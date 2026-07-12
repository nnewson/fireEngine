#version 450

// Per-object data (set 0, pushed per draw). previousModel is last frame's world for motion vectors.
layout(binding = 0) uniform ObjectUBO {
    mat4 model;
    int hasSkin;
    int _pad1;
    int _pad2;
    int _pad3;
    mat4 previousModel;
} ubo;

// Per-frame camera data (set 0, binding 29 — pushed per draw with the same handle every draw, so the
// depth prepass gets it too). proj is jittered (TAA) for rasterisation; the two view-projections are
// jitter-free so motion vectors are independent of the sub-pixel jitter.
layout(binding = 29) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    mat4 currentViewProj;
    mat4 previousViewProj;
} camera;

layout(binding = 3) uniform SkinUBO {
    mat4 joints[64];
} skin;

layout(binding = 4) uniform MorphUBO {
    int hasMorph;
    int morphTargetCount;
    int vertexCount;
    int _pad0;
    vec4 weights[2];
    float morphFactor;    // VIPM geomorph amount (0 = discrete / no morph)
    int vipmTargetLevel;  // vertices removed by this 1-based LOD level morph in this transition
} morph;

// Morph target deltas: [pos0..posN, norm0..normN] packed as vec4 (w unused)
layout(std430, binding = 5) readonly buffer MorphTargets {
    vec4 data[];
} morphTargets;

// VIPM per-vertex geomorph data (Continuous LOD): full render-attribute target so the morph
// doesn't warp the texture or TBN basis. Four vec4s per vertex.
struct VipmVert {
    vec4 posLevel;  // xyz = target position, w = 1-based LOD level where this vertex disappears
    vec4 normalPad; // xyz = target normal
    vec4 tangent;   // xyzw = target tangent + handedness
    vec4 uvPad;     // xy = target TEXCOORD_0, zw = target TEXCOORD_1
};
layout(std430, binding = 28) readonly buffer VipmMorph {
    VipmVert v[];
} vipm;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uvec4 inJoints;
layout(location = 5) in vec4 inWeights;
layout(location = 6) in vec4 inTangent;
// Second UV set; loader falls back to inTexCoord values when the mesh only
// has TEXCOORD_0, so this is always defined.
layout(location = 8) in vec2 inTexCoord1;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out mat3 fragTBN;
// Positive distance along the view forward axis. The forward fragment shader
// uses this for cascade selection against light.cascadeSplits.
layout(location = 7) out float fragViewDepth;
layout(location = 8) out vec2 fragTexCoord1;
// Jitter-free clip positions for TAA motion vectors. The fragment stage does
// the perspective divide and writes the UV-space velocity.
layout(location = 9) out vec4 fragCurClip;
layout(location = 10) out vec4 fragPrevClip;

// The depth prepass reuses this exact vertex shader (depth-only) and the forward
// pass loads that depth with a LESS_OR_EQUAL test. Mark gl_Position invariant so
// both pipelines compute bit-identical clip positions — otherwise per-pipeline
// optimisation could shift a vertex by 1 ULP and punch holes in the forward pass.
invariant gl_Position;

void main() {
    vec3 pos = inPos;
    vec3 normal = inNormal;
    vec3 tangent = inTangent.xyz;
    float tangentSign = inTangent.w;
    vec2 uv = inTexCoord;
    vec2 uv1 = inTexCoord1;

    // Apply morph targets. SSBO layout (per Object::load):
    //   [pos_0..N-1, norm_0..N-1, tang_0..N-1] each as vec4(xyz, 0).
    // Tangent slice is all-zero when the source asset doesn't ship morph
    // TANGENT data, so the additive blend is a no-op then.
    if (morph.hasMorph == 1) {
        int nTargets = morph.morphTargetCount;
        int nVerts = morph.vertexCount;
        for (int i = 0; i < nTargets; i++) {
            int posOffset = i * nVerts + gl_VertexIndex;
            int normOffset = (nTargets + i) * nVerts + gl_VertexIndex;
            int tangOffset = (2 * nTargets + i) * nVerts + gl_VertexIndex;
            float w = morph.weights[i / 4][i % 4];
            pos += w * morphTargets.data[posOffset].xyz;
            normal += w * morphTargets.data[normOffset].xyz;
            tangent += w * morphTargets.data[tangOffset].xyz;
        }
    }

    // VIPM geomorph (Continuous LOD): morph a collapsing vertex's complete render attributes
    // toward the exact wedge drawn by the next LOD. morphFactor is 0 in Discrete mode / for non-VIPM
    // meshes, so the && short-circuits and the (dummy) buffer is never read.
    if (morph.morphFactor > 0.0 &&
        int(vipm.v[gl_VertexIndex].posLevel.w + 0.5) == morph.vipmTargetLevel) {
        float f = morph.morphFactor;
        pos = mix(pos, vipm.v[gl_VertexIndex].posLevel.xyz, f);
        normal = mix(normal, vipm.v[gl_VertexIndex].normalPad.xyz, f);
        tangent = mix(tangent, vipm.v[gl_VertexIndex].tangent.xyz, f);
        tangentSign = mix(inTangent.w, vipm.v[gl_VertexIndex].tangent.w, f) >= 0.0 ? 1.0 : -1.0;
        uv = mix(uv, vipm.v[gl_VertexIndex].uvPad.xy, f);
        uv1 = mix(uv1, vipm.v[gl_VertexIndex].uvPad.zw, f);
    }

    mat4 transform;
    mat3 normalTransform;
    if (ubo.hasSkin == 1) {
        transform = inWeights.x * skin.joints[inJoints.x]
                  + inWeights.y * skin.joints[inJoints.y]
                  + inWeights.z * skin.joints[inJoints.z]
                  + inWeights.w * skin.joints[inJoints.w];
        // Skin matrices can contain armature conversion and blended joint
        // scale/shear, so normals need the same inverse-transpose treatment as
        // static meshes.
        normalTransform = transpose(inverse(mat3(transform)));
    } else {
        transform = ubo.model;
        normalTransform = transpose(inverse(mat3(transform)));
    }

    vec4 worldPos = transform * vec4(pos, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;

    // TAA motion vectors. Skinned meshes have no previous joint data, so they
    // fall back to camera-only velocity (previous == current world position);
    // rigid/animated nodes use the node's previous world matrix. Both view-
    // projections are jitter-free so the velocity is independent of the jitter.
    vec4 prevWorldPos = (ubo.hasSkin == 1) ? worldPos : ubo.previousModel * vec4(pos, 1.0);
    fragCurClip = camera.currentViewProj * worldPos;
    fragPrevClip = camera.previousViewProj * prevWorldPos;

    fragColor = inColor;
    fragViewDepth = -(camera.view * worldPos).z;
    fragTexCoord1 = uv1;

    fragNormal = normalize(normalTransform * normal);
    fragWorldPos = worldPos.xyz;
    fragTexCoord = uv;

    // TBN matrix for normal mapping. Use the morph-blended tangent so facial
    // expression rigs get correct normal-mapped lighting per blend.
    vec3 N = normalize(fragNormal);
    vec3 T = normalTransform * tangent;
    T = normalize(T - N * dot(N, T));
    vec3 B = normalize(cross(N, T)) * tangentSign;
    fragTBN = mat3(T, B, N);
}
