#version 450

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    int hasSkin;
} ubo;

layout(binding = 3) uniform SkinUBO {
    mat4 joints[64];
} skin;

layout(binding = 4) uniform MorphUBO {
    int hasMorph;
    int morphTargetCount;
    int vertexCount;
    int _pad0;
    vec4 weights[2];
} morph;

// Morph target deltas: [pos0..posN, norm0..normN] packed as vec4 (w unused)
layout(std430, binding = 5) readonly buffer MorphTargets {
    vec4 data[];
} morphTargets;

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

void main() {
    vec3 pos = inPos;
    vec3 normal = inNormal;
    vec3 tangent = inTangent.xyz;

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
    gl_Position = ubo.proj * ubo.view * worldPos;
    fragColor = inColor;
    fragViewDepth = -(ubo.view * worldPos).z;
    fragTexCoord1 = inTexCoord1;

    fragNormal = normalize(normalTransform * normal);
    fragWorldPos = worldPos.xyz;
    fragTexCoord = inTexCoord;

    // TBN matrix for normal mapping. Use the morph-blended tangent so facial
    // expression rigs get correct normal-mapped lighting per blend.
    vec3 N = normalize(fragNormal);
    vec3 T = normalTransform * tangent;
    T = normalize(T - N * dot(N, T));
    vec3 B = normalize(cross(N, T)) * inTangent.w;
    fragTBN = mat3(T, B, N);
}
