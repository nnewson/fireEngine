#version 450

// SHADOW_TOTAL_MATRIX_COUNT — the shared declaration graphics/gpu_limits.hpp re-exports as
// kShadowTotalMatrixCount, which is what sizes the C++ ShadowUBO this block must match.
#include "gpu_limits.glsl"

layout(binding = 0) uniform ShadowUBO {
    mat4 model;
    mat4 lightViewProj[SHADOW_TOTAL_MATRIX_COUNT];
    int hasSkin;
} shadow;

#include "shadow_push.glsl"

layout(binding = 1) uniform SkinUBO {
    mat4 joints[MAX_JOINTS];
} skin;

layout(binding = 2) uniform MorphUBO {
    int hasMorph;
    int morphTargetCount;
    int vertexCount;
    int _pad0;
    vec4 weights[MORPH_WEIGHT_VEC4_COUNT];
} morph;

layout(std430, binding = 3) readonly buffer MorphTargets {
    vec4 data[];
} morphTargets;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uvec4 inJoints;
layout(location = 5) in vec4 inWeights;
layout(location = 6) in vec4 inTangent;
// Second UV set (SH-05). glTF lets a material's base-colour texture read TEXCOORD_0 or TEXCOORD_1,
// so the masked fragment path needs both and the material picks — the loader copies set 0 into set 1
// when a mesh authors only one, so this attribute is always meaningful.
layout(location = 8) in vec2 inTexCoord1;

// Forwarded to the fragment stage so the point-shadow branch can write a
// linear distance/range depth value via gl_FragDepth.
layout(location = 0) out vec3 worldPos;
// SH-05: the caster's UVs, for the masked fragment path's cutout test. Emitted unconditionally
// rather than from a second shadow vertex shader — the skinning and morph maths below is the thing
// that must not be duplicated, and shadow.frag simply ignores these. Skinning and morphing move
// positions, never UVs, so no deformation applies here.
layout(location = 1) out vec2 uv0;
layout(location = 2) out vec2 uv1;

void main() {
    vec3 pos = inPos;

    if (morph.hasMorph == 1) {
        int nTargets = morph.morphTargetCount;
        int nVerts = morph.vertexCount;
        for (int i = 0; i < nTargets; i++) {
            int posOffset = i * nVerts + gl_VertexIndex;
            float w = morph.weights[i / 4][i % 4];
            pos += w * morphTargets.data[posOffset].xyz;
        }
    }

    mat4 transform;
    if (shadow.hasSkin == 1) {
        transform = inWeights.x * skin.joints[inJoints.x]
                  + inWeights.y * skin.joints[inJoints.y]
                  + inWeights.z * skin.joints[inJoints.z]
                  + inWeights.w * skin.joints[inJoints.w];
    } else {
        transform = shadow.model;
    }

    vec4 wp = transform * vec4(pos, 1.0);
    worldPos = wp.xyz;
    uv0 = inTexCoord;
    uv1 = inTexCoord1;
    mat4 lightMatrix = pc.matrixIndex < 0 ? pc.lightViewProj : shadow.lightViewProj[pc.matrixIndex];
    gl_Position = lightMatrix * wp;
}
