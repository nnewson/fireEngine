#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(binding = 4) uniform texture2DArray selfShadowFirstMapTex;
layout(binding = 5) uniform sampler selfShadowDepthSampler;

layout(push_constant) uniform ShadowPushConstants {
    int matrixIndex;
    int selfShadowSlot;
    float selfShadowDepthEpsilon;
    float _pad0;
    vec4 lightPosRange;
    mat4 lightViewProj;
} pc;

layout(location = 0) in vec3 worldPos;

void main() {
    // Depth-only pass: no colour output. The discards reject same-surface
    // fragments (dual-depth self-shadow rejection); surviving fragments keep the
    // fixed-function depth.
    if (pc.selfShadowSlot < 0 || pc.selfShadowSlot >= 4) {
        discard;
    }

    vec2 extent = vec2(textureSize(selfShadowFirstMapTex, 0).xy);
    vec2 uv = gl_FragCoord.xy / extent;
    float firstDepth = texture(sampler2DArray(selfShadowFirstMapTex, selfShadowDepthSampler),
                               vec3(uv, float(pc.selfShadowSlot))).r;
    float currentDepth = gl_FragCoord.z;
    if (currentDepth <= firstDepth + pc.selfShadowDepthEpsilon) {
        discard;
    }
}
