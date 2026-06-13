#version 450

const int SHADOW_POINT_MATRIX_BASE = 8;

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
    // Depth-only pass: no colour output. Point shadow faces store linear
    // distance / range into gl_FragDepth so the main pass' samplerCubeArrayShadow
    // compares the same ratio. Cascade and spot passes write nothing here and
    // keep the fixed-function hardware depth so contact shadows stay attached.
    if (pc.matrixIndex >= SHADOW_POINT_MATRIX_BASE) {
        float range = max(pc.lightPosRange.w, 1e-4);
        gl_FragDepth = clamp(length(worldPos - pc.lightPosRange.xyz) / range, 0.0, 1.0);
    }
}
