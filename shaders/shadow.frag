#version 450

const int SHADOW_POINT_MATRIX_BASE = 8;

layout(push_constant) uniform ShadowPushConstants {
    int matrixIndex;
    vec4 lightPosRange;
} pc;

layout(location = 0) in vec3 worldPos;
layout(location = 0) out vec4 outColor;

void main() {
    // Point shadow faces store linear distance / range, so the main pass'
    // samplerCubeArrayShadow compares the same ratio. Cascade and spot passes
    // keep the hardware depth value so contact shadows stay attached.
    float storedDepth = gl_FragCoord.z;
    if (pc.matrixIndex >= SHADOW_POINT_MATRIX_BASE) {
        float range = max(pc.lightPosRange.w, 1e-4);
        storedDepth = clamp(length(worldPos - pc.lightPosRange.xyz) / range, 0.0, 1.0);
        gl_FragDepth = storedDepth;
    }
    outColor = vec4(storedDepth, storedDepth, storedDepth, 1.0);
}
