#version 450

const int SHADOW_POINT_MATRIX_BASE = 8;

layout(push_constant) uniform ShadowPushConstants {
    int matrixIndex;
    vec4 lightPosRange;
} pc;

layout(location = 0) in vec3 worldPos;

void main() {
    // Point shadow faces store linear distance / range, so the main pass'
    // samplerCubeArrayShadow compares the same ratio. Cascade and spot passes
    // keep gl_FragCoord.z (default depth output) — perspective NDC z is fine
    // for those because the lit shader replays the same projection.
    if (pc.matrixIndex >= SHADOW_POINT_MATRIX_BASE) {
        float range = max(pc.lightPosRange.w, 1e-4);
        gl_FragDepth = clamp(length(worldPos - pc.lightPosRange.xyz) / range, 0.0, 1.0);
    }
}
