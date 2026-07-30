#version 450

layout(binding = 0) uniform SkyboxUBO {
    vec4 cameraForward;
    vec4 cameraRight;
    vec4 cameraUp;
    vec4 viewParams; // x = tanHalfFov, y = aspect
} sky;

layout(binding = 1) uniform samplerCube skyboxMap;

// The skybox needs exactly one field of this buffer — environmentParams.x, the sky intensity — but it
// must declare the WHOLE block, because a uniform block's field offsets depend on every field before
// them. Hence the shared include rather than a hand-written subset.
#define LIGHT_UBO_SET 0
#define LIGHT_UBO_BINDING 2
#include "light_ubo.glsl"

layout(location = 0) in vec2 fragUv;

layout(location = 0) out vec4 outColor;
// TAA velocity attachment. The skybox writes zero motion (treated as static);
// camera-rotation reprojection of the background is a possible follow-up.
layout(location = 1) out vec2 outVelocity;

void main() {
    vec2 ndc = fragUv * 2.0 - 1.0;

    float tanHalfFov = sky.viewParams.x;
    float aspect = sky.viewParams.y;

    vec3 forward = sky.cameraForward.xyz;
    vec3 right = sky.cameraRight.xyz;
    vec3 up = sky.cameraUp.xyz;

    // Flip ndc.y because Vulkan screen y is downward but world up should be at top of screen.
    vec3 dir = normalize(forward
                         + ndc.x * aspect * tanHalfFov * right
                         - ndc.y * tanHalfFov * up);

    vec3 skyColor = texture(skyboxMap, dir).rgb * light.environmentParams.x;
    outColor = vec4(skyColor, 1.0);
    outVelocity = vec2(0.0);
}
