#version 450

// Particle billboard fragment shader (Roadmap Milestone B). Soft round sprite,
// additively blended into the HDR target so bloom picks up the glow. Soft
// particles: fades the sprite out as it approaches scene geometry (sampled
// scene depth) so there is no hard clip edge at intersections.

layout(location = 0) in vec2 fragUv;
layout(location = 1) in vec3 fragColour;
layout(location = 2) in float fragViewDepth; // eye-space distance of this fragment

layout(binding = 2) uniform sampler2D sceneDepth;

layout(push_constant) uniform Soft {
    float nearPlane;
    float farPlane;
    float softRange;
    float _pad0;
} soft;

layout(location = 0) out vec4 outColour;

// Convert a non-linear [0,1] depth-buffer value to positive eye-space distance.
float lineariseDepth(float d) {
    return (soft.nearPlane * soft.farPlane) /
           (soft.farPlane - d * (soft.farPlane - soft.nearPlane));
}

void main() {
    // Radial falloff: opaque at centre, fading to the quad edge.
    vec2 d = fragUv * 2.0 - 1.0;
    float r2 = dot(d, d);
    float alpha = clamp(1.0 - r2, 0.0, 1.0);
    alpha *= alpha;

    // Soft particles: scene depth at this pixel vs the particle's eye depth.
    float sceneEye = lineariseDepth(texelFetch(sceneDepth, ivec2(gl_FragCoord.xy), 0).r);
    float fade = clamp((sceneEye - fragViewDepth) / max(soft.softRange, 1e-4), 0.0, 1.0);

    outColour = vec4(fragColour * alpha * fade, alpha * fade);
}
