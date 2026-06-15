#version 450

// Temporal anti-aliasing resolve. Reprojects the previous frame's accumulated
// history along the screen-space velocity buffer, clamps it to the local
// neighbourhood of the current frame (kills ghosting on disocclusion / fast
// motion), and blends the two. Output is written to the history ping-pong slot
// and then blitted into the offscreen HDR target by the Taa subsystem.

layout(location = 0) in vec2 fragUv;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D currentColor;
layout(binding = 1) uniform sampler2D velocityTex;
layout(binding = 2) uniform sampler2D historyTex;

layout(push_constant) uniform Push {
    vec2 texelSize;     // 1 / render-target resolution
    float historyBlend; // weight of history in the blend (kTaaHistoryBlend)
    float sharpen;      // post-resolve unsharp amount (0 = off)
    int historyValid;   // 0 on the first frame after (re)create
} pc;

void main() {
    vec3 current = texture(currentColor, fragUv).rgb;

    // Velocity is the UV-space motion (current - previous), so the matching
    // history sample lives at uv - velocity.
    vec2 velocity = texture(velocityTex, fragUv).rg;
    vec2 historyUv = fragUv - velocity;

    // 3x3 neighbourhood AABB of the current frame, used to clamp the history
    // sample back into plausible range. nsum accumulates the box mean for the
    // optional sharpen below.
    vec3 nmin = current;
    vec3 nmax = current;
    vec3 nsum = current;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) {
                continue;
            }
            vec3 c = texture(currentColor, fragUv + vec2(x, y) * pc.texelSize).rgb;
            nmin = min(nmin, c);
            nmax = max(nmax, c);
            nsum += c;
        }
    }

    // No usable history: first frame after a (re)create, or the reprojected UV
    // fell outside the screen (disocclusion at the frame edge). Use the current
    // frame so nothing ghosts in from undefined / off-screen data.
    bool offScreen = any(lessThan(historyUv, vec2(0.0))) || any(greaterThan(historyUv, vec2(1.0)));
    vec3 resolved;
    if (pc.historyValid == 0 || offScreen) {
        resolved = current;
    } else {
        vec3 history = clamp(texture(historyTex, historyUv).rgb, nmin, nmax);
        resolved = mix(current, history, pc.historyBlend);
    }

    // Unsharp mask against the 3x3 box mean to recover the slight softness TAA
    // trades for stability.
    if (pc.sharpen > 0.0) {
        vec3 mean = nsum / 9.0;
        resolved = max(resolved + pc.sharpen * (resolved - mean), vec3(0.0));
    }

    outColor = vec4(resolved, 1.0);
}
