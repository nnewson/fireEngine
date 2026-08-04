#version 450

// SH-05: the second-depth self-shadow layer for an ALPHA-MASKED caster. The dual-depth rejection of
// self_shadow_second.frag plus the visible material's cutout.
//
// Needed for the same reason the main pass needs its masked path, and not one step less: a cutout
// character (hair cards, a fringed skirt) whose first layer masks and whose second layer does not
// would record a second surface where the first recorded nothing, so the receiver test would compare
// against geometry that casts no shadow — self-shadowing through a hole.
#include "shadow_push.glsl"
#include "material.glsl"
#include "self_shadow_second.glsl"

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec2 uv0;
layout(location = 2) in vec2 uv1;

void main() {
    // Cutout FIRST: a masked-out fragment is not a surface at all, so it cannot be anybody's second
    // depth, and testing it against the first layer would ask a question about geometry that is not
    // there.
    if (materialAlphaCutoutFails(materialAlpha(materialBaseColourTexel(uv0, uv1)))) {
        discard;
    }
    if (selfShadowSecondRejects()) {
        discard;
    }
}
