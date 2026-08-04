#version 450

#include "shadow_push.glsl"
#include "self_shadow_second.glsl"

layout(location = 0) in vec3 worldPos;

void main() {
    // Depth-only pass, OPAQUE caster (SH-05). The rejection drops same-surface fragments (dual-depth
    // self-shadow rejection); surviving fragments keep the fixed-function depth.
    if (selfShadowSecondRejects()) {
        discard;
    }
}
