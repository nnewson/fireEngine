#version 450

#include "shadow_push.glsl"
#include "shadow_depth.glsl"

layout(location = 0) in vec3 worldPos;
// shadow.vert also forwards this caster's two UV sets for the masked path; an opaque caster reads
// neither, and an unread vertex output is free.

void main() {
    // Depth-only pass, OPAQUE caster (SH-05): every rasterised fragment occludes, so there is no
    // material to read and nothing to test. That is why this path stays separate from
    // shadow_masked.frag instead of sampling a cutoff which would be 0 for every material routed
    // here — the fetch is the whole cost, and it would be paid per shadow fragment per view.
    writeShadowDepth(worldPos);
}
