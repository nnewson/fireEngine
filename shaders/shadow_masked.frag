#version 450

// SH-05: the ALPHA-MASKED shadow fragment path. Same depth-only pass as shadow.frag, plus the
// visible material's alpha cutout — so a leaf card casts a leaf and not the quad it is drawn on.
//
// The cutout comes from the bindless material authority the forward shader indexes (material.glsl,
// set 2), reached through `pc.materialIndex`. Deliberately NOT a shadow-only material format: the
// silhouette a shadow must cast is the one the surface itself shows, so any second copy of the
// cutoff, the UV set, the KHR_texture_transform or the sampler is a way for the two to disagree —
// and the disagreement reads as a shadow bias artefact rather than as a mask bug.
#include "shadow_push.glsl"
#include "material.glsl"
#include "shadow_depth.glsl"

layout(location = 0) in vec3 worldPos;
// Both authored UV sets, forwarded by shadow.vert. glTF lets the base-colour texture choose either,
// and the material says which — the same choice the forward pass makes for this material.
layout(location = 1) in vec2 uv0;
layout(location = 2) in vec2 uv1;

void main() {
    // Tested BEFORE the depth write. A discarded fragment must record no depth at all: recording it
    // and then discarding would leave the cutout's holes occluding, which is the bug this path
    // exists to fix.
    if (materialAlphaCutoutFails(materialAlpha(materialBaseColourTexel(uv0, uv1)))) {
        discard;
    }
    writeShadowDepth(worldPos);
}
