// The FORWARD-family push-constant block — declared ONCE, here, and included by every stage that
// reads it: `shader.frag` and, since the depth prepass learned to apply the alpha cutout,
// `depth_prepass.frag`.
//
// Push constants are a raw byte range with no driver-side reflection, so a member added to one copy
// and not another silently reinterprets every field after it — a shifted `materialIndex` indexes a
// different bindless material, which is a wrong texture rather than an error. The C++ side is
// ForwardPushConstants in render/ubo.hpp, whose static_asserts pin these offsets; the range the two
// recorders push and the range the pipeline layout declares must cover the whole block, even in a
// stage that reads only one field of it.
layout(push_constant) uniform ForwardPushConstants {
    int selfShadowSlot;
    uint materialIndex; // index into the global materials[] SSBO for this draw
    uint lodLevel;      // selected discrete LOD level (read only for the LOD debug tint)
    // Level this mesh's shadow draw selected, or 0xFFFFFFFF when it casts no shadow (kNoShadowLod
    // in graphics/shadow_diagnostics.hpp). Read only for the Shadow-LOD debug tint.
    uint shadowLodLevel;
} pc;
