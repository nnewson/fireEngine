// The SHADOW pass' push-constant block — declared ONCE, here, and included by every shadow stage.
//
// Push constants are a raw byte range with no driver-side reflection: a field added to one copy of
// this block and not another silently reinterprets every field after it, and nothing fails. This
// block had been hand-copied into shadow.vert, shadow.frag and self_shadow_second.frag, so SH-05's
// `materialIndex` would have been the fourth copy to keep in step — exactly the drift that made the
// sky get multiplied by a shadow matrix (see cmake/check_shader_blocks.cmake). The C++ side is
// ShadowPushConstants in render/ubo.hpp, whose static_asserts pin these offsets.
//
// Both stages see the whole range (the pipeline declares it vertex | fragment), so a stage that
// reads only some fields still declares all of them.
layout(push_constant) uniform ShadowPushConstants {
    // Selects lightViewProj[] in the vertex stage; < 0 means "use pc.lightViewProj" (self-shadow).
    int matrixIndex;
    // Per-skinned-object self-shadow layer for the dual-depth self pass.
    int selfShadowSlot;
    // Normalized-depth gap before a fragment counts as the second surface.
    float selfShadowDepthEpsilon;
    // SH-05: index into the global bindless materials[] SSBO for this draw — the SAME authority the
    // forward pass indexes, so a cutout's shadow tests the material the surface is shaded with
    // rather than a shadow-only copy of it. Read by the masked fragment path only; occupies what
    // used to be explicit padding, so every offset around it is unchanged.
    uint materialIndex;
    // Point shadow (matrixIndex >= SHADOW_POINT_MATRIX_BASE): xyz = light world position,
    // w = effective range. Zero for cascade/spot/self passes.
    vec4 lightPosRange;
    // Used when matrixIndex < 0, for the tightly-fit per-object self-shadow views.
    mat4 lightViewProj;
} pc;
