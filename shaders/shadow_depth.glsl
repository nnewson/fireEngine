// The depth a shadow fragment records, shared by the shadow pass' fragment stages.
//
// Requires `pc` (shaders/shadow_push.glsl) to be declared first. Shared because both the opaque and
// the SH-05 masked fragment paths must record the SAME depth: a point-shadow face that stored raw
// hardware depth in one path and a linear distance/range ratio in the other would compare against
// the main shader's samplerCubeArrayShadow on one of them only, and cutout casters would lose their
// point shadows for a reason that looks like a bias problem.
//
// Point faces store linear distance / range into gl_FragDepth so the main pass' comparison sampler
// tests the same ratio. Cascade, spot and self views write nothing here and keep the fixed-function
// hardware depth, which is what keeps contact shadows attached.
//
// The discriminator is the view's DEPTH MODE, pushed as `radialDepth` from
// `PreparedShadowView::depthMode()`. It used to be "is this matrix index at or past the point base"
// — the right answer inferred from an unrelated fact, which stopped being available at all once the
// matrix table was retired.
void writeShadowDepth(vec3 worldPos)
{
    if (pc.radialDepth != 0) {
        float range = max(pc.lightPosRange.w, 1e-4);
        gl_FragDepth = clamp(length(worldPos - pc.lightPosRange.xyz) / range, 0.0, 1.0);
    }
}
