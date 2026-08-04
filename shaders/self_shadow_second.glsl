// The dual-depth self-shadow rejection test, shared by the second-depth pass' fragment stages
// (opaque and SH-05 masked). Requires `pc` (shaders/shadow_push.glsl) first.
//
// Shared rather than copied because the test IS the pass: a masked variant that drifted on the
// epsilon, the slot bound, or the layer it samples would write a second-depth layer that disagrees
// with the opaque one, and the symptom — self-shadow acne on cutout characters only — would look
// like a bias problem instead of a duplicated formula.
#extension GL_EXT_samplerless_texture_functions : require

layout(binding = 4) uniform texture2DArray selfShadowFirstMapTex;
layout(binding = 5) uniform sampler selfShadowDepthSampler;

// True when this fragment is the SAME surface the first (light-facing) layer already recorded, so it
// must not become the second depth. The slot bound mirrors kMaxSkinnedSelfShadowCasters
// (graphics/gpu_limits.hpp): a fragment carrying a slot outside the array has no layer to compare
// against, and rejecting it is the only answer that cannot sample someone else's caster.
bool selfShadowSecondRejects()
{
    if (pc.selfShadowSlot < 0 || pc.selfShadowSlot >= 4) {
        return true;
    }
    vec2 extent = vec2(textureSize(selfShadowFirstMapTex, 0).xy);
    vec2 uv = gl_FragCoord.xy / extent;
    float firstDepth = texture(sampler2DArray(selfShadowFirstMapTex, selfShadowDepthSampler),
                               vec3(uv, float(pc.selfShadowSlot))).r;
    float currentDepth = gl_FragCoord.z;
    return currentDepth <= firstDepth + pc.selfShadowDepthEpsilon;
}
