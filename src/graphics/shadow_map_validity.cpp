#include "fire_engine/graphics/shadow_map_validity.hpp"

namespace fire_engine
{

std::int32_t ShadowMapValidity::packedMask() const noexcept
{
    std::int32_t mask = 0;
    if (cascades)
    {
        mask |= kShadowMapValidCascades;
    }
    if (worldOnly)
    {
        mask |= kShadowMapValidWorldOnly;
    }
    if (self)
    {
        mask |= kShadowMapValidSelf;
    }
    if (spot)
    {
        mask |= kShadowMapValidSpot;
    }
    if (point)
    {
        mask |= kShadowMapValidPoint;
    }
    return mask;
}

bool ShadowMapValidity::none() const noexcept
{
    return !cascades && !worldOnly && !self && !spot && !point;
}

ShadowMapValidity shadowMapValidity(const ShadowMapValidityInputs& inputs) noexcept
{
    // One early return, not a `shadowsDisabled` term repeated in five expressions: "no shadows"
    // means no family records and no family is sampled, and stating it once makes that total.
    if (inputs.shadowsDisabled)
    {
        return ShadowMapValidity{};
    }

    constexpr auto cascadeCount = static_cast<std::size_t>(kShadowCascadeCount);
    constexpr auto faceCount = static_cast<std::size_t>(kCubeFaceCount);

    ShadowMapValidity validity{};
    // EVERY cascade, not a non-zero count — a fragment picks its layer by depth and would sample an
    // unfitted one. See the header.
    validity.cascades = inputs.primaryDirectionalLight && inputs.activeCascadeViews == cascadeCount;
    validity.worldOnly =
        inputs.primaryDirectionalLight && inputs.activeWorldOnlyViews == cascadeCount;
    validity.self = inputs.primaryDirectionalLight && inputs.activeSelfViews > 0;
    validity.spot = inputs.activeSpotViews > 0;
    // Whole cubes only. A remainder means the atomic six-face installation was bypassed, and half a
    // cube is a light whose shadow depends on which way the receiver happens to face.
    validity.point = inputs.activePointViews > 0 && inputs.activePointViews % faceCount == 0;
    return validity;
}

} // namespace fire_engine
