#include <fire_engine/graphics/shadow_caster_bounds_frame.hpp>

#include <format>
#include <stdexcept>
#include <utility>

namespace fire_engine
{

void ShadowCasterBoundsFrame::reset() noexcept
{
    entries_.clear();
    index_.clear();
}

void ShadowCasterBoundsFrame::add(const ShadowCasterBounds& caster)
{
    if (caster.casterId == ShadowCasterId::Invalid)
    {
        throw std::runtime_error(
            "shadow caster prepass: a binding reported bounds under an invalid caster id");
    }
    const auto [it, inserted] = index_.emplace(caster.key(), entries_.size());
    if (!inserted)
    {
        throw std::runtime_error(std::format(
            "shadow caster prepass: duplicate caster id {} generation {} — two bindings claim one "
            "identity",
            std::to_underlying(caster.casterId), std::to_underlying(caster.generation)));
    }
    entries_.push_back(caster);
}

const ShadowCasterBounds*
ShadowCasterBoundsFrame::find(ShadowCasterId casterId,
                              ShadowCasterGeneration generation) const noexcept
{
    const auto it = index_.find(ShadowCasterKey{.casterId = casterId, .generation = generation});
    return it == index_.end() ? nullptr : &entries_[it->second];
}

const ShadowCasterBounds& ShadowCasterBoundsFrame::require(ShadowCasterId casterId,
                                                           ShadowCasterGeneration generation) const
{
    if (const ShadowCasterBounds* found = find(casterId, generation))
    {
        return *found;
    }
    throw std::runtime_error(std::format(
        "shadow caster prepass: no bounds recorded for caster id {} generation {} — the "
        "prepass and the draw walk disagree about this frame's casters",
        std::to_underlying(casterId), std::to_underlying(generation)));
}

} // namespace fire_engine
