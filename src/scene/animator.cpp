#include <fire_engine/scene/animator.hpp>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/animation/animation_selection.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

std::size_t Animator::findAnimationIndex(std::size_t id) const noexcept
{
    return findAnimationEntryIndex(std::span<const AnimationEntry>{animations_}, id);
}

void Animator::addAnimation(std::size_t id, Animation* anim)
{
    animations_.push_back({id, anim});
    if (animations_.size() == 1)
    {
        activeIndex_ = 0;
        activeAnimationId_ = id;
    }
}

void Animator::activeAnimation(std::size_t id) noexcept
{
    (void)selectAnimationEntry(std::span<const AnimationEntry>{animations_}, id, activeIndex_,
                               activeAnimationId_, initialized_);
}

void Animator::update(const InputState& input_state, const Transform& /*transform*/)
{
    if (animations_.empty())
    {
        return;
    }

    const auto& animState = input_state.animationState();
    if (animState.hasActiveAnimation())
    {
        auto id = *animState.activeAnimation();
        auto index = findAnimationIndex(id);
        if (index < animations_.size() && index != activeIndex_)
        {
            (void)selectAnimationEntry(std::span<const AnimationEntry>{animations_}, id,
                                       activeIndex_, activeAnimationId_, initialized_);
        }
    }

    if (!initialized_)
    {
        startTime_ = input_state.time();
        initialized_ = true;
    }

    float t = static_cast<float>(input_state.time() - startTime_);
    modelMatrix_ = animations_[activeIndex_].animation->sample(t);
}

Mat4 Animator::render(const RenderContext& /*ctx*/, const Mat4& world)
{
    return render(world);
}

Mat4 Animator::render(const Mat4& world) const noexcept
{
    return world * modelMatrix_;
}

} // namespace fire_engine
