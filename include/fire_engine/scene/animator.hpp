#pragma once

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/scene/transform.hpp>

#include <cstddef>
#include <vector>

namespace fire_engine
{

class Animation;
struct RenderContext;

class Animator
{
public:
    struct AnimationEntry
    {
        std::size_t id{0};
        Animation* animation{nullptr};
    };

    Animator() = default;
    ~Animator() = default;

    Animator(const Animator&) = default;
    Animator& operator=(const Animator&) = default;
    Animator(Animator&&) noexcept = default;
    Animator& operator=(Animator&&) noexcept = default;

    void update(const InputState& input_state, const Transform& transform);

    [[nodiscard]]
    Mat4 render(const RenderContext& ctx, const Mat4& world);

    [[nodiscard]]
    Mat4 render(const Mat4& world) const noexcept;

    [[nodiscard]]
    Mat4 modelMatrix() const noexcept
    {
        return modelMatrix_;
    }

    void addAnimation(std::size_t id, Animation* anim);
    void addAnimation(Animation* anim)
    {
        addAnimation(animations_.size(), anim);
    }

    void activeAnimation(std::size_t id) noexcept;

    [[nodiscard]] std::size_t activeAnimation() const noexcept
    {
        return activeAnimationId_;
    }

    [[nodiscard]] std::size_t animationCount() const noexcept
    {
        return animations_.size();
    }

    [[nodiscard]] Animation* animation() noexcept
    {
        if (animations_.empty())
        {
            return nullptr;
        }
        return animations_[activeIndex_].animation;
    }

    [[nodiscard]] const Animation* animation() const noexcept
    {
        if (animations_.empty())
        {
            return nullptr;
        }
        return animations_[activeIndex_].animation;
    }

private:
    [[nodiscard]] std::size_t findAnimationIndex(std::size_t id) const noexcept;

    std::vector<AnimationEntry> animations_;
    std::size_t activeIndex_{0};
    std::size_t activeAnimationId_{0};
    Mat4 modelMatrix_{Mat4::identity()};
    double startTime_{0.0};
    bool initialized_{false};
};

} // namespace fire_engine
