#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fire_engine/graphics/object.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

class Animation;
struct RenderContext;
class Skin;

class Mesh
{
public:
    struct MorphAnimationEntry
    {
        std::size_t id{0};
        Animation* animation{nullptr};
    };

    explicit Mesh(Object object);
    ~Mesh() = default;

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;

    void skin(Skin* s) noexcept
    {
        object_.skin(s);
    }

    // Read-only access to the wrapped renderable — the scene culler reads its local
    // bounds and deformable flag to build/maintain its spatial proxy.
    [[nodiscard]] const Object& object() const noexcept
    {
        return object_;
    }

    void addMorphAnimation(std::size_t id, Animation* anim);
    void addMorphAnimation(Animation* anim)
    {
        addMorphAnimation(morphAnimations_.size(), anim);
    }

    void activeMorphAnimation(std::size_t id) noexcept;

    [[nodiscard]] std::size_t activeMorphAnimation() const noexcept
    {
        return activeMorphAnimationId_;
    }

    [[nodiscard]] std::size_t morphAnimationCount() const noexcept
    {
        return morphAnimations_.size();
    }

    [[nodiscard]] Animation* morphAnimation() noexcept
    {
        if (morphAnimations_.empty())
        {
            return nullptr;
        }
        return morphAnimations_[activeMorphIndex_].animation;
    }

    [[nodiscard]] const Animation* morphAnimation() const noexcept
    {
        if (morphAnimations_.empty())
        {
            return nullptr;
        }
        return morphAnimations_[activeMorphIndex_].animation;
    }

    void initialMorphWeights(std::vector<float> w) noexcept
    {
        morphWeights_ = std::move(w);
    }

    void update(const InputState& input_state, const Transform& transform);

    [[nodiscard]]
    Mat4 render(const RenderContext& ctx, const Mat4& world, const Mat4& previousWorld);

    void variantNames(std::vector<std::string> names) noexcept
    {
        variantNames_ = std::move(names);
    }

private:
    [[nodiscard]] std::size_t findMorphAnimationIndex(std::size_t id) const noexcept;
    void cycleVariant(int delta) noexcept;
    [[nodiscard]] bool isSelectableVariantState(int state) const noexcept;

    Object object_;
    std::vector<MorphAnimationEntry> morphAnimations_;
    std::size_t activeMorphIndex_{0};
    std::size_t activeMorphAnimationId_{0};
    std::vector<float> morphWeights_;
    std::vector<std::string> variantNames_;
    std::optional<std::size_t> activeVariant_{};
    double startTime_{0.0};
    bool morphInitialized_{false};
};

} // namespace fire_engine
