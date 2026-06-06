#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_handle.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/controllable.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

class Node
{
public:
    Node() = default;
    explicit Node(std::string name);
    ~Node() = default;

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) noexcept = default;
    Node& operator=(Node&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept
    {
        return name_;
    }
    void name(std::string n) noexcept
    {
        name_ = std::move(n);
    }

    [[nodiscard]] Transform& transform() noexcept
    {
        return transform_;
    }
    [[nodiscard]] const Transform& transform() const noexcept
    {
        return transform_;
    }

    [[nodiscard]] Components& component() noexcept
    {
        return component_;
    }
    [[nodiscard]] const Components& component() const noexcept
    {
        return component_;
    }

    // Typed access to the component variant. Returns nullptr when the active
    // alternative is not T — so `if (auto* m = node.componentAs<Mesh>())` reads
    // the intent without spelling out std::get_if at every call site.
    template <typename T>
    [[nodiscard]] T* componentAs() noexcept
    {
        return std::get_if<T>(&component_);
    }
    template <typename T>
    [[nodiscard]] const T* componentAs() const noexcept
    {
        return std::get_if<T>(&component_);
    }

    // Dispatch a visitor over whichever component is active.
    template <typename Visitor>
    decltype(auto) visitComponent(Visitor&& visitor)
    {
        return std::visit(std::forward<Visitor>(visitor), component_);
    }
    template <typename Visitor>
    decltype(auto) visitComponent(Visitor&& visitor) const
    {
        return std::visit(std::forward<Visitor>(visitor), component_);
    }

    [[nodiscard]] bool hasControllable() const noexcept
    {
        return controllable_.has_value();
    }
    [[nodiscard]] Controllable* controllable() noexcept
    {
        return controllable_ ? &controllable_.value() : nullptr;
    }
    [[nodiscard]] const Controllable* controllable() const noexcept
    {
        return controllable_ ? &controllable_.value() : nullptr;
    }
    Controllable& emplaceControllable()
    {
        return controllable_.emplace();
    }

    [[nodiscard]]
    PhysicsBodyHandle physicsBodyHandle() const noexcept
    {
        return physicsBodyHandle_;
    }

    void physicsBodyHandle(PhysicsBodyHandle handle) noexcept
    {
        physicsBodyHandle_ = handle;
    }

    [[nodiscard]]
    bool hasPhysicsBodyHandle() const noexcept
    {
        return physicsBodyHandle_.valid();
    }

    [[nodiscard]]
    PhysicsColliderHandle physicsColliderHandle() const noexcept
    {
        return physicsColliderHandle_;
    }

    void physicsColliderHandle(PhysicsColliderHandle handle) noexcept
    {
        physicsColliderHandle_ = handle;
    }

    [[nodiscard]]
    bool hasPhysicsColliderHandle() const noexcept
    {
        return physicsColliderHandle_.valid();
    }

    [[nodiscard]] Node* parent() const noexcept
    {
        return parent_;
    }

    [[nodiscard]] const std::vector<std::unique_ptr<Node>>& children() const noexcept
    {
        return children_;
    }

    [[nodiscard]] const Mat4& composedWorld() const noexcept
    {
        return composedWorld_;
    }

    Node& addChild(std::unique_ptr<Node> child);

    void update(const InputState& input_state, const Mat4& parentComposedWorld);
    void resolve(const Mat4& parentComposedWorld);
    void render(const RenderContext& ctx, const Mat4& parentWorld);

private:
    std::string name_;
    Transform transform_;
    Components component_;
    std::optional<Controllable> controllable_;
    PhysicsBodyHandle physicsBodyHandle_;
    PhysicsColliderHandle physicsColliderHandle_;
    Mat4 composedWorld_{Mat4::identity()};
    Node* parent_{nullptr};
    std::vector<std::unique_ptr<Node>> children_;
};

} // namespace fire_engine
