#pragma once

#include <memory>
#include <optional>
#include <string>
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
