#include <fire_engine/scene/node.hpp>

namespace fire_engine
{

Node::Node(std::string name)
    : name_{std::move(name)}
{
}

Node& Node::addChild(std::unique_ptr<Node> child)
{
    child->parent_ = this;
    children_.push_back(std::move(child));
    return *children_.back();
}

void Node::update(const InputState& input_state, const Mat4& parentComposedWorld)
{
    if (controllable_)
    {
        controllable_->update(input_state.controllerState(), transform_, parentComposedWorld);
    }
    else
    {
        transform_.update(parentComposedWorld);
    }

    visitComponent([&input_state, this](auto& component)
                   { component.update(input_state, transform_); });

    // Composed world includes component effects (e.g. Animator's model matrix)
    Mat4 componentMatrix = componentModelMatrix(component_);
    Mat4 newComposedWorld = parentComposedWorld * transform_.local() * componentMatrix;

    // Carry last frame's world for motion vectors (TAA) and continuous
    // collision / constraint solving. First frame: previous == current so the
    // motion vector is zero rather than a jump from the identity default.
    previousComposedWorld_ = hasComposedWorld_ ? composedWorld_ : newComposedWorld;
    hasComposedWorld_ = true;
    composedWorld_ = newComposedWorld;

    for (auto& child : children_)
    {
        child->update(input_state, composedWorld_);
    }
}

void Node::resolve(const Mat4& parentComposedWorld)
{
    transform_.update(parentComposedWorld);

    Mat4 componentMatrix = componentModelMatrix(component_);
    Mat4 newComposedWorld = parentComposedWorld * transform_.local() * componentMatrix;

    previousComposedWorld_ = hasComposedWorld_ ? composedWorld_ : newComposedWorld;
    hasComposedWorld_ = true;
    composedWorld_ = newComposedWorld;

    for (auto& child : children_)
    {
        child->resolve(composedWorld_);
    }
}

void Node::render(const RenderContext& ctx, const Mat4& parentWorld)
{
    Mat4 world = parentWorld * transform_.local();
    // Components that contribute to rendering (Animator, Mesh) define
    // render(ctx, world); the rest (Empty, Camera, Light) are no-ops that just
    // pass the world matrix down, handled here instead of each defining a
    // trivial render().
    Mat4 childWorld = visitComponent(
        [&ctx, &world, this](auto& component) -> Mat4
        {
            // Geometry components (Mesh) take the previous world too, for motion
            // vectors (TAA). Others keep the 2-arg form; the rest are no-ops.
            if constexpr (requires { component.render(ctx, world, previousComposedWorld_); })
            {
                return component.render(ctx, world, previousComposedWorld_);
            }
            else if constexpr (requires { component.render(ctx, world); })
            {
                return component.render(ctx, world);
            }
            else
            {
                return world;
            }
        });

    for (auto& child : children_)
    {
        child->render(ctx, childWorld);
    }
}

} // namespace fire_engine
