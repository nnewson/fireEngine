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
    composedWorld_ = parentComposedWorld * transform_.local() * componentMatrix;

    for (auto& child : children_)
    {
        child->update(input_state, composedWorld_);
    }
}

void Node::resolve(const Mat4& parentComposedWorld)
{
    transform_.update(parentComposedWorld);

    Mat4 componentMatrix = componentModelMatrix(component_);
    composedWorld_ = parentComposedWorld * transform_.local() * componentMatrix;

    for (auto& child : children_)
    {
        child->resolve(composedWorld_);
    }
}

void Node::render(const RenderContext& ctx, const Mat4& parentWorld)
{
    Mat4 world = parentWorld * transform_.local();
    Mat4 childWorld = visitComponent([&ctx, &world](auto& component) -> Mat4
                                     { return component.render(ctx, world); });

    for (auto& child : children_)
    {
        child->render(ctx, childWorld);
    }
}

} // namespace fire_engine
