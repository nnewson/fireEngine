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
    // A world-override (ragdoll drive) is authoritative: the physics body's world pose IS the
    // composed world, bypassing the parent chain, the local transform and the component matrix.
    // The early return is about SIDE EFFECTS — an overridden node runs no controllable, no
    // transform update and no component update, because physics has already decided where it is.
    // The matrices themselves come from the shared helpers below, so this walk and the draw walk
    // cannot express different rules.
    if (worldOverride_)
    {
        setComposedWorld(drawWorld(parentComposedWorld));
        for (auto& child : children_)
        {
            child->update(input_state, composedWorld_);
        }
        return;
    }

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

    // Composed world includes component effects (e.g. Animator's model matrix) — and is what
    // children inherit, which is exactly what `childWorld` decides for the draw walk.
    const Mat4 world = drawWorld(parentComposedWorld);
    setComposedWorld(childWorld(world, world * componentModelMatrix(component_)));

    for (auto& child : children_)
    {
        child->update(input_state, composedWorld_);
    }
}

void Node::gatherShadowCasters(ShadowCasterBoundsFrame& out) const
{
    // `composedWorld_` is what the last update left, which is the same matrix the draw walk will
    // hand the object — the prepass and the draw therefore describe one pose, not two.
    if (const auto* mesh = componentAs<Mesh>())
    {
        mesh->gatherShadowCasters(composedWorld_, out);
    }
    for (const auto& child : children_)
    {
        child->gatherShadowCasters(out);
    }
}

void Node::resolve(const Mat4& parentComposedWorld)
{
    // Same rule, same helpers as `update` and the draw walk — see `Node::drawWorld` /
    // `Node::childWorld`.
    if (worldOverride_)
    {
        setComposedWorld(drawWorld(parentComposedWorld));
        for (auto& child : children_)
        {
            child->resolve(composedWorld_);
        }
        return;
    }

    transform_.update(parentComposedWorld);

    const Mat4 world = drawWorld(parentComposedWorld);
    setComposedWorld(childWorld(world, world * componentModelMatrix(component_)));

    for (auto& child : children_)
    {
        child->resolve(composedWorld_);
    }
}

void Node::setComposedWorld(const Mat4& newComposedWorld) noexcept
{
    // Carry last frame's world for motion vectors (TAA) and continuous
    // collision / constraint solving. First frame: previous == current so the
    // motion vector is zero rather than a jump from the identity default.
    const bool changed = !hasComposedWorld_ || composedWorld_ != newComposedWorld;
    previousComposedWorld_ = hasComposedWorld_ ? composedWorld_ : newComposedWorld;
    hasComposedWorld_ = true;
    composedWorld_ = newComposedWorld;
    if (changed)
    {
        ++worldRevision_;
    }
}

void Node::render(const SceneDrawContext& ctx, const Mat4& parentWorld)
{
    // ONE transform source, shared with `update` / `resolve` / the shadow-caster prepass — see
    // `Node::drawWorld`. This walk used to recompute `parentWorld * local` and ignore a
    // world-override, which drew a ragdoll-driven caster at a different pose than the one its
    // SH-06 bounds were measured at.
    Mat4 world = drawWorld(parentWorld);

    // The scene culler may have found this node outside every frustum. Skip its draw-building (no
    // UBO writes, no draw commands) but still recurse — children have independent bounds.
    // Shadow-caster bounds are NOT skipped by this: the prepass walks separately and
    // unconditionally, because a caster outside the camera frustum still casts into it.
    //
    // The inherited transform is derived the SAME way as below, from the component matrix rather
    // than from running the component. Handing `world` down directly would have been an escape from
    // the shared rule: it happens to be identical today only because the culler marks Mesh nodes,
    // whose component matrix is identity — a culler that ever tracked an Animator parent would
    // silently reintroduce the mismatch this rule exists to remove.
    if (ctx.culledNodes != nullptr && ctx.culledNodes->contains(this))
    {
        const Mat4 inherited = childWorld(world, world * componentModelMatrix(component_));
        for (auto& child : children_)
        {
            child->render(ctx, inherited);
        }
        return;
    }

    // Components that contribute to rendering (Animator, Mesh) define
    // render(ctx, world); the rest (Empty, Camera, Light) are no-ops that just
    // pass the world matrix down, handled here instead of each defining a
    // trivial render().
    // The component's world has NO NAME on purpose: it exists only as the argument to `childWorld`,
    // so it cannot be handed to the children by accident. An overridden node still emits its
    // component's render work — draws are recorded as usual — but does not let the result move
    // anything below it, the same rule `update` and `resolve` apply by returning early. (Nothing is
    // "advanced" here: `Animator::render` is pure, and an overridden node's `update` is skipped
    // entirely, so its animation clock does not run.)
    const Mat4 childWorldMatrix = childWorld(
        world,
        visitComponent(
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
            }));

    for (auto& child : children_)
    {
        child->render(ctx, childWorldMatrix);
    }
}

} // namespace fire_engine
