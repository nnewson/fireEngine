#include <fire_engine/core/node_component_layout.hpp>

#include <memory>
#include <string>

#include <fire_engine/scene/node.hpp>

namespace fire_engine
{

NodeComponentLayout planNodeComponents(bool hasTransformAnim, bool hasMesh, bool hasLight,
                                       bool hasCamera) noexcept
{
    NodeComponentLayout layout{};

    if (hasTransformAnim)
    {
        // The animator cannot move to a child: the child would then inherit an unanimated parent
        // transform and everything else on the node would stop following the animation.
        layout.primary = NodeComponentLayout::Primary::Animator;
        layout.meshOnChild = hasMesh;
        layout.lightOnChild = hasLight;
        layout.cameraOnChild = hasCamera;
        return layout;
    }

    if (hasMesh)
    {
        layout.primary = NodeComponentLayout::Primary::Mesh;
        layout.lightOnChild = hasLight;
        layout.cameraOnChild = hasCamera;
        return layout;
    }

    if (hasLight)
    {
        layout.primary = NodeComponentLayout::Primary::Light;
        layout.cameraOnChild = hasCamera;
        return layout;
    }

    // A camera alone sits on the node; with anything else it took a child already, before this rule
    // existed. An empty layout — a pure transform node — is the remaining case.
    layout.primary =
        hasCamera ? NodeComponentLayout::Primary::Camera : NodeComponentLayout::Primary::Empty;
    return layout;
}

NodeComponentTargets materializeNodeComponentLayout(Node& node, const NodeComponentLayout& layout,
                                                    std::string_view meshChildName)
{
    // Children are created with the default (identity) transform and never given one: the glTF
    // node's own transform stays on the parent, which is what makes an animated parent drive its
    // mesh, light and camera together instead of one of them.
    const auto addChild = [&node](std::string name) -> Node*
    { return &node.addChild(std::make_unique<Node>(std::move(name))); };

    NodeComponentTargets targets{};
    if (layout.primary == NodeComponentLayout::Primary::Mesh)
    {
        targets.mesh = &node;
    }
    else if (layout.meshOnChild)
    {
        targets.mesh =
            addChild(meshChildName.empty() ? node.name() + "_Mesh" : std::string(meshChildName));
    }

    if (layout.primary == NodeComponentLayout::Primary::Light)
    {
        targets.light = &node;
    }
    else if (layout.lightOnChild)
    {
        targets.light = addChild(node.name() + "_Light");
    }

    if (layout.primary == NodeComponentLayout::Primary::Camera)
    {
        targets.camera = &node;
    }
    else if (layout.cameraOnChild)
    {
        targets.camera = addChild(node.name() + "_Camera");
    }

    return targets;
}

} // namespace fire_engine
