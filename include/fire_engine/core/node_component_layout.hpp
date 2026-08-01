#pragma once

#include <cstdint>
#include <string_view>

namespace fire_engine
{

class Node;

// How one glTF node's contents are laid out across engine nodes.
//
// A `Node` holds ONE component (`Components` is a variant: Empty/Animator/Camera/Mesh/Light), while
// a glTF node may legitimately carry a mesh, a light, a camera and animation channels at once. The
// engine therefore decomposes such a node into a parent that owns the TRANSFORM plus identity
// children for whatever else it carries.
//
// This was previously decided implicitly by the order of the attach calls, and the order was wrong
// in a way nothing reported: the light was attached while the variant still held `Empty`, then
// `emplace<Animator>()` (animated node) or `emplace<Mesh>()` (static mesh + light) overwrote it. No
// warning fired — the guard inside the light attach had already passed — so the light simply ceased
// to exist. `ShadowLodMotionDemo` lost its authored sun that way and rendered under the engine's
// fallback directional for every measurement taken on it.
//
// Making the layout an explicit value fixes the class of bug rather than the instance: the rule is
// stated once, tested exhaustively, and the loader cannot express "attach A then silently destroy
// it with B".
struct NodeComponentLayout
{
    // What the transform-owning node itself holds.
    enum class Primary : std::uint8_t
    {
        // Nothing of its own — a pure transform, or a node whose only content moved to a child.
        Empty,
        // Transform animation. The animated node MUST own this: every child inherits the animated
        // transform, which is exactly how an animated light or mesh follows its channel.
        Animator,
        // A static mesh with nothing competing for the slot.
        Mesh,
        // A light with nothing competing for the slot.
        Light,
        // A camera on a node that carries nothing else.
        Camera,
    };

    Primary primary{Primary::Empty};
    // Each of these means "this content exists and needs its own child node". A child is created
    // with an IDENTITY transform: the glTF transform stays on the parent, so animation drives the
    // mesh, the light and the camera together rather than any one of them individually.
    bool meshOnChild{false};
    bool lightOnChild{false};
    bool cameraOnChild{false};
};

// The decomposition rule. Precedence is by what CANNOT move: an Animator has to sit on the animated
// node, so it wins outright; a mesh takes the node only when no animator needs it; a light takes it
// only when nothing else does. A camera never claims the node when anything else is present, which
// is the behaviour the camera path already had and which this rule now states for all four.
//
// `hasTransformAnim` is specifically TRANSFORM animation (translation/rotation/scale). Weight-only
// animation drives morph targets through the Mesh component and needs no Animator of its own, so a
// weight-animated node still lays out as a plain mesh node.
[[nodiscard]] NodeComponentLayout planNodeComponents(bool hasTransformAnim, bool hasMesh,
                                                     bool hasLight, bool hasCamera) noexcept;

// Where each payload is to be attached, after the layout has been applied to a real node.
//
// The point of returning NODES rather than booleans: an attach site that reads a flag can still
// decide placement for itself, and then the planner's tests pass while production drifts. A site
// handed a target has nothing left to decide — and nothing left to get wrong when a future change
// reorders the calls.
//
// A null pointer means the glTF node did not declare that payload. It is never a fallback: if the
// layout says a light exists, `light` is non-null.
struct NodeComponentTargets
{
    Node* mesh{nullptr};
    Node* light{nullptr};
    Node* camera{nullptr};
};

// Applies a layout to `node`, creating exactly the identity-transform children it calls for, and
// returns the node each payload belongs on. Vulkan-free and scene-only, so the whole topology is
// testable headlessly — the production rule and the production node tree are the same code.
//
// `meshChildName` names the mesh child when one is needed (glTF meshes carry their own names, and
// the loader prefers them); empty falls back to `<node>_Mesh`. Light and camera children are always
// `<node>_Light` / `<node>_Camera`.
[[nodiscard]] NodeComponentTargets
materializeNodeComponentLayout(Node& node, const NodeComponentLayout& layout,
                               std::string_view meshChildName = {});

} // namespace fire_engine
