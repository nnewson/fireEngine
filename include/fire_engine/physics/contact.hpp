#pragma once

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// (The earlier handle-based Contact / ContactManifold stubs were never populated
// and have been removed — the real geometry manifold lives in
// collision/contact_manifold.hpp. This header now carries only DebugContact.)

// Lightweight per-step contact record for debug visualisation only. Now sourced
// from the shape-specific `ContactManifold` (one `DebugContact` per manifold
// point, with the manifold normal); captured before `applyResponses` advances
// bodies. Read-only — it does not affect the simulation. Vulkan-free.
struct DebugContact
{
    Vec3 point{};
    Vec3 normal{};
};

} // namespace fire_engine
