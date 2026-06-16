#pragma once

#include <cstdint>
#include <vector>

#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

enum class ClothColliderType : int32_t
{
    Plane = 0,
    Sphere = 1,
    Box = 2,
    Capsule = 3,
};

// A world-space collision primitive the cloth solver projects particles out of.
// Vulkan-free, but laid out to match the std140 element stride of the solver's
// collider UBO (64 bytes), so an array of these uploads with a plain memcpy. The
// a/b/c payload is interpreted per type — see the builders below.
struct ClothCollider
{
    float a[4]{};
    float b[4]{};
    float c[4]{};
    int32_t type{0};
    int32_t pad[3]{};
};

// type 0: a = (normal.xyz, offset)            — half-space dot(p,n) >= offset
[[nodiscard]] ClothCollider makePlaneCollider(Vec3 normal, float offset);
// type 1: a = (center.xyz, radius)
[[nodiscard]] ClothCollider makeSphereCollider(Vec3 center, float radius);
// type 2: a = center, b = halfExtents, c = orientation quat (x,y,z,w)
[[nodiscard]] ClothCollider makeBoxCollider(Vec3 center, Vec3 halfExtents, Quaternion orientation);
// type 3: a = (p0.xyz, radius), b = (p1.xyz, _) — segment p0..p1
[[nodiscard]] ClothCollider makeCapsuleCollider(Vec3 p0, Vec3 p1, float radius);

// One simulated cloth particle. invMass == 0 pins the particle (immovable).
struct ClothParticle
{
    Vec3 position{};
    float invMass{1.0f};
};

// XPBD distance constraint between two particles. compliance == 0 is fully rigid.
struct ClothConstraint
{
    uint32_t a{0};
    uint32_t b{0};
    float restLength{0.0f};
    float compliance{0.0f};
};

// CPU description of a cloth: simulated particles + distance constraints +
// renderable triangle mesh. Vulkan-free — the render layer builds GPU buffers
// from this. Constraints are graph-coloured into race-free batches so the GPU
// solver can run Gauss-Seidel per colour (no two constraints in a colour share a
// particle). `colourRanges` is CSR: colour c spans constraints
// [colourRanges[c], colourRanges[c + 1]).
struct ClothMesh
{
    std::vector<ClothParticle> particles;
    std::vector<ClothConstraint> constraints;
    std::vector<uint32_t> colourRanges;
    std::vector<Vertex> vertices; // initial render vertices, 1:1 with particles
    std::vector<uint32_t> indices;
    uint32_t resX{0};
    uint32_t resZ{0};

    [[nodiscard]] uint32_t numColours() const noexcept
    {
        return colourRanges.empty() ? 0u : static_cast<uint32_t>(colourRanges.size() - 1);
    }
};

// Parameters for a procedural grid cloth.
struct ClothGridParams
{
    uint32_t resX{32};        // particles along local X
    uint32_t resZ{32};        // particles along local Z
    float spacing{0.05f};     // metres between adjacent particles
    float mass{1.0f};         // total cloth mass, distributed across particles
    float compliance{0.0f};   // XPBD compliance (0 = rigid)
    bool pinTopCorners{true}; // pin the two far-Z corners (a hanging banner)
    Vec3 origin{};            // world-space placement (the cloth simulates in world space)
};

// Build a flat XZ-plane grid cloth in local space with structural + shear + bend
// distance constraints, graph-coloured. Render vertices mirror the particles
// (UVs across [0,1], up-facing normals; the solver recomputes normals at runtime).
[[nodiscard]] ClothMesh makeGridCloth(const ClothGridParams& params);

// Greedy edge-colouring. Reorders `constraints` so each colour span shares no
// particle, and fills `colourRanges` (CSR offsets, size numColours + 1).
// `particleCount` sizes the adjacency scratch. Exposed for unit testing.
void colourConstraints(std::vector<ClothConstraint>& constraints,
                       std::vector<uint32_t>& colourRanges, uint32_t particleCount);

} // namespace fire_engine
