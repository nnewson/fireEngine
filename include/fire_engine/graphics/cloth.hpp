#pragma once

#include <cstdint>
#include <vector>

#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

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
