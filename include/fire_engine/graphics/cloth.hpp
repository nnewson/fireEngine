#pragma once

#include <cstdint>
#include <span>
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

// XPBD distance constraint between two particles. compliance == 0 is fully rigid;
// larger is softer. The solver scales this by a global live multiplier, so the
// per-constraint value is the *base* stiffness — set per constraint type at build
// (structural/shear stiff, bend soft).
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
//
// `normalAdjOffsets` / `normalAdjTris` are a CSR adjacency the solver uses to
// recompute per-vertex normals at runtime from arbitrary topology (not just a
// grid): vertex v's incident triangles are normalAdjTris[normalAdjOffsets[v] ..
// normalAdjOffsets[v + 1]), each entry a triangle index into `indices`.
struct ClothMesh
{
    std::vector<ClothParticle> particles;
    std::vector<ClothConstraint> constraints;
    std::vector<uint32_t> colourRanges;
    std::vector<Vertex> vertices; // initial render vertices, 1:1 with particles
    std::vector<uint32_t> indices;
    std::vector<uint32_t> normalAdjOffsets; // CSR offsets, size particles + 1
    std::vector<uint32_t> normalAdjTris;    // flat triangle indices, one ring per vertex

    [[nodiscard]] uint32_t numColours() const noexcept
    {
        return colourRanges.empty() ? 0u : static_cast<uint32_t>(colourRanges.size() - 1);
    }
};

// Parameters for a procedural grid cloth. Compliance is split by constraint type
// so bend can be softer than the stretch-resisting structural/shear links —
// that's what makes the sheet drape like cloth rather than sheet metal.
struct ClothGridParams
{
    uint32_t resX{32};                // particles along local X
    uint32_t resZ{32};                // particles along local Z
    float spacing{0.05f};             // metres between adjacent particles
    float mass{1.0f};                 // total cloth mass, distributed across particles
    float structuralCompliance{0.0f}; // axis-aligned neighbours (0 = rigid)
    float shearCompliance{0.0f};      // diagonal neighbours
    float bendCompliance{1.0e-5f};    // skip-one neighbours (softer)
    bool pinTopCorners{true};         // pin the two far-Z corners (a hanging banner)
    Vec3 origin{};                    // world-space placement (cloth simulates in world space)
};

// Build a flat XZ-plane grid cloth in local space with structural + shear + bend
// distance constraints (each tagged with its per-type compliance), graph-coloured,
// with the normal adjacency filled. Render vertices mirror the particles (UVs
// across [0,1], up-facing normals; the solver recomputes normals at runtime).
[[nodiscard]] ClothMesh makeGridCloth(const ClothGridParams& params);

// Authoring parameters for cloth built from an arbitrary triangle mesh (glTF).
struct ClothMeshParams
{
    float structuralCompliance{0.0f}; // mesh edges (0 = rigid)
    float bendCompliance{1.0e-5f};    // across shared edges of adjacent triangles (softer)
    enum class Pin
    {
        None,       // fully free
        TopCorners, // the two extreme corners of the max-Z edge (local bounds)
        TopEdge,    // the whole max-Z edge
    } pin{Pin::None};
};

// Build a cloth from an arbitrary indexed triangle mesh: positions are welded by
// proximity into particles (shared edges become single constraints, so seams don't
// tear), structural constraints come from mesh edges and bend constraints from
// each interior edge's two opposite vertices. `vertices` carries the original
// render vertices (1:1 with positions, remapped to welded particle indices for the
// solver via the returned mesh's `indices`). Pinning resolves against local bounds.
[[nodiscard]] ClothMesh makeClothFromMesh(std::span<const Vertex> vertices,
                                          std::span<const uint32_t> indices,
                                          const ClothMeshParams& params);

// Greedy edge-colouring. Reorders `constraints` so each colour span shares no
// particle, and fills `colourRanges` (CSR offsets, size numColours + 1).
// `particleCount` sizes the adjacency scratch. Exposed for unit testing.
void colourConstraints(std::vector<ClothConstraint>& constraints,
                       std::vector<uint32_t>& colourRanges, uint32_t particleCount);

// Build the CSR per-vertex → incident-triangle adjacency the runtime normal pass
// reads. `indices` is a triangle list; fills offsets (size particleCount + 1) and
// the flat triangle-index array. Exposed for unit testing.
void buildNormalAdjacency(std::span<const uint32_t> indices, uint32_t particleCount,
                          std::vector<uint32_t>& offsets, std::vector<uint32_t>& tris);

} // namespace fire_engine
