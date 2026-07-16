#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

// Independent VDPM front validators, shared by the sequential-oracle (ActiveFront) and the
// GPU-shaped parallel (ParallelFront) repair tests. They recompute the foldover / coverage
// invariants from FIRST PRINCIPLES over the mesh + the front's PUBLIC state (`forest()` +
// `active(v)`), so they never touch the repair implementation they judge — the same yardstick
// applied to both fronts. Templated on the front type: any front exposing `const VertexForest&
// forest()` and `bool active(std::uint32_t)` works.
//
// The COVERAGE geometry is deliberately kept independent of the runtime classifier (it must be an
// external check, not a re-run of the code under test), but it SHARES the one policy knob that is a
// tuning choice rather than geometry — `detail::kMinCoverageScreenAreaPx`, the min screen area a
// coverage hole must reach to count — so the validator's "worth-fixing" threshold tracks the
// runtime's exactly (a fixed NDC constant would drift from the viewport-derived runtime policy and
// could flag a face the runtime intentionally skipped, or vice-versa). It checks only PROJECTABLE
// coverage: a face any of whose corners (or whose replacement) is behind the near plane can't be
// projected to test containment, so it is left to the separate near-plane repair path and not
// counted here.

namespace fire_engine::test
{

// Count emitted triangles whose active-ancestor replacement winds AGAINST the original triangle — a
// foldover the rasteriser back-face-culls into a hole. World space matches what the rasteriser
// culls on. A degenerate replacement (collapsed to a sliver) is legitimately covered by a
// neighbour, not a foldover.
template <class Front>
[[nodiscard]] std::size_t foldoverCount(const Front& front, std::span<const Vertex> vertices,
                                        std::span<const std::uint32_t> indices, const Mat4& world)
{
    const VertexForest& f = front.forest();
    const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(vertices);
    auto ancestor = [&](std::uint32_t v)
    {
        while (!front.active(v))
        {
            v = f.splits[f.removingSplit[v]].parent;
        }
        return v;
    };
    auto wp = [&](std::uint32_t v)
    {
        const Vec3 p = vertices[v].position();
        const Vec4 w = world * Vec4{p.x(), p.y(), p.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };
    std::size_t folds = 0;
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::uint32_t o0 = indices[i];
        const std::uint32_t o1 = indices[i + 1];
        const std::uint32_t o2 = indices[i + 2];
        const std::uint32_t a0 = ancestor(weld[o0]);
        const std::uint32_t a1 = ancestor(weld[o1]);
        const std::uint32_t a2 = ancestor(weld[o2]);
        if (a0 == a1 || a1 == a2 || a0 == a2)
        {
            continue; // legitimately collapsed away
        }
        const Vec3 og = Vec3::crossProduct(wp(o1) - wp(o0), wp(o2) - wp(o0));
        const Vec3 rg = Vec3::crossProduct(wp(a1) - wp(a0), wp(a2) - wp(a0));
        if (Vec3::dotProduct(og, rg) < 0.0f)
        {
            ++folds;
        }
    }
    return folds;
}

// Count VISIBLE original triangles whose projected centroid is NOT covered by their active-ancestor
// replacement in NDC — a silhouette coverage hole (closed + non-folded, yet leaks the background).
// Follows the `rasterBackfaceCulling` policy: with culling ON only front-facing faces are visible;
// with it OFF (double-sided/blend) back-faces render too and count. Sub-pixel slivers below the
// shared screen-area policy don't count. Faces that can't be fully projected (near-plane straddle)
// are left to the near-plane path and skipped — "projectable coverage".
template <class Front>
[[nodiscard]] std::size_t coverageFailures(const Front& front, std::span<const Vertex> vertices,
                                           std::span<const std::uint32_t> indices,
                                           const Mat4& viewProj, const Vec3& cameraPos,
                                           const Mat4& world, float viewportWidth,
                                           float viewportHeight, bool rasterBackfaceCulling)
{
    const VertexForest& f = front.forest();
    const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(vertices);
    auto ancestor = [&](std::uint32_t v)
    {
        while (!front.active(v))
        {
            v = f.splits[f.removingSplit[v]].parent;
        }
        return v;
    };
    auto wp = [&](std::uint32_t v)
    {
        const Vec3 p = vertices[v].position();
        const Vec4 w = world * Vec4{p.x(), p.y(), p.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };
    auto ndc = [&](const Vec3& p, Vec2& out)
    {
        const Vec4 c = viewProj * Vec4{p.x(), p.y(), p.z(), 1.0f};
        if (c.w() <= 1e-6f)
        {
            return false;
        }
        out = Vec2{c.x() / c.w(), c.y() / c.w()};
        return true;
    };
    auto edge = [](const Vec2& a, const Vec2& b, const Vec2& p)
    { return ((p.s() - a.s()) * (b.t() - a.t())) - ((p.t() - a.t()) * (b.s() - a.s())); };
    auto inside = [&](const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c)
    {
        const float d0 = edge(a, b, p);
        const float d1 = edge(b, c, p);
        const float d2 = edge(c, a, p);
        return !((d0 < 0.0f || d1 < 0.0f || d2 < 0.0f) && (d0 > 0.0f || d1 > 0.0f || d2 > 0.0f));
    };
    // Same viewport-derived screen-area policy as the runtime classifier — shared constant, so the
    // "worth fixing" threshold is identical to what the repair applied.
    const float minNdcArea =
        detail::kMinCoverageScreenAreaPx / std::max(1.0f, 0.25f * viewportWidth * viewportHeight);

    std::size_t fails = 0;
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const Vec3 p0 = wp(indices[i]);
        const Vec3 p1 = wp(indices[i + 1]);
        const Vec3 p2 = wp(indices[i + 2]);
        const Vec3 ctr = (p0 + p1 + p2) * (1.0f / 3.0f);
        if (rasterBackfaceCulling &&
            Vec3::dotProduct(Vec3::crossProduct(p1 - p0, p2 - p0), cameraPos - ctr) <= 0.0f)
        {
            continue; // culled back-face — not visible
        }
        Vec2 s0;
        Vec2 s1;
        Vec2 s2;
        if (!ndc(p0, s0) || !ndc(p1, s1) || !ndc(p2, s2))
        {
            continue; // straddles the near plane — the near-plane path handles it, not coverage
        }
        if (std::abs(edge(s0, s1, s2)) * 0.5f < minNdcArea)
        {
            continue; // sub-pixel: not a visible hole
        }
        const std::uint32_t a0 = ancestor(weld[indices[i]]);
        const std::uint32_t a1 = ancestor(weld[indices[i + 1]]);
        const std::uint32_t a2 = ancestor(weld[indices[i + 2]]);
        if (a0 == a1 || a1 == a2 || a0 == a2)
        {
            ++fails; // degenerate replacement of a non-trivial visible face — a dropped hole
            continue;
        }
        Vec2 sc;
        Vec2 sa0;
        Vec2 sa1;
        Vec2 sa2;
        if (!ndc(ctr, sc) || !ndc(wp(a0), sa0) || !ndc(wp(a1), sa1) || !ndc(wp(a2), sa2))
        {
            continue; // replacement not projectable — near-plane path, not coverage
        }
        if (!inside(sc, sa0, sa1, sa2))
        {
            ++fails;
        }
    }
    return fails;
}

} // namespace fire_engine::test
