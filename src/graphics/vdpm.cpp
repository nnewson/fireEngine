#include <fire_engine/graphics/vdpm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

namespace
{

// The finest triangle set in canonical space: each input triangle welded, degenerate (post-weld
// duplicate-vertex) triangles dropped.
[[nodiscard]] std::vector<std::array<std::uint32_t, 3>>
canonicalFaces(std::span<const std::uint32_t> weld, std::span<const std::uint32_t> indices)
{
    std::vector<std::array<std::uint32_t, 3>> faces;
    faces.reserve(indices.size() / 3);
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::array<std::uint32_t, 3> f{weld[indices[i]], weld[indices[i + 1]],
                                             weld[indices[i + 2]]};
        if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
        {
            continue;
        }
        faces.push_back(f);
    }
    return faces;
}

} // namespace

VertexForest buildVertexForest(std::span<const Vertex> vertices,
                               std::span<const MeshCollapse> collapses)
{
    const auto n = static_cast<std::uint32_t>(vertices.size());

    VertexForest forest;
    forest.vertexCount = vertices.size();
    forest.removingSplit.assign(n, kNoSplit);
    forest.splits.reserve(collapses.size());

    // The simplifier recorded each collapse's vsplit apexes (vl/vr) as it coarsened the true
    // canonical topology, so the forest is a faithful transcription of the stream — no
    // re-derivation by replay, no risk of diverging from the simplifier's actual decisions. A
    // collapse whose position-welded edge was non-manifold carries kNoCollapseApex for vl (the
    // fixed-arity vsplit can't encode >2 apexes); it is skipped, leaving `removed` a root (always
    // active) at that isolated spot — a conservative fallback that can't cascade into a topology
    // desync.
    for (const MeshCollapse& c : collapses)
    {
        if (c.vl == kNoCollapseApex)
        {
            continue;
        }
        const auto splitIndex = static_cast<std::uint32_t>(forest.splits.size());
        const std::uint32_t vr = c.vr == kNoCollapseApex ? kInvalidVertex : c.vr;
        forest.splits.push_back(VertexSplit{c.kept, c.removed, c.vl, vr, c.deviationRadius,
                                            c.uvDeviationRadius, c.normalDeviationRadius,
                                            c.tangentDeviationRadius});
        forest.removingSplit[c.removed] = splitIndex;
    }

    // Note: the deviation radii here are the simplifier's cumulative values (already accumulated up
    // the collapse tree), so no propagation pass is needed. Whether they are monotone across vl/vr
    // dependencies (not just endpoint ancestry) is checked by the [vdpm] tests, not assumed.
    return forest;
}

ActiveFront ActiveFront::build(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                               std::span<const MeshCollapse> collapses)
{
    ActiveFront front;
    front.forest_ = buildVertexForest(vertices, collapses);
    front.weld_ = mesh_topology::weldByPosition(vertices);
    front.finestFaces_ = canonicalFaces(front.weld_, indices);

    const std::size_t n = vertices.size();
    front.canonicalWedges_ =
        mesh_topology::canonicalWedges(front.weld_); // for seam-preserving emit

    // Coarsest state: only never-removed (root) canonical vertices are active; no split refined.
    front.active_.assign(n, 0);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        if (front.forest_.removingSplit[v] == kNoSplit)
        {
            front.active_[v] = 1;
        }
    }
    front.refined_.assign(front.forest_.splits.size(), 0);
    front.dependents_.assign(n, 0);
    return front;
}

std::uint32_t ActiveFront::activeAncestor(std::uint32_t canonicalVertex) const
{
    // A root is always active and has removingSplit == kNoSplit, so an inactive vertex always has a
    // valid removing split whose parent is one step nearer an active ancestor.
    std::uint32_t v = canonicalVertex;
    while (active_[v] == 0)
    {
        v = forest_.splits[forest_.removingSplit[v]].parent;
    }
    return v;
}

bool ActiveFront::refine(std::uint32_t splitIndex)
{
    if (splitIndex >= forest_.splits.size() || refined_[splitIndex] != 0)
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (active_[s.parent] == 0 || active_[s.vl] == 0 ||
        (s.vr != kInvalidVertex && active_[s.vr] == 0))
    {
        return false;
    }
    refined_[splitIndex] = 1;
    active_[s.child] = 1;
    ++dependents_[s.parent];
    ++dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        ++dependents_[s.vr];
    }
    return true;
}

bool ActiveFront::coarsen(std::uint32_t splitIndex)
{
    if (splitIndex >= forest_.splits.size() || refined_[splitIndex] == 0)
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (dependents_[s.child] != 0)
    {
        return false; // the child props up a refined split — not a leaf
    }
    refined_[splitIndex] = 0;
    active_[s.child] = 0;
    --dependents_[s.parent];
    --dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        --dependents_[s.vr];
    }
    return true;
}

void ActiveFront::refineAll()
{
    // Reverse stream order: a split's parent/vl/vr are removed only by later collapses, so undoing
    // coarsest-first guarantees each split's dependencies are already active.
    for (std::uint32_t i = static_cast<std::uint32_t>(forest_.splits.size()); i-- > 0;)
    {
        refine(i);
    }
}

void ActiveFront::coarsenAll()
{
    // Fixpoint: repeatedly collapse every currently-leaf refined split until none remain.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::uint32_t i = 0; i < forest_.splits.size(); ++i)
        {
            if (refined_[i] != 0 && coarsen(i))
            {
                changed = true;
            }
        }
    }
}

bool ActiveFront::forceRefine(std::uint32_t splitIndex)
{
    if (splitIndex >= forest_.splits.size())
    {
        return false;
    }
    if (refined_[splitIndex] != 0)
    {
        return true;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    // Bring in any dependency neighbourhood that isn't active yet. Parent is monotone so it is
    // usually already active; vl/vr are not, so this is where the non-monotonicity is absorbed.
    const std::array<std::uint32_t, 3> deps{s.parent, s.vl, s.vr};
    for (const std::uint32_t dep : deps)
    {
        if (dep == kInvalidVertex || active_[dep] != 0)
        {
            continue;
        }
        const std::uint32_t depSplit = forest_.removingSplit[dep];
        if (depSplit == kNoSplit || !forceRefine(depSplit))
        {
            return false;
        }
    }
    return refine(splitIndex);
}

void ActiveFront::refineForView(std::span<const Vertex> vertices, const Mat4& world,
                                const Vec3& cameraPos, float projScaleY, float viewportHeight,
                                float pixelBudget, float silhouetteBoost, float backfaceThreshold,
                                float uvScale, float normalScale, float tangentScale)
{
    coarsenAll();
    // New per-frame cycle: reset the repair diagnostics (repairFoldovers below + the caller's
    // repairCoverage accumulate into these).
    foldoversRepaired_ = 0;
    coverageRepaired_ = 0;
    const float halfViewport = viewportHeight * 0.5f;

    // Normal matrix = inverse-transpose of the world's linear part, so normals stay perpendicular
    // to the surface under NON-uniform scale / shear (`world · vec4(n,0)` skews them, flipping
    // facing and silhouette decisions). For a rigid / uniform-scale world this reduces to the
    // world's rotation.
    const Mat3 linear = Mat3::fromColumns({world[0, 0], world[1, 0], world[2, 0]},
                                          {world[0, 1], world[1, 1], world[2, 1]},
                                          {world[0, 2], world[1, 2], world[2, 2]});
    const Mat3 normalMatrix = linear.inverse().transpose();

    // Signed facing of one canonical vertex's world normal vs its own view direction: +1 toward the
    // camera, 0 edge-on, -1 away. Each vertex uses its own world position for the view direction. A
    // missing normal returns +1 (front-facing) so it never contributes to a back-face suppression.
    // Memoised per canonical vertex for this call: a vertex is a witness (child/parent/vl/vr) of
    // many splits, so it would otherwise be recomputed repeatedly. facingOf is a pure function of
    // the per-frame-constant world/cameraPos/normalMatrix, so the cache is behaviour-identical.
    facingCache_.assign(vertices.size(), 0.0f);
    facingValid_.assign(vertices.size(), 0);
    auto facingOf = [&](std::uint32_t v) -> float
    {
        if (facingValid_[v] != 0)
        {
            return facingCache_[v];
        }
        const Vec3 lp = vertices[v].position();
        const Vec4 wp = world * Vec4{lp.x(), lp.y(), lp.z(), 1.0f};
        const Vec3 wpos{wp.x(), wp.y(), wp.z()};
        const float d = std::max(1e-3f, (wpos - cameraPos).magnitude());
        const Vec3 wnrm = normalMatrix * vertices[v].normal();
        const float nlen = wnrm.magnitude();
        const float facing =
            nlen <= 1e-6f ? 1.0f
                          : Vec3::dotProduct(wnrm * (1.0f / nlen), (cameraPos - wpos) * (1.0f / d));
        facingCache_[v] = facing;
        facingValid_[v] = 1;
        return facing;
    };

    // Coarsest split first (reverse stream order): a split's parent/vl/vr are removed only by later
    // collapses, so refining coarse-first keeps every legal refine's dependencies already active.
    for (std::uint32_t i = static_cast<std::uint32_t>(forest_.splits.size()); i-- > 0;)
    {
        const VertexSplit& s = forest_.splits[i];

        const Vec3 local = vertices[s.child].position();
        const Vec4 wp4 = world * Vec4{local.x(), local.y(), local.z(), 1.0f};
        const Vec3 worldPos{wp4.x(), wp4.y(), wp4.z()};
        const float distance = std::max(1e-3f, (worldPos - cameraPos).magnitude());

        // Signed facing of the child rep's world normal vs the view direction: +1 toward the
        // camera, 0 edge-on (silhouette), -1 away (back). Drives the silhouette boost below.
        const float signedFacing = facingOf(s.child);

        // Back-face gate: skip *budget-driven* refinement of a split whose whole support region is
        // clearly back-facing — it is raster back-face-culled, so refining it is vertex work for no
        // visible pixels. This suppresses only discretionary refinement; forceRefine can still pull
        // a back-facing split in as the dependency of a visible/silhouette split.
        //
        // CONSERVATIVE multi-witness: a single smooth vertex normal is a poor stand-in for raster
        // back-face culling (which uses the geometric triangle winding), so one back-pointing child
        // normal must NOT suppress a split whose triangles are actually front-facing — that punched
        // a visible triangle out to the skybox. Test the child AND its dependency vertices (parent,
        // vl, vr — the split's support neighbourhood) and suppress only if EVERY witness is clearly
        // back-facing; if any faces the camera the split may protect a visible triangle, so keep it
        // eligible. (The exact test is a per-split facing cone; this 4-witness bound is the cheap
        // conservative version, and the threshold keeps near-edge-on reps silhouette-refined.)
        float maxFacing = signedFacing;
        maxFacing = std::max(maxFacing, facingOf(s.parent));
        if (s.vl != kInvalidVertex)
        {
            maxFacing = std::max(maxFacing, facingOf(s.vl));
        }
        if (s.vr != kInvalidVertex)
        {
            maxFacing = std::max(maxFacing, facingOf(s.vr));
        }
        if (maxFacing < -backfaceThreshold)
        {
            continue;
        }

        // Silhouette: near edge-on (|signedFacing| ~ 0) gets a tighter budget so contours stay
        // dense.
        const float boost = 1.0f + silhouetteBoost * (1.0f - std::abs(signedFacing));

        // Four independent channels, each projected e·projScaleY·(vh/2)/d pixels: geometry
        // (silhouette-boosted), UV texture stretch, shading-normal deviation, and tangent-frame
        // deviation (the normal/tangent pair also silhouette-boosted — a lighting error at a
        // grazing contour is the most visible). Any one over budget refines the split.
        const float geomScreenError = s.error * boost * projScaleY * halfViewport / distance;
        const float uvScreenError = s.uvError * uvScale * projScaleY * halfViewport / distance;
        const float normalScreenError =
            s.normalError * normalScale * boost * projScaleY * halfViewport / distance;
        const float tangentScreenError =
            s.tangentError * tangentScale * boost * projScaleY * halfViewport / distance;
        if (geomScreenError > pixelBudget || uvScreenError > pixelBudget ||
            normalScreenError > pixelBudget || tangentScreenError > pixelBudget)
        {
            forceRefine(i);
        }
    }

    repairFoldovers(vertices, world);
}

void ActiveFront::repairFoldovers(std::span<const Vertex> vertices, const Mat4& world)
{
    // Sweep the finest faces; where the active-ancestor replacement is wound against the original
    // face, force-refine the collapsed corners back in. Each repair only activates vertices (never
    // coarsens), so it strictly progresses toward the original geometry and terminates; refining
    // one face can re-fold a neighbour, so repeat until a full sweep finds nothing (bounded by the
    // forest depth — at worst the whole neighbourhood reaches full detail, which is flip-free).
    //
    // Winding is compared in WORLD space (not object space): the rasteriser culls on the post-world
    // winding, and a non-uniform-scale / mirroring world can flip the relative orientation of the
    // original vs replacement triangle, so an object-space test would mis-classify foldovers there.
    auto worldPos = [&](std::uint32_t v)
    {
        const Vec3 l = vertices[v].position();
        const Vec4 w = world * Vec4{l.x(), l.y(), l.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const std::array<std::uint32_t, 3>& fc : finestFaces_)
        {
            const std::uint32_t a0 = activeAncestor(fc[0]);
            const std::uint32_t a1 = activeAncestor(fc[1]);
            const std::uint32_t a2 = activeAncestor(fc[2]);
            if (a0 == a1 || a1 == a2 || a0 == a2)
            {
                continue; // legitimately collapsed to a degenerate — a neighbour covers it
            }
            const Vec3 p0 = worldPos(fc[0]);
            const Vec3 orig = Vec3::crossProduct(worldPos(fc[1]) - p0, worldPos(fc[2]) - p0);
            const Vec3 wa0 = worldPos(a0);
            const Vec3 repl = Vec3::crossProduct(worldPos(a1) - wa0, worldPos(a2) - wa0);
            if (Vec3::dotProduct(orig, repl) >= 0.0f)
            {
                continue; // replacement keeps the original winding — not a foldover
            }
            // Foldover: pull each collapsed corner back toward its finest position.
            for (const std::uint32_t c : fc)
            {
                if (active_[c] == 0 && forceRefine(forest_.removingSplit[c]))
                {
                    changed = true;
                    ++foldoversRepaired_;
                }
            }
        }
    }
}

void ActiveFront::repairCoverage(std::span<const Vertex> vertices, const Mat4& world,
                                 const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                                 float viewportHeight)
{
    auto worldPos = [&](std::uint32_t v)
    {
        const Vec3 l = vertices[v].position();
        const Vec4 w = world * Vec4{l.x(), l.y(), l.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };
    // NDC xy of a world point; false if behind the camera (w <= 0).
    auto toNdc = [&](const Vec3& wp, Vec2& out)
    {
        const Vec4 c = viewProj * Vec4{wp.x(), wp.y(), wp.z(), 1.0f};
        if (c.w() <= 1e-6f)
        {
            return false;
        }
        out = Vec2{c.x() / c.w(), c.y() / c.w()};
        return true;
    };
    // Is p inside triangle (a,b,c)? (same-sign edge functions; on-edge counts as inside).
    auto edge = [](const Vec2& a, const Vec2& b, const Vec2& p)
    { return ((p.s() - a.s()) * (b.t() - a.t())) - ((p.t() - a.t()) * (b.s() - a.s())); };
    auto inside = [&](const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& cc)
    {
        const float d0 = edge(a, b, p);
        const float d1 = edge(b, cc, p);
        const float d2 = edge(cc, a, p);
        return !((d0 < 0.0f || d1 < 0.0f || d2 < 0.0f) && (d0 > 0.0f || d1 > 0.0f || d2 > 0.0f));
    };

    // Force-refine every inactive corner of a face (un-collapse it back toward the finest). Returns
    // whether anything changed.
    auto refineCorners = [&](const std::array<std::uint32_t, 3>& fc)
    {
        bool did = false;
        for (const std::uint32_t c : fc)
        {
            if (active_[c] == 0 && forceRefine(forest_.removingSplit[c]))
            {
                did = true;
                ++coverageRepaired_;
            }
        }
        return did;
    };

    // Refine below this SCREEN area only when it's worth it: a couple of pixels. Expressed in px²
    // (resolution-independent), then converted to NDC — a triangle of NDC area A covers
    // A·(w/2)·(h/2) px², since NDC spans [-1,1]. A fixed NDC constant would mean different pixel
    // sizes per viewport and could skip visible holes on a high-res view.
    constexpr float kMinScreenAreaPx = 2.0f;
    const float minNdcArea =
        kMinScreenAreaPx / std::max(1.0f, 0.25f * viewportWidth * viewportHeight);

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const std::array<std::uint32_t, 3>& fc : finestFaces_)
        {
            const Vec3 w0 = worldPos(fc[0]);
            const Vec3 w1 = worldPos(fc[1]);
            const Vec3 w2 = worldPos(fc[2]);
            const Vec3 centroid = (w0 + w1 + w2) * (1.0f / 3.0f);
            const Vec3 gn = Vec3::crossProduct(w1 - w0, w2 - w0);
            if (Vec3::dotProduct(gn, cameraPos - centroid) <= 0.0f)
            {
                continue; // back-facing original: never a visible-coverage hole
            }
            // The ORIGINAL triangle's projected area gates whether a coverage miss is worth fixing.
            // A face straddling the near plane (some corners behind the camera) can't be projected
            // to test coverage, so refine it conservatively; one fully behind the camera isn't
            // visible.
            Vec2 s0;
            Vec2 s1;
            Vec2 s2;
            const int inFront = static_cast<int>(toNdc(w0, s0)) + static_cast<int>(toNdc(w1, s1)) +
                                static_cast<int>(toNdc(w2, s2));
            if (inFront == 0)
            {
                continue; // fully behind the camera — not visible
            }
            if (inFront < 3)
            {
                if (refineCorners(fc)) // straddles the near plane — can't test coverage, refine
                {
                    changed = true;
                }
                continue;
            }
            if (std::abs(edge(s0, s1, s2)) * 0.5f < minNdcArea)
            {
                continue; // sub-pixel: not a visible hole
            }
            const std::uint32_t a0 = activeAncestor(fc[0]);
            const std::uint32_t a1 = activeAncestor(fc[1]);
            const std::uint32_t a2 = activeAncestor(fc[2]);
            if (a0 == a1 || a1 == a2 || a0 == a2)
            {
                // DEGENERATE replacement: the face collapsed to a sliver and was dropped from the
                // emit. At a silhouette / high-curvature contour no neighbour covers its footprint,
                // so a front-facing non-trivial-area face here is a real hole — refine it back
                // (the earlier "a neighbour covers it" assumption is exactly wrong on a contour).
                if (refineCorners(fc))
                {
                    changed = true;
                }
                continue;
            }
            Vec2 sc;
            Vec2 sa0;
            Vec2 sa1;
            Vec2 sa2;
            if (!toNdc(centroid, sc) || !toNdc(worldPos(a0), sa0) || !toNdc(worldPos(a1), sa1) ||
                !toNdc(worldPos(a2), sa2))
            {
                // The replacement (or the centroid) straddles the near plane — can't test coverage,
                // so refine conservatively rather than silently skipping.
                if (refineCorners(fc))
                {
                    changed = true;
                }
                continue;
            }
            if (inside(sc, sa0, sa1, sa2))
            {
                continue; // the replacement still covers this face's centroid
            }
            // Coverage hole: refine the collapsed corner whose active ancestor is displaced
            // furthest on screen from its finest position — that is the corner whose recession
            // opened the gap.
            std::uint32_t worst = kInvalidVertex;
            float worstDisp = -1.0f;
            const std::array<std::uint32_t, 3> anc{a0, a1, a2};
            for (int k = 0; k < 3; ++k)
            {
                if (active_[fc[k]] != 0)
                {
                    continue; // this corner is already at finest — cannot displace
                }
                Vec2 sFine;
                Vec2 sAnc;
                if (!toNdc(worldPos(fc[k]), sFine) || !toNdc(worldPos(anc[k]), sAnc))
                {
                    continue;
                }
                const Vec2 d = sFine - sAnc;
                const float disp = Vec2::dotProduct(d, d);
                if (disp > worstDisp)
                {
                    worstDisp = disp;
                    worst = fc[k];
                }
            }
            if (worst != kInvalidVertex && forceRefine(forest_.removingSplit[worst]))
            {
                changed = true;
            }
        }
    }
}

std::vector<std::array<std::uint32_t, 3>> ActiveFront::emitActiveCanonical() const
{
    std::vector<std::array<std::uint32_t, 3>> out;
    out.reserve(finestFaces_.size());
    for (const std::array<std::uint32_t, 3>& f : finestFaces_)
    {
        const std::array<std::uint32_t, 3> a{activeAncestor(f[0]), activeAncestor(f[1]),
                                             activeAncestor(f[2])};
        if (a[0] == a[1] || a[1] == a[2] || a[0] == a[2])
        {
            continue; // the face collapsed to a degenerate at the current front
        }
        out.push_back(a);
    }
    return out;
}

void ActiveFront::emitActiveIndices(std::span<const Vertex> vertices,
                                    std::span<const uint32_t> indices,
                                    std::vector<std::uint32_t>& out) const
{
    out.clear();
    out.reserve(indices.size());

    // The front is settled at emit time, so activeAncestor(v) is stable — memoise it for every
    // canonical vertex once (a vertex is a corner of many faces) instead of re-walking the split
    // parent chain per corner. Reuses its capacity across frames.
    ancestorCache_.assign(active_.size(), 0);
    for (std::uint32_t c = 0; c < ancestorCache_.size(); ++c)
    {
        ancestorCache_[c] = activeAncestor(c);
    }

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::array<std::uint32_t, 3> oc{indices[i], indices[i + 1], indices[i + 2]};
        const std::array<std::uint32_t, 3> anc{ancestorCache_[weld_[oc[0]]],
                                               ancestorCache_[weld_[oc[1]]],
                                               ancestorCache_[weld_[oc[2]]]};
        if (anc[0] == anc[1] || anc[1] == anc[2] || anc[0] == anc[2])
        {
            continue; // collapsed to a degenerate at the current front
        }
        // Restore each corner to the nearest render wedge at its active-ancestor position, so a
        // seam corner keeps its own chart/shading identity instead of snapping to one canonical.
        for (std::size_t k = 0; k < 3; ++k)
        {
            out.push_back(
                mesh_topology::nearestWedge(vertices, canonicalWedges_[anc[k]], vertices[oc[k]]));
        }
    }
}

std::vector<std::uint32_t> ActiveFront::emitActiveIndices(std::span<const Vertex> vertices,
                                                          std::span<const uint32_t> indices) const
{
    std::vector<std::uint32_t> out;
    emitActiveIndices(vertices, indices, out);
    return out;
}

} // namespace fire_engine
