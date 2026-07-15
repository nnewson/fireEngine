#include <fire_engine/graphics/vdpm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fire_engine/graphics/lod.hpp>
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

// Largest singular value of a 3x3 (its spectral norm): the greatest factor by which the matrix
// stretches any direction. Power iteration on mᵀm (symmetric PSD) converges to its largest
// eigenvalue; σ_max is its square root. EXACT for a uniform scale (mᵀm = s²I ⇒ σ = s); a
// CONSERVATIVE upper bound over all directions for non-uniform scale / shear. Lets an instance
// transform's object-space deviation/support radii be bounded into world space by one scalar, so a
// scaled instance refines correctly instead of using the unscaled object-space radius.
[[nodiscard]] float largestSingularValue(const Mat3& m) noexcept
{
    const Mat3 ata = m.transpose() * m;
    Vec3 v{1.0f, 1.0f, 1.0f};
    float lambda = 0.0f;
    for (int iter = 0; iter < 24; ++iter)
    {
        const Vec3 av = ata * v;
        const float len = av.magnitude();
        if (len <= 1e-20f)
        {
            return 0.0f;
        }
        v = av * (1.0f / len);
        lambda = Vec3::dotProduct(v, ata * v);
    }
    return std::sqrt(std::max(0.0f, lambda));
}

} // namespace

namespace detail
{

ConeVisibility coneVisibility(const Vec3& coneAxis, float coneCos, float supportRadius,
                              const Vec3& regionCenter, const Vec3& cameraObj, float facingSign,
                              bool cullEnabled) noexcept
{
    const Vec3 toCam = cameraObj - regionCenter;
    const float d = toCam.magnitude();
    if (d <= supportRadius || d < 1e-6f)
    {
        return {false, 1.0f}; // camera within the support sphere ⇒ potential silhouette, never cull
    }
    // Combined half-angle θn + θv via the cosine sum identity — no acos/asin/sin (GPU-friendly).
    // cosN is the stored cone cosine (≤ 0 is the no-cull sentinel: θn ≥ 90° ⇒ cosSum ≤ 0 below, so
    // it falls through to "never cull, max straddle" uniformly); sinV = r/d is the sphere's view
    // spread.
    const float cosN = std::clamp(coneCos, -1.0f, 1.0f);
    const float sinN = std::sqrt(std::max(0.0f, 1.0f - (cosN * cosN)));
    const float sinV = std::clamp(supportRadius / d, 0.0f, 1.0f);
    const float cosV = std::sqrt(std::max(0.0f, 1.0f - (sinV * sinV)));
    const float cosSum = (cosN * cosV) - (sinN * sinV); // cos(θn + θv)
    const float sinSum = (sinN * cosV) + (cosN * sinV); // sin(θn + θv)
    if (cosSum <= 0.0f)
    {
        return {false, 1.0f}; // θn + θv ≥ 90° ⇒ cone + spread straddles edge-on, never cull
    }
    // facingSign folds in a reflection (negative-determinant world flips winding ⇒ the
    // raster-culled side is the object-space FRONT); +1 for a normal transform. The cone + sphere
    // are already conservative bounds, so the cull needs only a float-safety margin (NOT the old
    // smooth-normal era's 0.5 heuristic): cosAC and sinSum each carry O(few·ε) rounding, so 16·ε
    // safely dominates that while never widening the cull materially.
    constexpr float safety = 16.0f * std::numeric_limits<float>::epsilon();
    const float cosAC = facingSign * Vec3::dotProduct(coneAxis, toCam * (1.0f / d));
    const bool backFacing = cullEnabled && cosAC < -sinSum - safety;
    // straddle = how centrally edge-on sits in the cone+spread. When the cone+spread has zero width
    // (sinSum ≈ 0: a flat cone with no view spread), the region is a silhouette iff it is itself
    // exactly edge-on (|cosAC| ≈ 0) — otherwise fully one-sided.
    const float straddle = sinSum > safety
                               ? std::clamp((sinSum - std::abs(cosAC)) / sinSum, 0.0f, 1.0f)
                               : (std::abs(cosAC) <= safety ? 1.0f : 0.0f);
    return {backFacing, straddle};
}

} // namespace detail

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
                                            c.tangentDeviationRadius, c.supportRadius,
                                            c.normalConeAxis, c.normalConeCos});
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
                                float pixelBudget, float silhouetteBoost,
                                bool rasterBackfaceCulling, float uvScale, float normalScale,
                                float tangentScale)
{
    // The active front PERSISTS across frames — no coarsenAll(). The score pass below tags every
    // split for this view; a refine pass then pulls in splits over the pixel budget, and a coarsen
    // pass drops those under kVdpmCoarsenRatio × budget. The dead band between the two thresholds
    // is the hysteresis that stops a split whose score hovers at the budget (small camera moves,
    // TAA jitter) from popping in and out each frame. New per-frame cycle: reset the score
    // diagnostics + per-split scratch. (The repair counters reset in repairFront — it owns them.)
    channelStats_ = ChannelStats{};
    splitScore_.assign(forest_.splits.size(), 0.0f);
    splitBackface_.assign(forest_.splits.size(), 0);
    const float halfViewport = viewportHeight * 0.5f;

    const Mat3 linear = Mat3::fromColumns({world[0, 0], world[1, 0], world[2, 0]},
                                          {world[0, 1], world[1, 1], world[2, 1]},
                                          {world[0, 2], world[1, 2], world[2, 2]});

    // The deviation + support radii are OBJECT-space lengths, but the split is projected against a
    // WORLD-space distance, so a mesh instanced at a non-unit world scale must have its radii
    // bounded into world space or it would refine as if unscaled (a 10× instance under-refining).
    // The largest singular value of the linear part is that bound — exact for uniform scale,
    // conservative for non-uniform. (UV error is texture-space, not a world length, so it is NOT
    // scaled.)
    const float worldLengthScale = largestSingularValue(linear);

    // Per-split visibility from the precomputed normal cone (built from FACE normals — what the
    // rasteriser back-face-culls on) runs in OBJECT space: the SIGN of dot(worldNormal,
    // worldViewDir) equals the sign of dot(objNormal, objViewDir) for any linear world transform
    // (the normal's M⁻ᵀ and the view direction's M cancel), so back-facing is exact under
    // non-uniform scale / shear AND the stored cone stays circular (a world-space test would shear
    // it). Transform the camera into object space once. A near-singular world has no reliable
    // inverse — then the cone is unusable, so disable culling and treat every split as a potential
    // silhouette. The conditioning test is SCALE-INVARIANT: |det| = σ₁σ₂σ₃, so |det|/σ_max³ ∈ [0,1]
    // is a pure shape measure (≈1 well-conditioned, →0 as an axis collapses) — unlike an absolute
    // |det| threshold, which would wrongly reject a tiny-but-uniform instance (e.g. scale 1e-4).
    // Otherwise fold the determinant SIGN into the facing (a reflection flips winding, so the
    // raster-culled side is the object-space front) — `coneVisibility` handles it exactly.
    const float det = linear.determinant();
    const float sigmaMax = worldLengthScale;
    const bool coneUsable = std::abs(det) > 1e-6f * sigmaMax * sigmaMax * sigmaMax;
    const bool coneCullEnabled = rasterBackfaceCulling && coneUsable;
    const float facingSign = det >= 0.0f ? 1.0f : -1.0f;
    const Vec3 worldTranslation{world[0, 3], world[1, 3], world[2, 3]};
    const Vec3 cameraObj = coneUsable ? linear.inverse() * (cameraPos - worldTranslation) : Vec3{};
    auto visibilityOf = [&](const VertexSplit& s) -> detail::ConeVisibility
    {
        if (!coneUsable)
        {
            return {false, 1.0f}; // singular world: never cull, max potential silhouette
        }
        return detail::coneVisibility(s.normalConeAxis, s.normalConeCos, s.supportRadius,
                                      vertices[s.parent].position(), cameraObj, facingSign,
                                      coneCullEnabled);
    };

    // Score pass: tag every split with its max screen-space channel score + back-face flag for this
    // view (order-independent — no front mutation here).
    for (std::uint32_t i = 0; i < forest_.splits.size(); ++i)
    {
        const VertexSplit& s = forest_.splits[i];

        const Vec3 local = vertices[s.child].position();
        const Vec4 wp4 = world * Vec4{local.x(), local.y(), local.z(), 1.0f};
        const Vec3 worldPos{wp4.x(), wp4.y(), wp4.z()};
        const float distance = std::max(1e-3f, (worldPos - cameraPos).magnitude());

        // The support sphere is centred on `kept` (= s.parent), so its footprint must project from
        // the PARENT's world position, not the child's — otherwise the radius and the projection
        // centre describe different spheres and a large collapse mis-estimates its footprint.
        const Vec3 parentLocal = vertices[s.parent].position();
        const Vec4 pw4 = world * Vec4{parentLocal.x(), parentLocal.y(), parentLocal.z(), 1.0f};
        const float parentDistance =
            std::max(1e-3f, (Vec3{pw4.x(), pw4.y(), pw4.z()} - cameraPos).magnitude());

        // Back-face gate: skip *budget-driven* refinement of a split whose whole support region is
        // PROVABLY raster back-face-culled (refining it is vertex work for no visible pixels).
        // Suppresses only discretionary refinement; forceRefine can still pull a back-facing split
        // in as the dependency of a visible/silhouette split. This is an exact evaluation of a
        // conservative bound (the cone + support sphere over-bound the surface), so `backFacing` is
        // a one-sided proof of hiddenness — never a claim that a non-back-facing split is visible.
        const detail::ConeVisibility vis = visibilityOf(s);
        if (vis.backFacing)
        {
            splitBackface_[i] = 1; // culled: score 0, skip refine, eligible to coarsen
            continue;
        }

        // Silhouette: a cone straddling edge-on gets a tighter budget so contours stay dense.
        const float boost = 1.0f + silhouetteBoost * vis.straddle;

        // Four independent channels in screen pixels; any one over budget refines the split. The
        // GEOMETRIC channels are world-length deviations projected e·projScaleY·(vh/2)/d: geometry
        // (silhouette-boosted; its object-space radius bounded into world space by worldLengthScale
        // so a scaled instance refines correctly) and UV (texture-space, so NOT world-scaled). The
        // ANGULAR channels (normal/tangent, radians) matter in proportion to the projected SCREEN
        // EXTENT of the region they affect, so they project the split's world-space support radius
        // as that same geometric extent and multiply by the chord 2·sin(θ/2) — the dimensionless
        // angular magnitude (≈θ small-angle, saturating at 2 for a flipped normal). This makes the
        // shading score scale exactly like geometry (angle dimensionless, extent scales with the
        // mesh), unlike the old fixed-length projection which shrank the shading score when model +
        // camera scaled together. The support extent uses the PARENT-centred near-sphere depth
        // (parentDistance − worldSupport), a conservative projection of a sphere centred on kept.
        // Both angular channels are silhouette-boosted (a lighting error at a grazing contour is
        // the most visible).
        const float worldSupport = s.supportRadius * worldLengthScale;
        const float geomScreenError =
            s.error * worldLengthScale * boost * projScaleY * halfViewport / distance;
        const float uvScreenError = s.uvError * uvScale * projScaleY * halfViewport / distance;
        const float nearDistance = std::max(1e-3f, parentDistance - worldSupport);
        const float extent = worldSupport * projScaleY * halfViewport / nearDistance;
        const float normalScreenError =
            2.0f * std::sin(0.5f * s.normalError) * normalScale * boost * extent;
        const float tangentScreenError =
            2.0f * std::sin(0.5f * s.tangentError) * tangentScale * boost * extent;

        // Metric instrumentation: track how hard each channel is pushing (max ratio over all
        // splits) regardless of whether this split refines, so an under-firing channel is still
        // visible.
        const float invBudget = pixelBudget > 0.0f ? 1.0f / pixelBudget : 0.0f;
        channelStats_.maxGeometryRatio =
            std::max(channelStats_.maxGeometryRatio, geomScreenError * invBudget);
        channelStats_.maxUvRatio = std::max(channelStats_.maxUvRatio, uvScreenError * invBudget);
        channelStats_.maxNormalRatio =
            std::max(channelStats_.maxNormalRatio, normalScreenError * invBudget);
        channelStats_.maxTangentRatio =
            std::max(channelStats_.maxTangentRatio, tangentScreenError * invBudget);

        splitScore_[i] =
            std::max({geomScreenError, uvScreenError, normalScreenError, tangentScreenError});

        // A split that is NOT yet refined and is over the budget will be pulled in by the refine
        // pass below — attribute that new *trigger* to its winning channel (the one furthest over
        // budget), so the counts read as "which channel is driving new detail this frame".
        // Already-refined splits that merely stay refined are not re-counted (steady-state ⇒ few
        // triggers, which is the point of persistence). The max ratios above track every split,
        // refined or not.
        if (refined_[i] == 0 && splitScore_[i] > pixelBudget)
        {
            const float winner = splitScore_[i];
            if (winner == geomScreenError)
            {
                ++channelStats_.geometryTriggers;
            }
            else if (winner == uvScreenError)
            {
                ++channelStats_.uvTriggers;
            }
            else if (winner == normalScreenError)
            {
                ++channelStats_.normalTriggers;
            }
            else
            {
                ++channelStats_.tangentTriggers;
            }
        }
    }

    // Refine pass: coarse-first (reverse stream order, so a split's parent/vl/vr are already
    // active), pull in every front-facing split over the budget that isn't already refined.
    // forceRefine brings any inactive dependency neighbourhood with it.
    for (std::uint32_t i = static_cast<std::uint32_t>(forest_.splits.size()); i-- > 0;)
    {
        if (splitBackface_[i] == 0 && refined_[i] == 0 && splitScore_[i] > pixelBudget)
        {
            forceRefine(i);
        }
    }

    // Coarsen pass: a fine-first fixpoint collapsing every refined split now under the coarsen
    // budget (or back-face-culled) whose child is a leaf. A split can't coarsen until the finer
    // splits that depend on it have, so repeat until a full sweep changes nothing (bounded by the
    // forest depth). Only removes sub-band detail, so the front strictly relaxes toward the current
    // score.
    const float coarsenBudget = pixelBudget * kVdpmCoarsenRatio;
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::uint32_t i = 0; i < forest_.splits.size(); ++i)
        {
            if (refined_[i] != 0 && (splitBackface_[i] != 0 || splitScore_[i] < coarsenBudget) &&
                coarsen(i))
            {
                changed = true;
            }
        }
    }
    // Repairs are NOT run here: refineForView only scores + refines/coarsens. The foldover and
    // coverage repairs run together in repairFront (mandatory before emission), because they must
    // reach a JOINT fixed point — coverage force-refines can re-fold, and vice-versa.
}

ActiveFront::RepairSweepResult ActiveFront::repairFoldoversSweep(std::span<const Vertex> vertices,
                                                                 const Mat4& world)
{
    // ONE sweep of the finest faces: where the active-ancestor replacement is wound against the
    // original face, force-refine the collapsed corners back in. Only activates (never coarsens),
    // so it strictly progresses; refining one face can re-fold a neighbour, so the joint loop in
    // repairFront repeats this sweep until a full cycle changes nothing.
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
    RepairSweepResult result;
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
        // Foldover: pull each collapsed (inactive) corner back toward its finest position. An
        // inactive corner whose valid removing split fails to force-refine is a forest
        // inconsistency.
        for (const std::uint32_t c : fc)
        {
            if (active_[c] != 0)
            {
                continue; // already at finest here — nothing to advance
            }
            if (forceRefine(forest_.removingSplit[c]))
            {
                result.changed = true;
                ++foldoversRepaired_;
            }
            else
            {
                result.failedToProgress = true;
            }
        }
    }
    return result;
}

ActiveFront::RepairSweepResult
ActiveFront::repairCoverageSweep(std::span<const Vertex> vertices, const Mat4& world,
                                 const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                                 float viewportHeight, bool rasterBackfaceCulling)
{
    RepairSweepResult result;
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

    // The ONE place a coverage repair force-refines a corner + accounts for it, so the counter and
    // the sweep result can't diverge between the paths that call it (near-plane / degenerate
    // corners vs the worst-displaced corner). A corner already at finest is a no-op (clean — e.g. a
    // full-detail face that still straddles the near plane); an INACTIVE corner whose valid
    // removing split fails is a forest inconsistency ⇒ failedToProgress.
    auto refineCoverageCorner = [&](std::uint32_t c)
    {
        if (active_[c] != 0)
        {
            return;
        }
        if (forceRefine(forest_.removingSplit[c]))
        {
            result.changed = true;
            ++coverageRepaired_;
        }
        else
        {
            result.failedToProgress = true;
        }
    };
    auto refineCorners = [&](const std::array<std::uint32_t, 3>& fc)
    {
        for (const std::uint32_t c : fc)
        {
            refineCoverageCorner(c);
        }
    };

    // Refine below this SCREEN area only when it's worth it: a couple of pixels. Expressed in px²
    // (resolution-independent), then converted to NDC — a triangle of NDC area A covers
    // A·(w/2)·(h/2) px², since NDC spans [-1,1]. A fixed NDC constant would mean different pixel
    // sizes per viewport and could skip visible holes on a high-res view.
    constexpr float kMinScreenAreaPx = 2.0f;
    const float minNdcArea =
        kMinScreenAreaPx / std::max(1.0f, 0.25f * viewportWidth * viewportHeight);

    for (const std::array<std::uint32_t, 3>& fc : finestFaces_)
    {
        const Vec3 w0 = worldPos(fc[0]);
        const Vec3 w1 = worldPos(fc[1]);
        const Vec3 w2 = worldPos(fc[2]);
        const Vec3 centroid = (w0 + w1 + w2) * (1.0f / 3.0f);
        const Vec3 gn = Vec3::crossProduct(w1 - w0, w2 - w0);
        if (rasterBackfaceCulling && Vec3::dotProduct(gn, cameraPos - centroid) <= 0.0f)
        {
            // Back-facing original AND the draw culls back-faces ⇒ the rasteriser never shows it,
            // so it can't leak. A double-sided / blended draw (rasterBackfaceCulling false) renders
            // this face, so it still needs coverage — fall through. `gn` is the WORLD winding
            // (built from world positions), so this stays correct under a reflected
            // (negative-determinant) world, which flips which side faces the camera.
            continue;
        }
        // The ORIGINAL triangle's projected area gates whether a coverage miss is worth fixing. A
        // face straddling the near plane (some corners behind the camera) can't be projected to
        // test coverage, so refine it conservatively; one fully behind the camera isn't visible.
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
            refineCorners(fc); // straddles the near plane — can't test coverage, refine
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
            // DEGENERATE replacement: the face collapsed to a sliver and was dropped from the emit.
            // At a silhouette / high-curvature contour no neighbour covers its footprint, so a
            // visible non-trivial-area face here is a real hole — refine it back (the earlier "a
            // neighbour covers it" assumption is exactly wrong on a contour).
            refineCorners(fc);
            continue;
        }
        Vec2 sc;
        Vec2 sa0;
        Vec2 sa1;
        Vec2 sa2;
        if (!toNdc(centroid, sc) || !toNdc(worldPos(a0), sa0) || !toNdc(worldPos(a1), sa1) ||
            !toNdc(worldPos(a2), sa2))
        {
            // The replacement (or the centroid) straddles the near plane — can't test coverage, so
            // refine conservatively rather than silently skipping.
            refineCorners(fc);
            continue;
        }
        if (inside(sc, sa0, sa1, sa2))
        {
            continue; // the replacement still covers this face's centroid
        }
        // Coverage hole: refine the collapsed corner whose active ancestor is displaced furthest on
        // screen from its finest position — that is the corner whose recession opened the gap.
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
        // worst == kInvalidVertex means every corner is already at finest (the face covers itself)
        // — clean. Otherwise refine it through the SAME helper as the other paths (the main
        // coverage-hole path), so the counter and result accounting can't diverge between branches.
        if (worst != kInvalidVertex)
        {
            refineCoverageCorner(worst);
        }
    }
    return result;
}

void ActiveFront::repairFront(std::span<const Vertex> vertices, const Mat4& world,
                              const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                              float viewportHeight, bool rasterBackfaceCulling)
{
    // The repair API owns its diagnostics — reset here so an independent call never shows stale
    // counts (refineForView no longer touches them).
    foldoversRepaired_ = 0;
    coverageRepaired_ = 0;
    jointRepairSweeps_ = 0;

    // Each sweep only ACTIVATES splits (refinement-only / inflationary), so a cycle that changes
    // anything strictly grows the refined set — the loop can run at most (initially-unrefined + 1)
    // cycles, the +1 being the final cycle that proves convergence. Capture that bound to hard-fail
    // rather than hang if a future change ever breaks the inflationary property.
    const auto unrefined =
        static_cast<std::uint32_t>(std::ranges::count(refined_, static_cast<std::uint8_t>(0)));

    // JOINT fixed point: alternate a foldover sweep and a coverage sweep until a COMPLETE cycle
    // refines nothing — a coverage force-refine can re-fold a neighbour, and a foldover
    // force-refine can open a coverage hole, so a single phase order does not leave the front clean
    // of both.
    RepairSweepResult r;
    do
    {
        r = repairFoldoversSweep(vertices, world);
        const RepairSweepResult cov =
            repairCoverageSweep(vertices, world, cameraPos, viewProj, viewportWidth, viewportHeight,
                                rasterBackfaceCulling);
        r.changed = r.changed || cov.changed;
        r.failedToProgress = r.failedToProgress || cov.failedToProgress;
        ++jointRepairSweeps_;
        if (r.failedToProgress)
        {
            // A repairable violation had an inactive target whose valid removing split failed to
            // force-refine — a forest/logic inconsistency, not a "can't refine at full detail"
            // case.
            throw std::logic_error(
                "VDPM repairFront: a repairable violation could not be advanced");
        }
        if (jointRepairSweeps_ > unrefined + 1)
        {
            // Unreachable while the sweeps stay refinement-only — a guard that turns a broken
            // inflationary property into a diagnosis instead of a hang.
            throw std::logic_error("VDPM repairFront: exceeded the inflationary sweep bound");
        }
    } while (r.changed);
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
