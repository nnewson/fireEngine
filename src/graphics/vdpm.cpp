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

// A CONSERVATIVE bound on the largest singular value of a 3x3 (its spectral norm): the greatest
// factor by which the matrix stretches any direction. σ_max(m) = sqrt(λ_max(mᵀm)); bound
// λ_max(mᵀm) by the Gershgorin / induced-∞-norm bound ‖mᵀm‖∞ (the max absolute row sum of the
// symmetric Gram matrix), which for any symmetric PSD matrix is a GUARANTEED upper bound on its
// spectral radius. So σ_max is never UNDER-estimated — the object-space deviation/support radii
// bounded into world space stay conservative (an instance never under-refines) and the conditioning
// test never wrongly trusts a near-singular transform. EXACT for orthogonal (mᵀm = I) and diagonal
// (scale) transforms; a modest over-estimate for a sheared/rotated mix (off-diagonal mass inflates
// the row sum), which only over-refines — the safe direction.
//
// The Gram matrix + row sums are computed in DOUBLE and the float result is outward-rounded one ULP
// toward +∞, so float rounding of the products can't push the answer below the true bound (the
// bound is only conservative in exact arithmetic otherwise).
//
// A finite power iteration is NOT usable here: a fixed start vector orthogonal to the dominant
// eigenvector converges to the WRONG (smaller) eigenvalue no matter how many iterations run — e.g.
// the in-plane shear [[5.5,-4.5,0],[-4.5,5.5,0],[0,0,1]] has σ_max = 10 but its dominant mᵀm
// eigenvector (1,-1,0) is orthogonal to (1,1,1), so power iteration returns 1 (a 10×
// under-estimate).
[[nodiscard]] float largestSingularValue(const Mat3& m) noexcept
{
    double maxRowSum = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        double rowSum = 0.0;
        for (int j = 0; j < 3; ++j)
        {
            double gram = 0.0; // (mᵀm)_ij = Σ_k m_ki · m_kj
            for (int k = 0; k < 3; ++k)
            {
                gram += static_cast<double>(m[k, i]) * static_cast<double>(m[k, j]);
            }
            rowSum += std::abs(gram);
        }
        maxRowSum = std::max(maxRowSum, rowSum);
    }
    const double sigma = std::sqrt(std::max(0.0, maxRowSum));
    // Outward round: the returned float is the smallest float >= the double bound >= true σ_max.
    return std::nextafter(static_cast<float>(sigma), std::numeric_limits<float>::infinity());
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

bool isFoldover(const std::array<Vec3, 3>& original, const std::array<Vec3, 3>& replacement,
                bool replacementDegenerate) noexcept
{
    if (replacementDegenerate)
    {
        return false; // legitimately collapsed to a degenerate — a neighbour covers it
    }
    // Winding compared in WORLD space: the rasteriser culls on the post-world winding, and a
    // non-uniform-scale / mirroring world can flip the relative orientation of the original vs
    // replacement triangle, so an object-space test would mis-classify foldovers there.
    const Vec3 orig = Vec3::crossProduct(original[1] - original[0], original[2] - original[0]);
    const Vec3 repl =
        Vec3::crossProduct(replacement[1] - replacement[0], replacement[2] - replacement[0]);
    return Vec3::dotProduct(orig, repl) < 0.0f;
}

CoverageRepair
classifyCoverageRepair(const std::array<Vec3, 3>& original, const std::array<Vec3, 3>& replacement,
                       bool replacementDegenerate, const std::array<bool, 3>& cornerInactive,
                       const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                       float viewportHeight, bool rasterBackfaceCulling) noexcept
{
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

    // When no corner is inactive, the face is already at full detail here: the conservative
    // "refine every inactive corner" outcomes have no target and collapse to a clean None no-op.
    const bool anyInactive = cornerInactive[0] || cornerInactive[1] || cornerInactive[2];
    const CoverageRepair allInactive =
        anyInactive ? CoverageRepair{CoverageRepairKind::AllInactiveCorners, kInvalidCorner}
                    : CoverageRepair{};

    const Vec3 centroid = (original[0] + original[1] + original[2]) * (1.0f / 3.0f);
    const Vec3 gn = Vec3::crossProduct(original[1] - original[0], original[2] - original[0]);
    if (rasterBackfaceCulling && Vec3::dotProduct(gn, cameraPos - centroid) <= 0.0f)
    {
        // Back-facing original AND the draw culls back-faces ⇒ the rasteriser never shows it, so it
        // can't leak. A double-sided / blended draw renders it and still needs coverage — fall
        // through. `gn` is the WORLD winding, so this stays correct under a reflected world.
        return {};
    }
    // The ORIGINAL triangle's projected area gates whether a coverage miss is worth fixing. A face
    // straddling the near plane (some corners behind the camera) can't be projected to test
    // coverage, so refine it conservatively; one fully behind the camera isn't visible.
    Vec2 s0;
    Vec2 s1;
    Vec2 s2;
    const int inFront = static_cast<int>(toNdc(original[0], s0)) +
                        static_cast<int>(toNdc(original[1], s1)) +
                        static_cast<int>(toNdc(original[2], s2));
    if (inFront == 0)
    {
        return {}; // fully behind the camera — not visible
    }
    if (inFront < 3)
    {
        return allInactive; // straddles the near plane — can't test coverage, refine
    }
    // Refine below this SCREEN area only when it's worth it: a couple of pixels. Expressed in px²
    // (resolution-independent), then converted to NDC — a triangle of NDC area A covers
    // A·(w/2)·(h/2) px², since NDC spans [-1,1].
    const float minNdcArea =
        kMinCoverageScreenAreaPx / std::max(1.0f, 0.25f * viewportWidth * viewportHeight);
    if (std::abs(edge(s0, s1, s2)) * 0.5f < minNdcArea)
    {
        return {}; // sub-pixel: not a visible hole
    }
    if (replacementDegenerate)
    {
        // DEGENERATE replacement: the face collapsed to a sliver and was dropped from the emit. At
        // a silhouette / high-curvature contour no neighbour covers its footprint, so a visible
        // non-trivial-area face here is a real hole — refine it back.
        return allInactive;
    }
    Vec2 sc;
    Vec2 sa0;
    Vec2 sa1;
    Vec2 sa2;
    if (!toNdc(centroid, sc) || !toNdc(replacement[0], sa0) || !toNdc(replacement[1], sa1) ||
        !toNdc(replacement[2], sa2))
    {
        // The replacement (or the centroid) straddles the near plane — can't test coverage, so
        // refine conservatively rather than silently skipping.
        return allInactive;
    }
    if (inside(sc, sa0, sa1, sa2))
    {
        return {}; // the replacement still covers this face's centroid
    }
    // Coverage hole: refine the collapsed corner whose active ancestor is displaced furthest on
    // screen from its finest position — that is the corner whose recession opened the gap.
    std::uint32_t worst = kInvalidCorner;
    float worstDisp = -1.0f;
    for (std::uint32_t k = 0; k < 3; ++k)
    {
        if (!cornerInactive[k])
        {
            continue; // this corner is already at finest — cannot displace
        }
        Vec2 sFine;
        Vec2 sAnc;
        if (!toNdc(original[k], sFine) || !toNdc(replacement[k], sAnc))
        {
            continue;
        }
        const Vec2 d = sFine - sAnc;
        const float disp = Vec2::dotProduct(d, d);
        if (disp > worstDisp)
        {
            worstDisp = disp;
            worst = k;
        }
    }
    // worst == kInvalidCorner means every corner is already at finest (the face covers itself), or
    // every inactive corner was unprojectable — clean None no-op.
    if (worst == kInvalidCorner)
    {
        return {};
    }
    return {CoverageRepairKind::WorstInactiveCorner, worst};
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

void validateForest(const VertexForest& forest)
{
    const std::size_t n = forest.splits.size();
    const std::size_t vc = forest.vertexCount;
    constexpr std::size_t k32 = std::numeric_limits<std::uint32_t>::max();
    if (n > k32 || vc > k32)
    {
        throw std::runtime_error("VDPM forest exceeds 32-bit indexing");
    }
    if (forest.removingSplit.size() != vc)
    {
        throw std::runtime_error("VDPM forest: removingSplit size != vertexCount");
    }
    auto inRange = [vc](std::uint32_t v) { return static_cast<std::size_t>(v) < vc; };
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(n); ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        // parent/child/vl must be real vertices; vr may be the boundary sentinel.
        if (!inRange(sp.parent) || !inRange(sp.child) || !inRange(sp.vl) ||
            (sp.vr != kInvalidVertex && !inRange(sp.vr)))
        {
            throw std::runtime_error("VDPM forest: split references an out-of-range vertex");
        }
        // Each split must be the removing split of its own child — which also proves children are
        // unique (two splits can't both be removingSplit[child]).
        if (forest.removingSplit[sp.child] != s)
        {
            throw std::runtime_error("VDPM forest: split/child removing-split inconsistency");
        }
    }
    for (std::uint32_t v = 0; v < static_cast<std::uint32_t>(vc); ++v)
    {
        const std::uint32_t rs = forest.removingSplit[v];
        if (rs == kNoSplit)
        {
            continue;
        }
        if (static_cast<std::size_t>(rs) >= n)
        {
            throw std::runtime_error("VDPM forest: removingSplit references an out-of-range split");
        }
        // Reverse of the split-loop check: the split that claims to remove `v` must have `v` as its
        // child. Without this a spare vertex can alias another's removing split — the DAG would
        // treat it as activated by that split, but refineOne only activates the split's recorded
        // child.
        if (forest.splits[rs].child != v)
        {
            throw std::runtime_error("VDPM forest: removingSplit/child reverse inconsistency");
        }
    }
}

ActiveFront ActiveFront::build(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                               std::span<const MeshCollapse> collapses)
{
    ActiveFront front;
    front.forest_ = buildVertexForest(vertices, collapses);
    front.weld_ = mesh_topology::weldByPosition(vertices);
    front.finestFaces_ = mesh_topology::canonicalFaces(front.weld_, indices);

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

VdpmViewParams makeVdpmViewParams(const Mat4& world, const Vec3& cameraPos, float projScaleY,
                                  float viewportHeight, float silhouetteBoost,
                                  bool rasterBackfaceCulling, float uvScale, float normalScale,
                                  float tangentScale)
{
    const Mat3 linear = Mat3::fromColumns({world[0, 0], world[1, 0], world[2, 0]},
                                          {world[0, 1], world[1, 1], world[2, 1]},
                                          {world[0, 2], world[1, 2], world[2, 2]});
    // Object-space deviation/support radii are projected against a WORLD distance, so they must be
    // bounded into world space by σ_max (exact for uniform scale, conservative otherwise) or a
    // scaled instance would under-refine. (UV error is texture-space, so NOT scaled — see below.)
    const float worldLengthScale = largestSingularValue(linear);

    // The cone predicate runs in OBJECT space (sign of dot(normal, viewDir) is
    // transform-invariant), so the camera is inverse-transformed once. A near-singular world has no
    // reliable inverse — the cone is then unusable (never cull, treat every split as a potential
    // silhouette). The conditioning test |det|/σ_max³ ∈ [0,1] is a pure SHAPE measure
    // (scale-invariant), unlike an absolute |det| which would wrongly reject a tiny-but-uniform
    // instance. A reflection's determinant sign folds into the cone facing.
    const float det = linear.determinant();
    const float sigmaMax = worldLengthScale;
    const bool coneUsable = std::abs(det) > 1e-6f * sigmaMax * sigmaMax * sigmaMax;
    const Vec3 worldTranslation{world[0, 3], world[1, 3], world[2, 3]};

    VdpmViewParams p;
    p.worldLinear = linear;
    p.worldTranslationMinusCamera = worldTranslation - cameraPos;
    p.cameraObj = coneUsable ? linear.inverse() * (cameraPos - worldTranslation) : Vec3{};
    p.worldLengthScale = worldLengthScale;
    p.facingSign = det >= 0.0f ? 1.0f : -1.0f;
    p.projScaleY = projScaleY;
    p.halfViewport = viewportHeight * 0.5f;
    p.silhouetteBoost = silhouetteBoost;
    p.uvScale = uvScale;
    p.normalScale = normalScale;
    p.tangentScale = tangentScale;
    p.coneUsable = coneUsable;
    p.coneCullEnabled = rasterBackfaceCulling && coneUsable;
    return p;
}

VdpmSplitScore scoreVdpmSplit(const VdpmViewParams& params, const VertexSplit& s,
                              const Vec3& parentPos, const Vec3& childPos)
{
    // Camera-relative affine: worldLinear·local + (worldTranslation − camera) = world·local −
    // camera, so this is EXACTLY the world-space offset from the camera under any linear transform
    // (incl. singular). The support sphere is centred on the PARENT (= kept), so the footprint
    // projects from the parent's distance, not the child's.
    auto worldDelta = [&](const Vec3& local)
    { return (params.worldLinear * local) + params.worldTranslationMinusCamera; };
    const float distance = std::max(1e-3f, worldDelta(childPos).magnitude());
    const float parentDistance = std::max(1e-3f, worldDelta(parentPos).magnitude());

    // Object-space cone visibility (region centre = the parent). Singular world ⇒ never cull, max
    // potential silhouette.
    const detail::ConeVisibility vis =
        params.coneUsable
            ? detail::coneVisibility(s.normalConeAxis, s.normalConeCos, s.supportRadius, parentPos,
                                     params.cameraObj, params.facingSign, params.coneCullEnabled)
            : detail::ConeVisibility{false, 1.0f};

    VdpmSplitScore r;
    r.straddle = vis.straddle;
    if (vis.backFacing)
    {
        r.backface = 1; // provably raster back-face-culled: score 0, skip discretionary refine
        return r;
    }

    // Silhouette boost tightens the budget for a cone straddling edge-on so contours stay dense.
    const float boost = 1.0f + (params.silhouetteBoost * vis.straddle);
    // Four channels in screen pixels. GEOMETRIC (geometry silhouette-boosted + world-scaled; UV
    // texture-space, NOT world-scaled) project e·projScaleY·(vh/2)/d. ANGULAR (normal/tangent,
    // radians) project the support radius as a screen EXTENT and multiply by the chord 2·sin(θ/2) —
    // dimensionless, so shading scales like geometry. Both angular channels are silhouette-boosted.
    const float worldSupport = s.supportRadius * params.worldLengthScale;
    r.geometry = s.error * params.worldLengthScale * boost * params.projScaleY *
                 params.halfViewport / distance;
    r.uv = s.uvError * params.uvScale * params.projScaleY * params.halfViewport / distance;
    const float nearDistance = std::max(1e-3f, parentDistance - worldSupport);
    const float extent = worldSupport * params.projScaleY * params.halfViewport / nearDistance;
    r.normal = 2.0f * std::sin(0.5f * s.normalError) * params.normalScale * boost * extent;
    r.tangent = 2.0f * std::sin(0.5f * s.tangentError) * params.tangentScale * boost * extent;
    return r;
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

    // Derive the per-instance view params ONCE (σ_max, the determinant conditioning, the
    // object-space camera), then score every split through the shared Vulkan-free authority — the
    // exact derivation + math the GPU port (render/vdpm_gpu) uploads and reproduces.
    const VdpmViewParams params =
        makeVdpmViewParams(world, cameraPos, projScaleY, viewportHeight, silhouetteBoost,
                           rasterBackfaceCulling, uvScale, normalScale, tangentScale);
    const float invBudget = pixelBudget > 0.0f ? 1.0f / pixelBudget : 0.0f;

    // Score pass: tag every split with its max screen-space channel score + back-face flag for this
    // view (order-independent — no front mutation here).
    for (std::uint32_t i = 0; i < forest_.splits.size(); ++i)
    {
        const VertexSplit& s = forest_.splits[i];
        const VdpmSplitScore r =
            scoreVdpmSplit(params, s, vertices[s.parent].position(), vertices[s.child].position());
        if (r.backface != 0)
        {
            splitBackface_[i] = 1; // culled: score 0, skip refine, eligible to coarsen
            continue;
        }

        // Metric instrumentation: track how hard each channel is pushing (max ratio over all
        // front-facing splits) regardless of whether this split refines, so an under-firing channel
        // is still visible.
        channelStats_.maxGeometryRatio =
            std::max(channelStats_.maxGeometryRatio, r.geometry * invBudget);
        channelStats_.maxUvRatio = std::max(channelStats_.maxUvRatio, r.uv * invBudget);
        channelStats_.maxNormalRatio = std::max(channelStats_.maxNormalRatio, r.normal * invBudget);
        channelStats_.maxTangentRatio =
            std::max(channelStats_.maxTangentRatio, r.tangent * invBudget);

        splitScore_[i] = r.score();

        // A split that is NOT yet refined and is over the budget will be pulled in by the refine
        // pass below — attribute that new *trigger* to its winning channel. Already-refined splits
        // that merely stay refined are not re-counted (steady-state ⇒ few triggers).
        if (refined_[i] == 0 && splitScore_[i] > pixelBudget)
        {
            const float winner = splitScore_[i];
            if (winner == r.geometry)
            {
                ++channelStats_.geometryTriggers;
            }
            else if (winner == r.uv)
            {
                ++channelStats_.uvTriggers;
            }
            else if (winner == r.normal)
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
        const bool degenerate = (a0 == a1 || a1 == a2 || a0 == a2);
        // Shared classifier owns the winding math (a degenerate replacement is never a foldover —
        // a neighbour covers it), so the sequential sweep and the parallel snapshot detector agree.
        const std::array<Vec3, 3> original{worldPos(fc[0]), worldPos(fc[1]), worldPos(fc[2])};
        const std::array<Vec3, 3> replacement{worldPos(a0), worldPos(a1), worldPos(a2)};
        if (!detail::isFoldover(original, replacement, degenerate))
        {
            continue;
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
    for (const std::array<std::uint32_t, 3>& fc : finestFaces_)
    {
        const std::uint32_t a0 = activeAncestor(fc[0]);
        const std::uint32_t a1 = activeAncestor(fc[1]);
        const std::uint32_t a2 = activeAncestor(fc[2]);
        const bool degenerate = (a0 == a1 || a1 == a2 || a0 == a2);
        const std::array<Vec3, 3> original{worldPos(fc[0]), worldPos(fc[1]), worldPos(fc[2])};
        const std::array<Vec3, 3> replacement{worldPos(a0), worldPos(a1), worldPos(a2)};
        const std::array<bool, 3> inactive{active_[fc[0]] == 0, active_[fc[1]] == 0,
                                           active_[fc[2]] == 0};
        // The shared classifier decides the per-face outcome (all pure projection math); this sweep
        // only applies the targets, so it stays byte-identical to the parallel snapshot detector's
        // per-face policy. Every force-refine still routes through refineCoverageCorner so the
        // counter and result accounting can't diverge between the target kinds.
        const detail::CoverageRepair repair = detail::classifyCoverageRepair(
            original, replacement, degenerate, inactive, cameraPos, viewProj, viewportWidth,
            viewportHeight, rasterBackfaceCulling);
        switch (repair.kind)
        {
        case detail::CoverageRepairKind::None:
            break;
        case detail::CoverageRepairKind::AllInactiveCorners:
            for (const std::uint32_t c : fc)
            {
                refineCoverageCorner(c);
            }
            break;
        case detail::CoverageRepairKind::WorstInactiveCorner:
            refineCoverageCorner(fc[repair.worstCorner]);
            break;
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
