# Mesh Level of Detail

How the mesh LOD system works, and the design decisions behind it, for an engineer who will
**maintain and improve it**. This is the physics track's [`collision.md`](collision.md) equivalent for the rendering
spine's LOD arc (roadmap "#3 — view-dependent progressive meshes").

The full ladder is **done**: **discrete** LOD (Phase 1), **view-independent continuous** geomorphing
(VIPM, Phase 2), and **view-dependent** progressive meshes (VDPM, Phase 3). All three are selected in
the overlay's "Mesh LOD" panel and all three consume the *same* recorded collapse stream from one
simplifier. Discrete swaps index buffers with a pop; VIPM dissolves the pop by geomorphing into the
exact next level; VDPM refines *different regions of one mesh to different detail* per frame from a
per-region active front, so at a matched error budget it matches the discrete mesh's silhouette and
shading with a fraction of the triangles. VDPM is documented in its own section
([View-dependent progressive meshes](#view-dependent-progressive-meshes-vdpm)).

---

## What it does

At load time, each static mesh above a triangle threshold gets a small progressive artifact: an
ordered collapse stream, exact LOD cuts through that stream, progressively simpler index buffers, a
per-original-vertex geomorph buffer (VIPM), and a per-instance vertex forest + active front (VDPM).
At draw time the discrete/continuous paths pick the coarsest whole-mesh LOD whose on-screen error
stays within a pixel budget; the view-dependent path instead refines a per-instance active front for
the camera and uploads a per-frame index buffer, so different regions of one mesh sit at different
detail. On the DamagedHelmet (~15.5k triangles) the discrete/VIPM levels are ~15452 → 7725 → 2776
(the coarsest is held above the naive ~1900 target by the chart veto, which preserves the UV-seam
skeleton — see [simplifier decision 8](#the-simplifier-graphicsmesh_simplifier)); VDPM lands between
LOD0 and the budget, tracking the visible silhouette.

Two properties are load-bearing and shape everything below:

- **All LODs index the same, unchanged vertex buffer.** A LOD is *only* an alternate index buffer.
  No vertex data is moved or duplicated, so a mesh with N levels costs one vertex upload + N index
  uploads, and switching LOD is just swapping `indexBuffer`/`indexCount` on the draw.
- **The collapse cut, not the error value, is topology identity.** Each LOD records the exact number
  of collapses replayed to produce it. `error` is only the screen-space selection metric; it must
  never be used to infer which collapses belong to a level, because individual collapse errors are
  not guaranteed to be monotonic.
- **Render attributes survive simplification.** UV, normal, tangent, and TEXCOORD_1 seams are
  preserved through the near levels and only shift at the coarsest — not smeared — because the
  simplifier is attribute-aware and the emit is wedge-aware.

---

## Runtime data flow

```
Geometry::load()                         [once, load time, CPU]
  QuadricSimplifier::buildProgressive(kLodRatios)
    → collapse stream + exact LOD cuts + index set per LOD (into the base vertex buffer)
  lods_ = { LOD0 = full, LOD1, LOD2, … } each { indexBuffer, indexCount, error }
  buildVipmMorphData(vertices, collapses, progressive.lods)
    → per-original-vertex geomorph SSBO

Object::buildDrawCommands()              [per draw, per frame]
  selectLod(geometry.lods(), distance, proj[1,1], viewportHeight, pixelErrorBudget)
  → cmd.indexBuffer / cmd.indexCount / cmd.lodLevel
  (shadow draws use the same, biased coarser)

Object::writeForwardUniforms()            [per draw, per frame]
  selectVipm(...) in Continuous mode
  → MorphUBO { morphFactor, vipmTargetLevel }

  ViewDependent mode (per instance):
    ActiveFront::refineForView(...)         persistent front, refine by screen-space error
                                            (geometry / UV-seam / normal / tangent), cone
                                            silhouette boost + back-face gate
    ActiveFront::repairFront(...)           JOINT fixpoint: un-flip foldovers + un-recede coverage
                                            holes together, until a full cycle changes nothing
    ActiveFront::emitActiveIndices(...)     → a per-frame dynamic index buffer (wedge-restored)
```

- **Build** — `Geometry::load()` (`src/graphics/geometry.cpp`) builds LODs when
  `!storageVertices_ && triangles ≥ kMinLodTriangles`. Deformable meshes (cloth `storageVertices`,
  and by policy skinned/morph if we ever exclude them) are skipped or kept at LOD0. Simplification is
  entirely at load time — **zero per-frame cost**, and it does not touch the physics sim, so
  determinism is unaffected.
- **Select** — `selectLod` (`include/fire_engine/graphics/lod.hpp`, header-only, headless-testable)
  runs in `Object`'s draw-build (`src/graphics/object.cpp`) using `FrameInfo`'s camera position,
  `proj[1][1]`, and viewport height. It sets the chosen level's `indexBuffer`/`indexCount` on the
  `DrawCommand`, plus `lodLevel` (used only by the LOD-tint debug view).
- **Morph** — `selectVipm` (`include/fire_engine/graphics/vipm.hpp`) uses the same screen-space
  selection curve as `selectLod`, but returns the next exact LOD level and a 0→1 factor. The vertex
  shader morphs only vertices whose `collapseLevel` equals that target level.
- **Refine (VDPM)** — for a mesh with a collapse stream, `Geometry::load` also stores the raw
  collapses; `Object` builds a per-instance `ActiveFront` (over a `VertexForest`) at load, and in
  `writeForwardUniforms` runs `refineForView` + `repairFront` per frame, emitting the active index
  set into a per-`currentFrame` dynamic buffer. `buildDrawCommands` points the draw at it, taking
  precedence over discrete/VIPM. The per-frame path is **allocation-free steady-state**: `emit` fills a
  reused per-binding scratch vector (not a fresh `std::vector` each frame), and `emit` precomputes
  `activeAncestor` once (the front is settled by then) — pure per-frame functions, so behaviour is
  identical to the inline compute.
  `ActiveFront` also exposes **per-frame repair counters** (vertices each pass pulled back in), summed
  over instances and shown in the overlay LOD panel as a regression watch. Vulkan-free +
  headless-testable; see the VDPM section.

---

## The simplifier (`graphics/mesh_simplifier`)

A from-scratch **Garland–Heckbert quadric-error-metric (QEM) edge-collapse** simplifier. Vulkan-free,
headless-testable, no third-party dependency. Behind a `MeshSimplifier` interface (a backend could be
swapped — e.g. meshoptimizer — but see "Why from scratch" below).

Public surface:
- `simplify(vertices, indices, targetRatio) → SimplifiedMesh { indices, error }` — indices into the
  original vertices; `error` is the world-space geometric deviation (RMS) of the coarsest applied
  collapse.
- `collapseSequence(vertices, indices) → std::vector<MeshCollapse>` — the full ordered coarsening
  stream.
- `buildProgressive(vertices, indices, ratios) → ProgressiveMesh` — the runtime path. It runs one
  greedy replay, emits LOD0 plus each requested cut, records each cut's `collapseCount`, and returns
  the same collapse stream that VIPM consumes.

### Algorithm, and the decisions inside it

1. **Position welding for connectivity.** glTF splits a vertex at every attribute seam (a corner with
   two UVs / normals becomes two vertices). Left split, the index topology is shattered into
   boundary-locked islands that barely simplify (the DamagedHelmet went 15452 → 15361 — 0.6%). So
   collapse connectivity is built on **position-welded** vertices: all vertices at one exact position
   collapse to a canonical. *Decision:* weld by position only (not position+UV) — welding by
   position+UV re-shatters at seams and caps simplification (~44% on the helmet). The weld, wedge
   distance, nearest-wedge and canonical-wedge grouping primitives live in one shared module,
   `graphics/mesh_topology.{hpp,cpp}` (`weldByPosition` / `wedgeDistance` / `nearestWedge` /
   `canonicalWedges`), consumed identically by the simplifier, VIPM and VDPM so the canonical-id
   contract can't drift between them.

2. **Wedge-preserving emit.** Welding by position throws away seam identity (both sides collapse to
   one canonical wedge). To get it back, the emit keeps `origTris_` (original per-corner vertices)
   and `canonicalWedges_` (the native wedges at each position). Each output corner emits the wedge at
   its surviving position whose UV0 / UV1 / normal / tangent are **nearest** the corner's own render
   attributes (`nearestWedge`). On a preserved seam (a position carrying wedges from two charts or
   normals) each side keeps its closest chart/shading identity. *Decision:* this wedge machinery is
   needed regardless of whether UV is in the error metric — it's what makes the output attributes
   correct.

3. **Subset (endpoint) collapse.** An edge collapse merges to one of the two endpoints (whichever
   costs less), never to an optimal off-vertex position. *Decision:* this keeps the surviving vertex
   set a strict subset of the originals, so all LODs index the unchanged vertex buffer (property #1).
   Optimal placement would move vertices and force a per-LOD vertex buffer.

4. **Attribute-aware R⁵ quadric.** The error quadric lives in **R⁵ = (x, y, z, w·u, w·v)**: each
   triangle contributes the squared distance to its 2-plane in R⁵, so the metric penalises geometric
   deviation *and* UV stretch in one number. Collapses that would distort the texture cost more and
   are ordered **last** — so seams and high-UV-gradient interior detail survive into the near LODs and
   only give way at the coarsest. It reduces *exactly* to the geometric plane quadric when all UVs are
   equal (so UV-free meshes behave as pure QEM). Stored as the full quadratic form `VᵀAV + 2b·V + c`.
   Boundary quadrics are **position-only** (UV components zero) — a border is a geometric constraint.

5. **RMS error for screen-space, not the raw quadric.** The accumulated quadric cost is a *sum* of
   squared distances over every plane folded in, so it inflates as collapses chain (it came out ~60×
   too large — LODs never got selected). The reported per-LOD error normalises it:
   `sqrt(quadricCost / accumulatedPlaneWeight)` — a root-mean-square deviation in world units, ~0 for
   a flat surface, a small fraction of the mesh for a curved one. The boundary weight is added to
   **both** the quadric and the weight denominator, or boundary collapses inflate the RMS.

6. **Error ceiling refuses only geometrically-un-simplifiable shapes.** The greedy stops when the
   cheapest remaining collapse exceeds `kErrorCeilingFactor × boundsDiagonal²`. Its job is to refuse a
   shape that *can't* simplify (a cube — every collapse folds a face, driven far up by the boundary
   weight), **not** to stop at UV-costly seams. *Decision:* the factor is deliberately generous
   (`40`), because on-screen quality is guaranteed by the runtime screen-space selection, not by
   keeping build-time error low. Too tight and UV-aware ordering caps simplification at the seams
   (~45%); at `40` the boundary weight still refuses the cube while seams collapse last.

7. **Normal-flip veto.** A collapse that would invert an incident triangle's normal is rejected — a
   validity guard, not part of the cost.

8. **Chart veto — don't collapse across a render seam.** Position welding (decision 1) fuses UV /
   normal / tangent charts that authoring kept separate, which lets the collapse stream merge a vertex
   into a *different* chart. Discrete and VDPM tolerate that (VDPM's UV channel just refines the seam
   back), but **VIPM cannot**: its geomorph slides a vertex's whole render identity toward its
   collapse survivor, so a cross-chart collapse has to *shear or pop the texture* during the
   transition — there is no runtime metric that refines a morph out of a bad target. So a cross-chart
   collapse is vetoed topologically. Each original wedge gets a **chart id** (union-find over wedges,
   joined across every edge whose two incident faces *agree* on attributes — attribute-aware, so a
   benign position duplicate with identical attributes still merges and simplifies), and
   `canonicalCharts_[v]` is the set of charts meeting at position `v`. A collapse is rejected if the
   survivor lacks a chart the removed vertex carries; the error-chosen direction is tried first, then
   the reverse (so an interior vertex is forced *into* its seam rather than the seam into the
   interior), and only an edge between two *different* seams is skipped outright. *Decision:* a veto,
   not a seam-boundary quadric — a single quadric weight can't separate a legal *along*-seam collapse
   from an illegal *cross*-seam one (measured: any weight that stops crossing also locks curved seams
   so the mesh can't reach its target). This also makes `nearestWedge` correct *by construction*: the
   survivor is guaranteed to carry a same-chart wedge, so the attribute-distance pick lands in-chart.
   Cost: the coarsest LODs keep the seam skeleton (DamagedHelmet's L2 lands ~2800 tris vs a ~1900
   target), which is the price of a shear-free morph.

### The two tuning dials

Both live in `src/graphics/mesh_simplifier.cpp` (`QemRun`):

| Constant | Meaning | Effect of raising |
|---|---|---|
| `kUvWeightFactor` (0.1) | UV weight as a fraction of the bbox diagonal in the R⁵ metric | More UV fidelity (seams held longer), less triangle reduction |
| `kErrorCeilingFactor` (40) | Collapse-cost ceiling as a multiple of bbox-diag² | Allows coarser LODs; too high risks over-collapsing genuinely un-simplifiable shapes (the boundary weight, 1000, is the real cube guard) |

LOD ratios and the selection budget live in `include/fire_engine/graphics/lod.hpp`:
`kLodRatios {0.5, 0.125}`, `kMinLodTriangles 512`, `kLodPixelErrorBudget 2`, `kShadowLodBias 3`.

### Why from scratch (not meshoptimizer)

`meshopt_simplify` returns only a final index buffer — it does **not** expose the ordered collapse
sequence, and successive calls at different ratios aren't nested. The continuous phases (VIPM/VDPM)
*are* that collapse sequence, so we need a simplifier we can read it out of. Building our own also
keeps it dependency-free and in-spirit with the from-scratch physics solver. meshoptimizer remains a
viable backend for discrete-only work, or a validation oracle in tests, behind the `MeshSimplifier`
interface.

---

## Selection and VIPM (`selectLod`, `selectVipm`)

`selectLod(lods, distance, projScaleY, viewportHeight, pixelErrorBudget)` picks the **coarsest** level whose
world error projects to ≤ `pixelErrorBudget` screen pixels. A world deviation `e` at view distance `d`
projects to `e · projScaleY · viewportHeight / (2d)` pixels, where `projScaleY = proj[1][1]`
(= 1/tan(fovY/2)). Levels are ordered fine→coarse with non-decreasing error, so the first that
overflows the budget ends the search; LOD0 is the fallback when nothing coarser fits.

Shadow draws pass `kLodPixelErrorBudget × kShadowLodBias` (coarser — silhouette detail matters less in a
shadow). Selection uses the camera's projection/viewport as an approximation for shadow passes too;
good enough because a mesh far from the camera has little shadow footprint.

Continuous mode deliberately keeps the same topology level that `selectLod` would choose. When level
`L` is selected, `selectVipm` computes how far the current tolerated error has advanced toward
`L + 1`, writes that as `morphFactor`, and writes `vipmTargetLevel = L + 1`. A `MorphVertex` stores
the 1-based level at which that original vertex first disappears. The shader compares levels, not
errors, so non-monotonic collapse costs cannot make a vertex morph during the wrong transition.

The morph target is a full render wedge: position, normal, tangent, TEXCOORD_0, and TEXCOORD_1. At
`morphFactor == 1`, a fine-level vertex removed by the next LOD has the same attributes as the wedge
the next LOD index buffer will draw. This is the load-bearing no-pop invariant.

---

## View-dependent progressive meshes (VDPM)

VIPM picks one detail level for the whole mesh. VDPM (`graphics/vdpm`) refines **different regions of
one mesh to different detail** each frame, from a per-region **active front** — so the near/edge-on
parts stay dense while the far/flat parts coarsen, and at a matched budget the result matches the
discrete mesh's silhouette and shading with far fewer triangles. Vulkan-free + headless-testable; a
GPU-driven front is the eventual follow-on. Everything indexes the **canonical (position-welded)**
vertex set, exactly like the simplifier, and render wedges are restored only at emit.

### The vertex forest (`buildVertexForest`)

Replaying the recorded collapse stream backwards is a sequence of **vertex splits** (the inverse of a
collapse). Each `MeshCollapse` carries a `VertexSplit`'s worth of data — `{parent, child, vl, vr,
error, uvError, normalError, tangentError}` — Hoppe's fixed-size vsplit encoding: splitting `parent`
reintroduces `child` between the two faces of the collapsed edge, whose far apexes are `vl`/`vr`
(`vr == kInvalidVertex` on a boundary edge). A split is *legal* iff `parent` and `vl` (and `vr` if
present) are active, so no variable-length dependency list is needed — that dependency neighbourhood
is what keeps adjacent regions at different detail crack-free (no T-junctions). The four `*Error`
fields are the per-collapse deviations the runtime projects to screen (below).

**The apexes are ground truth from the simplifier, not re-derived.** `vl`/`vr` are recorded by the
simplifier's `collapse()` on the exact live canonical topology it coarsens, then stored on the
`MeshCollapse`. `buildVertexForest` is a straight transcription of that stream — for each collapse it
emits one `VertexSplit` and marks `removingSplit[removed]`. This removes the old failure mode: the
forest used to *re-derive* the apexes by replaying the stream over an independent adjacency view,
which **desynced** at a **non-manifold welded edge** (position-welding — decision 1 — fuses coincident
chart pieces into >2-face edges the vsplit can't encode) and then cascaded (DamagedHelmet: 7 genuine
non-manifold edges snowballed to 19 skipped collapses, leaving the forest unfaithful to the stream).
Now a non-manifold collapse records `kNoCollapseApex` for `vl`; `buildVertexForest` skips only that
one collapse, leaving `removed` a **root** (always active) at that isolated spot — a conservative
fallback (an always-present extra vertex, never a crack) that **cannot cascade**. So the DamagedHelmet
goes from 19 desynced skips to 7 isolated roots, faithful everywhere else.

### The active front (`ActiveFront`)

Holds per-canonical-vertex `active_` and per-split `refined_` state, plus `dependents_[v]` = the count
of refined splits requiring `v` as parent/vl/vr (so `dependents_[child] == 0` is exactly "leaf",
the coarsen precondition). `refine`/`coarsen` enforce Hoppe legality; `refineAll`/`coarsenAll` drive
the extremes; `activeAncestor(v)` walks `removingSplit[·].parent` until it hits an active vertex.
`emitActiveIndices` maps each finest render triangle's corners to their active ancestors, drops the
ones that collapse to a degenerate, and restores each surviving corner to its nearest render wedge —
the same seam-preserving pattern the simplifier's emit uses, so UV/normal seams keep their identity.

### `refineForView` — the four-channel screen-space metric

Per frame: `coarsenAll()`, then walk splits coarsest-first (so a legal refine's dependencies are
already active) and refine any whose projected error exceeds the pixel budget. A world deviation `e`
at distance `d` projects to `e · projScaleY · viewportHeight / (2d)` px, exactly like `selectLod`.
**Four independent channels**, any one over budget triggers a refine — each with its own `kVdpm*Scale`
dial in `graphics/lod.hpp`:

| Channel | `VertexSplit` field | Source (`MeshCollapse`) | What it catches |
|---|---|---|---|
| Geometry δ | `error` | `deviationRadius` (point-to-**plane**, additive) | off-surface curvature; ~0 on a flat region |
| UV seam/stretch | `uvError` | `uvDeviationRadius` (**max**-accumulated, per-wedge) | texture stretch, and atlas seams the geometry can't see |
| Shading normal | `normalError` | `normalDeviationRadius` (angular, rad) | lighting flattening a smooth-shaded curve carries even when near-coplanar |
| Tangent frame | `tangentError` | `tangentDeviationRadius` (angular, rad) | drift of the shader's TBN frame — MAX of the T- and B-axis deviation, so it sees a tangent roll AND a handedness (w) flip — independent of the shading normal (0 without tangents) |

Point-to-**plane** (not point-to-triangle) keeps the geometry channel ~0 across the small in-plane gap
a collapse leaves, so flats stay flat. The UV channel accumulates by **max**, not the geometric
channel's running sum: a UV error is a screen-space *discontinuity* (the eye sees the worst jump in a
region, not a compounding envelope), and its per-collapse value is `max(smooth stretch on a containing
face, spread between the removed position's atlas wedges)` — the wedge-spread term is what makes an
atlas seam-crossing collapse read its true cost. Two view modifiers on top:

Both come from the split's precomputed **conservative normal cone** (`detail::coneVisibility`, an
**exact evaluation of a conservative bound** — see § Visibility cones):

- **Silhouette boost** — a cone straddling edge-on gets a tighter budget (`1 + silhouetteBoost·straddle`,
  `straddle ∈ [0,1]`), so contours stay dense.
- **Back-face gate** — a split whose *whole cone* provably faces away (over the support-sphere view
  spread) skips *discretionary* refinement (raster back-face-culled — wasted detail); `forceRefine` can
  still pull it in as a visible split's dependency. Only applied when the draw actually culls back-faces
  (`rasterBackfaceCulling` — false for double-sided/blended materials, whose back-faces are visible).

### The joint repair (`repairFront`) — why a correct emit still needs it

A selective front is a **non-prefix** cut of the collapse stream. The simplifier's `wouldFlip` only
certifies the *linear prefix* is flip-free, and the deviation channels are all *topological* — never
projected to screen. So two failure classes survive an otherwise-correct front, each classified by a **pure per-face
predicate** (in `detail::`, so both the sequential CPU sweeps and the parallel snapshot detector run
the identical projection math) and repaired by a **refinement-only sweep** (only activates splits —
inflationary):

1. **Foldovers** (`detail::isFoldover`, applied by `repairFoldoversSweep`). A finest face whose
   active-ancestor replacement winds *against* the original is back-face-culled by the rasteriser → a
   hole. Where replacement winding opposes original winding, force-refine the collapsed corners
   (world-space winding, so correct under non-uniform / reflected transforms). A degenerate replacement
   is not a foldover — a neighbour covers it.
2. **Coverage / silhouette holes** (`detail::classifyCoverageRepair`, applied by `repairCoverageSweep`,
   on the **jitter-free** `currentViewProj` — the TAA-jittered `frame.proj` would shift the test ±½px
   each frame and thrash the front). A *closed, non-folded* front can still leak the background: at a
   silhouette a coarse replacement recedes inside a fine visible face's *projected footprint*. Purely
   screen-space (boundary/foldover checks are blind to it). For each **visible** finest face above a
   small projected-area gate the classifier returns one of: **None** (covered / not visible / sub-pixel
   / already full-detail), **AllInactiveCorners** (degenerate replacement — a dropped sliver at a
   contour; a near-plane straddle that can't project; an unprojectable replacement — refine every
   inactive corner conservatively), or **WorstInactiveCorner** (an escaped centroid — refine the single
   inactive corner most screen-displaced from its ancestor). "Visible" follows the draw's cull mode
   (`rasterBackfaceCulling`): culling on ⇒ front faces only; off (double-sided/blend) ⇒ back faces too.

**They must run as a JOINT fixed point, not two sequential phases** — `repairFront` alternates the two
sweeps until a complete cycle refines nothing. A coverage force-refine can re-fold a neighbour and a
foldover force-refine can open a coverage hole, so running the foldover fixpoint and *then* coverage
leaves foldovers (a real bug that shipped before this). Inflationary ⇒ terminates in
`jointRepairSweeps() ≤ initially-unrefined-splits + 1` (≈2 in practice). The result is the deterministic
fixed point of *this* schedule (not a least/unique one). `repairFront` is the ONLY public repair
(`Object` calls it, mandatory before emission); the two sweeps are private so no caller can misorder
them. If a repairable violation has an inactive target whose valid removing split fails to force-refine,
`repairFront` throws (a forest inconsistency) rather than spin.

**These repairs cannot be replaced by the normal cone** (§ Visibility cones): coverage is a
*screen-space* property and foldover is *topological*, while the cone is a face-*orientation* bound.
A force-refine-on-straddle experiment reduced but could not zero coverage repairs, so both sweeps
stay. Retiring them eventually needs a GPU worklist/fixpoint or a representation-level guarantee.

Because the classifiers are pure `detail::` predicates over world positions, the GPU-shaped
`ParallelFront` (§ GPU-driven active front) reuses them verbatim — its snapshot repair shares P2's
**per-face policy and final invariants**, differing only in the *operator*: it detects every violation
against a **settled** front (no mutation while detecting), marks each target's removing split, closes
+ applies them in rank order, then re-detects. So it may reach a **different valid front** (over-
refining a region can dissolve a violation the sequential schedule would have repaired via a different
corner) — never the sequential joint operator's exact front. The `detail::kMinCoverageScreenAreaPx`
policy is the one shared knob, so a test validator's "worth-fixing" threshold tracks the runtime's.

### Visibility cones (`detail::coneVisibility`)

Each split carries a **conservative normal cone** (`{normalConeAxis, normalConeCos}`, built in the
simplifier — § below) bounding every finest face normal in its subtree. `refineForView` turns it into
the two view modifiers above via `detail::coneVisibility`, which is an **exact evaluation of a
conservative bound**, not an exact visibility oracle: the cone and the support sphere both *over*-bound
the real surface, so a `backFacing` result is a one-sided **proof of hiddenness**, while `!backFacing`
and any straddle only mean "not provably hidden / a silhouette *may* exist".

- **Object space, exact under any linear transform.** The sign of `dot(worldNormal, worldViewDir)`
  equals the sign of `dot(objNormal, objViewDir)` for any linear world matrix (the normal's `M⁻ᵀ` and
  the view direction's `M` cancel), so the whole test runs in object space with the camera inverse-
  transformed once — exact under non-uniform scale/shear, and the circular cone stays circular (a
  world-space test would shear it).
- **View-direction spread.** The camera subtends a half-angle `asin(r/d)` over the support sphere
  (radius `r`, object-space distance `d`), so a merely back-*centred* region near the camera is not
  provably hidden. Back-facing needs `dot(axis, dirToCam) < −sin(θn+θv)`; the straddle weight measures
  how centrally edge-on sits in `θn+θv`. Formed **trig-free** via the cosine sum identity (GPU-friendly).
- **Reflections & degeneracies.** A negative-determinant (mirrored) world flips winding — folded in
  exactly by multiplying the facing by `sign(det)`. A near-singular world (no reliable inverse) →
  never cull, max silhouette. `coneCos ≤ 0` (cone wider than a hemisphere) and a camera inside the
  support sphere are both no-cull / max-silhouette sentinels.
- **Material-gated.** Suppression is applied only when the draw actually culls back-faces
  (`rasterBackfaceCulling`); a double-sided or blended material renders its back-faces, so their
  refinement must not be suppressed.

This replaced the old smooth-vertex-normal 4-witness proxy and roughly **halved** `refineForView`
(it's a closed-form per-split test, no fixpoint — also the shape a GPU-driven front wants). It does
**not** retire the repair sweeps (above).

### Dials (`graphics/lod.hpp`)

`kVdpmSilhouetteBoost` (2.0) and the four channel scales `kVdpmUvScale` / `kVdpmNormalScale` /
`kVdpmTangentScale` (1.0 / 0.5 / 0.5 — the geometry channel is unscaled). Raising a scale refines that
channel sooner (more fidelity, more triangles). The back-face cull takes no threshold dial — the cone
bound is exact bar a tiny float-safety margin; whether it applies at all is the `rasterBackfaceCulling`
flag `refineForView` is passed (from the material's cull mode). The coverage sweep's screen-area gate
(`detail::kMinCoverageScreenAreaPx`, shared by the classifier + test validators) trades residual
sub-pixel holes against extra triangles;
lower it to catch smaller holes.

---

## Overlay & debug

`RenderTunables` (`lodEnabled`, `lodMode`, `lodPixelErrorBudget`) drives a **"Mesh LOD"** overlay panel
(toggle + a **mode selector** — Discrete / Continuous (VIPM) / View-dependent (VDPM) — + pixel error
budget slider + a live "Triangles drawn" readout from `FrameStats::trianglesDrawn`). The
**"LOD tint"** debug view (`DebugView::Lod`, view index 7) colours each mesh by its selected level —
green LOD0 / yellow LOD1 / red LOD2 / magenta 3+ — via the per-draw `lodLevel` threaded through
`ForwardPushConstants` into `shader.frag`. Toggling LOD off (all green, full mesh) is the A/B.

Confirm behaviour by backing the camera off or cranking the pixel error budget slider: the tint should step
green → yellow → red and the triangle count should drop.

---

## Tests

`tests/graphics/test_mesh_simplifier.cpp` (`[MeshSimplifier]`), headless:
- **Flat grid → 2 triangles at ~0 error** — coplanar interior + straight borders collapse fully.
- **UV sphere** — hits the target ratio, bounded error, output is a strict vertex subset.
- **Welds coincident seam vertices** — a per-triangle-duplicated grid must collapse like a welded one.
- **Preserves per-corner attributes across a seam** — a two-triangle seam keeps both sides' closest
  render wedges.
- **Deterministic** — identical output across runs.
- **Collapse-sequence replay** — replaying the recorded stream reproduces `simplify(0.0)`.
- **Progressive cuts** — `buildProgressive()` records increasing exact collapse counts for each LOD.

`tests/graphics/test_vipm.cpp` (`[vipm]`), headless:
- **Exact cuts, not errors** — non-monotonic collapse errors do not move a vertex into the wrong
  transition.
- **Collapse chains** — vertices removed in one cut resolve to the final survivor for that cut.
- **Duplicate wedges** — when a position-welded vertex collapses, every original same-position wedge
  receives morph data.
- **Selection parity** — `selectVipm` chooses the same topology level as `selectLod`.

`tests/graphics/test_vdpm.cpp` (`[vdpm]`), headless:
- **Forest + front invariants** — legal refine/coarsen, `refineAll` reproduces the finest index
  buffer, coarse-first refine keeps dependencies satisfied, seam-wedge emit.
- **Per-channel deviation** — geometry δ ~0 on a flat mesh / accumulates on a curved one; the UV,
  shading-normal, and tangent channels each fire on a mesh only *that* channel can see (flat grid with
  skewed UVs / fanned normals / fanned tangents).
- **Back-face suppression** — the hidden hemisphere of a sphere coarsens; near view resolves more than
  far.
- **Joint repair (`repairFront`)** — after refine + `repairFront`, BOTH invariants hold together: no
  emitted triangle winds against the original (foldover) AND every visible triangle stays covered by
  its replacement (coverage), following the draw's cull policy (front-face-only when culling; both
  windings for a no-cull double-sided/blend material; correct under a reflected world). A **named
  regression** pins the six sphere cases where the old sequential foldover-then-coverage order left
  foldovers (coverage re-folded after the foldover fixpoint finished).

`selectLod` is header-only and directly unit-testable. Beyond the headless tests, the render smoke
(validation on) must stay 0-VUID with LODs active across static / skinned / cloth meshes and all three
LOD modes.

---

## Known limits & future directions

The discrete → VIPM → VDPM ladder is complete; VDPM matches the discrete mesh's silhouette and shading
at a fraction of the triangles. The remaining residuals and follow-ons, in rough priority:

- ✅ **Metric fidelity (arc complete, steps 1–7).** The four-channel metric was correct in shape but
  not a reliable perceptual bound; the seven steps below closed that. Parked follow-ons (texel-density
  UV budget, the discrete `selectLod` instance-scale gap) are listed under **Parked** below and in
  [`roadmap.md`](roadmap.md). Progress:
  - **Step 1 — instrumentation (done).** Per-channel refine attribution (`ActiveFront::channelStats()`
    + the overlay "VDPM splits" line) exposes which channel drives each split; a `[!shouldfail]`
    scale-invariance test pins the invariant before the metric changes.
  - **Step 2 — shading correspondence decoupled (done).** `normalStep`/`tangentStep` used to be
    computed only when the removed vertex projected *inside* a surviving one-ring face, and unlike the
    UV channel (which has a post-loop wedge-spread fallback) they had no fallback — so an endpoint
    collapse whose point lands outside every survivor recorded **zero** shading error (a hole, the
    likely dominant cause of close-range interior faceting). They now correspond to the **closest point
    on the nearest surviving triangle** (clamped barycentric, `closestPointBary` in
    `mesh_simplifier.cpp`), independent of the UV containment rule — the UV channel keeps containment
    (an affine chart must read exactly 0) + its wedge-spread fallback. Collapse order is unchanged
    (the shading channels don't steer selection), so only the recorded normal/tangent radii change.
  - **Step 3 — geometry against the nearest actual triangle (done).** The geometry channel used to be
    the `min` over *every* one-ring face's *infinite plane* distance, so on a curved neighbourhood an
    unrelated face whose plane happened to pass near `removed` quieted it below what point-to-plane
    inherently requires. It now measures point-to-plane against the **nearest actual surviving
    triangle** (least point-to-triangle distance — the same selection the shading channels use). A flat
    gap still reads ~0 (the nearest triangle is coplanar); a curved patch can no longer borrow a
    coincident unrelated plane. VDPM-only, collapse order unchanged.
  - **Step 4 — per-split support bounds + scale-invariant angular projection (done).** Each collapse
    now records a **support radius** (`MeshCollapse::supportRadius` → `VertexSplit::supportRadius`): a
    bounding sphere around `kept` grown to enclose the collapsed subtree (conservative — centred on
    kept, radius = max(own, edge length + child's), so it can over-estimate the true diameter; a real
    merged sphere is the tightening follow-up). `refineForView` projects the angular channels as
    `2·sin(θ/2) · scale · boost · extent`, where `extent = worldSupport · projScaleY · halfViewport /
    nearDistance` is the same geometric projection the geometry channel uses, so an angular error
    refines in proportion to the projected **screen extent** of the region it affects, the angle is a
    dimensionless chord (`2·sin(θ/2)`, ≈θ small-angle, saturating at 2), and the shading score scales
    **exactly like geometry**. Three details make it robust: (a) the support extent is projected from
    the **parent** (= kept) near-sphere depth (`parentDistance − worldSupport`), matching the sphere's
    centre; (b) the object-space deviation + support radii are bounded into world space by
    **`worldLengthScale`** — the largest singular value of the world matrix's linear part (exact for
    uniform scale, conservative for non-uniform) — so a mesh **instanced at a non-unit world scale**
    refines correctly, not just one authored bigger; (c) the accumulated angular radii are **capped at
    π** (a directional angular error can't exceed it, and the chord is only monotone up to π). The
    scale-invariance test (formerly `[!shouldfail]`) drives the *same* forest at identity vs a
    `scale(k)` world with the camera at k× distance and requires an identical front — the production
    instance-transform path, with no QEM-ordering noise from rebuilding a scaled mesh. Still VDPM-only,
    collapse order unchanged. *(The discrete `selectLod` still projects object-space error against
    world distance without the instance scale — the same fix should reach it eventually.)*
  - **Step 5 — full TBN tangent metric (done).** The tangent channel used to compare raw tangent
    **xyz**, but the shader never samples that directly: it builds a per-vertex TBN frame (`shader.vert`)
    — `T` Gram-Schmidt-orthogonalised against `N`, `B = cross(N,T)·handedness` — and a normal map is
    sampled in *that* frame. So a **handedness (`w`) flip** (which flips `B`, very visible on a normal
    map) read as **zero** error. The simplifier now precomputes the frame axes per vertex (`tanFrameT_`
    / `tanFrameB_`) and the tangent channel measures the **MAX of the T- and B-axis angular deviation**,
    catching both a tangent roll and a handedness flip; still independent of the shading-normal channel
    (which covers `N`). Unit-tested through `measureCollapseDeviation` (a hand-built frame with an
    opposite-`w` bitangent registers ~π; consistent handedness reads 0).
  - **Step 6 — material-aware tolerances (done).** `vdpmChannelScales(material)` (a pure,
    Vulkan-free helper in `graphics/vdpm_material.*`) derives the per-channel refine scales at *refine
    time* from the binding's active material, so a channel a material can't show is disabled (scale 0
    ⇒ its score never exceeds budget) and the collapse stream stays material-agnostic / reusable across
    variants. Rules: **unlit** → normal + tangent off (no shading); **no tangent-space normal map**
    (base or clearcoat) → tangent off (a mesh shipping tangents no longer protects a frame nothing
    samples); **no textures at all** → UV off; **glossy** → the normal channel is scaled up
    (`kVdpmGlossyNormalBoost`, ramped by `1 − roughness`) since a shading error reads far more strongly
    in a sharp highlight. Passed straight into `refineForView`'s existing `uv/normal/tangentScale`
    args at the `object.cpp` call site — no change to the simplifier, forest, or front. Unit-tested
    per rule.
  - **Step 7 — persistent front + hysteresis (done).** `refineForView` no longer `coarsenAll()`s every
    frame — the front persists across frames. Each frame: a score pass tags every split (max of its
    four channels + a back-face flag), a refine pass (coarse-first) pulls in splits over `pixelBudget`,
    and a coarsen pass (fine-first fixpoint) drops refined splits under `kVdpmCoarsenRatio × pixelBudget`
    (or back-face-culled) whose child is a leaf. The **dead band** between the two thresholds is the
    hysteresis: a split whose score hovers at the budget stops popping in/out under small camera moves
    or TAA jitter. A static camera now yields an identical front every frame; the repair passes still
    run each frame so correctness is unchanged. Steady-state also does *less* work than the old rebuild
    (only the sub-band splits coarsen, vs coarsenAll's full collapse). *(This is the last metric-arc
    step.)*

- ✅ **VDPM visibility cones.** The precomputed conservative normal cone (§ Visibility cones) now
  drives the back-face gate + silhouette boost, replacing the smooth-normal proxy. **It does NOT retire
  the repair sweeps**, which was the original hope: coverage is a screen-space property and foldover is
  topological, while the cone is a face-orientation bound. A force-refine-on-straddle experiment reduced
  but could not zero coverage repairs (10/19/85 on the coverage test), confirming the categories differ.
  The win is real but narrower: the cone halved `refineForView` and is GPU-shaped, but the joint
  `repairFront` (now the dominant cost) stays. (Bounding-cone caveats per Hoppe, *View-Dependent
  Refinement of Progressive Meshes*, SIGGRAPH 97, §4.)
- **Parked:** texel-density UV budget; a GPU worklist/fixpoint (or representation-level guarantee) for
  the repair sweeps, which the cone cannot subsume.
- ✅ **GPU-driven active front (arc complete — Stage 0 → B5c-4; the GPU front is the default wherever
  the device supports it, with the CPU front as oracle + per-mesh fallback).** The whole
  per-frame front lifecycle (score → refine/coarsen → both repairs → emit) had to move to the GPU as a
  unit — a partial move round-trips the shared front state and erases the win — with an indirect draw
  (the CPU no longer knows the index count). The CPU `vdpm` stays the headless-tested oracle + fallback
  (the `cloth` CPU-ref ↔ `render/` GPU-impl pattern). **Stage 0 (`graphics/vdpm_parallel`, COMPLETE):**
  a GPU-shaped CPU model — the refine-dependency DAG
  with topological **ranks** (CSR by-rank layout) plus the per-split **dependency triple**
  (`dependencies[s]` = a `SplitDependencies{parent, vl, vr}` record = `removingSplit[parent/vl/vr]`,
  the splits that must fire before `s`) — the ONE authority the closure and the GPU B3 uploader share
  instead of each re-deriving the edges, mapping 1:1 onto the GPU record. The slots are kept separate
  (not de-duplicated) so the DAG's per-slot edge correspondence is preserved — a distinct concern from
  `dependents_`, which the refine increments from the VERTEX slots parent/vl/vr, NOT from these
  dependency splits. `ParallelFront` reproduces the recursive front by **rank-ordered data-parallel
  passes** (requirement-closure → ascending-rank refine → descending-rank coarsen) instead of
  recursion, proven byte-for-byte against the oracle. Front consistency is checked by the shared, pure
  `validateFrontInvariants` (reconstruct `dependents_` from the refined splits; verify dependency
  activeness + the active↔refined relation; require every flag to be exactly 0/1) — templated over the
  flag width with uint8 (CPU) and uint32 (GPU read-back) overloads, so the GPU's 32-bit flags are
  checked with **no lossy narrowing** (256 → 0 can't slip through). Called by
  `ParallelFront::validateInvariants` and, later, by the GPU B3 harness on read-back state. The **parallel repairs**: `ParallelFront::repairFront`
  reuses the shared `detail::` classifiers to detect every foldover ∪ coverage violation against a
  **settled** front, closes + applies the targets in rank order (`closeAndApplyRequired`), and iterates
  to an inflationary fixed point — sharing P2's per-face policy + final invariants (zero foldover, zero
  coverage) but not its sequential schedule, so it may reach a different valid front. Evidence on a
  curved sphere: it converges in **2 detection passes / 1 apply round** and over-refines the sequential
  joint repair by only **1–3 triangles** (≤0.2%), under the hard bound (the full finest detail). And the
  **deterministic seam-preserving emit**: `ParallelFront::emitActiveIndices` is the GPU-shaped compaction
  — memoise each vertex's active ancestor → a per-face **survival flag** → an exclusive **prefix sum** →
  a stable **scatter** of three wedge-restored corners at `out[3·slot]` (no atomic append, so triangle
  order matches the original faces — blend/transmission need that), restoring each corner to the
  `nearestWedge` at its ancestor via a **CSR** wedge adjacency (`mesh_topology::canonicalWedgesCsr`, the
  GPU uploader's shape). It is **byte-identical** to the oracle's emit (indices, order, wedges) for a
  given front. This was the arc's stop/go gate — it proved the parallel scheduling, repair convergence,
  AND emit on CPU in CI, plus the dispatch-depth (curved ~30 ranks; flat grids scale linearly) and
  wedge-ABI evidence (CSR vs a precomputed corner→wedge map) — before any GLSL. **Stage A** (indirect
  draw) then landed the `drawIndexedIndirect` plumbing (de-risked on MoltenVK). The **GPU port** is
  now underway, starting with **B1 — scoring**: `refineForView`'s per-instance derivation + per-split
  math were extracted into ONE Vulkan-free authority (`makeVdpmViewParams` / `scoreVdpmSplit`); a
  headless surface-free `Device` compute mode + compute-only `Resources` stand up an offscreen device;
  `shaders/vdpm_score.comp` (typed buffer_reference, no descriptors) reproduces `scoreVdpmSplit` split
  by split; `VdpmGpuMesh` (shared static splits + canonical positions) and `VdpmGpuFront` (per-instance
  score output + per-frame mapped params) own the buffers; and a local `[.][gpu]` readback harness
  cross-checks the shader against the CPU authority (scores close, back-face decisions exact). The
  std430 ABI (`render/ubo.hpp` `VdpmSplitGpu`/`VdpmScoreOut`/`VdpmScoreParams`) is fully
  offset-asserted; `worldLinear` is a padded 3-column mat3, all flags are uint32, and the params block
  carries the three buffer_reference addresses (only its own address is pushed). **B2 — emit** then
  ported `ParallelFront::emitActiveIndices` to the GPU, **byte-identically**, from a CPU-uploaded front.
  Two precomputed Vulkan-free structures make byte-identity *structural* rather than float-dependent:
  `buildWedgeChoices` bakes the CPU's `nearestWedge` decision for each (original vertex, ancestor depth)
  into a CSR — so GPU restoration is pure integer indexing (a GPU-recomputed distance could tie-break a
  near-equal wedge differently) — and `buildRemovalParent` collapses the removal chain to one dependent
  load per step. Four descriptor-free passes (`shaders/vdpm_ancestor|survival|scatter|emit_finalize`)
  around the shared `VdpmScan` exclusive-scan primitive (recursive Blelloch, 256-element blocks): the
  **ancestor** pass walks the removal-parent chain (bounded by `maxDepth`, one atomic failure counter)
  to each canonical's active ancestor + depth; **survival** flags a face iff its three ancestors are
  distinct and none failed; the **scan** turns the flags into a stable per-face output slot + the
  surviving total; **scatter** writes each surviving face's three restored-wedge corners at `3·slot` in
  original face order (no atomic append); a one-invocation **finalize** sets the emitted index count to
  `3·survivors`. `VdpmGpuMesh`'s full build uploads the static emit data (indices/weld/removal-parent/
  wedge CSR, index-range-validated); `VdpmGpuFront::recordEmit` clears a single 3-uint counters buffer
  once (`fillBuffer`, an `eClear`→compute barrier) so the atomic, scan total, and index count all start
  defined with no CPU readback, then records the passes with compute→compute barriers (one after the
  scan feeding BOTH scatter and finalize). The `[.][gpu]` harness proves byte-identity (indices, order,
  wedges) across coarsest/partial/full/repaired fronts on a sphere + per-corner-seamed grid, plus
  determinism, an empty mesh, an all-faces-collapse front (faceCount > 0), and a deepest-chain fixture
  pinning the ancestor bound's off-by-one. `VdpmGpuMesh::fitsComputeDispatchLimits` gates GPU
  eligibility (static dispatch counts vs the device cap) before allocation, so B5's backend selector
  can fall back to the CPU front rather than fault at record time. **B3 — refine/coarsen** then moved
  the persistent front STATE (active/refined/dependents/required, device-resident uint32) onto the GPU,
  updated by rank-ordered dispatches — the port of `ParallelFront::applyView`, matched INTEGER-EXACT.
  Two Vulkan-free prep pieces landed first (`DependencyDag`'s per-split dependency triple as the shared
  closure authority + a templated `validateFrontInvariants` the GPU harness reuses on read-back state).
  The four passes — `vdpm_mark` (seed `backface==0 && max(geometry,uv,normal,tangent) > budget` from
  the production `VdpmScoreOut` — straddle excluded, matching `VdpmSplitScore::score`), `vdpm_close`
  (requirement closure over the DAG, one rank/dispatch DESCENDING, atomic-OR), `vdpm_refine` (apply
  ASCENDING, atomic-add `dependents` per vertex-slot), `vdpm_coarsen` (collapse DESCENDING) — wrap a
  **shared `recordCloseAndRefineRequired`** that owns its boundary barriers (leading seed→closure,
  between ranks, trailing refine→consumer), so B4's repair round reuses the identical scheduler. A GPU
  **invariant-failure flag** (required-refine-with-inactive-dependency / dependents underflow) gives a
  scheduling bug a precise diagnosis with no runtime readback; the harness asserts it stays 0.
  `VdpmFrontSplitGpu` (32 B) carries the vertex slots (for `dependents`/child) + the dependency-split
  slots (for closure); `rankOffsets` stays CPU-side driving per-rank `(offset,count)` push constants
  (only `splitsByRank` is uploaded). Evidence (`[B3Evidence]`): `1 + 3(maxRank+1)` dispatches per
  instance-frame — sphere maxRank 18 ⇒ 58, a deep flat grid maxRank 312 ⇒ 940 — so B5 weighs batching
  same-mesh instances by rank / a work queue for deep forests. **B4 — repairs** then ported the
  snapshot foldover ∪ coverage fixpoint (`ParallelFront::repairFront`) to the GPU, run after `applyView`.
  Each round clears `required`, runs the **shared ancestor resolve** (factored out of `recordEmit`, one
  resolve/canonical-vertex against the live front), a **detect** pass (`vdpm_repair_detect.comp` — a
  faithful port of `detail::isFoldover` + `detail::classifyCoverageRepair`, marking each violation's
  inactive-corner removing split), then the shared `recordCloseAndRefineRequired`. As the classifiers
  are screen-space FP, the contract is the P2 one — zero CPU-classified foldovers + zero coverage
  failures on the read-back front, valid invariants, no failure flags, emitted ≤ full detail — NOT
  integer-exact. Convergence is bounded + GPU-resident with no per-round readback: a round budget then a
  final detect + a **full-detail fallback** (`vdpm_repair_fallback.comp`, seeds every unrefined split
  iff a violation remains → always hole-free), with a separate repair-control buffer `{anyMarked,
  ancestorFailure, fallbackFired}`. Sync: a leading compute→(compute|clear) barrier + a per-round
  compute→eClear→compute `required` reset. The `[.][gpu]` harness proves the contract, plus DIRECT
  per-face classification readback matching the CPU classifiers **exactly** across mixed fronts (every
  branch) and the fallback path (tiny budget → fires → hole-free). **B5a — combined runtime front**
  then assembled the production unit: `buildRuntime` puts scoring + persistent refine/coarsen state +
  repair scratch + emit workspace in ONE front, with the draw-consumed outputs (emitted indices,
  indirect command, counters) + host repair params RINGED per frame-in-flight (persistent front,
  transient output). `recordApplyScoredView` reads the front's own GPU score output and
  `recordEmitFromFront(frameIndex)` the live front state — no host round-trip (both delegate to shared
  `record*Impl` recorders); the emit finalize writes the full 5-word indirect command; `recordFrame`
  chains the lot GPU-only. The `[.][gpu]` harness splits the contract: off-threshold (cull-off, tiny
  budget) exact front + emit vs the CPU lifecycle; general fixtures need only a valid hole-free front +
  emitted ≤ finest + clean diagnostics + a correct 5-word indirect (`indexCount == counters[2]`); a
  two-frame back-to-back test proves the ring. **B5b-1 — renderer integration (registration + compute,
  CPU output still drawn)** then wired the manager into the live pipeline. A Vulkan-free registration
  seam `graphics/vdpm_gpu_registry.hpp` (`VdpmGpuRegistry` interface + the generational identity
  handles `VdpmMeshHandle`/`VdpmFrontHandle` in `gpu_handle.hpp` + the semantic `VdpmWorkRequest`)
  is implemented by `render::VdpmGpuManager` (`render/vdpm_gpu_manager.{hpp,cpp}`): it owns the reusable
  score/refine/repair/emit pipeline bundles + a per-geometry mesh table and per-instance front table
  (both `GenerationalSlotPool`-keyed), and is built by the Renderer **only** after the
  `VdpmScan::deviceSupported` capability check (an unsupported device leaves it null — construction
  never fails, the CPU front stays usable; a per-mesh dispatch-limit ineligibility falls back, logged
  once). The load path threads the registry (`GltfLoader::loadScene` → `Geometry::load` registers the
  shared forest once → `Object::load` creates a per-instance front). Each frame, when the backend is
  selected (`RenderTunables::vdpmGpuBackend`, CLI `--vdpm-gpu`) and the device is capable, `Object`
  tags each camera-visible forward `DrawCommand` with its `VdpmFrontHandle` and appends a
  `VdpmWorkRequest` to the renderer-owned sink; the Renderer distils the sink to the fronts whose
  FORWARD draw survived the camera cull (a shadow-only instance never runs compute), dedups by handle,
  and records `VdpmGpuManager::recordRequests` after `collectDrawCommands`. In B5b-1 the compute was a
  **shadow run** — the CPU front still emitted the drawn buffers. **B5b-2 flipped the draw to the GPU
  output.** For a GPU-backed instance (backend selected + a live front) `writeForwardUniforms` now SKIPS
  the CPU `refineForView`/`repairFront`/`emitActiveIndices` lifecycle (the per-instance CPU cost this
  arc set out to retire); `buildDrawCommands` tags the forward `DrawCommand` with the front handle, sets
  `indexType = UInt32` (the GPU stream is always uint32) + `indexCount = 0` (GPU-only, overlay shows it
  once B5c adds delayed diagnostics), and emits the work request. The Renderer, in the same forward-bucket
  walk that harvests handles, calls `VdpmGpuManager::resolveDrawBuffers(front, frame)` — ONE handle lookup
  returning the GPU-emitted index + indirect ring buffers, **throwing `std::logic_error` if a tagged front
  doesn't resolve** (an invariant violation: eligibility was decided at registration, and a zero-`indexCount`
  draw must never slip through) — and points `indexBuffer`/`indirectBuffer` at them; the shared
  `recordIndexedDraw` (prepass / forward / transmission) is UNCHANGED. The compute→(index-read +
  indirect-read) barrier (`eComputeShader`/`eShaderStorageWrite` → `eIndexInput|eDrawIndirect`/
  `eIndexRead|eIndirectCommandRead`) is delayed to just before the depth prepass, AFTER the shadow pass
  (shadows keep discrete/direct + their draw copies clear `vdpmGpuFront`, so shadow-pass GPU work overlaps
  the compute). Per-mesh fallback instances (ineligible mesh → NullVdpmFront) keep the CPU path; stale CPU
  repair/channel stats for GPU instances are suppressed via a per-binding `vdpmCpuRanThisFrame` flag until
  B5c. Verified 0-VUID drawing GPU output on the helmet (opaque + prepass) and TransmissionTest (13 fronts,
  transmission path). **Perf arc opened (B5c/default-flip paused):** the GPU backend is **dispatch-bound,
  not triangle-bound** — the front lifecycle records ≈ `B·(2R+2) + 5R + 12` compute dispatches + a matching
  barrier count per instance per frame (R = rank count, B = repair round budget = 24), ~94% of it the
  fixed-round repair that records ALL 24 rounds even after round-1 convergence (plus a full close/refine
  after the fallback). Instrumented (no behaviour change): `ProfilePass::VdpmCompute` GPU timestamp, CPU
  wall time around `recordRequests`, and the analytic dispatch/barrier tally per front
  (`VdpmGpuFront::analyticComputeCost`, `VdpmGpuManager::ComputeStats`, overlay + `FE_LOG=render:debug`).
  Measured: the 15k-tri helmet is **~1014 dispatches + ~1066 barriers/frame** (1 front, 18 ranks);
  TransmissionTest **~19,700 dispatches + ~20,400 barriers/frame** (13 fronts). MoltenVK's per-command
  Metal translation makes CPU recording rival shader time; vsync turns the 16.7ms crossing into an abrupt
  60→30. **The per-round convergence evidence is decisive** (captured GPU-resident via address
  redirection — each bounded detect atomic-ORs its `anyMarked` into a per-round history slot, no shader
  change, delayed readback): the helmet repair **converges in 2 of the 24 rounds** (marked-then-clean
  prefix, fallback never fires) — so ~22 rounds (~87% of the dispatches) do nothing. A repair-budget sweep
  confirms it: budget **2 emits the IDENTICAL front** (38970 indices, no fallback) at **~178 dispatches vs
  ~1014 — 5.7×** fewer; budget 1 is insufficient (a violation remains → fallback fires → over-refines to
  full detail 46356). **Wall-time baseline** (helmet, MoltenVK, timestamps valid): at B=24 the VDPM
  compute is **~17 ms GPU + ~2.2 ms CPU record** — it alone blows the 16.7 ms vsync budget (the 60→30
  drop); at B=2 it is **~3.6 ms GPU + ~0.5 ms CPU**. GPU compute dominates CPU record ~8:1, so most of the
  cost is the GPU serialising across ~1000 barriers between tiny dispatches, not just MoltenVK command
  translation — both fall ~4–5× at B=2. **Next: the repair-scheduling arc — NOT a fixed lower budget and
  NOT dispatchIndirect early-out alone (which still leaves ~1000 commands to translate + barrier).** The
  fix is a **scheduler replacement** that folds convergence in: one persistent workgroup per front (clear
  → resolve/classify in strided loops → shared-memory reduce anyMarked → uniform exit when clean → process
  ranks in-workgroup with barrier()/memoryBarrierBuffer() → final detect + the existing full-detail
  fallback), batched across all fronts in ONE dispatch — turning ~1000 commands into one while keeping
  genuine GPU-resident convergence; graduate to a multi-workgroup queue/wave design only if one workgroup
  is too serial for large meshes. Implications: upload the rank offsets (currently CPU-side), share the
  GLSL coverage/foldover classifier between the existing detector and the new kernel, keep the current
  recorder as the tested reference/fallback until invariants + output match. Hoisting the per-rank
  pipeline binds is now only a small cleanup to that retained reference path, not a priority. **Stage 1
  (groundwork) has landed:** the foldover/coverage screen-space policy is the PURE shared include
  `shaders/vdpm_repair_classify.glsl` (`classifyRepairFace → {foldover, coverageKind, worstLocal}`, no
  buffer access / atomics / marking — the detect shader + the coming kernel share one classifier;
  byte-identical, the classification `[.][gpu]` test unchanged), and the per-rank `RankRange{offset,count}`
  list is uploaded device-local (`binding.rankRangesAddress`, the `VdpmRankRangeGpu` ABI in ubo.hpp,
  derived + validated as a contiguous partition of `splitsByRank` at the build boundary BEFORE any upload;
  a `[vdpm]` CI test pins the reject branches). **Stage 2 (the persistent-workgroup kernel) has landed
  too:** `shaders/vdpm_repair_kernel.comp` runs the whole repair fixpoint for one front in ONE workgroup /
  ONE dispatch (the `VdpmRepairJobGpu` batch ABI in ubo.hpp; `s_anyMarked` shared reduction for the
  uniform convergence exit; `VdpmRepairKernel` capability-gated; `VdpmGpuFront::prepareRepairJob` (the
  public write-params-and-pack authority; the raw `makeRepairJob` packer is private) +
  `recordRepairKernel`), built beside the recorder and exercised by 6 `[.][gpu]` control-flow fixtures.
  **Stage 3 (the flip) has landed too:** `recordFrame` now drives repair through the kernel when the
  device supports it (`VdpmGpuManager` holds a `std::optional<VdpmRepairKernel>`, emplaced iff
  `deviceSupported`; `recordFrame` takes a nullable `const VdpmRepairKernel*` and falls back to the
  recorder — now `recordRepairRuntime` — when null), with an explicit kernel→emit compute barrier
  (the emit counter clear only orders the counter, not the kernel's active-front writes). It has no
  default, so the repair path is chosen explicitly at every call site (a forgotten argument can't
  silently pick the ~1000-command recorder). `analyticComputeCost` (signature detailed in the apply
  bullet below) is now EXACT: the always-present lifecycle barrier + emit are counted (emit runs over
  FACES, so a zero-rank front is not free), and the emit's scan cost is derived from `faceCount`
  (K internal levels → `{2K+1, 2K}`) instead of assuming one hierarchy — the two-level helmet (R=18)
  gives recorder-apply+kernel-repair **64/65** vs both-recorders ~1014/1066; a `[vdpm]` CI test pins
  the paths across scan tiers + the zero-rank case so a recorder change can't silently stale the overlay. **Cross-check gate (`[.][gpu]`):** two `buildRuntime` fronts settle from
  one mesh to a bit-identical pre-repair state (proven by reading back active/refined/dependents/required/
  failFlags), then one is repaired by the kernel and one by the recorder; post-repair front + emitted
  indices + control/history diagnostics + zero failure flags match **exactly** across normal-convergence,
  forced-fallback, reflected+cull, and non-uniform/no-cull cases — plus a CPU-oracle gate on the kernel
  front (`validateFrontInvariants`, `foldoverCount == 0`, `coverageFailures == 0`, emitted ≤ finest). This
  is GPU-vs-GPU **exact** on the tested fixtures; it does NOT claim universal visual parity (screen-space FP
  can still diverge from the CPU oracle at thresholds). **Measured (helmet, R=18): CPU record 2.1 → 0.26 ms
  (~8×), GPU compute ~17.5 → ~2.4 ms (~7×), 1014 → 64 dispatches** — single-workgroup occupancy did NOT cap
  the win.
- ✅ **Apply persistent-kernel arc — checkpoint 0 (the instrumentation that steered it).** After Stage 3 moved the
  bottleneck, `recordFrame` gained opt-in per-stage timing (CPU `steady_clock` around each of
  score/apply/repair/emit + GPU bottom-of-pipe boundary timestamps into new `ProfilePass::Vdpm{Score,
  Apply,Repair,Emit}`, written only when a SINGLE front records — the query slots are one-shot; a null
  `VdpmStageProfile*` is zero-overhead in production). **Measured per-stage** (`FE_LOG=render:debug`):
  *helmet, 1 front* — GPU score ~0.02 / **apply ~1.2 (~58%)** / **repair ~0.7 (~35%)** / emit ~0.06 ms
  (of ~2.0 ms VdpmCompute); CPU-record **apply ~0.17 (~80%)** / rest tiny. *TransmissionTest, 13 fronts*
  — CPU-record **apply ~2.0 (~89%)** of ~2.24 ms (GPU per-stage n/a for >1 front). **Conclusions:**
  apply dominates BOTH CPU-record and GPU time ⇒ the apply persistent-kernel (mark/close/refine/coarsen
  in one workgroup, reusing the repair kernel's rank loop) is justified — it sheds apply's ~54 barriers
  (the CPU-record cost + most of the 1.2 ms GPU). **Surprise the checkpoint caught:** the persistent
  *repair* kernel, though ONE dispatch, still costs ~0.7 ms / 35% GPU — a single workgroup is
  occupancy-bound. So the apply kernel (also one workgroup) will land near a repair-like floor, not
  free; expect a solid but not proportional GPU drop, and the emerging frontier past the apply kernel is
  single-workgroup occupancy / **front-batching** (N fronts = N workgroups fill the GPU — which is why
  the job ABI is kept batch-ready). **The apply kernel is now SELECTED by `recordFrame` in production**
  (the recorder, `recordApplyScoredView`, is retained as the reference/fallback for unsupported
  devices): the mark/coarsen policy + `ScoreOut` struct/reduction are the shared
  `shaders/vdpm_apply_classify.glsl` (one authority with `vdpm_score/mark/coarsen`, no behaviour
  change), and `shaders/vdpm_apply_kernel.comp` + `VdpmApplyKernel` run the whole apply (reset failFlags
  → mark → close → refine → coarsen) for one front in ONE dispatch — close/refine reuse the repair
  kernel's loops, coarsen ports `vdpm_coarsen.comp`. `recordFrame` takes a nullable
  `const VdpmApplyKernel*` (no default — every stage's path is explicit) and the manager holds a
  `std::optional<VdpmApplyKernel>` emplaced iff `deviceSupported`; `VdpmGpuFront::prepareApplyJob` (the
  public pack authority) + `recordApplyKernel`; batch-ready `VdpmApplyJobGpu` ABI (budgets in the job).
  The workgroup size is a
  spec constant (`ComputeSpecializationConstant`, `layout(local_size_x_id = 0)`); the
  `[.][gpu][ApplySizeSweep]` benchmark (3 independent fronts, GPU-timestamped, warm-up excluded, median
  over rotated alternating cycles) picked **256** — fastest on the curved sphere, a tie on a deep grid.
  `analyticComputeCost(rank, faceCount, finestFaceCount, roundBudget, persistentApply, persistentRepair)`
  gained `persistentApply` ({1 dispatch, 1 barrier}) and the `finestFaceCount` repair gate (repair
  early-outs at 0 canonical faces, distinct from the emit `faceCount`); a `[vdpm]` test pins all four
  combinations + the empty-boundary cases (a selected-but-early-out kernel counts as zero, not one
  dispatch — including ranks-present-but-no-repair-faces).
  `test_vdpm_apply_kernel.cpp` (`[.][gpu]`) cross-checks BIT-EXACT vs the recorder from proven-identical
  scores across refine→coarsen / alternating budgets / diamond deps / back-facing coarsen, plus gates
  for the in-kernel failFlags reset, the coarsen dependents-underflow flag, a zero-split no-dispatch
  no-op, and a deep rank chain. **Measured (helmet, both kernels now on): 64 → ~10 dispatches; apply GPU
  ~1.2 → ~0.4 ms, apply CPU-record ~0.17 → ~0.005 ms; total VdpmCompute ~2.0 → ~1.2–1.5 ms GPU, total
  CPU-record ~0.26 → ~0.06 ms; 0-VUID on helmet + TransmissionTest + DamagedHelmetBlend.** Apply landed
  BELOW repair's floor (~0.4 vs ~0.75 ms — apply is less work), so REPAIR is now the dominant GPU stage;
  both are occupancy-bound single workgroups (one workgroup can't fill the GPU — the residual is the
  occupancy floor, not dispatch/barrier overhead).
- **Front-batching (landed) — the occupancy lever for MULTI-front scenes.** A single-front scene can't
  gain (one front = one workgroup); but a multi-front scene (TransmissionTest = 13 fronts) previously ran
  13 per-front lifecycles back-to-back — 13 under-occupied single-workgroup apply + repair dispatches,
  serialised by per-front barriers. `VdpmGpuManager::recordBatched` collapses this to **STAGE-MAJOR**:
  ONE lifecycle barrier → score every front → ONE `dispatch(Na)` apply → ONE `dispatch(Nr)` repair → ONE
  final front-state→emit barrier → emit every front, so the N fronts' workgroups run concurrently and
  fill the GPU. Requests are resolved ONCE into a reused scratch (front* + derived params + work flags);
  the apply/repair job arrays are COMPACTED (apply: `splitCount>0`; repair: additionally
  `finestFaceCount>0`, so an apply-only front is in the apply batch but out of the repair batch) into two
  manager-owned host-visible BDA arrays (one per frame-slot, geometric growth, replaced arrays RETAINED
  until session end so an in-flight frame never reads a freed BDA); a batch exceeding the 1-D group cap
  CHUNKS by advancing the job BDA `firstJob*stride` (disjoint jobs → no inter-chunk barrier; stride is
  8-aligned, the shader's `buffer_reference_align`, `static_assert`ed). `VdpmApplyKernel`/
  `VdpmRepairKernel::recordDispatch(cmd, jobsAddress, jobCount)` is the shared push-ABI authority
  (per-front recorder = `recordDispatch(…, 1)`); `VdpmGpuFront::{recordLifecycleBoundary,
  recordComputeStageBoundary}` are the shared barrier authorities recordFrame + the batched path both
  record, so the two paths can't drift. The manager routes to the batched path when BOTH kernels are
  available and the frame is not a profiled single front (a profiled single front keeps `recordFrame` for
  its per-stage GPU timestamps — dispatch(N) can't stamp them); the per-front path is the unchanged
  fallback. `VdpmGpuFront::analyticBatchedCost(fronts, groupCap)` is the EXACT stage-major aggregate
  (lifecycle + Σ score + Σ emit + ⌈Na/cap⌉ + ⌈Nr/cap⌉ dispatches; +[Na>0]+[Nr>0]+[Na>0‖Nr>0] barriers),
  pinned by a `[vdpm]` CI test. `test_vdpm_gpu_manager.cpp` (`[.][gpu]`) drives N fronts BATCHED via the
  real manager vs per-front `recordFrame`, asserting active/refined/dependents/required/failFlags + the
  indirect command + emitted indices **BIT-EXACT** across distinct per-request params, compaction
  (full / apply-only / zero-split), shuffled order, N=1, a job-array growth boundary, and two frame slots.
  **Measured (state-fair GPU-timestamp benchmark, 13 fronts, `[.][gpu][BatchBench]`): apply serial 3.07 →
  batched 0.25 ms (~12×), repair serial 4.99 → batched 0.41 ms (~12×)** — the serialised multi-front cost
  collapses toward the concurrent floor. (The benchmark is authoritative because the in-app `VdpmCompute`
  GPU timestamp reads `gpuValid=false` on heavier multi-front frames.) 0-VUID on TransmissionTest +
  DamagedHelmet + DamagedHelmetBlend (`--vdpm-gpu`; AlphaBlendModeTest sits below the VDPM eligibility
  threshold, so it does not exercise this path).
- **Apply+repair FUSION — SKIPPED (complexity/value judgment).** A fused kernel (apply then repair per
  front in one workgroup, `wgsync` between, dropping the apply→repair barrier + a dispatch) can ONLY
  reclaim the per-workgroup tail across the global apply→repair barrier (early-finishing apply
  workgroups idling until the slowest clears before any repair starts). The fenced
  `[.][gpu][FusionCeiling]` benchmark bounds this on a PROCEDURAL PROXY multi-front set (state-fair,
  medians): the optimistic ceiling `min(Tapply,Trepair)` is ~25% of the full lifecycle, but the TAIL
  heuristic (separate-stage floor `max(Aᵢ)+max(Rᵢ)` vs fused ideal `max(Aᵢ+Rᵢ)`) is **0 ms
  reclaimable**, and the argmax evidence shows why: the SAME front (the heaviest) is the critical path
  in BOTH apply and repair (`argmax(Aᵢ) == argmax(Rᵢ)`) — apply cost (∝ splits/ranks) and repair cost
  (∝ faces) both grow with mesh complexity, so the heaviest front tends to dominate both stages and
  there is no cross-front tail to overlap. **CAVEAT — this is a proxy, not a scene-specific
  measurement.** The proxies match triangle counts only; they do NOT reproduce TransmissionTest's real
  collapse forests, DAG ranks (its live max rank ~45 vs the proxies'), per-instance transforms, or
  materials (every job shares one identity world/view/params — but VDPM work is instance- and
  view-dependent). So a scene-specific NO-GO would require loading the real meshes + transforms +
  acceptance camera. It is not built because the payoff is small even optimistically and a fused kernel
  is heavy machinery (fused ABI + shader + bit-exact cross-check) with real MoltenVK/Metal occupancy
  risk — a poor trade regardless of the exact per-scene tail. The occupancy arc closes here. (Side
  finding, MEASURED not inferred: emit is a large share of the lifecycle — ~17–56% depending on
  composition — so if a future dispatch lever is ever wanted, multi-front emit compaction is the
  candidate, though substantially more invasive.)
- **B5c-1 (GPU-sourced overlay diagnostics) — landed.** The overlay's triangle count + repair health
  for GPU-driven fronts (previously suppressed as stale CPU counters, `vdpmCpuRanThisFrame`) now come
  from the GPU. A single-invocation `shaders/vdpm_health_reduce.comp` folds every recorded front's
  health (per-front `VdpmHealthJobGpu` array — counters/repairControl/roundHistory/failFlags by BDA +
  a submitted-draw multiplier) into ONE scene-wide `VdpmSceneHealthGpu` (64-bit emitted total {lo,hi}
  with a manual carry; repair-front / max+sum marked-round / fallback / non-clean-prefix / ancestor /
  B3-fail counts — health-oriented, NOT the CPU foldover/coverage vertex counts). The reduction runs
  inside the `VdpmCompute` timestamp; a device-local scene-health ring (per frame-slot) is copied to a
  host readback ring OUTSIDE it, parsed a frames-in-flight cycle later, and COMBINED with that same
  frame's stored CPU triangle subtotal so the displayed total is frame-consistent. The `vdpmCpuRan-
  ThisFrame` guard is KEPT (CPU rows labelled "CPU fronts only"); channel attribution shows "n/a" for
  GPU fronts (no GPU counters built). Draw multipliers are matched BY HANDLE from the real forward
  buckets. `test_vdpm_health_reduce.cpp` drives the reducer directly (mixed repair, submitted-draw
  weighting, dirty-history prefix, ancestor/fallback, 64-bit carry); `test_vdpm_gpu_manager.cpp` pins
  the K-frame readback delay + slot isolation. 0-VUID on DamagedHelmet `--vdpm-gpu`.
- **B5c-2 (real-asset helmet evidence) — landed.** `test_vdpm_helmet_evidence.cpp` (`[.][gpu]`,
  local-only) decodes the REAL DamagedHelmet through the production `GltfLoader::loadScene` and uses the
  collapse stream `Geometry::load` already built (`geometry.collapses()` — no re-simplify), selecting the
  UNIQUE VDPM-eligible geometry. Two cleanly-separated claims: **(A) byte-identity** — from a CPU-uploaded
  active set the GPU emit is BYTE-IDENTICAL to `ParallelFront::emitActiveIndices` at roots / mid / full
  (mid asserted strictly between; the non-full cases asserted to actually reach nonzero ancestor depth +
  multi-wedge seam buckets **among surviving faces** — resolved-ancestor-distinct — so the match can't be
  vacuous); **(B) full lifecycle** — the complete GPU `recordFrame` is invariant-valid, hole-free
  (CPU-classified foldover/coverage == 0), clean-failFlags, and bounded, with the emitted-count guarded
  (`counters[0]==0`, `[1]≤faces`, `[2]==3·[1]≤3·faces`) before any mapped readback. The exact GPU==CPU
  lifecycle comparison is GUARDED on the reference first reaching full detail — the real helmet at
  tiny-budget + cull-off does NOT (8 zero-score splits remain), so the guard correctly skips, proving that
  "tiny budget" ≠ full refinement (Claim A already supplies the structural byte-identity). Evidence WARNs
  report maxRank + the uploaded static footprint via `VdpmGpuMesh::staticByteFootprint()` (accumulated at
  every upload site across all twelve static B2–B4 buffers — uploaded payload bytes, not VMA suballocation
  padding: helmet ≈ 1.55 MB) and its `wedgeMapByteFootprint()` subset (choices + offsets, ≈ 202 kB).
- **B5c-3 (parity sign-off) — landed (2026-07-24, macOS/arm64).** Runbook, runtime control, and the
  recorded manual sign-off are all in (`docs/acceptance-testing.md`): DamagedHelmet and TransmissionTest
  both passed the full same-camera parity gate at two framings each — GPU failure flags all 0, visually
  identical under the in-process toggle + orbit, convergence within budget (max 8/24, no fallback), CPU
  round-trip exact. Non-zero CPU↔GPU count deltas (helmet ≤0.22%; TransmissionTest, 13 fronts, up to 5.8%
  at coarse detail) were assessed as no visible difference — the GPU front is slightly coarser as
  per-front screen-space FP diverges, silhouette hole-free. DamagedHelmetBlend was omitted from that
  original visual sign-off because of a validation-layer crash; the later forward descriptor-order fix
  (resolved `roadmap.md` (C)) restores a 0-VUID CPU/GPU validation smoke for the blend consumer, without
  claiming a retroactive visual/count parity record. B5c-4 unblocked. The
  `RenderTunables::vdpmGpuBackend` selector is now a runtime
  **overlay checkbox** ("GPU-driven front", in the Mesh LOD panel's view-dependent block) — the manager
  is built whenever the device supports the front (independent of the selector), so the flip takes effect
  the next frame with NO reload and an unsupported device shows an explicit "unsupported" label + stays on
  the CPU front (`FrameStats::vdpmGpuAvailable`, set from `vdpmManager_ != nullptr`). That makes the
  CPU↔GPU A/B a **same-process, same-camera** flip. `docs/acceptance-testing.md` gains the **empirical
  parity gate**: freeze the camera at a stable low-detail plateau, toggle the backend, record and assess
  any triangle-count delta, and require 0 fallback/non-clean/ancestor/B3-fail health flags. It is framed
  HONESTLY as empirical: screen-space FP can produce different valid fronts, counts do NOT prove
  identical indices, and Claim A owns structural identity; the visual checks + B5c-2 carry the
  correctness weight. The overlay backend toggle was pulled forward from B5c-4.
- **B5c-4 (default flip) — landed (2026-07-24).** The GPU-driven front is now the **default wherever the
  device supports it**: the CLI backend request is tri-state (`std::optional<bool>` in `DebugOptions`) —
  `--vdpm-gpu` forces on, `--no-vdpm-gpu` forces off, unset resolves in the Renderer to
  `VdpmScan::deviceSupported` (the same predicate that builds the manager), and repeated flags are
  last-one-wins (`test_application_args`). The stale B5b-1 "shadow run" comments (renderer.hpp /
  render_tunables.hpp / frame_info.hpp / object.cpp) are cleared. Unsupported devices and per-mesh
  ineligibility still fall back to the CPU front; the overlay checkbox toggles at runtime. **This
  completes the GPU-driven-front productionization arc (B5b → B5c).**
- NEXT: see [`roadmap.md`](roadmap.md) — the post-VDPM review backlog is cleared; the live arcs are
  shadow LOD ([`shadowplans.md`](shadowplans.md)), the architectural-review remainder, and the tiered
  static review.
- **7 forest skips.** `buildVertexForest` skips collapses whose edge diverged from its adjacency
  replay (7 of ~6800 on the helmet); past the first skip the forest is slightly unfaithful. The repairs
  cover the visible symptoms; truncating the stream at the first skip would be the clean structural fix.
- **Repair never reaches zero on the real helmet (observed).** Backing the camera off, the CPU foldover
  repair count settles to a non-zero floor (~40) and rises again at the very coarsest levels — there is
  **no zero-repair plateau**. This is *observed real-helmet behaviour*, phrased as such, not proven
  causality: it is **consistent with, and likely amplified by, the forest skips above**, but selective
  non-prefix fronts can require foldover repair even with a perfectly faithful forest, so the skips are
  not established as the sole cause. Consequence for the parity sign-off (`docs/acceptance-testing.md`):
  the gate must NOT require foldover/coverage or GPU-marked-rounds == 0 — repair is load-bearing for
  correctness. Note a *stable* triangle count proves only a stable final front, not a quiescent
  lifecycle (a region can coarsen and be repaired back every frame at an identical emitted count — that
  repeated work is real churn, distinct from the settled output).
- **Coarsest-level seam shift.** A position-welded seam vertex carries one representative UV, so when a
  seam vertex finally collapses *along* its seam, its representative UV shifts slightly. The chart veto
  (decision 8) already removes the worse failure — a seam vertex collapsing *across* into another chart,
  which sheared the texture. Eliminating the residual drift too needs full **per-wedge** attribute
  quadrics (each chart's quadric kept separate) — real machinery, diminishing returns.
- **Shadow LOD remains discrete.** Continuous/view-dependent refinement is applied to the forward/depth
  vertex path. Shadow draws still choose their biased discrete LOD, so a shadow silhouette can step
  independently of the main-view detail.

---

## File map

| File | Role |
|---|---|
| `include/fire_engine/graphics/lod.hpp` | `GeometryLod`, ratios/thresholds, `selectLod` (header-only) |
| `include/fire_engine/graphics/mesh_simplifier.hpp` | `MeshSimplifier` interface, `QuadricSimplifier`, `MeshCollapse`, `SimplifiedMesh`, `ProgressiveMesh` |
| `include/fire_engine/graphics/vipm.hpp`, `src/graphics/vipm.cpp` | VIPM morph payload, exact-cut morph-data build, `selectVipm` |
| `include/fire_engine/graphics/vdpm.hpp`, `src/graphics/vdpm.cpp` | VDPM: `VertexForest`/`buildVertexForest`, `ActiveFront` (`refineForView`, the joint `repairFront` over private `repairFoldoversSweep`/`repairCoverageSweep`, `emitActiveIndices`) |
| `src/graphics/mesh_simplifier.cpp` | The QEM engine (`QemRun`): R⁵ quadric, welding, wedge emit, chart veto, the four VDPM deviation channels, the two dials |
| `src/graphics/geometry.cpp` | `Geometry::load()` builds `lods_` + VIPM morph buffers + stores the VDPM collapse stream from one progressive artifact |
| `src/graphics/object.cpp` | Per-draw `selectLod` (forward + shadow), Continuous `selectVipm` uniforms, and per-instance VDPM `refineForView` + joint `repairFront` → dynamic index buffer |
| `graphics/frame_info.hpp`, `render/render_tunables.hpp`, `render/renderer.cpp` | `lodEnabled`/`lodMode`/`lodPixelErrorBudget` plumbing + jitter-free `currentViewProj` + triangles-drawn stat |
| `graphics/draw_command.hpp`, `render/ubo.hpp`, `shaders/shader.vert`, `shaders/shader.frag`, `render/debug_overlay.cpp` | VIPM morph binding/uniforms + per-draw `lodLevel` → push constant → LOD-tint debug view + overlay panel (3-mode selector) |
| `tests/graphics/test_mesh_simplifier.cpp`, `tests/graphics/test_vipm.cpp`, `tests/graphics/test_vdpm.cpp` | Headless correctness tests |
