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
      → repairFoldovers(...)                (inside refineForView) un-flip any backward-wound face
    ActiveFront::repairCoverage(...)        un-recede any visible face that leaks the background
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
  `writeForwardUniforms` runs `refineForView` + `repairCoverage` per frame, emitting the active index
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

### The two repair passes — why a correct emit still needs them

A selective front is a **non-prefix** cut of the collapse stream. The simplifier's `wouldFlip` only
certifies the *linear prefix* is flip-free, and the deviation channels are all *topological* — never
projected to screen. So two failure classes survive an otherwise-correct front, and each is fixed by a
**monotone** repair (force-refine only → converges, at worst to full detail):

1. **Foldovers** (`repairFoldovers`, run at the end of `refineForView`). A finest face whose
   active-ancestor replacement winds *against* the original is back-face-culled by the rasteriser → a
   hole. Sweep the finest faces; where replacement winding opposes original winding, force-refine the
   collapsed corners.
2. **Coverage / silhouette holes** (`repairCoverage`, called from `object.cpp` after `refineForView`
   with the **jitter-free** `currentViewProj`). A *closed, non-folded* front can still leak the
   background: at a silhouette a coarse replacement recedes inside a fine visible triangle's
   *projected footprint*. This is purely a screen-space property — boundary/foldover/manifold checks
   are all blind to it. For each **visible** finest face above a small projected-area gate: if its
   replacement is **degenerate** (the face collapsed to a sliver and was dropped — *not* safe to skip
   at a contour, only at an interior), force-refine its corners; else if its projected centroid falls
   outside the replacement in NDC, force-refine the corner with the largest screen displacement.
   "Visible" follows the draw's cull mode via a `rasterBackfaceCulling` arg (same as `refineForView`):
   with culling on only front-facing faces are visible; with it off (double-sided / blended material)
   back-faces render too and are covered as well (`gn` is the WORLD winding, so this is correct under a
   reflected world).
   **It must use the jitter-free view-projection** — feeding it the TAA-jittered `frame.proj` shifts
   the coverage test ±½px each frame and thrashes the front (a borderline hole flickering with the
   camera still).

**These repairs cannot be replaced by the normal cone** (§ Visibility cones): coverage is a
*screen-space* property and foldover is *topological*, while the cone is a face-*orientation* bound.
A force-refine-on-straddle experiment reduced but could not zero coverage repairs, so both sweeps
stay. Retiring them eventually needs a GPU worklist/fixpoint or a representation-level guarantee.

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
flag `refineForView` is passed (from the material's cull mode). The coverage repair's screen-area gate
(`kMinNdcArea` in `repairCoverage`) trades residual sub-pixel holes against extra triangles; lower it
to catch smaller holes.

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
- **Foldover repair** — no emitted triangle winds against the original (verified against a version
  with the repair disabled, so it's non-vacuous).
- **Coverage repair** — every visible triangle stays covered by its replacement, following the draw's
  cull policy (front-face-only when culling; both windings for a no-cull double-sided/blend material,
  verified against the pre-fix back-face gap; correct under a reflected world), *including* the
  degenerate-replacement case (again verified non-vacuous).

`selectLod` is header-only and directly unit-testable. Beyond the headless tests, the render smoke
(validation on) must stay 0-VUID with LODs active across static / skinned / cloth meshes and all three
LOD modes.

---

## Known limits & future directions

The discrete → VIPM → VDPM ladder is complete; VDPM matches the discrete mesh's silhouette and shading
at a fraction of the triangles. The remaining residuals and follow-ons, in rough priority:

- **Metric fidelity (active arc).** The four-channel metric is correct in shape but not yet a fully
  reliable perceptual bound. Progress:
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
  The win is real but narrower: the cone halved `refineForView` and is GPU-shaped, but `repairCoverage`
  (now ~50% of the cycle) and `repairFoldovers` stay. (Bounding-cone caveats per Hoppe, *View-Dependent
  Refinement of Progressive Meshes*, SIGGRAPH 97, §4.)
- **Parked:** texel-density UV budget; a GPU worklist/fixpoint (or representation-level guarantee) for
  the repair sweeps, which the cone cannot subsume.
- 🔨 **GPU-driven active front (in progress).** `refineForView` + the repairs are CPU today. The whole
  per-frame front lifecycle (score → refine/coarsen → both repairs → emit) has to move to the GPU as a
  unit — a partial move round-trips the shared front state and erases the win — with an indirect draw
  (the CPU no longer knows the index count). The CPU `vdpm` stays the headless-tested oracle + fallback
  (the `cloth` CPU-ref ↔ `render/` GPU-impl pattern). **Stage 0 (`graphics/vdpm_parallel`, in progress —
  scheduling core complete):** a GPU-shaped CPU model — the refine-dependency DAG with topological
  **ranks** (CSR by-rank layout), and
  `ParallelFront`, which reproduces the recursive front by **rank-ordered data-parallel passes**
  (requirement-closure → ascending-rank refine → descending-rank coarsen) instead of recursion, proven
  byte-for-byte against the oracle. This is the arc's stop/go gate: it proves the parallel scheduling
  and gathers the dispatch-depth evidence (curved meshes ~30 ranks; flat grids scale linearly) before
  any GLSL. Still ahead: the parallel repairs (a monotone snapshot fixpoint), deterministic
  seam-preserving emit, the GPU port + harness, and the indirect-draw plumbing.
- **7 forest skips.** `buildVertexForest` skips collapses whose edge diverged from its adjacency
  replay (7 of ~6800 on the helmet); past the first skip the forest is slightly unfaithful. The repairs
  cover the visible symptoms; truncating the stream at the first skip would be the clean structural fix.
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
| `include/fire_engine/graphics/vdpm.hpp`, `src/graphics/vdpm.cpp` | VDPM: `VertexForest`/`buildVertexForest`, `ActiveFront` (`refineForView`, `repairFoldovers`, `repairCoverage`, `emitActiveIndices`) |
| `src/graphics/mesh_simplifier.cpp` | The QEM engine (`QemRun`): R⁵ quadric, welding, wedge emit, chart veto, the four VDPM deviation channels, the two dials |
| `src/graphics/geometry.cpp` | `Geometry::load()` builds `lods_` + VIPM morph buffers + stores the VDPM collapse stream from one progressive artifact |
| `src/graphics/object.cpp` | Per-draw `selectLod` (forward + shadow), Continuous `selectVipm` uniforms, and per-instance VDPM `refineForView`/`repairCoverage` → dynamic index buffer |
| `graphics/frame_info.hpp`, `render/render_tunables.hpp`, `render/renderer.cpp` | `lodEnabled`/`lodMode`/`lodPixelErrorBudget` plumbing + jitter-free `currentViewProj` + triangles-drawn stat |
| `graphics/draw_command.hpp`, `render/ubo.hpp`, `shaders/shader.vert`, `shaders/shader.frag`, `render/debug_overlay.cpp` | VIPM morph binding/uniforms + per-draw `lodLevel` → push constant → LOD-tint debug view + overlay panel (3-mode selector) |
| `tests/graphics/test_mesh_simplifier.cpp`, `tests/graphics/test_vipm.cpp`, `tests/graphics/test_vdpm.cpp` | Headless correctness tests |
