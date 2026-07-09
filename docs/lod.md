# Mesh Level of Detail

How the mesh LOD system works, and the design decisions behind it, for an engineer who will
**maintain and improve it**. This is the physics track's [`collision.md`](collision.md) equivalent for the rendering
spine's LOD arc (roadmap "#3 — view-dependent progressive meshes").

The current system is **Phase 2: view-independent progressive mesh (VIPM)**. Phase 1's discrete LOD
index buffers still exist, but continuous mode now geomorphs vertices into the exact next LOD before
the index-buffer swap. The remaining future step is view-dependent progressive meshes — see
[Future Directions](#future-directions).

---

## What it does

At load time, each static mesh above a triangle threshold gets a small progressive artifact: an
ordered collapse stream, exact LOD cuts through that stream, progressively simpler index buffers,
and a per-original-vertex geomorph buffer. At draw time, each mesh picks the coarsest LOD whose
on-screen error stays within a pixel budget, so distant/small meshes submit far fewer triangles. In
continuous mode the current LOD's collapsing vertices slide onto the exact render-attribute wedges
drawn by the next LOD before the topology switches. On the DamagedHelmet (~15.5k triangles) the
levels are ~15452 → 7725 → 1931.

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
   position+UV re-shatters at seams and caps simplification (~44% on the helmet).

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

## Overlay & debug

`RenderTunables` (`lodEnabled`, `lodPixelErrorBudget`) drives a **"Mesh LOD"** overlay panel (toggle +
pixel error budget slider + a live "Triangles drawn" readout from `FrameStats::trianglesDrawn`). The
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

`selectLod` is header-only and directly unit-testable. Beyond the headless tests, the render smoke
(validation on) must stay 0-VUID with LODs active across static / skinned / cloth meshes.

---

## Known limits & future directions

Phase 2 removes the main forward-pass geometry/attribute pop for static meshes. Two residuals shape
what comes next:

- **Coarsest-level seam shift.** A position-welded seam vertex carries one representative UV, so at the
  *coarsest* level seams still shift slightly when they finally collapse. Eliminating this entirely
  needs full **per-wedge** attribute quadrics (each chart's quadric kept separate) — real machinery,
  diminishing returns. We stopped here deliberately.
- **Shadow LOD remains discrete.** Continuous morphing is currently applied to the forward/depth
  vertex shader path. Shadow draws still choose their biased discrete LOD, so a shadow silhouette can
  still step independently of the main-view morph.

The ladder, all on this one simplifier's recorded collapse stream:

- **Phase 2 — view-independent continuous (VIPM):** done for static forward/depth draws. It uses a
  per-vertex SSBO, not a rewritten dynamic vertex buffer, so the base vertex upload stays immutable.
- **Phase 3 — view-dependent (VDPM):** promote the linear stream to a vertex forest with dependencies +
  a per-region **active front** (silhouette / near-edge refinement), so different parts of one mesh sit
  at different detail. Optionally GPU-driven.

---

## File map

| File | Role |
|---|---|
| `include/fire_engine/graphics/lod.hpp` | `GeometryLod`, ratios/thresholds, `selectLod` (header-only) |
| `include/fire_engine/graphics/mesh_simplifier.hpp` | `MeshSimplifier` interface, `QuadricSimplifier`, `MeshCollapse`, `SimplifiedMesh`, `ProgressiveMesh` |
| `include/fire_engine/graphics/vipm.hpp`, `src/graphics/vipm.cpp` | VIPM morph payload, exact-cut morph-data build, `selectVipm` |
| `src/graphics/mesh_simplifier.cpp` | The QEM engine (`QemRun`): R⁵ quadric, welding, wedge emit, the two dials |
| `src/graphics/geometry.cpp` | `Geometry::load()` builds `lods_` + VIPM morph buffers from one progressive artifact |
| `src/graphics/object.cpp` | Per-draw `selectLod` (forward + shadow) and Continuous-mode `selectVipm` uniforms |
| `graphics/frame_info.hpp`, `render/render_tunables.hpp`, `render/renderer.cpp` | `lodEnabled`/`lodPixelErrorBudget` plumbing + triangles-drawn stat |
| `graphics/draw_command.hpp`, `render/ubo.hpp`, `shaders/shader.vert`, `shaders/shader.frag`, `render/debug_overlay.cpp` | VIPM morph binding/uniforms + per-draw `lodLevel` → push constant → LOD-tint debug view + overlay panel |
| `tests/graphics/test_mesh_simplifier.cpp`, `tests/graphics/test_vipm.cpp` | Headless correctness tests |
