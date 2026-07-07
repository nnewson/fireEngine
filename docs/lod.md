# Mesh Level of Detail

How the mesh LOD system works, and the design decisions behind it, for an engineer who will
**maintain and improve it**. This is the physics track's [`collision.md`](collision.md) equivalent for the rendering
spine's LOD arc (roadmap "#3 — view-dependent progressive meshes").

The current system is **Phase 1: discrete LOD**. It is deliberately built so that the harder later
phases (continuous, then view-dependent) reuse its core — see [Future Directions](#future-directions).

---

## What it does

At load time, each static mesh above a triangle threshold gets a small set of **discrete LODs** —
progressively simpler index buffers. At draw time, each mesh picks the coarsest LOD whose on-screen
error stays within a pixel budget, so distant/small meshes submit far fewer triangles. On the
DamagedHelmet (~15.5k triangles) the levels are ~15452 → 7725 → 1931.

Two properties are load-bearing and shape everything below:

- **All LODs index the same, unchanged vertex buffer.** A LOD is *only* an alternate index buffer.
  No vertex data is moved or duplicated, so a mesh with N levels costs one vertex upload + N index
  uploads, and switching LOD is just swapping `indexBuffer`/`indexCount` on the draw.
- **Textures survive simplification.** UV seams are preserved through the near levels and only shift
  at the coarsest — not smeared — because the simplifier is UV-aware and the emit is wedge-aware.

---

## Runtime data flow

```
Geometry::load()                         [once, load time, CPU]
  QuadricSimplifier::simplify(ratio)  →  index set per LOD (into the base vertex buffer)
  lods_ = { LOD0 = full, LOD1, LOD2, … } each { indexBuffer, indexCount, error }

Object::buildDrawCommands()              [per draw, per frame]
  selectLod(geometry.lods(), distance, proj[1,1], viewportHeight, pixelError)
  → cmd.indexBuffer / cmd.indexCount / cmd.lodLevel
  (shadow draws use the same, biased coarser)
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
  stream. Phase 1 doesn't consume it, but it is **recorded from day one** as the raw material for the
  continuous phases.

### Algorithm, and the decisions inside it

1. **Position welding for connectivity.** glTF splits a vertex at every attribute seam (a corner with
   two UVs / normals becomes two vertices). Left split, the index topology is shattered into
   boundary-locked islands that barely simplify (the DamagedHelmet went 15452 → 15361 — 0.6%). So
   collapse connectivity is built on **position-welded** vertices: all vertices at one exact position
   collapse to a canonical. *Decision:* weld by position only (not position+UV) — welding by
   position+UV re-shatters at seams and caps simplification (~44% on the helmet).

2. **Wedge-preserving emit.** Welding by position throws away the seam UVs (both sides collapse to one
   canonical UV). To get them back, the emit keeps `origTris_` (original per-corner vertices) and
   `canonicalWedges_` (the native wedges at each position). Each output corner emits the wedge at its
   surviving position whose UV is **nearest** the corner's own UV (`nearestWedge`). On a preserved
   seam (a position carrying wedges from two charts) each side keeps its chart's UV. *Decision:* this
   wedge machinery is needed regardless of whether UV is in the error metric — it's what makes the
   output UVs correct.

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
`kLodRatios {0.5, 0.125}`, `kMinLodTriangles 512`, `kLodPixelError 2`, `kShadowLodBias 3`.

### Why from scratch (not meshoptimizer)

`meshopt_simplify` returns only a final index buffer — it does **not** expose the ordered collapse
sequence, and successive calls at different ratios aren't nested. The continuous phases (VIPM/VDPM)
*are* that collapse sequence, so we need a simplifier we can read it out of. Building our own also
keeps it dependency-free and in-spirit with the from-scratch physics solver. meshoptimizer remains a
viable backend for discrete-only work, or a validation oracle in tests, behind the `MeshSimplifier`
interface.

---

## Selection (`selectLod`)

`selectLod(lods, distance, projScaleY, viewportHeight, pixelError)` picks the **coarsest** level whose
world error projects to ≤ `pixelError` screen pixels. A world deviation `e` at view distance `d`
projects to `e · projScaleY · viewportHeight / (2d)` pixels, where `projScaleY = proj[1][1]`
(= 1/tan(fovY/2)). Levels are ordered fine→coarse with non-decreasing error, so the first that
overflows the budget ends the search; LOD0 is the fallback when nothing coarser fits.

Shadow draws pass `kLodPixelError × kShadowLodBias` (coarser — silhouette detail matters less in a
shadow). Selection uses the camera's projection/viewport as an approximation for shadow passes too;
good enough because a mesh far from the camera has little shadow footprint.

---

## Overlay & debug

`RenderTunables` (`lodEnabled`, `lodPixelError`) drives a **"Mesh LOD"** overlay panel (toggle +
pixel-error slider + a live "Triangles drawn" readout from `FrameStats::trianglesDrawn`). The
**"LOD tint"** debug view (`DebugView::Lod`, view index 7) colours each mesh by its selected level —
green LOD0 / yellow LOD1 / red LOD2 / magenta 3+ — via the per-draw `lodLevel` threaded through
`ForwardPushConstants` into `shader.frag`. Toggling LOD off (all green, full mesh) is the A/B.

Confirm behaviour by backing the camera off or cranking the pixel-error slider: the tint should step
green → yellow → red and the triangle count should drop.

---

## Tests

`tests/graphics/test_mesh_simplifier.cpp` (`[MeshSimplifier]`), headless:
- **Flat grid → 2 triangles at ~0 error** — coplanar interior + straight borders collapse fully.
- **UV sphere** — hits the target ratio, bounded error, output is a strict vertex subset.
- **Welds coincident seam vertices** — a per-triangle-duplicated grid must collapse like a welded one.
- **Preserves per-corner UV across a seam** — a two-triangle UV seam keeps both sides' UVs.
- **Deterministic** — identical output across runs.
- **Collapse-sequence replay** — replaying the recorded stream reproduces `simplify(0.0)`.

`selectLod` is header-only and directly unit-testable. Beyond the headless tests, the render smoke
(validation on) must stay 0-VUID with LODs active across static / skinned / cloth meshes.

---

## Known limits & future directions

Phase 1 pushes discrete LOD to its practical ceiling. Two known residuals shape what comes next:

- **Discrete popping.** A LOD swap changes geometry (and UVs) in one frame. That's the nature of
  discrete LOD — **Phase 2 (VIPM)** dissolves it with geomorphing.
- **Coarsest-level seam shift.** A position-welded seam vertex carries one representative UV, so at the
  *coarsest* level seams still shift slightly when they finally collapse. Eliminating this entirely
  needs full **per-wedge** attribute quadrics (each chart's quadric kept separate) — real machinery,
  diminishing returns. We stopped here deliberately.

The ladder, all on this one simplifier's recorded collapse stream:

- **Phase 2 — view-independent continuous (VIPM):** build a vertex hierarchy from `collapseSequence()`,
  pick one global refinement level, and **geomorph** vertices in/out across the transition (kills the
  pop) via the per-frame dynamic vertex buffer (the same path the cloth solver writes).
- **Phase 3 — view-dependent (VDPM):** promote the linear stream to a vertex forest with dependencies +
  a per-region **active front** (silhouette / near-edge refinement), so different parts of one mesh sit
  at different detail. Optionally GPU-driven.

---

## File map

| File | Role |
|---|---|
| `include/fire_engine/graphics/lod.hpp` | `GeometryLod`, ratios/thresholds, `selectLod` (header-only) |
| `include/fire_engine/graphics/mesh_simplifier.hpp` | `MeshSimplifier` interface, `QuadricSimplifier`, `MeshCollapse`, `SimplifiedMesh` |
| `src/graphics/mesh_simplifier.cpp` | The QEM engine (`QemRun`): R⁵ quadric, welding, wedge emit, the two dials |
| `src/graphics/geometry.cpp` | `Geometry::load()` builds `lods_` |
| `src/graphics/object.cpp` | Per-draw `selectLod` (forward + shadow) |
| `graphics/frame_info.hpp`, `render/render_tunables.hpp`, `render/renderer.cpp` | `lodEnabled`/`lodPixelError` plumbing + triangles-drawn stat |
| `graphics/draw_command.hpp`, `render/ubo.hpp`, `shaders/shader.frag`, `render/debug_overlay.cpp` | Per-draw `lodLevel` → push constant → LOD-tint debug view + overlay panel |
| `tests/graphics/test_mesh_simplifier.cpp` | Headless correctness tests |
