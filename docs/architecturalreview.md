# Fire Engine — Architectural Review

*Reviewed 2026-07-25, branch `architectural-review` (off `main` @ `b9b5136`).*

Scope: the four requested focus areas — (1) the overall rendering approach, (2) shadow
generation / anti-aliasing and their use by the graphics subsystem, (3) physics-system and
solver correctness, (4) the mesh simplifier and the view-dependent progressive-mesh (VDPM)
system — plus any critical or important issues found along the way. Findings already recorded
in [`roadmap.md`](roadmap.md) / [`codereview.md`](codereview.md) / [`lod.md`](lod.md) are not
re-litigated; where a finding overlaps a documented "known residual" it is cross-referenced
rather than re-opened.

Severity legend: **[A]** correctness / latent-bug risk · **[B]** architectural inefficiency
worth fixing · **[C]** minor / hygiene / hardening.

---

## Executive summary

The architecture is in genuinely good shape. The layering discipline (`graphics/` Vulkan-free,
opaque handles, the `RenderableScene` seam, the three-layer GPU resource model) is real and
enforced, not aspirational. The physics solver is a faithful, well-reasoned TGS soft-step in
the Box2D-v3 mould with documented rationale at every non-obvious decision. The simplifier /
VDPM tree is the strongest subsystem in the codebase: a single canonical collapse stream feeds
discrete/VIPM/VDPM, conservatism is engineered (outward rounding, monotone accumulation,
inflationary repair bounds) rather than asserted, and the CPU/GPU scoring shares one authority.

The most consequential findings are all in the **renderer's fixed per-frame cost structure**:

1. **[B] Shadow rendering has a large constant per-frame floor** — ≥16 depth passes every
   frame regardless of scene content, including a fully redundant second 4-cascade CSM when no
   skinned mesh exists (§2.1, §2.2).
2. **[B] Skinning/morphing is re-executed in every pass's vertex shader** — up to ~11× per
   skinned vertex per frame. A compute pre-skinning pass (the `SoftBodySystem` already proves
   the pattern in-engine) would collapse this (§1.3).
3. **[A] The TAA history index formula silently assumes `kMaxFramesInFlight == 2`** — correct
   today, but it breaks without a diagnostic if the constant ever changes (§2.4).
4. **[C]** A handful of smaller items: duplicated sleep-threshold magic number, per-step heap
   allocation in the physics hot path, unbatched barriers, double LightUBO upload per frame.

No critical correctness bug was found in the physics solvers or the LOD/VDPM system.

---

## 1. Rendering approach (`render/`, `Renderer::drawFrame`)

### Assessment

The frame is a modern forward+ pipeline: depth prepass → SSAO/contact shadows → forward HDR
(+velocity MRT) → transmission capture → TAA resolve → particles → bloom → tonemap → overlay.
Vulkan 1.4 dynamic rendering + synchronization2 throughout, no render-pass/framebuffer
objects, push descriptors for per-object set 0, bindless materials on set 2, a per-frame
camera UBO written once (the CR-ubo-split work paid off), timeline-semaphore frame pacing.
This is a coherent, current design and the pass ordering is correct — in particular the
VDPM compute → shadow-pass overlap with the consumer barrier delayed to just before the depth
prepass (`renderer.cpp:1192`) is a nice latency-hiding decision, correctly reasoned (shadows
never read VDPM output).

The draw-command model (Vulkan-free `DrawCommand` built in `graphics/`, resolved by the
renderer) keeps the seam clean, and the two-stage culling (coarse BVH pre-cull as a superset
union, precise per-bucket cull) is sound: conservative in the right direction at both stages.

### Findings

**1.1 [B] Opaque draws are unsorted.** *(→ §6 item 10 (open))* `buildDrawBuckets` (`renderer.cpp:562`) depth-sorts
`blend` and `transmissive` but leaves `opaque` in scene-traversal order. The depth prepass
makes overdraw a non-issue for *shading*, but the prepass itself pays full overdraw in
traversal order, and the forward pass still binds state per draw in arbitrary order. A cheap
front-to-back sort of `opaque` (you already compute `sortDepth`) would help the prepass on
dense scenes; a secondary key on pipeline/material would reduce state churn. Low effort,
modest win; becomes more relevant as scenes grow.

**1.2 [C] Per-barrier `pipelineBarrier2` calls.** *(→ §6 item 6 (open))* Every `forwardImageBarrier` /
`imageLayerBarrier` / SSAO / TAA barrier issues its own `cmd.pipelineBarrier2` with a single
`ImageMemoryBarrier2`. The shadow pass alone issues 2 barriers × (8 CSM layers + 8 self-shadow
layers + spot + 6×point faces) as separate calls. On MoltenVK each barrier call translates to
Metal encoder boundary work, so batching adjacent barriers into one `DependencyInfo` (e.g. the
three-image transition at `beginForwardRendering`, or per-shadow-map-group) is a real, cheap
CPU-side win. The helpers could accept a span.

**1.3 [B] Deformation runs in every pass's vertex shader.** *(→ §6 item 5 (open))* A skinned+morphed mesh runs full
skinning/morph in: 4 CSM cascades, 2 self-shadow passes, the depth prepass, and the forward
pass — up to 8 vertex-shader executions per frame, plus the `PrevSkin` path in forward. The
engine already contains the better pattern: `SoftBodySystem` solves cloth *once* into a
storage vertex buffer that the shadow and forward passes then consume as plain vertices. A
compute pre-skinning pass writing post-deformation positions (+ normals/tangents for forward)
into a per-object storage vertex buffer would:
- make every downstream pass a rigid draw (shadow vertex shader shrinks to matrix × position),
- give the deformable path real bounds → coarse-cullable (today skinned nodes are always
  drawn because their bind-pose AABB under-covers — `onboarding.md` § Frustum Culling),
- unify the "deformed geometry" story with cloth instead of having two mechanisms.
This is the single highest-leverage rendering refactor available. It is also a prerequisite
worth weighing against the GPU-driven direction already in the roadmap's "Larger arcs".

**1.4 [C] `view_` uses a fixed world-up.** *(→ §6 item 11, ✅ landed)* `drawFrame` builds
`view_ = Mat4::lookAt(cameraPosition, cameraTarget, {0,1,0})` (`renderer.cpp:1001`) while the
skybox/shadow/self-shadow paths go through `makeViewBasis`/`stableUpForForward`. Camera pitch
is clamped (`camera.cpp` `clampPitch`), so this is latent, not live — but any future camera
type (orbit, cinematic spline) that reaches ±90° pitch will produce a degenerate view matrix
here first. Routing the main view through the same `view_basis.hpp` helper closes the trap
and honours the documented convention ("use it for new view … code").

**1.5 [C] `LightUBO` is uploaded twice per frame.** *(→ §6 item 9, ✅ landed)* `updateLightData` writes the full UBO
(`renderer.cpp:346`), then `assignSelfShadowSlots` rewrites the whole struct again after
filling the self-shadow matrices (`renderer.cpp:549`). `LightUBO` carries 16 lights + all
cascade/spot/self matrices, so this is two multi-KB copies where one would do — defer the
single `writeMapped` until after slot assignment (both run inside `collectDrawCommands`'s
frame phase, so ordering is already safe).

**1.6 [C] `vdpmDrawCounts_` accumulation is O(fronts²)** (`renderer.cpp:1058-1084`, linear
`find` per draw). Front counts are small today; fine — just don't let it grow unnoticed if
instance counts rise. Same pattern in `findSelfShadowViewProj` (linear scan per slot ×
slots) — all micro, listed for completeness. *(→ §6 item 18 — open, watch-item)*

---

## 2. Shadows and anti-aliasing (focus 2)

### Assessment

The shadow *quality* engineering is above the bar for an experimental engine: bounding-sphere
cascade fits with texel snapping (stable under camera motion), log-uniform practical splits,
raster depth bias + shader-side min/slope bias split sanely, per-cascade/per-face frustum
filtering of casters, and the dual-depth skinned self-shadow scheme is a genuinely principled
solution to light-side acne (first pass captures nearest surface both-faces, second pass
front-culled so only the *next* occluder rasterises). The 16+1-tap PCF disk in `shader.frag`
is expensive but appropriate for the single primary directional.

TAA is likewise correct where it is easy to be wrong: the velocity buffer is rendered
jitter-free, VDPM repair is fed the jitter-free view-projection (documented invariant), the
`PrevSkin` UBO gives exact deformation velocity, and particles deliberately render un-jittered
after the resolve so they stay out of history.

### Findings

**2.1 [B] The shadow pass has a fixed per-frame floor independent of content.** *(→ §6 item 2 ✅ landed + §6 item 4 (open))*
`Shadows::recordPass` (`shadows.cpp:145`) unconditionally records, every frame:
- 4 cascades × 2 maps (CSM + world-only CSM) at 2048² = 8 depth passes,
- 4 self-shadow slots × 2 passes at 1024² = 8 more passes,
each with clear + 2 layout barriers — **16 begin/endRendering brackets minimum**, even for a
static scene with zero skinned meshes and an unmoved light. Two independent fixes:

1. *Skip empty iterations.* A self-shadow slot with no draws (`selfShadowDraws` filtered by
   slot is empty — trivially knowable up front from `activeSelfShadowSlots`) still pays
   clear+barriers ×2 at 1024². Skipping needs care only for staleness: the maps rest cleared
   to 1.0 (= no occlusion) and unused slots carry identity matrices, so the safe scheme is
   "clear a slot once when it becomes unused, then skip it" (a per-slot dirty bit), not
   "always re-clear".
2. *Cache static cascades.* When the light direction, cascade fit (camera cell after texel
   snap), and the caster set are unchanged, the CSM content is identical to last frame's.
   Even a coarse validity check (light dir + snapped cascade origin + a caster-set epoch
   counter) would let a static scene skip all 8 CSM passes. Far cascades can also update on a
   cadence (every N frames) with minimal visible cost — a standard CSM optimisation.

**2.2 [B] The world-only CSM is fully redundant when no skinned mesh exists.** *(→ §6 item 1, ✅ landed)*
`worldShadowMap` exists so skinned receivers can sample a CSM *without* their own geometry
(self-shadowing handled by the dual-depth maps). When the frame has zero skinned casters,
`worldOnlyShadowDraws == shadowDraws` and the engine renders the *same geometry twice* into
two 2048×2048×4 depth arrays. `Renderer::recordShadowPass` already knows whether any skinned
caster exists (it computed the self-shadow slots); pass that down and either skip the world
map (and have the shader sample the main CSM — same content) or alias the descriptor. This
halves directional shadow cost for every fully-rigid scene, which is most of the sample set.

**2.3 [B/C] No punctual-shadow change detection.** *(→ §6 item 15 (open))* Spot/point casters re-render all faces
every frame even when the light and the geometry within range are static. Point lights are
the worst case (6 × 1024² faces each). Same epoch/dirty-bit approach as 2.1 applies. (Point
face culling *is* already per-face frustum-filtered — good.)

**2.4 [A] TAA history index assumes exactly two frames in flight.** *(→ §6 item 3, ✅ landed)*
`taa.cpp:87`: `const uint32_t prev = (kMaxFramesInFlight - 1) - cur;` — this is only
"the other slot" for `kMaxFramesInFlight == 2`. At 3 it maps 1→1: the resolve would sample
its own uninitialised output with no validation error and produce subtle history corruption.
The history ping-pong is conceptually *previous frame*, not *frame-in-flight slot* — the
current code conflates the two and happens to be right at 2. Minimum fix: a
`static_assert(kMaxFramesInFlight == 2, "Taa history indexing assumes double-buffering")`
next to the formula; better: an explicit ping-pong index owned by `Taa`.

**2.5 [C] The resolve→blit round trip costs a full-res HDR copy per frame.** *(→ §6 item 16 (open, measure first))* The resolve
renders into `history[cur]`, then blits back into the offscreen HDR target so downstream
passes (particles/bloom/post) keep a stable input (`taa.cpp:140-171`). The blit is the price
of not re-pointing downstream descriptors. An alternative — treat `history[cur]` as *the*
scene target for the rest of the frame (particles render into it, bloom samples it) — removes
a full-res 16F copy at the cost of per-frame descriptor updates or double-buffered
descriptors for the three consumers. Worth measuring on MoltenVK before committing; the
current design is defensible, the cost just shouldn't be forgotten.

**2.6 [—] Shadow LOD is discrete while forward is VIPM/VDPM.** *(informational — no action item here; this IS arc 1, see shadowplans.md)* Already documented as a known
residual (`onboarding.md` § Mesh LOD); noted here only because it *is* an architectural
seam: shadow silhouettes can pop independently of the forward silhouette. No action beyond
what the roadmap already records.

---

## 3. Physics correctness (focus 3)

### Assessment

The solver stack (`contact_solver.cpp`, `joint_solver.cpp`, `physics_world.cpp:solveIsland`)
is a correct and well-motivated TGS soft-step implementation. Points verified specifically:

- **Substep structure** (`solveIsland`, `physics_world.cpp:1442`): prepare once at `h`,
  then per substep gravity → warm-start → biased solve (joints then contacts) → position
  integration → un-biased relax. This matches the Box2D-v3 "soft step" schedule, and the
  relax pass is what makes the soft bias dissipative rather than an energy pump — correctly
  implemented (`massScale=1, impulseScale=0, bias=0` in the relax branch of both solvers).
- **Contact rows**: normal-then-friction ordering with the friction cone clamped against the
  *fresh* normal impulse; the coupled 2×2 tangent effective mass with **disk** (not box)
  clamping is the correct treatment and the comment records why (torque mis-distribution at
  tipping contacts). The speculative-gap branch (`separation > 0 → bias = separation/h`,
  applied regardless of bias phase) is the standard anti-tunnelling formulation.
- **Warm-start policy**: normal impulse carried cross-frame by proximity match; friction
  deliberately *not* carried cross-frame (documented failure history — flipped settled
  bodies). This is an unusual but defensible, empirically-driven policy and it is applied
  consistently in both `prepare` and `refresh`.
- **Restitution** as a single end-of-step pass against the prepare-time approach velocity
  `relVelN0`, gated on threshold + engaged impulse — correct; avoids substep buzz.
- **Mid-step manifold refresh (P9.6)**: the rebuild preserves accumulated normal impulse,
  `relVelN0`, and `maxNormalImpulse` matched via the *tracked current* anchor (not the stale
  prepare point) — the subtle part, done right. World inverse inertias are rebuilt for all
  bodies, with the honest comment that unrefreshed rows become a preconditioner mismatch
  (convergence-rate, not fixed-point, error). Agreed.
- **COM handling**: solver integrates about the world COM with origin conversion on
  write-back and in `ownerPoseAt` — consistent in all three places it matters.
- **Determinism**: canonical pair sort before solve, island-local solve equivalence argument,
  insertion-order body indexing — coherent, and guarded by the golden-hash tests.
- **Joints**: full-Jacobian rows, per-substep anchor-error recomputation for point rows,
  swing-twist limit decomposition with w-canonicalisation — checked, no defect found.

**No correctness bug found.** The findings below are efficiency/hygiene.

### Findings

**3.1 [B] Per-step heap allocation in the hot path.** *(→ §6 item 7 (open))* Every `step()` allocates:
`contacts()`'s pair copy + result vector, `solverBodies`, `inputs`, `jointInputs`,
per-island `islandContacts`/`islandJoints` copies, `refresh`'s `newPoints`, and the
warm-start caches rehash (`next_` unordered_maps rebuilt per step). None of this affects
determinism; all of it is avoidable with persistent scratch members (the renderer side
already follows the `*Scratch_` pattern rigorously — the physics side should mirror it).
At current body counts this is noise; it becomes the first flat cost when scenes grow, and
it is mechanical to fix.

**3.2 [C] Duplicated magic sleep threshold.** *(→ §6 item 8, ✅ landed)* The jointed-island angular sleep threshold
`0.15f` is hardcoded twice (`physics_world.cpp:1563` and `:1716`). Two call sites that must
agree, one literal apart. Promote to `kJointedAngularSleepThreshold` in
`physics_constants.hpp` next to `kAngularSleepThreshold`.

**3.3 [C] Mesh-triangle warm-start key mixing can collide.** *(→ §6 item 17 (open))*
`in.key ^= subKey * 0x9E3779B97F4A7C15ULL` (`physics_world.cpp:1635`) is a decent mix but
XOR over the pair key admits theoretical collisions across (pair, triangle) combinations.
Consequence is only a wrong warm-start seed (self-correcting within iterations), so this is
cosmetic — but a `boost::hash_combine`-style mix would make it principled.

**3.4 [C] Hinge axis-alignment rows are prepare-frozen.** *(→ §6 item 14, ✅ landed)* `addAxisRows` computes the
2-DOF axis-alignment error and Jacobian from step-start orientations; unlike point-anchor
rows (`anchorError == true`, re-evaluated per substep) the axis error is not recomputed as
the bodies rotate through the substeps. For fast-spinning hinged bodies this lags the
constraint by up to one step's rotation. The TGS literature accepts this; worth a comment in
`addAxisRows` so the asymmetry with `addPointRows` reads as chosen, not overlooked.

**3.5 [—] Already-documented limits** (not re-opened): brute-force spatial queries, mesh
contacts not mid-step-refreshed, link-vs-static-only articulation contacts, single-threaded
solve. All recorded in the roadmap with rationale; islands are the natural parallelism seam
if the need ever materialises. *(informational — no action item; the parked list lives in roadmap.md § Parked & revisit and collision.md)*

---

## 4. Mesh simplification & VDPM (focus 4)

### Assessment

This is the most carefully engineered subsystem in the codebase, and the review confirms the
big invariants hold:

- **One collapse stream, three consumers.** The R⁵ QEM run records the stream once;
  discrete cuts, VIPM morph targets, and the VDPM forest are all derived from it, with the
  weld (`mesh_topology::weldByPosition`) shared so canonical ids cannot drift. The
  `vl`/`vr` apex recording at collapse time (rather than replay re-derivation) eliminates
  the historical desync class by construction.
- **Conservatism is engineered, not asserted.** The normal-cone union re-derives the
  half-angle against the *stored float* axis and outward-rounds one ULP; `largestSingularValue`
  uses a Gershgorin bound computed in double with outward rounding and documents why power
  iteration is unusable; deviation channels accumulate monotonically (sum for geometry,
  max for UV — with the correct perceptual argument — π-capped sums for angular). Each of
  these is the right call and the comments prove the failure mode was understood.
- **The `run()` early-`break` on the error ceiling is sound**: version-valid heap entries
  were costed against the *current* quadrics (versions bump on collapse), so the popped cost
  equals the recomputed error and min-heap monotonicity justifies the break. The chart-veto
  reversal correctly re-checks the ceiling and `continue`s (the earlier bug, fixed).
- **`wouldFlip` only inspects `removed`'s ring** — correct under subset placement, since
  `kept` does not move and its other faces are unchanged.
- **The repair fixed point** (foldover ⊕ coverage, refinement-only) with the inflationary
  sweep bound turned into a `logic_error` guard is exactly how to make a convergence
  argument enforceable.
- **Forest validation** checks both directions of the split↔child bijection — the reverse
  check (alias detection) is the one implementations usually forget.

### Findings

**4.1 [B, largely retired by the GPU front] CPU-front per-frame cost is O(mesh), not
O(front).** *(informational — retired by the GPU-driven front; no action item)* Per instance per frame, the CPU path runs: a full `O(splits)` score scan
(`refineForView`), repair sweeps at `O(finestFaces)` per sweep until fixpoint, and an emit at
`O(original indices)` with per-corner `nearestWedge`. None of this scales with the *active*
front size. This is exactly why the GPU-driven front exists and is now the default, so no
action for the GPU path — but the CPU path remains the fallback for unsupported devices and
per-mesh dispatch-limit fallbacks. If the CPU fallback is ever expected to carry real scenes,
the standard Hoppe-style fix is queue-driven refinement (only re-score splits whose
score-relevant inputs changed / a priority queue over the front neighbourhood) rather than
the full scan. Recorded here as the shape of the fix; not worth doing speculatively.

**4.2 [C] The chart veto uses static chart sets.** *(→ §6 item 13, ✅ landed)* `canonicalCharts_` is computed once from
the original mesh; `crossesChart` tests `removed ⊆ kept` against those initial sets, and
`kept` never inherits anything on collapse. This is *correct* (wedges never move, so a
position's native wedge set — and hence its chart set — is genuinely immutable), but the
reasoning is subtle enough that a comment on `canonicalCharts_` saying "immutable by design:
wedges are attached to positions, collapses never move them" would save the next reader the
same derivation this review needed.

**4.3 [C] Simplifier build-time allocation churn.** *(→ §6 item 19 (open, conditional))* `collapse()` allocates `oneRing`,
`removedWedgeUv`, and a `neighbours` map per collapse. The code already notes the
per-collapse allocation is "in the noise next to the neighbour re-cost" — accepted; if load
times ever matter, these three are the first hoist (member scratch, `clear()` per collapse).

**4.4 [—] Metric-fidelity residuals** (angular channels vs a true Hausdorff bound,
texel-density UV budget) are already exhaustively recorded in the roadmap's
`cr-vdpm-metric-instrumentation` arc and `lod.md` § Known limits. Nothing new found beyond
that record; the instrumentation (per-channel trigger counts + max ratios in the overlay) is
the right tooling to keep them observable. *(informational — no action item; parked in lod.md § Known limits)*

---

## 5. Cross-cutting observations

**5.1 [B] The renderer records everything, every frame.** *(informational — the general case; parked in roadmap.md § Parked & revisit)* There is no notion of "nothing
changed" anywhere in the frame graph: all shadow passes, full command re-record, full
`LightUBO`/`CameraUBO` writes, SSAO, bloom chain — every frame, even with a static camera
over a static scene. For an experimental engine this is a reasonable simplicity choice, but
the shadow findings (2.1–2.3) show where it bites first. A lightweight *epoch* system
(scene-graph transform epoch, light epoch, caster-set epoch) would let individual passes
make skip decisions without a full frame-graph rewrite, and is the incremental path if
render-cost ever becomes the constraint.

**5.2 [C] Barrier/pass micro-structure vs MoltenVK.** *(informational — a meta-observation over 1.2 / 2.1 / 2.5, not its own item)* Several findings (1.2, 2.1, 2.5)
compound specifically on MoltenVK, where encoder splits and barrier translation carry more
CPU cost than on native Vulkan. Given macOS is the primary platform, batching barriers and
reducing begin/endRendering bracket count is worth more here than generic Vulkan lore
suggests. The VDPM perf arc already discovered this empirically (command-translation cost
rivalling shader time); the same lesson applies to the shadow pass structure.

**5.3 [C] `Frustum::fromViewProj` is called ~29× per frame** in `collectDrawCommands`
(camera + all 28 shadow matrix slots, including identity placeholders — `renderer.cpp:840`)
plus per-cascade/per-face again inside `Shadows::recordPass`. Plane extraction is cheap, but
extracting frustums for *identity* matrices (inactive slots) to get "harmless degenerate
frustums" does real work to add nothing — skipping slots not currently active (the counts
are known: `activeSpotCasters_`, `activePointCasters_`, self-shadow slot count) is free. *(→ §6 item 12, ✅ landed)*

**5.4 [C] Two review docs now exist.** *(✅ resolved — see the note below the finding)* `docs/codereview.md` and this file overlap in genre.
Recommend making the relationship explicit so items don't fork, and folding the surviving
items of both into `roadmap.md`.

✅ *Resolved in [`roadmap.md`](roadmap.md) § How the review docs relate.* The split is:
`codereview.md` is the **rolling tiered static review** that walks the
[`review-order.md`](review-order.md) tiers (Tier 0 math, 18 Jul 2026; Tier 1 handles/limits/
tunables, 19 Jul 2026; further tiers expected) — all of its findings are open, tracked as the
roadmap's arc 3. This file is the **dated architectural review**, and its §6 table below stays
the status of record for its own findings. `roadmap.md` indexes the open items of both and
carries no landed work. *(An earlier draft of this finding described `codereview.md` as "the
post-VDPM static review" — that was a different, now-cleared review whose file no longer
exists; the tiered review inherited the filename.)*

---

## 6. Prioritised recommendations

| # | Finding | Severity | Effort | Where |
|---|---------|----------|--------|-------|
| 1 | ✅ Skip redundant world-CSM when no skinned casters *(branch `review-shadow-taa-fixes`)* | B | S | §2.2 |
| 2 | ✅ Skip empty self-shadow slots *(same branch; implementation is simpler than proposed — unassigned slots are provably never sampled, so no clear-once/dirty bit is needed, they are skipped outright)* | B | S | §2.1 |
| 3 | ✅ `static_assert(kMaxFramesInFlight == 2)` on the TAA history index *(same branch)* | A | XS | §2.4 |
| 4 | Static-scene CSM caching (light+fit+caster epoch) | B | M | §2.1 |
| 5 | Compute pre-skinning pass (unify with cloth pattern) | B | L | §1.3 |
| 6 | Batch image barriers into single `DependencyInfo`s | C | S | §1.2 |
| 7 | Physics per-step scratch persistence | B | S | §3.1 |
| 8 | ✅ Named constant for jointed sleep threshold *(branch `review-xs-cleanups`; `kJointedAngularSleepThreshold`, with the articulation constant now aliasing it as its comment claimed)* | C | XS | §3.2 |
| 9 | ✅ Single deferred `LightUBO` upload *(same branch; `updateLightData` no longer uploads — `assignSelfShadowSlots` is the sole per-frame write)* | C | XS | §1.5 |
| 10 | Front-to-back sort of opaque bucket | B | S | §1.1 |
| 11 | ✅ Route main view matrix through `view_basis.hpp` *(same branch; `stableUpForForward`, identical result off the poles)* | C | XS | §1.4 |
| 12 | ✅ Skip frustum extraction for inactive shadow slots *(same branch; coarse-cull pushes only active cascade/spot/point slots — also tightens the cull)* | C | XS | §5.3 |
| 13 | ✅ Comment: chart-set immutability rationale *(same branch)* | C | XS | §4.2 |
| 14 | ✅ Comment: hinge axis rows prepare-frozen by design *(same branch)* | C | XS | §3.4 |
| 15 | Punctual-shadow change detection (spot/point re-render every face every frame; point is 6×1024² each) — same epoch/dirty-bit mechanism as item 4 | B/C | M | §2.3 |
| 16 | `hash_combine`-style mix for the mesh-triangle warm-start key (XOR admits collisions across (pair, triangle); consequence is a wrong warm-start seed, self-correcting) | C | XS | §3.3 |
| 17 | Retire the TAA resolve→blit full-res 16F copy by treating `history[cur]` as the scene target — **measure on MoltenVK first**; costs per-frame or double-buffered descriptors for three consumers | C | M | §2.5 |
| 18 | `vdpmDrawCounts_` / `findSelfShadowViewProj` linear scans are O(fronts²) — a watch-item, not a defect at today's front counts | C | XS | §1.6 |
| 19 | Hoist the simplifier's per-collapse `oneRing` / `removedWedgeUv` / `neighbours` allocations to member scratch — **only if load times start to matter** | C | S | §4.3 |

Items 1–3 were the recommended first branch (`review-shadow-taa-fixes`, merged): small,
high-value, none moves the physics golden or any documented invariant. The six XS items
(3, 8, 9, 11, 12, 13, 14 — 3 landed with 1–2) followed on `review-xs-cleanups`. Item 5 is the
one genuinely architectural piece of work and should be weighed against the roadmap's existing
GPU-driven arc before starting.

**Items 15–19 were added later, by a coverage audit** (2026-07-26): the original table was a
*prioritised* list, not an exhaustive one, and five actionable findings in §1–§5 had no row.
With them added, **every finding in this document now resolves to either a table row or an
explicit "informational" tag** next to the finding itself — §2.6 (shadow LOD, which is arc 1 /
[`shadowplans.md`](shadowplans.md)), §3.5, §4.1, §4.4, §5.1, §5.2, and §5.4 (resolved). Note
15–19 are genuinely lower-value than 1–14: three are conditional or watch-items in the review's
own words. They are recorded so the arc can be scoped honestly, not because they are all worth
doing. The same list is mirrored in [`roadmap.md`](roadmap.md) § Arc 2, which is
self-contained — so this document can be retired once its content has been reviewed.
