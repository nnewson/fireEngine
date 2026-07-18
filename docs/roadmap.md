# Fire Engine — Roadmap

Open work only. Everything major is **complete** — the physics & collision track (P0–P9), the
rendering foundations (particles, cloth/XPBD, Vulkan 1.3/1.4 modernization, bindless materials, TAA,
SSAO, frustum culling, overlay), the 26-item staff-engineer code review (CR-01…26), and rendering
spine **#3 — progressive-mesh LOD** end to end (discrete → VIPM → VDPM). The design + rationale for
landed work lives in the authority docs, not here:

- **Physics / collision** → [`collision.md`](collision.md)
- **Mesh LOD (discrete / VIPM / VDPM)** → [`lod.md`](lod.md)
- **Everything else** (subsystems, invariants, reading order) → [`onboarding.md`](onboarding.md),
  [`review-order.md`](review-order.md), [`README.md`](../README.md)

Suggested order below — not binding.

---

## Immediate — post-VDPM review ([`codereview.md`](../codereview.md))

A refreshed static review of the VDPM/LOD tree, renderer resource paths, and C++23/Vulkan practice.
Handled as a unit, exactly as CR-01…26 was cleared before the spine. Two real correctness bugs, the
rest fidelity/robustness/cleanup; several items *subsume* the "Known limits" in [`lod.md`](lod.md).
One branch per item. (Findings map to `codereview.md` by **title** — its list restarts numbering per
severity tier, so titles are the stable reference, not a global number.)

**(A) VDPM hardening.**
- ✅ **Chart-veto reversal bypasses the error ceiling** *(branch `cr-vdpm-correctness`)* — when the
  veto flips the collapse direction, the reversed (larger) endpoint error is now re-checked against
  `kErrorCeilingFactor` (and `continue`, not `break` — the heap is ordered by each edge's *minimum*
  endpoint cost). Real bug, fixed. (No minimal regression test: the trigger needs a seam-reversal
  collapse exceeding the *generous* 40× ceiling, and its only public observable, `MeshCollapse::error`,
  is an RMS that normalises the boundary weight out — so the violation is masked in every black-box
  signal. Guarded by the existing determinism/replay tests.)
- ✅ **`QemRun::sequence()` `noexcept`+copy → `&&`-move** *(same branch)* — consuming rvalue accessor;
  `noexcept` is now honest (a vector move can't allocate).
- ✅ **Forest replay after an unreplayable collapse → fixed via the shared-topology util below**
  *(branch `cr08-shared-topology`)*. Root-caused: the "7 forest skips" are genuine **non-manifold
  welded edges** (position-welding fuses coincident chart pieces into >2-face edges), which the vl/vr
  vertex-split encoding can't represent; the old `buildVertexForest` re-derived that adjacency by an
  independent replay that then desynced and cascaded (19 skips). *Truncate* rejected (5.6× coarser
  floor); *veto in the simplifier* rejected (the two topology trackers diverge beyond the non-manifold
  edges, and it changes discrete/VIPM output). The fix landed: the simplifier now **records (vl, vr)
  per collapse** on the true canonical topology it coarsens (ground truth), and `buildVertexForest`
  transcribes that stream instead of re-deriving — faithful by construction. The few genuinely
  non-manifold edges carry `kNoCollapseApex` and leave `removed` a root (always active), isolated with
  no cascade and no coarsening loss (19 skips → 7 roots).
- ✅ **Coverage repair precision (partial)** *(branch `cr-vdpm-correctness`)* — the area gate is now
  **viewport-relative** (a pixel-area constant → NDC via the passed viewport, so it doesn't drift with
  resolution), and near-plane-crossing faces (some corners behind the camera) are now **conservatively
  refined** instead of silently skipped. The **multi-sample** corner-coverage idea was deliberately
  NOT taken: measured, sampling a fine triangle's corners against its OWN replacement flags ~5800
  mostly-false positives (the neighbour's replacement covers them), so it would massively over-refine;
  a correct version needs true point-location against the active mesh — deferred, and the centroid +
  degenerate handling already closes the visible holes. (Supersedes the `lod.md` coverage-precision
  note for the two parts done.)
- ✅ **Non-uniform-scale transforms** *(same branch)* — facing now uses the inverse-transpose normal
  matrix, and foldover winding is compared in **world space** (what the rasteriser culls on). Guarded
  by a non-uniform-scale case in the foldover test.
- ✅ **Cheap hygiene** *(same branch)* — `ActiveFront`'s `std::vector<bool>` → `uint8_t`; the
  `writeMapped` guard is now release-visible (clamps so it can't corrupt neighbouring GPU memory, and
  logs once instead of overflowing silently).
- ✅ **Shared mesh-topology / wedge util** *(branch `cr08-shared-topology`)* — position weld, wedge
  distance, nearest-wedge, canonical-wedge grouping were duplicated across simplifier / VIPM / VDPM /
  tests, exactly where the recent bugs clustered. Now one module `graphics/mesh_topology.{hpp,cpp}`
  (`weldByPosition`, `wedgeDistance`, `nearestWedge`, `canonicalWedges`) that all four consume, so the
  canonical-id contract can't drift between them. Step 2 recorded (vl, vr) per collapse in the
  simplifier (see the forest-replay item above), so `buildVertexForest` transcribes ground-truth
  adjacency instead of re-deriving it. De-risks all future LOD work.

**(B) Renderer hygiene — independent of VDPM.** *(all landed, branch `cr-renderer-hygiene`)*
- ✅ **Device suitability validates all requested features/limits** — `missingDeviceCapabilities`
  queries one `PhysicalDeviceFeatures2` chain (1.0/1.2/1.3 + portability-subset) plus the
  descriptor-indexing update-after-bind limits during selection, and rejects an unsuitable GPU with a
  named reason (logged) instead of failing an opaque call later. `createLogicalDevice` dropped its
  redundant per-feature re-checks (suitability is now authoritative) and the dead
  `descriptorBindingVariableDescriptorCount` request (the bindless array is fixed-size + partiallyBound,
  not an `eVariableDescriptorCount` binding).
- ✅ **Descriptor-pool lifecycle + `createMappedStorageBuffer` misnaming** — the per-group pools drop
  `eFreeDescriptorSet` (sets are never freed individually; the pool is destroyed as a unit at
  shutdown) and now own their sets as plain handles rather than per-set RAII. `createMappedStorageBuffer`
  (which built one shared buffer, duplicated its handle into every frame slot, and returned empty
  mapped spans) became `createSharedStorageBuffer` returning a single `BufferHandle`; particle-system
  and morph/VIPM-dummy call sites updated.
- ✅ **Static vertex/index buffers are now device-local** — new `createDeviceLocalBuffer` (device-local
  + transient staging buffer + one-time copy). `createVertexBuffer` / both `createIndexBuffer`
  overloads (so base mesh + every LOD cut) and the VIPM geomorph table (`createStaticStorageBuffer`)
  use it; per-frame dynamic buffers (UBOs, VDPM index sets, debug lines, cloth storage-vertex) stay
  host-visible/mapped. The batched-upload optimisation (join the image `uploadBatch_` instead of a
  per-buffer submit) is left as a follow-up under static-residency below.

---

## Could — opportunistic / supporting

- ✅ **VDPM per-frame CPU emission scratch + repair counters** *(branch `cr-vdpm-emit-scratch`; the
  CPU-only, no-GPU-rewrite subset of codereview.md "VDPM per-frame work is correct but expensive")*.
  Three pieces: **(a)** `emitActiveIndices` now fills a caller-owned reused buffer (`object.cpp` passes
  a per-binding scratch vector) — no per-frame heap allocation; a vector-returning overload is kept for
  tests. **(b)** `refineForView` memoised `facingOf(v)` per canonical vertex and `emit` precomputes
  `activeAncestor` once per frame (the front is settled there); both are pure per-frame functions, so
  behaviour is byte-identical to the inline computation. (The `facingOf` cache was later removed by the
  visibility-cone arc, which replaced the smooth-normal facing with the per-split normal cone.) **(c)**
  `ActiveFront` exposes per-frame repair counters (vertices each pass pulled back in, `active_==0`-guard
  dedup'd), plumbed `Object → SceneDrawContext → CullStats → FrameStats` into the overlay's LOD panel
  ("VDPM repairs (verts): foldover X, coverage Y", shown in View-dependent mode) so a repair-count
  regression is visible. Verified: build clean, fast suite (incl. `[vdpm]`) green, 0 VUID smoke. The
  full GPU active front stays the separate arc below.
- ✅ **Split the per-frame vs per-object forward UBO** *(branch `cr-ubo-split`)* — the old
  `UniformBufferObject` bundled per-frame camera data with per-object data and was re-uploaded per
  object per frame. Now split into **`CameraUBO`** (view / proj / cameraPos / view-projections) and a
  slim **`ObjectUBO`** (model / previousModel / hasSkin). `CameraUBO` is written **once per frame** by
  the Renderer (no more duplicating view/proj into every object). `ObjectUBO` is re-uploaded only when
  it changes — `Object` caches the last world/previousWorld/hasSkin per frame slot and skips a
  byte-identical rewrite, so a static object does no per-frame UBO write. **Design note:** camera went
  into the *push* set 0 (binding 29), **not** the global set 1 — the depth prepass reuses `shader.vert`
  but binds no globals, and set 0 is already pushed there, so set 0 is the only place both passes see
  it (a first cut using set 1 failed pipeline creation exactly here). Camera is still only *bound* per
  draw (a cheap push, not a re-upload). Touched `ubo.hpp` (+static_asserts), both forward shaders, the
  set-0 layout, `pushForwardObjectDescriptors`, `DrawCommand`/`FrameInfo`, `Object`, and the Renderer.
- ✅ **Route the active camera through the `RenderableScene` seam** *(branch `cr-camera-through-seam`)*
  — the camera was the one piece of scene-owned per-frame data that bypassed the seam (passed as two
  loose `drawFrame(… Vec3 cameraPosition, Vec3 cameraTarget …)` args the app extracted). Now
  `RenderableScene::activeCamera()` returns a `CameraView{position, target}` and `drawFrame` drops the
  args. `SceneGraph` owns the active camera as a **`Node*`** (`activeCamera(Node*)` setter, asserts the
  node carries a `Camera` component); it reads the node's **live** world pose, so a moved camera node
  (e.g. a future route system animating its `Transform`) moves the view for free. No authored camera →
  a fixed debug fallback pose ({2,2,2}→origin, **not** a scene node, warns once) so a missing camera is
  visible, not a crutch. Camera *types* (FlyCamera/FirstPersonCamera + per-type input) are deliberately
  out of scope — this is the seam they'll plug into. `[SceneGraph]` tests cover the setter/getter +
  fallback.
- ✅ **TAA skinned-deformation velocity** *(branch `cr-taa-skinned-velocity`)* — skinned meshes used
  to reproject on camera motion only (`prevWorldPos = worldPos` for `hasSkin`), so animated
  deformation ghosted. Now a **`PrevSkin`** UBO (forward set-0 binding 30) carries last frame's joint
  matrices, and `shader.vert` skins the bind-pose vertex with them to get exact per-vertex velocity.
  Unified with the rigid path via `prevTransform` (previous joints when skinned, `previousModel`
  otherwise). `Object` caches `previousJointMatrices_` (previous == current on frame one → zero
  velocity). Camera-only was the *v1*; this is the exact form. Verified: skinned (CesiumMan) + rigid
  render 0 VUID; the velocity debug view (=5) shows the deformation motion.
- ✅ **`Mat4::transformPoint` helper** *(branch `cr-mat4-transformpoint`)* — the free
  `transformPoint(const Mat4&, Vec3)` copy-pasted in `physics_world.cpp`, `physics_world_shapes.cpp`,
  `scene_culler.cpp` is now one affine `Mat4::transformPoint(Vec3)` method (drops the homogeneous w, no
  perspective divide — correct for the composed model/world matrices all three sites use; the
  `scene_culler` copy's defensive `/w` was dead for its affine input). All copies replaced; three
  `[Mat4TransformPoint]` tests added. Determinism golden unchanged (physics behaviour byte-identical).

---

## Maybe — cosmetic / on demand

- ✅ **Ragdoll per-joint hinge limits** *(branch `cr-ragdoll-hinge-limits`)* — knees/elbows are now
  authored as true 1-DOF **Revolute** hinges with an asymmetric angle range, not the uniform swing
  cone. Three parts: (1) **physics** — `ArticulationLinkDesc` gained `jointLowerLimit`/`jointUpperLimit`
  and `solveJointLimits` a Revolute branch enforcing them via the same velocity-level unilateral
  push-back as the cone-twist; `jointImpulseResponse` generalised from spherical-only to any DOF count
  (the **extensibility hook** for future joint types). (2) **authoring** — `extras.Ragdoll.Joints` maps
  a bone node name → `{Type:"Hinge", Axis, Min, Max}` (unlisted bones keep the uniform cone). (3)
  **build** — `makeArticulated` builds a Revolute for a hinge-authored bone. Purely additive/opt-in, so
  **the determinism golden is unchanged** (no re-baseline). Tests: `[Articulation]` limit,
  `[GltfNodeExtras]` parse ×2, `[Ragdoll]` build. (Post-settle arm-drift + settle-yaw were already
  fixed on `ragdoll-joint-settle` — see [`collision.md`](collision.md).)
  - **Joint debug view + authored ragdoll** *(same branch)* — a `DebugView::Joints` mode (overlay
    "View → Joints", or `--debug-joints`) **replaces the scene mesh** (keeping the collider wireframes
    for context) with a per-link RGB axis gizmo (the link's local frame — pick the hinge `Axis` off
    it), a **degree-of-freedom overlay** (bright hinge-axis line for a 1-DOF Revolute; the triad spans
    a Spherical), and an on-screen "index: bone-name (Type)" label, so a skeleton's joints — and their
    DOF — can be found without guessing.
  - **Base roll/pitch settle damper** *(same branch)* — with the knees now folding correctly, a
    collapsed ragdoll rocked left-right for a while: the base's **roll/pitch** was the one settled DOF
    with no dedicated damper (linear + yaw had one; full base-angular damping was left out because it
    destabilises a near-planar chain). Added `kBaseRockSettle*` — decays the residual non-yaw base
    spin, double-gated on *settling* + a slow angular so a violent impact is untouched. Articulation-
    only ⇒ **golden-neutral**; the `[slow]` settle/soak gate stays green.
  - **Hip-wiggle settle (landed-widened joint gate)** *(same branch)* — after the base-roll fix a
    residual *hip* wiggle remained (traced: `leg_joint_*_1` ringing at ~1–4 rad/s for ~1 s after
    landing, re-exciting the base rock). It sat in a band too fast for `kJointSettleSpeed` (0.5) yet
    far below the collapse (6–13 rad/s). Fix: once the base has **landed** (linear < `kBaseSettleSpeed`)
    the joint settle gate widens to `kJointSettleSpeedLanded` (4 rad/s), bleeding the wiggle while the
    speed gate still protects the fast airborne collapse (which can momentarily read "landed"). Cut
    the hip wiggle ~½ and killed the 1.5 s base-rock re-spike; collapse **shape** (gyration / head-foot
    chord gates) preserved, golden-neutral.
  - **Near-rest snap (straggler sleep)** *(same branch)* — after the wiggle fix a lone shoulder DOF
    still crept in at ~0.15 rad/s for ~1 s after the body stopped (traced: hand moved ~3 mm while the
    joint held ~terminal velocity — a straggler hovering at the 0.15 sleep threshold, resetting the
    dwell). Added `kArticulationRestSnap*`: once the whole articulation sits within a wider near-rest
    band (0.30) for a short dwell (0.25 s), zero the residual so it sleeps as a unit. Cut settle from
    ~4.8 s → ~2.9 s; collapse untouched (byte-identical early frames), shape gates green, golden-
    neutral. Snap magnitude is tiny (fires ~0.15 rad/s) so it shouldn't pop — pending visual confirm. Labels project through the ImGui foreground
    draw list (`DebugOverlay::drawWorldLabels`) using `DisplaySize` (retina-correct) and this frame's
    finalised `viewProj` (no lag). With this, `CesiumManRagdoll.gltf` is now authored with real 1-DOF
    knees (`leg_joint_L/R_3`), ankles (`leg_joint_L/R_5`), and elbows (`Skeleton_arm_joint_R__3_` /
    `Skeleton_arm_joint_L__2_` — note the L/R numbering asymmetry the labels expose).

---

## Parked with data — revisit only on a concrete need

- **P9.6 Stage 2** — per-substep re-detection for >20 rad/s spinners resting on floors (boundary
  quantified + gated by `Demos.RotationalTunnellingBoundedTo20RadPerSec`).
- **Mesh-contact mid-step refresh.**
- **Link-vs-dynamic-rigid articulation contacts** (link colliders are link-vs-static only today).
- **Joint split-position pass** (P9 item 4).

---

## Larger arcs — GPU-driven direction (optional; opened up by #3)

Not new features — the maturation of things already noted. `codereview.md`'s "Larger Rewrite
Candidates" section folds in here:

- 🔨 **VDPM metric fidelity** *(branch `cr-vdpm-metric-instrumentation`; in progress)* — the four-channel
  refine metric is correct in shape but not a reliable perceptual bound: the angular (normal/tangent)
  channels aren't scale-invariant (angular error projected as a world length), and the shading channels
  can silently read zero when a collapse's removed vertex projects outside every surviving face (no
  covering-face fallback, unlike UV) — the likely cause of close-range interior faceting. Sequenced:
  **(1) instrumentation + invariant tests** *(done)* — `ActiveFront::channelStats()` per-channel refine
  attribution + overlay "VDPM splits" line + a `[!shouldfail]` scale-invariance test.
  **(2) shading correspondence decoupled** *(done)* — normal/tangent now measure against the closest
  point on the nearest surviving triangle (`closestPointBary`), not only a *containing* face, so
  endpoint collapses stop silently recording zero shading error (the holes).
  **(3) geometry vs the nearest actual triangle** *(done)* — the geometry channel measures point-to-plane
  against the nearest surviving *triangle*, not the `min` over every one-ring *infinite plane* (an
  unrelated coincident plane no longer quiets a curved patch).
  **(4) support bounds + scale-invariant angular projection** *(done)* — each collapse records a support
  radius (bounding sphere); the angular channels project it as a screen extent × the chord `2·sin(θ/2)`
  from the parent near-sphere depth, with object-space radii bounded into world space by the world
  matrix's largest singular value (so instanced non-unit scale refines correctly), and the angular
  radii capped at π. Scale-invariance test now passes on the production instance path. Instrumentation
  refined to per-channel *triggers* + max score/budget ratios (a zero count with a near-1 ratio = a
  hair under budget). Review follow-ups also done: the collapse measurement is now a unit-testable
  `detail::measureCollapseDeviation` (the no-containing-face regression feeds it a hand-built one-ring
  where `removed` is provably outside every face and asserts non-zero shading — not a `(parent,vl,vr)`
  proxy); the two closest-point helpers merged into one `closestOnTriangle` (barycentric + squared
  distance) with a conservative MAX over equal-distance ties; and a normal-channel test at the
  production `kVdpmNormalScale`.
  **(5) full TBN tangent metric** *(branch `cr-vdpm-tbn-tangent`)* — the tangent channel compared raw
  tangent xyz, but the shader samples a normal map in the per-vertex TBN frame (T Gram-Schmidt'd
  against N, B = cross(N,T)·handedness), so a handedness (`w`) flip read as zero. The simplifier now
  precomputes the frame axes and the channel measures the MAX of the T- and B-axis deviation (catches
  roll + handedness flip); unit-tested via `measureCollapseDeviation`.
  **(6) material-aware tolerances** *(branch `cr-vdpm-material-tolerances`)* — `vdpmChannelScales(material)`
  (pure `graphics/vdpm_material.*`) derives the per-channel refine scales at refine time, so a channel a
  material can't show is disabled: unlit → normal+tangent off; no normal/clearcoat map → tangent off (a
  mesh with tangents stops protecting a frame nothing samples); no textures → UV off; glossy → normal
  channel scaled up. Passed into `refineForView`'s existing scale args at the `object.cpp` call site — no
  simplifier/forest/front change; unit-tested per rule.
  **(7) persistent front + hysteresis** *(branch `cr-vdpm-persistent-front`)* — `refineForView` no
  longer `coarsenAll()`s each frame; the front persists. Score pass → refine pass (over budget) →
  coarsen pass (under `kVdpmCoarsenRatio × budget`); the dead band between stops splits popping in/out
  under small camera moves / TAA jitter (a static camera now yields an identical front every frame).
  Repairs still run each frame, and steady-state does *less* work than the old full rebuild. Tests:
  static-view stability + sub-band hold. **This completes the VDPM metric-fidelity arc.** Parked
  next-steps if it's ever revisited: a GPU-driven active front (also the only path to retiring the
  per-frame repair sweeps — the visibility cones below could NOT), texel-density UV budget.
  See [`lod.md`](lod.md) § Known limits (Metric fidelity). Render-path only ⇒ golden-neutral.
- ✅ **VDPM visibility cones** *(branch `cr-vdpm-visibility-cones`)* — a precomputed per-split
  **conservative normal cone** replaces the unreliable smooth-vertex-normal visibility proxy with a
  GPU-compatible, conservative face-**orientation** bound. Outcome, narrower than first hoped: it
  **improves back-face suppression and silhouette targeting but CANNOT replace screen-space coverage
  or topological foldover repair** — both remain. **(1) measure** *(done)* — the retained hidden
  benchmark `[.][RepairBench]` (`test_vdpm.cpp`) times the per-frame cycle on a dense (~24.6k-face)
  silhouette-heavy sphere. **(2) precompute the cone** *(done)* — `MeshCollapse`/`VertexSplit` carry a
  normal cone `{axis, cosHalfAngle}` accumulated bottom-up like `supportRadius` (`mergeCones`),
  conservative *by construction* (double math with re-normalised axes; the union axis is re-checked
  against both children so rounding can only widen; a one-ULP outward round — no magic margin); a cone
  wider than a hemisphere is the `cosHalfAngle <= 0` no-cull sentinel. A headless test replays the raw
  collapse stream and proves the cone bounds *every* finest-face normal in each subtree. **(3) wire it
  into `refineForView`** *(done)* — `detail::coneVisibility` does an **exact evaluation of the
  conservative bound**: the region is back-face-culled only if its whole cone provably faces away over
  the support-sphere view-direction spread (a one-sided proof of hiddenness — never a claim of
  visibility); a straddle of edge-on drives the silhouette boost. Done in **object space** (the facing
  sign is invariant under any linear transform, so it's exact under non-uniform scale and keeps the
  cone circular), trig-free (cosine sum identity, GPU-friendly), reflection-exact via the determinant
  sign, and singular-transform-safe. Gated by `rasterBackfaceCulling` (a double-sided or blended
  material culls nothing, so its refinement must NOT be suppressed). The cone HALVED `refineForView`
  (~1.45 → ~0.69 ms on the bench), which now makes `repairCoverage` the dominant ~50% of the ~2.0 ms
  cycle. **Coverage is a screen-space property, not orientation** — a force-refine-on-straddle
  experiment reduced but could not zero coverage repairs (10/19/85), so `repairCoverage` (and
  `repairFoldovers`, topological) STAY and step 4 is dropped. Retiring them eventually needs a GPU
  worklist/fixpoint or a representation-level guarantee — not this cone. (Bounding-cone caveats per
  Hoppe, *View-Dependent Refinement of Progressive Meshes*, SIGGRAPH 97, §4.)
- 🔨 **GPU-driven active front (in progress)** — drive the whole per-frame front lifecycle (score →
  refine/coarsen → repair → emit) + indirect draw on the GPU; the CPU `vdpm` stays the tested oracle +
  fallback. **Stage 0** (`graphics/vdpm_parallel`, merged) proved the parallel rank-ordered scheduling
  byte-for-byte against the oracle. Two oracle prerequisites landed on the way: the no-cull coverage gap
  (P1), and the **joint foldover+coverage repair** *(branch `cr-vdpm-joint-repair`)* — the two repairs
  were sequential (`refineForView`'s foldover fixpoint, then `repairCoverage`), so a coverage
  force-refine could re-fold a neighbour *after* the foldover fixpoint finished, leaving foldovers (a
  real shipped silhouette-hole bug). Now one public `repairFront` alternates the two private sweeps to a
  JOINT fixed point (≈2 sweeps in practice); `Object` can't misorder them; a named regression pins the
  six cases. The **parallel repairs** then landed on the GPU-shaped model *(branch
  `cr-vdpm-parallel-repairs`)*: the joint repair's per-face geometry was extracted into pure `detail::`
  classifiers (`isFoldover` / `classifyCoverageRepair`) the sequential sweeps now route through, and
  `ParallelFront::repairFront` reuses them as a **snapshot** detector — detect every violation against a
  settled front, close + apply targets in rank order, re-detect — an inflationary fixed point sharing
  the per-face policy + final invariants but not the sequential schedule (may reach a different valid
  front). Evidence: converges in 2 detection passes / 1 apply round, over-refines the sequential by
  1–3 tris (≤0.2%). Finally the **deterministic seam-preserving emit** *(branch `cr-vdpm-parallel-emit`)*:
  `ParallelFront::emitActiveIndices` is the GPU-shaped compaction — per-face survival flag → exclusive
  prefix sum → stable scatter (no atomic append, so triangle order is preserved) with `nearestWedge`
  restoration via a CSR wedge adjacency (`mesh_topology::canonicalWedgesCsr`) — proven **byte-identical**
  to the oracle's emit (indices, order, wedges). **Stage 0 is complete**: scheduling, repair
  convergence + overhead, and emit are all proven on CPU in CI, with the rank-depth and wedge-ABI
  evidence reported. **Stage A — indirect draw** *(branch `cr-vdpm-indirect-draw`)* is done too: a
  Vulkan-free `DrawIndexedIndirectCommand` mirror + `DrawCommand::indirectBuffer`/`indirectOffset`
  sentinel, a per-instance per-frame host-visible indirect-command buffer CPU-written from the emit
  count, and the three VDPM draw sites (forward / depth prepass / transmission) routed through
  `recordIndexedDraw` — shadows keep discrete LOD + direct draw. Mechanical, no behaviour change; it
  de-risks `drawIndexedIndirect` on MoltenVK (smoke-tested 0 VUID on DamagedHelmet + TransmissionTest
  via the new `--lod-mode view-dependent` launch flag) before any compute writes that buffer. The GPU
  port then began. **B1 foundation** *(branch `cr-vdpm-gpu-score`)*: a surface-free `Device`
  compute mode (no swapchain/present; graphics+compute queue), the per-instance scoring extracted
  into ONE Vulkan-free authority (`makeVdpmViewParams` / `scoreVdpmSplit`, which `refineForView` now
  consumes — with a camera-relative affine that reproduces world distance under any linear transform),
  a corrected conservative σ_max bound (the old power iteration under-estimated an orthogonal shear
  10×), and the std430 GPU ABI (`ubo.hpp` structs + pack helpers, fully offset-asserted). **B1 GPU
  scoring** *(branch `cr-vdpm-gpu-score-shader`)*: `shaders/vdpm_score.comp` (typed buffer_reference,
  reproducing `scoreVdpmSplit`), `VdpmGpuMesh` (shared static splits + positions) + `VdpmGpuFront`
  (per-instance output + per-frame mapped params), a compute-only `Resources` path, and a `[.][gpu]`
  readback harness that cross-checks the shader against the CPU authority (scores close, back-face
  decisions exact) on sphere/synthetic/singular/zero-split cases. **Stage B1 complete.** **B2 emit**
  *(prep branches `cr-vdpm-gpu-emit-shaders`/scan; passes on `cr-vdpm-gpu-emit-passes`)*: the GPU
  reproduces `ParallelFront::emitActiveIndices` **byte-identically** from a CPU-uploaded front. A
  reusable **exclusive-scan** primitive (`VdpmScan`, recursive Blelloch, 256-element blocks) and
  precomputed **wedge choices** (`buildWedgeChoices` — the CPU's `nearestWedge` decision per (original
  vertex, ancestor depth), so restoration is pure integer indexing and byte-identity is *structural*,
  not float-dependent) plus a collapsed **removal-parent** chain feed four passes: ancestor resolution
  (bounded removal-parent walk → active ancestor + depth, one atomic failure counter), per-face
  survival (three distinct, non-failed ancestors), the scan (→ stable per-face output slot + surviving
  total), and a stable scatter (restored-wedge corners in original face order — no atomic append) +
  a one-invocation finalize (index count = 3·survivors). `VdpmGpuMesh` gained the static emit data
  (indices/weld/removal-parent/wedge CSR, index-range-validated); `VdpmGpuFront::recordEmit` clears a
  single 3-uint counters buffer once and records the passes with compute→compute barriers. The
  `[.][gpu]` harness proves byte-identity (indices, order, wedges) across coarsest/partial/full/repaired
  fronts on sphere + per-corner-seamed grid, plus determinism, an empty mesh, an all-faces-collapse
  front (faceCount > 0), and a deepest-chain fixture pinning the ancestor bound's off-by-one.
  `VdpmGpuMesh::fitsComputeDispatchLimits` is the B5 backend selector's GPU-eligibility gate (static
  dispatch counts vs the 1-D group cap, checked BEFORE allocation so the selector can pick the CPU
  fallback); `build` enforces the same bound. **Stage B2 complete.** Still ahead: B3 refine/coarsen →
  B4 repairs → B5 end-to-end (compute writes the indirect buffer + a compute→indirect-read barrier;
  **record the wedge-choice memory for a real loaded asset (helmet) here** — it needs the Vulkan glTF
  path, so it belongs to render integration, not the Vulkan-free `[vdpm]` evidence test).
- ✅ **Static GPU residency** — device-local static asset upload split from dynamic mapped buffers.
  Landed in block B (`createDeviceLocalBuffer` for static vertices/indices/LODs/VIPM), and the batching
  follow-up is now done too *(branch `cr-static-residency-batch`)*: when a load-time `uploadBatch_` is
  open, `createDeviceLocalBuffer` records its staging copy into the batch's shared command buffer and
  retains the staging buffer, so the whole scene's buffer **and** texture uploads ride one submit +
  fence instead of a per-buffer stall. Outside a batch it still submits immediately.
- ✅ **Capability-driven device setup** *(branch `cr-capability-driven-device`)* — the required
  features are now **one** `kRequiredFeatures*` table per feature struct (`{pointer-to-member, name}`
  entries) that drives *both* `missingDeviceCapabilities` (the suitability check) and
  `createLogicalDevice` (the enable-chain) via generic `collectMissingFeatures` / `enableFeatures`
  helpers — so the check and enable lists can no longer drift. Descriptor-indexing *limits* stay a
  separate properties check (no enable counterpart). Verified: build clean, device still suitable, 0
  VUID render smoke.

---

## Design-review revisits — trigger-based

- **Character-controller as a scene component** (P7 decision). Today `CharacterController` is a
  `physics/` engine class driven from the main loop, **not** a scene `Components` variant — a variant
  `update(InputState, Transform)` has no `PhysicsWorld` access, and the controller *is* a world query,
  so making it a component would force `PhysicsWorld` into the Vulkan-free scene layer. **Revisit when**
  a 2nd consumer appears (authored character nodes / NPCs): the clean upgrade is a small
  `SceneUpdateContext { const InputState&; PhysicsWorld*; }` threaded into component updates — *not* a
  per-component back-pointer.
