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
  tests. **(b)** `refineForView` memoises `facingOf(v)` per canonical vertex (a vertex is a witness of
  many splits) and `emit` precomputes `activeAncestor` once per frame (the front is settled there);
  both are pure per-frame functions, so behaviour is byte-identical to the inline computation. **(c)**
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
  simplifier/forest/front change; unit-tested per rule. **Next: persistent front + hysteresis** (stop
  rebuilding the front from `coarsenAll()` every frame; refine above a high threshold, coarsen below a
  low one, so small camera moves near the budget don't pop topology).
  See [`lod.md`](lod.md) § Known limits (Metric fidelity). Render-path only ⇒ golden-neutral.
- **VDPM exact-visibility cones** — replace the per-frame `repairFoldovers` / `repairCoverage` sweeps
  with precomputed per-split foldover / silhouette / coverage cones. This *is* the `lod.md`
  "repairs vs cones" note; promote it now the repairs are proven.
- **GPU-driven active front** — drive `refineForView` + emission on the GPU (the forest + errors are
  already just buffers). The indirect-draw direction #3 opened. (The CPU-side reusable-scratch-emit +
  repair-counters piece is split out as its own near-term item under "Could" above — do that first;
  this arc is the full GPU move.)
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
