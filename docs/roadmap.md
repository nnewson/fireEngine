# Fire Engine — Roadmap

A forward-looking plan for the next major capabilities. The spine is
**Particles → Soft-body → Progressive meshes**, sequenced so each feature pays
down the infrastructure debt the next one needs (hardest last).

## Where the engine is today

A strong **rendering** codebase with a now-**mature physics** core (the P0–P9 track is complete).

- **Rendering (deep):** forward PBR, full split-sum IBL with multi-scatter, CSM +
  spot + point shadows with skinned self-shadowing, transmission (thin +
  volumetric), clearcoat, bloom, ACES, normal mapping — on Vulkan 1.4 dynamic
  rendering + synchronization2.
- **Assets / animation (deep):** glTF 2.0 with a long KHR extension list,
  KTX2/Basis, 64-joint skinning, morph targets, full keyframe interpolation.
- **Physics (growing):** Static / Kinematic / Dynamic bodies; box / sphere /
  capsule / AABB colliders; a **dynamic AABB-tree broad phase** (sweep-and-prune
  selectable behind a `BroadPhase` interface); **shape-specific
  `ContactManifold`** narrow phase (analytic pairs + box/box SAT; P1 ✅); a
  **linear sequential-impulse solver** with warm starting, friction, restitution,
  and split-impulse correction (P2 ✅) — resting stacks settle; **speculative-margin
  CCD** so fast movers don't tunnel (P2.5 ✅); **full rotational dynamics** — inertia
  tensors, orientation integration, lever-arm torque, so boxes topple and rest on a
  face (P3 ✅); **GJK/EPA convex hulls** with a `ConvexHullShape` authored from glTF
  (P3.5 ✅); **constraints, joints & ragdolls** — distance/ball-socket/hinge with
  angle limits, reusing the solver, plus a skinned-skeleton `Ragdoll` driven through
  a `Node` world-override and authored via `extras.Ragdoll` (P4 ✅); **scale** —
  simulation islands with a per-island solve, body/island **sleeping**, and a
  **dynamic AABB-tree broadphase** behind a `BroadPhase` interface (P5 ✅); **real level
  geometry** — a reusable `AabbBvh<T>`, true centre-of-mass offset, **compound** colliders,
  and **static triangle meshes** authored via `extras.Physics` `Shape: "Mesh"`/`"Compound"`
  (P6 ✅); queries + a character controller (P7 ✅); and **P9 solver robustness** (soft joints,
  TGS soft-step, 2D friction, reduced-coordinate articulations, mid-step manifold refresh). The
  physics track is complete end to end — see [`collision.md`](collision.md).

**Structural gaps this roadmap set out to close — now closed:** the compute-shader path (#1),
GPU instancing (#1 particles), frustum culling + spatial structure (`SceneCuller`), dynamic
per-frame mesh-buffer updates (#2 cloth), and anti-aliasing (TAA) have all landed. What remains
of the original insight below: the **progressive-mesh LOD** feature (#3) and the GPU-driven
indirect-draw direction it opens up.

Key insight (historical, now realised): the three target features all secretly depended on the
same infrastructure, so the ordering front-loaded that infrastructure via the lowest-risk feature
(#1 → #2 → #3). #1 and #2 are done; #3 is the remaining arc.

## Status (July 2026)

Three large bodies of work are **complete**: the **physics & collision track** (P0–P9, end to
end), the **rendering foundations** (particles, soft-body/cloth, Vulkan 1.3/1.4 modernization,
bindless materials, TAA, SSAO, frustum culling, overlay), and the **26-item staff-engineer code
review** (formerly `codereview.md` — now cleared, all of CR-01…CR-26 done). The one remaining
*major* planned arc is **rendering spine #3 — view-dependent progressive meshes**; everything else
outstanding is smaller could/maybe work (listed below). Detailed design for each done item lives in
the narrative sections further down and in [`collision.md`](collision.md) (physics).

## ✅ Done — at a glance

**Rendering**
- **#1 Particles** — compute-pipeline path + GPU-instanced soft/additive billboards.
- **#2 Soft-body / cloth (XPBD)** — GPU solver, dynamic per-frame vertex buffers, primitive
  collision, glTF `extras.Cloth` authoring, `bufferDeviceAddress` (descriptor-free solver).
- **Vulkan 1.3/1.4 modernization** — persistent pipeline cache, dynamic-state pipeline collapse
  (3→2), push descriptors (forward + shadow), timeline-semaphore frame pacing.
- **Bindless materials** — global set-2 `sampler2D[]` + materials SSBO, indexed in-shader.
- **Anti-aliasing** — velocity-buffer TAA + live tuning dials.
- **SSAO + contact shadows** — depth prepass → half-res AO → bilateral blur → forward apply.
- **Frustum culling + persistent scene BVH** (`SceneCuller`) + overlay counts.
- **ImGui debug/profiling overlay** — per-pass GPU timestamps, CPU frame plot, `RenderTunables`.

**Physics & collision (P0–P9)** — full detail in [`collision.md`](collision.md).
- debug-draw + determinism harness → shape narrowphase (`ContactManifold`) → sequential-impulse
  solver → speculative CCD → full rotational dynamics → GJK/EPA convex → constraints/joints/
  ragdolls → islands/sleeping/AABB-tree broadphase → static mesh + compound → queries + character
  controller → GJK edge robustness → character climbing → demo assets → **P9 solver robustness**
  (soft joints, TGS soft-step, 2D friction, reduced-coordinate articulations incl. CesiumMan,
  P9.6 mid-step manifold refresh).

**Code review (all 26)** — CR-20 render interpolation; the VMA memory block (CR-15/16/17 + CR-12
gpu); physics free-lists (CR-11/12); render↔scene layering (CR-09); Assets pointer-stability
(CR-10); MappedMemory→spans (CR-21); plus CR-01…08, 13, 14, 18, 19, 22, 23…26.

## 📋 To do — outstanding work

Suggested execution order — not binding, adjust freely.

**Now / near-term (small, in-flight)**
1. ✅ **Acceptance-testing runbook** ([`acceptance-testing.md`](acceptance-testing.md), landed) — copy-pasteable command +
   Khronos source + reference image per sample scene, physics demo, and generated feature. Referenced
   from the README. Full image-link verification is a manual pass (in progress).
2. ✅ **TransmissionTest regression** *(fixed, branch `transmission-thinwalled-roughness`)* — the grid
   is entirely thin-walled (`KHR_materials_transmission`, no volume), and the shader collapsed *all*
   thin-walled transmission to a flat irradiance tint (regressed by commit 982bcab), discarding the
   roughness-blurred scene sample. Fix: clear/frosted thin-walled glass now screen-refracts the
   roughness-blurred scene (so `roughnessFactor` varies again), while emissive thin-walled shades (the
   LightsPunctualLamp / StainedGlassLamp paper) stay a view-independent diffuse scatter — keyed on the
   **emissive factor**, since thickness lumped clear glass in with the shades and a *textured*
   `roughnessFactor` couldn't separate them. Verified on TransmissionTest / TransmissionRoughnessTest /
   StainedGlassLamp / LightsPunctualLamp.
3. **Cleanup pass** — the tidy-up flagged as the review wrapped.

**The remaining major arc (should) — Rendering spine #3, laddered discrete → VIPM → VDPM**
4. ✅ **Phase 1 — mesh simplifier + discrete LOD** *(branch `progressive-meshes-phase1`)*: a from-
   scratch **Garland–Heckbert QEM** simplifier (`graphics/mesh_simplifier`) that records the ordered
   collapse stream (the raw material for Phase 2/3). Position-welded connectivity (so glTF's seam-
   split vertices still simplify), a **wedge-preserving emit** (each corner keeps its own UV via
   nearest-wedge matching), and an **attribute-aware R⁵ quadric** (position + weighted UV) so the
   ordering collapses UV seams *last* — texture holds through LOD1, only the coarsest level shifts.
   `Geometry` builds discrete LODs at load; selection is screen-space error at the draw-build (forward
   + coarser for shadows); overlay toggle + pixel-error slider + triangles-drawn + a per-LOD debug
   tint. Verified: helmet 15452→7725→1931 with a real 3-level band, correct UVs, 0 VUID. Headless
   `[MeshSimplifier]` correctness tests (target count, error bound, boundary lock, determinism,
   collapse-replay, seam-UV). Known residual = **discrete popping** at the transition → Phase 2's job.
   - **Phase 2 — view-independent continuous (VIPM)** *(next)*: turn the collapse stream into a
     vertex hierarchy, pick one global refinement level + **geomorph** between steps (kills the pop)
     via the per-frame dynamic vertex buffer (#2 cloth path).
   - **Phase 3 — view-dependent (VDPM)**: promote the linear stream to a vertex forest + per-region
     active front (silhouette / near-edge refinement). Full design under "### 3" below.

**Opportunistic / supporting (could)**
5. **Rendering optimisations** *(flagged during CR-09)* — skip redundant per-object UBO re-uploads
   for unchanged (static) objects via a dirty-flag / world-revision check (`Node::worldRevision()`,
   `SceneCuller::Proxy::worldRevision` are precedent); and revisit routing the active camera through
   the `RenderableScene` seam rather than an explicit `drawFrame` argument. Pairs well with #4's
   heavier scenes.
6. **TAA skinned-deformation velocity** — exact previous-joint-matrix velocity to replace the v1
   camera-motion-only fallback (skinned meshes currently reproject on camera motion only).
7. **`Mat4::transformPoint` helper** *(flagged during the clang-tidy cleanup)* — the free
   `transformPoint(const Mat4&, Vec3)` (`m * {p, 1}` → `Vec3`) is copy-pasted in `physics_world.cpp`,
   `physics_world_shapes.cpp`, and `scene_culler.cpp`. Its natural home is a `Mat4::transformPoint(Vec3)`
   method (`RigidTransform` already has one; `Mat4` doesn't). Lift it onto `Mat4`, replace the three
   copies + their call sites. Small + mechanical; kept out of `tidy-aabb-corners` to hold the impact
   radius (see `clang-tidy.md`).

**Cosmetic / on demand (maybe)**
8. **Ragdoll plausibility polish** *(branch `ragdoll-joint-settle`)* — ✅ **post-settle arm drift** +
   ✅ **residual settle-yaw** both fixed, via settle assists in `Articulation::integrateVelocities`
   that mirror the base's existing linear one: (a) a per-DOF joint settle assist strongly decays only
   joints already moving slowly (`kJointSettle{Speed,Damping}`), so limbs stop within ~0.5 s instead
   of crawling to the sleep threshold over ~5 s; (b) a yaw settle assist decays only the base's spin
   about the *vertical* (`kBaseYawSettleDamping`, projecting angular velocity onto world-up — the
   non-vertical axes stay undamped to keep a near-planar chain stable), killing the slow residual
   body rotation that was the last visible motion. Dramatic collapse untouched; determinism hash
   unaffected (rigid-only). Still open: per-joint hinge-limit authoring (knees/elbows vs uniform
   cones).

**Parked with data — revisit only on a concrete need (maybe)**
9. **P9.6 Stage 2** — per-substep re-detection for >20 rad/s spinners resting on floors (boundary
   quantified + gated by `Demos.RotationalTunnellingBoundedTo20RadPerSec`).
10. **Mesh-contact mid-step refresh.**
11. **Link-vs-dynamic-rigid articulation contacts** (link colliders are link-vs-static only today).
12. **Joint split-position pass** (P9 item 4).

**Design-review revisits — trigger-based (maybe)**
13. **Character-controller as a scene component** — when a 2nd consumer appears (authored character
    nodes / NPCs), via a `SceneUpdateContext { const InputState&; PhysicsWorld*; }` threaded into
    component updates (not a per-component back-pointer). See "Design reviews" at the end.

## Candidate assessment

| Feature | Complexity | Hidden prerequisites | Payoff |
|---|---|---|---|
| Particle systems | Low–med | compute path, GPU instancing, soft/additive transparency | High, immediate, visible |
| Soft-body / cloth | Med–high | compute, **dynamic mesh-buffer updates**, collision integration | High, very "alive" |
| View-dependent progressive meshes | **High** | dynamic mesh buffers, **frustum culling + spatial structure**, ideally GPU-driven refinement | High on heavy scenes; invisible on simple ones |

Progressive meshes (Hoppe-style edge-collapse / vertex-split with view-dependent
refinement) is the most research-heavy *and* the most prerequisite-laden, and it
only pays off once scenes are big enough to stress it. Particles is the opposite:
low risk, instantly gratifying (HDR + bloom + depth make soft/additive particles
look great), and the natural vehicle for building **compute + instancing**, which
the other two reuse.

## Other items worth considering

Weighed against the three simulation/geometry features:

- **Anti-aliasing (TAA). ✅ Done.** The single biggest *perceptual* quality win —
  the image had crawling jaggies. Shipped as velocity-buffer TAA (chosen over MSAA
  so shading/specular shimmer is covered too, and over camera-reprojection so
  animated meshes reproject correctly): per-node previous-world tracking → a
  screen-space motion-vector attachment → Halton(2,3) jitter + history resolve with
  neighbourhood clamp. Skinned deformation velocity is camera-motion-only in v1
  (a later pass could add previous joint data).
- **Frustum culling + bounding-volume hierarchy. ✅ Done.** Unglamorous but a hard
  prerequisite for progressive meshes being *meaningful*; pays for itself
  everywhere. Folded in as the on-ramp to LOD rather than a standalone item. Shipped
  in three phases on the reusable `AabbBvh<T>` core P6 left behind:
  1. **`graphics/Frustum`** (Gribb–Hartmann 6-plane extraction, Vulkan [0,1] depth) +
     a positive-vertex AABB test (no false negatives; invalid bounds always visible).
     Brute-force per-bucket cull — camera frustum drops non-shadow draws in
     `buildDrawBuckets`; each shadow cascade / spot / point-face filters its draw list
     against its own frustum in `Shadows::recordPass`. Reuses the per-draw world
     `shadowBounds` already computed every frame.
  2. **Persistent scene `AabbBvh<Node*>`** (`scene/SceneCuller`): each rigid renderable
     node owns a fat-AABB proxy (`composedWorld · localBounds`); `sync()` refreshes
     bounds + reconciles the proxy set, `cull()` unions the visible leaves across the
     camera + every shadow frustum and returns the nodes in *none* of them.
     `Node::render` skips draw-building for those (still recurses children). Deformable
     (skinned/morph) nodes are never tracked — always drawn, left to the precise
     per-bucket cull. `AabbBvh::traverse(predicate, fn)` is the generic descent that
     lets the culler bridge AABB↔Frustum without `collision/` depending on `graphics/`.
  3. **Overlay** — `RenderTunables::cullingEnabled` toggle + tracked/visible/culled
     counts in the ImGui panel (the toggle doubles as the A/B regression escape hatch).
  - **It stays a rendering item, not a physics P5.5.** Frustum culling is *not* a
    prerequisite for P6 (physics never tests the camera frustum), and it doesn't even
    need a tree to start: brute-force per-object frustum tests (loop renderables,
    6-plane test against `composedWorld · localBounds`) are O(N) and already a large
    win into the thousands of objects. The shared asset is the **AABB BVH machinery**,
    not the FC instance — and P6's triangle BVH forces that abstraction anyway (see
    P6 below). Sequence: **P6 extracts a reusable `AabbBvh<T>` core; frustum culling
    rides it afterward** (or ships brute-force first if a heavy scene needs it sooner).
    P5's `DynamicAabbTreeBroadPhase` is the template for that core.
- **Debug / profiling overlay (ImGui + GPU timestamps). ✅ Done.** A force-multiplier —
  tuning particle emitters, cloth stiffness, and LOD thresholds by recompiling
  constants gets painful fast. Now live: per-pass GPU timestamps + CPU frame plot and a
  `RenderTunables` panel (F1 / `--overlay`), so the knobs below are sliders, not rebuilds.
- **SSAO / contact shadows. ✅ Done.** Medium effort, reuses the depth buffer, adds
  grounding. A good optional rendering win — **planned in detail under "Slot in
  anywhere it fits" below.**

If adding one non-simulation item: **anti-aliasing** for visible quality, with
**frustum culling** coming along as part of LOD.

## Roadmap

### 1. Particle systems — build the foundation while getting a quick win

- **Milestone A — Compute pipeline path. ✅ Done.** `ComputePipeline` +
  storage-buffer dispatch + synchronization2 buffer barriers
  (`render/compute_pipeline.{hpp,cpp}`), proven by a debug-only headless
  self-test. Three of the planned features build on this.
- **Milestone B — GPU-instanced particles. ✅ Done.** Scene-graph
  `ParticleEmitter` component gathered like `Light`; renderer-owned
  `ParticleSystem` runs a compute simulation (spawn / age / integrate in a pooled
  SSBO) and renders instanced billboards additively into the HDR target (bloom),
  with **soft particles** (in-shader fade against sampled scene depth).
- **Leaves behind:** compute + instancing infrastructure, reusable everywhere.

### 2. Soft-body / cloth (XPBD) — reuse compute; build dynamic meshes

- **Stage 1 (solver + render) ✅ / Stage 2 (collision) ✅ / Stage 3 (authoring +
  tuning) ✅.** GPU XPBD cloth: a graph-coloured distance solver
  (`cloth_predict`/`solve`/`finalize` compute), the solved positions + normals
  written into a storage vertex buffer the forward/shadow passes read, and
  **collision** (`cloth_collide`) against world primitives gathered from
  `PhysicsWorld` (`gatherColliders` → plane/sphere/box/capsule) + a ground plane.
  `-c` drapes an unpinned sheet over a sphere onto the floor. **Stage 3** added
  **glTF `extras.Cloth` authoring** (`makeClothFromMesh` welds an arbitrary mesh →
  particles + structural/bend constraints, pin rules; sample `ClothBanner.gltf`),
  **per-type bend stiffness** (structural/shear stiff, bend soft, with a live global
  compliance multiplier), **wind** + overlay tuning, and a unified
  **CSR-adjacency normal** recompute (`buildNormalAdjacency`) that replaced the
  grid-only central-difference so any topology lights correctly.
- XPBD is robust, easy to keep stable, and pleasantly *independent* of the rigid
  solver (its own particle + constraint system) — not blocked on maturing
  rigid-body dynamics first.
- Introduced **dynamic per-frame vertex-buffer updates** (compute writing
  positions / normals) and **collision against the box / sphere / capsule / plane
  colliders** — the previously-missing mesh-deformation plumbing.
- Pairs naturally with the existing skinning / morph vertex path.
- **`bufferDeviceAddress` — ✅ done (the first customer was the cloth solver).** The
  solver first shipped on **descriptor sets** (the proven `ParticleSystem` pattern)
  to de-risk first light, then was converted: `device.cpp` enables
  `VkPhysicalDeviceVulkan12Features.bufferDeviceAddress` in the device pNext chain,
  `Device::createBuffer` threads the device-address memory-allocate flag, and the
  cloth particle / constraint / render-vertex / per-frame-collider buffers carry
  `eShaderDeviceAddress` usage. The four `cloth_*.comp` shaders now take their
  buffers as `GL_EXT_buffer_reference` pointers pushed in the push constant — the
  solver is descriptor-free. This is the foundation for the GPU-driven indirect
  draws #3 wants.

### Vulkan 1.3/1.4 modernization — before #3

The engine already targets `vk::ApiVersion14` and uses **dynamic rendering +
synchronization2** (1.3) and **bufferDeviceAddress + descriptorIndexing** (1.2). The
current MoltenVK exposes the rest of what's below (`VK_EXT_extended_dynamic_state1/2/3`,
`VK_KHR_push_descriptor`, `VK_KHR_timeline_semaphore`, `VK_KHR_maintenance5/6`,
`VK_KHR_dynamic_rendering_local_read`), and most are core at 1.4 — so no new feature
gates beyond the EDS3 flags. Sequenced before #3 because its GPU-driven / culling work
benefits from the leaner pipeline + descriptor model these leave behind. Prioritised:

1. **Pipeline cache. ✅ Done.** `Device` owns one shared `VkPipelineCache`
   (`createPipelineCache`, exposed via `Device::pipelineCache()`), fed to every graphics
   *and* compute pipeline creation (`Pipeline::createGraphicsPipeline`, `ComputePipeline`
   ctor) instead of a null cache. **Persisted to disk** (`pipeline_cache.bin`: loaded in
   `createPipelineCache`, written in `~Device`, and validated against the device's vendor/device
   IDs + cache UUID so a driver/GPU change starts cold) so the driver's compilation — on MoltenVK
   the deferred Metal compile, otherwise paid on every cold start — is paid once across runs; it
   also dedupes within a run and warms pipeline recreation on swapchain resize.
2. **Collapse the forward pipeline variants via dynamic state. ✅ Done (partial, 3 → 2).**
   `opaque` + `opaqueDoubleSided` are now **one** pipeline: cull mode is a core-1.3
   dynamic state (`VK_DYNAMIC_STATE_CULL_MODE`, opted in via `PipelineConfig::dynamicCullMode`),
   set per draw in `Renderer::recordDrawBucket` / the transmission bucket from
   `DrawCommand::doubleSided`. `blend` stays a **separate** pipeline: folding it in needs
   dynamic blend enable/equation (`VK_EXT_extended_dynamic_state3`), and the current MoltenVK
   advertises the extension but reports `extendedDynamicState3ColorBlendEnable` /
   `…ColorBlendEquation` as **`false`** (Metal bakes blend into the pipeline descriptor) — so
   the full 3 → 1 isn't reachable on this hardware. `depthWriteEnable` only differs for blend,
   so it stayed static too. Documented under README **Limitations**.
3. **Push descriptors for the forward set 0. ✅ Done.** The forward per-object set 0
   (frame/skin/morph UBOs + morph-targets SSBO) is now a `VK_KHR_push_descriptor` layout
   (`PipelineConfig::pushDescriptorSet0` → `ePushDescriptorKHR`), pushed inline at draw time
   by the shared `pushForwardObjectDescriptors` helper (forward + transmission passes) from
   buffer handles carried on `DrawCommand`. `createObjectDescriptors` / the forward
   `GeometryBindings::descSets` are gone — no per-object forward descriptor allocation.
   MoltenVK supports it (`pushDescriptor=true`, `maxPushDescriptors=287`); the extension is
   enabled in `device.cpp` and the **core** entry `vkCmdPushDescriptorSet` is called (the
   vcpkg loader exports the core symbol, not the `KHR` alias — static dispatch).
   **Shadow pass — ✅ done too.** The shadow set 0 is now a push-descriptor layout as well
   (`shadowConfig().pushDescriptorSet0`, inherited by the self-shadow first/second configs),
   pushed inline per draw by `pushShadowObjectDescriptors`: bindings 0..3 are the per-object
   ShadowUBO + reused skin/morph/morphSsbo buffers (carried on `DrawCommand`), bindings 4/5
   are the shared self-shadow image+sampler read straight from `Resources`. `createShadowDescriptors`
   and the `ShadowDescriptorRequest`/`Result`/`GeometryInfo` structs (and the now-dead forward
   `ObjectDescriptorRequest`/`GeometryDescriptorInfo` scaffolding) are gone — **no per-object
   descriptor allocation remains in either the forward or shadow path.**
4. **Timeline semaphores. ✅ Done.** Frame pacing moved from per-frame binary fences
   (`Frame::inFlight_` + the `imagesInFlight_` per-image fence tracking) to one monotonic
   **timeline semaphore** (`Frame::timeline_`, `timelineSemaphore` enabled in `device.cpp`).
   Submit signals it at an incrementing value alongside the binary present semaphore; acquire
   CPU-waits the timeline on the per-slot value (`frameTimelineValue_`, gates cmd-buffer / UBO
   reuse) and the per-image value (`imageTimelineValue_`, gates swapchain-image reuse) via
   `Renderer::waitTimeline` → `vkWaitSemaphores`. The WSI acquire/present semaphores
   (`imageAvail`/`renderDone`) **stay binary** — timeline semaphores aren't permitted with
   `acquireNextImageKHR`/`presentKHR`. No fences remain in the frame loop; the core
   `vkWaitSemaphores` entry is used (not the `*KHR` alias — static dispatch).

Also available if a future need arises: `maintenance5`'s `vkCmdBindIndexBuffer2`, and
`dynamic_rendering_local_read` (same-pixel attachment reads inside a dynamic render pass).

### 3. View-dependent progressive meshes — everything else now exists

- On-ramp: **frustum culling + a spatial / bounds structure** (build on the
  existing `Bounds3`). ✅ done.
- **Phase 1 — discrete LOD ✅ done** (branch `progressive-meshes-phase1`; see the "To do" item #4
  for the full write-up): from-scratch attribute-aware (R⁵ position+UV) QEM simplifier that records
  the collapse stream, position-welded connectivity + wedge-preserving UV emit, load-time discrete
  LODs, screen-space selection, overlay + debug tint. Residual = discrete popping.
- Then the progressive-mesh hierarchy and runtime view-dependent refinement,
  reusing dynamic mesh buffers (from #2) and compute (from #1):
  - **Phase 2 (VIPM)** — vertex hierarchy from the collapse stream + geomorph (dissolves the pop).
  - **Phase 3 (VDPM)** — vertex forest + per-region active front (silhouette / near-edge).
- Done last so the engine is mature enough to support it — with real scenes to
  test against.

### Bindless materials (descriptor indexing) — ✅ done

Off the simulation spine; the single biggest descriptor-model cleanup available,
and a foundation for the GPU-driven direction (#3).

- **Was:** descriptor **set 0 allocated per-object × per-frame** (15 bindings —
  material UBO + 10 textures + skin/morph), with a brittle *slot enum ↔
  `MaterialUBO::uv[]` ↔ descriptor bindings ↔ shader* coupling, plus a variant-
  switch descriptor re-point.
- **Now:** a new **forward set 2** holds one global `sampler2D[]` **texture array**
  (indexed by `TextureHandle`) + a global **materials[] SSBO** (indexed by a per-
  draw `ForwardPushConstants::materialIndex`). The shader samples
  `textures[material.textureIndex[slot]]` and reads `materials[pc.materialIndex]`.
  Per-object **set 0 collapsed to 4 vertex-stage bindings** (frame/skin/morph UBOs
  + morph SSBO); the per-object material UBO, the 10 texture descriptors, the
  variant re-point (`updateObjectGeometryTextures`), and the slot↔binding coupling
  are all gone. Materials are plain integer indices now.
- **Enabled (`VkPhysicalDeviceVulkan12Features`):** `descriptorIndexing`,
  `runtimeDescriptorArray`, `shaderSampledImageArrayNonUniformIndexing`,
  `descriptorBindingPartiallyBound`, `descriptorBindingSampledImageUpdateAfterBind`,
  `descriptorBindingVariableDescriptorCount`. The combined `sampler2D[]` array works
  on the current MoltenVK (Metal argument buffers); the device caps total
  combined-image-samplers per stage at 1024, so the array is sized 512
  (`kMaxBindlessTextures`) to leave headroom for the set-1 IBL samplers.
- **Touched:** `device` (features), `pipeline` (set-2 layout), `resources` (the
  bindless set + texture/material registries), `descriptor_bindings`, `object`/
  `descriptors` (set-0 shrink), `ubo`/`draw_command` (indices), and `shader.frag`.

### Slot in anywhere it fits

- **Anti-aliasing** (velocity-buffer TAA) — ✅ done; landed as the palate-cleanser between #1 and #2.
  - **TAA tuning dials** — ✅ done; **history blend** and **resolve sharpen** are now live
    sliders in the debug overlay (`RenderTunables`), not recompiled constants. Still
    pending: exact skinned-deformation velocity (previous joint matrices) to replace the
    v1 camera-motion-only fallback.
- **ImGui debug overlay** — ✅ done. Dear ImGui (1.92, dynamic rendering) drawn into the
  swap image after post-process; F1 toggles, `--overlay` starts it visible. Per-pass GPU
  timestamps (`GpuProfiler`, graceful MoltenVK fallback) + CPU frame plot, and a live
  `RenderTunables` panel (TAA, debug views, bloom/IBL/sun, particle emitter scales). ImGui
  core plus GLFW and Vulkan backends come from the vcpkg `imgui[glfw-binding,vulkan-binding]`
  manifest dependency; the engine still links Vulkan/GLFW directly, with
  `cmake/fireengine_imgui.cmake` wrapping the ImGui archive to avoid duplicate transitive links.

- **SSAO + contact shadows — ✅ done (all phases).**
  Screen-space ambient occlusion and short-range contact shadows, both driven off the depth
  buffer. Adds grounding/contact darkening that IBL ambient and the CSM can't capture. Pairs
  naturally with the existing TAA, which **denoises the noisy AO across frames for free** — so
  a low per-pixel sample count suffices. Landed: a **depth prepass** (`Pipeline::depthPrepassConfig`,
  reusing `shader.vert` with `invariant gl_Position` + an empty `depth_prepass.frag`; forward
  loads it with `LESS_OR_EQUAL`), a half… full-res **`Ssao` subsystem** (`render/ssao.{hpp,cpp}`,
  `ssao.frag`) that reconstructs view position+normal from depth (no normal G-buffer; analytic
  unprojection from `proj`), writes an **R8G8 target** (R = AO, G = contact), and the **forward
  shader** sampling it (forward set 1, binding 13) to multiply SSAO into the IBL terms and the
  contact term into the **direct sun** (ambient keeps pure CSM). A **depth-aware bilateral blur**
  (`ssao_blur.frag`, 5×5 with a view-space-Z edge-stop) then smooths the per-pixel sampling/march
  noise without bleeding across silhouettes; the forward pass samples the blurred target. Live
  overlay sliders (radius/bias/intensity/power, contact length) + a `--debug-ssao` view;
  `GpuProfiler` gains `DepthPrepass`/`Ssao` passes. **Contact shadows default ON** alongside SSAO:
  they fill the CSM's short-range contact gap, with an N·L gate, view-Z-scaled ray-march depth
  window, and depth-silhouette edge guard to keep screen-space streaks contained.

  **The ordering problem.** This is a *forward* renderer: depth is produced *by* the forward
  pass (cleared then written), and ambient/IBL is applied inside `shader.frag`. AO needs
  depth (+ normals) *before* lighting. The chosen fix is a **depth prepass** — it makes depth
  exist before lighting (correctly "reusing the depth buffer"), lets AO modulate **ambient
  only** (direct light stays correct), and hands the forward pass free early-Z. (Rejected
  alternative: a post-pass that darkens the tonemapped HDR — cheaper, but darkens direct light
  too and can't fold contact shadows into the sun term.)

  **Pipeline (depth prepass → AO pass → blur → forward applies):**
  1. **Depth prepass** — opaque geometry rendered depth-only into the shared D32 buffer before
     the forward pass, reusing the skin/morph-aware `shadow.vert` path with an empty fragment
     shader. Forward depth-clear becomes `loadOp = Load`.
  2. **AO pass** (`Ssao` subsystem, half-res) — reads depth only, **reconstructs view-space
     normals from depth derivatives** (no normal G-buffer — stays depth-only), and computes a
     range-checked hemisphere-kernel **SSAO** plus a sun-direction screen-space ray-march for
     **contact shadows**. Output: a small (R8/RG8) half-res target.
  3. **Bilateral (depth-aware) blur** — cleans the half-res noise; kept light because TAA
     finishes the denoise.
  4. **Forward applies** — `shader.frag` samples the AO target (screen-space UV) and multiplies
     **SSAO into the IBL terms** (combined with the existing material `SLOT_OCCLUSION`) and
     **contact shadows into `primaryDirectionalVisibility`**.

  **Touches:** new `render/ssao.{hpp,cpp}` (subsystem in the `Taa` mould — owns target, pipelines,
  descriptors, `recreate()`); new `ssao`/`ssao_blur` shaders + a depth-only prepass vert; `pipeline.cpp`
  factories (`depthPrepassConfig`/`ssaoConfig`/`ssaoBlurConfig`, covered in
  `test_pipeline_config.cpp`); `renderer.cpp` (prepass + AO pass + barriers + ctor/resize wiring,
  forward depth `loadOp`); `shader.frag` (sample + apply + a raw-AO debug view); forward set 1 gains
  the AO texture as a global binding; `GpuProfiler` (`DepthPrepass`/`Ssao` passes); `RenderTunables` +
  `constants.hpp` (enable/radius/intensity/bias + contact length/steps as live sliders).

  **Phasing (both, hardest-sharing-the-cheapest-last):** (1) depth prepass + plumbing; (2) SSAO
  + apply to ambient; (3) bilateral blur + TAA tuning; (4) **contact shadows** — a small add-on
  once the prepass + AO pass + depth-sampling infra from 1–3 exists, just another term in the AO
  pass applied to the sun visibility. Medium effort overall, roughly the size of the TAA addition.

## Physics & collision — adding weight

The rendering spine above is deep; the physics core is now a real rigid-body solver
([`collision.md`](collision.md) documents it in full). The narrowphase produces a **shape-specific
`ContactManifold`** (P1 ✅) and a **sequential-impulse solver** resolves it (P2 ✅) —
warm-started friction + restitution impulses with split-impulse positional correction,
so `PhysicsMaterial::friction` and `mass` are live and resting stacks settle.
**Speculative-margin CCD** (P2.5 ✅) stops fast movers tunnelling, and **full rotational
dynamics** (P3 ✅) — inertia tensors, orientation integration, lever-arm torque — make
`PhysicsBody::angularVelocity` live, so boxes topple, spin, and rest on a face. A
**GJK/EPA convex narrowphase** with an authored `ConvexHullShape` (P3.5 ✅) covers
arbitrary convex colliders. **Constraints, joints & ragdolls** (P4 ✅) reuse the solver
as generic constraint rows — distance/ball-socket/hinge with angle limits — and a
skinned `Ragdoll` driven through a `Node` world-override (authored via
`extras.Ragdoll`). Next is scale — sleeping + islands (P5).

**Key insight (same shape as the rendering spine):** every richer feature —
friction, resting stacks, toppling boxes, joints, ragdolls — is downstream of one
piece of infrastructure: a **contact manifold + an iterative constraint solver**.
Build those two first and the rest are rows fed into the same solver. So the
ordering front-loads the manifold and solver, then reuses them, hardest last.

Two invariants to preserve throughout (already true today, easy to lose):
the **scene/physics handle split** — `scene::Node` owns opaque handles, all state
lives behind `PhysicsWorld` — and the **authority-by-body-type rule**
(Static/Kinematic scene-authored in, Dynamic physics-authoritative out). Every
item below lands *behind* the `PhysicsWorld` boundary.

### P0. Debug draw + determinism harness — see it before you grow it — ✅ done

- **Debug draw ✅.** A renderer-owned `DebugDraw` subsystem draws immediate-mode
  wireframes — broadphase AABBs (green), authored collider shapes (cyan,
  sphere/box/capsule), and approximate contact points + normals (red/yellow) —
  into the HDR target after particles, via a line-list pipeline reusing the
  standard `Vertex`. Depth handling is a **dynamic depth-test toggle** (x-ray vs
  occluded). Per-category overlay checkboxes ("Physics debug" panel) +
  `--debug-physics`; a `DebugDraw` GPU-timing row. The data is Vulkan-free:
  `PhysicsWorld::gatherColliders()` (shapes, reused) + new `debugColliderBounds()`
  (AABBs) + `debugContacts()` (per-step `DebugContact`, captured before the TOI
  resolve — approximate point sharpens at P1). Sleep state lands with P5.
- **Determinism harness ✅.** `tests/physics/test_physics_determinism.cpp` +
  `tests/support/state_hash.hpp` (FNV-1a over body transform + velocity bits):
  **ReplayIsBitIdentical** (two runs of a scripted gravity+collision scene hash
  equal — guards against future non-determinism), **FreeFallMatchesClosedForm**
  (semi-implicit Euler sanity), and **GoldenHash** (recorded end-state tripwire).
  The existing `step()` was already deterministic (no RNG, stable sort, SAP
  tie-break by ColliderId), so no solver changes were needed.
- **Leaves behind:** the validation surface every later phase is debugged against.

### P1. Shape-specific narrowphase → `ContactManifold` — the spine — ✅ done

- **Manifold ✅.** The single `SweptAabbContact` is replaced by a real manifold
  (`collision/contact_manifold.hpp`):
  ```cpp
  struct ManifoldPoint { Vec3 position; float penetration; };
  struct ContactManifold { Vec3 normal; std::array<ManifoldPoint, 4> points; int count; };
  ```
- **Dispatch ✅.** `NarrowPhase::collide(WorldShape, WorldShape)` 2-arg-`std::visit`
  dispatches each broadphase pair to an analytic `collidePair` overload by shape:
  sphere/sphere, sphere/box, sphere/capsule, capsule/capsule, box/capsule
  (iterated segment↔OBB closest point), and **box/box SAT with Sutherland-Hodgman
  face clipping** for a up-to-4-point manifold (edge-edge falls back to a single
  point). Normal convention: points `b → a` (target → moving, the push-out
  direction). Closest-point primitives live in `collision/geometry.{hpp,cpp}`
  (unit-tested in isolation). `PhysicsWorld::worldShape` composes each authored
  `ColliderShape` + body transform into a neutral `collision/world_shape.hpp`
  `WorldShape` (`WorldSphere`/`WorldBox`/`WorldCapsule`), reused by
  `gatherColliders`. The general **GJK + EPA** convex path is still deferred — the
  analytic pairs cover every authored shape today.
- **Response (interim, since superseded by P2).** P1 shipped a discrete
  penetration push-out + velocity reflection as a placeholder; **P2 replaced it
  with the sequential-impulse solver** (see below).
- **Deferred → P2.5 (✅ done).** P1 left fast movers able to tunnel; **P2.5** added
  speculative-margin CCD (see below). The old `NarrowPhase::sweptAabb` seed was
  superseded by the analytic narrowphase gap and removed.
- **Leaves behind:** the manifold every solver and query consumes.

### P2. Sequential-impulse solver (linear) — make contact behave — ✅ done

- **`ContactSolver`** (`physics/contact_solver.{hpp,cpp}`) is a linear
  sequential-impulse solver, decoupled from `PhysicsWorld` (operates on a flat
  `SolverBody` array + `SolverContactInput` list, unit-tested in isolation):
  - ✅ **warm starting** — a persistent cache keyed by collider pair, proximity-
    matching contact points across frames to inherit impulses (kills resting-stack
    jitter). Lookup-only, so the `unordered_map` can't break determinism.
  - ✅ **restitution** as a normal bias impulse, gated by a resting-speed threshold
    so settling stacks don't buzz.
  - ✅ Coulomb **friction** as a clamped tangent impulse (two-tangent cone);
    `PhysicsMaterial::friction` is finally live (combined `sqrt(a*b)`).
  - ✅ correct **mass ratios** between two dynamic bodies (inverse-mass weighting).
  - ✅ **split-impulse** positional correction (slop-tolerant pseudo-velocity pass)
    so penetration removal doesn't add energy.
  - ✅ `kVelocityIterations` velocity + `kPositionIterations` position sweeps per
    fixed step (`physics/physics_constants.hpp`).
- **`step()` reordered** to the standard impulse-solver shape: integrate velocity →
  narrowphase → warm-start + velocity solve → integrate position → position solve →
  store cache. Free-fall stays exact (`FreeFallMatchesClosedForm` unchanged); the
  `GoldenHash` was re-recorded for the new solver math.
- Still **linear-only**: the win is **resting stacks of boxes that settle and stay
  still** instead of jittering or sinking (`StackOfBoxesSettlesAndStaysStill`) — the
  first time the engine looks like it has real physics. Rotation arrives in P3 ✅.
- **Leaves behind:** the iterative constraint solver every later phase reuses.

### P2.5. Speculative-margin CCD — anti-tunnelling, reusing the solver — ✅ done

- **Signed-separation narrowphase.** `NarrowPhase::collide` takes a
  `speculativeMargin`: every `collidePair` now reports *separated-but-within-margin*
  pairs with a **negative penetration** (`= -gap`); margin 0 reproduces the old
  overlap-only behaviour. The analytic pairs are a one-line threshold change; box/box
  SAT keeps the max separation (no early-out) and emits a single gap point.
- **Motion-aware broadphase.** `Collider::update(world, motion)` extends the swept
  AABB by the predicted displacement (`velocity × dt`, threaded from
  `PhysicsWorld::updateColliders`), so a body starting a step at high speed from rest
  still pairs with what it will reach — the P2 reorder (narrowphase *before* position
  integration) had otherwise broken that coverage.
- **Speculative solver bias.** `PhysicsWorld::contactForPair` computes a per-pair
  margin `(|v_moving| + |v_target|)·dt + kSpeculativeDistance`. In `ContactSolver`, a
  gap contact (penetration < 0) sets the normal bias to `-separation/dt`: the body may
  close the gap this step but the non-negative impulse clamp brakes any overshoot, so
  it reaches the surface without tunnelling. No rewind, no substep — entirely inside
  the existing velocity loop; the split-impulse position pass already no-ops on gaps.
- **Verified** by `FastBulletDoesNotTunnelThroughThinWall` (a 300 m/s box stops at a
  thin wall) + solver-unit and narrowphase-unit gap tests; resting/stack/friction
  behaviour unchanged. The old `sweptAabb` and its tests were removed; `GoldenHash`
  re-recorded.
- **Known limitation:** the speculative normal is the closest-feature normal at the
  *start* pose, so a steep corner graze can be slightly off — full conservative
  advancement (ideally on the P3.5 GJK distance) is the future refinement.

### P3. Rotational dynamics — translational → full rigid body — ✅ done

- **Inertia ✅.** Per-shape diagonal inertia tensors (box / sphere / capsule / aabb)
  computed in `createCollider` (`PhysicsBody::inverseInertiaLocal`); the solver builds
  the world inverse inertia `R·diag(invI_local)·Rᵀ` per body per step (new `math/mat3.hpp`).
- **Integration ✅.** Quaternion orientation integration via an exponential-map
  `Quaternion::integrate(ω, dt)` (+ a Hamilton `operator*`), applied to Dynamic bodies
  in `step()` alongside position.
- **Angular solver ✅.** `ContactSolver` carries angular velocity + orientation +
  world inverse inertia; every normal/friction impulse adds lever-arm torque
  `ω += I⁻¹(r×P)`, the effective mass gains `(r×d)·I⁻¹(r×d)`, relative velocity
  includes `ω×r`, warm starting applies angular, and the split-impulse position pass
  corrects orientation via a pseudo-angular velocity. A centred contact (`r=0`)
  reproduces the P2 linear result exactly.
- This is the leap the roadmap is named for: **boxes topple, spin, and come to rest on
  a face** (`TallTiltedBoxTopplesOntoItsSide`, `BoxDroppedFlatRestsFlatAndStill`). The
  `GoldenHash` was re-recorded (the determinism scene now rotates — already hashed).
- **Scope:** single-collider, centre of mass = body origin (compound / parallel-axis
  is P6); gyroscopic torque omitted (standard for stable game physics).

### P3.5. Convex narrowphase — GJK distance + EPA depth — ✅ done

The **shape-coverage** complement to the solver spine (orthogonal to the P2.5
*temporal* CCD work — this is about *which* shapes collide, not *when*).

- **GJK + EPA ✅** (`collision/gjk_epa.{hpp,cpp}`) over **support functions**
  (`collision/support.hpp`): GJK closest-point/distance for separated pairs (witnesses
  via simplex barycentrics) + EPA penetration depth/normal for overlap, with a
  directional-MTV fallback for the axis-aligned-symmetric EPA-init degeneracy. One
  universal path for any convex-involving pair (convex/convex, convex/box, convex/
  sphere, convex/capsule).
- **Manifold ✅:** `collideConvex` (in `narrow_phase.cpp`) builds the same
  `ContactManifold` as the analytic paths — a **precomputed-face clip** (the box/box
  clipper generalised to arbitrary convex faces) for polytope-polytope, a single
  witness point for curved contacts. Honours the speculative margin (GJK distance ≤
  margin → negative-penetration gap), so P2.5 CCD works for hulls too.
- **`ConvexHullShape` ✅** (vertices + face loops) authored from glTF
  `Shape: "ConvexHull"`, built from the node mesh by `buildConvexHull`
  (`core/convex_hull_builder.{hpp,cpp}`: weld + coplanar-triangle merge into ordered
  face loops; `isConvex` validation). The analytic primitive pairs are untouched.
- **Scope/limits:** convex inertia is approximated by the hull's AABB box inertia; the
  builder assumes a convex input mesh (quickhull-from-point-cloud is the future
  robustness upgrade). A convex cube rests/topples like the primitive box
  (`ConvexHullCubeRestsFlatLikeABox`).
- **Leaves behind:** the general convex contact query every later shape (hulls,
  P6 mesh triangles) reuses.

### P4. Constraints & joints → ragdolls — reuse the solver — ✅ done

- **Shared solver math ✅** (`physics/solver_math.hpp`): the per-body primitives
  (world inverse inertia, relative velocity, effective mass, apply-impulse) extracted
  from the contact solver so the joint solver reuses them. `ContactSolver` was
  refactored onto them bit-identically (the P2/P3 determinism + contact tests guarded
  it; `GoldenHash` unchanged).
- **Joints ✅** (`physics/joint.{hpp}` + `physics/joint_solver.{hpp,cpp}`): distance,
  ball-socket, and hinge expressed as **generic constraint rows** (full Jacobian
  linearA/angularA/linearB/angularB), solved Gauss-Seidel and **interleaved** with
  contacts over the same `SolverBody` array (joints first each sweep). Position error
  feeds back through a `kJointBaumgarte` velocity bias (no separate split-impulse
  pass); impulses warm-start across frames (cached per joint, lookup-only).
  `PhysicsWorld::createJoint` / `destroyJoint` store them tombstoned like bodies.
- **Limits ✅:** hinge `[lower, upper]` angle clamp and ball-socket **cone-twist**
  (swing cone + ±twist) via a **swing-twist quaternion decomposition**, each emitting a
  one-sided row only while violated (no energy at rest; warm-started by stable slot).
- **Ragdoll ✅** (`scene/ragdoll.{hpp,cpp}`): `Ragdoll::make` builds a capsule body +
  ball-socket(+cone-twist) joint per bone (parent resolved through the node hierarchy,
  self-collision filtered by a shared layer), seeded from each bone's bind-pose
  world. The **drive** is the new `Node` world-override: `activate()` sets each bone
  node's `worldOverride_`, which `resolve()` treats as the authoritative
  `composedWorld`, so the existing skinning path (Skin reads `composedWorld`) renders
  the simulated pose; `SceneGraph::applyPhysics` syncs the override each step.
- **glTF authoring ✅:** `extras.Ragdoll` on a skinned node auto-builds a ragdoll from
  its skin's joints (`Mass` / `Radius` / `BoneLength` / `ConeTwist` / `SwingLimit` /
  `TwistLimit`), activated at load and returned to the app to retain.
- **Scope/limits:** ragdoll capsules align to the body local-y (rough collision proxy);
  anchors assume unit-scale bodies; ragdoll→animation *recovery* (blend back) is not
  wired yet — `deactivate()` clears the overrides. Lives in `scene/` (not `physics/`)
  to keep the scene→physics dependency direction (physics stays scene-free).
- **Leaves behind:** the generic constraint row + world-override drive that any later
  articulated system (motors, springs, vehicle joints) reuses.

### P5. Scale — sleeping + islands + dynamic AABB-tree broadphase — ✅ done

- **Simulation islands ✅** (`physics/island.{hpp,cpp}`): union-find partitions the
  movable bodies into connected components linked by movable-movable contacts/joints
  (Static is a boundary; Kinematic is a node so its contacts still solve). The solve is
  now **per island** — `solveAndIntegrate` partitions the contact/joint inputs by island
  and runs the contact + joint solvers over each independently (islands share no bodies
  or constraints, so this is equivalent to the old global solve and sets up future
  parallelism). The warm-start cache lifecycle was split (`beginStore`/`store`/
  `commitStore`) so the per-island solves accumulate into one frame's cache. `GoldenHash`
  unchanged (the partitioned solve is bit-identical).
- **Sleeping ✅:** a Dynamic body below the linear + angular thresholds for `kSleepTime`
  becomes eligible; an **island sleeps** once all its dynamic members are eligible —
  zeroing their velocities and skipping integration + solving until disturbed. Islands
  are rebuilt from current contacts each step, so any awake member (a fresh contact, a
  moving Kinematic rider-platform) wakes the whole island. `PhysicsWorld::wake` /
  `sleeping` / `sleepingEnabled`, per-body `allowSleeping`, and wake-on-mutation
  (`setBodyVelocity`/`setBodyTransform`) round it out.
- **Dynamic AABB tree ✅** (`collision/broad_phase.hpp` interface +
  `collision/dynamic_aabb_tree_broad_phase.{hpp,cpp}`): a fat-AABB BVH (surface-area
  insertion + AVL-rotation balance, à la Box2D's `b2DynamicTree`), leaves sized from the
  swept world bounds + a margin so small motion needs no re-insert. Pairs are
  regenerated each `update()` by querying every leaf and **sorted by collider id** for
  determinism. `SweepAndPruneBroadPhase` was refactored behind the same `BroadPhase`
  interface; `PhysicsWorld` holds a `unique_ptr<BroadPhase>` defaulting to the tree.
  `validate()` cross-checks the pair set against brute force. **Follow-up (post-FC):** the
  broadphase is now genuinely injectable — `PhysicsWorld(unique_ptr<BroadPhase>)` lets a
  caller substitute an implementation (the seam the `unique_ptr` always implied but never
  exposed). `PhysicsWorld::contacts()` sorts pairs into a canonical `(firstId, secondId)`
  order so the order-dependent solve is independent of which broadphase produced them
  (a no-op for the already-id-sorted tree — GoldenHash unchanged), and a new
  `Determinism.BroadphasesAgree` test runs the same scene through the tree + an injected
  SAP and asserts an identical end-state hash — turning the `validate()`-style contract
  into a live cross-implementation correctness check.
- **Scope/limits:** pairs are regenerated in full each step (deterministic + simple) —
  a moved-leaf-only incremental pass is a future optimisation; broadphase/narrowphase
  still run for sleeping bodies (only the *solve* is skipped). A **debug-draw colour for
  asleep colliders** — sleeping bodies dim in the `--debug-physics` overlay — landed with the
  P8 SleepDemo (`PhysicsWorld::debugColliderSleeping()` → `PhysicsDebugData::shapesAsleep`).
- **Leaves behind:** the `BroadPhase` interface (mesh/compound colliders in P6 plug into
  it) and the island partition (the unit any future multithreaded solver parallelises).

### P6. Static mesh + compound colliders — real level geometry — ✅ done

- **Reusable `AabbBvh<T>` core ✅** (`collision/aabb_bvh.hpp`): a generic, payload-
  templated fat-AABB BVH (proxies + insert/remove/move/query) lifted out of P5's
  dynamic tree. `DynamicAabbTreeBroadPhase` was refactored to own an `AabbBvh<Collider*>`
  (behaviour-preserving — the P5 broadphase + determinism tests guard it), and the static
  mesh's triangle BVH reuses the same core. This is the shared spatial structure the
  rendering track's **frustum culling** rides later (FC stays a rendering item; the
  dependency runs P6 → FC, not the reverse).
- **True COM offset ✅:** `PhysicsBody::centerOfMassLocal` (set from the shape/compound
  centroid); `solveAndIntegrate` integrates about the world COM and converts back to the
  transform origin on write-back. Zero for a centred collider, so existing behaviour is
  bit-identical (`GoldenHash` held). An off-centre collider now spins about its real COM.
- **Compound colliders ✅:** `PhysicsWorld::createCompoundCollider` (`CompoundChild` list)
  makes one child collider per primitive at its local offset (each broadphase-registered;
  `ColliderEntry` carries a `localPosition`/`localRotation` composed in `worldShape`), and
  aggregates the children's mass properties — COM by volume-weighted centroid, inertia by
  parallel-axis sum (diagonalised). Per-child contacts flow through the existing pipeline.
- **Static triangle mesh ✅:** `createMeshCollider(StaticMeshShape)` (Static-only) builds a
  per-collider triangle `AabbBvh<int>`; `contacts()` expands a mesh pair into one contact
  per overlapping triangle (triangle-indexed warm-start sub-key). Per-triangle narrowphase
  reuses GJK/EPA, with a **planar-contact reconstruction** for stability: the contact
  normal is snapped to the triangle's CCW face normal and each point's penetration is
  re-measured against the face plane (EPA on a flat triangle gives unreliable
  normals/depths and *edge* normals where a body overhangs a shared edge — which would
  rock a box to a NaN); non-finite / implausibly-deep witness points are dropped. Meshes
  are one-sided (front face only) and assumed to keep a fixed transform.
- **glTF authoring ✅:** `extras.Physics` `Shape: "Mesh"` (static triangle mesh from the
  node geometry) and `Shape: "Compound"` with a `Children` array (per-child shape +
  `Position`/`Rotation` + material).
- **Scope/limits:** compound inertia keeps only the diagonal (exact for compounds
  symmetric about the body axes, approximate otherwise); mesh EPA contact-point quality
  degrades when a triangle is enormous relative to the body (a specialised
  primitive-vs-triangle narrowphase is the future hardening); static meshes don't move.
- **Leaves behind:** the generic `AabbBvh<T>` (frustum culling + future spatial queries
  reuse it) and a working level-geometry collision path.

### P7. Queries + character controller — gameplay surface — ✅ done

Three phases, all behind the existing `PhysicsWorld` boundary, reusing the convex core
(`gjkEpaContact`, `supportPoint`, `worldShape`, `NarrowPhase::collide`):

- **Query API ✅** — `raycast` / `raycastAll` / `shapecast` / `overlapSphere` /
  `overlapShape` with a layer/mask `QueryFilter`. `collision/ray.hpp` (analytic ray vs
  sphere/OBB/capsule/convex + Möller–Trumbore triangle + slab reject) and
  `collision/shape_cast.hpp` (**GJK conservative advancement**, reusing the gap distance +
  normal). Brute-force over active colliders (AABB reject → exact test); mesh colliders
  dispatch into their triangle BVH. Replaces the never-actually-consumed `possiblePairs()`
  access.
- **Trigger + collision events ✅** — a **per-collider `isTrigger` flag** (the roadmap's
  earlier "reserved trigger layer" was never real; a flag is orthogonal to layer/mask, the
  industry model). Trigger pairs route out of the solver (no response); enter/stay/exit
  events are diffed step-to-step from the overlap set (`triggerEvents()` /
  `collisionEvents()`, carrying public collider handles). Authored via glTF
  `extras.Physics` `"IsTrigger": true`.
- **Character controller ✅** — a kinematic-capsule `physics/CharacterController` over the
  queries: collide-and-slide, slope limit, step up/down, sweep+raycast grounding. Headless
  engine class (10 unit tests), driven from `FireEngine::mainLoop` rather than a scene
  component (see **Design reviews** below for why); a `-k` step-pyramid patrol demo.

**Bonus fix:** the query work surfaced that `gjkEpaContact` returned false collisions for
separated *large* boxes (a degenerate-tetra origin-containment FP error); guarded in
`closestOnSimplex`.

**Follow-ups (deferred):**
- **BVH-accelerated spatial queries.** The query API is brute-force O(n); the optimisation
  is a ray/AABB spatial query on the broadphase (the dynamic tree already has
  `AabbBvh::traverse`; the seam is a `BroadPhase` method the tree overrides, SAP
  brute-forces). Deferred so the API ships with behaviour-pinning tests first.
- **GJK edge/corner robustness** — see the dedicated item below; **✅ done** (was a P8 blocker).

### P7.5. GJK edge/corner robustness — foundation hardening — ✅ done

Not a P-series capability — a **correctness-hardening** pass on the P3.5 convex core
(`gjkEpaContact`), the analogue of *bindless materials* on the rendering side: off the
spine, but underpinning P3.5 (narrowphase) → P6 (mesh) → P7 (queries). Surfaced by P7,
whose `shapecast`/`overlap` exercise `gjkEpaContact` with arbitrary shape pairs near
edges — the solver mostly routes primitives through the analytic narrowphase and never hit
it.

- **Diagnosis (corrected from the original "signed-volumes" framing).** The *scale* part was
  already fixed (the committed degenerate-tetra guard in `closestOnSimplex`). The residual
  on box **edges/corners** turned out **not** to be the distance sub-algorithm. Building out
  the full **signed-volumes** sub-algorithm (Montanari et al. 2017) reproduced the bug *and
  regressed* sphere-sphere/sphere-box-face/large-floor — the sub-distance returns the correct
  closest point for the simplex it's handed. The real fault is the **GJK outer loop + support
  tie-breaking on flat features**: on a flat face/edge the box support tie-breaks to a corner,
  the loop adds a vertex that "makes progress" but the sub-distance keeps a stale vertex it
  can't drop, so the simplex stalls and the closest-point *magnitude* `|v|` over-estimates the
  gap (up to ~55%) while spinning to the iteration cap. Signed-volumes was therefore abandoned
  (net regression) in favour of the targeted loop fix.
- **Fix.** At every iteration the loop already has the exact answer in hand: the support
  projection `dot(w, v)/|v|` is the separating-plane distance *through the closest feature* —
  a **lower bound** that equals the gap, where `|v|` is only an upper bound. Track the tightest
  lower bound + its direction, add **duplicate-support termination** (stop when the support
  repeats — the simplex has stalled), and report the lower-bound gap + normal on the separated
  path instead of the stale `|v|`. They coincide on clean convergence; the projection wins only
  on the degenerate tie-break. EPA, the degenerate-tetra guard, `directionalMtv`, and the
  barycentric witness points all stay. `GoldenHash` unchanged (`0x98969fb8…`).
- **Gate (green).** The property/fuzz suite (`tests/collision/test_gjk_epa.cpp`) cross-checks
  `gjkEpaContact` against analytic ground truth (sphere/sphere, sphere/AA-box, capsule/large-
  floor) at scales 0.01→100, asserting **exact** depth + normal for face *and* edge/corner
  features. The former `[!shouldfail]` edge case is now a regular passing case. Full
  convex/narrowphase/compound/mesh/determinism suites + `GoldenHash` all green.
- **Cleanup done:** the `-k` demo dropped the stuck-escape hop band-aid. The patrol now applies
  gravity only while airborne (a grounded character no longer pushes down into the floor each
  frame, which — with the rounded capsule grazing the ground — could wedge it), giving an
  indefinite flat patrol with no freezes.
### P7.6. Character-controller climbing robustness — ✅ done

A follow-up surfaced by P7.5's `-k` demo: the controller wedged when climbing (separate from
the GJK core). Two gaps, both at the demo's low per-frame cadence (3 m/s @ 60 fps = 0.05/frame):

- **Step-up was speed-fragile.** The lift→walk→drop drop was a collide-and-slide; at 0.05/frame
  the capsule centre never reached over a step top in one lifted walk, so the rounded bottom slid
  off the step's top edge back to the floor and oscillated. **Fix:** the drop is now a downward
  **sweep-and-rest** (rest at first contact, no lateral projection) so the capsule rests on the
  step edge and gains height each frame, mounting over a run of frames. A step is accepted only
  when the **lifted walk clears the obstacle** (open space above a step's flat top — a wall or
  slope keeps blocking the raised capsule); gating on that, rather than the ambiguous step-top-
  edge surface normal, both rejects steep slopes and removes a frame-after-frame stutter at the
  riser.
- **Elevated wall/ground corner wedge** (capsule jamming in a wall/platform inner corner) was a
  symptom of the same step-up/slide interaction and is resolved by the same fix.
- **Smoothness pass** (a follow-up round on the live demo): (a) the horizontal collide-and-slide
  now ignores **walkable** ground/edge contacts — only walls block lateral travel — which fixes a
  ~0.6 s stall where the rounded capsule wedged against a step's top edge on the way *down* (it
  touched the edge → zero-advance nudge → the ground snap rested it back on the edge → jitter);
  (b) the patrol turns around on **position bounds** at the flat ends, never mid-stair (a fixed
  timer used to reverse it partway up a climb); (c) `updateCharacter` advances once per render
  frame by the **real frame time** — a fixed-1/60 accumulator was tried first, but above 60 fps
  (e.g. a 120 Hz ProMotion display) it does 0 sim steps on some frames and 2 on others, so the
  ball advanced in uneven chunks (a visible stutter, worst during the slow climb); the controller
  is robust to any per-frame distance, so stepping at the frame rate is smoother and simpler;
  (d) a `move()` output guard clamps any net-**backward** horizontal step — collide-and-slide can
  depenetrate the capsule off a step riser against its travel on a frame where the step-up isn't
  accepted — to a no-progress frame, removing a rare visible step-back.
- **Gate:** `tests/physics/test_character_controller.cpp` gained walk-speed (0.05/frame) coverage
  the old tests lacked (they moved 0.3/frame, masking the bug): `StepsUpLowLedgeAtWalkSpeed` plus
  two headless patrol replays (`PatrolsStepPyramidWithoutWedging`, `PatrolsRampToWalledPlatform…`)
  asserting full course traversal, no permanent freeze, and (pyramid) **no multi-frame stall**
  (`maxStallStreak`). `-k` now walks a step-pyramid course up and down smoothly.

### P8. Demonstration assets — show the work — ✅ done (ragdoll deferred to P9)

Not a new engine capability — a **capstone showcase + manual-validation** deliverable.
Every prior milestone is currently proven only by headless unit tests; this authors a
curated set of scenes that *show* each one running, doubling as a smoke-test gallery and
the reference for how to author physics content.

**Shipped:** 9 self-contained glTF demos in `assets/physics_demos/` (emitted by `generate.py`,
regenerated at build) — FallRest, Restitution, FrictionRamp, Stack, Topple, ConvexHull, Sleep,
StaticMesh, Compound — plus 2 main-loop/flag demos (`-k` character, `-q` query probe). Each has a
headless behaviour test in `tests/physics/test_demos.cpp` (`[Demos]`), loads at 0 `VUID`, and is
indexed in [`README.md`](../README.md) / [`collision.md`](collision.md) / [`onboarding.md`](onboarding.md). **Along the way the demos surfaced real
solver limits** — tall-stack chaotic instability and ragdoll joint-pumping — now driving **P9**
(the Stack demo was capped at 3 boxes — **now 5, re-enabled once P9.2's TGS solver quiesced tall
stacks**; the **ragdoll demo is deferred to P9**, authoring code preserved
in `generate.py`). A small renderer add landed too: the asleep-collider debug colour and a
`DebugLine` query-ray channel.

- Reuses the existing authoring path (glTF `extras.Physics` / `extras.Ragdoll` /
  `extras.Cloth`, plus the CLI demo flags — `-f` floor, `-c` cloth, the P7 `-k`
  character) rather than new code; most scenes are authored glTF + HDR, a few are
  programmatic demo builders behind a flag where authoring can't express them.
- One scene per capability, roughly: a **restitution/friction** ramp (P2), a **stack +
  topple** of boxes resting on a face (P3), a **convex-hull** pile (P3.5), a **ragdoll**
  dropped onto stairs (P4), a **sleeping** stack that settles then wakes on impact (P5),
  a **static-mesh** level surface + a **compound** collider body (P6), and a
  **query/character** playground — ray/overlap probes drawn via the debug overlay and the
  controller walking slopes/steps (P7).
- Verification is the point: each loads clean (0 `VUID`), behaves as labelled, and is
  listed in [`collision.md`](collision.md) as the authoring reference. A short README/onboarding "physics
  demos" section indexes them with the command to run each.

### P9. Solver robustness & modernization — in progress (P9.1 soft joints ✅, P9.2 TGS soft-step ✅, P9.3 friction patches ✅, reduced-coordinate articulations ✅, P9.6 mid-step manifold refresh ✅ — settle snap 10.5°→0.5°)

A grab-bag of solver-quality investigations, **all surfaced by the P8 demos stress-testing
the solver under uncomfortable conditions** (which is exactly what the demos are for — the
visual artefact is secondary to learning how the system reacts). Each changes solver
behaviour, so each needs its own branch, a re-baseline of the determinism `GoldenHash`, and a
green run of the contact/determinism suites. Off the P8 path (P8 demonstrates the physics;
this modifies it) — sequenced **after** P8. Several converge on one modern answer (a
TGS/soft-step solver), so they're grouped here.

**(A) Speculative-contact restitution.** Surfaced by the restitution demo: a **fast** approach is
caught by the speculative-margin CCD as a *gap* contact (`penetration < 0`) and braked to graze
the surface with **no restitution** (`contact_solver.cpp`: the gap branch sets
`normalBias = penetration/dt` and skips the bounce). So a tall drop barely bounces — the demo
works around it by dropping spheres from a modest height to stay in the overlap-restitution
regime.
- **Pragmatic fix (small, localized):** on the gap branch also compute the restitution rebound
  from the approach velocity and target whichever is bouncier —
  `normalBias = restitutionBias > 0 ? restitutionBias : (penetration < 0 ? penetration/dt : 0)`.
  Preserves current behaviour (restitution-0 gap still closes cleanly, resting stacks don't buzz,
  anti-tunnelling intact) and restores bounce on fast impacts. **Trade-off:** a body rebounds
  *within the speculative margin* (≈ `v·dt` early) rather than exactly at the surface — usually
  imperceptible.
- **Fully-correct alternative:** a **Box2D-2.4-style separate "relax"/restitution pass** that
  applies the rebound *after* the position-closing solve, so the bounce happens at the true
  surface with no early-bounce artefact.

**(B) Tall-stack convergence & stability.** Surfaced by the stack demo: with the fixed
**8 velocity / 3 position** iterations, settling time is **chaotic near the solver's stability
margin** and grows steeply with stack height — measured (programmatic, headless): N=3 ~1.1 s,
N=4 ~2.7 s, N=5 ~5 s, **N=6 ~59 s**, and **N=8 diverges** (velocity blows up). It's
non-monotonic (N=7 settles faster than N=6) and **hypersensitive to FP-epsilon** — a 1e-5
per-box offset swings settle time 2×+, which is why the glTF-loaded stack (FP-different from a
programmatically-built one via TRS decomposition) settled in ~25–30 s where the programmatic
one took ~5 s. The stack demo therefore capped at **3 boxes**, comfortably inside the margin. The
"sway" (lateral drift while ringing out) amplifies with height: 0.04 m at N=3 → 0.44 m at N=6.
**✅ Resolved by P9.2 (TGS soft-step):** N=3–8 now all settle and sleep in ~50–130 steps with
millimetre stack error (N=8 sleeps at ~step 105 vs the old solver *diverging*). The StackDemo is
re-enabled to **5 boxes** (`Demos.Stack.SettlesAndStaysStill`) — the substepped solve + relax pass
propagate the load down the stack instead of the bottom box ringing out under the whole tower.
- **Iteration count is the lever** (demonstrated): 24/8 iterations takes N=6 from 59 s → 5 s and
  stops N=8 diverging (→ stable). Cheapest fix, but more CPU per step for every island.
- **Adaptive iteration count** — scale iterations with island size / contact count / measured
  residual, so deep stacks get more passes without taxing simple scenes. Or a **forced-convergence
  pass**: detect non-convergence (residual above a threshold after the normal budget) and run
  extra iterations until it settles.
- **Contact solve ordering / shock propagation** — solve stacks bottom-up so the floor's
  immovability propagates up the stack in one pass; a classic tall-stack stabilizer.

**(B2) Ragdoll (many-joint) non-convergence.** Surfaced by the P8 ragdoll demo: a complex
articulated body (a ~17-bone humanoid, 16 ball-socket/cone-twist joints) **never settles** — it
sits in a **~0.4–1.5 m/s limit cycle forever** (well above the 0.05 sleep threshold, so it never
sleeps). Reproduced headlessly with a clean Y-up skeleton (not a CesiumMan/Z-up quirk; capsules
don't even self-collide — layer-filtered). Crucially, the joints are an **active energy source**,
not just poor dissipation:
- **Iterations help but don't fix it:** 24/8 → ~0.4 m/s (from ~1.5), still no sleep.
- **Damping alone does NOT fix it** (tested): even strong damping (≈3 %/step) leaves ~0.7 m/s;
  damping + 24/8 iterations only reaches a ~0.15–0.5 limit cycle. Damping is a *tool*, not the answer.
- **Gravity is not the source** (tested): with gravity *killed after* the figure lands it keeps
  twitching at ~1 m/s (full=1.7, ×0.1=0.34, off-after-landing≈1.0 indefinitely). So the joints
  pump energy intrinsically, regardless of gravity or contacts.
- **Root cause (precise).** Joint position error is corrected as **real velocity** via hard
  Baumgarte (`kJointBaumgarte = 0.2`; `joint_solver.cpp`: `bias = -(kJointBaumgarte/dt)·C`), and
  joints — unlike contacts — have **no split-position / pseudo-velocity pass** (`physics_world.cpp`):
  the bias integrates straight into real motion. So any joint row the 8 Gauss-Seidel sweeps leave
  unsatisfied is turned back into velocity next frame — a structural energy *source* (exactly why
  killing gravity doesn't help). **Secondary suspect:** the ball-socket point constraint is solved
  as **three independent scalar rows** on world X/Y/Z with a scalar diagonal effective mass
  (`joint_solver.cpp`), not a **3×3 block** — for off-centre anchors the translation↔rotation
  coupling is a block problem, so scalar rows are weaker and can inject angular ringing in chains.
  The corollary: **you cannot rely on the solver converging a complex joint network**, which is
  *why* the stability state machine below (force-settle on a detected limit cycle) is a *backstop*,
  not the primary fix — that's the (C) solver work.
- **The maximal-coordinate ragdoll demo was RETIRED** (deleted: `build_ragdoll_demo` +
  `RagdollDemo.gltf` + its tests). Its soft joints drift and then hard-snap on sleep — an inherent
  maximal-coordinate trade-off, not a bug — which is exactly what the reduced-coordinate
  articulation (item 5) avoids (zero joint error by construction). The real skinned **CesiumMan**
  now falls and settles through the articulated path (`CesiumManRagdoll.gltf`); it supersedes the
  synthetic maximal demo.
- **The promising direction — a body/island stability state machine** (extends the existing
  awake/asleep states). The trigger is detecting the "stuck" state: energy low-ish but **not
  decreasing** for N steps — a limit cycle the energy-threshold sleep misses. From there a body or
  island can take one of several responses depending on the scenario:
  - **FORCE-SETTLE** — ramp damping / extra iterations to drive the residual down; once "close
    enough" and clearly not improving, snap-to-sleep to guarantee rest. The blunt, reliable
    backstop.
  - **RELAX-CONSTRAINTS** — when the system is wedged in a *local minimum* (constraints fighting
    each other so the solver can't make progress), temporarily relax/soften the joint constraints
    (lower `kJointBaumgarte`, widen limits, drop constraint stiffness) to let it slide out of the
    minimum toward a solvable configuration, then re-tighten. Complements the author's side of the
    bargain: **ragdoll constraints should be authored to be easy to solve** (sane masses, limits,
    bone proportions) so the solver isn't handed an over-constrained rig in the first place.
  - **BLEND-TO-SLEEP** — instead of stopping a body dead (which can look like a hitch), **blend the
    current joint motion out** over a short window until it stops — a pure *animation* of the
    existing velocities decaying to zero (no constraints, no IK solve), purely for a nicer-looking
    settle. The visual finisher on top of FORCE-SETTLE.
  - These pair with softer joints / soft constraints (less energy to fight) and the (C) solver
    below. This state machine is the pragmatic robustness layer: it *guarantees* rest where the
    solver alone cannot.

**(C) Solver modernization — the real fixes (not "more Gauss-Seidel + Baumgarte").** The point
fixes above are stopgaps; the durable answer changes the solver. The modern-engine pattern (PhysX
5, Box2D v3) is *not* to keep adding PGS iterations + hard Baumgarte — it's soft constraints +
substepping + (for skeletons) reduced coordinates. In rough priority/order for this engine:

1. **Soft / compliant joint constraints (CFM or XPBD)** in place of hard Baumgarte — **✅ done
   (P9.1)**. The joint solver now uses a Box2D-v3 soft constraint (`b2MakeSoft`: `kJointHertz = 8`,
   `kJointDampingRatio = 5`; `joint_solver.cpp` — `bias = biasRate·C` + `massScale`/`impulseScale`,
   the `impulseScale·impulse` decay being the dissipative anti-pump term). **The structural energy
   pump is gone**: a representative 17-bone humanoid ragdoll now **settles to rest** (test
   `Ragdoll.HumanoidSettlesOnFloor`) where the old hard-Baumgarte solver left a 1.3 m/s limit cycle
   forever — and all `[Joint]` tests + `GoldenHash` (no joints in that scene) still pass.
   **But** at a *single* timestep there's a real **stiffness ↔ settling tradeoff**: low hertz settles
   the ragdoll but makes joints too soft (fails the anchor tests); high hertz keeps joints rigid but
   the settling basin is **narrow and skeleton-chaotic** — hz=8/ζ=5 settles the clean test humanoid
   but **not** the real CesiumMan skeleton (it oscillates 0.6–2.3 m/s). This was the signal that
   substepping was needed for robust *stiff-and-settled* — see item 2 (P9.2), which delivered TGS
   substepping and settled the synthetic humanoid, though CesiumMan still needs item 5.
2. **Substepping / TGS (temporal Gauss-Seidel) for joints and contacts** — **✅ done (P9.2).** The
   fixed step is split into `kSubstepCount = 8` substeps of `h = dt/8`; collision detection runs
   once per step, then per substep: integrate gravity → warm-start → **bias** solve (joints +
   contacts, separation/error recomputed from the moving pose) → integrate positions → **relax**
   solve (no bias). The relax pass removes the soft bias velocity each substep — the dissipation
   that a first, *whole-step* substepping attempt lacked (re-running collision+solve N× was no
   better than single-step; that result + the relax-pass insight are why this is true TGS soft-step,
   not whole-step). Contacts also became **soft** (`b2MakeSoft`, `kContactHertz = 90` /
   `kContactDampingRatio = 10`) replacing the split-impulse position pass for dynamics; split-impulse
   is kept *kinematic-only* (scene-driven bodies still slide out of static penetration). Restitution
   is now a proper end-of-step pass at the **true impact velocity** (subsumes (A)); the default
   material restitution was flipped `1.0 → 0.0` (Box2D/PhysX convention — a bouncy default that the
   old speculative-suppression masked was a footgun). Stacks settle and sleep cleanly (subsumes (B)).
   **Outcome on the gate:** the synthetic 17-bone humanoid (`Ragdoll.HumanoidSettlesOnFloor`) now
   settles to **0 m/s** — the key levers were **enough substeps (8)** *and* **stiff contacts (90 Hz)**:
   an overly compliant floor contact let the limbs bob and pumped that motion back through the joints.
   **But the real CesiumMan skeleton still does not settle** — it sits in a ~0.5–1 m/s limit cycle
   even at 16 substeps (a larger joint graph; chaotic basin). So the **P8 ragdoll demo stays deferred**
   (`build_ragdoll_demo` still re-commented in `generate.py`). That is the signal that maximal-coordinate
   PGS — even substepped + soft — is not enough for a full skeleton; the durable answer is item 5
   (reduced-coordinate articulations).

2.5. **2D Coulomb friction (coupled 2×2 mass + disk clamp) — ✅ done (P9.3).** *Resolves* the
   convex-tetra friction artifacts the P9.2 contact rewrite surfaced in `ConvexHullDemo`/`FrictionRampDemo`:
   (a) a tetra landing on an **edge** then **tipping onto a face** spun up about the vertical and even
   *flipped onto another face from rest* (energy injected — non-physical); (b) a box could be flung off a
   friction ramp. The real fix turned out to be the **friction model**, not warm-start tweaks: friction is
   now solved as a **coupled 2-vector** over the contact tangent plane against a **symmetric 2×2 effective
   mass** (which captures the two tangents' angular cross-coupling at an edge/vertex) and clamped to the
   friction **disk** `|λ| ≤ μ·N` (a circle, not an independent-axis box). The old per-axis box clamp +
   scalar masses over-budgeted diagonally and mis-distributed torque — that was the pump. Plus: friction
   has **no cross-frame memory** — the tangent impulse is re-derived from zero each step (the within-step
   per-substep warm-start still converges it); replaying it across frames fed the per-substep warm-start at
   a rocking contact and pumped energy (`|ω|` 0.6→81 rad/s flinging the ramp box; the edge-rock flip). Only
   the *normal* impulse is warm-started across frames (its one-sided clamp self-corrects).
   **Two approaches were tried and rejected on the way:** (i) Layer-1 warm-start count tweaks (substep-0-only
   tangent warm-start) — diverged kinetic friction (ramp box through the floor); (ii) a full **positional
   static-friction anchor / stick-slide** subsystem (body-local material anchors, a soft drift spring,
   per-substep mode transitions) — *implemented and removed*: per-point sticking anchors on a 2-point edge
   formed a torsional couple that **pumped** and flipped settled convex bodies, and it didn't even help the
   yaw (velocity-only + 2×2 + disk gave a *lower* yaw). The 2×2-mass + disk-clamp + no-friction-memory model
   is simpler and strictly better. Result: the tipping tetra settles (no flip, no pump), the ramp holds, and
   every friction/stack/ragdoll/mesh test stays green; `GoldenHash` untouched (its scene is frictionless).
   Guarded by `PhysicsWorld.SettledTetrahedronDoesNotFlipToAnotherFace`,
   `PhysicsWorld.TippingTetrahedronDoesNotSpinUpAboutVertical`, `ContactSolver.FrictionDiskClampLimits…`.
   **Deferred:** a true **per-manifold patch clamp** (sum the normal load over a manifold, solve one shared
   2D friction impulse) — per-point disks can in principle still over-budget on multi-point face contacts,
   though no case currently needs it.
   **Separately noted (pre-existing, *not* a friction issue):** a few violent tumble orientations of the
   tetra sink through the floor and never settle even frictionless — a convex narrowphase / deep-penetration
   instability present already in P9.2 (the once-per-step manifold misses a fast vertex until it's deeply
   embedded), tracked for the (A)/CCD + narrowphase work, not P9.3 — now item 6 (P9.6).
   **❌ P9.5 attempted & reverted — an *angular* speculative-CCD margin does NOT fix rotational tunnelling.**
   Hypothesis (from the once-per-step CCD's linear-only motion expansion): a spinning collider's corner
   moves at `|ω|·r` tangentially, so the swept-AABB expansion and the speculative `closingReach` undercount
   a fast spinner and it tunnels. Implemented an `angularReach = |ω|·angularRadius` term (per-collider
   `angularRadius` = max COM-to-scaled-corner distance, cached on `ColliderEntry`) and added it to the
   swept-bound isotropic expansion + both narrowphase `closingReach` sites. **It changed essentially
   nothing** — across ~45 spin/drop/shape configs, pre- vs post-fix `minY` was bit-identical on the
   boundary cases and the violent tunnels (`minY ≈ −75…−96`) stayed tunnelled. **Root cause of the
   non-result:** a fast spinner's `previous→current` swept AABB *already* spans its full rotational arc
   (at 10 rad/s ≈ 0.17 rad/step the two poses merge to cover the swing), so broadphase **already pairs** it
   with the floor — pairing was never the bottleneck. The real failure is **narrowphase**: the speculative
   manifold is built **once at the start pose**, and the solver can't track the contact as the body keeps
   rotating through the surface mid-step; a larger margin can't fix a stale manifold. The angular term only
   bites in a razor-thin band (body wholly above the floor at step start, a corner grazing down within one
   step), with no demonstrable win — so the change (and its `GoldenHash` re-baseline) was reverted whole.
   **Also corrected:** the earlier "`(0.3,1,1)` family sinks" claim was config-specific and stale — that
   exact drop now settles both before and after, so it is *not* a CCD repro. **The genuine fix is sub-step
   narrowphase / conservative advancement** (re-detect the manifold per substep, or root-find time-of-impact
   under rotation), a substantially larger piece than a margin tweak — folded into the (A)/CCD + narrowphase
   track, not the speculative-margin path. Now a first-class item: see item 6 (P9.6).
   **🟡 Convex contact-offset + proximity-ramp friction (branch `convex-contact-quality`) — PARTIAL.**
   Targeting the `ConvexHullDemo` red-tetra "rapid Y-spin", added (1) a **contact offset** (`kContactOffset`
   in `contact_manifold.hpp`, collision layer, distinct from `kSpeculativeDistance`): `clipFaceManifold`
   keeps clipped points within the offset *above* the reference face, recording the true signed gap as a
   **negative penetration**, so a near-flat landing forms a small support patch instead of one vertex; and
   (2) a **proximity-ramp on friction** (`contact_solver.cpp`): a contact's friction budget scales
   `clamp(1 − separation/kContactOffset, 0, 1)` — zero at the outer gap, full at real contact — so a
   speculative gap point brakes but never acts as a full tangential motor. **Offset-alone (full friction on
   gap points) was a regression** — the gap points became a tangential motor that spun the tetra a full turn
   and *walked* it across the floor; the proximity ramp fixed the walking. **But the headline 180° Y-spin
   remains:** it is the genuine **steep vertex-first** landing (a single-point support the offset can't reach
   — the neighbours are too far above the floor), which no contact-quality lever resolves. Confirmed the
   walking is *inherent* to a tumbling drop (baseline slides ~0.35 m too, with worse chaotic spikes the ramp
   removes). **Lesson: contact offset + ramped friction de-sensitises near-flat landings and is more-correct
   friction (a not-yet-touching point shouldn't carry full Coulomb), but the steep-vertex spin needs the
   sub-step narrowphase above.** Suite green, GoldenHash untouched (its scene is frictionless → ramp inert),
   render smoke 0 VUID. **Key debugging caveat: the headless direct-value repro is only *partially* faithful
   to the glTF-loaded app** — it reproduces the walking but not the 180° spin (TRS-decompose FP-epsilon puts
   the app in a different chaotic microstate), so the perturbation sweep judges *relative* de-sensitisation,
   not the exact app symptom; the user's visual verification is ground truth.
3. **Block-solve the ball-socket anchor** as one 3D constraint with a 3×3 effective mass, not three
   scalar rows — kills the anchor jitter / angular pumping from the secondary suspect above.
   **❌ Tried (P9.4) and reverted — it does NOT help ragdoll chains, it breaks them.** Implemented a 3×3
   `PointBlock` (coupled effective mass `K`, `Mat3::inverse`, per-substep anchor tracking, soft bias +
   relax + warm-start) replacing the 3 scalar rows. It passed every `[Joint]` test and the synthetic
   17-bone gate, but the **real 19-bone CesiumMan ragdoll diverged to NaN at the floor impact** (the block
   impulse ran away to ~1e21). The K was well-conditioned (det 15–1949, `maxInvK ≈ 0.5`), so it isn't the
   inverse — it's the classic failure: solving each joint *exactly* in a **maximal-coordinate chain**
   over-corrects and diverges the Gauss-Seidel at a violent impact, where the scalar rows' partial solves
   stay stable. Under-relaxation (SOR factor 0.5) only traded the NaN for a 26 rad/s spin that sank through
   the floor. **Lesson: a per-joint block solve is the wrong lever for chains — it can't fix (and worsens)
   the maximal-coordinate instability; the durable answer is item 5 (reduced coordinates).** Kept the
   general-purpose `Mat3::inverse()` (it'll be needed there); reverted the rest. The ball-socket stays
   three soft scalar rows.
4. **A joint positional-correction path that adds no kinetic energy** — a joint split-position pass
   (contacts already have one) or XPBD projection, so "fix the pose" is separate from "add velocity."
5. **Reduced-coordinate articulations (the strongest answer for ragdolls).** Represent the skeleton
   as a tree of *parent pose + joint coordinates* rather than independent rigid bodies glued by
   constraints afterward — joint error is then **zero by construction**, and high mass ratios behave.
   PhysX models ragdolls this way (articulations). Keep ordinary maximal-coordinate rigid-body
   joints for breakable / looped / general constraints.
   **🚧 In progress (branch `reduced-coordinate-articulations`), mirroring Bullet's `btMultiBody`:**
   the real deliverable is **not "add ABA" but "add an articulated constraint-body type to the
   contact solver"** — a `ConstraintBody` seam (velocityAt / effectiveMassAlong / applyImpulse) the
   contact/friction/limit rows talk to, with a rigid impl (today's flat `SolverBody`) and an
   articulation-link impl (impulse mapped through the articulated response into q̇ + floating-base
   velocity). Phasing: **A** foundations (spatial algebra, floating-base FK, link colliders as a
   collider-owner variant in the broadphase) → **B** ABA free dynamics (fixed + floating revolute
   chains) → **C** the seam, minimal (one contact row through the articulated effective mass — the
   go/no-go) → **D** full manifolds + friction through the seam → **E** spherical joints + cone-twist
   limits + passive drives → **F** `Ragdoll` integration + the CesiumMan settle gate + re-enable the
   P8 ragdoll demo. Revolute before spherical (no quaternion/singularity cost while proving the
   architecture); floating base is first-class from A (ragdolls aren't fixed-base arms).
   **✅ Phase A done:** `physics/spatial.hpp` (`RigidTransform`, `SpatialVector`),
   `physics/articulation.hpp/.cpp` (link/joint model, floating base, generalized q/q̇, forward
   kinematics), and the collider-owner change — `ColliderEntry` owner is now a body *or* an
   articulation link (`PhysicsWorld::createArticulation` / `attachLinkCollider`), resolved uniformly
   through `colliderOwnerPose` so link colliders track their FK pose, pair in the broadphase, and
   answer spatial queries. Link-collider contact *pairs* are skipped by the existing null-body guard
   (no rigid response yet — that's the Phase C seam). Tests in `tests/physics/test_articulation.cpp`;
   `Mat3::inverse` (kept from P9.4) will be used in the dynamics phases.
   **✅ Phase B done (branch `articulation-aba-dynamics`): Featherstone ABA free dynamics
   (fixed-base, revolute + fixed joints).** `spatial.hpp` gained the spatial algebra —
   `crossMotion`/`crossForce`, a block-3×3 `SpatialMatrix`, the Plücker `motionTransform`/
   `forceTransform`, and rigid `spatialInertia` (+ `Mat3` add/sub/scalar-mul/`skew`); `Articulation`
   gained `computeAccelerations(gravity, jointDamping)` (the 3-pass ABA: outward velocities/bias →
   inward articulated inertia → outward accelerations) and semi-implicit `integrate(dt)`. Validated
   hard: single-pendulum release accel matches the analytic −mgL/(I_com+mL²) to 3 sig figs;
   single-pendulum energy conserves (0.4%→0.005% as dt shrinks); the inter-link coupling (the
   articulated-inertia transform, which the single pendulum never exercises since its parent is the
   *fixed* root) is stable under zero gravity; a hanging double pendulum oscillates stably.
   **Key finding for Phase F:** a *chaotic* gravity-driven double pendulum **diverges** under the
   non-symplectic explicit integrator at the physics substep rate (h = 1/480) when undamped — a known
   limitation of explicit integration of chaotic articulations, **not** an ABA fault (all the
   energy-conservation checks pass). **Passive joint damping ≥ ~0.2 stabilises it**; a robust ragdoll
   (Phase F) needs adequate joint damping + limits (Phase E) + contact dissipation, or an
   implicit/stabilised integrator. Assumption to revisit: the ABA takes the revolute axis through the
   child link origin (jointToChild a pure rotation); geometry goes in parentToJoint + comLocal.
   **✅ Phase C done — the ConstraintBody seam PROVEN (branch `articulation-constraint-seam`), the
   go/no-go.** `Articulation` gained the articulated impulse response: `factorizeArticulatedInertia()`
   caches the ABA inertia factorization (Iᴬ/U/D/transforms, geometry+mass only), and
   `inverseEffectiveMass(link, point, dir)` + `applyImpulse(link, point, impulse)` +
   `pointVelocity`/`computeLinkVelocities` let the contact solver treat a link like any constraint
   body — read a contact point's velocity, ask its operational-space effective mass, and apply an
   impulse that propagates through the whole chain into q̇. Validated: single-pendulum effective mass
   = L²/I_pivot and Δq̇ = P·L/I_pivot to 3 sig figs; **and the decisive test — one impulse
   `J = −vₙ/invEffMass` drives a *two*-link chain tip's constraint velocity to exactly zero**, i.e. a
   contact solved through the inter-link coupling. (Sign gotcha fixed: the impulse enters as bias
   pᴬ = −impulse, matching the accel-ABA's `−f_ext`; first cut gave negative effective mass.) Suite
   green (14605), 0 VUID. **The seam lands — reduced-coordinate contacts are viable.**
   **🟡 Phase D core done (branch `articulation-contact-solver`): the `ConstraintBody` seam + a
   contact solve that rests an articulated link on a floor.** `constraint_body.hpp` is the seam the
   plan named — a `ConstraintBody` (static | rigid `SolverBody` | articulation link) exposing
   `velocityAt` / `inverseEffectiveMassAlong` / `applyImpulse`; the rigid path is behaviour-preserving,
   the link path routes through the Phase-C articulated response. `articulation_contact.{hpp,cpp}`
   (`stepArticulationOnPlanes`) advances a fixed-base articulation one step under gravity + damping and
   resolves link-vs-static-plane contacts (soft non-penetration bias + velocity-only Coulomb friction,
   cone-clamped) over the TGS substep loop through the seam. **Gate met:** a fixed-base pendulum swings
   onto a floor plane and **rests exactly at the surface, no tunnel** (test
   `Articulation.LinkRestsOnFloorThroughConstraintBody`). Rigid `ContactSolver` untouched → GoldenHash
   unchanged; suite green (14608), 0 VUID. **Bug caught: the normal solve braked the bob mid-air —
   gate on `separation ≤ kSpeculativeDistance`.** Friction path runs + cone-clamps but a 1-DOF pendulum
   has no sustained slip to exercise it (real sliding friction comes with multi-link / the ragdoll).
   **Remaining in D:** wire link colliders into the real `contacts()` pipeline (un-skip the null-body
   guard, generate manifolds via `worldShape`, feed the seam) so contacts flow automatically instead of
   via hand-fed planes; mixed rigid-vs-link and link-vs-link. Then the **floating base** (6×6 base solve
   → needs a `SpatialMatrix` inverse; the current response is fixed-base) before Phase E/F.
   **✅ Phase E1 done (branch `articulation-spherical-limits`): spherical joints + multi-DOF ABA.** Added
   a 3-DOF `Spherical` joint (a ball joint: hip/shoulder) whose state is a quaternion (`jointRotation`)
   with its three q̇ as the joint-frame angular velocity. Generalised the whole ABA — `computeAccelerations`,
   `factorizeArticulatedInertia`, `impulseResponse` — from rank-1 (scalar D) to rank-n via a small
   `invertDof` (scalar reciprocal for revolute, `Mat3::inverse` for spherical), unifying the factorization
   so all three methods share it. Revolute behaviour bit-preserved (all prior tests green). Validated:
   spherical release accel and in-plane trajectory match the revolute joint exactly; **out-of-plane
   precession conserves energy (0.04%).** **Bug caught: the joint quaternion must right-multiply (R·Δ) —
   q̇ is the *body*-frame rate; `Quaternion::integrate` left-multiplies (world frame), which pumped 70%
   energy out-of-plane (in-plane didn't catch it, ω∥axis commutes).** `Mat3` gained `+`/`−`/scalar-`*`/
   `skew` (Phase B) and now drives the spherical D⁻¹. Suite green (14613), 0 VUID.
   **✅ Phase E2 done (same branch): cone-twist limits on the spherical joint.** A passive restoring
   *generalized torque* (not a separate constraint row): past the swing cone (`swingLimit` about
   `jointAxis`) or the ±`twistLimit`, a spring (`limitStiffness`·excess) + damping (`limitDamping`)
   pushes the joint back inside; zero within range. Injected into the ABA joint torque `τ` (reduced
   coords — no extra solver), using the same swing-twist decomposition as the maximal-coord
   `JointSolver`. Validated: a gravity-loaded spherical rod held near a 0.5-rad cone (vs a free joint
   swinging far past); the soft-limit overshoot tightens monotonically with stiffness (k=40→0.69,
   k=800→0.51) and stays NaN-free; a twist-axis spin arrests at ±twistLimit; within-cone motion is
   untouched. Fits the B stability finding (passive, moderate stiffness — a rigid limit would ring
   under explicit Euler). Suite green (14618), 0 VUID.
   **✅ Phase E3 done (same branch) — passive drives; Phase E complete.** A per-joint drive spring
   toward a target pose (a muscle / rest-pose bias), active when `driveStiffness > 0`:
   τ = driveStiffness·(target − q) − driveDamping·q̇. Revolute uses a scalar `driveTarget`; spherical a
   `driveTargetRotation` with the body-frame orientation error (2·log of q⁻¹·q_target). Injected into
   the ABA `τ` alongside damping + limits. Validated: a revolute drive holds a gravity-loaded rod near
   its target (undriven it falls to hang); a spherical drive poses the joint exactly to a target
   orientation. Suite green (14621), 0 VUID. **Phase E (spherical + cone-twist limits + drives) done.**
   **✅ Floating base done — free 6-DOF root.** `SpatialMatrix::inverse` (3×3 Schur-complement blocks),
   base velocity/accel state (`baseVel_`), and the base solve `a₀ = −Iᴬ₀⁻¹·pᴬ₀` wired through
   `computeAccelerations` / `impulseResponse` / `computeLinkVelocities` / `integrate` (base pose advances
   by its body-frame spatial velocity). **Bug found & fixed: `spatialOuter` built `w·uᵀ` instead of
   `u·wᵀ`, transposing the off-diagonal blocks of the `U·D⁻¹·Uᵀ` articulated-inertia update.** Invisible
   for centred-COM links (planar/fixed-base tests all passed), but it corrupted the articulated inertia of
   any **off-centre-COM** link folded into the free root — a free body fell ~4.4× too fast. Now the COM
   free-falls at exactly g (regression test `Articulation.FloatingBaseConservesLinearMomentum`).
   **Two suspected "velocity-product" momentum bugs turned out to be measurement artifacts, not code**:
   spinning about an offset origin gives the COM an initial velocity (a wrong free-fall baseline), and a
   second-difference "acceleration" of ~10 m positions at dt² ≈ 2.5e-7 is below float32 precision — the
   robust first-difference velocity-change check reads −9.8100. Suite green (29414), 0 VUID.
   **✅ Phase F1 done — articulation contacts through real geometry, own solve pass.**
   `PhysicsWorld::stepArticulations` advances every articulation one fixed step (gravity + joint
   damping) and resolves link-collider-vs-**static**-rigid contacts through the `ConstraintBody` seam:
   broadphase pairs → real capsule/box manifolds (`worldShape` + narrowphase) → per-link plane contacts
   tracked in link-local space across the TGS substep loop. Separate from the rigid islands (link-vs-
   dynamic and self-collision deferred, per the agreed option 1). **Gate met** — a free-floating
   spherical-joint capsule chain with cone-twist limits drops, flops, and settles flat on a box floor
   (`Articulation.SphericalChainSettlesOnFloor`). Suite green (29418), 0 VUID.
   **Two bugs fixed en route:** (1) the contact push-out used a *hard* Baumgarte (`−sep·kBaumgarte/h`),
   which at the substep rate launched the chain to +2.5 m — replaced with the rigid solver's **soft
   contact** (b2MakeSoft + `kMaxBiasVelocity` cap + speculative brake). (2) **`computeLinkVelocities`
   silently dropped a *spherical* joint's angular velocity** (only Revolute was handled; Phase C/D
   tested revolute only) — so `pointVelocity` was wrong for spherical links and the contact solve read
   garbage. Verified the fix against finite-difference. **Solver-structure lesson:** the contact sweep
   refreshes link velocities before *each* contact (sequential Gauss-Seidel) — a single stale snapshot
   makes coupled contacts over-correct; but *iterating* the soft bias without a warm-start step pumps
   energy, so it's one sweep per substep, convergence from the substep cadence. **Also: spawning a body
   already interpenetrating a static collider isn't recovered** (deep-penetration is a separate problem)
   — normal drops are fine.
   **✅ Phase F2 done — articulated `Ragdoll` binding; the humanoid settles.** `Ragdoll::makeArticulated`
   builds one `Articulation` from a bone tree: the single root bone becomes the floating-base pelvis,
   every child bone a spherical joint seeded from the bind pose (relative parent→child transform so
   FK at q = 0 reproduces the skeleton), a capsule link collider per bone, and uniform cone-twist limits
   + damping from `RagdollParams` (per-joint authoring layers on top later, per the agreed design).
   Bones are topologically ordered (BFS from the root) so `addLink` always sees its parent; capsule
   inertia is a rod-plus-radius approximation floored against degeneracy. `activate()` drives the bone
   nodes' world-overrides from the link FK transforms (vs the bodies, for a maximal ragdoll). The F1
   `stepArticulations` pass drives it automatically. **Gate met** — the generated **two-joint** and the
   full **17-bone branching humanoid** both fall and settle on the floor through the reduced-coordinate
   path, no tunnelling, no limit-cycle (`Ragdoll.ArticulatedTwoJointDemoSettlesOnFloor`,
   `Ragdoll.ArticulatedHumanoidSettlesOnFloor`). That is the CesiumMan mechanism at demo scale. Suite
   green (29471), 0 VUID. The maximal-coordinate ragdoll (1/2-joint + full demos) stays a separate track.
   **✅ Phase F3 done — the real CesiumMan skeleton settles, with cone-twist limits on.** The F2 mechanism
   held for the synthetic humanoid but the *real* 19-bone CesiumMan (much smaller bones → tiny link
   inertias) exposed two things the explicit solver couldn't handle: (a) the explicit cone-twist limit
   *spring* produced ~10⁴–10⁷ rad/s² joint accelerations on a low-inertia bone and blew up; (b) the
   single-sweep contact solve under-converged the stiff, many-contact collapse. The **"robust solve"**
   (branch `articulation-robust-solve`) fixed both, three parts: **(1) Unified velocity model** —
   `impulseResponse` folds every per-link `dv[i]` into the cached `linkVelWorld_` on commit, so the
   contact solve can *iterate* without the drifting `computeLinkVelocities` refresh (the classical
   transport and the Plücker impulse response are two propagations that diverge under repeated
   intra-substep refresh — the real reason single-sweep was forced). **(2) Iterated TGS contact solve** —
   `integrate` split into `integrateVelocities`/`integratePositions`; per substep = velocities → one
   *biased* sweep (soft push-out) → integrate positions → `kArticulationVelocityIterations−1` *relax*
   sweeps (bias 0). **(3) Velocity-level joint limits** — the explicit limit spring replaced by
   `solveJointLimits`: a unilateral velocity projection through a new generalized `jointImpulseResponse`
   (impulse on a joint's DOFs, floating base recoils, conserves linear momentum — tested), bounded
   Baumgarte push-back, solved *inside* the contact sweeps so limits + contacts converge together. No
   stiff spring → stable at any bone inertia. Gate met headlessly — the real CesiumMan skin bound via
   `makeArticulated` falls and settles (`Ragdoll.ArticulatedCesiumManSkeletonSettles`; baseY steady, no
   NaN /1500 steps). App path: a per-frame `Ragdoll::syncNodes` (articulated bones aren't body-bound, so
   `applyPhysics` doesn't reach them) pushes link FK → bone `worldOverride`; the loader builds articulated
   ragdolls from `extras.Ragdoll.Articulated`; `addFloorPlane` gained a physics collider; CesiumMan.gltf
   tagged + raised so it drops onto the floor (`./fireEngineApp CesiumMan/CesiumMan.gltf skybox.hdr -f` —
   builds a 19-bone articulated ragdoll, 0 VUID). Suite green (31010). The dead `limitStiffness`/
   `limitDamping` desc fields (velocity-level limits ignore them) were removed. **Plausibility polish
   (the (C) track):** ✅ the post-settle **arm drift** (a limb curling for a second or two after the
   body rests) and ✅ the settled **residual yaw** (the whole body slowly rotating about the vertical,
   which contact friction doesn't resist) are both fixed on branch `ragdoll-joint-settle` — settle
   assists in `Articulation::integrateVelocities` that mirror the base's linear one (a per-DOF joint
   settle decay for slow-moving joints; a yaw-only base decay projected onto world-up). Still open:
   per-joint limit authoring (hinges for knees/elbows). The maximal P8 ragdoll demo was retired
   (deleted), superseded by the articulated CesiumMan.

6. **Sub-step narrowphase — manifold refresh under fast rotation (P9.6) — ✅ Stage 1 done
   (branch `substep-narrowphase-refresh`); Stage 2 stays deferred, gate met with 20× margin.**
   **Stage-1 result:** the gated mid-step manifold refresh collapses the settle snap from
   **10.51°/step → 0.53°/step** (tetra 1: 0.30°, tetra 2: 0.002°) — far beyond the ≤3° gate — and
   the tetra also comes to rest *faster* (rest by step ~59 vs ~63) in the same final pose. Suite +
   [slow] + layering CTest green, 0 VUID on the demo. **Implementation:** (1)
   `ContactSolver::refresh(bodies, contacts, refreshed)` — row *splice*: unflagged contacts' rows
   copied verbatim (re-preparing a stale manifold at an advanced pose would corrupt separations),
   flagged rows rebuilt at the current pose; matched points (same pair, nearest *tracked current*
   anchor within kWarmStartMatchRadius — the prepare-time point is rotation-stale by exactly the
   sweep being corrected) inherit normalImpulse + relVelN0 + maxNormalImpulse (restitution still
   sees the true approach speed and its engagement flag); friction not carried (cross-frame
   convention); world inverse inertias recomputed (equally stale). (2) `worldShapeAt(entry,
   OwnerPose)` factored from `worldShape` so shapes compose at *hypothetical* poses. (3)
   `PhysicsWorld::refreshIslandContacts`, called at substep kSubstepCount/2: gate =
   Dynamic && |ω|·dt > `kSubstepRefreshRotation` (0.03 rad ⇒ ≥ ~1.8 rad/s — 0.05 would MISS the
   2.9 rad/s motivating case) on the *current solver ω* (the impact generates the spin);
   non-mesh (subKey 0) pairs only; owner pose rebuilt from SolverBody exactly as the writeback
   (origin = COM − R·comLocal); speculative reach sized for the *remaining* step; empty re-collide
   drops the rows. Deterministic (index-order iteration; bit-identical replay asserted).
   **Gates:** `Demos.ConvexHull.TetraSettleTwistBounded` (settle twist < 3°/step on the faithful
   gltf replay, pile sleeps, bit-identical replay — the scene provably trips the refresh);
   4 `ContactSolver.Refresh*` unit tests; hidden `[ConvexProbe]` trace kept for diagnostics.
   **GoldenHash intentionally re-baselined** (the 3-box drop trips the gate at impact): macOS
   0x2b31386354735d8b; the Linux/x86_64 golden is STALE pending a run of the local Docker CI
   replica (per-platform goldens must be recorded on their platform).
   **Collateral claims VERIFIED by an A/B tunnelling sweep** (27 configs: box(0.3,1,1) / thin
   plate / tetra hull × three spin axes × 10/20/40 rad/s, dropped onto a wide floor; hidden
   `[TunnelSweep]` probe prints the table; A/B by temporarily disabling the gate constant):
   gate OFF = **5/27 tunnel** (minY −5…−45, the P9.5-era failures), refresh ON = **2/27** — every
   ≤20 rad/s config now rests on the surface, and the (0.3,1,1) "sinking family" settles cleanly
   (finalY 0.300). Locked in as `Demos.RotationalTunnellingBoundedTo20RadPerSec` (18 ≤20 rad/s
   configs must never tunnel). **The two survivors are 40 rad/s (~380 RPM) flat-box spinners** —
   at 38°/step, even the mid-step refresh is ~19° stale — which quantifies the Stage-2 boundary.
   **Stage 2 (per-substep re-detection / rotational TOI) stays deferred**: the known failure
   family below 20 rad/s is retired; revisit only if gameplay needs >20 rad/s spinners resting
   on floors. Original evidence and design follow.**
   Promoted from prose in the (A)/CCD + narrowphase track now that the failure is quantified and
   reproducible. **The evidence (ConvexHullDemo Tetra0, the orange tetra's settle "snap"):** its final
   tip-onto-face is a *ballistic face-slam* — it loses all contacts for ~10 steps while tipping
   (tumbling at ~2.9 rad/s), then impacts in ONE step where twist decomposition shows **−10.5° of true
   world-Y rotation inside a single 16 ms step** while the *end-of-step* yaw rate is only −0.42 rad/s:
   an intra-step yaw burst (~630°/s equivalent) is generated and cancelled within the same TGS substep
   loop, but the rotation is already integrated into the pose — the visible snap. **Root cause:** the
   contact manifold is built **once per step at the step-start pose**; the body sweeps a large rotation
   across the 8 substeps, so 1–2 *stale* contact points take the full impact impulses, whose pattern has
   a net moment about the vertical; friction (μN) can only respond after the normal impulses exist.
   Sleep contributes nothing (velocities are ~0 long before the island sleeps). This is the same
   staleness behind the P9.5 verdict (rotational tunnelling), the P9.3-deferred "steep-vertex spin",
   and the deep-penetration sinking family — one fix retires all three.
   **Reproducer / gate:** `Demos.ConvexHull.Tetra0YawProbe` (`[.][ConvexProbe]`, tests/physics/
   test_demos.cpp) replays the *actual* gltf (bit-identical fastgltf TRS + `GltfLoader::meshConvexHull`)
   — the first faithful headless repro of this family. The P9.6 gate: promote it to an asserting test —
   max per-step twist about any axis during settle below a threshold (~2–3°/step), plus the existing
   `[Demos]`/`GoldenHash` suites staying green (a narrowphase-cadence change re-baselines GoldenHash).
   **Design, two stages:**
   - *Stage 1 — gated mid-step manifold refresh (cheap first cut):* when a body's intra-step rotation
     exceeds a threshold (|ω|·dt gate, e.g. > ~0.05 rad), re-run narrowphase for its contacts once at
     substep N/2 and re-prepare those constraint rows (anchors/normal/separation; keep accumulated
     impulses via the warm-start proximity match). Costs nothing for the 99% of slow bodies; should
     roughly halve the snap and may drop it below visibility.
   - *Stage 2 — per-substep re-detection / conservative advancement (the full fix):* re-detect
     manifolds every substep for gated bodies (or root-find rotational TOI), giving the solver correct
     geometry for the whole step. Larger: narrowphase moves inside the substep loop for hot bodies, and
     warm-start identity across refreshes needs care (reuse the triangle-indexed sub-key pattern from P6).
   Measure stage 1 against the probe before committing to stage 2 — if the snap drops under the gate
   threshold, stage 2 stays deferred.

**Plus the non-solver production layer** that shipped ragdolls (Havok/PhysX) lean on heavily —
joint projection, limit softness, per-joint mass/inertia scaling, angular damping, deactivation
thresholds, animation blending, and a "force-sleep once it looks good" backstop — because shipped
ragdolls are judged on *visual plausibility*, not physical truth. The (B2) state machine
(FORCE-SETTLE / RELAX-CONSTRAINTS / BLEND-TO-SLEEP) is exactly this layer.

**Steering for fireEngine:**
- **Short term:** replace hard joint Baumgarte with soft/compliant constraints + add TGS substeps. ✅ (P9.1/9.2)
- **Medium term:** ~~block-solve ball-socket anchors~~ — *tried (P9.4), reverted: it diverges ragdoll
  chains, doesn't fix them (see item 3)._ A joint split-position pass (item 4) is still worth trying.
- **Long term:** ~~a reduced-coordinate articulation path for ragdolls~~ ✅ (item 5 — CesiumMan
  settles), leaving rigid-body joints for general / breakable / looped constraints. Next: P9.6
  sub-step narrowphase (item 6).

This attacks the oscillation at the root: reduced coordinates remove joint drift, and TGS / soft
constraints stop the correction step from acting as an energy pump.

**Sequencing rationale:** P0 makes everything debuggable; P1+P2 are the manifold +
solver spine; P3 adds rotation on top of a proven solver; P4 reuses the solver for
joints/ragdolls; P5–P7 are scale and gameplay surface that only matter once bodies
actually behave; **P7.5 hardens the convex core (`gjkEpaContact`) before** P8 leans on the
queries for its showcase; P8 demonstrates the whole track end to end. Mirrors the rendering
spine's "front-load the shared infrastructure via the lowest-risk feature, hardest last"
discipline.

## Summary

Spine: **Particles → Soft-body → Progressive meshes.** Compute + instancing land
in #1, dynamic meshes in #2, culling kicks off #3 — each feature paying down
infrastructure debt for the next, hardest last. **Bindless materials** (✅ done) was
a standalone rendering refactor off the spine — material textures + scalars are now
a global set-2 array + SSBO indexed in-shader; **`bufferDeviceAddress`** is adopted
with #2 (the cloth solver is descriptor-free). Both underpin the GPU-driven direction.

Separate **physics & collision** track ([`collision.md`](collision.md) for the current system): its
own spine — **debug-draw ✅ → contact manifold ✅ → impulse solver ✅ → CCD ✅ → rotation ✅ →
convex narrowphase ✅ → joints ✅ → scale ✅ → mesh/compound ✅ → queries ✅ → GJK hardening ✅ → showcase ✅ → solver robustness (P9)**, front-loading the
manifold + solver (both ✅) the way the rendering spine front-loads compute +
instancing. The solver spine (temporal CCD, rotation, joints, sleeping) and the
shape-coverage strand (GJK/EPA convex narrowphase, static mesh) are orthogonal —
"when" vs "which shapes" — interleaved by payoff. Takes the core from translational
AABB dynamics to full rigid bodies, all behind the existing `PhysicsWorld` boundary.

## Design reviews

Decisions worth revisiting as the engine grows — recorded where the reasoning isn't
obvious from the code.

### Character-controller location (engine class vs scene component) — P7

**Decision:** the `CharacterController` is a `physics/` engine class driven from
`FireEngine::mainLoop` (the `-k` demo), **not** a member of the scene `Components` variant.

**Why:** a variant component's `update(InputState, Transform)` has no `PhysicsWorld`
access, and the controller *is* a world query (shapecast/raycast every move). The scene
layer is deliberately physics-agnostic — it holds only opaque physics handles and bridges
to `PhysicsWorld` solely through `submitPhysics`/`applyPhysics`. Making the controller a
first-class component would force a `PhysicsWorld&` into the component-update signature (or
a back-pointer inside the component), coupling the Vulkan-free scene layer to `PhysicsWorld`
and eroding that separation. So the engine class stays the reusable, headless-tested core,
and the main-loop hook is a thin app-level driver — mirroring how `Controllable` is an
optional-on-`Node` helper rather than a variant member.

**Trade-off:** there's no declarative "attach a character controller to this node"
authoring path yet; gameplay code wires it by hand.

**Revisit when:** a second consumer appears (e.g. authored character nodes, or NPCs). The
clean upgrade is a small update context carrying the world (`SceneUpdateContext { const
InputState&; PhysicsWorld*; }`) threaded into component updates — *not* a per-component
back-pointer. Until then the main-loop driver is the right amount of structure (avoids
speculative generality).
