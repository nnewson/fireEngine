# Code Review

Scope: refreshed static review of the current `progressive-meshes-phase3` working tree, with extra attention on the new VDPM / LOD code, representative renderer resource paths, modern C++23 usage, Vulkan practice, and redundant code. This is not a request for a rewrite; larger candidates are separated from the actionable findings below.

Verification performed for this refresh: `cmake --build build --target test_fire_engine`, `ctest --preset fast`, and `git diff --check` all pass. Full CI and a validation-layer render smoke test were not run.

## Findings

### High

1. **VDPM forest replay continues after an unreplayable collapse.**

   `buildVertexForest` documents zero live edge faces and non-manifold edges as unsupported, but the implementation just skips recording that split and continues replaying later collapses (`src/graphics/vdpm.cpp:190-196`). That means the evolving topology used to recover all later `vl` / `vr` dependencies no longer matches the simplifier stream. The current foldover and coverage repairs can mask symptoms, but the forest is structurally unfaithful after the first skip.

   Recommendation: make this explicit and deterministic. Either truncate the forest at the first unreplayable collapse, fail/disable VDPM for that mesh, or rebuild the stream so every collapse can be replayed. Add a test/assertion for the chosen behavior; the header already says these cases are rejected (`include/fire_engine/graphics/vdpm.hpp:81-86`), so the code should not silently continue as if the stream were intact.

### Medium

2. **Reversing a chart-vetoed collapse can bypass the simplifier's error ceiling.**

   The run loop checks the cheaper endpoint error against `ceiling` before applying the render-chart veto (`src/graphics/mesh_simplifier.cpp:500-510`). If that direction would lose a chart, it swaps `kept` / `removed` and replaces `err` with the other endpoint's error, but never checks the new, potentially much larger value against the ceiling (`src/graphics/mesh_simplifier.cpp:516-524`). The collapse can therefore exceed the mesh-scale quality bound that the loop claims to enforce. It also makes `maxError` and LOD selection faithfully report a collapse that should never have been accepted.

   Recommendation: after choosing the final legal direction, re-check its error against `ceiling` before `wouldFlip` / `collapse`. Do not `break` on the reversed cost—the heap is ordered by each edge's minimum endpoint cost, so other edges may still have a legal direction under the ceiling. Add a seam case where the cheap direction is illegal and the reverse direction is over budget.

3. **Coverage repair proves centroid coverage, not triangle coverage.**

   `repairCoverage` checks whether the finest triangle's projected centroid lies inside the active replacement (`src/graphics/vdpm.cpp:606-618`). The test helper mirrors that exact centroid-only condition (`tests/graphics/test_vdpm.cpp:220-303`), so the regression is useful but not an independent proof that the replacement covers the visible footprint. A replacement can contain the centroid while still receding past an edge or corner, especially at silhouettes and skinny projected triangles.

   Recommendation: treat coverage as a projected-triangle property. A reasonable next step is to sample centroid, vertices, and edge midpoints, or implement a conservative 2D containment/intersection test. Also move `kMinNdcArea` away from a fixed NDC constant (`src/graphics/vdpm.cpp:559-562`): its pixel meaning changes with viewport size and aspect ratio, so it can skip visible holes on high-resolution views. The current `w <= 0` early-out also leaves triangles crossing the near plane unchecked; clipping to the view volume before the coverage test would handle that case correctly.

4. **VDPM transform handling is only correct for rigid or uniform-scale instances.**

   `refineForView` transforms normals with `world * vec4(normal, 0)` (`src/graphics/vdpm.cpp:385-402`), rather than the inverse-transpose normal matrix. That changes the facing and silhouette decisions under non-uniform scale. Separately, `repairFoldovers` compares original and replacement winding using object-space cross products (`src/graphics/vdpm.cpp:472-499`), while the result is culled after model/view/projection transforms. Both are fine for identity, rotation, translation, and uniform scale, but the API accepts an unrestricted `Mat4` and the scene transform supports non-uniform scale.

   Recommendation: compute a normal matrix once per call for facing tests, and perform foldover orientation in the same transformed/projected space that determines raster winding (with a deliberate near-plane policy). If VDPM intentionally forbids non-uniform scale, enforce and document that invariant instead of silently producing different refinement.

5. **Device suitability does not validate all features and limits later assumed by device/pipeline creation.**

   `isDeviceSuitable` checks queues, extensions, and swapchain support only (`src/render/device.cpp:165-197`). `createLogicalDevice` checks `imageCubeArray`, synchronization2, and dynamic rendering, but then unconditionally requests `samplerAnisotropy`, `independentBlend`, timeline semaphores, buffer device address, several descriptor-indexing features, and `mutableComparisonSamplers` without querying them (`src/render/device.cpp:246-316`). The bindless layout also assumes the relevant update-after-bind descriptor limits can accommodate 512 textures plus the global samplers (`include/fire_engine/graphics/gpu_limits.hpp:55-64`). On a device missing any one of these, selection succeeds and a later Vulkan call fails with a much less useful error.

   Recommendation: query one complete `vk::PhysicalDeviceFeatures2` chain (including Vulkan 1.2/1.3 and portability-subset features) plus the descriptor-indexing properties during suitability evaluation. Reject with a precise reason if a mandatory feature or limit is absent. Also remove `descriptorBindingVariableDescriptorCount` from the requested feature set unless the layout actually gains an `eVariableDescriptorCount` binding.

6. **Static vertex and index buffers are allocated as host-visible/coherent memory.**

   `createVertexBuffer` and both static `createIndexBuffer` overloads call `createHostVisibleBuffer` directly (`src/render/resources.cpp:139-142`, `src/render/resources.cpp:179-188`). That is simple and works, especially on UMA/macOS, but it is not the Vulkan best-practice path for long-lived static geometry. Static meshes, LOD buffers, and VIPM data are good candidates for device-local buffers with staging uploads; persistently mapped host-visible memory should stay for per-frame dynamic buffers such as UBOs and VDPM index buffers (`src/render/resources.cpp:1193-1197`).

   Recommendation: add a device-local static-buffer upload path and keep the existing mapped helpers for dynamic CPU-written data. This is a larger resource-system cleanup, not a correctness bug.

7. **VDPM per-frame work is correct but expensive and allocation-heavy.**

   Each VDPM frame coarsens/refines, runs repair passes, allocates a fresh `std::vector<uint32_t>` from `emitActiveIndices`, then copies it into the mapped index buffer (`src/graphics/object.cpp:455-473`). The repair loops also repeatedly recompute active ancestors and projected positions across all finest faces (`src/graphics/vdpm.cpp:564-649`).

   Recommendation: keep the current CPU-first design for correctness, but consider a follow-up that emits into a reusable per-binding scratch buffer or directly into a caller-owned span, caches per-iteration active ancestors / projected positions, and exposes per-frame repair counters so regressions are visible.

### Low / Refactor

8. **Position-weld and render-wedge matching logic is duplicated.**

   The same conceptual machinery appears in the simplifier (`src/graphics/mesh_simplifier.cpp:273-317`, `src/graphics/mesh_simplifier.cpp:564-580`), VIPM (`src/graphics/vipm.cpp:15-66`), VDPM (`src/graphics/vdpm.cpp:20-110`), and tests (`tests/graphics/test_vdpm.cpp:181-235`). The duplication is now risky because chart identity, seam preservation, and nearest-wedge behavior are exactly where recent bugs occurred.

   Recommendation: extract a small mesh-topology / render-wedge utility for exact position keys, canonical welds, wedge distance, canonical-wedge tables, and nearest-wedge lookup. Tests should call the same canonical helper where they are setting up data, while property checks should stay independent where possible.

9. **`std::vector<bool>` is avoidable in a hot, stateful data structure.**

   `ActiveFront` stores `active_` and `refined_` as `std::vector<bool>` (`include/fire_engine/graphics/vdpm.hpp:189-191`). The proxy specialization is compact, but it is awkward for debugging, can generate surprising code, and is not usually a win for per-frame mutation-heavy logic.

   Recommendation: prefer `std::vector<std::uint8_t>` or an explicit bitset wrapper. The latter is only worth it if memory pressure is measurable; `uint8_t` is simpler and more transparent.

10. **Mapped-buffer writes are debug-checked only.**

   `writeMapped` carries the destination size in a `std::span`, which is good, but the overflow guard is an `assert` before unconditional `memcpy` (`include/fire_engine/graphics/mapped_buffer.hpp:18-21`). In release builds this becomes unchecked. Most current call sites are size-stable, and the VDPM index buffer is sized to the finest index count (`src/graphics/object.cpp:205-214`), so this is low risk.

   Recommendation: consider a release-visible guard for dynamic model-driven writes, even if it terminates or logs once. The span API is already the right foundation.

11. **A vector-copying accessor is incorrectly marked `noexcept`.**

   `QemRun::sequence()` returns `sequence_` by value while declaring `noexcept` (`src/graphics/mesh_simplifier.cpp:582-585`). The copy allocates and can throw; allocation failure would therefore terminate instead of propagating. Both callers are consuming a completed run (`src/graphics/mesh_simplifier.cpp:1041,1076`), so this is also an avoidable copy.

   Recommendation: make it an `&&`-qualified consuming accessor that returns `std::move(sequence_)`, then call it on `std::move(run)`, or simply remove `noexcept` if keeping copy semantics is intentional.

12. **Descriptor and shared-storage helpers encode lifecycle details awkwardly.**

   Every `buildFrameSets` call creates a separate pool with `eFreeDescriptorSet`, even though sets are retained for the pool's whole lifetime and never individually freed (`src/render/descriptors.cpp:69-80,104-138`). Separately, `createMappedStorageBuffer` creates one shared host-visible buffer, duplicates its handle into every frame slot, and returns empty mapped spans (`src/render/resources.cpp:1200-1216`); the name and return type promise a per-frame mapped set that does not exist.

   Recommendation: group long-lived descriptor sets into a small number of lifetime-based pools and omit `eFreeDescriptorSet` when pool reset/destruction is the reclamation mechanism. Rename the storage helper to describe shared ownership and return a `BufferHandle` (or introduce an explicit `SharedBufferSet` if uniform frame indexing is valuable). These are simplifications, not correctness fixes.

## Larger Rewrite Candidates

- **VDPM exact visibility constraints:** replace per-frame foldover / coverage repairs with precomputed per-split foldover, silhouette, and coverage cones. The current repair passes are good proof machinery; the exact cones would be the principled long-term form.
- **Static GPU resource residency:** split static asset upload from dynamic mapped buffers so vertex/index/material-heavy resources default to device-local memory.
- **Shared mesh attribute identity layer:** centralize canonical position welding, chart identity, wedge restoration, and attribute-distance policy so discrete LOD, VIPM, VDPM, and tests cannot drift.
- **VDPM emission pipeline:** if CPU cost becomes visible, rewrite active-front emission around reusable scratch buffers first, then consider GPU-driven refinement/emission later.
- **Capability-driven device setup:** represent required renderer capabilities once, use that description for physical-device filtering and feature-chain construction, and make optional paths genuinely conditional rather than relying on device-creation failure.

## Things That Look Good

- The VDPM implementation is Vulkan-free and headless-testable, which is the right boundary.
- Keeping `MeshCollapse::error` untouched while adding VDPM-only fields avoided discrete/VIPM regressions.
- The frame path uses jitter-free `currentViewProj` for coverage repair (`src/graphics/object.cpp:462-468`), which is the correct fix for no-camera-motion flicker.
- The renderer already uses modern Vulkan patterns in the inspected paths: VMA-backed RAII, synchronization2 barriers, dynamic rendering, timeline semaphore frame pacing, and static layout assertions for shader-visible CPU structs.
- The acquire semaphore waits at colour-attachment output, which allows offscreen/compute work to start early while still protecting the first swapchain-image access; the frame timeline uses `eAllCommands` for safe CPU reuse.
- Push descriptors for per-object data plus a global bindless material set remove a large amount of descriptor allocation/update churn from the draw path.
