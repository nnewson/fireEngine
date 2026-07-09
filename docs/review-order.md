# Fire Engine — File Review Order

A complete, file-by-file reading order for understanding the entire system, arranged so each
tier depends only on the tiers above it. Read headers before their `.cpp`; within a tier, go
top-to-bottom.

This complements [`onboarding.md`](onboarding.md) (architecture + recommended route) and [`collision.md`](collision.md) (physics
authoring). Where the two overlap, [`onboarding.md`](onboarding.md) carries the prose; this file is the exhaustive
ordered checklist with per-file attention points.

---

## Build and tooling files

Read these first when a change touches build configuration, CI, or local tooling:

| File | Pay attention to |
|---|---|
| `CMakePresets.json` | The `vcpkg` preset pins Apple Clang on macOS, selects `Dev`, and exports `compile_commands.json` for `clangd`. |
| `CMakeLists.txt` | `Dev` is the default build type (`-O2 -g`, no `NDEBUG`); `FIRE_ENGINE_WARNINGS_AS_ERRORS` is CI-only by default; CTest registers `test_fire_engine` (`~[slow]`) so plain CTest stays fast; `tests-full` runs the all-tags Catch2 binary plus the layering guard; `run-clang-tidy` appears only when `clang-tidy` is installed. |
| `.clang-format` / `.clang-tidy` | Formatting is CI-gated. Tidy is the first-pass static-analysis config for engine `src/` plus `include/fire_engine/`; disabled checks are documented inline. |
| `.github/workflows/ci.yml` | Builds with vcpkg + warnings-as-errors, runs `run-clang-tidy`, runs the full `tests-full` target, and has a separate clang-format dry-run job. |

## Tier 0 — Math & value types (foundation, no dependencies)

| File | Pay attention to |
|---|---|
| `math/constants.hpp` | Just π/epsilon constants — orient quickly. |
| `math/vec_base.hpp` | CRTP base for the vec types; compound-assign are primitives, binary ops delegate. |
| `math/vec2.hpp` / `vec3.hpp` / `vec4.hpp` | `Vec3` is the workhorse. **`magnitude()`/`normalise()` are deliberately NOT constexpr** (sqrt). `operator==` is **strict bit equality** — use `approxEqual` for tolerance. Vec3↔Vec4 conversions are `explicit` both ways. |
| `math/quaternion.hpp` | SLERP, `fromVectors`, Hamilton `operator*`, and `integrate(ω, dt)` (exponential-map orientation integration for the rigid-body solver). Used for all scene rotation; glTF round-trips through this. |
| `math/mat3.hpp` | Column-major 3×3 (mirrors `Mat4`'s `[row,col]`): `fromQuaternion`, `diagonal`, `transpose`, `Mat3·Mat3` / `Mat3·Vec3`. Holds the world inverse inertia `R·diag(invI)·Rᵀ` in the physics solver. |
| `math/mat4.hpp` | **Column-major.** Translation/rotation/scale/perspective/look-at. Everything downstream trusts this — verify the multiplication and handedness conventions. |
| `math/view_basis.hpp` | Shared right/up construction that stays finite for zero-length or vertical look dirs. Used by view, skybox, shadow-fit, sort-depth. |
| `graphics/colour3.hpp` | Trivial RGB value type. |

## Tier 1 — Handles, limits, tunables (vocabulary types)

| File | Pay attention to |
|---|---|
| `graphics/gpu_handle.hpp` | Opaque handle types — the contract that keeps `graphics/` Vulkan-free. |
| `graphics/gpu_limits.hpp` | `kMaxLights`, `kMaxJoints`, `kMaxMorphTargets`, shadow caster caps, `kShadowTotalMatrixCount`. **These must equal shader array sizes** — cross-file invariant. |
| `render/constants.hpp` | Scalar render tunables (biases, IBL strengths, extents, FOV). Includes gpu_limits. Note the bias values — comments explain why they're conservative. |
| `physics/physics_handle.hpp` / `collision/collider_id.hpp` | Stable opaque IDs linking scene↔physics↔broadphase. |
| `core/log.hpp` | Runtime diagnostics. `FE_LOG` is parsed once; levels are `debug`/`info`/`warn`/`error`/`off`, with category overrides like `ragdoll:debug`. Engine code should use `log::debug/info/warn/error`, not direct stream or printf diagnostics. |

## Tier 2 — Input (pre-packaged before scene)

| File | Pay attention to |
|---|---|
| `input/camera_state.hpp`, `controller_state.hpp`, `animation_state.hpp`, `variant_state.hpp` | Small per-domain state bundles. |
| `input/input_state.hpp` | The per-frame bundle handed to `SceneGraph::update`. |
| `input/input.hpp` + `src/input/input.cpp` | GLFW-facing collector that fills `InputState`. **Key point: components never poll GLFW — they read normalized `InputState`.** |

## Tier 3 — Platform & process (thin OS wrappers)

| File | Pay attention to |
|---|---|
| `core/system.hpp` + `system.cpp` | GLFW init/destroy, time. |
| `platform/keyboard.hpp` + `keyboard.cpp` | **`enum class Key` order must match `kGlfwKeyCodes` table** — a reorder silently maps wrong keys (count mismatch fails compile, reorder doesn't). |
| `platform/mouse.hpp` + `mouse.cpp`, `window.hpp` + `window.cpp` | GLFW window/surface lifetime, resize callback. |
| `platform/application_args.hpp` + `src/platform/application.cpp` | CLI parsing → `RendererDebug{ DebugView view; bool noShadows; }`. This is the entry `main`. |

## Tier 4 — Scene graph (runtime behaviour, easiest to reason about)

| File | Pay attention to |
|---|---|
| `scene/transform.hpp` + `transform.cpp` | Local TRS + cached local/world matrices. Cache invalidation is the thing to watch. |
| `scene/components.hpp` | `std::variant<Empty,Animator,Camera,Mesh,Light,ParticleEmitter>`. Access via `componentAs<T>()`/`visitComponent` — not raw `std::get_if`. |
| `scene/empty.hpp` + `empty.cpp`, `controllable.hpp` + `controllable.cpp` | Empty = structural no-op/joints. Controllable = optional gameplay movement, **separate from the variant**. |
| `scene/camera.hpp` + `camera.cpp` | Consumes `CameraState`; exposes world position/target for `drawFrame`. |
| `scene/light.hpp` + `light.cpp` | Point/spot/directional runtime light. |
| `scene/particle_emitter.hpp` + `particle_emitter.cpp` | GPU particle emitter component. Transform-driven; gathered (not drawn) like `Light` via `toEmitterState` → `gatherEmitters`. The GPU pool/sim lives in `render/particle_system`. |
| `scene/mesh.hpp` + `mesh.cpp` | Wraps graphics `Object`; updates morph weights; emits draw commands in render traversal. |
| `scene/node.hpp` + `node.cpp` | Owns transform, component, optional controllable, optional physics handles, children, parent, cached `composedWorld`. The `requires`-based no-op render is here. `render` takes a Vulkan-free `SceneDrawContext` and skips draw-building for nodes in `SceneDrawContext::culledNodes` (still recurses children — Mesh returns `world`, so it's equivalent). |
| `scene/node_format.hpp` / `scene_graph_format.hpp` | Formatters split out so `<format>` doesn't bleed into hot headers — include only at print sites. |
| `scene/scene_graph.hpp` + `scene_graph.cpp` | **Core runtime loop**: `update` → `submitPhysics` → `applyPhysics`/`resolve`, `gatherLights`/`gatherEmitters`, render traversal. Study the physics submit/apply ordering and the Static/Kinematic/Dynamic authority model. Owns the `SceneCuller`; `cull(frustums)` runs the coarse pre-cull before `render`. |
| `scene/scene_culler.hpp` + `scene_culler.cpp` | Persistent `AabbBvh<Node*>` over rigid renderable nodes. `sync()` reconciles the proxy set and re-transforms a proxy's world bound only when the node's `worldRevision()` changed (a per-`Node` counter bumped on world-matrix change), so a static scene does no per-frame bound recompute; `cull(frustums)` unions the leaves visible in *any* frustum (camera + every shadow caster) and returns the nodes in **none** of them — the set `Node::render` skips. Deformable (skinned/morph) nodes are **never tracked** (always drawn, left to the precise per-bucket cull). |

## Tier 5 — Animation (data-first, renderer-independent)

| File | Pay attention to |
|---|---|
| `animation/animation_selection.hpp` | Clip selection for switching. |
| `animation/animation.hpp` + `src/animation/animation.cpp` | LINEAR/STEP/CUBICSPLINE sampling; rotation via SLERP. **Morph weights sampled separately from the TRS matrix** — mesh can animate vertex deltas without moving the node. |
| `scene/animator.hpp` + `animator.cpp` | Scene component that samples `Animation` and pushes the sampled matrix down the tree. |

## Tier 6 — Graphics data (still Vulkan-free)

| File | Pay attention to |
|---|---|
| `graphics/vertex.hpp`, `joints4.hpp` | Vertex layout (pos/colour/normal/tangent/2×UV/joints/weights); fixed 4-joint skinning influence. |
| `graphics/bounds.hpp`, `sampler_settings.hpp` | Value types; sampler filter/wrap from glTF. `Bounds3` is the growable world-space AABB used for shadow fit + frustum culling. |
| `graphics/frustum.hpp` | Vulkan-free, header-only. 6 Gribb–Hartmann planes from a viewProj (**Vulkan [0,1] depth** — near plane = row 2, not row3+row2) + a positive-vertex `intersects(Bounds3)` test (**conservative — no false negatives**; invalid bounds always visible). Tested in `test_frustum.cpp`. |
| `graphics/image.hpp` + `image.cpp` | **Reference class for the whole code style.** Read it as the style exemplar. |
| `graphics/ktx_image.hpp` + `ktx_image.cpp`, `texture.hpp` + `texture.cpp` | KTX2/Basis CPU container; `Texture` holds a `TextureHandle` after upload. |
| `graphics/material.hpp` + `material_binding.hpp`/`.cpp` | Ten textures in one array via `material.texture(MaterialTextureSlot::X)` → `TextureSlot{texture,texCoord,transform,has()}`. Clearcoat/transmission(+ior)/volume are `std::optional<…Params>`. `toMaterialUBO` packs into `MaterialUBO` (`value_or({})` for absent blocks), including each slot's **bindless `textureIndex`** (= the texture's handle). **Slot enum order ↔ `MaterialUBO::uv[]`/`textureIndex[]` ↔ shader `SLOT_*` must move together** (no longer tied to descriptor bindings — textures are bindless). |
| `graphics/geometry.hpp` + `geometry.cpp` | Vertices/indices/material ptr/GPU handles/morph deltas. Note the `castsShadow` flag + `storageVertices` (cloth's compute-writable vertex buffer). `load()` also builds discrete **LODs** for static meshes >`kMinLodTriangles` (`std::vector<GeometryLod>`, all indexing the same vertex buffer). |
| `graphics/lod.hpp`, `mesh_simplifier.hpp` + `mesh_simplifier.cpp` | **Discrete mesh LOD (spine #3, Phase 1).** `lod.hpp`: `GeometryLod{indexBuffer,indexCount,error}`, the ratios/budgets, and the headless-testable `selectLod` (projects a level's world error to screen pixels, picks the coarsest under budget). `mesh_simplifier`: a from-scratch **Garland–Heckbert QEM** with an **attribute-aware R⁵ (position + weighted-UV) quadric** so seams collapse *last*; **position-welded** connectivity (glTF's seam-split verts still simplify) + a **wedge-preserving emit** (each corner keeps its own UV, nearest-wedge). Subset (endpoint) placement → LODs share the base vertex buffer. Records the ordered collapse stream (raw material for Phase 2/3 VIPM/VDPM). Headless `test_mesh_simplifier.cpp` (target count, error bound, boundary lock, determinism, replay, seam-UV). Two dials in the `.cpp`: `kUvWeightFactor` (UV vs simplification), `kErrorCeilingFactor` (only refuses *geometrically* un-simplifiable shapes). |
| `graphics/cloth.hpp` + `cloth.cpp` | Vulkan-free cloth data model: `ClothMesh` (particles + distance constraints + render verts + CSR normal adjacency), `makeGridCloth` and `makeClothFromMesh` (welds an arbitrary glTF mesh into particles, builds structural mesh-edge + triangle-adjacency bend constraints, resolves pin rules), the **greedy edge-colouring** (`colourConstraints`) that makes the GPU solve race-free, and `buildNormalAdjacency` (per-vertex→triangle CSR for runtime normals). Per-type compliance (structural/shear/bend). Plus `ClothCollider` + builders (plane/sphere/box/capsule). Unit-tested headless (`test_cloth.cpp`). |
| `graphics/skin.hpp` + `skin.cpp` | Joint node ptrs + inverse bind matrices; computes joint matrices from cached node worlds. |
| `graphics/lighting.hpp`, `particle.hpp`, `draw_command.hpp`, `frame_info.hpp` | Packed render-facing light record; world-space `EmitterState` (the `ParticleEmitter` gather output); backend-agnostic handle bundle; Vulkan-free per-frame data. |
| `graphics/assets.hpp` | Central arrays (textures/materials/geometries/skins/animations/lights). **Backed by `std::deque`** so pointers survive insertion — important. |
| `graphics/object.hpp` + `src/graphics/object.cpp` | **High-attention (~560 lines).** Writes mapped UBOs, **selects alpha pipeline from material state** (but only returns a `PipelineHandle` — binding happens later), emits `DrawCommand`s. Also runs the **per-draw LOD selection** (`selectLod` from the geometry's `lods()` using camera distance + `frame.proj`/viewport; forward + coarser for shadows) and sets `cmd.indexBuffer`/`indexCount`/`lodLevel`. Watch the per-`currentFrame` mapped-UBO writes. |

## Tier 7 — Collision & physics (read [`collision.md`](collision.md) alongside)

| File | Pay attention to |
|---|---|
| `collision/aabb.hpp` | Single AABB definition + `Axis` enum used by everyone. No per-TU axis helpers. |
| `collision/aabb_bvh.hpp` | Generic fat-AABB BVH (`AabbBvh<T>`, SAH insert + AVL balance, à la Box2D). `createProxy`/`moveProxy`/`destroyProxy`/`query`. **The reusable core (P6):** the physics `DynamicAabbTreeBroadPhase`, the static-mesh triangle BVH, **and** the render-side `SceneCuller` all build on it. `traverse(predicate, fn)` is the generic descent that lets a frustum query it without `collision/` depending on `graphics/`. |
| `collision/end_point.hpp`, `collider.hpp` + `collider.cpp` | Endpoints live inside `Collider`; local/world/swept AABB, layer/mask, six endpoints. `update(world, motion)` extends the swept AABB by a predicted displacement so the broadphase stays motion-aware for CCD. |
| `collision/broad_phase.hpp` | The `BroadPhase` interface (`addCollider`/`removeCollider`/`update`/`rebuild`/`possiblePairs`/`validate`) + the `CollisionPair` value type. `PhysicsWorld` owns one through a `unique_ptr` — **default `DynamicAabbTreeBroadPhase`, injectable via `PhysicsWorld(unique_ptr<BroadPhase>)`.** `validate()` cross-checks the maintained pair set against brute-force O(n²). |
| `collision/dynamic_aabb_tree_broad_phase.hpp` + `.cpp` | The default broadphase: wraps `AabbBvh<Collider*>` + a proxy-by-id map. Emits `possiblePairs_` **sorted by `(firstId, secondId)`** (independent of tree shape / map order) — the canonical order the solver relies on. |
| `collision/sweep_and_prune_broad_phase.hpp` + `.cpp` | The selectable alternative. **Sharp edge: non-owning.** Stores pointers to endpoints owned inside colliders — never move/destroy a registered collider; layer/mask change needs `rebuild()`. Emits pairs in update order; `PhysicsWorld::contacts()` re-sorts to the canonical order, so the solve matches the tree (the `Determinism.BroadphasesAgree` test asserts this). |
| `collision/world_shape.hpp`, `contact_manifold.hpp`, `geometry.hpp` + `geometry.cpp` | Neutral world-space shapes (sphere/box/capsule/**convex hull**), the `ContactManifold` value type, and reusable closest-point primitives (segment, OBB, segment-segment). Read before `narrow_phase`. |
| `collision/support.hpp`, `gjk_epa.hpp` + `gjk_epa.cpp` | **Convex query core (P3.5).** `supportPoint` (farthest point of a shape along a direction); `gjkEpaContact` = GJK distance/witnesses (separated) + EPA depth/normal (overlap) over support functions, with a directional-MTV fallback for the symmetric degeneracy. The reusable convex contact the P6 mesh + the P7 queries reuse. **Edge/corner robustness (P7.5 ✅):** on a flat face/edge the box support tie-breaks to a corner and the GJK loop stalls with a stale simplex vertex, so the closest-point *magnitude* over-estimated the gap (up to ~55%). Fixed in the outer loop, not the sub-distance: track the lower-bound separation `dot(w, v)/|v|` (the separating plane *through* the closest feature, exact at convergence), add duplicate-support termination, and report that gap + normal rather than the stale `|v|`. (Signed-volumes was tried first and abandoned — it reproduced the bug and regressed face cases, because the sub-distance was never the fault.) The `closestOnSimplex` degenerate-tetra guard still handles the *separated-large-box* false-collision. Gated by the `test_gjk_epa.cpp` fuzz suite (exact face + edge/corner). |
| `collision/ray.hpp` + `.cpp`, `shape_cast.hpp` + `.cpp` | **Query primitives (P7).** `rayIntersect` (analytic ray vs sphere/OBB/capsule/convex + Möller–Trumbore triangle + slab reject) and `shapeCast` (GJK conservative advancement, reusing `gjkEpaContact`). Tested in `test_shape_cast.cpp` (+ via `test_physics_query.cpp`). |
| `collision/narrow_phase.hpp` + `.cpp` | `collide(a, b, speculativeMargin)` = shape-specific manifold dispatch. Primitive pairs → analytic `collidePair` (+ box/box SAT); **any convex-involving pair → `collideConvex`** (GJK/EPA → polytope face-clip or single point). Normal points `b → a`. A non-zero margin emits *separated-but-approaching* gap contacts with **negative penetration** (speculative-margin CCD); margin 0 = overlap-only. |
| `physics/collider_shape.hpp`, `contact.hpp`, `physics_body.hpp` + `physics_body.cpp` | Authored shapes (incl. `ConvexHullShape` = vertices + face loops); `PhysicsWorld::worldShape` composes them with the body transform into a `WorldShape` for the narrowphase. `PhysicsBody` carries `inverseInertiaLocal` (diagonal, set from shape+mass in `createCollider`). `DebugContact` carries real manifold points. |
| `core/convex_hull_builder.hpp` + `.cpp` | `buildConvexHull(positions, indices)` welds a triangle mesh and coplanar-merges it into ordered polygon faces (a `ConvexHullShape`); `isConvex` validates. Used by the glTF `Shape: "ConvexHull"` path (`GltfLoader::meshConvexHull`). |
| `physics/contact_solver.hpp` + `.cpp`, `physics_constants.hpp` | **TGS soft-step contact solver with rotation (P9.2).** Decoupled from `PhysicsWorld` (flat `SolverBody` + `SolverContactInput`): `prepare`(at substep `h`)/`warmStart`/`solveVelocity(useBias)`/`applyRestitution`/`solvePosition`(kinematic-only)/`store`. Normal + **2D Coulomb friction** (P9.3): friction is solved as a coupled 2-vector over the contact tangent plane against a symmetric **2×2 tangent mass** (captures the tangents' angular cross-coupling) and clamped to the friction **disk** `|λ| ≤ μ·N` (a circle, not a per-axis box) — that coupling + circular clamp is what stops a tipping/edge contact pumping spurious torque. Friction has **no cross-frame memory** (only the normal impulse warm-starts across frames; replaying friction pumps energy at a rocking contact). Penetration is a **soft constraint** (`b2MakeSoft`: `biasRate`/`massScale`/`impulseScale`) with separation recomputed each substep from body-local anchors (`adjustedSeparation`). The caller runs the substep loop (warm-start → `solveVelocity(true)` bias → integrate → `solveVelocity(false)` relax); restitution is one end-of-step pass at the true impact velocity (`relVelN0`, gated below `kRestitutionThreshold`, only where `maxNormalImpulse > 0`); `solvePosition` is the leftover **kinematic-only** split-impulse pass (Dynamic `positionWeight` is 0). Speculative gaps (separation > 0) brake to `separation/h`. Persistent warm-start cache (lookup-only → determinism-safe). **Angular (P3):** effective mass `invMassA+invMassB+(rA×d)·IA⁻¹(rA×d)+(rB×d)·IB⁻¹(rB×d)`, every impulse adds `I⁻¹(r×P)` torque. Unit-tested in `test_contact_solver.cpp`. |
| `physics/physics_world.hpp` + `src/physics/physics_world*.cpp` | **High-attention.** The public API is in the header; implementation is split by responsibility. `physics_world.cpp` keeps body/collider/joint creation, fixed-step contact generation, island solving, and sleep. `physics_world_shapes.cpp` owns owner-pose/world-shape composition and collider AABB refresh/reset. `physics_world_queries.cpp` owns `raycast`/`shapecast`/`overlap*`. `physics_world_articulation.cpp` owns articulation handles, link colliders, self/static contact gather, and articulation stepping/sleep. `physics_world_debug.cpp` owns debug extraction. `physics_world_events.cpp` owns cloth-collider export and trigger/collision event bookkeeping. Study `step` (narrowphase → `solveAndIntegrate`), `contactForPair`, `solveAndIntegrate` + `solveIsland` (the TGS substep loop: prepare-once → per substep gravity/warm-start/bias-solve/integrate/relax → restitution → kinematic position pass), and the Static/Kinematic/Dynamic authority rules. `contacts()` sorts pairs into canonical `(firstId, secondId)` order so the order-dependent solve is broadphase-independent. The broadphase is injectable via `PhysicsWorld(unique_ptr<BroadPhase>)`. |
| `physics/physics_query.hpp`, `collision_event.hpp` | **P7 value types.** `QueryFilter` + `Raycast`/`Shapecast`/`OverlapHit` (carry collider+body handles); `EventPhase` + `ContactEvent` (ordered collider-handle pair + phase). |
| `physics/character_controller.hpp` + `.cpp` | **Kinematic capsule controller (P7).** `move(world, displacement)` = collide-and-slide over the queries: vertical/horizontal swept separately, blocked motion projected, slope-limit drops upward slide off too-steep faces, lift→walk→**sweep-and-rest** step-up, capsule-sweep distance + raycast-normal grounding. The step-up drop rests on the step's top edge (a downward sweep, no slide) so a rounded capsule climbs at walk speed instead of sliding off; a step is accepted only when the **lifted walk clears the obstacle** (rejects walls/steep slopes without the ambiguous edge-normal stutter). The horizontal slide ignores **walkable** ground/edge contacts (only walls block lateral travel) — without this the capsule wedges against a step's top edge descending. A `move()` output guard clamps any net-**backward** horizontal step (a rare riser depenetration against the travel) to a no-progress frame. Headless-tested (`test_character_controller.cpp`) at the demo's 0.05/frame cadence — walk-speed step-up + patrol-replay traversal/no-wedge/no-stall tests, the coverage the old 0.3/frame tests lacked. Driven from `FireEngine::mainLoop` (`-k` demo: a step-pyramid patrol; advanced at the **real per-frame dt** for smooth motion at any refresh rate, bounds-based turnaround at the flat ends, gravity only while airborne), not a scene-`Components` member — [`roadmap.md`](roadmap.md) § Design reviews explains why. |

## Tier 8 — Loading (largest translation layer)

| File | Pay attention to |
|---|---|
| `core/shader_loader.hpp` + `.cpp` | SPIR-V loading — simple, read first. |
| `core/tangent_generator.hpp` + `.cpp` | Generates tangents when base/clearcoat normal mapping needs them. |
| `core/gltf_loader.hpp` | One header declaring all `GltfLoader::` members; the `.cpp` is split by concern across `gltf_loader_*.cpp`. |
| `src/core/gltf_loader_scene.cpp` | `loadScene` entry point — drives parse → asset load → node/scene build. |
| `src/core/gltf_loader_nodes.cpp` | Node loading + hierarchy build, physics-body creation, `Controllable`/TRS application. **Note the animated-node hierarchy split** that protects authored base transforms from sampled animator transforms. |
| `src/core/gltf_loader_assets.cpp` | `parseAsset`, extension validation, asset presize, `extras` dispatch. |
| `src/core/gltf_loader_extras.cpp` | `extras.Controllable`/`Physics`/`Cloth`/`Ragdoll` parsing. |
| `src/core/gltf_loader_mesh.cpp` | Primitives/material factors/textures/samplers/UV transforms, convex-hull + triangle extraction (physics shapes), `loadMesh` assembling texture slots. **Sharp edge: base-normal and clearcoat-normal are both tangent-space — if tangents fail, skip both consistently.** |
| `src/core/gltf_loader_geometry.cpp` | **High-attention.** Geometry/vertex loading, smooth-normal + tangent generation (feeds the mesh sharp edge above). |
| `src/core/gltf_loader_material.cpp` | Material factor + texture-slot resolution. |
| `src/core/gltf_loader_images.cpp` | Centralized image source loading: direct local URI path or resolved bytes, then `Image`/`KtxImage`; texture-index lookup. |
| `src/core/gltf_loader_animation.cpp` | Animation channels + skin/inverse-bind loading. |

## Tier 9 — Vulkan resources & pipeline plumbing

| File | Pay attention to |
|---|---|
| `render/device.hpp` + `device.cpp` | Instance/surface/physical+logical device/queues, and the process-wide **VMA allocator** (`allocator()`, owned here as a `VmaAllocatorHandle`; buffers/images sub-allocate from it — `createBuffer` returns a `UniqueVmaBuffer`). Built with `VULKAN_HPP_NO_CONSTRUCTORS` — designated initializers everywhere. |
| `render/vma.hpp` | Layer 3 of the GPU resource model (see CLAUDE.md): the VMA include shim + `VmaAllocatorHandle` and the move-only `UniqueVmaBuffer`/`UniqueVmaImage` RAII wrappers (own resource + sub-allocation together, Approach A; `operator*` yields the `vk::` handle to mirror vk::raii). Implementation TU: `render/vma_impl.cpp`. |
| `render/generational_slot_pool.hpp` + `.cpp` | Pure index+generation slot allocator behind the texture free-list (CR-17): `acquire`/`release`/`valid`, recycles released slots so the table stays bounded across resize churn. Unit-tested headless in `tests/render/test_generational_slot_pool.cpp`. |
| `render/swapchain.hpp` + `swapchain.cpp` | Images/views/extent/depth/recreation. |
| `render/frame.hpp` + `frame.cpp` | Command pool/buffers + sync objects: binary `imageAvail`/`renderDone` semaphores (WSI acquire/present), plus one monotonic **timeline semaphore** for CPU↔GPU frame pacing (replaced the in-flight fences). Renderer holds the per-slot/per-image timeline values. |
| `graphics/renderable_scene.hpp` | The Vulkan-free render↔scene seam (CR-09): the `RenderableScene` interface (`gatherLights`/`gatherEmitters`/`buildDrawCommands`) that `SceneGraph` implements and `Renderer` pulls through. The renderer builds a `graphics/frame_info.hpp` `FrameInfo` per frame (retired the old `RenderContext`) and passes it plus cull frustums in; the scene emits `DrawCommand`s back. |
| `render/viewport.hpp`, `cubemap_basis.hpp` | Small render helpers. |
| `render/descriptor_bindings.hpp` | **Cross-file lynchpin.** `ForwardBinding`/`ForwardGlobalBinding`/`ShadowBinding`/`SkyboxBinding`/`PostProcessBinding` must match every `layout(set,binding)` in GLSL. |
| `render/ubo.hpp` | **High-attention.** Structs memcpy'd into mapped GPU memory — field order/`alignas`/padding must mirror GLSL std140 exactly. `test_ubo.cpp` guards sizes/offsets. Note `LightUBO::environmentParams` packs debug-view + no-shadows. |
| `render/descriptors.hpp` + `descriptors.cpp` | Forward set 0 (per-object: frame/skin/morph UBOs + morph SSBO — 4 bindings) is **pushed inline per draw** via `pushForwardObjectDescriptors` (`VK_KHR_push_descriptor`) — not allocated. The shadow set 0 is **pushed inline per draw** too (`pushShadowObjectDescriptors`: per-object ShadowUBO + reused skin/morph buffers + the shared self-shadow image/sampler from `Resources`) — no per-object allocation in either pass. Set 1 (forward globals): `createGlobalDescriptors` once at startup; **`updateGlobalDescriptors` after resize** so recreated sceneColor sampler isn't dangling. Watch info-object lifetime around `updateDescriptorSets`. Set 2 (bindless materials) is owned by `Resources`, not here. |
| `render/pipeline.hpp` + `pipeline.cpp` | Forward pipelines declare `bindings` (set 0), `globalBindings` (set 1), and `bindlessSet` (set 2: a partially-bound `sampler2D[kMaxBindlessTextures]` + materials SSBO, with update-after-bind binding flags); non-forward pipelines leave globals/bindless off. Fullscreen/fragment-only config factories share private helpers; `test_pipeline_config.cpp` guards the returned state. |
| `render/resources` (bindless) | Owns the global set-2 descriptor (update-after-bind pool). `registerBindlessTexture` writes each 2D material texture into the array at its handle index (called from the 2D `createTexture` paths). `registerMaterial` assigns a material its slot in the persistently-mapped materials[] SSBO (dedup by `Material*`), returning the index drawn with. |
| `render/render_target.hpp` + `render_target.cpp` | `RenderTarget` descriptor (attachment formats for `VkPipelineRenderingCreateInfo`) + `makeRenderingInfo` helper. Dynamic rendering — no `VkRenderPass`/`VkFramebuffer`; passes transition layouts with explicit synchronization2 barriers. |
| `render/compute_pipeline.hpp` + `compute_pipeline.cpp` | Compute counterpart to `Pipeline` (single compute stage, `VkComputePipelineCreateInfo`) + synchronization2 **buffer**-barrier helpers (`makeBufferMemoryBarrier`/`recordBufferBarrier`). Used by `ParticleSystem`. |
| `render/resources.hpp` + `src/render/resources.cpp` | **The bridge. High-attention (~1300 lines).** Buffer/texture creation (`createHostVisibleBuffer`, mapped-buffer helpers, `uploadImageFromHost`, `createTexture2DTarget`, `withOneTimeSubmit`), descriptor pools, pipeline registry, all shadow-map resources, HDR/bloom/IBL/BRDF targets, `SharedTextures` aggregate. Buffers/images are VMA-backed (`UniqueVmaBuffer`/`UniqueVmaImage`); texture slots are managed by a `GenerationalSlotPool` so `releaseTexture`/`createTexture` recycle slots and stale handles are detectable (`validTexture`). Load-time uploads coalesce through `begin/endUploadBatch` (one submit + fence, bracketed in `FireEngine::run`); `withOneTimeSubmit` waits on a per-submit fence, not `queue.waitIdle`. **Mapped UBO pointers: always write the `currentFrame` slot.** |

## Tier 10 — Frame rendering & orchestration (hardest bugs live here)

| File | Pay attention to |
|---|---|
| `render/environment_precompute.hpp` + `.cpp` | Equirect→cubemap, irradiance, prefilter, BRDF LUT at startup. |
| `render/shadows.hpp` + `shadows.cpp` | **High-attention.** CSM directional + world-only CSM, spot layers, point cubemap-array, **dual-depth per-skinned-object self-shadow** (two passes: capture nearest surface, then `cullMode=eFront` for next occluder; in-shader `skinnedSelfShadowDepthEpsilon` safety net). `kMaxSkinnedSelfShadowCasters` cap. `recordPass` takes the shadow matrices + `cullingEnabled` and filters each cascade/spot/point-face draw list against its own `Frustum` (self-shadow slots aren't culled). |
| `render/post_processing.hpp` + `post_processing.cpp` | HDR target, bloom chain, ACES/gamma. |
| `render/transmission.hpp` + `transmission.cpp` | **High-attention.** `KHR_materials_transmission` off the captured `sceneColor`. The `shader.frag` split (post-fix): clear/frosted glass does screen-space refraction (roughness-blurred by the sceneColor mip chain); a thin-walled surface that is **also emissive** (a self-lit paper lamp shade) instead scatters to a view-independent irradiance tint — so a bright bulb behind it doesn't beam a camera-tracking blob. Discriminator is the **emissive factor**, NOT thickness. Plus back-face normal flip. |
| `render/debug_draw.hpp` + `debug_draw.cpp` | Physics debug wireframes. `PhysicsDebugData` (AABBs + `ClothCollider` shapes + `DebugContact`s, all Vulkan-free) → CPU-built line list in a per-frame mapped `Vertex` buffer (`Resources::createMappedVertexBuffers`) → line-list pipeline (`Pipeline::debugLineConfig`, `PipelineConfig::topology` + `dynamicDepthTest`) drawn into HDR after particles. Brackets the HDR target ShaderReadOnly↔ColorAttachment + depth ReadOnly→Attachment. Physics side: `PhysicsWorld::debugColliderBounds()` / `debugContacts()` (captured in `step()` pre-resolve). |
| `tests/physics/test_physics_determinism.cpp` + `tests/support/state_hash.hpp` | Determinism harness: FNV-1a body-state hash; ReplayIsBitIdentical / FreeFallMatchesClosedForm / GoldenHash / **BroadphasesAgree** (same scene through the tree + an injected SAP → identical hash). The GoldenHash constant is a behaviour tripwire — update it intentionally when the solver math changes. |
| `render/ssao.hpp` + `ssao.cpp` | **High-attention.** SSAO + contact shadows. Runs after the depth prepass: borrows the shared scene depth (attachment → read-only → attachment), reconstructs view position+normal from depth alone (analytic unprojection from `proj`, no normal G-buffer), writes R8G8 (R = hemisphere-kernel AO, G = sun-direction contact-shadow ray-march; `ssao.frag`), then a **depth-aware bilateral blur** (`ssao_blur.frag`, view-space-Z edge-stop) into a second target. Always runs (disabled = intensity 0). Forward set 1 binding 13 samples the blurred target. The depth prepass itself is `Pipeline::depthPrepassConfig` + `Renderer::recordDepthPrepass` (reuses `shader.vert` w/ `invariant gl_Position`; forward loads depth `LESS_OR_EQUAL`). |
| `render/taa.hpp` + `taa.cpp` | **High-attention.** Temporal AA subsystem. Owns the RG16F velocity target (written by the forward/transmission passes as a 2nd colour attachment), two ping-pong history HDR targets, and the resolve pass (`taa.frag`): reproject history along velocity → 3×3 neighbourhood clamp → blend → blit into the offscreen HDR target. `historyWritten_` guards the first frame after (re)create. Sub-pixel jitter lives in `Renderer::drawFrame`; motion vectors are jitter-free. |
| `render/particle_system.hpp` + `particle_system.cpp` | Renderer-owned GPU particle system. Pooled SSBO partitioned per emitter; compute sim (`particle_simulate.comp`) → buffer barrier → instanced additive billboards into HDR (soft particles via sampled scene depth). Records after the TAA resolve (un-jittered, kept out of history), before post-process. |
| `render/soft_body_system.hpp` + `soft_body_system.cpp` | **High-attention.** GPU XPBD cloth solver. Descriptor-free: four compute pipelines (`cloth_predict`/`solve`/`collide`/`finalize`) take every buffer as a `bufferDeviceAddress` pointer in the push constant — per-cloth particle/constraint buffers + the render vertex buffer + a per-frame collider buffer, all `eShaderDeviceAddress`. `recordSolve` = per-substep `predict → per-colour solve → collide`, then `finalize` writes solved positions + normals (recomputed from the per-cloth CSR adjacency, arbitrary topology) into the cloth's storage vertex buffer (compute-write → vertex-input-read barrier). Reads `ClothSimParams` (overlay; compliance is a global multiplier on each constraint's authored per-type stiffness); colliders from `PhysicsWorld::gatherColliders`. Cloths come from the `-c` demo or glTF `extras.Cloth`. |
| `render/render_tunables.hpp` | Plain struct of live, overlay-editable render params (TAA, **`cullingEnabled`**, **`lodEnabled`/`lodPixelErrorBudget`**, debug view, bloom/IBL/sun, particle scales) + the `DebugView` enum (incl. `Lod` tint). Seeded from `constants.hpp` + CLI flags; the renderer reads it instead of the `constexpr`s. Read this first — it's the contract between the overlay and the renderer. |
| `render/gpu_profiler.hpp` + `gpu_profiler.cpp` | Timestamp `VkQueryPool` ring (`kMaxFramesInFlight` slots). `begin/end(pass)` write a pair; `resolve` reads the slot a cycle later (safe — the acquire timeline-wait guarantees that frame finished) into `FrameStats`. `slotUsed_` guards reading never-reset queries; `eWithAvailability` skips passes that didn't run. Disabled when `timestampPeriod==0` / `timestampValidBits==0`. `FrameStats` also carries the frustum-cull tracked/culled counts (populated in `collectDrawCommands`, shown a frame later). |
| `render/debug_overlay.hpp` + `debug_overlay.cpp` | Dear ImGui owner (context + GLFW/Vulkan backends, dynamic rendering). `buildUi(stats, tunables)` builds the panels (incl. the **Culling** group: `cullingEnabled` toggle + tracked/visible/culled readout); `record` draws into the swap image (loadOp Load). Non-movable (ImGui global state). ImGui core plus GLFW/Vulkan backends come from the vcpkg `imgui[glfw-binding,vulkan-binding]` manifest dependency; `cmake/fireengine_imgui.cmake` wraps the ImGui archive so `fireengine` keeps direct ownership of Vulkan/GLFW linkage. |
| `render/renderer.hpp` + `src/render/renderer.cpp` | **Capstone.** `drawFrame` = named phases (`updateFrameLighting`/`collectDrawCommands`/`recordShadowPass`/`recordForwardPass`/`recordTransmissionPass`/`recordPostProcessing`) plus the inline `taa_.recordResolve`, `recordParticlePass`, `overlay_.record`, and `transitionSwapchainToPresent` (present-split: post-process leaves the swap image in colour-attachment layout). Each pass is wrapped in a `profiler_` scope. Reads `tunables_` for debug view, IBL/bloom/sun, TAA params, particle scales. Study pass ordering, the jitter-free `currentViewProj_`/`previousViewProj_` matrices, `recreateSwapchain` + `buildGlobalDescriptorRequest`. `collectDrawCommands` runs the coarse frustum pre-cull (builds camera + shadow frustums → `scene.cull`); `buildDrawBuckets` does the precise per-camera cull. |

## Tier 11 — Shaders (verify against the C++ they mirror)

Read paired with `ubo.hpp`, `descriptor_bindings.hpp`, `gpu_limits.hpp`.

| File | Pay attention to |
|---|---|
| `shader.vert` / `shader.frag` | Main forward PBR. `shader.vert` also emits jitter-free current/previous clip positions for TAA motion vectors (skinned meshes fall back to camera-only velocity). **`shader.frag` is the densest shader**: writes `outVelocity` (location 1) before any early return, then IBL, CSM/spot/point shadow sampling, double-sided normal flip (`if (!gl_FrontFacing) N = -N;` for `N`, `shadowNormal`, clearcoat `N_cc`), debug-view branches (incl. velocity = 5, SSAO = 6, **LOD tint = 7** — colours by `pc.lodLevel`), transmission. Array sizes must equal `gpu_limits.hpp`. |
| `taa.frag` | TAA resolve: `historyUV = uv − velocity`, 3×3 neighbourhood clamp, `mix(current, history, historyBlend)`; falls back to current when history is invalid or the reprojected UV is off-screen. Fullscreen triangle (`postprocess.vert`). |
| `shadow.vert`/`.frag`, `self_shadow_second.frag` | Skinning+morph still run here so animated geo casts matching shadows; second pass = back-faces only. |
| `skybox.*`, `postprocess.*` | Cubemap sample; ACES/gamma. |
| `environment_convert`, `irradiance_convolution`, `prefilter_environment`, `brdf_integration`, `bloom_downsample`/`upsample` | IBL precompute + bloom kernels. |
| `particle_simulate.comp`, `particle.vert`/`.frag` | GPU particle sim (spawn-claim atomics + integrate) and instanced billboards (`gl_InstanceIndex` reads the pool SSBO; soft-fade vs sampled scene depth). UBO layout mirrors `ParticleFrameUBO`/`ParticleEmitterGpu`. |
| `cloth_predict`/`solve`/`collide`/`finalize.comp` | XPBD cloth substep chain: integrate gravity+wind → per-colour distance projection (race-free, `compliance = c.compliance * scale`) → push out of colliders → write positions + adjacency-gathered normals into the `Vertex` buffer (flat-float writes, `sizeof(Vertex)==100` asserted C++-side). All buffers arrive via `GL_EXT_buffer_reference`; the shared 112-byte `Push` block (vec4s first for 16-byte alignment) mirrors `ClothPush`. |

## Tier 12 — Top of the tree (everything comes together)

| File | Pay attention to |
|---|---|
| `fire_engine.hpp` + `src/fire_engine.cpp` | **Read last.** Owns Window/Renderer/Input/SceneGraph/Assets/PhysicsWorld/Camera*. The **fixed-timestep accumulator inside a variable-rate render loop** (`mainLoop`/`stepSimulation`), which also derives `alpha = accumulator / fixedDt` and threads it into `applyPhysics`/`syncNodes` for CR-20 render interpolation, plus default-sun seeding, `-f` floor plane, and category-gated ragdoll diagnostics (`FE_LOG=ragdoll:debug`). This is your whole-frame mental model. |

---

## How to use this

- **Stop at Tier 6** if you only need to add a material field or scene feature.
- **Tiers 9–10 are the deep end** — `resources.cpp`, `renderer.cpp`, `shadows.cpp`, `shader.frag`.
  Budget the most time there.
- **Read the cross-file invariants** ([`onboarding.md`](onboarding.md) § Cross-File Invariants) before touching
  anything in Tiers 6, 9, or 11 — the compiler won't catch a slot-order or std140 mismatch.
</content>
</invoke>
