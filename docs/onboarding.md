# Fire Engine Onboarding

This guide is for engineers who know C++ and graphics programming but are new to this codebase,
and who will **maintain and extend** the engine — not just run it. It gives a practical route
through the engine from the smallest classes to the highest-risk systems, then covers the
conventions, build wiring, and cross-file invariants you need so that changes land in the right
layer and don't silently break a matching declaration somewhere else.

If you only need the high-level feature list and how to run it, read [`README.md`](../README.md) first; this file
assumes you are about to change code.

## Core Mental Model

Fire Engine is split around ownership boundaries:

- `math/` and simple value types are header-only or mostly value-style.
- `graphics/` is Vulkan-free. It stores renderable CPU data and opaque GPU handles.
- `render/` owns Vulkan objects, descriptor layouts, pipelines, command buffers, and
  presentation. Targets Vulkan 1.4: dynamic rendering (no `VkRenderPass`/`VkFramebuffer`) and
  synchronization2 barriers/submits throughout.
- `scene/` owns node hierarchy, transforms, components, input-driven scene state, and physics
  handles.
- `physics/` owns runtime physics state.
- `collision/` owns low-level collision primitives, broadphase, and narrowphase.
- `core/` loads files and translates external formats into engine data.

The most important rule: `graphics/` does not include Vulkan. Graphics code asks `Resources`
for GPU resources and receives opaque handles. `Renderer` resolves those handles when
recording Vulkan commands.

```text
glTF file
  -> GltfLoader
  -> Assets + SceneGraph + PhysicsWorld
  -> Resources creates GPU buffers/textures/descriptors
  -> Mesh/Object emits DrawCommand handles
  -> Renderer resolves handles and records Vulkan commands
  -> submit + present
```

## Code Style & Conventions

`.clang-format` is the formatter of record (Allman braces, 4-space indent, 100 columns,
left-aligned pointers, no single-line functions, constructor initializers each on their own
line). CI checks `clang-format --dry-run -Werror`, so format before you commit:

```bash
clang-format -i <files you touched>
```

`.clang-tidy` is checked in for the first-pass static-analysis gate over engine `src/` and
`include/fire_engine/` (`bugprone-*`, `performance-*`, selected `modernize-*`, implicit bool
conversions, and special-member checks). If `clang-tidy` is installed locally, run:

```bash
cmake --build build --target run-clang-tidy
```

Conventions that are load-bearing (a reviewer will push back if you break them):

- **C++23**, `constexpr` wherever it can hold (math/value types especially), `[[nodiscard]]` on
  getters and pure functions, `noexcept` where the body can't throw.
- **Private members carry a trailing underscore** (`x_`). Getter and setter **share the name**:
  `float x() const` / `void x(float)` — no `getX`/`setX`.
- **Named constants use the Google `k` style** (`kMaxFramesInFlight`), not `SCREAMING_SNAKE`.
- **Explicit rule-of-five** on every owning type — default or delete each member, don't rely on
  implicit generation.
- **Compound-assignment operators are the primitives**; binary operators delegate to them.
- **Static factories for file loading**: `Class::load_from_file(path)`.
- **A declaration from a `.hpp` is defined in the matching `.cpp`** — never parked in an
  unrelated translation unit. New types get a matching `include/` + `src/` pair.

`include/fire_engine/graphics/image.hpp` is the reference class for all of the above. [`CLAUDE.md`](../CLAUDE.md)
holds the same rules in condensed form.

## Recommended Onboarding Route

Work through the code in this order. Each step introduces one new layer of complexity.

### 1. Value Types

Start here because these classes are small, heavily tested, and used everywhere.

- `Vec2`, `Vec3`, `Vec4`: numeric vector types with constexpr arithmetic and component
  accessors. `Vec3` also provides operations used by lighting, transforms, normals, and
  physics response. `magnitude()` / `normalise()` call `std::sqrt` and are intentionally
  *not* `constexpr` (sqrt only became constexpr in C++26). `operator==` is strict bit
  equality — use `approxEqual(rhs, eps)` for tolerance-based comparison (or `bitwiseEqual`
  if you want to name the bit-identity intent explicitly). Vec3 ↔ Vec4 conversion is
  `explicit` in both directions to prevent silent w-component loss/gain.
- `Mat4`: column-major transform/projection matrix type. Look at translation, rotation,
  scale, perspective, and look-at helpers. Renderer, scene traversal, skinning, and physics
  transforms all depend on this behaving predictably. Same `approxEqual` / `bitwiseEqual`
  convention as the vector types.
- `Quaternion`: runtime rotation representation for scene transforms. glTF rotations round
  trip better through quaternions than Euler angles. Animation uses SLERP for rotation
  interpolation.
- `Colour3`: simple RGB value used by materials and lights.

Useful tests:

- `tests/math/`
- `tests/graphics/test_colour3.cpp`

### 2. Input State

Input is deliberately packaged before it reaches the scene.

- `CameraState`: movement, yaw, pitch, and zoom deltas for camera control.
- `ControllerState`: gameplay-style movement input used by controllable nodes.
- `AnimationState`: requested animation selection and playback state.
- `VariantState`: helper for typed state bundles.
- `InputState`: per-frame bundle passed into `SceneGraph::update`.
- `Input`: GLFW-facing input collector that fills `InputState`.

Interesting detail: components do not poll GLFW directly. They read the already-normalized
`InputState`, which keeps platform input out of scene component code.

### 3. Scene Basics

The scene layer is the easiest place to understand engine runtime behaviour.

- `Transform`: local position, quaternion rotation, scale, and cached local/world matrices.
- `Components`: `std::variant<Empty, Animator, Camera, Mesh, Light, ParticleEmitter>`. Inspect it through
  `node.componentAs<T>()` (returns `T*`, or nullptr if that's not the active alternative) and
  `node.visitComponent(visitor)` rather than reaching for `std::get_if` / `std::visit` at call
  sites.
- `Empty`: structural no-op component for grouping and joints.
- `Camera`: consumes `CameraState` and exposes world position/target for rendering.
- `Animator`: samples an `Animation` and contributes a model matrix to descendants.
- `Mesh`: wraps a graphics `Object`; updates morph weights and emits draw commands during
  render traversal.
- `Light`: runtime point, spot, or directional light component. `SceneGraph::gatherLights`
  converts light nodes into packed render-facing `Lighting` records.
- `Controllable`: optional node-side gameplay movement component, separate from the variant.
- `Node`: owns transform, component, optional controllable, optional physics handles, child
  nodes, parent pointer, and cached `composedWorld`.
- `SceneGraph`: owns root nodes, updates traversal, render traversal, light gathering, and
  physics submit/apply passes.

`std::formatter<Node>` and `std::formatter<SceneGraph>` live in dedicated headers
(`scene/node_format.hpp`, `scene/scene_graph_format.hpp`) so the hot `node.hpp` /
`scene_graph.hpp` headers do not drag `<format>` into every TU. Include the format header
only at call sites that need `std::format("{}", node)` or `log::info(..., "{}", scene)`.

### Runtime Diagnostics

Use `fire_engine/core/log.hpp` for runtime diagnostics. Engine code should not write directly to
`std::clog`, `std::cout`, `std::cerr`, `std::print`, or printf-family APIs; the logger owns stderr
formatting and level/category filtering.

`FE_LOG` is read once on first log use. With no `FE_LOG`, the global level is `warn`, so warnings
and errors are visible while informational/debug noise stays quiet. Set a global level with:

```bash
FE_LOG=debug ./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr
FE_LOG=off ./fireEngineApp
```

Set category-specific levels with comma-separated `category:level` entries. Later entries for the
same category win:

```bash
FE_LOG=ragdoll:debug ./fireEngineApp CesiumMan/CesiumManRagdoll.gltf skybox.hdr -f
FE_LOG=warn,gltf:info,render:debug ./fireEngineApp
```

Supported levels are `debug`, `info`, `warn`, `error`, and `off`. Current categories are:
`app`, `general`, `gltf`, `physics`, `ragdoll`, and `render`.

Per-frame scene flow:

```cpp
scene_.update(input_state);
scene_.submitPhysics(physics_);
accumulator += dt;
while (accumulator >= fixedDt) { physics_.step(fixedDt); accumulator -= fixedDt; }
const float alpha = accumulator / fixedDt;   // CR-20 render interpolation
scene_.applyPhysics(physics_, alpha);
renderer_->drawFrame(...);
```

`submitPhysics()` pushes static and kinematic scene transforms into `PhysicsWorld`.
`applyPhysics(physics, alpha)` pulls dynamic and corrected kinematic transforms back into
nodes, then calls `resolve()` so composed-world matrices are current before rendering. The
sim runs at a fixed 60 Hz but the display is faster, so `alpha = accumulator / fixedDt` blends
each body between its pose at the start of the last step and its current pose (position lerp,
orientation slerp) — smooth motion on a 120 Hz panel. It is purely visual: `bodyTransform()`
(the sim state) is untouched, so determinism is unaffected. Articulated ragdoll bones aren't
body-bound, so `Ragdoll::syncNodes(alpha)` interpolates their link transforms the same way.

`drawFrame` takes no camera argument — it pulls the active camera from the scene through the seam
(`scene.activeCamera()`). That's why `scene_.update(input_state)` (which refreshes camera world
transforms) **must** run before `drawFrame` in the loop above. The scene owns which node is the
active camera (`SceneGraph::activeCamera(Node*)`); the renderer only reads its pose.

Useful files:

- `src/scene/node.cpp`
- `src/scene/scene_graph.cpp`
- `tests/scene/`

### 4. Animation

Animation is data-first and renderer-independent.

- `Animation`: stores translation, rotation, scale, and morph-weight keyframes. It supports
  LINEAR, STEP, and CUBICSPLINE interpolation. Rotation interpolation uses SLERP.
- `AnimationSelection`: used by input/runtime selection so an `Animator` can switch clips.
- `Animator`: scene component that samples an `Animation` each frame and passes its sampled
  transform down the node tree.

Interesting detail: morph weights are sampled separately from the TRS matrix so a mesh can
animate vertex deltas without changing node transforms.

### 5. Graphics Data

The graphics layer is still Vulkan-free.

- `Vertex`: position, colour, normal, tangent, two UV sets, joints, and weights.
- `Joints4`: fixed four-joint influence container for skinning.
- `Image` and `KtxImage`: CPU-side image containers.
- `SamplerSettings`: filter and wrap modes imported from glTF samplers.
- `Texture`: stores a `TextureHandle` after upload through `Resources`.
- `Material`: PBR state, alpha mode, unlit flag, plus the texture and extension data. The ten
  material textures live in one array reached by `material.texture(MaterialTextureSlot::X)`,
  returning a `TextureSlot{ texture, texCoord, transform, has() }` — there are no per-slot
  `baseColorTexture()`/`hasBaseColorTexture()` getters any more. The clearcoat / transmission(+ior)
  / volume extensions are `std::optional<ClearcoatParams>` / `<TransmissionParams>` / `<VolumeParams>`
  blocks (`material.clearcoat()` etc.), engaged only when the glTF authored the extension. Core PBR
  scalars (`baseColor()`, `roughness()`, `metallic()`, `normalScale()`, …) stay as direct
  getters/setters. The GPU-side UBO base-colour field is still called `diffuseAlpha` to match the
  shader.
- `Geometry`: vertices, indices, material pointer, GPU buffer handles, and morph target
  deltas. `load()` also builds discrete **LODs** for static meshes above `kMinLodTriangles`
  (`graphics/lod.hpp` + `graphics/mesh_simplifier.hpp` — see § Mesh LOD); all levels index the
  same vertex buffer, so a LOD is only extra index data.
- `Skin`: joint node pointers and inverse bind matrices; computes joint matrices from cached
  node world transforms.
- `Assets`: central arrays of textures, materials, geometries, skins, animations, and lights.
- `Object`: graphics-side renderable. It writes mapped UBOs and emits `DrawCommand`s.
- `DrawCommand`: backend-agnostic handle bundle consumed by the renderer.
- `FrameInfo`: Vulkan-free per-frame data passed from render to graphics code.

Interesting detail: alpha pipeline selection happens in `Object::render()` from material
state, but the selected pipeline is only a `PipelineHandle`. Vulkan binding happens later in
`Renderer`.

### 6. Collision And Physics

Read [`collision.md`](collision.md) alongside this section for authoring details.

Low-level collision:

- `AABB` (in `collision/aabb.hpp`): pure value type with `min`/`max` plus `axisMin(Axis)`,
  `axisMax(Axis)`, `center()`, `extent()`. The `Axis` enum lives next to it. Every
  collision/physics consumer (Collider, narrowphase, broadphase, gltf_loader, object.cpp)
  uses this single definition — there is no per-TU axis-switch helper any more.
- `ColliderId`: stable id assigned by the broadphase.
- `EndPoint`: one min/max endpoint on one SAP axis. Endpoints live inside `Collider`.
- `Collider`: local bounds, current world AABB, swept AABB, collision layer/mask, id, and six
  endpoints.
- `BroadPhase`: interface (`addCollider`/`removeCollider`/`update`/`rebuild`/`possiblePairs`/
  `validate`) implemented by two interchangeable broadphases. `DynamicAabbTreeBroadPhase`
  (the default) is a fat-AABB BVH; `SweepAndPruneBroadPhase` owns sorted endpoint arrays +
  pair-state masks. Both emit `CollisionPair`s from overlapping swept world bounds. The
  implementation is injectable through `PhysicsWorld(unique_ptr<BroadPhase>)`; the
  `Determinism.BroadphasesAgree` test runs a scene through both and asserts an identical
  end-state hash (`PhysicsWorld::contacts()` sorts pairs into a canonical order so the
  order-dependent solve doesn't depend on which broadphase produced them).
- `NarrowPhase`: `collide(a, b, speculativeMargin)` builds a shape-specific `ContactManifold`
  (sphere/box/capsule analytic + box/box SAT; any **convex-hull** pair via GJK/EPA +
  face-clip). A non-zero margin also emits *separated-but-approaching* gap contacts (negative
  penetration) for speculative-margin CCD.
- `support.hpp` / `gjk_epa.hpp`: support functions + `gjkEpaContact` (GJK distance + EPA depth)
  — the universal convex query the convex pairs and future mesh collision use.
- `WorldShape` / `ContactManifold` / `geometry.hpp`: neutral world-space shapes
  (`WorldSphere`/`WorldBox`/`WorldCapsule`/`WorldConvex`), the manifold value type (normal +
  up to four points), and reusable closest-point primitives.

Runtime physics:

- `PhysicsBodyHandle` / `PhysicsColliderHandle`: opaque scene-to-physics links.
- `PhysicsBody`: body type, linear/angular velocity, mass/inverse mass, local inverse inertia
  (diagonal), gravity scale, and material.
- `ColliderShape`: authored AABB, box, sphere, capsule, or convex hull. The broadphase uses a
  local AABB; `PhysicsWorld::worldShape` composes the authored shape + body transform into a
  `WorldShape` for the shape-specific narrowphase. `ConvexHullShape` is built from a mesh by
  `buildConvexHull` (`core/convex_hull_builder.hpp`), authored via glTF `Shape: "ConvexHull"`.
- `AabbBvh<T>` (`collision/aabb_bvh.hpp`, P6): the generic fat-AABB BVH core (proxies +
  insert/remove/move/query). The dynamic-tree broadphase wraps `AabbBvh<Collider*>`; a static
  mesh collider wraps `AabbBvh<int>` of triangles.
- Compound + static mesh (P6): `createCompoundCollider` (a body with offset child colliders +
  aggregated COM/inertia) and `createMeshCollider` (`StaticMeshShape`, a per-collider triangle
  BVH; `contacts()` expands a mesh pair into per-triangle contacts). Authored via glTF
  `Shape: "Compound"`/`"Mesh"`. Bodies integrate about a true `centerOfMassLocal`.
- `DebugContact`: per-manifold-point debug record (point + normal), Vulkan-free.
- `solver_math.hpp`: the shared per-body solver primitives (`SolverBody`, world inverse inertia,
  relative velocity, effective mass, apply-impulse) used by both the contact and joint solvers.
- `ContactSolver`: **TGS soft-step** solver (P9.2) (`prepare`/`warmStart`/`solveVelocity(useBias)`/
  `applyRestitution`/`solvePosition`/`store`) over a flat `SolverBody` view. Normal + friction
  impulses with lever-arm torque; penetration is a **soft constraint** (`b2MakeSoft`) resolved by
  the caller's per-substep bias→integrate→relax loop, restitution is one end-of-step pass at the
  true impact velocity, and `solvePosition` is the leftover **kinematic-only** split-impulse pass.
  Persistent warm-start cache. **Full rotation (P3):** per-body world inverse inertia, angular
  velocity, orientation; a centred contact reproduces the linear case. Constants in
  `physics_constants.hpp` (`kSubstepCount`, `kContactHertz`, …).
- `Joint` / `JointSolver` (P4, soft P9.1): distance/ball-socket/hinge constraints as generic
  constraint rows (full Jacobian), interleaved with contacts in the substep loop; a **soft /
  compliant** (`b2MakeSoft`) bias — not hard Baumgarte — so the joint correction dissipates rather
  than pumps energy, with point-anchor error recomputed per substep. Warm-started. Hinge-angle and
  ball-socket cone-twist **limits** via swing-twist quaternion decomposition.
  `PhysicsWorld::createJoint`/`destroyJoint` manage them.
- `Ragdoll` (`scene/ragdoll.hpp`, P4): builds a capsule body + joint per skeleton bone and
  drives the bones through a `Node` **world-override** (`activate()`), so the skinning path
  (Skin reads `composedWorld`) renders the simulated pose. Authored via glTF `extras.Ragdoll`.
- `buildIslands` (`physics/island.hpp`, P5): union-find partition of the movable bodies into
  connected components (linked by movable-movable contacts/joints). `PhysicsWorld` solves
  **per island** (`solveIsland`), and **sleeps** a whole island once its dynamic members settle
  (`sleepTimer ≥ kSleepTime`) — zeroing velocities and skipping it until disturbed. `wake` /
  `sleeping` / `sleepingEnabled` + per-body `allowSleeping` control it.
- `PhysicsWorld`: owns bodies, colliders, shapes, the broadphase (`unique_ptr<BroadPhase>`,
  default AABB tree, injectable via the constructor), narrowphase, the `ContactSolver`, the
  `JointSolver`, joints, the fixed step, and transform queries. The implementation is split across
  `src/physics/physics_world*.cpp`: core creation/solve/sleep in `physics_world.cpp`, shape
  composition in `_shapes.cpp`, queries in `_queries.cpp`, articulations in `_articulation.cpp`,
  debug extraction in `_debug.cpp`, and collider/event export in `_events.cpp`.
- Spatial queries (P7): `raycast`/`raycastAll`/`shapecast`/`overlapSphere`/`overlapShape` with a
  layer/mask `QueryFilter` (`physics/physics_query.hpp`; `collision/ray.hpp` +
  `collision/shape_cast.hpp` are the primitives — analytic ray/shape, GJK conservative-advancement
  sweep). Brute-force over active colliders today (BVH acceleration is a recorded follow-up).
- Trigger / collision events (P7): a collider with `isTrigger` generates overlap **events**, not a
  solver response; `triggerEvents()`/`collisionEvents()` return per-step enter/stay/exit
  `ContactEvent`s (`physics/collision_event.hpp`), diffed from the overlap set each step.
- `CharacterController` (P7, `physics/character_controller.hpp`): a kinematic-capsule
  collide-and-slide controller over the queries (slope limit, step up/down, grounded snap). A
  headless engine class driven from `FireEngine::mainLoop` (`-k` demo), **not** a scene component —
  see [`roadmap.md`](roadmap.md) § Design reviews for why.

Authority rules:

- `Static`: scene transform is submitted to physics; physics does not move it; scene does not
  apply it back.
- `Kinematic`: scene/input submits a target transform; physics may slide/correct it; scene
  applies the corrected transform back.
- `Dynamic`: physics owns transform and velocity; scene does not submit it; scene applies it
  back after stepping.

Sharp edge: `SweepAndPruneBroadPhase` is non-owning. Do not move or destroy a registered
`Collider` before unregistering it, because endpoint arrays store pointers to endpoints owned
inside the collider.

### 7. Loading

`GltfLoader` is the largest translation layer in the engine. It maps glTF into `Assets`,
`SceneGraph`, `Resources`, and `PhysicsWorld`.

What it handles:

- required-extension validation
- buffer/accessor reads
- mesh primitives and index data
- material factors, textures, samplers, UV sets, and UV transforms
- normal generation when source normals are missing
- tangent generation when base or clearcoat normal mapping needs tangents
- skins and inverse bind matrices
- TRS and morph-weight animations
- `KHR_lights_punctual`
- camera adoption
- `extras.Controllable`
- `extras.Physics`
- `extras.Cloth` — turns the node's mesh into a GPU-simulated cloth (`SoftBodySystem`): welds
  vertices to particles, builds structural + triangle-adjacency bend constraints, and registers it
  with the solver. Fields: `Pin` (`None`/`TopCorners`/`TopEdge`), `Compliance`, `BendCompliance`.
  Sample assets: `assets/ClothSheet/ClothSheet.gltf` (a small `TopCorners`-pinned sheet) and
  `assets/ClothBanner/ClothBanner.gltf` (a larger `TopEdge`-pinned banner).

Interesting detail: if a node has animation channels, the loader may build a hierarchy that
keeps the original glTF transform separate from the sampled animator transform. That protects
authored base transforms while still letting animation override the intended channel.

Useful files (all define `GltfLoader::` members declared in one `gltf_loader.hpp`; the `.cpp` is
split by concern):

- `src/core/gltf_loader_scene.cpp` — `loadScene` entry point (drives parse → asset load → node/scene build)
- `src/core/gltf_loader_nodes.cpp` — node loading + hierarchy, physics-body creation, `Controllable`/TRS application
- `src/core/gltf_loader_assets.cpp` — `parseAsset`, extension validation, asset presize, `extras` dispatch
- `src/core/gltf_loader_extras.cpp` — `extras.Controllable`/`Physics`/`Cloth`/`Ragdoll` parsing
- `src/core/gltf_loader_mesh.cpp` — primitive types/bounds, convex-hull + triangle extraction, `loadMesh` (assembles texture slots)
- `src/core/gltf_loader_geometry.cpp` — geometry/vertex loading, UV transforms, smooth-normal + tangent generation
- `src/core/gltf_loader_material.cpp` — material factor + texture-slot resolution
- `src/core/gltf_loader_images.cpp` — centralized image source resolution → `Image`/`KtxImage`, texture-index lookup
- `src/core/gltf_loader_animation.cpp` — animation channels + skin loading
- `tests/core/test_gltf_loader.cpp`

Sharp edge: material texture slots are resolved and applied through one centralized loader path.
Base normal and clearcoat normal textures are both tangent-space inputs; if tangents cannot be
loaded or generated, both slots must be skipped consistently with a warning.

Image source loading is also centralized. `src/core/gltf_loader_images.cpp` resolves each glTF image
once into either a direct local file path or an in-memory byte buffer, then feeds that source to
`Image` or `KtxImage`. Keep PNG/JPEG/HDR and KTX2 changes on that shared path so embedded images
and local URI images stay behaviorally aligned.

### 8. Render Resources

`Resources` is the bridge between Vulkan-free graphics data and Vulkan objects.

- Creates vertex/index/storage/uniform buffers. Three private building blocks pick the memory class;
  public methods only choose usage flags + initial data: **`createDeviceLocalBuffer`** for long-lived
  GPU-only data that's written once and never CPU-touched again (static vertices/indices — including
  every LOD cut — and the VIPM geomorph table via `createStaticStorageBuffer`); it stages through a
  transient host-visible buffer + a one-time copy. **`createHostVisibleBuffer` / `createMappedHostVisibleBuffers`**
  for data the CPU (re)writes — per-frame UBOs, the VDPM dynamic index set, debug lines, and the
  soft-body/cloth buffers (`createStorageBuffer`, `createSharedStorageBuffer`, cloth's
  `createStorageVertexBuffer`). **Invariant: static geometry is device-local; anything CPU-written
  per frame stays host-visible** — don't route a per-frame buffer through the staging path (it would
  pay a copy every frame), and don't leave bulk static geometry host-visible. Buffers and images are
  sub-allocated from the **VMA arena** (`Device::allocator()`) via the `UniqueVmaBuffer`/`UniqueVmaImage`
  RAII wrappers — never a `vkAllocateMemory` per resource (see CLAUDE.md "GPU resource model").
- Uploads textures and fallback textures (8-bit RGBA, HDR float, Basis/KTX2). A single
  `uploadImageFromHost` helper drives the staging buffer / barrier / copy / transition flow
  for every texture variant; a `withOneTimeSubmit` template wraps the boilerplate for any
  one-off command-buffer submission (waiting on a per-submit fence, not `queue.waitIdle`).
  During scene load, `begin/endUploadBatch` coalesces every texture **and device-local buffer**
  upload (static vertices/indices/LODs/VIPM) into one submit + one fence — `createDeviceLocalBuffer`
  records its staging copy into the open batch instead of submitting per buffer. Texture slots are tracked by a `GenerationalSlotPool`, so `releaseTexture` (e.g. on
  resize) recycles the slot and bumps its generation — a stale handle is detectable via
  `validTexture`, not a silent alias.
- Creates common 2D render/sample targets through one private target descriptor helper. Offscreen
  HDR, velocity, bloom, and sceneColor still keep their special cases (nearest velocity sampling,
  bloom per-mip views, sceneColor initial shader-read layout), but shared image/view/sampler setup
  belongs in that helper.
- Owns descriptor pools and descriptor sets for the per-object **shadow** set 0 plus the
  per-frame forward-globals set 1. The **forward** set 0 is no longer allocated — it is a
  `VK_KHR_push_descriptor` layout pushed inline per draw (`pushForwardObjectDescriptors`). Forward
  set 0 carries two UBOs: a slim per-object **`ObjectUBO`** (model/hasSkin/previousModel, binding 0)
  and a per-frame **`CameraUBO`** (view/proj/cameraPos/view-projections, binding 29). `CameraUBO`
  lives in set 0 (not the global set 1) on purpose — the depth prepass reuses `shader.vert` but binds
  no globals, and set 0 is already pushed there. The Renderer writes `CameraUBO` **once per frame**;
  `Object` re-writes its `ObjectUBO` only when world/previousWorld/hasSkin change (a static object
  skips it).
  `Descriptors::createGlobalDescriptors` allocates set 1 once at startup;
  `Descriptors::updateGlobalDescriptors` rewrites it after swapchain resize. These pools are
  renderer-lifetime: their sets are allocated once and **never freed individually**, so the pools
  carry **no `eFreeDescriptorSet`** and hold their sets as plain `vk::DescriptorSet` handles (the pool
  owns them; destroying it reclaims them). Don't reintroduce the flag or per-set RAII for this path.
  (The bindless set 2 pool is the exception — it keeps one `vk::raii` set + `eFreeDescriptorSet`.)
- Owns the global **bindless materials set 2** (update-after-bind): one `sampler2D[]`
  texture array (`registerBindlessTexture`, indexed by texture handle) + a materials[]
  SSBO (`registerMaterial`, dedup by `Material*` → the per-draw material index).
- Owns pipeline registry and maps `PipelineHandle` to Vulkan pipeline/layout pairs.
- Owns full directional, world-only directional, dual-depth per-skinned-object self-shadow,
  spot, and point shadow map resources and descriptors. Shadow passes are depth-only (dynamic
  rendering commits depth-only stores on current MoltenVK); the shadow-depth debug view samples
  the CSM depth map directly.
- Owns HDR offscreen targets, bloom chain, post-process descriptors, IBL cubemaps, and BRDF
  LUT handles.
- Exposes a `SharedTextures` aggregate (`Resources::sharedTextures()`) holding the eleven
  globals that the forward pass binds via set 1 (shadow maps, IBL, sceneColor, debug image).

Interesting detail: many objects keep mapped pointers for per-frame UBO writes. Always write
the slot for `currentFrame`; do not treat mapped UBO memory as a single global struct.

Camera basis construction uses the shared math helper in `view_basis.hpp`. Use it for new
view, skybox, sort-depth, or shadow-fit code so zero-length camera vectors and vertical
look directions keep producing finite right/up vectors.

### 9. Vulkan Render Layer

The render layer is where most difficult bugs live.

- `Device`: instance, surface, physical/logical device, queues, memory type selection, and
  buffer helpers.
- `Swapchain`: swapchain images, views, extent, depth resources, and recreation.
- `Frame`: command pool, command buffers, and sync objects — binary `imageAvail` / `renderDone`
  semaphores for the WSI acquire/present path, plus one monotonic **timeline semaphore** for
  CPU↔GPU frame pacing (replaced the per-frame in-flight fences). The renderer tracks per-slot
  and per-image timeline values to gate command-buffer / swapchain-image reuse.
- `RenderTarget` (`render_target.hpp`): a thin descriptor carrying attachment formats (fed to
  `VkPipelineRenderingCreateInfo` at pipeline creation) plus a `makeRenderingInfo` helper. Each
  subsystem builds its `vk::RenderingInfo` from image views at record time and brackets the draw
  with explicit synchronization2 layout-transition barriers — there are no `VkRenderPass` or
  `VkFramebuffer` objects.
- `Pipeline`: descriptor layouts, pipeline layouts, and graphics pipelines. Forward
  pipelines declare `PipelineConfig::bindings` (set 0, per-object), `globalBindings`
  (set 1, forward globals), and `bindlessSet` (set 2, the bindless texture array +
  materials SSBO with update-after-bind binding flags); non-forward pipelines leave
  globals/bindless off and use a single-set layout. Fullscreen / fragment-only config factories
  share small helpers in `pipeline.cpp`; preserve their returned state and update
  `tests/render/test_pipeline_config.cpp` when intentionally changing it.
- `Descriptors`: descriptor layout/write helpers. Neither the forward nor the shadow set 0
  is allocated — the free functions `pushForwardObjectDescriptors` /
  `pushShadowObjectDescriptors` push their per-object buffers inline per draw
  (`VK_KHR_push_descriptor`); the shadow push also carries the shared self-shadow
  image+sampler read from `Resources`. `createGlobalDescriptors` allocates `kMaxFramesInFlight`
  set-1 descriptors with the shared globals (light UBO, shadow maps, IBL textures, scene
  colour). `updateGlobalDescriptors` rewrites those globals after swapchain resize so
  Transmission's recreated sceneColor sampler doesn't leave dangling descriptor references.
- `EnvironmentPrecompute`: startup equirectangular-to-cubemap, irradiance, prefilter, and
  BRDF LUT generation.
- `Shadows`: cascaded directional shadows, spot shadow layers, point cubemap-array shadows,
  depth-only shadow rendering (per-layer depth views bound directly), and shadow pass replay.
- `PostProcessing`: HDR target, bloom, and post-process setup helpers.
- `ComputePipeline` (`compute_pipeline.hpp`): compute counterpart to `Pipeline` (single compute
  stage), plus synchronization2 buffer-barrier helpers (`makeBufferMemoryBarrier` /
  `recordBufferBarrier`).
- `ParticleSystem`: renderer-owned GPU particle system. Owns a pooled particle SSBO, simulates
  it with compute, and renders instanced additive billboards into the HDR target (soft particles
  via sampled scene depth). Emitters come from `SceneGraph::gatherEmitters` (the `ParticleEmitter`
  component, gathered like `Light`).
- `SoftBodySystem`: renderer-owned GPU XPBD cloth solver (`-c`, or any glTF `extras.Cloth` node).
  Descriptor-free — the four compute pipelines (`cloth_predict`/`solve`/`collide`/`finalize`) reach
  every buffer (particles, constraints, render verts, index list, CSR normal adjacency, per-frame
  colliders) as a `bufferDeviceAddress` pointer in the push constant. `recordSolve` runs the substep
  dispatch chain (predict → per-colour solve → collide → finalize) writing solved positions +
  recomputed normals into the cloth's storage vertex buffer, which the shadow/forward passes read.
  Normals come from the CSR per-vertex→triangle adjacency (arbitrary topology, not just a grid).
  Colliders come from `PhysicsWorld::gatherColliders`; solver params (substeps/compliance-scale/
  damping/gravity/wind) come from `RenderTunables` — the compliance slider is a global multiplier on
  each constraint's authored per-type stiffness. The CPU-side cloth mesh, per-type constraint build,
  graph-colouring, and normal adjacency are the Vulkan-free `graphics/cloth.hpp` (`makeGridCloth` /
  `makeClothFromMesh`).
- `Taa`: temporal anti-aliasing. Owns the velocity target + ping-pong history targets and the
  resolve pass (reproject along velocity → neighbourhood clamp → blend → optional sharpen → blit
  into the HDR target).
- `Ssao`: screen-space ambient occlusion + contact shadows. Runs after the depth prepass
  (`recordDepthPrepass`), borrows the shared scene depth (attachment → read-only → attachment),
  reconstructs view position+normal from depth alone, writes a raw R8G8 target (R = AO,
  G = contact), then a **depth-aware bilateral blur** into a second target the forward shader
  samples. Always runs (disabled = intensity 0 → AO 1, no conditional barriers). `recreate()` on
  resize.
- `DebugDraw`: physics debug wireframes (broadphase AABBs, collider shapes, contact normals)
  built CPU-side into a per-frame mapped vertex buffer (reusing `Vertex`) and drawn via a
  line-list pipeline into the HDR target after particles. Depth test is a dynamic state (x-ray
  vs occluded). Fed each frame by the main loop from `PhysicsWorld::gatherColliders()` /
  `debugColliderBounds()` / `debugContacts()` (all Vulkan-free); gated on `physicsDebugWanted()`.
- `DebugOverlay`: Dear ImGui context + GLFW/Vulkan backends. Draws the overlay into the swap image
  (dynamic rendering, after post-process) and forwards `WantCaptureMouse/Keyboard` so the main loop
  can gate camera input. Non-movable (owns ImGui global state).
- `GpuProfiler`: timestamp `VkQueryPool` ring. Each pass writes a begin/end pair; results are read
  back a frame-cycle later into a `FrameStats`. Disabled gracefully when timestamps are unsupported.
- `RenderTunables` (`render_tunables.hpp`): the live, overlay-editable render parameters (TAA,
  debug view, bloom/IBL/sun, particle scales, cloth solver substeps/compliance/damping/gravity/wind).
  Seeded from `constants.hpp` + the CLI debug flags; the `Renderer` reads it every frame instead of
  the `constexpr`s. `DebugView` lives here too.
- `Renderer`: frame orchestration, light gathering, command recording, pass ordering, submit,
  present, and swapchain recreation. Holds `tunables_` and reads it for the debug view, IBL/bloom/
  sun strengths, TAA jitter + resolve params, and particle emitter scales.

Per-frame render order:

1. acquire swapchain image
2. gather lights and update frame data
3. collect scene draw commands
4. run the soft-body/cloth solver (compute) — writes solved positions + normals into the cloth vertex buffers the shadow/forward passes read
5. record directional, spot, and point shadow passes
6. record HDR forward pass (writes HDR colour + a screen-space velocity attachment)
7. capture scene colour and record the transmission pass when needed
8. resolve TAA — reproject + accumulate history along the velocity buffer, blit back into the HDR target (skipped under `--no-taa`)
9. simulate particles (compute) and render them additively into the (resolved) HDR target
10. record bloom downsample and upsample passes
11. record ACES/gamma post-process into the swapchain image (leaves the swap image in colour-attachment layout)
12. draw the ImGui debug overlay over the swap image, then transition it to present
13. submit and present

Each GPU pass is bracketed by `GpuProfiler` timestamp writes; the results feed the overlay one frame-cycle later.

Interesting detail: shadow rendering replays compatible draw commands through different
pipelines. Skinning and morph targets still run in the shadow vertex shader so animated
geometry casts matching shadows. Skinned meshes receive world shadows from a world-only CSM
and self-shadow from a per-object dual-depth map that rejects the first light-facing surface.
Punctual shadow receiving is still active.

The dual-depth self-shadow path deliberately uses two passes. A single depth map stores the
nearest light-facing surface, which is usually the receiver itself and causes light-side acne.
The first self-shadow pass renders both face orientations to capture the nearest surface; the
second pass culls front faces (`cullMode = eFront`) so only back-facing geometry rasterises,
giving the forward shader the next useful occluder behind the front surface. An in-shader
discard against `skinnedSelfShadowDepthEpsilon` remains as a safety net for non-closed meshes
but does not gate per-fragment coverage on closed meshes. Only `kMaxSkinnedSelfShadowCasters`
objects get these per-object slots each frame; additional skinned objects still cast into the
full directional CSM for world receivers.

Debugging flags are parsed in `ApplicationArgs` into a `RendererDebug { DebugView view;
bool noShadows; bool taa; }` struct, passed through `FireEngine::run` into the `Renderer` ctor.
The `DebugView` integer and `noShadows` are packed into `LightUBO::environmentParams` (`.z` / `.w`)
and consumed by `shader.frag`; `taa` is read directly in `Renderer::drawFrame` to gate the jitter
and resolve pass:

- `--debug-normals`: show fragment normals.
- `--debug-ndotl`: show primary directional `NdotL`.
- `--debug-shadow`: show primary directional shadow visibility.
- `--debug-shadow-depth`: show directional receiver/stored depth and cascade index.
- `--debug-velocity`: visualise the TAA motion-vector buffer (|x|, |y| scaled). Zero for a still
  camera on rigid/skinned geometry; grows with camera or node-transform motion.
- `--no-shadows`: disable shadow-map visibility lookups.
- `--no-taa`: skip the projection jitter and TAA resolve — reverts to the raw aliased image.
- `--overlay`: start with the ImGui debug overlay visible (also toggled at runtime with **F1**).
- `-f`: add a receiver-only floor plane at y=0.
- `-p`: seed the demo GPU particle fountain (off by default).
- `-c`: drop a demo cloth that drapes over a sphere onto the ground (GPU XPBD soft-body). Cloth can
  also be authored on any glTF mesh via `extras.Cloth` (e.g. `assets/ClothSheet/ClothSheet.gltf`).

Most of these flags are now also live controls in the debug overlay (`RenderTunables`): the debug
view is a dropdown, `--no-shadows`/`--no-taa` are checkboxes, and TAA blend/sharpen, bloom, IBL/sun
strengths, and particle emitter scales are sliders. The CLI flags seed the overlay's initial state.

### 10. FireEngine

`FireEngine` is the top-level application object.

- Owns `Window`, `Renderer`, `Input`, `SceneGraph`, `Assets`, `PhysicsWorld`, and active
  `Camera*`.
- Loads a scene through `GltfLoader`.
- Seeds a default directional sun only when the loaded scene has no authored directional
  light.
- Optionally adds a receiver-only floor plane from `-f` so shadow-casting scenes have a
  simple ground target.
- Optionally seeds the demo GPU particle fountain from `-p` (a `ParticleEmitter` node).
- Runs the fixed-timestep physics accumulator inside the variable-rate render loop.

Start at `src/fire_engine.cpp` when you need to understand whole-frame behaviour.

## Common Engineering Tasks

### Trace A Mesh From glTF To Pixels

Start in `GltfLoader`, follow primitive loading into `Assets`, `Geometry`, `Material`, and
`Object::load`. GPU allocations go through `Resources`. At runtime, `Mesh::render` calls
`Object::render`, which returns `DrawCommand`s. `Renderer` buckets those commands, resolves
their handles, binds Vulkan state, and records `drawIndexed`.

### Frustum Culling

Two stages, both gated by `RenderTunables::cullingEnabled` (overlay toggle; off = submit
everything, the A/B + regression escape hatch). `graphics/Frustum` (Vulkan-free, testable)
extracts 6 Gribb–Hartmann planes from a viewProj and does a positive-vertex AABB test —
*conservative*, never a false negative.

- **Coarse pre-cull** (`scene/SceneCuller`, owned by `SceneGraph`): a persistent
  `AabbBvh<Node*>` over rigid renderable nodes. `collectDrawCommands` builds the frustum
  set (camera + every shadow caster) and calls `scene.cull()`; `Node::render` skips
  draw-building for nodes outside *all* of them (still recurses children). Deformable
  (skinned/morph) nodes aren't tracked — their bind-pose AABB under-covers the animated
  mesh, so they're always drawn. The BVH is the same `AabbBvh<T>` the physics broadphase
  uses; `traverse(predicate, fn)` is the generic descent the culler queries with a frustum.
- **Precise per-bucket cull**: `buildDrawBuckets` drops non-shadow draws outside the camera
  frustum; `Shadows::recordPass` filters each cascade/spot/point-face draw list against its
  own frustum. Uses the exact per-draw world `shadowBounds` computed every frame, so it is
  always tight even for the deformable nodes the coarse stage skips.

The coarse union is a *superset* of what the per-bucket stage keeps, so a node dropped by
the pre-cull is never wanted by any pass. Overlay shows tracked/visible/culled counts.

### Mesh LOD

Mesh level-of-detail (rendering spine #3 — the full **discrete → VIPM → VDPM** ladder is done), gated
by `RenderTunables::lodEnabled` (overlay toggle + pixel error budget slider; a per-LOD **"LOD tint"**
debug view colours each mesh green/yellow/red by its selected level). `RenderTunables::lodMode`
switches between hard **Discrete** swaps, Continuous **VIPM** geomorphs, and **View-dependent (VDPM)**
per-region refinement. One recorded collapse stream feeds all three. The authority is [`lod.md`](lod.md).

- **Build (load time, `Geometry::load`):** a from-scratch **Garland–Heckbert QEM** simplifier
  (`graphics/mesh_simplifier`) builds one `ProgressiveMesh` per static mesh >
  `kMinLodTriangles` (skipped for deformable/`storageVertices` meshes). The artifact contains the
  ordered collapse stream, exact collapse-count cuts for each LOD, and the coarser index sets. The
  error quadric lives in **R⁵ (position + weighted UV)** so collapses that stretch the texture are
  ordered *last*; **position welding** keeps glTF's seam-split verts connected, a
  **wedge-preserving emit** keeps each corner's closest UV/normal/tangent identity, and **subset
  (endpoint) placement** means every level indexes the base vertex buffer.
- **Select (per draw, `object.cpp`):** `selectLod` picks the coarsest level whose world error (an RMS
  deviation) projects within `lodPixelErrorBudget` pixels, from camera distance + `frame.proj`/viewport;
  shadow draws bias coarser. It sets `cmd.indexBuffer`/`indexCount`/`lodLevel`.
- **Morph (per draw, `object.cpp` + `shader.vert`):** Continuous mode uses `selectVipm` to compute
  a `morphFactor` toward the next exact LOD level. The VIPM SSBO stores, per original vertex, the
  1-based level where that vertex first disappears plus its target position/normal/tangent/UV0/UV1.
  The shader morphs only vertices whose removal level equals `MorphUBO::vipmTargetLevel`.
- **Refine (per draw, `object.cpp` + `graphics/vdpm`):** View-dependent mode builds a per-instance
  `ActiveFront` over a `VertexForest` (the collapse stream promoted to Hoppe vertex-splits with
  dependencies) at load. Each frame `refineForView` resets to coarsest and refines by **four
  screen-space channels** (geometry δ / UV-seam / shading-normal / tangent, each with a `kVdpm*Scale`
  dial) + silhouette boost + a conservative multi-witness back-face gate; then two **monotone repair
  passes** — `repairFoldovers` (backward-wound faces) and `repairCoverage` (silhouette/degenerate
  coverage holes, on the **jitter-free** `currentViewProj`) — close the holes a selective (non-prefix)
  front introduces that `wouldFlip` and the deviation metrics can't see. `emitActiveIndices` restores
  render wedges into a per-frame dynamic index buffer.
- **Two dials** live in `mesh_simplifier.cpp`: `kUvWeightFactor` (UV fidelity vs simplification) and
  `kErrorCeilingFactor` (must only refuse *geometrically* un-simplifiable shapes — the cube via its
  boundary weight — not UV-costly seams). `kLodRatios` / `kLodPixelErrorBudget` / the `kVdpm*` dials
  are in `graphics/lod.hpp`. A simplifier-side **chart veto** (`canonicalCharts_`) forbids a collapse
  from crossing a UV/normal seam, so the VIPM morph never shears the texture across charts.
- **Known residual:** shadow LOD is still discrete and biased coarser, so shadow silhouettes can step
  independently of the forward/depth VIPM/VDPM detail; the VDPM repair passes run on the CPU per frame
  (a per-split cone or GPU-driven front would retire that cost).

### Add A Material Field

Add the CPU field to `Material` — a core scalar getter/setter, a new `MaterialTextureSlot` if it's
a texture, or a member on the relevant `ClearcoatParams`/`TransmissionParams`/`VolumeParams` block
(extend the optional, don't add a loose scalar). Load it in `GltfLoader::loadMaterial`, pack it into
`MaterialUBO` in `toMaterialUBO` (`src/graphics/material_binding.cpp`) — use `value_or({})` for
optional blocks so absent extensions keep their defaults — update
`include/fire_engine/render/ubo.hpp` and the shader, and add/update tests in
`tests/graphics/test_material.cpp` and `tests/render/test_ubo.cpp`.

### Add A Texture Slot Or Shader Binding

First decide which descriptor set the binding belongs on:

- **Set 0 (per-object)** — per-object vertex-stage UBOs (frame/skin/morph) + morph SSBO.
  Lives in `ForwardBinding`, declared in `PipelineConfig::bindings` (a push-descriptor
  layout — `PipelineConfig::pushDescriptorSet0`), pushed inline per draw by
  `pushForwardObjectDescriptors` (carry the new buffer on `DrawCommand` and add it to that
  helper's writes), consumed by `shader.frag`/`shader.vert` as `layout(binding = N)`.
- **Set 1 (forward globals)** — anything shared across every draw, like a global texture or
  per-frame UBO. Lives in `ForwardGlobalBinding`, declared in
  `PipelineConfig::globalBindings`, written by `Descriptors::createGlobalDescriptors` plus
  `Descriptors::updateGlobalDescriptors` (for swapchain-driven recreation), and consumed by
  `shader.frag` as `layout(set = 1, binding = N)`.
- **Set 2 (bindless materials)** — material textures + scalars. Textures go in the global
  `sampler2D[]` array (`Resources::registerBindlessTexture`), material records in the
  materials[] SSBO (`Resources::registerMaterial`); the draw selects its material via
  `ForwardPushConstants::materialIndex`. Lives in `BindlessBinding`; the shader reads
  `materials[pc.materialIndex]` and samples `textures[material.textureIndex[slot]]`. A new
  material *field* means updating `MaterialUBO`/`MaterialData` (C++ + shader) in lockstep;
  no descriptor-binding change.

Then update material storage and glTF loading if applicable, plus the relevant test in
`tests/render/test_pipeline_config.cpp`. If the new binding references a texture that gets
recreated on resize, make sure `Renderer::recreateSwapchain` includes it in the
`buildGlobalDescriptorRequest` snapshot it passes into `updateGlobalDescriptors`.

### Add A Physics Body Feature

Start with `PhysicsBody` and `PhysicsBodyDesc`, then update `PhysicsWorld::createBody`,
solver/step behaviour, glTF `extras.Physics` parsing, tests in `tests/physics/`, and docs in
[`collision.md`](collision.md).

### Extend Shape-Specific Collision

The shape-specific narrowphase already exists: keep broadphase AABB-based, then add the shape
to `collision/world_shape.hpp`, compose it in `PhysicsWorld::worldShape`, and add a
`collidePair(a, b, margin)` overload (returning a `ContactManifold`, normal `b → a`) in
`narrow_phase.cpp` — the 2-arg `std::visit` in `NarrowPhase::collide` picks it up. Respect the
speculative `margin`: reject only when separated by more than it, and report a separated pair
with a **negative** penetration (so CCD works). Put reusable closest-point math in
`collision/geometry.{hpp,cpp}` with isolated tests. For *convex* shapes, add a support function
in `collision/support.hpp` and the pair routes through `collideConvex` (GJK/EPA) automatically —
no per-pair overload needed.

### Add A Render Pass

Create attachments/resources through `Resources`, then create a pipeline/layout whose
`PipelineConfig` declares the attachment formats (`colourFormats` / `depthFormat`) it renders
into — dynamic rendering means there is no `VkRenderPass`/`VkFramebuffer` to set up. At record
time, transition the attachment images into the right layouts with synchronization2 barriers,
build a `vk::RenderingInfo` from their views (see `makeRenderingInfo` in `render_target.hpp`),
and bracket the draws with `cmd.beginRendering` / `cmd.endRendering`. `Renderer::drawFrame` is a
sequence of named phase methods
(`updateFrameLighting`, `collectDrawCommands`, `recordShadowPass`, `recordForwardPass`,
`recordTransmissionPass`, `recordPostProcessing`) — add a matching `recordXxxPass(cmd, ...)`
method and call it from `drawFrame` at the right point in that order rather than inlining the
recording. Update `Renderer::recreateSwapchain` if the resource depends on the swapchain extent.

### Debug A Lighting Or Shadow Issue

Prefer the existing runtime flags before adding one-off shader edits. `--debug-normals`,
`--debug-ndotl`, `--debug-shadow`, `--debug-shadow-depth`, `--debug-velocity`, `--no-shadows`,
and `--no-taa` are wired through the normal app argument path and can be combined with a scene
path, environment path, and `-f`.

## Cross-File Invariants

These are the pairs the compiler can't fully police. When you touch one side, update the other in
the same change — most have a test or guard that will catch you, but not all.

- **GLSL bindings ↔ `render/descriptor_bindings.hpp`.** Every `layout(set = S, binding = N)` in a
  shader must match the corresponding `ForwardBinding` / `ForwardGlobalBinding` / `ShadowBinding` /
  `SkyboxBinding` / `PostProcessBinding` enumerator. `tests/render/test_pipeline_config.cpp` checks
  the C++ side; the GLSL side is on you.
- **UBO/push-constant structs ↔ GLSL std140 layout.** `render/ubo.hpp` structs are memcpy'd into
  mapped GPU memory, so their field order, `alignas`, and padding must mirror the matching GLSL
  block exactly. `tests/render/test_ubo.cpp` asserts sizes/offsets — extend it when you add a field.
- **`MaterialTextureSlot` order ↔ `MaterialUBO::uv[]` / `textureIndex[]` ↔ shader `SLOT_*`.** The
  slot enum (in `graphics/material.hpp`) indexes the per-slot UV array and the per-slot bindless
  texture index. Add a slot in all of these (and the shader's `SLOT_*` constants) together. (Since
  textures are bindless, slots no longer map to fixed descriptor bindings.)
- **`enum class Key` ↔ `kGlfwKeyCodes`** (`platform/keyboard.hpp` / `keyboard.cpp`). The table is
  indexed by enum order and sized to `Key::Count`, so a count mismatch fails to compile — but a
  *reordering* silently maps the wrong physical key. Keep both in the same order.
- **GPU array sizes ↔ shader array sizes.** `graphics/gpu_limits.hpp` (`kMaxLights`, `kMaxJoints`,
  `kMaxMorphTargets`, shadow caster caps, `kShadowTotalMatrixCount`) must equal the array sizes
  declared in the shaders that consume those UBOs.
- **Progressive LOD cuts ↔ VIPM morph targets.** `Geometry::load()` must build runtime LOD index
  buffers and VIPM morph data from the same `ProgressiveMesh`. `ProgressiveLod::collapseCount` is
  topology identity; `GeometryLod::error` is only a screen-space selection metric. Do not reconstruct
  a LOD's collapse set from error thresholds. The shader-side `MorphVertex` layout in
  `graphics/vipm.hpp` must match `shader.vert`'s `VipmVert`, and `MorphUBO::vipmTargetLevel` in
  `render/ubo.hpp` must match the shader block.
- **VDPM shares the simplifier's canonical topology, and refines on screen space not topology.** The
  simplifier, VIPM and the `VertexForest` / `ActiveFront` (`graphics/vdpm`) all weld positions through
  the *one* shared `graphics/mesh_topology::weldByPosition` (glTF seam duplicates fuse to one canonical
  vertex) — a single impl so the welds *cannot* diverge and the recorded collapses' canonical indices
  always line up. The forest no longer re-derives split adjacency by replaying the stream: the
  simplifier records each collapse's `vl`/`vr` apexes as it coarsens (`kNoCollapseApex` where a
  non-manifold welded edge can't be encoded), and `buildVertexForest` transcribes them — so keep the
  weld shared and keep `vl`/`vr` recorded on `MeshCollapse` (don't reintroduce a replay).
  `refineForView`/`repairCoverage` are
  screen-space: `repairCoverage` **must** be fed the **jitter-free** `FrameInfo::currentViewProj`, not
  the TAA-jittered `FrameInfo::proj` (the sub-pixel jitter would thrash the front frame-to-frame). The
  two repair passes exist because a *selective* front is a non-prefix cut of the stream, so the
  simplifier's linear `wouldFlip` and the deviation metrics don't cover its foldover / coverage holes —
  do not delete them assuming a "closed, non-folded" emit is hole-free.
- **The scene owns the active camera; the renderer reads it through the seam.** The camera crosses
  `graphics/renderable_scene.hpp` via `activeCamera()` (returns a `CameraView{position, target}`),
  exactly like lights/emitters/draws — it is *not* a `drawFrame` argument. `SceneGraph` holds the
  active camera as a `Node*` and returns its **live** world pose, so `SceneGraph::update()` (which
  refreshes camera world transforms) **must** run before `Renderer::drawFrame` reads it (the main loop
  already orders them so). Don't reintroduce a camera parameter on `drawFrame`, and don't cache the
  camera pose across the update→draw boundary.
- **Where a constant lives.** Scalar render tunables (biases, strengths, extents, FOV) go in
  `render/constants.hpp`; GPU data-layout limits the Vulkan-free graphics layer also needs go in
  `graphics/gpu_limits.hpp`. `constants.hpp` includes the latter, so render-side code still sees
  everything through one include.

## Sharp Edges

- `graphics/` must remain Vulkan-free, and `render/`/`scene/` are siblings that meet only through `graphics/` — enforced by the `layering_guards` CTest case (`cmake/check_layering.cmake`): `graphics/` and `scene/` *headers* must not include `render/`, and `render/` headers must not include `scene/` (CR-09). The scene reaches the renderer through the Vulkan-free `graphics/renderable_scene.hpp` `RenderableScene` interface, not a concrete type. Shared GPU data-layout limits live in `graphics/gpu_limits.hpp` so graphics headers can size arrays without reaching into `render/`. Graphics `.cpp` files may still include `render/resources.hpp` to allocate GPU resources.
- Vulkan-Hpp is built with `VULKAN_HPP_NO_CONSTRUCTORS`; use designated initializers.
- Vulkan structs often contain pointers. Keep pointed-to arrays and descriptor infos alive
  until the Vulkan call using them has returned.
- Descriptor writes are consumed immediately by `updateDescriptorSets`, but the info objects
  they point at must live until that call completes.
- Swapchain-sized resources must be recreated on resize.
- Alpha sorting is per draw command, not per triangle.
- Skinned meshes receive the world-only directional CSM plus their second-depth self-shadow
  map. Keep both maps in mind when evaluating character/object shadow interactions.
- If a skinned mesh casts onto the world but lacks self-shadowing, check whether it exceeded
  `kMaxSkinnedSelfShadowCasters` or failed to produce valid shadow bounds.
- Directional shadow bias is intentionally conservative to keep contact shadows attached.
  Raising caster or receiver bias can reopen visible gaps at ground contact points.
- Collision layer/mask changes on registered colliders require a broadphase `rebuild()`.
- Registered colliders must not move in memory while owned by `SweepAndPruneBroadPhase`.
- Physics contacts are shape-specific (a real `ContactManifold` per pair); the broadphase
  still proxies every shape as an AABB. Contact response is a **TGS soft-step**
  `ContactSolver` (P9.2: substepped solve, warm-started friction + soft-normal impulses, a
  per-substep relax pass, end-of-step restitution) with **full rotational dynamics** (inertia
  tensors, orientation integration, lever-arm torque — boxes topple and rest on a face). Fast movers don't tunnel:
  speculative-margin CCD (motion-expanded swept broadphase + negative-penetration gap
  contacts the solver brakes) catches them.
- Kinematic nodes are submitted from scene to physics before fixed steps, then applied back
  after stepping so collision correction is visible to rendering.
- Double-sided geometry: `shader.frag` flips the shading normal to face the viewer on back faces
  (`if (!gl_FrontFacing) N = -N;`, also `shadowNormal` and clearcoat `N_cc`). Keep this when
  touching normal mapping or any view-dependent term.
- `KHR_materials_transmission` (in `shader.frag`) does screen-space scene-behind-glass refraction
  off the captured `sceneColor`, roughness-blurred via its mip chain — for clear/frosted glass,
  thin-walled or volumetric alike. The **exception**: a thin-walled surface that is *also emissive*
  (a self-lit paper lamp shade) instead scatters to a view-independent irradiance tint, so a bright
  bulb behind it doesn't beam a camera-tracking screen-space blob. The discriminator is the
  **emissive factor**, not thickness (a thickness-based split used to wrongly flatten the
  TransmissionTest grid — see [`roadmap.md`](roadmap.md)).

## Where To Look First

- Startup and frame loop: `src/fire_engine.cpp`
- CLI arguments: `src/platform/application.cpp`, `include/fire_engine/platform/application_args.hpp`
- glTF loading: `src/core/gltf_loader*.cpp` (entry in `_scene.cpp`, nodes/hierarchy in `_nodes.cpp`; assets/extras/mesh/geometry/material/images/animation in the matching `_*.cpp` files)
- Scene traversal: `src/scene/scene_graph.cpp`, `src/scene/node.cpp`
- Physics world: `src/physics/physics_world.cpp` plus the responsibility-specific
  `src/physics/physics_world_*.cpp` files
- Physics demos: `assets/physics_demos/` (one glTF per capability, generated by `generate.py`); behaviour tests in `tests/physics/test_demos.cpp` (`[Demos]`). See [`collision.md`](collision.md) § Physics demos and [`README.md`](../README.md) § Physics Demos for the run commands + what each shows.
- Collision broadphase: `src/collision/dynamic_aabb_tree_broad_phase.cpp` (default), `src/collision/sweep_and_prune_broad_phase.cpp` (alternative), behind `collision/broad_phase.hpp`
- Narrowphase: `src/collision/narrow_phase.cpp`
- Mesh component: `src/scene/mesh.cpp`
- Draw command generation + LOD selection: `src/graphics/object.cpp`
- Mesh LOD / simplifier: `include/fire_engine/graphics/lod.hpp`, `src/graphics/mesh_simplifier.cpp`
- GPU resource registry: `src/render/resources.cpp`
- Frame orchestration: `src/render/renderer.cpp`
- Render tunables: `include/fire_engine/render/constants.hpp`
- GPU data-layout limits (graphics-visible): `include/fire_engine/graphics/gpu_limits.hpp`
- Physics authoring docs: [`collision.md`](collision.md)

## Build And Verification

Configure and build from the repository root:

```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
```

The `vcpkg` preset selects the `Dev` build type (`-O2 -g`, no `NDEBUG`) and writes
`build/compile_commands.json` for `clangd`.

Run tests — the Catch2 binary plus the CTest cases (the latter includes the
`layering_guards` guard, which needs no GPU). Run from `build/`
(or pass `--test-dir build`) so the relative `test_assets/` paths resolve:

```bash
(cd build && ./test_fire_engine)               # all Catch2 tests
(cd build && ./test_fire_engine "[UBO]")       # one tag/area while iterating
ctest --preset fast                            # source-root fast CTest preset
ctest --test-dir build --output-on-failure     # build-dir equivalent
cmake --build --preset full                    # all Catch2 tests + layering guard
cmake --build build --target tests-full        # build-dir equivalent
```

Graphics-layer tests run headless because the layer only stores opaque handles — keep it that way
so the suite stays GPU-free. Test files mirror their source path (`src/foo/bar.cpp` →
`tests/foo/test_bar.cpp`). Shared Catch2 helpers and compile-time test traits live in
`tests/support/`; assets in `tests/assets/` are copied to `build/test_assets/`.

### Adding code to the build

The build uses **explicit source lists**, not globbing (only `tests/assets/` is globbed). When you
add a file you must register it in `CMakeLists.txt`:

- **New library source** (`src/.../foo.cpp`) → add to the `add_library(fireengine SHARED ...)` list.
- **New test** (`tests/.../test_foo.cpp`) → add to the `add_executable(test_fire_engine ...)` list.
  Long-running settle/soak coverage should carry `[slow]`; `test_fire_engine` runs `~[slow]`,
  while `tests-full` runs the whole binary.
- **New shader** (`shaders/foo.frag`) → add to `SHADER_SOURCES`; it's compiled to `foo.frag.spv`
  via `glslc` at build time and copied next to the binary. Forgetting this means the shader simply
  won't exist at runtime.
- **ImGui/vcpkg link wiring** → keep Vulkan and GLFW linked by `fireengine` directly. The helper
  `cmake/fireengine_imgui.cmake` creates the local `fireengine_imgui` target from vcpkg's ImGui
  archive without inheriting ImGui's transitive Vulkan/GLFW link interface.

A pure header-only addition needs no CMake change, but its test still does.

### Before you open a PR

- Format touched files with `clang-format -i`; CI enforces a repository dry-run.
- Keep `./build/test_fire_engine`, default `ctest`, and `tests-full` green at the PR boundary.
- **Run the Linux CI locally in Docker** — the macOS build is more lenient than CI, so a green local
  build can still fail the Linux/clang-tidy/format gate. Reproduce it before you commit:

  ```bash
  tools/ci/run-local-ci.sh all      # format + configure + build (warnings-as-errors) + tidy + tests
  ```

  It copies the working tree into an Ubuntu 24.04 container and keeps the Linux build + vcpkg state in
  Docker volumes (host artifacts untouched). Sub-stages `format` / `configure` / `build` / `tidy` /
  `test` / `shell` isolate a step. It defaults to `linux/amd64` (matches GitHub Actions); set
  `DOCKER_PLATFORM=linux/arm64` for a faster Apple-Silicon smoke that does *not* match CI's platform.
- **If a change to the physics solver moves `Determinism.GoldenHash`**, that golden is
  *platform-specific*. Update the macOS/arm64 value from your local run **and** the Linux/x86_64 value
  from `tools/ci/run-local-ci.sh test` — CI runs Linux/x86_64 and will fail on a stale golden. A hash
  move you didn't intend is a determinism regression, not something to rebaseline (see CLAUDE.md
  § Testing).
- Sweep [`README.md`](../README.md), `onboarding.md`, [`review-order.md`](review-order.md), and the relevant subsystem doc
  ([`collision.md`](collision.md) / [`lod.md`](lod.md) / [`roadmap.md`](roadmap.md)) for references your change made stale — the constant you
  moved, the binding you renamed, the file you split. Treat the docs as part of the change, not an
  afterthought (see CLAUDE.md § Documentation).

Run the app:

```bash
cd build
./fireEngineApp
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr
./fireEngineApp -f RiggedSimple/RiggedSimple.gltf night.hdr
./fireEngineApp --debug-shadow-depth -f RiggedSimple/RiggedSimple.gltf night.hdr
```

Assets and shaders are copied or compiled as part of the build, so runtime paths are normally
relative to the build directory.
