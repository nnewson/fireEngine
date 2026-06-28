# Fire Engine

A Vulkan-based 3D renderer written in C++23, built on macOS with MoltenVK.

After I came out of Uni, my first real job was working as a software engineer for [Rare](https://www.rare.co.uk).
I started there working on their R&D team, developing what would be called the 'REngine' - a shared 3D engine that was used be used for a bunch of future Gamecube releases (but not Perfect Dark Zero they did their own).
During those days I had a design for a 3D engine I'd create it Gamecube limitations wheren't a problem (tesselated surfaces, spline based animation and softbody skinning, and 'correct' collision detection).
I've no doubt these are all solved problems nowadays with the Unreal engine et all, but since I've been reading up on the tech again, I thought it would fun to dip my toes in again with Vulkan.

## Features

- **glTF 2.0 model loading** via [fastgltf](https://github.com/spnda/fastgltf) — geometry, full PBR material set (base-colour, metallic-roughness, normal, occlusion + `occlusionStrength`, emissive, **transmission**, **clearcoat**, **thickness**), per-texture sampler settings, **per-slot UV-set selection (TEXCOORD_0 / TEXCOORD_1)**, skeletal skins, morph targets (POSITION + NORMAL + TANGENT deltas), keyframe animations, and alpha-mode state (OPAQUE / MASK / BLEND, `alphaCutoff`, `doubleSided`). Supported extensions: `KHR_materials_emissive_strength`, `KHR_texture_transform`, **`KHR_texture_basisu`** (Basis Universal / KTX2 textures), **`KHR_materials_variants`**, `KHR_materials_unlit`, **`KHR_lights_punctual`**, **`KHR_materials_transmission`**, **`KHR_materials_ior`**, **`KHR_materials_clearcoat`**, **`KHR_materials_volume`**. Authored cameras are adopted as the engine's runtime view; authored lights drive the scene. Unsupported `extensionsRequired` rejected with a clear error; non-Triangles primitives skipped with a warning
- **Tangent-space normal mapping** — tangents generated on load when a material uses a base normal or clearcoat normal texture (per-triangle UV derivatives, Gram-Schmidt orthogonalisation, handedness preserved in `tangent.w`). **Smooth-normal fallback** synthesises per-vertex normals when the source mesh omits NORMAL (e.g. Fox.gltf)
- **Physically based shading with split-sum IBL + multi-scatter compensation** — equirectangular HDR skybox is converted to an environment cubemap (1024², 11 mip levels), a diffuse irradiance cubemap (32²), a GGX prefiltered specular cubemap (128², 8 mips, importance-sampled with 256 Hammersley samples and Filament-style mip-LOD weighting against the source cubemap's mip chain), and a BRDF integration LUT (256²) at startup. Forward fragment shader uses Fdez-Aguera multi-scatter compensation so rough conductors stay energy-conserving across the roughness range
- **Cascaded directional shadows with skinned self-shadowing** — 4 cascades, 2048×2048 per cascade in a 2D-array depth image, Practical-Split-Scheme cascade boundaries (λ = 0.5 blend of linear and log-uniform splits) over 0.1m–50m. Directional shadows use shared comparison sampling with per-cascade bias scaling and 10% blend bands at cascade boundaries; the shader still has a rotated Poisson PCF path, currently configured with zero filter radius for crisp contact. Skinned meshes avoid same-map self-shadow acne by receiving world shadows from a world-only directional map plus a tightly-fit dual-depth per-object self-shadow map. The self-shadow second pass culls front faces (`vk::CullModeFlagBits::eFront`) so only back-facing geometry rasterises, eliminating per-fragment discard coin-flips on marginal surfaces that previously produced random per-pixel flicker
- **KHR_materials_unlit** — flagged materials skip BRDF/IBL/shadow entirely and output the textured base colour directly. Used for skybox cards, foliage, decals, UI quads
- **KHR_texture_transform** — per-slot UV offset/scale/rotation from the extension is applied to each texture sample. Identity by default
- **Multi-light scenegraph** — `Light` is a first-class component variant alongside Camera / Mesh / Animator / Empty. Type enum is **Directional / Point / Spot**, with colour, intensity, range, and inner/outer cone angles per spec. Each frame the scenegraph walks all lights into a packed `Lighting` array (cap `kMaxLights = 8`), the renderer picks the first directional as the CSM source, and the forward shader runs a per-fragment loop over the array. Point/spot use the KHR_lights_punctual attenuation (`windowing² / d²` with `windowing = clamp(1 − (d / range)⁴, 0, 1)`); spot adds a smooth cone factor on top
- **KHR_lights_punctual import** — glTF lights become `Light` nodes, transform-driven. FireEngine seeds a default directional Sun only when the asset hasn't authored one
- **KHR_materials_transmission** — diffuse is attenuated by `(1 − transmission)` and a transmission lobe is added on top, gated on `KHR_materials_volume` thickness. **Thin-walled** materials (no volume, e.g. a paper lamp shade) scatter to a uniform basecolor × env-irradiance tint — view-independent, so a bright source behind the surface can't smear into a camera-tracking highlight. **Volumetric** materials (thickness > 0, e.g. frosted glass / the TransmissionRoughnessTest panels) sample the captured post-opaque sceneColor mip chain along the refracted ray for screen-space scene-behind-glass refraction, blurred by roughness, then attenuated by Beer-Lambert absorption over the volume
- **Authored-camera adoption** — `GltfLoader::findFirstCamera` walks the default scene's node tree DFS and returns the first camera-bearing node's view; FireEngine reframes its runtime camera to match before the first frame
- **GPU particle system (compute-driven)** — `ParticleEmitter` is a scene component, gathered each frame like `Light` into a Vulkan-free `EmitterState`. A renderer-owned `ParticleSystem` simulates a pooled particle SSBO with a **compute shader** (spawn dead slots at the emitter up to a per-frame budget via an atomic spawn-claim; integrate the rest under gravity), then renders the pool as **instanced camera-facing billboards** (`cmd.draw(6, poolCount)`, per-instance data read from the SSBO by `gl_InstanceIndex`) blended **additively into the HDR target** so bloom catches the glow. **Soft particles**: the fragment shader fades against sampled scene depth so particles dissolve smoothly into geometry with no hard clip edge. Built on the compute-pipeline + synchronization2 buffer-barrier path
- **HDR offscreen forward pass + bloom + ACES post-process** — forward writes into an R16G16B16A16 target. **Dual-filter bloom** (6-mip RGBA16F chain at half-screen res, 13-tap CoD downsample with Karis-average on the first pass to suppress fireflies, 9-tap tent upsample with additive blend) produces a low-pass HDR contribution. Post-process mixes the HDR target with bloom mip 0 (`bloomStrength = 0.04` default; `0` is bit-identical to a no-bloom path), then ACES tonemap + gamma 2.2 before presenting
- **SSAO + contact shadows** — a **depth prepass** (reusing the forward vertex shader with `invariant gl_Position`; the forward pass loads it with `LESS_OR_EQUAL`) fills the shared depth buffer before lighting. A renderer-owned `Ssao` subsystem then reconstructs view-space position + normal from depth alone (no normal G-buffer — analytic unprojection from the projection matrix) and writes an **R8G8** target: R = hemisphere-kernel ambient occlusion, G = a sun-direction screen-space **contact-shadow** ray-march. The forward shader samples it to multiply SSAO into the IBL/ambient terms and the contact term into the **direct sun** (ambient stays on pure CSM). A **depth-aware bilateral blur** (5×5, view-space-Z edge-stop) smooths the per-pixel sampling/march noise without bleeding across silhouettes, with TAA carrying the temporal denoise. SSAO and contact shadows are on by default; contact shadows fill the CSM's short-range contact gap and use an N·L gate plus view-Z-scaled depth window and silhouette edge guard to avoid screen-space streaks. Live overlay sliders (radius / bias / intensity / power, contact length) and a `--debug-ssao` view
- **GPU soft-body / cloth (XPBD)** — `-c` drops a cloth that simulates entirely on the GPU, or author one on any glTF mesh with `extras.Cloth` (samples: `assets/ClothSheet/ClothSheet.gltf`, `assets/ClothBanner/ClothBanner.gltf`). A renderer-owned `SoftBodySystem` runs an XPBD compute solver each substep: `cloth_predict` integrates gravity + wind, `cloth_solve` projects distance constraints **graph-coloured** into race-free batches (Gauss-Seidel by colour), `cloth_collide` pushes particles out of world colliders, and `cloth_finalize` writes solved positions + normals (recomputed from a per-vertex→triangle **CSR adjacency**, so arbitrary meshes work, not just grids) into a storage **vertex buffer** the forward/shadow passes read — so the cloth renders, lights, and casts shadows through the normal forward path (double-sided), no new render shaders. The solver is **descriptor-free**: every buffer reaches the shaders as a `bufferDeviceAddress` pointer. Collision primitives (plane / sphere / box / capsule) are gathered each frame from `PhysicsWorld` (`gatherColliders`) plus a ground plane. Constraint stiffness is authored per type (structural/shear stiff, bend soft); substeps, a global **compliance multiplier**, damping, gravity, and wind are **live overlay sliders**. Built on the same compute + buffer-barrier path as particles
- **Temporal anti-aliasing (TAA)** — sub-pixel Halton(2,3) projection jitter plus velocity-buffer history accumulation anti-aliases geometry edges *and* specular/shading shimmer (unlike MSAA, which only covers geometry edges). The forward + transmission passes write a screen-space motion-vector attachment; the resolve reprojects the previous frame's history along it (`historyUV = uv − velocity`), neighbourhood-clamps to the current 3×3 to suppress ghosting/disocclusion, and blends. Motion vectors are jitter-free so the jitter cancels in accumulation. Per-node previous-world-matrix tracking feeds rigid + animated motion (skinned deformation is camera-motion-only in v1); particles render after the resolve, kept out of history. `--no-taa` reverts to the raw image, `--debug-velocity` visualises the buffer
- **Frustum culling (camera + shadow casters)** — built on a reusable fat-AABB BVH (`AabbBvh<T>`, the same core the physics broadphase and static-mesh triangle index use). Two stages: a **persistent scene BVH** (`SceneCuller`, an `AabbBvh<Node*>` over rigid renderables) pre-culls each frame against the union of the camera frustum and every shadow caster's frustum, so off-screen nodes skip draw-building entirely (no UBO writes, no per-vertex bounds) — `O(log N + visible)` instead of `O(N)`; then a **precise per-pass cull** drops the survivors that fall outside a given pass's frustum (`buildDrawBuckets` for the camera, per-cascade/spot/point-face in the shadow pass). Frustums are 6 Gribb–Hartmann planes (Vulkan `[0,1]` depth) with a conservative positive-vertex AABB test (no false negatives). Deformable (skinned/morph) meshes skip the coarse BVH (bind-pose bounds under-cover the animated pose) and rely on the precise stage's exact per-frame world bounds. The coarse union is a strict superset of what any pass keeps, so culling never drops a visible draw. Live overlay toggle + tracked/visible/culled counts; off submits everything (A/B + regression escape hatch)
- **Debug + profiling overlay (Dear ImGui)** — a runtime overlay (Dear ImGui 1.92 on the Vulkan dynamic-rendering backend, drawn into the swapchain after post-process) toggled with **F1** (`--overlay` to start visible). Shows a **CPU frame-time/FPS plot** and **per-pass GPU timings** via a timestamp `VkQueryPool` (`GpuProfiler`, with graceful fallback when the device/queue doesn't support timestamps), plus a **live tunables panel** (`RenderTunables`) for TAA (history blend, sharpen, on/off), frustum culling (on/off + tracked/visible/culled counts), the debug-view dropdown + no-shadows, bloom/diffuse-IBL/specular-IBL/sun-intensity, and particle emitter rate/lifetime/size — all editable without a recompile. Camera input is suppressed while a widget is being driven
- **Keyframe animation** with per-channel interpolation (LINEAR with SLERP for quaternions, STEP, CUBICSPLINE with in/out tangents) across rotation, translation, scale, and morph weight channels; looping playback; runtime animation selection via `AnimationState`
- **Skeletal skinning** — GPU joint matrix blending, up to 64 joints per skin
- **Morph target animation** — vertex POSITION + NORMAL + TANGENT deltas uploaded as a single packed SSBO, blended by weights in the vertex shader (up to 8 targets per mesh). Tangent morphs feed the TBN reconstruction so facial-rig normal mapping stays correct mid-blend
- **Alpha blending and masking** — three forward pipeline variants (opaque, double-sided opaque, blend) dispatched per draw from the material's `AlphaMode`/`doubleSided` flags. MASK handled via a fragment-shader discard test; BLEND uses straight-alpha blending with depth-write disabled and back-to-front sort of translucent draws
- **Scenegraph architecture** — tree of Nodes with Component variants (Camera, Animator, Mesh, Empty, **Light**, **ParticleEmitter**) that propagate transforms, an `InputState` bundle, and draw commands. Node transforms store rotation as a quaternion so orientations from glTF round-trip exactly
- **Custom collision and physics path** — glTF `extras.Physics` can create `Static`, `Kinematic`, and `Dynamic` bodies with layer/mask filtering, authored AABB/box/sphere/capsule/convex-hull collider shapes, linear velocity, mass, restitution, friction, and gravity scale. `PhysicsWorld` owns body/collider state, a `BroadPhase` (default `DynamicAabbTreeBroadPhase` — a fat-AABB BVH; `SweepAndPruneBroadPhase` injectable via the `PhysicsWorld(unique_ptr<BroadPhase>)` constructor) gathers AABB candidate pairs, `NarrowPhase` builds a shape-specific `ContactManifold` per pair (sphere/box/capsule analytic + box/box SAT, plus a **GJK/EPA convex path** for `ConvexHullShape` colliders and a speculative-margin path that emits gap contacts for separated-but-approaching pairs so fast movers don't tunnel), a **sequential-impulse `ContactSolver`** resolves them with **full rigid-body rotation** (warm-started normal + friction impulses with lever-arm torque, gated restitution, split-impulse positional/orientation correction, per-shape inertia tensors and quaternion orientation integration — resting stacks settle and boxes topple and rest on a face), and `SceneGraph::submitPhysics` / `SceneGraph::applyPhysics` bridge scene-authored and physics-authored transforms each frame. **Debug tooling**: `--debug-physics` (and the overlay "Physics debug" panel) draws immediate-mode wireframes for broadphase AABBs, collider shapes, and contact normals via a renderer-owned `DebugDraw` line pass (x-ray or depth-tested), and a **determinism harness** (`test_physics_determinism`) replays a fixed-step scene and hashes body state so accidental non-determinism or behaviour drift is caught in CI
- **Physics queries, triggers, and a character controller** — first-class spatial queries on `PhysicsWorld`: `raycast`/`raycastAll` (analytic ray vs sphere/OBB/capsule/convex + Möller–Trumbore triangle), `shapecast` (GJK conservative advancement), and `overlapSphere`/`overlapShape`, all with layer/mask `QueryFilter`ing (brute-force over active colliders with an AABB reject; mesh colliders dispatch into their triangle BVH). A collider flagged `isTrigger` (glTF `extras.Physics` `"IsTrigger": true`) generates overlap **events** instead of a solver response — `triggerEvents()`/`collisionEvents()` return per-step enter/stay/exit `ContactEvent`s diffed from the overlap set. On top of the queries, a kinematic-capsule **`CharacterController`** does collide-and-slide movement with slope limits, step up/down, and grounded snapping (`-k` runs a patrol demo)
- **Backend-decoupled graphics layer** — graphics classes use opaque handles (`BufferHandle`, `TextureHandle`, `DescriptorSetHandle`, `PipelineHandle`) and emit `DrawCommand` structs with no Vulkan dependencies. IBL cubemaps, BRDF LUT, shadow map, bloom chain are all owned by the render layer and referenced through the same handle types
- **Vulkan rendering** via vulkan.hpp C++ bindings, targeting Vulkan 1.4 with **dynamic rendering** (no `VkRenderPass`/`VkFramebuffer` objects) and **synchronization2** barriers/submits throughout. CPU↔GPU frame pacing uses a single monotonic **timeline semaphore** (the swapchain acquire/present semaphores stay binary, as WSI requires). Built around a **frequency-split forward descriptor layout**: set 0 holds 4 per-object vertex-stage bindings (frame UBO, skin/morph buffers), set 1 holds 13 globals shared by every draw (light UBO, five shadow maps, debug image, compare/debug samplers, three IBL textures, sceneColor), and **set 2 is bindless** — one global `sampler2D[]` texture array (indexed by texture handle) plus a global materials SSBO (indexed by a per-draw push constant). Set 0 is **pushed inline per draw** (`VK_KHR_push_descriptor`), not allocated; sets 1 and 2 are allocated once and bound at the start of each forward pass. The shadow pass uses the same push-descriptor model for its set 0. Separate descriptor layouts exist for skybox, shadow, post-process, and bloom passes
- **Single source of truth for tunables** — every scalar rendering knob (light intensity, IBL strengths, shadow biases, cascade split λ, bloom strength, IBL extents, camera FOV) lives in `include/fire_engine/render/constants.hpp`. GPU data-layout limits that the Vulkan-free graphics layer also needs (frames-in-flight, joint/morph/light counts, shadow caster caps + matrix layout, cascade count) live one layer down in `include/fire_engine/graphics/gpu_limits.hpp`, which `constants.hpp` includes — so render-side code still sees every constant through one include, while graphics headers stay free of `render/`
- **Texture mapping** via [stb_image](https://github.com/nothings/stb), including HDR equirectangular loading for the skybox; uploaded to GPU through staging buffers
- **First-person camera** with keyboard (WASD + E/F for vertical) and mouse controls
- **GLSL shaders** compiled to SPIR-V at build time via `glslc`

## How It Works

### Scenegraph

The engine uses a scenegraph where each `Node` holds a `Transform` (position, unit-quaternion rotation, scale), a `Component`, optional `Controllable` input behaviour, optional physics handles, child ownership, and a cached composed-world matrix. Components are stored as a `std::variant<Empty, Animator, Camera, Mesh, Light>`. Each component implements the update/render behaviour relevant to that component:

- **Camera** integrates per-frame deltas from `InputState::cameraState()` (deltaPosition, deltaYaw, deltaPitch, deltaZoom) into absolute position and orientation. Pitch is clamped to ±`kCameraMaxPitch` (1.5 rad ≈ 85.9°) from `math/constants.hpp`. FireEngine owns the active Camera directly and passes its position/target to the renderer each frame.
- **Animator** owns an `Animation` (per-channel Linear/Step/CubicSpline interpolation across rotation, translation, scale, and morph weight keyframes). Each frame it samples the animation at the current elapsed time, producing a TRS matrix that is applied to all child nodes. It also consults `InputState::animationState()` to switch between animations at runtime.
- **Mesh** wraps an `Object` that manages GPU resources via opaque handles. During render traversal it calls `Object::render()` which writes UBO data to mapped memory and returns `DrawCommand` structs.
- **Light** carries a `Type` (Directional / Point / Spot), `Colour3`, intensity, range, and inner/outer cone angles. Position and forward direction come from the node's `composedWorld` matrix at gather time — KHR_lights_punctual convention has the light forward as the node's local −Z. `SceneGraph::gatherLights()` walks the tree once per frame and returns a `std::vector<Lighting>` for the renderer to pack into the LightUBO array. `SceneGraph::hasDirectionalLight()` lets FireEngine skip seeding its default Sun when an asset has authored its own.
- **Empty** is a no-op component for structural nodes (joint bones, group nodes).

`Node` does not own physics objects directly. It stores `PhysicsBodyHandle` and `PhysicsColliderHandle` values when the glTF node has `extras.Physics`. `PhysicsWorld` owns the actual bodies/colliders and `SceneGraph` synchronizes transforms across that boundary with `submitPhysics()` and `applyPhysics()`.

### Collision And Physics

Physics is split across three layers:

- **`collision/`** contains low-level collision primitives. `AABB` (in `collision/aabb.hpp`) is the shared value type used by every collision/physics consumer, with `axisMin(Axis)` / `axisMax(Axis)` / `center()` / `extent()` accessors; it's used directly by `Collider`, the narrow-/broad-phase, glTF loading, and graphics. `Collider` stores local/world/swept AABBs, collision layer/mask filtering, a stable `ColliderId`, and six owned SAP `EndPoint`s. `SweepAndPruneBroadPhase` owns endpoint lists and emits broadphase `CollisionPair`s. `NarrowPhase::collide` produces a shape-specific `ContactManifold` (neutral `WorldShape` primitives + closest-point math in `collision/geometry.{hpp,cpp}`; convex hulls via GJK/EPA in `collision/gjk_epa.{hpp,cpp}`), and with a speculative margin also emits negative-penetration gap contacts for continuous collision.
- **`physics/`** owns simulation state. `PhysicsWorld` stores bodies, colliders, shapes, materials, the broadphase, the narrowphase, and the `ContactSolver` (a sequential-impulse solver: warm-started normal/friction impulses with lever-arm torque, gated restitution, split-impulse positional/orientation correction, and **full rotational dynamics** — per-shape inertia tensors and quaternion orientation integration, so boxes topple and rest on a face). `PhysicsBody` stores type, velocity, mass/inverse mass, angular velocity, local inverse inertia, gravity scale, restitution, and friction. `ColliderShape` supports AABB, box, sphere, capsule, and convex-hull authoring (the hull built from a mesh by `core/convex_hull_builder`); the broadphase uses a local AABB while `PhysicsWorld::worldShape` composes the authored shape with the body transform into a `WorldShape` for the shape-specific narrowphase.
- **`scene/`** stores only opaque physics handles. Scene nodes stay responsible for transforms, hierarchy, input, animation, and rendering; physics ownership stays in `PhysicsWorld`.

The frame loop makes the authority split explicit:

```cpp
scene_.update(input_state);
scene_.submitPhysics(physics_);

while (accumulator >= fixedDt)
{
    physics_.step(fixedDt);
    accumulator -= fixedDt;
}

scene_.applyPhysics(physics_);
```

`submitPhysics()` pushes non-dynamic scene transforms into `PhysicsWorld`: static bodies are scene-authored, and kinematic bodies are gameplay/input-authored. `PhysicsWorld::step()` integrates dynamic-body velocity, refreshes collider AABBs, updates broadphase candidates, builds shape-specific contact manifolds, runs the sequential-impulse velocity solve, integrates positions, applies the split-impulse penetration correction, and captures previous positions. `applyPhysics()` pulls non-static physics transforms back onto scene nodes, so dynamic simulation and kinematic collision correction are visible before rendering.

Physics can be authored in glTF through node `extras.Physics`. The loader creates bodies/colliders, assigns handles to the node, and rejects unsupported combinations such as a `Dynamic` body on a `Controllable` node.

### Physics Demos

`assets/physics_demos/` holds one minimal, self-contained glTF scene per physics capability — simple untextured geometry whose dimensions match its collider, authored purely to *show* a feature behaving. The scenes are emitted by `assets/physics_demos/generate.py` (regenerated automatically at build time) and each is mirrored by a headless replay test in `tests/physics/test_demos.cpp` that rebuilds an equivalent `PhysicsWorld`, steps the fixed-step solver, and asserts the labelled outcome — so each demo is both a visual showcase and an automated regression guard.

| Demo (`physics_demos/…`) | What it verifies | Headless test (`tests/physics/test_demos.cpp`) |
|---|---|---|
| `FallRestDemo.gltf` | A Dynamic box falls onto a Static floor, settles flat, and goes fully still (it sleeps). End-to-end author→simulate smoke. | `Demos.FallRest.BoxComesToRestOnFloor` |
| `RestitutionDemo.gltf` | Three spheres with restitution 0.0 / 0.5 / 0.9 dropped from the same height bounce to visibly different rebound heights. | `Demos.Restitution.HigherRestitutionBouncesHigher` |
| `FrictionRampDemo.gltf` | Two boxes on a 25° ramp: a high-friction box holds while a low-friction box slides off and grinds to a halt on the rough floor (combined friction is `sqrt(a·b)`). | `Demos.Friction.HighFrictionStaysLowFrictionSlides` |

Run a demo (add `--debug-physics` to overlay collider/contact wireframes); paths are relative to `build/`:

```bash
cd build
./fireEngineApp physics_demos/FallRestDemo.gltf      skybox.hdr --debug-physics
./fireEngineApp physics_demos/RestitutionDemo.gltf   skybox.hdr --debug-physics
./fireEngineApp physics_demos/FrictionRampDemo.gltf  skybox.hdr --debug-physics
```

Run all the demo behaviour tests headlessly:

```bash
./test_fire_engine "[Demos]"
```

### Graphics/Render Boundary

The `graphics/` layer is fully decoupled from Vulkan:

- **Opaque handles** — `BufferHandle`, `TextureHandle`, `DescriptorSetHandle`, and `PipelineHandle` are scoped enums backed by `uint32_t`. Graphics classes store these instead of Vulkan objects. The IBL cubemaps, BRDF LUT, and shadow map live behind the same handle types.
- **Resources** (in `render/`) — owns all Vulkan GPU resources (buffers, textures, descriptor pools/sets, registered pipelines, IBL cubemaps, BRDF LUT, shadow map). Graphics classes call `Resources` methods during `load()` to create buffers, textures, and descriptor sets, and receive handles back.
- **DrawCommand** — a backend-agnostic struct containing handle references (including the `PipelineHandle` to bind), an index count, and a `sortDepth` used for back-to-front ordering of translucent draws. `Object::render()` returns a vector of these; the Renderer resolves handles to Vulkan objects and records the actual draw calls.
- **FrameInfo** — plain data struct carrying frame index, viewport dimensions, camera vectors, and the `AlphaPipelines` bundle so graphics code can pick the right pipeline per material without touching Vulkan types.

### Loading

`GltfLoader` parses a glTF 2.0 file using fastgltf and builds the scenegraph. For each glTF node:

- If the node has an animation channel, it creates a three-level hierarchy: root Node (with the node's TRS transform) -> Animator Node (with keyframe data, preserving each channel's interpolation mode) -> Mesh Node.
- Otherwise, the node maps directly with its transform and mesh data.
- Skin data (joint references and inverse bind matrices) is loaded and attached to the relevant Mesh nodes.
- Morph target deltas (position, normal, **and tangent**) are stored per-geometry and uploaded as a single packed SSBO.
- Materials gather texture slots for base-colour, emissive, normal, metallic-roughness, occlusion, **transmission**, clearcoat, clearcoat roughness, clearcoat normal, and thickness. Each slot carries its own `SamplerSettings`, `TextureEncoding` (Srgb or Linear), UV-set index (TEXCOORD_0 / TEXCOORD_1), and `UvTransform` from KHR_texture_transform (offset / scale / rotation; identity by default).
- Texture image sources go through a shared loader path: local file URIs use the direct file loaders, while embedded/data/buffer-view sources are resolved to bytes and then loaded through the `Image` or `KtxImage` memory APIs.
- **Smooth-normal fallback** runs when the source mesh omits the `NORMAL` attribute (Fox.gltf and similar). A static `GltfLoader::generateSmoothNormals` builds per-vertex normals from positions + indices via area-weighted accumulate-and-normalize, with an up-pointing fallback for unreferenced vertices.
- **Tangent generation** runs automatically when a material has a base normal or clearcoat normal texture and the glTF did not already supply TANGENT data. A custom per-triangle routine computes T and B from UV derivatives, Gram-Schmidts T against the vertex normal, and writes handedness into `tangent.w`. Degenerate UVs fall back to a normal-derived tangent so the mesh still shades reasonably.
- **Material extensions** — `KHR_materials_emissive_strength` is multiplied into emissive at load time so HDR emissives reach the bloom chain at the authored magnitude. `KHR_materials_unlit` flips a flag on the Material that the fragment shader uses to skip BRDF/IBL/shadow. `KHR_texture_transform` is read per slot and applied in shader before each sample. **`KHR_materials_transmission`**, `KHR_materials_ior`, `KHR_materials_clearcoat`, and `KHR_materials_volume` populate the extra transmission, clearcoat, and thickness slots consumed by the forward shader.
- **Light extensions** — `KHR_lights_punctual.lights` are loaded into the asset's lights array; nodes carrying a `lightIndex` get a `Light` component (skipped with a warning if the node already holds a Mesh / Animator). Type / colour / intensity / range / cone angles all map directly. `FireEngine::loadScene` checks `SceneGraph::hasDirectionalLight()` after load and seeds a default Sun only when no directional was authored.
- **Camera extension** — `GltfLoader::cameraViewFromMatrix` resolves a node's accumulated world transform into a `(position, target)` viewpoint (glTF cameras look down −Z in local space). FOV / near / far stay engine-side; first-cut adoption is position + look direction only. View-basis construction uses a shared fallback path so zero-length targets and straight-up/down views still produce finite right/up vectors.
- **Physics extras** — `extras.Physics` can create `Static`, `Kinematic`, or `Dynamic` bodies. Supported custom fields include `Layer`, `Mask`, `Velocity`, `Mass`, `Restitution`, `Friction`, `GravityScale`, `Shape`, `Center`, `HalfExtents`, `Radius`, and `HalfHeight`. If no shape is supplied, the loader uses the mesh POSITION bounds as an AABB proxy.
- **Safety checks** — `GltfLoader::ensureSupportedExtensions` walks `asset.extensionsRequired` and throws if any aren't in our supported set (so e.g. draco-compressed assets fail fast instead of producing corrupt geometry). Non-triangle primitives are skipped with a `std::clog` warning rather than rendered as garbage.

Animation keyframes (input times and output quaternions/vectors/weights, plus CUBICSPLINE tangents) are read from glTF accessor data and set on the Animator's `Animation`.

### Startup: IBL Precompute

Before the first frame, the renderer runs a one-shot precompute chain using transient pipelines and one-time-submit command buffers:

1. **Equirectangular → cubemap** — load the HDR skybox and render 6 cubemap faces into a 1024² RGBA32F cubemap by sampling the equirectangular map along per-face direction vectors. After the 6-face pass, a `vkCmdBlitImage` chain generates the full 11-mip pyramid (1024 → 1) so the prefilter pass can do mip-weighted importance sampling.
2. **Irradiance convolution** — produce a 32² RGBA32F cubemap via cosine-weighted hemisphere integration (diffuse IBL).
3. **GGX specular prefilter** — produce a 128² RGBA32F cubemap with 8 mip levels. Roughness is pushed as a push constant per mip; each fragment importance-samples GGX with Hammersley + 256 samples, picking a blurrier source-cubemap mip when the PDF is low (Filament's mip-weighted importance sampling) so rough lobes stay shimmer-free.
4. **BRDF integration LUT** — 256² RGBA32F 2D; x = NdotV, y = roughness; outputs (scale, bias) for the Fresnel split-sum. No input textures required.

The transient pipelines are destroyed once the bake completes; only the resulting cubemaps + LUT remain and are bound into the forward shader's descriptor set every frame.

### Frame Loop

1. `FireEngine::mainLoop()` polls GLFW, calls `input_.update(window, dt)` to produce an `InputState`, then `scene_.update(inputState)`.
2. `SceneGraph::update()` propagates `InputState` and transforms down the node tree; each Node caches its `composedWorld` matrix for skin joint lookups.
3. `scene_.submitPhysics(physics_)` pushes static/kinematic scene transforms into `PhysicsWorld`.
4. `PhysicsWorld::step(1.0f / 60.0f)` runs zero or more fixed substeps from the frame accumulator.
5. `scene_.applyPhysics(physics_)` pulls dynamic and corrected kinematic transforms back into scene nodes and resolves composed-world matrices.
6. `Renderer::drawFrame()` acquires a swapchain image and records the frame passes:
   - **Shadow passes** — directional cascades render both the full CSM and a world-only CSM that excludes skinned casters. Each skinned self-shadow slot renders two tightly-fit passes: the first captures the nearest light-facing surface, and the second samples that first depth and discards it so the forward shader can sample the next useful self-occluder. Spot and point shadow passes replay the same compatible shadow draw commands through their per-layer/per-face depth attachment views. Skin and morph still apply in the shadow vertex shader
   - **Forward pass** — begin the HDR offscreen pass, draw the skybox (LEQUAL depth, no write), then call `scene.render(ctx)`; Mesh/Object emit `DrawCommand`s that the Renderer buckets into opaque, transmissive, and blend lists, sorts the blend bucket back-to-front by `sortDepth`, and replays through the same bind/draw loop resolving handles via `Resources`
   - **Transmission pass** — when transmissive draws are present, capture the opaque scene colour mip chain and replay transmissive draws so the shader can sample scene-behind-glass data
   - **TAA resolve** — the forward/transmission passes also write a screen-space velocity (motion-vector) attachment; the resolve reprojects the previous frame's accumulated history along that buffer, neighbourhood-clamps it against the current 3×3 to kill ghosting, blends, and blits the result back into the HDR target. Sub-pixel projection jitter (Halton(2,3)) drives the accumulation; particles render afterwards with the un-jittered projection so they stay out of history. Skipped under `--no-taa`
   - **Bloom downsample chain** — 6 fullscreen-triangle passes. Pass 0 reads the HDR target with the Karis-average 13-tap kernel (firefly suppression), writing mip 0 of the bloom chain. Passes 1..5 read the previous bloom mip and write the next, plain CoD weights
   - **Bloom upsample chain** — 5 fullscreen-triangle passes back up the chain (mip 5 → mip 4 → … → mip 0). Each samples its source mip with a 9-tap tent kernel and **additively blends** onto the destination mip (preserved by `loadOp=eLoad`). The final write to mip 0 carries the summed contribution from every coarser mip
   - **Post-process pass** — begin the swapchain-format pass, draw a fullscreen triangle that samples both the HDR target and bloom mip 0, mixes them by `bloomStrength`, and applies ACES + gamma 2.2. The swap image is left in colour-attachment layout (the present transition is deferred) so the overlay can draw over it
   - **Debug overlay** — when the ImGui overlay has content, draw it into the swap image (dynamic rendering, loadOp Load); then `transitionSwapchainToPresent` performs the final colour-attachment → present transition
   - Per-pass GPU timestamps wrap each of the above via `GpuProfiler`; results are read back a frame-cycle later for the overlay
7. Renderer submits the command buffer and presents

### Rendering Pipeline

- Forward descriptor layout is split by update frequency into **set 0 (per-object, 4 bindings)**, **set 1 (forward globals, 13 bindings)**, and **set 2 (bindless materials, global)**. Set 0 is tiny per-object state, **pushed inline at draw time via `VK_KHR_push_descriptor`** (`vkCmdPushDescriptorSet`) — no per-object descriptor set is allocated; set 1 and set 2 are bound once per frame and survive pipeline transitions inside the forward bucket.
- **Set 0 — per-object vertex-stage state**:
  - 0 frame UBO (model / view / projection + camera position)
  - 3 skin UBO (joint matrices, `mat4[64]`)
  - 4 morph UBO (metadata + weights)
  - 5 morph targets SSBO — `[positions, normals, tangents]` per target as `vec4[]`
  - (bindings 1, 2 are intentional gaps — the old Material UBO + base-colour sampler, now bindless)
- **Set 2 — bindless materials** (global, bound once per forward pass):
  - 0 `sampler2D textures[]` — one global combined-image-sampler array (capacity `kMaxBindlessTextures` = 512), indexed by `TextureHandle`; partially-bound + update-after-bind, written as 2D material textures load
  - 1 `materials[]` SSBO — an array of the material record (`diffuseAlpha`, `emissiveRoughness`, `materialParams`, `textureFlags`, `extraFlags`/**unlit flag**, `texCoordIndices`, `transmissionParams`, `clearcoatParams`/`clearcoatFlags`/`clearcoatTexCoords`, `volumeParams`, `attenuation`, `UvXform uv[10]`, and a per-slot bindless `textureIndex[]`), indexed by the per-draw `ForwardPushConstants::materialIndex`. The shader reads `materials[pc.materialIndex]` and samples `textures[material.textureIndex[slot]]`.
- **Set 1 — forward globals** (renumbered locally within the set):
  - 0 Light UBO — `cascadeViewProj[4]`, `cascadeSplits`, IBL params, shadow bias/filter params, environment params, `lightCount`, and `LightData lights[MAX_LIGHTS]` (per-light position/direction/colour/cone in std140-aligned `vec4`s)
  - 1 cascaded shadow map sampled image (`texture2DArray`, 4 layers)
  - 2 world-only directional shadow map sampled image (`texture2DArray`, excludes skinned casters)
  - 3 skinned second-depth self-shadow map sampled image (`texture2DArray`, up to 4 per-object layers)
  - 4 spot shadow sampled image (`texture2DArray`)
  - 5 point shadow sampled image (`textureCubeArray`)
  - 6 shadow-depth debug image (`texture2DArray`)
  - 7 shared shadow comparison sampler used with CSM, spot, and point sampled-image bindings
  - 8 shadow-depth debug sampler
  - 9 irradiance cubemap
  - 10 prefiltered environment cubemap
  - 11 BRDF integration LUT (2D)
  - **12 captured scene-colour mip chain for screen-space transmission/refraction**

  On swapchain resize, only the `kMaxFramesInFlight` set-1 descriptors need rewriting (sceneColor, post-process targets, and any future recreated globals) via `Descriptors::updateGlobalDescriptors`; the global set-2 bindless descriptors are untouched, and forward set 0 is never allocated (pushed per draw).
- Separate descriptor layouts for the skybox (SkyboxUBO + samplerCube + LightUBO), shadow (ShadowUBO with `lightViewProj[]` + SkinUBO + MorphUBO + MorphTargets SSBO + first self-shadow depth/sampler, plus `ShadowPushConstants` on the vertex/fragment stages), post-process (HDR sampler at 0 + bloom mip 0 sampler at 1, plus `PostProcessPushConstants { float bloomStrength }`), and bloom-down / bloom-up (single input mip sampler + `BloomPushConstants` on the fragment stage)
- Two forward pipeline variants share the shader + binding layout:
  - **opaque** (no blend, depth write) — OPAQUE and MASK materials. Cull mode is a **dynamic state** (`VK_DYNAMIC_STATE_CULL_MODE`, core Vulkan 1.3) set per draw, so single-sided (cull back) and double-sided (cull none) geometry share this one pipeline; `DrawCommand::doubleSided` carries the choice.
  - **blend** (cull none, `SRC_ALPHA / ONE_MINUS_SRC_ALPHA` blend, no depth write) — BLEND materials. Kept as a separate static-blend pipeline because dynamic blend state isn't available on MoltenVK (see [Limitations](#limitations)).
- All graphics and compute pipelines are created through one shared `VkPipelineCache` (owned by `Device`), so the driver can dedupe compilation work and warm pipeline recreation on resize. It is **persisted to disk** (`pipeline_cache.bin`, validated against the device's vendor/device IDs + cache UUID) so the driver's compilation — on MoltenVK the deferred Metal compile — is paid once across runs, not every cold start.
- The forward, blend, skybox, and transmission pipelines write **two colour attachments** — HDR colour + an RG16F screen-space velocity buffer (TAA) — with per-attachment blend state (velocity never blends; needs the `independentBlend` device feature)
- Additional persistent pipelines: **skybox** (fullscreen triangle, LEQUAL depth, no write), **shadow** (front-face cull, depth bias enabled, debug colour depth write), **self-shadow-first** and **self-shadow-second** (no cull; second pass rejects the first-depth surface), **transmission**, **TAA resolve** (fullscreen triangle, samples current colour + velocity + previous history, no depth), **post-process** (bloom mix + ACES + gamma, no depth), **bloom-down** (no blend, no depth, fullscreen triangle), **bloom-up** (additive eOne/eOne blend, no depth, fullscreen triangle)
- Transient IBL pipelines (`environment_convert`, `irradiance_convolution`, `prefilter_environment`, `brdf_integration`) exist only during the startup precompute
- Fullscreen and fragment-only pipeline configs share small factory helpers in `pipeline.cpp`; keep their returned `PipelineConfig` values stable because `tests/render/test_pipeline_config.cpp` locks the binding and state surface.
- MASK is implemented via a fragment-shader `discard` when `alpha < alphaCutoff`; OPAQUE/BLEND write `alphaCutoff = 0.0` so the discard is inert
- Resources class owns all GPU resources and exposes opaque handles (pipeline registry, IBL cubemaps, BRDF LUT, shadow maps with a shared comparison sampler, **bloom chain**). Internally it centralizes host-visible buffer creation and common 2D render-target setup so usage flags, views, samplers, per-mip views, and initial layouts stay consistent.
- Each Object owns its per-frame UBO/SSBO buffers; both the forward and the shadow set 0 are pushed inline per draw via `VK_KHR_push_descriptor` (no per-object descriptor set allocated in either pass)
- Depth buffering and swapchain recreation on window resize (HDR offscreen target, bloom chain, post-process descriptors, and the TAA velocity + history targets all rebuilt at new extent; the TAA history-valid guard resets so the first post-resize frame uses current colour only)

### Vertex Shader Pipeline

1. **Morph targets** (if enabled): accumulates weighted **position / normal / tangent** deltas from the packed SSBO (layout `[positions, normals, tangents]` per target)
2. **Skinning** (if enabled): blends joint matrices using per-vertex joint indices and weights
3. **Transform**: applies either the blended skin matrix or the model matrix to produce world-space position
4. **TBN construction**: transforms the morph-blended normal and tangent by the normal matrix, orthogonalises T against N, builds B via `cross(N, T) * tangent.w`, and passes `mat3(T, B, N)` to the fragment shader for tangent-space normal mapping
5. **Second UV set + view-space depth**: forwards `fragTexCoord1 = inTexCoord1` (location 8) and `fragViewDepth = -(view * worldPos).z` (location 7) for cascade selection

### Fragment Shader

The forward fragment shader picks a UV stream per sample (`pickUv(material.texCoordIndices.X)` returns TEXCOORD_0 or TEXCOORD_1), applies KHR_texture_transform (`applyUvTransform` does scale → CCW rotate → translate) and samples the right texture. If `material.extraFlags.z == 1` (KHR_materials_unlit) it writes `vec4(baseColor, alpha)` and returns immediately, skipping all lighting. Otherwise it runs a PBR Cook-Torrance BRDF (GGX + Schlick Fresnel + Smith G) **per light in a fixed-size loop over `light.lights[0..lightCount]`** — directional, point, and spot all share the same BRDF; point/spot add the KHR_lights_punctual `windowing² / d²` distance attenuation, spot adds a smooth cone factor, and only the first directional (`i == 0 && type == 0`) carries the directional shadow term. After the loop it adds diffuse IBL from the irradiance cubemap and specular IBL via the prefiltered cubemap + BRDF LUT split-sum **with Fdez-Aguera multi-scatter compensation** for energy-conserving rough conductors. **KHR_materials_transmission** then attenuates the diffuse lobes by `(1 − transmission)` and adds a separate transmission lobe on top, gated on `KHR_materials_volume` thickness: thin-walled materials (no volume) use a basecolor × env-irradiance-tint scatter (view-independent — no screen-space image to track the camera), while volumetric materials (thickness > 0) sample the captured sceneColor along the refracted ray, blurred by roughness, for scene-behind-glass refraction. Double-sided surfaces flip the shading normal to face the viewer on back faces (`gl_FrontFacing`) so the view-dependent terms don't evaluate against an inward-facing normal. The directional shadow term chooses one of 4 cascades with a 10% blend band at boundaries. Non-skinned receivers sample the full CSM; skinned receivers combine the world-only CSM with their second-depth per-object self-shadow map.

## Limitations

- **Forward pipeline collapse is partial — 2 variants, not 1.** The opaque and
  double-sided forward pipelines are merged into a single pipeline using dynamic
  cull mode (`VK_DYNAMIC_STATE_CULL_MODE`, core Vulkan 1.3, set per draw). The
  BLEND pipeline is *not* folded in: collapsing it too would require dynamic
  colour-blend state (`VK_EXT_extended_dynamic_state3`'s
  `extendedDynamicState3ColorBlendEnable` / `…ColorBlendEquation`), which the
  current MoltenVK reports as **unsupported** — the extension is advertised but
  those two feature bits are `false`, because Metal bakes blend state into the
  render-pipeline descriptor rather than letting it vary dynamically. On a
  desktop driver that exposes those features the blend variant could fold into
  the same pipeline; on MoltenVK it stays a separate static-blend pipeline.

## Setup

Setup [vcpkg](https://vcpkg.io/en/) on the build machine, and ensure that `VCPKG_ROOT` is available in the `PATH` environment variable.
Details of how to do this can be found at steps 1 and 2 in this [getting started doc](https://learn.microsoft.com/en-gb/vcpkg/get_started/get-started).
Ensure the `vcpkg` executable is available in your `PATH`.

Configure CMake, which will install and build dependencies via vcpkg.
Additionally, since I use `NeoVim`, I export the `compile_commands.json` to the build directory for use with `clangd`:

```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

Build:

```bash
cmake --build build
```

Run the Catch2 test binary:

```bash
./build/test_fire_engine
```

Run the full CTest suite, including the graphics-layer include guard:

```bash
ctest --test-dir build --output-on-failure
```

Run the application:

```bash
cd build && ./fireEngineApp
```

The app accepts two optional positional arguments:

```bash
./fireEngineApp <scene.gltf> <skybox.hdr>
```

Both fall back to built-in defaults when omitted. A single `.hdr`/`.exr` argument is treated as a
skybox path, so `./fireEngineApp nightbox.hdr` keeps the default scene and swaps only the
environment.

## Dependencies

Managed via the vcpkg manifest (`vcpkg.json`):

- `vulkan-headers` — Vulkan API headers
- `fastgltf` — glTF 2.0 parser
- `stb` — image loading (stb_image, incl. HDR)
- `ktx` — KTX2 / Basis Universal textures
- `catch2` — Catch2 v3 test framework
- `imgui[glfw-binding,vulkan-binding]` — debug overlay (ImGui core + GLFW platform
  backend + Vulkan renderer backend). The engine links Vulkan and GLFW directly; the local
  `cmake/fireengine_imgui.cmake` helper wraps vcpkg's `imgui::imgui` archive without inheriting
  its transitive Vulkan/GLFW link interface, avoiding duplicate static-library entries.

**Toolchain: Current built with Apple Clang** (`/usr/bin/clang++`). The vcpkg toolchain inherits the
project's compiler (via `CC`/`CXX`), so all ports build from the manifest. The project
formerly used Homebrew g++-15, which can't parse the Apple SDK framework headers — that
broke the vcpkg builds of gtest/glfw3/imgui and forced classic-mode global installs plus
a vendored imgui backend; the Clang switch removed all of that.

Also requires:

- Vulkan SDK (for `glslc` shader compiler)

## Assets

All glTF models are from the Khronos glTF Sample Models [repository](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main): AlphaBlendModeTest, AnimatedCube, AnimatedMorphCube, BoomBoxWithAxes, BoxAnimated, BrainStem, CesiumMan, DamagedHelmet, Fox, InterpolationTest, LightsPunctualLamp (exercises KHR_lights_punctual + KHR_materials_transmission), MetalRoughSpheres, MorphPrimitivesTest, OrientationTest, RecursiveSkeletons, RiggedSimple, TextureCoordinateTest, TextureLinearInterpolationTest, TextureSettingsTest, VertexColorTest.

HDR equirectangular skyboxes (`skybox.hdr`, `nightbox.hdr`) drive the IBL precompute.
