[![CI](https://github.com/nnewson/fireEngine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nnewson/fireEngine/actions/workflows/ci.yml)

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
- **Cascaded directional shadows with skinned self-shadowing** — 4 cascades, 2048×2048 per cascade in a 2D-array depth image, Practical-Split-Scheme cascade boundaries (λ = 0.5 blend of linear and log-uniform splits) over 0.1m–50m. Directional shadows use shared comparison sampling with 10% blend bands at cascade boundaries and a rotated 17-tap Poisson PCF kernel at a 2-texel radius. **Bias is derived per view, not per cascade index**: one law (`graphics/shadow_bias.hpp`, mirrored by `shaders/shadow_bias.glsl` and shared by the cascade, self-shadow, spot and point receivers) converts a texel-relative policy into stored depth using each view's own fitted world-per-texel and depth conversion, so nothing assumes a scale that doubles per cascade — on the acceptance scene the real footprint ratios are 1.84 / 1.71 / 2.11. Spot maps carry the ray-forward cosine and an inverse-square depth term; point maps size their footprint by the cube face's major axis while comparing radial distance. The kernel's taps are normalised to unit support so "radius" means the same thing to the sampler and to the bias that must clear it. Skinned meshes avoid same-map self-shadow acne by receiving world shadows from a world-only directional map plus a tightly-fit dual-depth per-object self-shadow map. The self-shadow second pass culls front faces (`vk::CullModeFlagBits::eFront`) so only back-facing geometry rasterises, eliminating per-fragment discard coin-flips on marginal surfaces that previously produced random per-pixel flicker. Shadow casters are **material-aware**: an `alphaMode: MASK` caster casts its cutout, not its quad — the depth-only pass applies the same base-colour alpha test as the forward shader, from the same bindless material entry (the cutoff, UV set, `KHR_texture_transform` and sampler all come from one declaration in `shaders/material.glsl`, so the two cannot drift) — and a **double-sided** caster culls nothing rather than front faces, so a sheet authored face-on to the light casts a shadow instead of nothing at all. Cull mode is dynamic state on all four shadow pipelines (opaque / masked x the per-family face policy). Alpha-masked casters draw at full detail: a cutout's silhouette comes from alpha and UVs, which the simplifier's deviation channels do not measure, so shadow LOD reports `AlphaMaskedFallback` rather than guessing. A mesh can be authored **receive-only** with glTF `extras.Shadow = {"Casts": false}` — required for any wide flat receiver, since a floor that casts writes its own depth into every cascade and self-shadows its entire surface
- **KHR_materials_unlit** — flagged materials skip BRDF/IBL/shadow entirely and output the textured base colour directly. Used for skybox cards, foliage, decals, UI quads
- **KHR_texture_transform** — per-slot UV offset/scale/rotation from the extension is applied to each texture sample. Identity by default
- **Multi-light scenegraph** — `Light` is a first-class component variant alongside Camera / Mesh / Animator / Empty. Type enum is **Directional / Point / Spot**, with colour, intensity, range, and inner/outer cone angles per spec. Each frame the scenegraph walks all lights into a packed `Lighting` array (cap `kMaxLights = 8`), the renderer picks the first directional as the CSM source, and the forward shader runs a per-fragment loop over the array. Point/spot use the KHR_lights_punctual attenuation (`windowing² / d²` with `windowing = clamp(1 − (d / range)⁴, 0, 1)`); spot adds a smooth cone factor on top
- **KHR_lights_punctual import** — glTF lights become `Light` nodes, transform-driven. FireEngine seeds a default directional Sun only when the asset hasn't authored one
- **KHR_materials_transmission** — diffuse is attenuated by `(1 − transmission)` and a transmission lobe is added on top, gated on `KHR_materials_volume` thickness. **Thin-walled** materials (no volume, e.g. a paper lamp shade) scatter to a uniform basecolor × env-irradiance tint — view-independent, so a bright source behind the surface can't smear into a camera-tracking highlight. **Volumetric** materials (thickness > 0, e.g. frosted glass / the TransmissionRoughnessTest panels) sample the captured post-opaque sceneColor mip chain along the refracted ray for screen-space scene-behind-glass refraction, blurred by roughness, then attenuated by Beer-Lambert absorption over the volume
- **Authored-camera adoption** — `GltfLoader::findFirstCamera` walks the default scene's node tree DFS and returns the first camera-bearing node's view; FireEngine reframes its runtime camera to match before the first frame
- **GPU particle system (compute-driven)** — `ParticleEmitter` is a scene component, gathered each frame like `Light` into a Vulkan-free `EmitterState`. A renderer-owned `ParticleSystem` simulates a pooled particle SSBO with a **compute shader** (spawn dead slots at the emitter up to a per-frame budget via an atomic spawn-claim; integrate the rest under gravity), then renders the pool as **instanced camera-facing billboards** (`cmd.draw(6, poolCount)`, per-instance data read from the SSBO by `gl_InstanceIndex`) blended **additively into the HDR target** so bloom catches the glow. **Soft particles**: the fragment shader fades against sampled scene depth so particles dissolve smoothly into geometry with no hard clip edge. Built on the compute-pipeline + synchronization2 buffer-barrier path
- **HDR offscreen forward pass + bloom + ACES post-process** — forward writes into an R16G16B16A16 target. **Dual-filter bloom** (6-mip RGBA16F chain at half-screen res, 13-tap CoD downsample with Karis-average on the first pass to suppress fireflies, 9-tap tent upsample with additive blend) produces a low-pass HDR contribution. Post-process mixes the HDR target with bloom mip 0 (`bloomStrength = 0.04` default; `0` is bit-identical to a no-bloom path), then ACES tonemap + gamma 2.2 before presenting
- **SSAO + contact shadows** — a **depth prepass** (reusing the forward vertex shader with `invariant gl_Position`; the forward pass loads it with `LESS_OR_EQUAL`) fills the shared depth buffer before lighting. It applies the material's **alpha cutout** through the same shared test the forward and shadow passes use, so a MASK material writes depth only where it is actually opaque — otherwise it occludes across its own holes, hiding geometry behind a leaf card and making the AO below treat it as a solid sheet. A renderer-owned `Ssao` subsystem then reconstructs view-space position + normal from depth alone (no normal G-buffer — analytic unprojection from the projection matrix) and writes an **R8G8** target: R = hemisphere-kernel ambient occlusion, G = a sun-direction screen-space **contact-shadow** ray-march. The forward shader samples it to multiply SSAO into the IBL/ambient terms and the contact term into the **direct sun** (ambient stays on pure CSM). A **depth-aware bilateral blur** (5×5, view-space-Z edge-stop) smooths the per-pixel sampling/march noise without bleeding across silhouettes, with TAA carrying the temporal denoise. SSAO and contact shadows are on by default; contact shadows fill the CSM's short-range contact gap and use an N·L gate plus view-Z-scaled depth window and silhouette edge guard to avoid screen-space streaks. Live overlay sliders (radius / bias / intensity / power, contact length) and a `--debug-ssao` view
- **GPU soft-body / cloth (XPBD)** — `-c` drops a cloth that simulates entirely on the GPU, or author one on any glTF mesh with `extras.Cloth` (samples: `assets/ClothSheet/ClothSheet.gltf`, `assets/ClothBanner/ClothBanner.gltf`). A renderer-owned `SoftBodySystem` runs an XPBD compute solver each substep: `cloth_predict` integrates gravity + wind, `cloth_solve` projects distance constraints **graph-coloured** into race-free batches (Gauss-Seidel by colour), `cloth_collide` pushes particles out of world colliders, and `cloth_finalize` writes solved positions + normals (recomputed from a per-vertex→triangle **CSR adjacency**, so arbitrary meshes work, not just grids) into a storage **vertex buffer** the forward/shadow passes read — so the cloth renders, lights, and casts shadows through the normal forward path (double-sided), no new render shaders. The solver is **descriptor-free**: every buffer reaches the shaders as a `bufferDeviceAddress` pointer. Collision primitives (plane / sphere / box / capsule) are gathered each frame from `PhysicsWorld` (`gatherColliders`) plus a ground plane. Constraint stiffness is authored per type (structural/shear stiff, bend soft); substeps, a global **compliance multiplier**, damping, gravity, and wind are **live overlay sliders**. Built on the same compute + buffer-barrier path as particles
- **Temporal anti-aliasing (TAA)** — sub-pixel Halton(2,3) projection jitter plus velocity-buffer history accumulation anti-aliases geometry edges *and* specular/shading shimmer (unlike MSAA, which only covers geometry edges). The forward + transmission passes write a screen-space motion-vector attachment; the resolve reprojects the previous frame's history along it (`historyUV = uv − velocity`), neighbourhood-clamps to the current 3×3 to suppress ghosting/disocclusion, and blends. Motion vectors are jitter-free so the jitter cancels in accumulation. Per-node previous-world-matrix tracking feeds rigid + animated motion (skinned deformation is camera-motion-only in v1); particles render after the resolve, kept out of history. `--no-taa` reverts to the raw image, `--debug-velocity` visualises the buffer
- **Frustum culling (camera + shadow casters)** — built on a reusable fat-AABB BVH (`AabbBvh<T>`, the same core the physics broadphase and static-mesh triangle index use). Two stages: a **persistent scene BVH** (`SceneCuller`, an `AabbBvh<Node*>` over rigid renderables) pre-culls each frame against the union of the camera frustum and every shadow caster's frustum, so off-screen nodes skip draw-building entirely (no UBO writes, no per-vertex bounds) — `O(log N + visible)` instead of `O(N)`; then a **precise per-pass cull** drops the survivors that fall outside a given pass's frustum (`buildDrawBuckets` for the camera, per-cascade/spot/point-face in the shadow pass). Frustums are 6 Gribb–Hartmann planes (Vulkan `[0,1]` depth) with a conservative positive-vertex AABB test (no false negatives). Deformable (skinned/morph) meshes skip the coarse BVH (bind-pose bounds under-cover the animated pose) and rely on the precise stage's exact per-frame world bounds. The coarse union is a strict superset of what any pass keeps, so culling never drops a visible draw. Live overlay toggle + tracked/visible/culled counts; off submits everything (A/B + regression escape hatch)
- **Progressive mesh level-of-detail (discrete → VIPM → VDPM)** — a from-scratch **attribute-aware Garland–Heckbert quadric-error simplifier** (`graphics/mesh_simplifier`) records an ordered edge-collapse stream per static mesh at load time; all levels index the *same* vertex buffer (only index data is added). The error quadric lives in **R⁵ (position + weighted UV)**, so collapses that would stretch the texture parameterisation are ordered *last*; **position welding** restores connectivity across glTF's seam-split vertices, a **wedge-preserving emit** keeps each corner's own UV (nearest-wedge), and a **chart veto** stops a collapse from crossing a UV/normal seam. That one collapse stream drives three selectable modes (overlay: Discrete / Continuous / View-dependent): **Discrete** picks the coarsest whole-mesh level whose screen-space error fits a pixel budget; **Continuous (VIPM)** geomorphs the collapsing vertices' full render attributes into the exact next level to dissolve the pop; **View-dependent (VDPM)** promotes the stream to a per-instance **vertex forest + active front** that refines *different regions of one mesh to different detail* each frame — from four screen-space channels (geometry, UV-seam, shading-normal, tangent) plus silhouette boost and a conservative back-face gate, with a joint refinement-only foldover/coverage repair — so it matches the discrete mesh's silhouette and shading at a fraction of the triangles. Overlay toggle + mode selector + pixel error budget slider + triangles-drawn readout + a per-LOD debug tint; `--lod-mode discrete|continuous|view-dependent` selects the starting mode at launch. VDPM draws are issued via **`drawIndexedIndirect`** from a per-instance indirect-command buffer (the count on the GPU) — the plumbing the GPU-driven front consumes; every other draw stays direct. A **GPU-driven front** backend (the **"GPU-driven front" overlay checkbox** in the Mesh LOD panel — a reload-free runtime toggle, needs `--lod-mode view-dependent` and a compute-capable device; shown "unsupported" otherwise) runs the whole per-frame front lifecycle — score, refine/coarsen, foldover/coverage repair, and seam-preserving emit — on the GPU in compute, and the draw consumes the GPU-emitted index/indirect buffers directly (the per-instance CPU front work is skipped); the CPU front is retained as an automatic per-mesh fallback and the same-camera A/B reference. It is **on by default** wherever the device supports it; force it with `--vdpm-gpu` or disable it with `--no-vdpm-gpu` (repeated flags are last-one-wins). See [`docs/lod.md`](docs/lod.md)
- **Debug + profiling overlay (Dear ImGui)** — a runtime overlay (Dear ImGui 1.92 on the Vulkan dynamic-rendering backend, drawn into the swapchain after post-process) toggled with **F1** (`--overlay` to start visible). Shows a **CPU frame-time/FPS plot** and **per-pass GPU timings** via a timestamp `VkQueryPool` (`GpuProfiler`), which distinguishes a device that cannot time (zero `timestampPeriod` / `timestampValidBits`, reported once at startup with both numbers) from a ring slot that has nothing to report yet — the panel says which. Every pass stamps both boundaries at bottom-of-pipe, so the values are consecutive deltas on one timeline, and deltas are modular in the queue's `timestampValidBits` (Vulkan defines timestamp overflow as wrapping inside that width). The figure below the rows is labelled **measured pass sum**, not a total: it adds the instrumented passes only, plus a **live tunables panel** (`RenderTunables`) for TAA (history blend, sharpen, on/off), frustum culling (on/off + tracked/visible/culled counts), mesh LOD (on/off + mode + pixel error budget + triangles-drawn + a reload-free GPU-driven-front backend toggle), a **Shadows (SH-01)** panel (per shadow-view-family and per physical slot: raster passes, drawn/candidate draws + triangles, per-view LOD histograms, and the family's GPU time), the debug-view dropdown (incl. a **LOD tint** (`--debug-lod`), a **Shadow LOD tint** (`--debug-shadow-lod`, with `--no-shadow-lod` as the full-detail control and `--shadow-budget` / `--shadow-ratio` to sweep the calibration — see `tools/shadow_lod_sweep.sh`, with `--shadow-focus <group>:<slot>` to pick the view it follows — e.g. `cascade:3`, `point:0:4` — resolved once at startup to that slot's logical view and then followed across slot compaction) that colours each mesh by the level ONE shadow view picked for it — the view focused in the Shadows panel, or cascade 0 by default, since after SH-03 a caster holds a different level per view — with neutral grey for meshes that view has no level for, and a **Joints** view — `--debug-joints` — that replaces the scene mesh with a per-link RGB axis gizmo + "index: bone-name" labels to identify ragdoll joints for hinge authoring) + no-shadows, bloom/diffuse-IBL/specular-IBL/sun-intensity, and particle emitter rate/lifetime/size — all editable without a recompile. Camera input is suppressed while a widget is being driven
- **Reproducible frame capture** — `--capture <path.png>` writes the numbered frame (`--capture-frame N`, counted in frames rather than seconds, so any machine captures the same render ordinal — identical *content* additionally needs a static scene, since animation and physics still advance on wall-clock `dt`) and exits (non-zero if the file could not be written), copying the **final swapchain image** — post-process and overlay included — straight before present. Swapchain `TRANSFER_SRC` is requested only when capture is asked for, after checking the surface supports it; 8-bit BGRA/RGBA formats are supported and anything else is rejected rather than guessed. With `--no-lod` (full detail, forward *and* shadow selection) it makes an A/B pair, which is how the shadow-LOD reference images in [`docs/acceptance-testing.md`](docs/acceptance-testing.md) are regenerated
- **Runtime logging** — diagnostics route through `core/log.hpp` with `debug`/`info`/`warn`/`error`/`off` levels and categories (`app`, `general`, `gltf`, `physics`, `ragdoll`, `render`). `FE_LOG` controls the global threshold and per-category overrides, e.g. `FE_LOG=ragdoll:debug` for ragdoll settle diagnostics or `FE_LOG=render:debug` for Vulkan extension dumps
- **Keyframe animation** with per-channel interpolation (LINEAR with SLERP for quaternions, STEP, CUBICSPLINE with in/out tangents) across rotation, translation, scale, and morph weight channels; looping playback; runtime animation selection via `AnimationState`
- **Skeletal skinning** — GPU joint matrix blending, up to 64 joints per skin
- **Morph target animation** — vertex POSITION + NORMAL + TANGENT deltas uploaded as a single packed SSBO, blended by weights in the vertex shader (up to 8 targets per mesh). Tangent morphs feed the TBN reconstruction so facial-rig normal mapping stays correct mid-blend
- **Alpha blending and masking** — three forward pipeline variants (opaque, double-sided opaque, blend) dispatched per draw from the material's `AlphaMode`/`doubleSided` flags. MASK handled via a fragment-shader discard test; BLEND uses straight-alpha blending with depth-write disabled and back-to-front sort of translucent draws
- **Scenegraph architecture** — tree of Nodes with Component variants (Camera, Animator, Mesh, Empty, **Light**, **ParticleEmitter**) that propagate transforms, an `InputState` bundle, and draw commands. Node transforms store rotation as a quaternion so orientations from glTF round-trip exactly
- **Custom collision and physics path** — glTF `extras.Physics` can create `Static`, `Kinematic`, and `Dynamic` bodies with layer/mask filtering, authored AABB/box/sphere/capsule/convex-hull collider shapes, linear velocity, mass, restitution, friction, and gravity scale. `PhysicsWorld` owns body/collider state, a `BroadPhase` (default `DynamicAabbTreeBroadPhase` — a fat-AABB BVH; `SweepAndPruneBroadPhase` injectable via the `PhysicsWorld(unique_ptr<BroadPhase>)` constructor) gathers AABB candidate pairs, `NarrowPhase` builds a shape-specific `ContactManifold` per pair (sphere/box/capsule analytic + box/box SAT, plus a **GJK/EPA convex path** for `ConvexHullShape` colliders and a speculative-margin path that emits gap contacts for separated-but-approaching pairs so fast movers don't tunnel), a **TGS soft-step `ContactSolver`** (P9.2: the fixed step is substepped — warm-start → soft/compliant normal constraint → per-substep relax pass → end-of-step restitution at the true impact velocity, replacing the old split-impulse position pass) resolves them with **full rigid-body rotation** (warm-started normal + friction impulses with lever-arm torque, per-shape inertia tensors and quaternion orientation integration — resting stacks settle and sleep, boxes topple and rest on a face), and `SceneGraph::submitPhysics` / `SceneGraph::applyPhysics` bridge scene-authored and physics-authored transforms each frame. **Debug tooling**: `--debug-physics` (and the overlay "Physics debug" panel) draws immediate-mode wireframes for broadphase AABBs, collider shapes, and contact normals via a renderer-owned `DebugDraw` line pass (x-ray or depth-tested), and a **determinism harness** (`test_physics_determinism`) replays a fixed-step scene and hashes body state so accidental non-determinism or behaviour drift is caught in CI
- **Physics queries, triggers, and a character controller** — first-class spatial queries on `PhysicsWorld`: `raycast`/`raycastAll` (analytic ray vs sphere/OBB/capsule/convex + Möller–Trumbore triangle), `shapecast` (GJK conservative advancement), and `overlapSphere`/`overlapShape`, all with layer/mask `QueryFilter`ing (brute-force over active colliders with an AABB reject; mesh colliders dispatch into their triangle BVH). A collider flagged `isTrigger` (glTF `extras.Physics` `"IsTrigger": true`) generates overlap **events** instead of a solver response — `triggerEvents()`/`collisionEvents()` return per-step enter/stay/exit `ContactEvent`s diffed from the overlap set. On top of the queries, a kinematic-capsule **`CharacterController`** does collide-and-slide movement with slope limits, step up/down, and grounded snapping (`-k` runs a patrol demo)
- **Backend-decoupled graphics layer** — graphics classes use opaque handles (`BufferHandle`, `TextureHandle`, `DescriptorSetHandle`, `PipelineHandle`) and emit `DrawCommand` structs with no Vulkan dependencies. IBL cubemaps, BRDF LUT, shadow map, bloom chain are all owned by the render layer and referenced through the same handle types
- **Vulkan rendering** via vulkan.hpp C++ bindings, targeting Vulkan 1.4 with **dynamic rendering** (no `VkRenderPass`/`VkFramebuffer` objects) and **synchronization2** barriers/submits throughout. CPU↔GPU frame pacing uses a single monotonic **timeline semaphore** (the swapchain acquire/present semaphores stay binary, as WSI requires). Built around a **frequency-split forward descriptor layout**: set 0 holds 7 per-object/per-draw buffer bindings (frame, camera, skin, previous-skin, morph, morph-target, VIPM), set 1 holds 14 globals shared by every draw (light UBO, five shadow maps, debug image, compare/debug samplers, three IBL textures, sceneColor, SSAO), and **set 2 is bindless** — one global `sampler2D[]` texture array (indexed by texture handle) plus a global materials SSBO (indexed by a per-draw push constant). Set 0 is **pushed inline per draw** (core 1.4 push descriptors), not allocated; when a forward pipeline becomes active it is established before allocated sets 1 and 2 are bound through the same compatible layout. The shadow pass uses the same push-descriptor model for its set 0, and opts into set 2 as well (its masked fragment paths read the same materials SSBO + texture array) — a pipeline that wants bindless without forward globals declares an **empty set 1**, so "bindless is set 2" holds in every pipeline rather than shifting with each one's set list. Separate descriptor layouts exist for skybox, shadow, post-process, and bloom passes
- **Single source of truth for tunables** — every scalar rendering knob (light intensity, IBL strengths, shadow biases, cascade split λ, bloom strength, IBL extents, camera FOV) lives in `include/fire_engine/render/constants.hpp`. GPU data-layout limits that the Vulkan-free graphics layer also needs (frames-in-flight, joint/morph/light counts, shadow caster caps + matrix layout, cascade count) live one layer down in `include/fire_engine/graphics/gpu_limits.hpp`, which `constants.hpp` includes — so render-side code still sees every constant through one include, while graphics headers stay free of `render/`
- **GPU memory** is sub-allocated through the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) rather than one `vkAllocateMemory` per resource — buffers and images are owned as a resource + sub-allocation unit behind small `vk::raii`-style RAII wrappers, resource handles carry a generation so recycled slots are detectable, and load-time texture uploads coalesce into a single batched submit
- **Texture mapping** via [stb_image](https://github.com/nothings/stb), including HDR equirectangular loading for the skybox; uploaded to GPU through staging buffers
- **First-person camera** with keyboard (WASD + E/F for vertical) and mouse controls
- **GLSL shaders** compiled to SPIR-V at build time via `glslc` (from the vcpkg `shaderc` port)

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
- **`physics/`** owns simulation state. `PhysicsWorld` stores bodies, colliders, shapes, materials, the broadphase, the narrowphase, and the `ContactSolver` (a TGS soft-step solver: a substepped solve with warm-started normal/friction impulses, a soft/compliant normal constraint, a per-substep relax pass, end-of-step restitution, and **full rotational dynamics** — per-shape inertia tensors and quaternion orientation integration, so boxes topple and rest on a face). Its implementation is split across `src/physics/physics_world*.cpp` by responsibility: core create/step/solve/sleep, shape composition, spatial queries, articulation stepping, debug extraction, and event/collider export. `PhysicsBody` stores type, velocity, mass/inverse mass, angular velocity, local inverse inertia, gravity scale, restitution, and friction. `ColliderShape` supports AABB, box, sphere, capsule, and convex-hull authoring (the hull built from a mesh by `core/convex_hull_builder`); the broadphase uses a local AABB while `PhysicsWorld::worldShape` composes the authored shape with the body transform into a `WorldShape` for the shape-specific narrowphase.
- **`scene/`** stores only opaque physics handles. Scene nodes stay responsible for transforms, hierarchy, input, animation, and rendering; physics ownership stays in `PhysicsWorld`.

The frame loop makes the authority split explicit:

```cpp
scene_.update(input_state);
scene_.submitPhysics(physics_);

accumulator += dt;
while (accumulator >= fixedDt)
{
    physics_.step(fixedDt);
    accumulator -= fixedDt;
}

const float alpha = accumulator / fixedDt;   // fixed-step render interpolation
scene_.applyPhysics(physics_, alpha);
```

`submitPhysics()` pushes non-dynamic scene transforms into `PhysicsWorld`: static bodies are scene-authored, and kinematic bodies are gameplay/input-authored. `PhysicsWorld::step()` snapshots each body's start-of-step pose (the render-interpolation baseline), refreshes collider AABBs, updates broadphase candidates, builds shape-specific contact manifolds, then runs the TGS soft-step solve (substepped gravity → warm-start → bias solve → integrate → relax, with end-of-step restitution and a kinematic-only position pass), and captures previous positions. `applyPhysics(physics, alpha)` pulls non-static physics transforms back onto scene nodes, so dynamic simulation and kinematic collision correction are visible before rendering. Because the sim advances in fixed 60 Hz increments while the display refreshes faster, `alpha = accumulator / fixedDt` blends each body between its previous and current pose (position lerp, orientation slerp) so motion stays smooth on a 120 Hz panel; this is purely visual, leaving the simulated state (and thus determinism) untouched. Articulated ragdoll bones are driven separately by `Ragdoll::syncNodes(alpha)`, which interpolates the articulation's link transforms the same way.

Physics can be authored in glTF through node `extras.Physics`. The loader creates bodies/colliders, assigns handles to the node, and rejects unsupported combinations such as a `Dynamic` body on a `Controllable` node.

### Physics Demos

`assets/physics_demos/` holds one minimal, self-contained glTF scene per physics capability — simple untextured geometry whose dimensions match its collider, authored purely to *show* a feature behaving. The scenes are emitted by `assets/physics_demos/generate.py` (regenerated automatically at build time; the shared geometry/glTF machinery lives in `tools/assetgen/`) and each is mirrored by a headless replay test in `tests/physics/test_demos.cpp` that rebuilds an equivalent `PhysicsWorld`, steps the fixed-step solver, and asserts the labelled outcome — so each demo is both a visual showcase and an automated regression guard.

| Demo (`physics_demos/…`) | What it verifies | Headless test (`tests/physics/test_demos.cpp`) |
|---|---|---|
| `FallRestDemo.gltf` | A Dynamic box falls onto a Static floor, settles flat, and goes fully still (it sleeps). End-to-end author→simulate smoke. | `Demos.FallRest.BoxComesToRestOnFloor` |
| `RestitutionDemo.gltf` | Three spheres with restitution 0.0 / 0.5 / 0.9 dropped from the same height bounce to visibly different rebound heights. | `Demos.Restitution.HigherRestitutionBouncesHigher` |
| `FrictionRampDemo.gltf` | Two boxes on a 25° ramp: a high-friction box holds while a low-friction box slides off and grinds to a halt on the rough floor (combined friction is `sqrt(a·b)`). | `Demos.Friction.HighFrictionStaysLowFrictionSlides` |
| `StackDemo.gltf` | A 5-high tower of boxes dropped with small gaps settles into a resting stack and sleeps rather than buzzing apart — the TGS soft-step solver (P9.2) quiesces a tall tower cleanly (the old solver was capped at three). | `Demos.Stack.SettlesAndStaysStill` |
| `ToppleDemo.gltf` | A tall box tilted 30° — past its ~16.7° balance angle — topples onto its long side and comes to rest (full rotational dynamics: inertia + lever-arm torque). | `Demos.Topple.TallBoxTopplesOntoSide` |
| `ConvexHullDemo.gltf` | Tetrahedra (collider = `ConvexHull` built from the mesh) tumble through the GJK/EPA convex narrowphase, land on a face, and settle into a loose pile. | `Demos.ConvexHull.PileSettlesAtRest` |
| `SleepDemo.gltf` | A small stack settles and the island goes to **sleep** (with `--debug-physics` its colliders dim to the asleep colour); a striker then slides in along the floor, **wakes** it on impact, and friction stops the striker against the stack — so everything ends asleep on the floor. | `Demos.Sleep.StackSleepsThenWakesOnImpact` |
| `StaticMeshDemo.gltf` | Boxes + a sphere dropped into a triangulated valley (a Static `Shape:"Mesh"` triangle-mesh collider, not a box) land on the mesh surface and settle — contacts against the mesh's actual triangles. | `Demos.StaticMesh.BodiesSettleInValley` |
| `CompoundDemo.gltf` | An L-shaped body whose collider is a `Shape:"Compound"` of two boxes; its engine-aggregated centre of mass is offset toward the corner, so it rests stably on its bar instead of tipping. | `Demos.Compound.LShapeRestsOnFloor` |
| `SingleJointRagdollDemo.gltf` | The smallest ragdoll-authored scene: one skinned joint tagged with `extras.Ragdoll`, producing one capsule body and no parent-child constraints. | `Ragdoll.GeneratedSingleJointDemoAssetRestsOnFloor` |
| `TwoJointRagdollDemo.gltf` | A deliberately tiny skinned skeleton with two joints and one ragdoll constraint; used to verify the basic parent→child joint connection before scaling up to a humanoid graph. | `Ragdoll.GeneratedTwoJointDemoAssetStaysConnected` |
| `CesiumMan/CesiumManRagdoll.gltf` | The full showcase: the Khronos CesiumMan (19-bone skinned humanoid) tagged `extras.Ragdoll{Articulated}`, built as a **reduced-coordinate articulation** (Featherstone ABA, floating base, spherical joints with velocity-level cone-twist limits, self-collision). It drops, crumples believably, and settles cleanly at rest — the failure mode maximal-coordinate ragdolls limit-cycle on — coming to a natural, dead-still stop (settle assists in the articulation integrator damp the residual limb drift and the body's slow yaw once it has landed). Run with `-f` for the floor. | `Ragdoll.CesiumManRagdollAppFaithful` |

Run a demo (add `--debug-physics` to overlay collider/contact wireframes); paths are relative to `build/`:

```bash
cd build
./fireEngineApp physics_demos/FallRestDemo.gltf      skybox.hdr --debug-physics
./fireEngineApp physics_demos/RestitutionDemo.gltf   skybox.hdr --debug-physics
./fireEngineApp physics_demos/FrictionRampDemo.gltf  skybox.hdr --debug-physics
./fireEngineApp physics_demos/StackDemo.gltf         skybox.hdr --debug-physics
./fireEngineApp physics_demos/ToppleDemo.gltf        skybox.hdr --debug-physics
./fireEngineApp physics_demos/ConvexHullDemo.gltf    skybox.hdr --debug-physics
./fireEngineApp physics_demos/SleepDemo.gltf         skybox.hdr --debug-physics
./fireEngineApp physics_demos/StaticMeshDemo.gltf    skybox.hdr --debug-physics
./fireEngineApp physics_demos/CompoundDemo.gltf      skybox.hdr --debug-physics
./fireEngineApp physics_demos/SingleJointRagdollDemo.gltf skybox.hdr --debug-physics
./fireEngineApp physics_demos/TwoJointRagdollDemo.gltf skybox.hdr --debug-physics
./fireEngineApp CesiumMan/CesiumManRagdoll.gltf      skybox.hdr -f
```

Run all the demo behaviour tests headlessly:

```bash
./test_fire_engine "[Demos]"
```

Two **query/character** demos are driven programmatically (the queries and the controller are issued from the main loop, not authored in glTF), so they run behind CLI flags rather than a `.gltf`:

```bash
./fireEngineApp -k skybox.hdr   # kinematic character controller walking a step-pyramid course
./fireEngineApp -q skybox.hdr   # query probe: a rotating fan of raycasts + overlap on a ring of bodies
```

Their physics is covered headlessly by `tests/physics/test_character_controller.cpp`, `test_physics_query.cpp`, and `Demos.Query.RaycastAndOverlapFindBodies`.

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
- **Safety checks** — `GltfLoader::ensureSupportedExtensions` walks `asset.extensionsRequired` and throws if any aren't in our supported set (so e.g. draco-compressed assets fail fast instead of producing corrupt geometry). Non-triangle primitives are skipped with a `log::warn` diagnostic rather than rendered as garbage.

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
5. `scene_.applyPhysics(physics_, alpha)` pulls dynamic and corrected kinematic transforms back into scene nodes — interpolated by `alpha = accumulator / fixedDt` between the last two simulated poses for smooth motion above 60 Hz — and resolves composed-world matrices.
6. `Renderer::drawFrame()` acquires a swapchain image and records the frame passes:
   - **Shadow passes** — directional cascades render both the full CSM and a world-only CSM that excludes skinned casters. Each skinned self-shadow slot renders two tightly-fit passes: the first captures the nearest light-facing surface, and the second samples that first depth and discards it so the forward shader can sample the next useful self-occluder. Spot and point shadow passes replay the same compatible shadow draw commands through their per-layer/per-face depth attachment views. Skin and morph still apply in the shadow vertex shader. **Which families record is one decision per frame** (`ShadowMapValidity`): the slot-addressed families disappear when nothing assigns them — no self-shadow caster, no spot or point caster, no skinned draw for the world-only CSM — while the main cascades still record (and clear) whenever there is a sun to fit them to, casters or not. The directional families additionally require a primary directional light, and `--no-shadows` clears everything. A skipped family costs nothing at all — no draws, no clears, no timestamps — and the *same* value is uploaded as a bit mask the forward shader reads, so its sampling path answers fully lit instead of reading depth left over from an earlier frame
   - **Forward pass** — begin the HDR offscreen pass, draw the skybox (LEQUAL depth, no write), then call `scene.buildDrawCommands(frameInfo, frustums, out)` through the Vulkan-free `RenderableScene` interface (the scene culls internally and emits draws); Mesh/Object emit `DrawCommand`s that the Renderer buckets into opaque, transmissive, and blend lists, sorts the blend bucket back-to-front by `sortDepth`, and replays through the same bind/draw loop resolving handles via `Resources`
   - **Transmission pass** — when transmissive draws are present, capture the opaque scene colour mip chain and replay transmissive draws so the shader can sample scene-behind-glass data
   - **TAA resolve** — the forward/transmission passes also write a screen-space velocity (motion-vector) attachment; the resolve reprojects the previous frame's accumulated history along that buffer, neighbourhood-clamps it against the current 3×3 to kill ghosting, blends, and blits the result back into the HDR target. Sub-pixel projection jitter (Halton(2,3)) drives the accumulation; particles render afterwards with the un-jittered projection so they stay out of history. Skipped under `--no-taa`
   - **Bloom downsample chain** — 6 fullscreen-triangle passes. Pass 0 reads the HDR target with the Karis-average 13-tap kernel (firefly suppression), writing mip 0 of the bloom chain. Passes 1..5 read the previous bloom mip and write the next, plain CoD weights
   - **Bloom upsample chain** — 5 fullscreen-triangle passes back up the chain (mip 5 → mip 4 → … → mip 0). Each samples its source mip with a 9-tap tent kernel and **additively blends** onto the destination mip (preserved by `loadOp=eLoad`). The final write to mip 0 carries the summed contribution from every coarser mip
   - **Post-process pass** — begin the swapchain-format pass, draw a fullscreen triangle that samples both the HDR target and bloom mip 0, mixes them by `bloomStrength`, and applies ACES + gamma 2.2. The swap image is left in colour-attachment layout (the present transition is deferred) so the overlay can draw over it
   - **Debug overlay** — when the ImGui overlay has content, draw it into the swap image (dynamic rendering, loadOp Load); then `transitionSwapchainToPresent` performs the final colour-attachment → present transition
   - Per-pass GPU timestamps wrap each of the above via `GpuProfiler`; results are read back a frame-cycle later for the overlay
7. Renderer submits the command buffer and presents

### Rendering Pipeline

- Forward descriptor layout is split by update frequency into **set 0 (per-object/per-draw, 7 bindings)**, **set 1 (forward globals, 14 bindings)**, and **set 2 (bindless materials, global)**. Set 0 is **pushed inline at draw time via core 1.4 push descriptors** (`vkCmdPushDescriptorSet`) — no per-object descriptor set is allocated. On each transition to a forward pipeline, set 0 is pushed before allocated sets 1/2 are rebound through the same compatible layout; subsequent draws only replace set 0.
- **Set 0 — per-object/per-draw buffer state**:
  - 0 frame UBO (model / view / projection + camera position)
  - 3 skin UBO (joint matrices, `mat4[64]`)
  - 4 morph UBO (metadata + weights)
  - 5 morph targets SSBO — `[positions, normals, tangents]` per target as `vec4[]`
  - 28 VIPM geomorph SSBO
  - 29 camera UBO (view/projection/camera position/current+previous view-projection)
  - 30 previous-skin UBO (last-frame joint matrices for deformation motion vectors)
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
  - 13 SSAO/contact-shadow texture

  On swapchain resize, only the `kMaxFramesInFlight` set-1 descriptors need rewriting (sceneColor, post-process targets, and any future recreated globals) via `Descriptors::updateGlobalDescriptors`; the global set-2 bindless descriptors are untouched, and forward set 0 is never allocated (pushed per draw).
- Separate descriptor layouts for the skybox (SkyboxUBO + samplerCube + LightUBO), shadow (ShadowUBO — per-object model + hasSkin — plus SkinUBO + MorphUBO + MorphTargets SSBO + first self-shadow depth/sampler, with the view's own transform arriving in `ShadowPushConstants` on the vertex/fragment stages), post-process (HDR sampler at 0 + bloom mip 0 sampler at 1, plus `PostProcessPushConstants { float bloomStrength }`), and bloom-down / bloom-up (single input mip sampler + `BloomPushConstants` on the fragment stage)
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
- Each Object owns its per-frame UBO/SSBO buffers; both the forward and the shadow set 0 are pushed inline per draw via core 1.4 push descriptors (no per-object descriptor set allocated in either pass)
- Depth buffering and swapchain recreation on window resize (HDR offscreen target, bloom chain, post-process descriptors, and the TAA velocity + history targets all rebuilt at new extent; the TAA history-valid guard resets so the first post-resize frame uses current colour only)

### Vertex Shader Pipeline

1. **Morph targets** (if enabled): accumulates weighted **position / normal / tangent** deltas from the packed SSBO (layout `[positions, normals, tangents]` per target)
2. **Skinning** (if enabled): blends joint matrices using per-vertex joint indices and weights
3. **Transform**: applies either the blended skin matrix or the model matrix to produce world-space position
4. **TBN construction**: transforms the morph-blended normal and tangent by the normal matrix, orthogonalises T against N, builds B via `cross(N, T) * tangent.w`, and passes `mat3(T, B, N)` to the fragment shader for tangent-space normal mapping
5. **Second UV set + view-space depth**: forwards `fragTexCoord1 = inTexCoord1` (location 8) and `fragViewDepth = -(view * worldPos).z` (location 7) for cascade selection

### Fragment Shader

The forward fragment shader picks a UV stream per sample and applies KHR_texture_transform in one call — `materialSlotUv(SLOT_*, material.texCoordIndices.X, uv0, uv1)` chooses TEXCOORD_0 or TEXCOORD_1 as the material asks, then scales → CCW-rotates → translates — and samples the right texture. Those helpers, the material struct and the bindless arrays live in the shared `shaders/material.glsl`, which the shadow pass' masked paths include too, so a cutout's shadow is tested against exactly the cutoff and UVs its surface uses. If `material.extraFlags.z == 1` (KHR_materials_unlit) it writes `vec4(baseColor, alpha)` and returns immediately, skipping all lighting. Otherwise it runs a PBR Cook-Torrance BRDF (GGX + Schlick Fresnel + Smith G) **per light in a fixed-size loop over `light.lights[0..lightCount]`** — directional, point, and spot all share the same BRDF; point/spot add the KHR_lights_punctual `windowing² / d²` distance attenuation, spot adds a smooth cone factor, and only the first directional (`i == 0 && type == 0`) carries the directional shadow term. After the loop it adds diffuse IBL from the irradiance cubemap and specular IBL via the prefiltered cubemap + BRDF LUT split-sum **with Fdez-Aguera multi-scatter compensation** for energy-conserving rough conductors. **KHR_materials_transmission** then attenuates the diffuse lobes by `(1 − transmission)` and adds a separate transmission lobe on top, gated on `KHR_materials_volume` thickness: thin-walled materials (no volume) use a basecolor × env-irradiance-tint scatter (view-independent — no screen-space image to track the camera), while volumetric materials (thickness > 0) sample the captured sceneColor along the refracted ray, blurred by roughness, for scene-behind-glass refraction. Double-sided surfaces flip the shading normal to face the viewer on back faces (`gl_FrontFacing`) so the view-dependent terms don't evaluate against an inward-facing normal. The directional shadow term chooses one of 4 cascades with a 10% blend band at boundaries. Non-skinned receivers sample the full CSM; skinned receivers combine the world-only CSM with their second-depth per-object self-shadow map.

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

Configure CMake, which will install and build dependencies via vcpkg. The `vcpkg`
preset selects the `Dev` build type (`-O2 -g`, assertions/validation still enabled)
and exports `build/compile_commands.json` for `clangd`:

```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

Build:

```bash
cmake --build build
```

Run the Catch2 test binary directly:

```bash
./build/test_fire_engine
```

Run the fast CTest suite (source root, via preset):

```bash
ctest --preset fast
```

Or from the build directory:

```bash
ctest --test-dir build --output-on-failure
```

Run the full Catch2 suite, including `[slow]` tests, plus the build-time guards (graphics-layer
includes, shared shader blocks, the shadow bias law, the shared GPU limits, the per-view shadow
matrix, and the `[release-contract]` tags):

```bash
cmake --build --preset full
```

The equivalent build-directory command is:

```bash
cmake --build build --target tests-full
```

Hidden `[.][gpu]` tests need a real Vulkan device (the VDPM GPU-front compute path, cross-checked
against the CPU authority on a headless surface-free device). They are excluded from CTest and CI (no
GPU/ICD on the runners); run them locally from `build/`:

```bash
cd build && ./test_fire_engine "[gpu]"
```

Optional local tooling targets:

```bash
cmake --build build --target run-clang-tidy   # if clang-tidy is installed
```

`run-clang-tidy` is split per source file, so Ninja can run it in parallel. Use
`cmake --build build --target run-clang-tidy --parallel <jobs>` or
`CMAKE_BUILD_PARALLEL_LEVEL` to cap local CPU/memory use.

CI (GitHub Actions, all `FIRE_ENGINE_WARNINGS_AS_ERRORS=ON`) runs five parallel jobs:

- **`clang-format`** and **`clang-tidy`** — platform-independent lint gates, run once (Ubuntu).
- **`build-test-linux`** — build + `tests-full` on Ubuntu (validates the **Linux/x86_64**
  determinism golden).
- **`build-test-macos`** — build + `tests-full` on macOS/arm64 (validates the **macOS/arm64**
  golden). Like Linux, it gets Vulkan + GLFW + `glslc` from vcpkg — the runner only adds `ninja`.
- **`release-contract`** — the only job that builds `Release`. Every other job builds `Dev`, so the
  suite's `#ifdef NDEBUG` bodies (what a writer returns once its assertion is compiled away)
  vanish; this one builds `test_fire_engine` from the `vcpkg-release` preset and runs
  `test_fire_engine "[release-contract]"`. Ubuntu only — that behaviour does not vary by platform —
  and deliberately not the whole Release suite, which would drag in the optimisation-sensitive
  physics goldens.

Each platform's `Determinism.GoldenHash` golden is now enforced by its own job — see
[`docs/collision.md`](docs/collision.md) and CLAUDE.md § Testing.

To reproduce the CI checks locally:

```bash
tools/ci/run-local-ci.sh all      # Linux, in Docker (Ubuntu 24.04)
tools/ci/run-local-macos.sh all   # macOS, native (no container)
```

The **Docker** runner copies the working tree into an Ubuntu 24.04 container, keeps Linux
build/vcpkg state in Docker volumes, and accepts `format`, `configure`, `build`, `tidy`, `test`,
`release-contract`, `all`, or `shell` to isolate a stage (`all` includes the Release contract on
Linux; the macOS runner has no such stage, by design). It defaults to `linux/amd64` to match GitHub Actions; set
`DOCKER_PLATFORM=linux/arm64` for a faster native Apple Silicon check. The **native macOS** runner
takes the same stages and runs them directly on your host toolchain (it installs nothing — Vulkan,
GLFW, and `glslc` all come from vcpkg, so it just needs your existing vcpkg + compiler + `ninja`).
Both share their stage bodies via `tools/ci/ci-stages.sh`.

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

Use `--maximized` (or `--maximised`) to ask GLFW to create the application window maximized.

Runtime diagnostics use `FE_LOG`. With no `FE_LOG`, warnings and errors are printed; info/debug
logs are quiet. Set a global level (`debug`, `info`, `warn`, `error`, `off`) or category-specific
levels (`category:level`) with comma-separated entries:

```bash
FE_LOG=debug ./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr
FE_LOG=warn,gltf:info,ragdoll:debug ./fireEngineApp CesiumMan/CesiumManRagdoll.gltf skybox.hdr -f
FE_LOG=render:debug ./fireEngineApp
```

Current categories are `app`, `general`, `gltf`, `physics`, `ragdoll`, and `render`.

`render:debug` also prints a periodic **shadow recording** line — per family, whether it was recorded
or skipped, its raster passes and its GPU milliseconds — which is how `--no-shadows` is checked: it
suppresses the *recording*, not only the sampling, so every family must read `skipped passes=0
0.000ms`. A frame that still rendered into maps nobody samples would look identical on screen.

## Dependencies

Managed via the vcpkg manifest (`vcpkg.json`); every version comes from the baseline pinned in
`vcpkg-configuration.json`, currently `ea1a7396` (Aug 2026) — Vulkan headers + loader **1.4.357.0**,
`glfw3 3.5.1`, `glslang 16.4.0`, `spirv-tools 1.4.357.0`, `imgui 1.92.8`, `shaderc 2026.2`,
`ktx 4.4.2`, `fastgltf 0.9.0`, `catch2 3.15.3`, `vulkan-memory-allocator 3.4.0`:

- `vulkan-headers` — Vulkan API headers (the Vulkan **loader** + `glfw3` arrive transitively, so
  both come from vcpkg — no system Vulkan SDK / GLFW needed to build)
- `vulkan-memory-allocator` — VMA GPU sub-allocator
- `shaderc` — provides the `glslc` GLSL→SPIR-V compiler as a vcpkg tool, so every platform compiles
  shaders with the same baseline-pinned compiler (no system `glslang-tools` / Vulkan-SDK glslc)
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

Also requires a C++23 toolchain, CMake, and Ninja. Building and the headless test suite need no
system Vulkan — the loader, headers, GLFW, and `glslc` all come from vcpkg. On **Linux** add the X11
development packages and autotools, which vcpkg needs to build GLFW's X11 dependency chain (some of
those ports use autotools rather than CMake and stop with a clear message if they are absent):

```bash
sudo apt install xorg-dev libxinerama-dev libxcursor-dev libglu1-mesa-dev pkg-config \
                 autoconf autoconf-archive automake libtool
```

macOS needs none of that — GLFW uses the Cocoa backend there. To actually *run* the app you
additionally need a Vulkan ICD at runtime: **MoltenVK** on macOS, a GPU driver on Linux.

The GPU must expose **Vulkan 1.4** (the renderer uses core 1.4 push descriptors); a device below
that is rejected at startup with the version named in the log. Instance layers and platform
extensions are resolved against what the loader actually offers, so one binary runs on a MoltenVK
Mac (which requires `VK_KHR_portability_subset`) and on a conformant Linux driver (which must not
be asked for it) with no build flags.

The **validation layer is optional at runtime**: a `Dev` build enables it when present, logs
`Vulkan validation enabled` and continues with a `NOT INSTALLED` warning when it isn't. It ships
with the [Vulkan SDK](https://vulkan.lunarg.com/) (on Linux, the tarball — `source setup-env.sh`),
not with the vcpkg loader. Pass **`--require-validation`** to refuse to start unless validation is
actually active — worth doing in any automated run, since a zero-VUID report from an unvalidated
build means only that nothing was checking.

## Assets

All glTF models are from the Khronos glTF Sample Models [repository](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main): AlphaBlendModeTest, AnimatedCube, AnimatedMorphCube, BoomBoxWithAxes, BoxAnimated, BrainStem, CesiumMan, DamagedHelmet, Fox, InterpolationTest, LightsPunctualLamp (exercises KHR_lights_punctual + KHR_materials_transmission), MetalRoughSpheres, MorphPrimitivesTest, OrientationTest, RecursiveSkeletons, RiggedSimple, TextureCoordinateTest, TextureLinearInterpolationTest, TextureSettingsTest, VertexColorTest.

Two asset sets are **generated, not downloaded** — regenerated at build time from Python under `assets/`, sharing the glTF machinery in `tools/assetgen/`: the physics demos above, and `assets/shadow_lod/` (`ShadowLodDemo.gltf`, the static shadow-LOD measurement baseline, plus `ShadowLodMotionDemo.gltf`, its animated stability loop). The shadow scenes carry their own floor, sun, spot and point lights, a shared dense caster at several scales and distances, a skinned limb, a morph target, an alpha-masked cutout and a double-sided sheet — cases the sample models don't provide together in one frame. Their generator validates the finished document structurally, so a later edit can't quietly drop an exposure.

HDR equirectangular skyboxes (`skybox.hdr`, `nightbox.hdr`) drive the IBL precompute.

Both CLI args are optional. With **no scene** the app renders only programmatically-added content (the `-p`/`-c`/`-k`/`-q` demo flags) — there is no fallback asset. With **no skybox** the scene is still lit by the default environment's IBL, but no sky is drawn behind it (a neutral background) and the IBL runs at a calmer level; pass `skybox.hdr` or `nightbox.hdr` to draw *and* light with that environment at full strength.

For manual visual sign-off, [`docs/acceptance-testing.md`](docs/acceptance-testing.md) is a per-asset runbook: a copy-paste command for every sample scene, physics demo, and generated feature (particles/cloth/character/query), each paired with its upstream Khronos source and reference image — the checklist to walk before a branch lands.
