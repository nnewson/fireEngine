# Fire Engine — Acceptance Testing

Manual, visual sign-off for the renderer + physics. This is the runbook to walk before a branch
lands, to catch feature regressions a headless smoke test can't see — the kind that only show on
specific multi-material scenes (e.g. the thin-walled TransmissionTest roughness regression this
runbook surfaced, since fixed).

## How to run

- **Run every command from the `build/` directory** — assets are copied there flat, so scene paths
  have **no `assets/` prefix** (e.g. `DamagedHelmet/DamagedHelmet.gltf`).
- The app opens a window. Compare it against the reference image, then **close the window** (or
  `Ctrl-C` the terminal) to move on.
- **Validation layers are on** in the default (non-`NDEBUG`) build, so a clean render is also a
  **0-VUID** render. For a headless pass/fail, background the app and grep stderr (see [`CLAUDE.md`](../CLAUDE.md)).
- **Both CLI args are optional.** With **no scene**, only programmatically-added content renders
  (the `-p`/`-c`/`-k`/`-q` demos) — there is no fallback asset. With **no skybox**, the scene is lit
  by the default environment's IBL but **no sky is drawn** (neutral background), at a calmer IBL
  level. Pass `skybox.hdr` (bright) or `nightbox.hdr` (dark) to draw + light with that environment.

### ⚠️ Verify the reference images

The reference-image links point at the Khronos **glTF-Sample-Models** repo and are **best-guess
URLs** (most models use `screenshot/screenshot.jpg`; some use a different name — `.gif`, `.png`,
`screenshot_large*.jpg`, etc., and a few newer models may only live in the successor
`glTF-Sample-Assets` repo). **Expect some to 404.** When one is wrong, replace it with the correct
filename or with `placeholder.jpg`. Engine-authored / physics / generated scenes use `placeholder.jpg`
by design (no upstream image).

---

## Khronos glTF 2.0 sample models

Source repo: <https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0>

### AlphaBlendModeTest — alpha OPAQUE / MASK / BLEND + double-sided
```bash
./fireEngineApp AlphaBlendModeTest/AlphaBlendModeTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/AlphaBlendModeTest)

![AlphaBlendModeTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/AlphaBlendModeTest/screenshot/screenshot_large.jpg)

### AnimatedCube — node rotation animation
```bash
./fireEngineApp AnimatedCube/AnimatedCube.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/AnimatedCube)

![AnimatedCube](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/AnimatedCube/screenshot/screenshot.gif)

### AnimatedMorphCube — morph-target animation
```bash
./fireEngineApp AnimatedMorphCube/AnimatedMorphCube.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/AnimatedMorphCube)

![AnimatedMorphCube](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/AnimatedMorphCube/screenshot/screenshot.gif)

### BoomBoxWithAxes — PBR metal/rough + orientation axes
```bash
./fireEngineApp BoomBoxWithAxes/BoomBoxWithAxes.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/BoomBoxWithAxes)

![BoomBoxWithAxes](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/BoomBoxWithAxes/screenshot/screenshot.jpg)

### BoxAnimated — hierarchical node animation
```bash
./fireEngineApp BoxAnimated/BoxAnimated.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/BoxAnimated)

![BoxAnimated](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/BoxAnimated/screenshot/screenshot.gif)

### BrainStem — skinned skeletal animation
```bash
./fireEngineApp BrainStem/BrainStem.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/BrainStem)

![BrainStem](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/BrainStem/screenshot/screenshot.gif)

### Cameras — embedded camera nodes
```bash
./fireEngineApp Cameras/Cameras.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Cameras)

![Cameras](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/Cameras/screenshot/screenshot.png)

### CesiumMan — skinned character (walk)
```bash
./fireEngineApp CesiumMan/CesiumMan.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/CesiumMan)

![CesiumMan](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/CesiumMan/screenshot/screenshot.gif)

### ClearCoatTest — KHR_materials_clearcoat
```bash
./fireEngineApp ClearCoatTest/ClearCoatTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/ClearCoatTest)

![ClearCoatTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/ClearCoatTest/screenshot/screenshot_large.jpg)

### DamagedHelmet — opaque PBR (normal / emissive / AO / metal-rough)
```bash
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/DamagedHelmet)

![DamagedHelmet](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/DamagedHelmet/screenshot/screenshot.png)

### EmissiveStrengthTest — KHR_materials_emissive_strength + bloom
```bash
./fireEngineApp EmissiveStrengthTest/EmissiveStrengthTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/EmissiveStrengthTest)

![EmissiveStrengthTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/EmissiveStrengthTest/screenshot/screenshot_large_bloom.jpg)

### Fox — skinned run cycle
```bash
./fireEngineApp Fox/Fox.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Fox)

![Fox](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/Fox/screenshot/screenshot.jpg)

### InterpolationTest — STEP / LINEAR / CUBICSPLINE animation
```bash
./fireEngineApp InterpolationTest/InterpolationTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/InterpolationTest)

![InterpolationTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/InterpolationTest/screenshot/screenshot.gif)

### LightsPunctualLamp — KHR_lights_punctual (point / spot / directional)
```bash
./fireEngineApp LightsPunctualLamp/LightsPunctualLamp.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/LightsPunctualLamp)

![LightsPunctualLamp](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/LightsPunctualLamp/screenshot/lights_on_off.gif)

### MetalRoughSpheres — metal × roughness grid
```bash
./fireEngineApp MetalRoughSpheres/MetalRoughSpheres.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/MetalRoughSpheres)

![MetalRoughSpheres](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/MetalRoughSpheres/screenshot/screenshot.png)

### MorphPrimitivesTest — morph targets across primitives
```bash
./fireEngineApp MorphPrimitivesTest/MorphPrimitivesTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/MorphPrimitivesTest)

![MorphPrimitivesTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/MorphPrimitivesTest/screenshot/screenshot.jpg)

### OrientationTest — node orientation / coordinate handedness
```bash
./fireEngineApp OrientationTest/OrientationTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/OrientationTest)

![OrientationTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/OrientationTest/screenshot/screenshot.png)

### RecursiveSkeletons — nested / recursive skins
```bash
./fireEngineApp RecursiveSkeletons/RecursiveSkeletons.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/RecursiveSkeletons)

![RecursiveSkeletons](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/RecursiveSkeletons/screenshot/screenshot.jpg)

### RiggedSimple — simple single skin
```bash
./fireEngineApp RiggedSimple/RiggedSimple.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/RiggedSimple)

![RiggedSimple](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/RiggedSimple/screenshot/screenshot.gif)

### StainedGlassLamp (KTX2) — KTX2/Basis textures + transmission + emissive
```bash
./fireEngineApp StainedGlassLampKTX/StainedGlassLamp.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/StainedGlassLamp)

![StainedGlassLamp](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/StainedGlassLamp/screenshot/screenshot_large.jpg)

### TextureCoordinateTest — UV coordinate mapping
```bash
./fireEngineApp TextureCoordinateTest/TextureCoordinateTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/TextureCoordinateTest)

![TextureCoordinateTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/TextureCoordinateTest/screenshot/screenshot.png)

### TextureLinearInterpolationTest — sRGB vs linear texture filtering
```bash
./fireEngineApp TextureLinearInterpolationTest/TextureLinearInterpolationTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/TextureLinearInterpolationTest)

![TextureLinearInterpolationTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/TextureLinearInterpolationTest/screenshot/screenshot.png)

### TextureSettingsTest — sampler wrap / filter modes
```bash
./fireEngineApp TextureSettingsTest/TextureSettingsTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/TextureSettingsTest)

![TextureSettingsTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/TextureSettingsTest/screenshot/screenshot.png)

### TransmissionRoughnessTest — transmission × roughness
```bash
./fireEngineApp TransmissionRoughnessTest/TransmissionRoughnessTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/TransmissionRoughnessTest)

![TransmissionRoughnessTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/TransmissionRoughnessTest/screenshot/screenshot-large.png)

### TransmissionTest — KHR_materials_transmission
Multi-material transmission grid. The **roughness column** should vary visibly frosted → clear, and
the **rows** should vary by transmission factor / opacity / texture — match the reference grid.
(This scene surfaced a regression where all thin-walled spheres rendered identically regardless of
`roughnessFactor`; fixed 2026-07-06 by making thin-walled transmission sample the roughness-blurred
scene per spec.)
```bash
./fireEngineApp TransmissionTest/TransmissionTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/TransmissionTest)

![TransmissionTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/TransmissionTest/screenshot/screenshot_large.png)

### VertexColorTest — per-vertex COLOR_0
```bash
./fireEngineApp VertexColorTest/VertexColorTest.gltf skybox.hdr
```
[Source](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/VertexColorTest)

![VertexColorTest](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/VertexColorTest/screenshot/screenshot.png)

---

## Environment maps (skyboxes)

The two shipped HDR environments, each verified against a known subject (`DamagedHelmet`) so the sky,
its reflections, and the IBL tint are all visible. Passing the skybox both draws it as the background
and drives the IBL. (Omitting it renders no background + a calmer default IBL — see the note at the
top.) Images are `placeholder.jpg` for now.

### skybox.hdr — bright daytime environment
```bash
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr
```
![skybox.hdr](placeholder.jpg)

### nightbox.hdr — dark night environment
```bash
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf nightbox.hdr
```
![nightbox.hdr](placeholder.jpg)

---

## Engine-authored scenes

Original assets (no upstream source / image). Reference images are `placeholder.jpg` for now.

### CesiumManRagdoll — articulated ragdoll settling on a floor (needs `-f`)
Falls and settles into a plausible pose; without `-f` there is no floor and it falls through. The
knees and elbows are authored as true 1-DOF hinges (`extras.Ragdoll.Joints`), so the limbs fold one
way about their hinge axis rather than swinging in a cone.
```bash
./fireEngineApp CesiumMan/CesiumManRagdoll.gltf skybox.hdr -f
```
![CesiumManRagdoll](placeholder.jpg)

Add `--debug-joints` (or overlay **View → Joints**) to replace the mesh with the per-joint RGB axis
gizmo + "index: bone-name" labels — used to author/verify the hinge axes. The labels should track the
joints as the camera moves, and turning the view off should restore the mesh with no leftover gizmo.
```bash
./fireEngineApp CesiumMan/CesiumManRagdoll.gltf skybox.hdr -f --debug-joints
```
![CesiumManRagdoll joints](placeholder.jpg)

### ClothBanner — authored `extras.Cloth` (pinned banner)
Cloth hangs from its pinned edge and sways.
```bash
./fireEngineApp ClothBanner/ClothBanner.gltf skybox.hdr
```
![ClothBanner](placeholder.jpg)

### ClothSheet — authored `extras.Cloth` (free sheet)
```bash
./fireEngineApp ClothSheet/ClothSheet.gltf skybox.hdr
```
![ClothSheet](placeholder.jpg)

### Pong — the bundled mini-game
```bash
./fireEngineApp Pong/Pong.gltf skybox.hdr
```
![Pong](placeholder.jpg)

---

## Physics demos

Generated by `assets/physics_demos/generate.py`, each mirrored by a headless replay test in
`tests/physics/test_demos.cpp`. They carry their own floor — no `-f` needed. Enable
`--debug-physics` to overlay collider / contact wireframes. Images are `placeholder.jpg` for now.

### CompoundDemo — compound (multi-shape) collider settles upright
```bash
./fireEngineApp physics_demos/CompoundDemo.gltf skybox.hdr
```
![CompoundDemo](placeholder.jpg)

### ConvexHullDemo — GJK/EPA convex hull rests on the floor
```bash
./fireEngineApp physics_demos/ConvexHullDemo.gltf skybox.hdr
```
![ConvexHullDemo](placeholder.jpg)

### FallRestDemo — a body falls and comes to rest
```bash
./fireEngineApp physics_demos/FallRestDemo.gltf skybox.hdr
```
![FallRestDemo](placeholder.jpg)

### FrictionRampDemo — boxes on ramps of increasing friction slide differently
```bash
./fireEngineApp physics_demos/FrictionRampDemo.gltf skybox.hdr
```
![FrictionRampDemo](placeholder.jpg)

### RestitutionDemo — bouncing bodies (restitution)
```bash
./fireEngineApp physics_demos/RestitutionDemo.gltf skybox.hdr
```
![RestitutionDemo](placeholder.jpg)

### SingleJointRagdollDemo — single-joint ragdoll settles
```bash
./fireEngineApp physics_demos/SingleJointRagdollDemo.gltf skybox.hdr
```
![SingleJointRagdollDemo](placeholder.jpg)

### SleepDemo — bodies settle, then sleep (stop integrating)
```bash
./fireEngineApp physics_demos/SleepDemo.gltf skybox.hdr
```
![SleepDemo](placeholder.jpg)

### StackDemo — a box stack stays standing (solver stability)
```bash
./fireEngineApp physics_demos/StackDemo.gltf skybox.hdr
```
![StackDemo](placeholder.jpg)

### StaticMeshDemo — bodies rest on a static triangle mesh
```bash
./fireEngineApp physics_demos/StaticMeshDemo.gltf skybox.hdr
```
![StaticMeshDemo](placeholder.jpg)

### ToppleDemo — a box topples onto a face and rests
```bash
./fireEngineApp physics_demos/ToppleDemo.gltf skybox.hdr
```
![ToppleDemo](placeholder.jpg)

### TwoJointRagdollDemo — two-joint ragdoll settles
```bash
./fireEngineApp physics_demos/TwoJointRagdollDemo.gltf skybox.hdr
```
![TwoJointRagdollDemo](placeholder.jpg)

---

## Generated feature demos (command-line flags)

These add procedurally-built content at startup; they carry no `.gltf`. The commands pass **no scene
and no skybox** — so only the generated content renders (no fallback asset), lit by the default
environment's IBL against a neutral background (no sky drawn). Add `skybox.hdr` to draw a sky behind
them. Flags are position-independent. Images are `placeholder.jpg` for now.

### Particles — `-p` (GPU compute + instanced additive billboards + bloom)
```bash
./fireEngineApp -p
```
![Particles](placeholder.jpg)

### Cloth — `-c` (GPU XPBD sheet drapes over a sphere onto the ground)
```bash
./fireEngineApp -c
```
![Cloth](placeholder.jpg)

### Character controller — `-k` (kinematic capsule on an obstacle course: walk / step / climb / slide)
```bash
./fireEngineApp -k
```
![Character](placeholder.jpg)

### Query probe — `-q` (ring of static bodies queried each frame; run with `--debug-physics` to see it)
```bash
./fireEngineApp -q --debug-physics
```
![QueryProbe](placeholder.jpg)

### Floor — `-f` (adds a receiver-only ground plane; combine with any of the above)
```bash
./fireEngineApp -f -p
```
![Floor](placeholder.jpg)

---

## Feature checks (overlay-driven)

### Mesh LOD — discrete / VIPM / VDPM
```bash
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr --overlay
```
In the **"Mesh LOD"** overlay panel, with the helmet filling most of the screen and the pixel-error
budget near default:
- **Discrete** — backing the camera off (or raising the budget) should **step** the mesh coarser; the
  "LOD tint" debug view (View dropdown → LOD tint) colours by level (green → yellow → red) and you
  should see a hard swap at each step.
- **Continuous (VIPM)** — the same transitions should now **dissolve** instead of popping; the texture
  must not shear at the mirrored front seam as a level changes.
- **View-dependent (VDPM)** — at a matched budget the mesh should look **the same as Discrete's finest
  visible detail but with far fewer triangles** (watch "Triangles drawn"). Flipping between VDPM and
  Discrete/VIPM should show **no background leaking through** (no missing front-facing triangles), no
  silhouette holes, and no flicker with the camera still. This is the case the VDPM foldover +
  coverage repair passes exist to guarantee — see [`lod.md`](lod.md).

#### VDPM indirect draw (Stage A of the GPU-driven front)
VDPM draws are issued with `drawIndexedIndirect`. Launch straight into it (no overlay toggle needed):
```bash
./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr --lod-mode view-dependent
./fireEngineApp TransmissionTest/TransmissionTest.gltf skybox.hdr --lod-mode view-dependent
```
The image must be **identical** to the overlay-selected VDPM (the indirect command carries the same
index count the direct path used), and the validation layers must stay **silent** — the smoke check
`grep -icE 'VUID|validation error'` on the backgrounded run's stderr should print `0`. DamagedHelmet
exercises the forward + depth-prepass sites; TransmissionTest's dense transmissive meshes exercise the
transmission site. This is the MoltenVK `drawIndexedIndirect` de-risk.
