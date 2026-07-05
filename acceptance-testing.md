# Fire Engine — Acceptance Testing

Manual, visual sign-off for the renderer + physics. This is the runbook to walk before a branch
lands, to catch feature regressions a headless smoke test can't see (e.g. the TransmissionTest
regression that only shows on multi-material scenes).

## How to run

- **Run every command from the `build/` directory** — assets are copied there flat, so scene paths
  have **no `assets/` prefix** (e.g. `DamagedHelmet/DamagedHelmet.gltf`).
- The app opens a window. Compare it against the reference image, then **close the window** (or
  `Ctrl-C` the terminal) to move on.
- **Validation layers are on** in the default (non-`NDEBUG`) build, so a clean render is also a
  **0-VUID** render. For a headless pass/fail, background the app and grep stderr (see `CLAUDE.md`).
- Default skybox is `skybox.hdr`; `nightbox.hdr` is also available.

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

### TransmissionTest — KHR_materials_transmission ⚠️ *known regression — scrutinise*
Multi-material transmission spheres. **Currently renders identically regardless of transmission
variant / opacity / texture** — this is the pre-existing regression the runbook exists to catch.
The spheres in the reference vary visibly across the grid; if ours all look the same, it's still broken.
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

## Engine-authored scenes

Original assets (no upstream source / image). Reference images are `placeholder.jpg` for now.

### CesiumManRagdoll — articulated ragdoll settling on a floor (needs `-f`)
Falls and settles into a plausible pose; without `-f` there is no floor and it falls through.
```bash
./fireEngineApp CesiumMan/CesiumManRagdoll.gltf skybox.hdr -f
```
![CesiumManRagdoll](placeholder.jpg)

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

These add procedurally-built content to the scene at startup; they carry no `.gltf`. Images are
left as `placeholder.jpg` for now. (Flags are position-independent; a scene arg is optional.)

### Particles — `-p` (GPU compute + instanced additive billboards + bloom)
```bash
./fireEngineApp skybox.hdr -p
```
![Particles](placeholder.jpg)

### Cloth — `-c` (GPU XPBD sheet drapes over a sphere onto the ground)
```bash
./fireEngineApp skybox.hdr -c
```
![Cloth](placeholder.jpg)

### Character controller — `-k` (kinematic capsule on an obstacle course: walk / step / climb / slide)
```bash
./fireEngineApp skybox.hdr -k
```
![Character](placeholder.jpg)

### Query probe — `-q` (ring of static bodies queried each frame; run with `--debug-physics` to see it)
```bash
./fireEngineApp skybox.hdr -q --debug-physics
```
![QueryProbe](placeholder.jpg)

### Floor — `-f` (adds a receiver-only ground plane; combine with any of the above)
```bash
./fireEngineApp skybox.hdr -f -p
```
![Floor](placeholder.jpg)
