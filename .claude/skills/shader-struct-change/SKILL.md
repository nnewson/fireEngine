---
name: shader-struct-change
description: >
  Safely change a GPU struct shared with a shader (UBO / SSBO / push constant) in fireEngine — the
  cross-file std140 / std430 dance. Use when adding or changing a field in render/ubo.hpp, a shader
  block, a descriptor binding, or a material field, or when the user says "add a UBO field", "shader
  struct", "push constant", "std140", "layout mismatch".
---

# Shader-visible struct change

Every CPU struct shared with a shader lives in `render/ubo.hpp` with `alignas` + `static_assert`s
pinning its std140/std430 offsets and size. A host↔GPU layout mismatch is **silent** — those
static_asserts and `test_ubo.cpp` are the only guard. Touch both sides together.

## Steps

1. **CPU struct** — edit `include/fire_engine/render/ubo.hpp`. Keep field order / `alignas` / padding
   matching std140 (UBO) or std430 (SSBO); a `vec3` is 16-byte aligned. Update the
   `static_assert(offsetof(...))` and `sizeof` lines.
2. **GLSL block** — mirror the exact field order + trailing padding in the shader's
   `layout(std140/std430)` block.
3. **Binding (only if new)** — add the enumerator to `render/descriptor_bindings.hpp`
   (`ForwardBinding` / `ForwardGlobalBinding` / `ShadowBinding` / `SkyboxBinding` /
   `PostProcessBinding`) and the matching `layout(set = S, binding = N)` in GLSL. Set 0 is pushed
   per-draw (`pushForwardObjectDescriptors`); set 1 is written in `Descriptors::createGlobalDescriptors`
   + `updateGlobalDescriptors`; set 2 is bindless (owned by `Resources`).
4. **Mapped writes** — go through `graphics/mapped_buffer.hpp` `writeMapped` (a bounded span), never a
   raw `void*`, and always the `currentFrame` slot.
5. **Tests** — extend `tests/render/test_ubo.cpp` (sizes/offsets); for a binding change also
   `tests/render/test_pipeline_config.cpp`.
6. **clang-format** the touched files.

## Material-field variant
A new material field: update `MaterialUBO` / `MaterialData` (C++ + shader) in lockstep, pack it in
`toMaterialUBO` (`src/graphics/material_binding.cpp`; `value_or({})` for optional blocks), load it in
`GltfLoader::loadMaterial`, add a `MaterialTextureSlot` + shader `SLOT_*` if it's a texture, and extend
`tests/graphics/test_material.cpp`. No descriptor-binding change (textures are bindless).

## Verify
Build (the static_asserts fire on mismatch) → `(cd build && ./test_fire_engine "[UBO]")` →
**render-smoke** skill (0 VUID). See CLAUDE.md § Architecture "GPU data-layout discipline".
