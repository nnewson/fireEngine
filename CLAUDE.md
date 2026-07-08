# CLAUDE.md

Vulkan-based 3D renderer in C++23. Loads glTF 2.0 with PBR materials, IBL, CSM shadows, bloom. macOS via MoltenVK.

## Ethos

**This is an experimental project — it will never "ship".** So always take the *correct* / most principled
approach, not the pragmatic-but-compromised one, unless you have a genuine concern that the "correct" choice
may itself be wrong (surface that concern). Don't cheese a result, tune magic numbers to squeak a test past,
or ship a half-fix to move on — if the right fix is a bigger piece of work, do that. Assets are ours to edit
(reposition, retag, fix a skeleton, regenerate) when that's the correct way to exercise a feature.

## Workflow

**Every new plan or piece of work gets its own git branch, created off `main` before any code is written** — one branch per roadmap item / feature / fix, so each can be reviewed and committed independently. **Branch off *local* `main`** (`git switch -c <name> main`), **not off `origin/main`**: branching from the remote-tracking ref sets the branch's upstream to `origin/main`, which suppresses `push.autoSetupRemote` and makes `git push` fail with an "upstream branch name doesn't match" error. Off local `main` the branch starts with no upstream, so the first push auto-creates a correctly-named remote branch. Don't work on `main` directly, and don't commit or push unless explicitly asked; just leave the work on its branch. Keep the docs (§ Documentation) in sync **in the same branch** — treat them as part of the change, not an afterthought.

## Documentation

The repo carries several Markdown docs. The maintainer docs live in **`docs/`**; `README.md` and
this file stay at the repo root. Each has a distinct audience; **when you touch
code, update the docs that describe it in the same change**:

- **[`README.md`](README.md)** (root) — outward-facing: feature list, how to build/run, the local Docker CI, dependencies, assets, the [`docs/acceptance-testing.md`](docs/acceptance-testing.md) pointer. Update when a user-visible capability, CLI flag, dependency, or build step changes.
- **[`docs/onboarding.md`](docs/onboarding.md)** — for engineers who will **maintain/extend** the engine: the mental model, a recommended reading route (smallest types → highest-risk systems), conventions, **cross-file invariants**, common "how do I…" tasks, and sharp edges. Update when you add a subsystem, a cross-file invariant, or a common task.
- **[`docs/review-order.md`](docs/review-order.md)** — the exhaustive, **tiered file-by-file reading order** (headers before `.cpp`) with per-file attention points. Update when you add/rename/split a source file or change a file's responsibility.
- **[`docs/collision.md`](docs/collision.md)** — the **physics & collision** track: runtime model, glTF authoring (`extras.*`), per-system detail. The authority for physics behaviour + authoring.
- **[`docs/lod.md`](docs/lod.md)** — the **mesh level-of-detail / simplifier** system: how it works and the design decisions, for a maintainer. The authority for LOD behaviour + tuning.
- **[`docs/roadmap.md`](docs/roadmap.md)** — the **rendering-spine roadmap**: what's done, an ordered to-do (could/should/maybe), and per-item design notes. Update status when an item lands. (The physics track's detail lives in [`docs/collision.md`](docs/collision.md).)
- **[`docs/acceptance-testing.md`](docs/acceptance-testing.md)** — the **manual visual sign-off runbook** (per-asset commands + reference images). Update when assets, scenes, or flags change.
- **[`CLAUDE.md`](CLAUDE.md)** / **[`AGENTS.md`](AGENTS.md)** (root) — agent instructions (this file; `AGENTS.md` is a symlink). Update when a convention, invariant, or workflow rule changes.

Scope by change type: a **rendering** feature → README + [`docs/onboarding.md`](docs/onboarding.md) + [`docs/review-order.md`](docs/review-order.md) + roadmap; a **physics** feature → those + [`docs/collision.md`](docs/collision.md); an **LOD** change → [`docs/lod.md`](docs/lod.md) (+ README/onboarding if user-visible); a **new cross-file invariant** → [`docs/onboarding.md`](docs/onboarding.md) § Cross-File Invariants + this file's § Architecture.

## Build

```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
ctest --preset fast
cmake --build --preset full                 # includes [slow] settle/soak tests
./build/fireEngineApp [scene.gltf] [skybox.hdr]
```

The `vcpkg` preset selects the `Dev` build type (`-O2 -g`, no `NDEBUG`) and exports
`build/compile_commands.json` for `clangd`. Shaders compile via `glslc` at build time.
Assets copied via `cmake/copy_assets.cmake`.

**Runtime layout (run from `build/`):**
- Assets are copied **flat** into the build dir, so scene paths are relative to `build/` with **no `assets/` prefix** — e.g. `./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr`. Skyboxes: `skybox.hdr` (default), `nightbox.hdr`. Useful test scenes: `AlphaBlendModeTest/AlphaBlendModeTest.gltf` (blend + double-sided), `TransmissionTest/TransmissionTest.gltf` (transmission), `DamagedHelmet/DamagedHelmet.gltf` (opaque PBR).
- **Validation layers are on in non-`NDEBUG` builds** (`Device::enableValidation`), so a quick render smoke-test catches Vulkan misuse (dynamic-state / barrier / descriptor errors). The app opens a window; for a headless-ish check, background it and grep stderr:
  ```bash
  cd build && (./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr >/tmp/fe.log 2>&1 & p=$!; sleep 6; kill $p; wait $p 2>/dev/null)
  grep -icE 'VUID|validation error' /tmp/fe.log   # expect 0; SIGTERM exit 143 = ran fine
  ```
  Runtime diagnostics go through `FE_LOG` (`debug`, `info`, `warn`, `error`, `off`; categories:
  `app`, `general`, `gltf`, `physics`, `ragdoll`, `render`). Vulkan instance/device extension
  lists are quiet by default; enable them with `FE_LOG=render:debug`. Ragdoll settle diagnostics
  use `FE_LOG=ragdoll:debug`.

## Dependencies

vcpkg manifest (`vcpkg.json`; versions from the default-registry baseline in `vcpkg-configuration.json`): `vulkan-headers`, `vulkan-memory-allocator`, `shaderc` (provides the `glslc` tool under `<vcpkg-installed>/<triplet>/tools/shaderc`), `catch2`, `stb`, `fastgltf`, `ktx`, `imgui[glfw-binding,vulkan-binding]`. The Vulkan **loader** and `glfw3` arrive transitively, so Vulkan + GLFW + the shader compiler all build from vcpkg — **no system Vulkan SDK / GLFW / glslang-tools**. System requirement is just a C++23 toolchain + Ninja; a Vulkan ICD (MoltenVK on macOS) is needed only at *runtime* to render, not to build or run the headless tests. `fireengine` links Vulkan/GLFW directly; `cmake/fireengine_imgui.cmake` wraps vcpkg's ImGui archive without its transitive Vulkan/GLFW link interface to avoid duplicate static-library warnings.

**Toolchain: Apple Clang** (`/usr/bin/clang++`, set in `CMakePresets.json`). The vcpkg toolchain inherits this compiler, so all ports build from the manifest. (The project previously used Homebrew g++-15, which can't parse the Apple SDK framework headers — that broke the vcpkg builds of gtest/glfw3/imgui and forced classic-mode global installs + a vendored imgui backend; switching to Clang removed all of that.) Note `clang++` on `PATH` may be Homebrew clang — the preset pins `/usr/bin/clang++` for Apple's.

## Layout

```
include/fire_engine/  animation/ collision/ core/ graphics/ input/
                      math/ physics/ platform/ render/ scene/
src/                  mirrors include/
shaders/              GLSL → SPIR-V
tests/                Catch2, mirrors src/
tests/support/        Shared Catch2 helpers and compile-time test traits
assets/               glTF samples + HDR skyboxes
```

## Architecture

- **graphics/** — backend-agnostic. No Vulkan headers. Communicates via opaque handles (`gpu_handle.hpp`) and `DrawCommand` structs.
- **render/** — all Vulkan code. `Resources` owns GPU resources, hands out handles, and centralizes host-visible buffer creation plus common 2D render-target setup. `PipelineConfig` factories use shared helpers for fullscreen/fragment-only pass boilerplate; keep returned config values covered by `tests/render/test_pipeline_config.cpp`. `Renderer` orchestrates passes.
- **scene/** — scenegraph. `Node` holds a `Components` variant (Empty/Animator/Camera/Mesh/Light). `SceneGraph::update()` propagates `InputState`; `gatherLights()` resolves world-space lights.
- **render/constants.hpp** — single source of truth for scalar render tunables (shadow biases, IBL strengths, shadow/IBL extents, camera FOV, bloom config). GPU data-layout limits (frames-in-flight, joint/morph/light counts, shadow caster caps + matrix layout) live in **graphics/gpu_limits.hpp** so the Vulkan-free graphics layer can size its arrays; constants.hpp includes it, so all constants stay reachable through one include.
- **Layering guard** — `graphics/` and `scene/` headers must not include `render/`, and `render/` headers must not include `scene/` (`render`↔`scene` meet only through the Vulkan-free `graphics/renderable_scene.hpp` seam). Enforced by the `layering_guards` CTest case (`cmake/check_layering.cmake`). Graphics `.cpp` files may include `render/resources.hpp` — that's the documented GPU-allocation bridge.
- **GPU resource model** — three orthogonal layers; never conflate them. **(1) Identity:** opaque handles (`graphics/gpu_handle.hpp`) into `Resources`' tables — an index packed with a generation so a stale handle to a reused slot is detectably invalid. **(2) Lifetime:** off-the-shelf `vk::raii` owns every Vulkan object whose lifetime is *independent of a device-memory allocation* (image views, samplers, pipelines, descriptor layouts, command pools). `vk::raii` is a lifetime tool, not a memory allocator — it imposes nothing on allocation strategy. **(3) Memory / sub-allocation:** VMA is the arena, used the idiomatic way (`vmaCreateBuffer`/`vmaCreateImage`) so it owns the resource + its sub-allocation and frees them together. Because `vmaDestroy*` couples the two, buffers and images are the *one* exception to layer 2: their `VkBuffer`/`VkImage` lifetime lives inside a small custom move-only VMA RAII wrapper (`UniqueVmaBuffer`/`UniqueVmaImage`) — still RAII, just VMA-backed. This is **not** a migration away from `vk::raii`; revisit layer 2 only if runtime streaming (mid-frame create/destroy needing a deletion queue) ever arrives.
- **GPU data-layout discipline** — every CPU struct shared with a shader (UBO/SSBO) lives in `render/ubo.hpp` with `alignas` + `static_assert`s pinning its std140/std430 offsets and size. Preserve this: when you change a shader-visible struct, update both sides and keep the static_asserts — they are the only thing catching a silent host↔GPU layout mismatch. Mapped host-visible writes go through `graphics/mapped_buffer.hpp` `writeMapped` (a bounds-checked `std::span<std::byte>`), never a raw `void*`.

## Code Style

**Always run `clang-format -i <file>` on any C++ file (`.hpp`/`.cpp`) after editing it.** The codebase is already formatted; keep it that way per-edit.

`.clang-format` enforces: Allman braces, 4-space indent, 100-col, left-aligned pointers, no single-line functions, ctor initializers each on own line.

`.clang-tidy` enables the first-pass bugprone/performance/modernize/readability checks for
engine `src/` and `include/fire_engine/`. If `clang-tidy` is installed, run
`cmake --build build --target run-clang-tidy`.

- C++23
- `constexpr` where possible (math types especially)
- `[[nodiscard]]` on getters/pure functions
- `noexcept` where it can't throw
- Private members trailing underscore: `x_`
- Getter/setter share name: `float x() const` / `void x(float)`
- Static factories for file loading: `Class::load_from_file(path)`
- Compound assignment as primitives, binary operators delegate
- Explicit rule-of-five (defaulted or deleted)
- Declarations from a `.hpp` go in the matching `.cpp`, never an unrelated one
- Runtime diagnostics go through `include/fire_engine/core/log.hpp` (`log::debug/info/warn/error`
  with categories); don't add direct `std::cout`/`std::cerr`/`std::clog`/`std::print` or
  printf-family diagnostics outside the logger/tests, and don't add one-off debug env vars.

Reference class: `include/fire_engine/graphics/image.hpp`.

## Testing

Catch2 (v3, `Catch2::Catch2WithMain`). Single binary `test_fire_engine`. CTest registers
`test_fire_engine` as the fast `~[slow]` entry, so plain `ctest` and `ctest --preset fast`
stay fast. The `tests-full` target runs the all-tags Catch2 binary plus the graphics-layer
include guard; from the source root use `cmake --build --preset full`. Test files mirror
source paths. Shared helpers/traits live in `tests/support/`. Test assets in `tests/assets/`
→ copied to `build/test_assets/`.

Graphics-layer tests run without a GPU (opaque handles).

**CI has four parallel jobs** (GitHub Actions, all `FIRE_ENGINE_WARNINGS_AS_ERRORS=ON`): `clang-format`
+ `clang-tidy` (platform-independent lint, run once on Ubuntu) and `build-test-linux` +
`build-test-macos` (build + `tests-full` on Ubuntu and macOS/arm64). Each build job validates *its*
platform's determinism golden. The build/test/lint stage bodies are shared across the Docker replica
and the native macOS replica via `tools/ci/ci-stages.sh` — edit stages there, not in each script.

**Local CI parity.** `tools/ci/run-local-ci.sh [format|configure|build|tidy|test|all|shell]`
reproduces the **Linux** checks (Ubuntu 24.04) in Docker; `tools/ci/run-local-macos.sh
[format|configure|build|tidy|test|all]` runs the same stages **natively on macOS** (no container —
uses your existing vcpkg + C++ toolchain, installs nothing; Vulkan/GLFW/glslc all come from vcpkg). The Docker runner copies the
working tree into volumes (host artifacts untouched), defaults to `linux/amd64` to match CI
(`DOCKER_PLATFORM=linux/arm64` is faster but off-platform). Run the relevant one before committing
anything that could trip the stricter warnings / clang-tidy / format gate.

**GoldenHash is platform-specific — re-baseline BOTH on a solver change.** `Determinism.GoldenHash`
(`tests/physics/test_physics_determinism.cpp`) compares the physics end-state hash (raw float bits) to
a recorded golden. macOS/arm64 and Linux/x86_64 diverge by a few last-bit contact-solver ops, so each
platform records its **own** golden (`#if defined(__linux__) && defined(__x86_64__)`). **Both are now
CI-enforced** (the Linux and macOS build jobs each run `tests-full`), so an *intended* change to solver
/ step math must update **both**: the macOS/arm64 golden from your local run, **and** the Linux/x86_64
golden read off `tools/ci/run-local-ci.sh test`. Never edit a golden to silence an *unexplained* hash
move — that's a determinism regression to investigate, not to rebaseline.

## Assets

glTF: Khronos sample models in `assets/`. HDR: `skybox.hdr` (default), `nightbox.hdr`. Fallback `default.png` for materials lacking base-colour.

CLI: `./fireEngineApp <scene.gltf> <skybox.hdr>` — both optional.
