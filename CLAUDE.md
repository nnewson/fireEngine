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
- **[`docs/roadmap.md`](docs/roadmap.md)** — the **index of open work only**: the current arcs (each pointing at the review/plan doc that owns its detail) plus the trigger-based parked items. It carries **no record of landed work** — when an item lands, delete it here and make sure its rationale sits in the authority doc ([`docs/lod.md`](docs/lod.md), [`docs/collision.md`](docs/collision.md), [`docs/onboarding.md`](docs/onboarding.md)…).
- **[`docs/codereview.md`](docs/codereview.md)** — the rolling **tiered static code review** (follows the [`docs/review-order.md`](docs/review-order.md) tiers). Findings are open until tracked as landed in the arc that clears them; add a tier when one is reviewed.
- **[`docs/architecturalreview.md`](docs/architecturalreview.md)** — the dated **architectural review** (rendering, shadows/AA, physics, simplifier/VDPM). Its §6 priority table is the status of record for its own findings — tick items there as they land.
- **[`docs/shadowplans.md`](docs/shadowplans.md)** — the **shadow-LOD improvement plan** (SH-01…SH-09): principles, per-item scope, verification gates. The authority for shadow-LOD selection design.
- **[`docs/acceptance-testing.md`](docs/acceptance-testing.md)** — the **manual visual sign-off runbook** (per-asset commands + reference images). Update when assets, scenes, or flags change.
- **[`CLAUDE.md`](CLAUDE.md)** / **[`AGENTS.md`](AGENTS.md)** (root) — agent instructions (this file; `AGENTS.md` is a symlink). Update when a convention, invariant, or workflow rule changes.

Scope by change type: a **rendering** feature → README + [`docs/onboarding.md`](docs/onboarding.md) + [`docs/review-order.md`](docs/review-order.md) + roadmap; a **physics** feature → those + [`docs/collision.md`](docs/collision.md); an **LOD** change → [`docs/lod.md`](docs/lod.md) (+ README/onboarding if user-visible); a **shadow-LOD / shadow-selection** change → [`docs/shadowplans.md`](docs/shadowplans.md) + [`docs/lod.md`](docs/lod.md); a **new cross-file invariant** → [`docs/onboarding.md`](docs/onboarding.md) § Cross-File Invariants + this file's § Architecture.

## Build

```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
ctest --preset fast
cmake --build --preset full                 # includes [slow] settle/soak tests
cd build && ./fireEngineApp [scene.gltf] [skybox.hdr]   # MUST run from build/ (see Runtime layout)
```

The `vcpkg` preset selects the `Dev` build type (`-O2 -g`, no `NDEBUG`) and exports
`build/compile_commands.json` for `clangd`. Shaders compile via `glslc` at build time.
Assets copied via `cmake/copy_assets.cmake`.

**Runtime layout (run from `build/`):**
- Assets are copied **flat** into the build dir, so scene paths are relative to `build/` with **no `assets/` prefix** — e.g. `./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr`. Skyboxes: `skybox.hdr` (default), `nightbox.hdr`. Useful test scenes: `AlphaBlendModeTest/AlphaBlendModeTest.gltf` (blend + double-sided), `TransmissionTest/TransmissionTest.gltf` (transmission), `DamagedHelmet/DamagedHelmet.gltf` (opaque PBR).
- **Validation layers are on in non-`NDEBUG` builds** (`Device::enableValidation`), so a quick render smoke-test catches Vulkan misuse (dynamic-state / barrier / descriptor errors). The app opens a window; for a headless-ish check, background it and grep stderr:
  ```bash
  cd build
  FE_LOG=render:info ./fireEngineApp --require-validation DamagedHelmet/DamagedHelmet.gltf skybox.hdr >/tmp/fe.log 2>&1 & p=$!
  sleep 6; alive=0; kill -0 $p 2>/dev/null && { alive=1; kill $p; }; rc=0; wait $p || rc=$?
  vuid=$(grep -icE 'VUID|validation error' /tmp/fe.log || true)
  active=$(grep -c 'Vulkan validation enabled' /tmp/fe.log || true)
  test $alive -eq 1 && test $rc -eq 143 && test $vuid -eq 0 && test $active -eq 1 \
    && echo PASS || { echo "FAIL (alive=$alive rc=$rc vuid=$vuid validation=$active)"; false; }
  ```
  All four conditions are the pass. **`alive` + status 143** prove the app survived to *our*
  SIGTERM (an early exit — no suitable GPU, a startup throw — otherwise yields a clean-looking log);
  **0 VUIDs** proves nothing went wrong; **the token** proves something was checking. The layer is
  enable-iff-available (it ships with the Vulkan SDK, not vcpkg's loader), so without
  `--require-validation` a machine lacking it — or an `NDEBUG` build — runs unvalidated and reports
  zero VUIDs *vacuously*.
  Runtime diagnostics go through `FE_LOG` (`debug`, `info`, `warn`, `error`, `off`; categories:
  `app`, `general`, `gltf`, `physics`, `ragdoll`, `render`). Vulkan instance/device extension
  lists are quiet by default; enable them with `FE_LOG=render:debug`. Ragdoll settle diagnostics
  use `FE_LOG=ragdoll:debug`.

## Dependencies

vcpkg manifest (`vcpkg.json`; versions from the default-registry baseline in `vcpkg-configuration.json`): `vulkan-headers`, `vulkan-memory-allocator`, `shaderc` (provides the `glslc` tool under `<vcpkg-installed>/<triplet>/tools/shaderc`), `catch2`, `stb`, `fastgltf`, `ktx`, `imgui[glfw-binding,vulkan-binding]`. The Vulkan **loader** and `glfw3` arrive transitively, so Vulkan + GLFW + the shader compiler all build from vcpkg — **no system Vulkan SDK / GLFW / glslang-tools**. System requirement is a C++23 toolchain + Ninja on macOS; **on Linux, add X11 dev packages and autotools** (`xorg-dev`, `libxinerama-dev`, `libxcursor-dev`, `libglu1-mesa-dev`, `autoconf`, `autoconf-archive`, `automake`, `libtool`) — glfw3's X11 backend pulls ports that vcpkg builds with autotools rather than CMake, and they fail to configure without them. A Vulkan ICD (MoltenVK on macOS) is needed only at *runtime* to render, not to build or run the headless tests. `fireengine` links Vulkan/GLFW directly; `cmake/fireengine_imgui.cmake` wraps vcpkg's ImGui archive without its transitive Vulkan/GLFW link interface to avoid duplicate static-library warnings.

**Pinned headers must beat `/usr/local/include`, and `-isystem` cannot do it.** Clang searches
`/usr/local/include` ahead of every `-isystem` path, which is where CMake puts an imported target's
includes. Install a Vulkan SDK there and our own translation units compile against *its* vulkan-hpp
while anything resolved relative to a vcpkg header (`vulkan_raii.hpp` including its sibling
`vulkan.hpp`) still gets vcpkg's — two vulkan-hpp versions in one binary, and the first RAII call
aborts on `m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION` with no bad line of C++ behind
it. This actually happened when SDK 1.4.357 landed beside the pinned 1.4.335. The top-level
`CMakeLists.txt` therefore sets `CMAKE_NO_SYSTEM_FROM_IMPORTED` and adds the vcpkg include dir with
`include_directories(BEFORE)`, so dependency headers arrive as `-I` and the manifest's versions win.
The cost is that third-party headers are no longer warning-suppressed: silence a finding **at the
include site** with a narrow `#pragma GCC diagnostic ignored` and a reason (see
`src/graphics/frame_capture.cpp`), never by weakening a flag for our own code.

**Upgrade the vcpkg baseline after each major item lands**, not mid-arc. The baseline in
`vcpkg-configuration.json` pins every dependency version, so it only moves when someone moves it —
and a stale pin quietly drifts from the SDK on the machine (it had once sat on a Feb 2026 commit
carrying `vulkan-headers 1.4.335.0` while the installed SDK reached 1.4.357). Bumping it is its own
branch with its own verification: full rebuild, `tests-full` on both platforms, and the render smoke,
since a loader/ICD change can alter device capabilities. **Do it in the gap BETWEEN items, and
before a perf item rather than after one** — measurements taken across a toolchain move cannot
attribute a change to the work.

Current pin: `ea1a7396` (Aug 2026) — `vulkan-headers`/`vulkan-loader` **1.4.357.0**, matching the
SDK installed here, which is what keeps the mixed-vulkan-hpp trap below out of reach; plus `glfw3
3.5.1`, `glslang 16.4.0`, `spirv-tools 1.4.357.0`, `imgui 1.92.8#1`, `shaderc 2026.2`, `ktx 4.4.2`,
`fastgltf 0.9.0`, `catch2 3.15.3`, `vulkan-memory-allocator 3.4.0`.

**"Full rebuild" there means `--clean-first`, and that is not pedantry.** vcpkg preserves each
port's *upstream* file timestamps, so an upgraded header can land with an mtime OLDER than the object
files that include it, and ninja will not rebuild them. The result links a stale object against a new
library and fails at runtime, not build time: after the imgui 1.91.9 → 1.92.8 bump every scene
aborted on `IMGUI_CHECKVERSION`'s "Mismatched version string" because `debug_overlay.cpp.o` still had
1.91.9 compiled into it while `libimgui.a` was 1.92.8. An incremental build reported success. Treat
any post-upgrade runtime assert about versions as a stale-object symptom first.

Expect *source* breakage from the upgrade too, and fix it at the call site rather than pinning back:
imgui 1.92 (2025/09/26) moved `RenderPass` / `Subpass` / `MSAASamples` / `PipelineRenderingCreateInfo`
out of `ImGui_ImplVulkan_InitInfo` into the nested `PipelineInfoMain`.

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
- **render/constants.hpp** — single source of truth for scalar render tunables (shadow biases, IBL strengths, shadow/IBL extents, camera FOV, bloom config). GPU data-layout limits (frames-in-flight, joint/morph/light counts, shadow caster caps) live in **graphics/gpu_limits.hpp** so the Vulkan-free graphics layer can size its arrays; constants.hpp includes it, so all constants stay reachable through one include.
- **Layering guard** — `graphics/` and `scene/` headers must not include `render/`, and `render/` headers must not include `scene/` (`render`↔`scene` meet only through the Vulkan-free `graphics/renderable_scene.hpp` seam). Enforced by the `layering_guards` CTest case (`cmake/check_layering.cmake`). Graphics `.cpp` files may include `render/resources.hpp` — that's the documented GPU-allocation bridge.
- **GPU resource model** — three orthogonal layers; never conflate them. **(1) Identity:** opaque handles (`graphics/gpu_handle.hpp`) into `Resources`' tables — an index packed with a generation so a stale handle to a reused slot is detectably invalid. **(2) Lifetime:** off-the-shelf `vk::raii` owns every Vulkan object whose lifetime is *independent of a device-memory allocation* (image views, samplers, pipelines, descriptor layouts, command pools). `vk::raii` is a lifetime tool, not a memory allocator — it imposes nothing on allocation strategy. **(3) Memory / sub-allocation:** VMA is the arena, used the idiomatic way (`vmaCreateBuffer`/`vmaCreateImage`) so it owns the resource + its sub-allocation and frees them together. Because `vmaDestroy*` couples the two, buffers and images are the *one* exception to layer 2: their `VkBuffer`/`VkImage` lifetime lives inside a small custom move-only VMA RAII wrapper (`UniqueVmaBuffer`/`UniqueVmaImage`) — still RAII, just VMA-backed. This is **not** a migration away from `vk::raii`; revisit layer 2 only if runtime streaming (mid-frame create/destroy needing a deletion queue) ever arrives.
- **One declaration of every shared GPU data-layout limit** — the sizes and indices that a C++ block and a shader block must agree on (caster/light/joint/morph/emitter/kernel counts, the map-validity bits). Purely GLSL-side algorithm constants are NOT in scope and this mechanism does not own them: a compute workgroup size, a scan radix, a tap count with no C++ counterpart stays where it is used. `shaders/gpu_limits.glsl` is written in the subset that is valid GLSL *and* valid C++; shaders `#include` it and `graphics/gpu_limits.hpp` includes it inside a `shader_limits` namespace, re-exporting each value under its `k`-name. Add a shader-visible limit **there**, never as a literal on either side, and keep the file inside the common subset (no `constexpr`, `inline`, `namespace`, `static_cast`, unsigned suffixes — each breaks the *other* language, in files that never mention this one). The `gpu_limits_guard` CTest case sweeps `shaders/` for a re-declaration, requires each consumer to use the name rather than a literal, and requires each `k`-constant to be defined *as* the shared declaration, so C++ cannot drift back to hard-coded values behind green shader checks.
- **A shadow family's recording and its uploaded validity are one value** — `ShadowMapValidity` (`graphics/shadow_map_validity.hpp`) is applied twice per frame in `Renderer::prepareShadowPlan`, in a fixed order, both from the COMPLETED view set: as ELIGIBILITY, deciding which families may be PREPARED at all (preparation resolves casters and stages hysteresis, so a family that will be neither recorded nor sampled must not be resolved); then as CONFIRMATION (`shadowMapValidityFromPlan`), derived from the finished plan and judged against the counts eligibility expected, which is what `uploadFrameLighting` writes to `LightUBO::shadowMapValidMask` for every sampling path in `shader.frag`. Never skip a family's recording without routing the decision through it — a skipped family's depth image holds an earlier frame's content, and sampling it produces no error, no crash, and shadows from a frame that is gone.
- **The shadow pass decides in preparation and records from the plan** — `prepareShadowFrame` (`graphics/shadow_pass_prepare.hpp`) filters, resolves each caster's LOD per view, claims the diagnostic row and builds a `ShadowFramePlan`; `Shadows::recordPass` consumes that plan and nothing else (no draw spans, no view set, no resolver). Anything the pass rasterises with must live in the prepared view or draw: a value read at record time that the comparison never saw is a cached shadow map kept when it should have been re-rendered.
- **A reused shadow map is a claim about the GPU, so it is only ever made after the submit** — `ShadowResidencyStore` (`graphics/shadow_pass_plan.hpp`) records what each physical view's depth image HOLDS, and `prepareShadowFrame` compares this frame's prepared content against it to mark each view `Reused` or `Recorded`. It is owned by `Shadows`, beside the images it describes: that is the whole invalidation story, and why there is no `invalidate()` to forget to call — recreating the images means reconstructing the object that owns both. Two rules the type enforces rather than its callers: only a `Recorded` view commits (a `Reused` one never touched its image, so committing its prepared work would replace the record of what the image holds with a description of a frame that wrote nothing), and an `Invalid` slot is left alone (nothing recorded means nothing overwrote the image, so its record is still true). The commit sits beside `shadowLodResolver_.commitFrame()` BETWEEN `submitFrame` and `presentFrame`, for the same reason: content adopted by a frame that was abandoned would claim an image holds pixels the GPU never drew, and committing after PRESENTATION would be worse still — raii `presentKHR` throws on an out-of-date swapchain, so a resize would skip the commit for a frame whose depth was already being rasterised — and it is `noexcept`, adopting by MOVE out of the plan (`ShadowFramePlan::takeRecorded`, with `static_assert`s pinning the no-throw moves), because on the far side of a submit there is no useful answer to a failed allocation. `RenderTunables::shadowResidencyReuseEnabled` (overlay: "Reuse unchanged shadow views") forces every engaged view to record; it is SCHEDULING, so it is an argument to the law and never part of the content descriptor — a frame recorded with reuse off commits as usual and is reusable the moment it is switched back on. Each SH-01 row carries the disposition it ended up with, because zero raster passes alone cannot separate "reused" from "never engaged".
- **GPU data-layout discipline** — every CPU struct shared with a shader (UBO/SSBO) lives in `render/ubo.hpp` with `alignas` + `static_assert`s pinning its std140/std430 offsets and size. Preserve this: when you change a shader-visible struct, update both sides and keep the static_asserts — they are the only thing catching a silent host↔GPU layout mismatch. Mapped host-visible writes go through `graphics/mapped_buffer.hpp` `writeMapped` (a bounds-checked `std::span<std::byte>`), never a raw `void*`. **And a block bound by more than one shader is declared ONCE, in a shared `shaders/*.glsl` include** (`light_ubo.glsl` for `LightUBO`, `material.glsl` for the bindless `Materials` SSBO + `MaterialData`, `shadow_push.glsl` for the `ShadowPushConstants` push block), never hand-copied per shader: field offsets depend on every field before them, so a copy missing an inserted field misreads everything after it, with no validation error and no crash. That is how the sky came to be multiplied by a shadow matrix — `selfShadowViewProj` was added to the struct and `shader.frag`, not to `skybox.frag`, and the wrong value read 1.0 until a scene had two skinned self-shadow casters. The `shader_block_guards` CTest case (`cmake/check_shader_blocks.cmake`) fails on a re-declared block *and* on a shared include that stops declaring it.

## Code Style

**Always run `clang-format -i <file>` on any C++ file (`.hpp`/`.cpp`) after editing it.** The codebase is already formatted; keep it that way per-edit.

`.clang-format` enforces: Allman braces, 4-space indent, 100-col, left-aligned pointers, no single-line functions, ctor initializers each on own line.

`.clang-tidy` enables an allowlist of bugprone/performance/modernize/readability checks for engine
`src/` and `include/fire_engine/`, run **warnings-as-errors** (`WarningsAsErrors: '*'`) — any finding
fails `run-clang-tidy` and the CI clang-tidy job, so the checked set stays clean. Run it with
`cmake --build build --target run-clang-tidy`; note local Apple-Clang can't be parsed by a Homebrew
clang-tidy (libc++ mismatch), so the **Ubuntu CI job is the source of truth** (§ Testing).
**Prefer fixing a finding over suppressing it.** Use `// NOLINT(check): reason` only when the check is a
provable false-positive that can't be cleanly restructured, and — where the invariant is
compile-time-expressible — back the reason with a `static_assert` so it can never silently go stale
(as the `std::visit`/never-valueless and `MaterialUBO`-memcmp guards do).

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
stay fast. The `tests-full` target runs the all-tags Catch2 binary plus the six build-time guards —
graphics-layer includes, shared shader blocks, the shadow bias law, the shared GPU limits, the
per-view shadow matrix, and the `[release-contract]` tags; from the source root use
`cmake --build --preset full`. Test files mirror
source paths. Shared helpers/traits live in `tests/support/`. Test assets in `tests/assets/`
→ copied to `build/test_assets/`.

Graphics-layer tests run without a GPU (opaque handles).

**CI has five parallel jobs** (GitHub Actions, all `FIRE_ENGINE_WARNINGS_AS_ERRORS=ON`): `clang-format`
+ `clang-tidy` (platform-independent lint, run once on Ubuntu), `build-test-linux` +
`build-test-macos` (build + `tests-full` on Ubuntu and macOS/arm64), and `release-contract` (below).
Each build job validates *its* platform's determinism golden. The build/test/lint stage bodies are
shared across the Docker replica and the native macOS replica via `tools/ci/ci-stages.sh` — edit
stages there, not in each script, and the GitHub `release-contract` job sources that file rather
than restating its commands.

**A test whose body is `#ifdef NDEBUG` is tagged `[release-contract]`.** Every preset here builds
`Dev`, so those bodies — the release half of a writer that asserts in Dev and returns `false` under
`NDEBUG` — compile to nothing: the cases run, pass, and assert nothing at all. The Linux-only
`release-contract` job is the only place they are real code (`cmake --preset vcpkg-release`, build
`test_fire_engine`, run `test_fire_engine "[release-contract]"`; locally,
`tools/ci/run-local-ci.sh release-contract`). Two things keep the job honest, and both must survive
any edit to it: Catch2 exits non-zero when a spec matches nothing, so a renamed tag fails rather
than passing quietly; and `tests/release_contract.cpp` holds a HIDDEN (`[.]`) sentinel case in the
same selection that fails unless `NDEBUG` is defined, so the job cannot pass by building `Dev` by
mistake. The `[.]` is what keeps it out of the Dev suites — `ctest` (`~[slow]`) and `tests-full` (no
filter) are both default runs, which exclude hidden cases, while an explicit tag selection includes
them. **Only tag a case that actually has an `#ifdef NDEBUG` in it**: the tag means "this asserts
something a Dev build cannot see". Both halves of that rule are enforced by the
`release_contract_guard` CTest case (`cmake/check_release_contract.cmake`), which fails on a
conditional case that is untagged — the silent failure, since nothing else would ever run it as real
code — on a tagged case with no conditional, on a missing sentinel, and on the tag vanishing
altogether; so the convention no longer depends on the next person knowing it. The `vcpkg-release`
preset builds into `build-release/` and points `VCPKG_INSTALLED_DIR` back at
`build/vcpkg_installed`, so it reuses the ports the Dev tree and the CI cache already hold instead
of building a second copy of every one.

**Local CI parity.** `tools/ci/run-local-ci.sh [format|configure|build|tidy|test|release-contract|all|shell]`
reproduces the **Linux** checks (Ubuntu 24.04) in Docker — its `all` includes the Release contract;
`tools/ci/run-local-macos.sh [format|configure|build|tidy|test|all]` runs the same stages
**natively on macOS**, minus that one (it is platform-independent, so one job proves it) (no container —
uses your existing vcpkg + C++ toolchain, installs nothing; Vulkan/GLFW/glslc all come from vcpkg). The Docker runner copies the
working tree into volumes (host artifacts untouched), defaults to `linux/amd64` to match CI
(`DOCKER_PLATFORM=linux/arm64` is faster but off-platform). Run the relevant one before committing
anything that could trip the stricter warnings / clang-tidy / format gate.

**The Linux image needs system packages the manifest cannot supply**, so a baseline bump can break
the build with no source change: some vcpkg ports build with `vcpkg_make` rather than CMake and
refuse to configure without autotools. `glfw3 3.5.1` pulls `pthread-stubs` on Linux for exactly this
reason, which is why `autoconf autoconf-archive automake libtool` sit in `tools/ci/Dockerfile` **and**
in both Linux jobs of `.github/workflows/ci.yml` — `autoconf-archive` is not preinstalled on
`ubuntu-latest` either. macOS is unaffected: glfw3 uses the Cocoa backend and never reaches that
dependency chain, which is exactly why a green macOS run does not clear a bump. When a port fails to
build after a bump, read the error before assuming a flake — vcpkg prints the missing programs and
the `apt install` line for them.

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
