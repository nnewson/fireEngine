---
name: add-source-file
description: >
  Register a new fireEngine source, test, or shader in the explicit CMake lists (the build does not
  glob). Use when adding a .cpp / test / shader, or when the user says "add a source file", "new file
  won't build", "register in CMake", "shader missing at runtime".
---

# Add a source file (CMake registration)

The build uses **explicit source lists**, not globbing (only `tests/assets/` is globbed). A new file
is invisible to the build until registered in `CMakeLists.txt`.

| New file | Add to |
|---|---|
| Library source `src/.../foo.cpp` | the `add_library(fireengine SHARED ...)` source list |
| Test `tests/.../test_foo.cpp` | the `add_executable(test_fire_engine ...)` list — tag long settle/soak coverage `[slow]` (`test_fire_engine` runs `~[slow]`; `tests-full` runs everything) |
| Shader `shaders/foo.frag` | `SHADER_SOURCES` — compiled to `foo.frag.spv` via `glslc` and copied next to the binary. **Forgetting this = the shader silently doesn't exist at runtime.** |

- A **header-only** addition needs no CMake change — but its **test** still does.
- New type → a matching `include/` + `src/` pair; the declaration's definition goes in the matching
  `.cpp`, never an unrelated translation unit.
- Then give the new file a tier entry in `docs/review-order.md` (see the **doc-sync** skill).

## Verify
```bash
cmake --build build            # reconfigures + compiles the new file
ctest --preset fast            # the new test runs
```
For a new shader, run the app (or the **render-smoke** skill) — a missing `.spv` fails at load.
