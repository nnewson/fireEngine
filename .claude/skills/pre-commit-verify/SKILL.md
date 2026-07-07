---
name: pre-commit-verify
description: >
  Full pre-commit verification gate for fireEngine, in order: clang-format, warning-clean build,
  fast tests, slow tests, render smoke, then the Linux / clang-tidy / format Docker CI. Use before
  committing or opening a PR, or when the user says "verify", "pre-commit", "ready to commit",
  "run the checks".
---

# Pre-commit verify

Run in order; stop at the first failure and fix before continuing. **macOS builds are lenient — the
Docker CI (step 6) is the real gate GitHub Actions enforces.**

## 1. Format
```bash
clang-format -i <every .hpp/.cpp you touched>
```

## 2. Build (warning-clean)
```bash
cmake --preset=vcpkg -DCMAKE_EXPORT_COMPILE_COMMANDS=1   # only if not configured
cmake --build build
```

## 3. Fast tests
```bash
ctest --preset fast          # or: (cd build && ./test_fire_engine)
```

## 4. Slow tests (settle / soak)
```bash
cmake --build --preset full  # builds + runs the all-tags binary incl. [slow] + the layering guard
```

## 5. Render smoke (0 VUID)
Use the **render-smoke** skill on the scene(s) your change affects (default: DamagedHelmet). Expect 0.

## 6. Linux CI parity (the real gate)
```bash
tools/ci/run-local-ci.sh all   # warnings-as-errors build + run-clang-tidy + tests-full + clang-format dry-run
```
Ubuntu 24.04, `linux/amd64` (matches GitHub Actions). Isolate a stage with `format|configure|build|tidy|test`.

## 7. If physics / solver math changed
`Determinism.GoldenHash` will move. Re-baseline BOTH platforms — see the **goldenhash-rebaseline**
skill. Never rebaseline an *unexplained* hash move (that's a determinism regression to investigate).

## 8. Docs
Sync any docs your change made stale — see the **doc-sync** skill. Same branch as the code.
