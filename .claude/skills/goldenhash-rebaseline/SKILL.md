---
name: goldenhash-rebaseline
description: >
  Re-baseline the platform-specific physics determinism golden hash after an INTENDED solver / step
  math change in fireEngine. Updates both the macOS/arm64 and Linux/x86_64 goldens. Use when
  Determinism.GoldenHash fails after a deliberate physics change, or the user says "rebaseline
  golden", "goldenhash", "determinism hash changed".
---

# GoldenHash re-baseline

`Determinism.GoldenHash` (`tests/physics/test_physics_determinism.cpp`) compares the physics
end-state hash (raw float bits) to a recorded golden. macOS/arm64 and Linux/x86_64 diverge by a few
last-bit contact-solver ops, so `goldenHash()` records one per platform:

```cpp
#if defined(__linux__) && defined(__x86_64__)
    return 0x...ULL;   // Linux/x86_64 — recorded via the Docker CI
#else
    return 0x...ULL;   // macOS/arm64 — recorded locally
#endif
```

## FIRST: is the move intended?
Only rebaseline a hash move caused by a **deliberate** change to solver / step math. An *unexplained*
move is a determinism regression — investigate (non-determinism, uninitialised state, order
dependence). Do **not** silence it by rebaselining.

## Update BOTH goldens
CI runs a Linux/x86_64 job **and** a macOS/arm64 job, each running `tests-full` — so a stale golden on
*either* platform fails CI. Update both.

1. **macOS/arm64** (local) — run the test and read the actual hash from the Catch2 failure
   (`hash == kGoldenHash` prints both sides):
   ```bash
   (cd build && ./test_fire_engine "Determinism.GoldenHash")
   ```
   Put the actual value in the `#else` branch.
2. **Linux/x86_64** (Docker) — read that platform's hash off the local CI replica:
   ```bash
   tools/ci/run-local-ci.sh test
   ```
   Read the actual hash from the same failing CHECK in the container output; put it in the `#if` branch.
3. Rebuild and re-run both paths; both green.

## Note
This is the only test whose golden is platform-split. Reference: CLAUDE.md § Testing.
