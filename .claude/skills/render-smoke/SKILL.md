---
name: render-smoke
description: >
  Vulkan validation smoke test for fireEngine. Backgrounds the app on a scene for a few seconds,
  greps stderr for VUID / validation errors, reports pass/fail. Use after any render / shader /
  pipeline / descriptor / barrier change, before committing, or when the user says "smoke test",
  "render smoke", "check for VUIDs", "does it still render".
---

# Render smoke (Vulkan validation)

Validation layers are ON in the Dev build (`Device::enableValidation`, non-`NDEBUG`), so running the
app for a few seconds catches dynamic-state / barrier / descriptor / layout misuse. The window opens;
background it, kill it, grep stderr.

## Run

Build first (`cmake --build build`). Then from the repo root:

```bash
log=$(mktemp); scene=(DamagedHelmet/DamagedHelmet.gltf skybox.hdr)
cd build
FE_LOG=render:info ./fireEngineApp --require-validation "${scene[@]}" >"$log" 2>&1 & p=$!
sleep 6
alive=0; kill -0 "$p" 2>/dev/null && { alive=1; kill "$p"; }
rc=0; wait "$p" || rc=$?
vuid=$(grep -icE 'VUID|validation error' "$log" || true)
active=$(grep -c 'Vulkan validation enabled' "$log" || true)
if test "$alive" -eq 1 && test "$rc" -eq 143 && test "$vuid" -eq 0 && test "$active" -eq 1
then echo "SMOKE PASS"
else echo "SMOKE FAIL (alive=$alive rc=$rc vuid=$vuid validation=$active)"; tail -25 "$log"; false; fi
```

All four conditions are the pass, and every one of them closes a way to pass vacuously:

- **`alive` after 6s** — the process must still have been running when we went to kill it. Without
  this an early exit (no suitable GPU, missing asset, a throw during startup) leaves a log with 0
  VUIDs and, if it got far enough, the validation token: the exact false pass.
- **`rc == 143`** — it died from *our* SIGTERM, not on its own. `wait` is what reports it. (Named
  `rc`, not `status`: `$status` is read-only in zsh, the default shell here.)
- **`vuid == 0`** — nothing went wrong. Non-zero: open `$log`, read the first `VUID-...`.
- **`active == 1`** — something was actually checking. The token is logged only *after* the
  instance is successfully created, so it means validation is live, not merely requested.

Shell details that are load-bearing, not style: `scene` is an **array** (`"${scene[@]}"`) because
zsh doesn't word-split a scalar, so `$scene` would hand the app one combined path; the failure
branch ends in **`false`** so automation sees a non-zero status instead of a success that merely
printed FAIL; `rc=0; wait || rc=$?` and `|| true` on the greps keep it correct under `set -e`
(`grep -c` exits 1 on a zero count).

Both switches matter: **`--require-validation`** makes a missing layer a startup failure (exit 1,
`Fatal: --require-validation: …`) instead of a silent unvalidated run — including in an `NDEBUG`
build where the layer is compiled out; **`FE_LOG=render:info`** because the token is an `info` on
the `render` category and a quieter level would hide it. `mktemp` per run so parallel smokes don't
clobber one log. Swap `$scene` for any row in the matrix below.

## Scene matrix — cover the surface your change touches

Run from `build/`; scene paths are flat (no `assets/` prefix). Skyboxes: `skybox.hdr`, `nightbox.hdr`.

| Feature | Command tail |
|---|---|
| Opaque PBR | `DamagedHelmet/DamagedHelmet.gltf skybox.hdr` |
| Alpha blend + double-sided | `AlphaBlendModeTest/AlphaBlendModeTest.gltf skybox.hdr` |
| Transmission | `TransmissionTest/TransmissionTest.gltf skybox.hdr` |
| Skinned / morph | `RiggedSimple/RiggedSimple.gltf skybox.hdr` |
| Ragdoll (needs floor) | `-f CesiumMan/CesiumManRagdoll.gltf skybox.hdr` |
| Cloth (GPU XPBD) | `-c` |
| Particles | `-p` |
| Character controller | `-k` |
| Query probe | `-q` |
| Mesh LOD | any dense static mesh (DamagedHelmet); tint via `--overlay` + the LOD debug view |

Physics / gameplay scenes (`-f` / `-k` / `-q`) usually want `-f` for a receiver floor.

## Report

State the scene(s) run and the VUID count. If non-zero, quote the first `VUID-...` line — that's the
fix target. See CLAUDE.md § Build for the canonical recipe.
