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
log=$(mktemp)
cd build && (./fireEngineApp DamagedHelmet/DamagedHelmet.gltf skybox.hdr >"$log" 2>&1 & p=$!; sleep 6; kill $p 2>/dev/null; wait $p 2>/dev/null)
grep -icE 'VUID|validation error' "$log"   # expect 0
```

- **0 = pass.** Non-zero = a validation error; open `$log` and read the first `VUID-...` message.
- SIGTERM exit **143** is normal (we killed it) — not a failure.
- `mktemp` per run so parallel smokes don't clobber one log.

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
