# Fire Engine — Roadmap

**Open work only.** This file is the *index* of what is still to do; it deliberately carries no
record of landed work. Everything major is complete — the physics & collision track (P0–P9), the
rendering foundations (particles, cloth/XPBD, Vulkan 1.3/1.4 modernization, bindless materials, TAA,
SSAO, frustum culling, overlay), the 26-item staff-engineer code review (CR-01…26), the post-VDPM
review pass (VDPM hardening, renderer hygiene, static GPU residency, capability-driven device
setup), and rendering spine **#3 — progressive-mesh LOD** end to end (discrete → VIPM → VDPM →
**GPU-driven front, now the default on any device that supports it**).

The design + rationale for landed work lives in the authority docs, never here:

- **Physics / collision** → [`collision.md`](collision.md)
- **Mesh LOD (discrete / VIPM / VDPM, CPU + GPU front)** → [`lod.md`](lod.md)
- **Everything else** (subsystems, invariants, reading order) → [`onboarding.md`](onboarding.md),
  [`review-order.md`](review-order.md), [`README.md`](../README.md)

## How the review docs relate

Three review/plan documents feed this roadmap. Each owns its own detail; the roadmap only indexes
the open items so they can't fork:

| Doc | What it is | Status of its items |
|---|---|---|
| [`codereview.md`](codereview.md) | Rolling **tiered static review**, following the [`review-order.md`](review-order.md) tiers (Tier 0 math, 18 Jul 2026; Tier 1 handles/limits/tunables, 19 Jul 2026). Further tiers expected. | **All open** — arc 3 below |
| [`architecturalreview.md`](architecturalreview.md) | One-shot **architectural review** (25 Jul 2026) of rendering, shadows/AA, physics, simplifier/VDPM. Audited 26 Jul so every finding now maps to a §6 row or an explicit "informational" tag. **Retire it once reviewed** — arc 2 below is self-contained. | 8 of 19 landed; the rest is arc 2 |
| [`shadowplans.md`](shadowplans.md) | The **shadow-LOD improvement plan** (SH-01…SH-09) spun out of the architectural review's §2. | SH-01…SH-03 + SH-05 landed, SH-04's deformation half landed (proxy half open); SH-06…SH-09 open — arc 1 |

Suggested order below — not binding. One branch per item, off local `main`.

---

## Arc 1 — Shadow LOD & shadow correctness ([`shadowplans.md`](shadowplans.md))

The central defect: a shadow caster's LOD is inherited from the camera view instead of being
selected for the shadow view that rasterises it, in shadow-map texels. Milestones 0–2 are the
correctness work; milestone 3 is evidence-gated. Detail, contracts, and verification gates are in
the plan; the priority order is its § Suggested priority.

**Milestone 0 — evidence before policy**
- ~~**SH-01** — shadow diagnostics + a purpose-built owned acceptance scene~~ ✅ **landed**
  (`shadow-lod-diagnostics`): per-group GPU time, per-view candidate/drawn counts, LOD histograms
  and selection reasons, the `ShadowLod` debug view, the Shadows panel, `assets/shadow_lod/`, and
  scriptable `--capture`. Runbook in [`acceptance-testing.md`](acceptance-testing.md); its captures
  are the measurement baseline SH-03 is read against.

**Milestone 1 — correct discrete shadow LOD**
- ~~**SH-02** — the pure, Vulkan-free shadow-view projection model~~ ✅ **landed**
  (`shadow-view-lod-model`): `ShadowView` + `projectShadowErrorTexels` + `selectShadowLod` +
  hysteresis, plus a dedicated per-cut Euclidean shadow-deviation channel through the simplifier
  (the RMS error measured 2x BELOW true deviation; point-to-plane misses in-plane silhouette
  movement; the support radius measured 12x-21,000x loose). Deliberately an **estimate, not a
  bound** — see [`shadowplans.md`](shadowplans.md) § SH-02 for why, and for the one-sided limitation
  it carries. No runtime behaviour changed; SH-03 threads it through.
- **SH-03** — thread per-shadow-view discrete LOD through the renderer (the requested architectural
  fix: one caster may select different levels for different shadow views). Slices 1–3 have landed —
  identity, the per-frame view set, and the unresolved command seam with per-view resolution, which
  also brought forward per-view diagnostic reasons and moved the tuning into
  `render/constants.hpp` (`kShadowLodPixelBudget` + `kShadowLodCoarsenRatio`, `kShadowLodBias`
  retired), plus per-view diagnostics with a focused-view reason breakdown (slice 4), a ShadowLod
  tint driven by that focused view (slice 5, with `--shadow-focus` for scripted captures), and the
  calibration (slice 6: budget 1 texel, no dead band, both measured against a stated threshold,
  CSM-only — see
  [`shadowplans.md`](shadowplans.md) § SH-03).
- **SH-04** — deformation / proxy policy. **Deformation half landed** (`shadow-deformation-policy`):
  skinned, morph-capable and storage-vertex casters are classified `Deformable` and resolve to full
  detail with their own `DeformableFallback` reason, infinite projected error and no hysteresis
  history — closing a hole SH-03 left open, where deformable casters selected levels from a
  BIND-POSE deviation (live, not theoretical: BrainStem transitioned a skinned caster's shadow level
  within seconds). Calibration re-run: the error column did not move (the demo's deformable casters
  are not in the measured directional view), but self-shadow cost went 184/1248 → 1248/1248 and the
  cascade group 59.9% → 68.2% of full detail; the 0.1% threshold still selects budget 1.
  **Proxy half still open** — `Object::shadowGeometry` was REMOVED rather than documented as unsafe,
  so there is currently no way to author a proxy; reinstating a validated setter (deformation
  compatibility, morph contract, proxy-derived bounds, enforced at load time) is what closes it.

**Milestone 2 — shadow silhouette correctness**
- **SH-05 follow-ups** — the item itself landed (`shadow-material-casters`; rationale in
  [`shadowplans.md`](shadowplans.md) § SH-05). Two things it deliberately left open:
  - **A coarser masked-LOD policy** needs a silhouette-error argument. Cutouts are pinned to level 0
    (`AlphaMaskedFallback`) because no simplifier channel measures where a binary alpha boundary
    lands; VDPM's UV-deviation channel is an input, not a proof.
  - **A cutout with a real LOD chain** to price that pin. The sweep was re-run on this branch and
    found SH-05's pin costs ~nothing on `ShadowLodDemo` — its masked caster is a two-triangle quad,
    so it drew its whole mesh either way. The budget/ratio calibration itself is settled and was
    re-measured on merged `main` after SH-06 (still budget 1, ratio 1.0; table in
    `render/constants.hpp`), but the pin's cost is untested until such a caster exists.
- ~~**SH-06** — cascade caster fit~~ ✅ **landed** (`shadow-cascade-caster-depth-fit`): the fixed
  `kShadowDepthBackExtend` is retired as policy. The cascade fit is split into a stable receiver
  half and a depth half; a Vulkan-free per-frame caster prepass
  (`RenderableScene::gatherShadowCasters` → `ShadowCasterBoundsFrame`) is the single authority on
  caster bounds for the fit, the draws and the diagnostics; and `fitCasterAwareCascadeDepth` places
  the near plane at the furthest-upstream candidate caster and the far plane at the receiver volume.
  Each cascade is fitted from the start of its predecessor's blend band, with the fraction uploaded
  in `LightUBO::cascadeParams.x`. Acceptance on `ShadowDepthClipDemo`: 26166 → 35324 shadow pixels.
  Cloth still forces a marked `LegacyStaleFallback` — see below.
- **SH-07** — scale-derived bias & filtering tied to each map's actual texel footprint. Better
  positioned since SH-06: the per-view metrics it needs are already returned by the fit, and the
  depth span is no longer a fixed constant.

**Open questions and follow-ups left by the milestone-2 work** (each is its own branch):

- **Suggested next: SH-07.** SH-05 landed, so milestone 2 is complete apart from the follow-ups
  listed above it. SH-07 is better positioned than it was: the per-view metrics it needs already
  come back out of SH-06's fit, and the depth span is no longer a fixed constant.
- **The historical half-ellipse is NOT SH-06's motivation and remains unexplained.** It was observed
  on `ShadowLodMotionDemo` under the engine's FALLBACK sun (the glTF loader was dropping lights on
  animated nodes), and measurement excluded depth clipping as the cause: zero `clippedNear` events
  across a 676-row live trace, closest approach 20.7 m. Diagnosing it needs the symptom re-confirmed
  under the repaired sun, the shadow pass's own per-cascade drawn verdict beside the placement
  trace, and per-pixel cascade / blend factor / projected shadow U/V at the affected receivers —
  see [`shadowplans.md`](shadowplans.md) § SH-06.
- **Cloth cannot be fitted to.** A storage-vertex caster's bounds are its bind pose, so any frame
  containing one falls back to the legacy depth range for every directional cascade. Closing this
  needs a conservative simulation or authored envelope for storage geometry; until then the
  fallback is marked `LegacyStaleFallback` in the fit result and the panel, not silently taken.
- **SH-04's proxy half** — `Object::shadowGeometry` was removed rather than documented as unsafe, so
  there is currently no way to author a shadow proxy at all.

**Milestone 3 — only if measured**
- **SH-08** — shadow VIPM, *if* discrete transitions remain visibly popping.
- **SH-09** — shadow VDPM checkpoint; highest complexity, requires evidence before committing.

**Independent shadow hygiene** (need not block the milestones; see the plan's § Independent shadow
hygiene): make `noShadows` suppress *recording*, not just sampling; skip directional/world/self maps
with no active primary directional light; generate the GLSL shadow-limits include from the C++
authority instead of repeating `SHADOW_TOTAL_MATRIX_COUNT` / `SHADOW_POINT_MATRIX_BASE` / the
self-shadow slot count (**same fix as arc 3's Tier 1 finding 2**; SH-05 reduced each of the three to
ONE hand-written copy — `shadow.vert`, `shaders/shadow_depth.glsl`, `shaders/self_shadow_second.glsl`
— but they are still hand-written, so the finding stands); keep map validity explicit when a
family is skipped.

---

## Arc 2 — Architectural-review remainder ([`architecturalreview.md`](architecturalreview.md) §6)

The eight small/XS items landed on `review-shadow-taa-fixes` + `review-xs-cleanups`. Ten remain —
the five the review prioritised, then five a later coverage audit found had no action item. In the
review's priority order:

- **#4 [B/M] Static-scene CSM caching** (§2.1) — the renderer currently re-records every shadow pass
  every frame. Needs the lightweight **epoch** idea from §5.1 (scene-transform / light / caster-set
  epochs) so individual passes can skip without a frame-graph rewrite. **Consumes SH-01's
  diagnostics and SH-03's per-view LOD contract** — a map's content signature must include the
  shadow view descriptor and every selected LOD/front generation, not just a camera epoch
  ([`shadowplans.md`](shadowplans.md) § Interaction).
- **#5 [B/L] Compute pre-skinning pass** (§1.3) — skinning/morphing re-runs in every pass's vertex
  shader (~11× per skinned vertex per frame). `SoftBodySystem` already proves the compute pattern
  in-engine. The one genuinely architectural piece here; it also retires SH-04's deformable
  full-detail fallback by exposing pre-deformed vertices + exact deformed bounds + a deformation
  revision.
- **#7 [B/S] Physics per-step scratch persistence** (§3.1) — remove the per-step heap allocation in
  the solver hot path. Golden-neutral if done as pure allocation reuse.
- **#10 [B/S] Front-to-back sort of the opaque bucket** (§1.1) — improves depth-prepass rejection.
- **#6 [C/S] Batch image barriers into single `DependencyInfo`s** (§1.2) — compounds on MoltenVK
  (§5.2); coordinate with SH-* so barrier grouping doesn't change per-view LOD decisions.

**Added by a coverage audit** (2026-07-26). The review's §6 table was a *prioritised* list, not an
exhaustive one: five actionable findings in its body had no row. They are now rows 15–19 there and
items here. All five are genuinely lower-value than the above — three are conditional or watch-items
in the review's own words — and are recorded so the arc is scoped honestly, not because each is
worth doing:

- **#15 [B/C, M] Punctual-shadow change detection** (§2.3) — spot and point casters re-render every
  face every frame even when the light and the geometry in range are static; a point light is
  6 × 1024² per frame. Same epoch/dirty-bit mechanism as #4, so do them together. (Per-face frustum
  filtering already exists and is correct — this is about skipping the re-render entirely.)
- **#16 [C, XS] `hash_combine`-style mix for the mesh-triangle warm-start key** (§3.3) —
  `in.key ^= subKey * 0x9E3779B97F4A7C15ULL` (`physics_world.cpp`) is a decent mix, but XOR over the
  pair key admits collisions across (pair, triangle) combinations. The consequence is only a wrong
  warm-start seed, which self-corrects within iterations — cosmetic, but a proper combine makes it
  principled. **Touches the solver ⇒ re-baseline the determinism golden on BOTH platforms.**
- **#17 [C, M] Retire the TAA resolve→blit full-res copy** (§2.5) — the resolve renders into
  `history[cur]`, then blits back into the offscreen HDR target so particles/bloom/post keep a stable
  input. Treating `history[cur]` as *the* scene target for the rest of the frame removes a full-res
  16F copy per frame, at the cost of per-frame (or double-buffered) descriptor updates for those
  three consumers. **Measure on MoltenVK before committing** — the current design is defensible.
- **#18 [C, XS] `vdpmDrawCounts_` accumulation is O(fronts²)** (§1.6) — linear `find` per draw
  (`renderer.cpp`), and the same pattern in `findSelfShadowViewProj`. Harmless at today's front
  counts; a **watch-item** to revisit if instance counts rise, not a defect.
- **#19 [C, S] Hoist the simplifier's per-collapse allocations** (§4.3) — `collapse()` allocates
  `oneRing`, `removedWedgeUv`, and a `neighbours` map per collapse. Already assessed as "in the noise
  next to the neighbour re-cost". Load-time only; do it **only if load times start to matter**, as
  member scratch + `clear()` per collapse.

---

## Arc 3 — Tiered static code review ([`codereview.md`](codereview.md))

Handled as a unit the way CR-01…26 was, one branch per phase. Findings map to `codereview.md` by
**title** — its numbering restarts per tier, so titles are the stable reference.

**Tier 0 — math & value types.** 3 high (`Mat3::inverse()` rejects valid small transforms;
`approxEqual()` accepts NaNs as equal; rotation quaternions don't enforce their invariant), 5 medium
(non-robust norms, "bitwise equality" isn't bitwise, affine/projective mixed, hidden projection
conventions, duplicated conversion authority) + a standardisation list. Sequenced by the doc:
1. **Correctness foundation** — NaN/tiny-matrix regression tests, shared scalar comparison, robust
   scaled norms, scale-aware `Mat3::tryInverse()` + caller migration, fix/remove the bitwise API.
2. **Rotation redesign** — `UnitQuaternion`/`Rotation3`, one quaternion→matrix authority, migrate
   transform/animation/render/physics users. *(Touches physics orientation ⇒ expect a determinism
   golden re-baseline on BOTH platforms — see CLAUDE.md § Testing.)*
3. **Transform & API redesign** — `Affine3` + direct TRS, split affine point/vector/normal from
   projective, explicit projection conventions, standardise the access/operator surface.

**Tier 1 — handles, limits, tunables.** 4 high (texture generations not enforced on lookup/release;
GPU layout limits have no machine-enforced C++/GLSL authority; `ColliderId` registration inconsistent
between broadphases; handle packing silently aliases invalid inputs), 5 medium, 2 low. Sequenced:
1. **Close correctness holes** — enforce texture generation everywhere, fix dynamic-tree collider-ID
   clearing/re-registration + broadphase parity tests, derive the shadow matrix ranges with
   compile-time relationship asserts, add the build-enforced C++/GLSL limits authority (**the same
   authority arc 1's hygiene item needs**).
2. **Identity foundation** — neutral raw-index + generational strong-handle primitives, move
   `GenerationalSlotPool` out of `graphics/` with occupancy tracking and invalid-release rejection,
   pick a no-resurrection generation policy, migrate the GPU/physics handle families.
3. **Configuration & diagnostics** — frame-ring logic independent of the literal 2, derived mip
   counts + compile-time render-default relationships, enum logger categories with an indexed
   immutable config, parser/output state into a `.cpp` with precedence tests.

Further tiers of this review are expected to follow the [`review-order.md`](review-order.md) tiers.

---

## Parked & revisit — trigger-based

Not a backlog. Each item was investigated, has data behind the decision, and is picked up only on
the stated trigger.

### Physics / collision

- **P9.6 Stage 2 — per-substep re-detection** for >20 rad/s spinners resting on floors. Boundary is
  quantified and gated by `Demos.RotationalTunnellingBoundedTo20RadPerSec`. **Trigger:** a scene that
  genuinely needs faster resting spinners.
- **Mesh-contact mid-step refresh.** **Trigger:** observed tunnelling/jitter against triangle-mesh
  level geometry.
- **Link-vs-dynamic-rigid articulation contacts** — link colliders are link-vs-static only today.
  **Trigger:** a ragdoll that must interact with dynamic props.
- **Joint split-position pass** (P9 item 4). **Trigger:** a joint-stretch case the soft-joint path
  can't hold.

### LOD / VDPM — see [`lod.md`](lod.md) § Known limits & future directions

- **Texel-density UV budget** for the UV channel.
- **Retiring the per-frame repair sweeps.** The visibility cones provably *cannot* (coverage is
  screen-space, foldover topological) and the GPU front made them cheap rather than unnecessary; a
  real retirement needs a representation-level guarantee. **Trigger:** repair becomes the measured
  bottleneck again.
- **The 7 forest skips + the non-zero repair floor on the real helmet** — genuinely non-manifold
  welded edges are isolated as roots today; a per-wedge representation would remove the residual.
- **Coarsest-level seam shift** — needs full per-wedge attribute quadrics. Real machinery,
  diminishing returns.
- **Discrete `selectLod` lacks instance-scale bounding** — VDPM got the `worldLengthScale` fix
  (metric step 4); the discrete path still projects object-space error against world distance.
  **Trigger:** picked up naturally by SH-02, which needs exactly this bound for shadow views.
- **Multi-front emit compaction** — the only remaining dispatch lever (emit is a measured ~17–56% of
  the GPU lifecycle). Apply+repair **fusion was measured and skipped** (0 ms reclaimable tail on the
  proxy; same front is critical path in both stages). **Trigger:** a scene with many fronts where
  emit dominates.

### Architecture

- **Character-controller as a scene component** (P7 decision). Today `CharacterController` is a
  `physics/` engine class driven from the main loop, **not** a scene `Components` variant — a variant
  `update(InputState, Transform)` has no `PhysicsWorld` access, and the controller *is* a world query,
  so making it a component would force `PhysicsWorld` into the Vulkan-free scene layer. **Trigger:**
  a 2nd consumer (authored character nodes / NPCs). The clean upgrade is a small
  `SceneUpdateContext { const InputState&; PhysicsWorld*; }` threaded into component updates — *not*
  a per-component back-pointer.
- **"The renderer records everything, every frame"** ([`architecturalreview.md`](architecturalreview.md)
  §5.1) — the general epoch/frame-graph question. Arc 2 #4 is the first concrete slice; the wider
  rewrite is parked until render cost is actually the constraint.
