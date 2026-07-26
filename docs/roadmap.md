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
| [`architecturalreview.md`](architecturalreview.md) | One-shot **architectural review** (25 Jul 2026) of rendering, shadows/AA, physics, simplifier/VDPM. Its §6 table is the status of record. | 8 of 14 landed; the rest is arc 2 |
| [`shadowplans.md`](shadowplans.md) | The **shadow-LOD improvement plan** (SH-01…SH-09) spun out of the architectural review's §2. | All open — arc 1 |

Suggested order below — not binding. One branch per item, off local `main`.

---

## Arc 1 — Shadow LOD & shadow correctness ([`shadowplans.md`](shadowplans.md))

The central defect: a shadow caster's LOD is inherited from the camera view instead of being
selected for the shadow view that rasterises it, in shadow-map texels. Milestones 0–2 are the
correctness work; milestone 3 is evidence-gated. Detail, contracts, and verification gates are in
the plan; the priority order is its § Suggested priority.

**Milestone 0 — evidence before policy**
- **SH-01** — shadow diagnostics (per-group GPU time, draw/triangle counts, per-view LOD histograms,
  projected-deviation estimate, LOD0-fallback reasons, a shadow-LOD debug view) + a purpose-built
  owned glTF acceptance scene, recorded in [`acceptance-testing.md`](acceptance-testing.md).

**Milestone 1 — correct discrete shadow LOD**
- **SH-02** — the pure, Vulkan-free shadow-view projection model (`ShadowView`, per-cut shadow
  deviation metric, `projectShadowErrorTexels`, `selectShadowLod`, hysteresis), headless-tested.
- **SH-03** — thread per-shadow-view discrete LOD through the renderer (the requested architectural
  fix: one caster may select different levels for different shadow views).
- **SH-04** — deformation / proxy policy (skinned, morphed, cloth: no invalid error claims, explicit
  conservative full-detail fallback).

**Milestone 2 — shadow silhouette correctness**
- **SH-05** — material-aware casters (alpha-mask cutout, double-sided sheets).
- **SH-06** — cascade caster fit (remove fixed-depth clipping, align candidate sets).
- **SH-07** — scale-derived bias & filtering tied to each map's actual texel footprint.

**Milestone 3 — only if measured**
- **SH-08** — shadow VIPM, *if* discrete transitions remain visibly popping.
- **SH-09** — shadow VDPM checkpoint; highest complexity, requires evidence before committing.

**Independent shadow hygiene** (need not block the milestones; see the plan's § Independent shadow
hygiene): make `noShadows` suppress *recording*, not just sampling; skip directional/world/self maps
with no active primary directional light; generate the GLSL shadow-limits include from the C++
authority instead of repeating `SHADOW_TOTAL_MATRIX_COUNT` / `SHADOW_POINT_MATRIX_BASE` / the
self-shadow slot count (**same fix as arc 3's Tier 1 finding 2**); keep map validity explicit when a
family is skipped.

---

## Arc 2 — Architectural-review remainder ([`architecturalreview.md`](architecturalreview.md) §6)

The eight small/XS items landed on `review-shadow-taa-fixes` + `review-xs-cleanups`. What remains,
in the review's priority order:

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
