# Shadow System Improvement Plan

*Prepared 2026-07-25 from `main` at `f525115`.*

## Purpose and scope

This plan focuses on making shadow geometry selection correct, predictable, and observable. Its
central recommendation is:

> A shadow caster's LOD must be selected for the shadow view that rasterises it, in shadow-map
> texels, rather than inherited from the camera view or resolved once for every shadow pass.

The small fixes and broad architectural actions in
[`architecturalreview.md`](architecturalreview.md) §6 are separate workstreams. In particular, this
document does not re-plan static-shadow caching, compute pre-skinning, or barrier batching. It does
make the contracts introduced here compatible with those changes, because cache invalidation and
pre-deformed geometry both need to know exactly which shadow geometry was rendered.

The current cascade fitting, cascade blending, caster frustum filtering, comparison sampling, and
dual-depth skinned self-shadow technique are sound foundations. They should be evolved rather than
replaced wholesale.

## Executive recommendation

Do this in four steps:

1. Build a shadow-specific measurement and test harness.
2. Replace the camera-derived, once-per-object discrete LOD with a per-shadow-view discrete
   selection based on the actual light projection and map resolution.
3. Close geometric and material correctness gaps around deformation, alpha masks, double-sided
   materials, cascade depth fitting, and scale-derived bias.
4. Only then evaluate continuous or view-dependent shadow LOD. Start with shadow VIPM if discrete
   transitions remain visible; add shadow VDPM only if measurements show that whole-mesh cuts leave
   material performance on the table.

The discrete baseline is not a temporary shortcut. A shadow map is itself a finite-resolution
representation, so choosing the coarsest geometry whose projected error remains below a shadow
texel budget is a principled final policy for many scenes.

## What the engine does today

The shadow path currently has this shape:

```text
Renderer computes all light-view matrices
    ↓
FrameInfo carries camera data + the light matrices into graphics/
    ↓
Object builds a forward DrawCommand
    ↓ clone
Object builds one shadow DrawCommand and resolves one discrete LOD
    ↓
Renderer buckets that same command into full/world/self shadow lists
    ↓
Shadows replays it into every intersecting cascade, spot map, point face,
and (for a skinned object) both self-shadow depth passes
```

The shadow LOD in `Object::buildDrawCommands` is selected with:

- distance from the camera to the node translation;
- the camera's perspective projection scale and viewport height; and
- `camera pixel budget × kShadowLodBias`.

That produces several mismatches:

- The selected level is independent of shadow-map resolution and light projection.
- One level is shared by near/far cascades, spot maps, point faces, and self-shadow maps even though
  they project the caster at different sizes.
- An off-screen caster can be made coarse because it is far from the camera even when its shadow is
  large on a visible receiver.
- The node translation is not necessarily the geometry/bounds centre.
- `GeometryLod::error` is the simplifier's accumulated maximum **RMS quadric error**, not a
  Hausdorff or silhouette-displacement bound. Projecting it is a useful quality heuristic, but it
  cannot support a claim that the shadow silhouette is bounded.
- That object-space error is not scaled by the world transform. This is already a known
  discrete-LOD residual in [`lod.md`](lod.md), but it is especially visible in shadow silhouettes.
- A static rest-pose simplification bound is used for skinned and morphed geometry even though the
  deformation can invalidate it.
- Continuous and VDPM forward geometry deliberately do not reach the shadow pass, so the visible
  and shadow silhouettes can change independently.

The existing decision to keep GPU VDPM output out of the shadow pass is nevertheless correct for
the current implementation. The forward front is scored for the camera, is mutable per instance,
and is produced concurrently with shadow rendering. Reusing it for a light view would be incorrect,
not merely lower quality.

## Design principles

### The pass owns shadow LOD selection

`Object` should describe the shadow geometry and its available LODs. `Shadows`, which knows the
actual matrix, projection type, map extent, and layer being rendered, should select the level for
that iteration.

Do not keep cloning a fully resolved forward `DrawCommand` and adding more shadow exceptions. Add an
explicit Vulkan-free shadow draw description at the `graphics`/`render` seam. Conceptually it needs:

- a stable per-binding draw identity, not only the object id;
- finest vertex/index buffers and the immutable discrete LOD chain;
- world bounds and the world transform's conservative length scale;
- skin/morph buffers and a deformation classification;
- material shadow mode (opaque, alpha-masked, double-sided, or non-casting);
- an optional authored shadow proxy; and
- the per-object shadow UBO handle.

The implementation can use a dedicated `ShadowDrawCommand` or a well-contained shadow payload on
`DrawCommand`. A dedicated type is preferable: it makes it impossible for a forward-only indirect
buffer or VDPM front to leak into a shadow draw, and it stops shadow metadata from expanding every
forward command.

### Error is measured in the target representation

Define a Vulkan-free `ShadowView`/`ShadowProjection` value for one renderable map layer. It should
carry a stable view id, view/projection data, map width and height, and enough projection metadata
to measure a world-space error in texels.

Do not use `GeometryLod::error` unchanged as the shadow authority. Add a per-cut shadow geometric
deviation measure derived from the collapse stream. At minimum it should max-reduce the collapse
tree's cumulative geometric deviation over the regions represented at that exact cut, rather than
using the QEM RMS that primarily exists to rank/choose general render LOD. Preserve the forward
metric so this work does not silently retune the visible LOD system.

There is an important limit to surface honestly: the current VDPM `deviationRadius` is documented as
a conservative screen-space estimate, not a rigorous Hausdorff bound. If the goal is a formal
guarantee, the load-time simplifier needs a genuine conservative surface/silhouette deviation bound
for each cut (or an intentionally looser bound over the collapse regions). If that larger metric
project is deferred, name the value an estimate, validate it against full-detail shadow masks, and
do not describe the resulting texel budget as a mathematical guarantee.

For a rigid caster with a suitable per-cut shadow deviation:

```text
world error = LOD object-space shadow deviation × σmax(world linear transform)
```

where `σmax` is the same conservative largest-singular-value idea already used by VDPM. Then:

- **Directional/self orthographic maps:** projected error is world error divided by the fitted
  world-units-per-texel.
- **Spot/point perspective maps:** use the nearest positive light-view depth of the caster bounds,
  the light projection scale, and the map extent. If the bounds cross the near plane, select LOD0.

The selection should choose the coarsest level below a named shadow-texel error budget. Do not
preserve `kShadowLodBias` as a different magic multiplier: replace it with a quantity whose unit is
documented and visible in diagnostics. A starting budget must be justified against the sampler
footprint (for example, sub-texel geometric displacement for a one-texel filter), then validated
with the test scene rather than tuned until one asset looks acceptable.

Bounds must be the per-binding geometry bounds where possible. The current object-wide
`shadowBounds` remains useful for coarse culling and self-shadow fitting, but using the same union
for every primitive can make a small primitive appear artificially near or large.

### Conservative fallbacks are explicit

Use LOD0 when the engine cannot establish a valid bound. Initially that includes:

- skinned or morphed geometry, because rest-pose simplification error is not a deformation bound;
- an alpha-masked caster until the shadow pass respects the mask and has a suitable UV/cutout
  fidelity policy;
- a caster intersecting a perspective shadow near plane;
- a singular or non-finite transform; and
- a malformed/non-monotonic LOD chain.

Cloth already follows the right policy indirectly: storage-vertex geometry does not build
simplified levels.

These fallbacks should increment diagnostics rather than silently appearing as ordinary LOD0
choices. Once pre-skinning or deformation-safe authored proxies exist, they can replace the
deformable fallback without changing the pass contract.

### Temporal stability is part of correctness

Shadow texel snapping stabilises the cascade projection, but LOD thresholds can still toggle as a
caster or light moves. Keep a small persistent history keyed by:

```text
(stable draw id, shadow view id)
```

Apply refine/coarsen hysteresis in texel-error space, analogous to the VDPM front's dead band. Reset
history when a view slot changes owner, a geometry/proxy changes, or an LOD chain is rebuilt.

For the two self-shadow passes, resolve the level once per slot and reuse exactly the same topology
for both depth layers. Their nearest/second-surface relationship is invalid if the two passes choose
different geometry.

## Roadmap

Each item below should be its own branch off local `main`, in keeping with the repository workflow.

### Milestone 0 — evidence before policy

#### SH-01: Shadow diagnostics and a purpose-built validation scene — ✅ landed (branch `shadow-lod-diagnostics`)

What landed: per-group GPU timing (directional / world-only / self / spot / point) and per-view
candidate-vs-drawn draw and triangle counts keyed by PHYSICAL slot; per-view LOD histograms and
selection reasons; the `ShadowLod` debug view (a mesh tinted by the level its shadow draw picked,
neutral grey when it casts none); the Shadows overlay panel; `tools/assetgen/` extracted from the
physics generators; the owned acceptance scenes `assets/shadow_lod/ShadowLodDemo.gltf` (static, for
reproducible captures) and `ShadowLodMotionDemo.gltf` (animated, for the stability loop), whose
`validate()` asserts every coverage claim the runbook makes; and `--capture` / `--capture-frame`,
which write one frame and exit so a reference image is scriptable.

Two things it deliberately did NOT do. It cannot say how WRONG a level was — that needs the
projected shadow-texel deviation, which is SH-02's metric — only which level was chosen and what it
cost. And its captures are the **measurement baseline**: the record of what the engine did with
camera-derived shadow LOD, which is what SH-03's per-view selection is read against.

The original plan follows, for the reasoning behind each counter.

Add diagnostics before changing selection:

- shadow GPU time split into directional, world-only, self, spot, and point groups;
- submitted and surviving draw/triangle counts per group and cascade/face;
- per-view LOD histograms;
- LOD-selection reasons, recorded at the decision (selected / LOD disabled / single-level); and
- a debug view that identifies the selected shadow LOD independently of the forward LOD.

The current single `ProfilePass::Shadow` total cannot say whether a change moved cost between
cascades, punctual lights, or self-shadowing.

**Projected shadow-texel deviation belongs to SH-02, not here.** SH-01 can honestly report which
level a view rasterised and what it cost; it cannot say how wrong that level was, because the metric
that would answer it — per-cut shadow deviation projected into shadow-map texels — is the thing
SH-02 defines. Reporting a number before then would mean inventing one.

Create a small owned glTF acceptance scene instead of relying only on sample assets. It should
contain:

- a large off-camera caster whose shadow crosses a camera-visible receiver;
- the same mesh under identity and non-uniform scale;
- casters spanning adjacent cascade bands;
- a moving rigid caster and a moving light for threshold stability;
- a spot light and a point light close enough to magnify a caster;
- an animated skinned/morphed caster;
- an alpha-masked cutout and a double-sided sheet; and
- a recognisable high-detail silhouette at several distances.

**Split the static baseline from the motion check.** `assets/shadow_lod/generate.py` emits two files:
`ShadowLodDemo.gltf` is entirely static, so its authored poses make overlay numbers and screenshots
reproducible run to run; `ShadowLodMotionDemo.gltf` adds the moving caster, swinging sun, swinging
skinned limb and pulsing morph. An animated frame has no reproducible timestamp, so the motion file
is a qualitative "no chatter, no flicker" loop, never a screenshot reference. Fixed poses live in
[`acceptance-testing.md`](acceptance-testing.md).

SH-01's captures are the **measurement baseline** — the record of what the engine does today, taken
before any selection change. They are not yet an acceptance gate: the criterion "a bounded
silhouette displacement, not merely *looks similar*" needs a bound, and the bound is a shadow-texel
number SH-02 defines. When SH-02 lands, these images become the reference the bound is measured
against.

**Known exposures the baseline records, each labelled by the item that owns the fix.** They are
present in the scene deliberately: a scene that only contained cases the engine already handles
would certify nothing.

| Observed today | Owner |
|---|---|
| Every shadow view rasterises the level the **camera** picked; one command is replayed into all of them | **SH-03** |
| Skinned and morphed casters select simplified levels like any rigid mesh — there is no deformation fallback, and their error claims are unverified | **SH-04** |
| ~~The alpha-masked cutout casts a solid silhouette — `shadow.frag` samples no texture, so the mask has no effect on its shadow~~ — **fixed by SH-05** | **SH-05** ✅ |
| ~~The double-sided sheet casts nothing at all: it is authored face-on to the sun, and the shadow pipeline fixes `cullMode = eFront` while the forward pass draws both sides of a double-sided material, so the shadow pass culls the only faces it has~~ — **fixed by SH-05** (cull mode is per-draw dynamic state now; double-sided casters cull nothing). The verdict had been per-light — a punctual light on the quad's *back* side keeps exactly the faces the sun's view culls, so the same quad would have cast a grazing sliver from it; the scene places both flat quads out of punctual reach so that second effect never sat on top of the first | **SH-05** ✅ |
| Node scale (including the non-uniform caster) does not participate in a projected-texel policy, because there is no such policy yet | **SH-02** |

Likely branch: `shadow-lod-diagnostics`.

### Milestone 1 — correct discrete shadow LOD

#### SH-02: Pure shadow-view projection model — ✅ landed (branch `shadow-view-lod-model`)

The Vulkan-free selection model, headless-testable, with no renderer changes: SH-03 threads it
through. What landed:

**The metric.** `GeometryLod::error` is an RMS quadric value — an AVERAGE, which a locally bad
region hides inside; measured on a sphere it read **2x BELOW** the surface's true movement, so it is
unsafe as a shadow authority. `MeshCollapse::deviationRadius` is point-to-PLANE and reads ~0 across
an in-plane gap, so it can miss the tangential displacement a shadow shows and a camera view barely
does. `supportRadius` is safe but measured **12x-21,000x** loose (it grows toward the object's own
size), which would pin every planar caster to LOD0 permanently.

So SH-02 added a dedicated channel: `CollapseDeviation::shadow`, the EUCLIDEAN distance to the
nearest surviving triangle — a value `measureCollapseDeviation` already computed and discarded. It
accumulates with the same running-sum envelope as the geometric channel into
`MeshCollapse::shadowDeviationRadius`, reduces per cut via the pure `shadowDeviationForCut` (max over
the EXACT `collapseCount` prefix) onto `ProgressiveLod::shadowDeviation`, and is carried — validated
— to `GeometryLod::shadowDeviation`.

**It is an ESTIMATE, not a bound, and this doc must keep saying so.** It measures the tangential
displacement of the REMOVED SAMPLE and is one-sided: a coplanar collapse bridging a concavity adds
simplified surface outside the original outline that the removed vertex never sees. Measured
directly: on a sphere at LOD2 the simplified surface sat **0.069** from the original while
original->simplified sampling reported only **0.044**. The accumulated estimate covered both, at
2.27x, but that is empirical cushion rather than guarantee. A certified symmetric surface bound
remains a swappable future metric, not hidden unfinished work inside SH-02.

**Invalid-value policy, everywhere in the chain.** A deviation is a magnitude, so negative, NaN and
non-finite all mean invalid data — never a cheap collapse. Each becomes infinity, which forces LOD0.
`std::max(0.0f, x)` is specifically wrong for this (it returns 0 for both NaN and negatives) and is
not used anywhere in the chain. An invalid collapse prefix asserts in debug and returns infinity in
release rather than clamping to a shorter one.

**The model** (`graphics/shadow_view.hpp`):

- `ShadowView` — an encapsulated value with private state and static factories, so an invalid view
  cannot be assembled by bypassing a convention. Orthographic covers directional cascades, the
  world-only cascades AND the self-shadow layers; perspective covers spot maps and each point face
  separately, carrying a factory-normalised forward so depth is light-view z.
- `nearestForwardDepth` — the minimum signed forward projection over all EIGHT bounds corners. A
  centre distance hides a caster straddling the light; a radial distance over-states depth off-axis
  (a caster 40 units to the side reads 9 deep, not 41).
- `projectShadowErrorTexels` — RADIAL texel displacement, stated explicitly rather than per-axis.
  Orthographic is `error / worldUnitsPerTexel`. Perspective is
  `error * extent * sqrt(1 + 2*tanHalfFov^2) / (2 * tanHalfFov * (depth - error))`: the square-root
  factor is the Jacobian's largest singular value at a frustum corner, and `depth - error` evaluates
  it at the closest depth the displaced surface can reach. That second term is not cosmetic — a
  frustum-corner test measured an actual displacement of 7.4256 texels against a first-order bound
  of 7.4021.
- `selectShadowLod` — validates the LOD chain first (LOD0 exactly zero; coarser cuts finite,
  non-negative, monotonic), then selects. Refinement may jump multiple levels at once to restore the
  budget; coarsening reaches the coarsest level clearing the stricter threshold. Object-space
  deviation is bounded into world space by the conservative sigma_max shared with VDPM
  (`math/singular_value.hpp`), so a scaled instance can never under-refine.
- Distinct fallback reasons — `InvalidView`, `InvalidCaster`, `NearPlane`, `InvalidPreviousLevel` —
  so SH-01's panel never reports a forced LOD0 as a deliberate `Selected`.
- `previousLevel` is valid ONLY for the same draw and the same LOGICAL view. An out-of-range value
  is not clamped: clamping would hide a punctual slot reassigned to another light applying one
  caster's hysteresis to another's geometry.

**Acceptance statement** (replacing the earlier "bounded silhouette displacement", which promised a
guarantee the metric does not provide):

> Select the coarsest cut whose conservatively accumulated shadow-deviation estimate projects within
> the texel budget; validate the estimate empirically against full-detail shadow masks.

**Deliberately deferred to SH-03**, because both are tuning values that must be measured rather than
guessed: `ShadowLodHysteresis::coarsenRatio` defaults to an INVALID sentinel (the selector rejects
it) so no caller silently inherits an unmeasured number, and `selectShadowLod` takes its texel budget
as an argument with no default.

#### SH-03: Resolve LOD per shadow iteration — ✅ landed (branch `shadow-per-view-discrete-lod`)

Change the command seam so `Object` emits an unresolved shadow geometry description. Resolve its
index buffer/count inside each directional cascade, world-only cascade, self-shadow slot, spot map,
and point face.

Important invariants:

- culling and LOD use the same `ShadowView` descriptor;
- full and world-only CSM use the same choice for a given rigid draw/cascade;
- both passes of one self-shadow slot use one choice;
- point-face selection uses the face actually being rendered;
- forward `lodLevel`, indirect buffers, and `vdpmGpuFront` are irrelevant to shadow recording; and
- ~~the shadow LOD chain belongs to `shadowGeometry`, not necessarily the visible geometry~~ —
  **superseded by SH-04**, which removed the unvalidated shadow-proxy setter. The caster IS the
  visible geometry until a validated proxy API reintroduces the distinction with the compatibility
  rules that make it safe.

The draw loop should resolve a buffer view without copying the whole command per pass. This is also
the right seam for future cache signatures: the selected geometry identity and level become explicit
inputs to shadow content.

Likely branch: `shadow-per-view-discrete-lod`.

**Implementation slices** (each verifiable on its own; **all six landed**, merged as PR #126):

1. **Identity foundations.** `NodeId` on the scene node and into `Lighting`; `ShadowCasterId` +
   generation on `GeometryBindings`; `ShadowLogicalViewId` / `ShadowLodStateKey`. Hysteresis is
   frame-to-frame state, so before anything is cached its key must name the same thing next frame as
   it did last frame — a physical spot/point slot does not.
2. **`ShadowRenderViewSet`.** One per-frame authority holding each view's matrix, projection
   descriptor and logical identity together. The renderer's parallel `shadowViewProjs_` array is
   GONE: every fit populates the set, and the ShadowUBO matrix array, the LightUBO cascade/spot/self
   arrays, the coarse cull frustums and the shadow pass are all derived from it. Population order is
   forced by what is known when — reset + cascades + punctual views in `updateFrameLighting`, self
   layers once the draws exist, world-only once `anySkinned` does.
3. **Unresolved command seam + per-view resolution.** `Object` emits a `ShadowGeometryRequest` — LOD
   span, conservative world scale, caster id, generation — and CLEARS the shadow command's inherited
   index buffer, count and level, so an unresolved command cannot be mistaken for a resolved one.
   `shadows.cpp` filters first and resolves second (a caster a view rejects acquires no dead band
   against it), through `ShadowLodResolver`: a per-frame cache keyed on the full
   `(ShadowCasterId, generation, ShadowLogicalViewId)` — the LOGICAL view rather than the physical
   slot, which is what makes a cascade and its world-only twin, and a self-shadow slot's two depth
   layers, share one decision; the caster and generation are in the key because a view-only key
   would hand one caster's answer to another — plus a hysteresis history under that SAME key,
   STAGED during recording and committed
   only after a successful submit. Only a genuinely `Selected` level enters the history; an invalid
   key enters neither store. `Shadows::recordPass` takes the view set directly, so each iteration's
   cull frustum, projection and history key come from one entry.
   *Diagnostics moved with it* (slice 4's first half, pulled forward because leaving it would have
   left a lying counter): reasons are now per view in `ShadowViewStats::lodReasons`, and
   `candidateTriangles` means FULL-DETAIL triangles offered — a rejected caster is never resolved,
   so it has no level to count. `candidateDraws − drawnDraws` remains the filter yield exactly.
   *Tunables moved with it* (from slice 5, because the first runtime call needed them):
   `kShadowLodPixelBudget` + `kShadowLodCoarsenRatio` in `render/constants.hpp`, threaded explicitly;
   `kShadowLodBias` retired. The ShadowLod tint is neutral grey meanwhile — a caster holds one level
   per view, so there is no single number to tint by until slice 5 picks a view.
4. **Diagnostics.** The panel names the view it is reporting: clicking a slot row sets
   `RenderTunables::shadowViewFocus` to that row's LOGICAL identity plus its group, and the reason
   table retargets to that view alone. Identity, never slot: punctual and self slots compact in
   scene-gather order, so a slot-keyed focus silently retargets to whichever light replaced the one
   selected — and that gets worse in slice 5, where the tint reads the same focus from the CURRENT
   frame while the panel shows a COMPLETED ring frame. `ShadowViewStats` therefore records the
   identity it was rasterised with (`beginRasterPass` requires it), and `focused()` searches the
   group for it, returning the slot it was found in. The group stays part of the key because a
   cascade and its world-only twin deliberately share one identity.
   Three outcomes read differently: the rollup; "selection is not a valid view" (structurally
   unaddressable — an invalid identity, or one whose kind cannot occur in that group, so no frame
   can satisfy it); and "not present in this frame" (well-formed but not found, which deliberately
   does NOT claim whether it will return — a deleted light and a view that simply did not rasterise
   are the same thing here, and separating them needs scene liveness the diagnostics do not have).
   A row also refuses a second, different identity within a frame (`beginRasterPass` returns false,
   changing nothing) and the shadow pass makes that terminal, because merging two views' counters
   under one name produces a row that reads like a measurement of something that never existed.
   The three labels SH-03 falsified are corrected: draws d/c is the cull yield, tris d/c is culling
   AND LOD (its candidate is full detail), and the level columns are LOD selections of DRAWN casters
   (a rejected candidate is never resolved, so it has no level to contribute), counted once per
   logical view. The four observation rules are stated on `ShadowViewStats::observe`, where the
   invariants they protect live. Column widths are explicit because the level columns — the one
   thing the table exists to show — were being ellipsised to a single character.
5. **Tint.** The ShadowLod debug view colours each mesh by the level ONE shadow view resolved for
   it — the panel's focused view, or cascade 0 by default, named in the overlay so the default is
   never silent. The level is READ BACK through `ShadowLodResolver::drawnResolution(group, key)` —
   what that FAMILY drew for this caster — never re-selected: a second selection would see a
   different history state and the picture would contradict the geometry it claims to describe. It
   is not `frameResolution(key)`, which returns the decision SHARED by every view with that
   identity; a cascade and its world-only twin share one while drawing different casters, so the
   level alone would attribute one pass's choice to another. (`frameResolution` remains for
   inspecting the shared decision itself.) Filled between the shadow pass and the forward pass, the
   one window where the levels exist and the forward push constants have not been written yet. Grey
   means that view has no level for that mesh — it casts no shadow, or that view culled it — never
   level 0, which would read as "full detail chosen". `--debug-lod` / `--debug-shadow-lod` make both
   tints capturable without a human at the keyboard, and `--shadow-focus <group>:<slot>` picks the
   view: resolved once at startup into that slot's logical identity, then followed across
   compaction, failing by name if the slot is not active.
6. **Calibration + docs.** Measured, not guessed — reproduce with `tools/shadow_lod_sweep.sh`; the
   numbers and the reasoning live on `kShadowLodPixelBudget` / `kShadowLodCoarsenRatio` in
   `render/constants.hpp`. Summary:

   *The reference is `--no-shadow-lod`* — forward LOD untouched, every caster at shadow LOD0. Two
   near-misses were rejected on the way: `--no-lod` (also disables FORWARD LOD, so the comparison
   contains visible-geometry differences — the first table was measured this way and discarded) and
   a tiny budget (selection still runs, and a zero-deviation cut stays eligible, so it means "almost
   always LOD0", not "LOD0").

   *The metric is the shadow MASK, not the frame, and its denominator is measured.* Whole-image PSNR
   dilutes a localised silhouette error into invisibility, so the sweep compares the `--debug-shadow`
   visibility image and counts pixels whose shadow state differs by more than 8/255. The shadowed
   area is the pixels that differ between the reference and the same view with `--no-shadows`
   (10.4% of the frame) — an earlier pass called "darker than half" shadowed, which counted the night
   skybox and flattered every percentage by ~3.8x. The reference captured twice gives a noise floor
   of exactly zero, so every reported number is signal.

   *Budget → 1 texel*, against a threshold registered before the numbers were corrected: at most
   0.1% of shadowed pixels may differ, and the differences must sit on silhouette edges rather than
   in filled regions. Only budget 1 passes (0.003%, 59.9% of the full-detail triangles); 2 (0.243%)
   and 4 (0.356%) fail. With the inflated denominator the same table put 4 at 0.093% and selected
   it — the threshold was deliberately NOT widened to preserve that answer, since moving a stated
   criterion after seeing the data turns a calibration into a rationalisation. Holding the line costs
   6.6 percentage points of geometry and buys a hundredfold smaller error. There is no knee in the
   savings curve to appeal to — each doubling keeps buying triangles — so a threshold is the only
   honest basis.

   *Ratio → 1.0 (no dead band).* Note the direction: a SMALLER ratio WIDENS the band, costing
   triangles to buy stability. Chatter is a reversal that undoes a RECENT transition, within a stated
   window; counting every return scored the scene's periodic animation as instability (a time-blind
   count reported 0.31 reversals per 100 frames, which was a caster walking L1 → L2 and back as the
   sun swung). Swept at the SELECTED budget of 1 — a ratio measured at a different budget would not
   reproduce this decision — reversals are 0.00 per 100 frames at ratios 1.0, 0.75 and 0.5, against
   0.27 / 0.09 / 0.09 plain transitions. There is no chatter to buy off.

   *SCOPE — this calibrates the CSM, not every family.* `--debug-shadow` visualises the primary
   directional visibility and the triangle column is the cascade row, so cascade and world-only are
   measured; spot, point and self share the constant but their quality is not measured here. Their
   savings are visible in the panel's other rows. Extending the evidence to the punctual families
   needs a per-family visibility view and should happen before the constant is treated as globally
   validated.

   *Instruments added for it*, because "chatter" had no number: `ShadowLodTransitions`
   (held / transitions / reversed / first-seen, counted at commit, with reversal time-boxed so
   movement and oscillation are different measurements), shown in the panel and logged per frame at
   `FE_LOG=render:debug`; `--shadow-budget` / `--shadow-ratio`, both TERMINAL on unusable input so a
   mistyped sweep step cannot quietly re-measure the default; and `--no-shadow-lod` for the
   reference, which leaves forward LOD alone (`--no-lod` still means full detail everywhere). The
   sweep script asserts the repository's liveness contract on each animated run — alive before the
   SIGTERM, exit 143 after, a minimum record count — so a crashed run fails instead of reporting a
   reassuring zero. Re-run the sweep when SH-04 and SH-05 change which casters are selected.

#### SH-04: Conservative deformation and proxy policy — ✅ deformation half landed (branch `shadow-deformation-policy`), proxy half open

**Deformation half: landed** (`shadow-deformation-policy`). The proxy half is deliberately deferred —
see below.

The hole SH-03 left open: `object.cpp` built every shadow request with the frame's LOD toggle and
nothing else, so skinned and morphed casters selected levels from a deviation the simplifier measured
on the **bind pose**. That is not a loose estimate, it is an unfounded one — a joint rotation can
carry a vertex arbitrarily far from where its rest-pose deviation was taken, so the number describes
a mesh that is never drawn. SH-02 was careful to call the metric an estimate; this is the case where
that word stopped covering it. It was live, not theoretical: BrainStem logged a shadow-level
transition on a skinned caster within seconds of starting.

What landed:

- **`ShadowCasterDeformation` on the request** (`Rigid` / `Deformable`) — a classification, not a
  policy switch. Deliberately NOT expressed by passing `lodEnabled = false`, which would have
  reported `LodDisabled` and conflated a user's toggle with a safety fallback: the panel would then
  answer "why is this caster at full detail?" with somebody else's reason. It defaults to
  `Deformable` on the same principle as `worldScale`'s NaN — a producer that forgets the field must
  not receive the optimistic answer.
- **`ShadowLodReason::DeformableFallback`**, resolving to the whole mesh with an **infinite**
  projected error. Not zero: zero would rank a deformable caster as the most accurate in the frame,
  the exact inversion of what is known about it.
- **Precedence**, most-specific-cause-first: `InvalidCaster` (a producer bug outranks a property of
  well-formed geometry) → `LodDisabled` (a global switch is the operative fact about every caster;
  a deformable one is not *more* disabled) → `DeformableFallback` → `SingleLevel` → selection.
  Deformation is checked **before** the chain length so a single-level deformable caster still says
  why it may not select.
- **Classification at the object seam** (`graphics/shadow_caster_deformation.hpp`), covering all
  three carriers: skinned instances, **morph-capable geometry regardless of current weights** (an
  all-zero weight set is this frame's value, not a property — classifying by weights would swap a
  caster's error model mid-animation), and **storage-vertex geometry** (cloth, whose vertices a
  compute pass rewrites). Cloth is single-level today and so was safe *by accident*; classifying it
  explicitly is what stops that accident from becoming load-bearing the day storage-vertex geometry
  gains an LOD chain.
- **No hysteresis history** for a fallback — already guaranteed by the `Selected`-only staging rule,
  now pinned by a test so it cannot regress silently.
- **`Object::shadowGeometry` removed.** It was unused, and it was a public route around every rule
  above: substituting a proxy silently substitutes its topology, its deformation carriers (a rigid
  proxy for a skinned mesh casts a shadow frozen in bind pose) and its bounds — which drive per-view
  LOD and culling — with nothing checking that any of them match. Documenting it as unsafe would have
  left the hole open while claiming it was closed.

**The proxy half remains open**, and reinstating a setter is what closes it. A validated proxy API
must verify deformation compatibility (a skinned proxy shares a vertex/joint mapping or is
rejected), state an explicit morph contract, allow a rigid proxy its own LOD chain, and take bounds
from the **proxy** rather than the visible geometry — enforced at load time, where a rejection can
still be reported. Until then there is no way to author one, which is the intended state: no route
in beats an unchecked route in.

**Calibration was re-run, not assumed.** `ShadowLodDemo` carries a skin and a morph primitive by
design, so forcing them to LOD0 changes the caster mix and the measured tables move — see
`render/constants.hpp` for the re-derived numbers against the same pre-registered 0.1% threshold.

### Milestone 2 — shadow silhouette correctness

#### SH-05: Material-aware caster pipelines — ✅ landed (branch `shadow-material-casters`)

The shadow fragment path could not apply the visible material's alpha mask, and the generic shadow
pipeline front-culled even double-sided materials. Both were visible in `ShadowLodDemo`: the
alpha-masked cutout cast a solid rectangle, and the double-sided sheet — authored face-on to the sun —
cast nothing at all, because `cullMode = eFront` discarded the only faces it had.

The item asked for four shadow material modes (opaque / alpha-masked x single- / double-sided). They
landed as **two pipelines, not four**: only the ALPHA half needs a different fragment shader, and the
SIDEDNESS half is dynamic cull state, which the shadow pipelines had never used.

What landed:

- **`ShadowCasterAlpha` (`Opaque` / `Masked`)** on the request and the command, classified once at
  the object seam by `shadowCasterAlpha(const Material&)`
  (`graphics/shadow_caster_alpha.hpp`) — the SH-04 shape, for the SH-04 reasons: a classification
  rather than a policy switch, testable without a GPU, and derived ONCE so the LOD decision and the
  fragment path cannot disagree about what a caster is. It defaults to **Masked**, the pessimistic
  answer: a caster wrongly classified Masked pays a fetch and full detail and *says so* in the panel,
  while one wrongly classified Opaque silently reinstates the solid rectangle. BLEND maps to Opaque
  deliberately — its shadow semantics (opaque / dithered / transmittance) remain an open design
  decision, and the cutout path would settle it by accident.
- **`shadow_masked.frag` + `self_shadow_second_masked.frag`**, applying the cutout through the
  BINDLESS MATERIAL AUTHORITY (`shaders/material.glsl`, set 2) reached by a new
  `ShadowPushConstants::materialIndex` — the same `materials[]` SSBO and texture array the forward
  shader indexes, with the material's UV set, `KHR_texture_transform`, sampler and factor. The
  masked *second* self-shadow layer is not optional detail: a cutout caster whose first layer masks
  and whose second does not would record a second surface where the first recorded none, and
  self-shadow through its own holes.
- **One implementation of the cutout test.** `materialBaseColourTexel`, `materialSlotUv`,
  `materialAlpha` and `materialAlphaCutoutFails` moved into `shaders/material.glsl`, and
  `shader.frag` now calls them too, so forward and shadow apply the same cutoff to the same UVs by
  construction rather than by review. Four shared shader includes were extracted with it
  (`material.glsl`, `shadow_push.glsl`, `shadow_depth.glsl`, `self_shadow_second.glsl`), and `Materials` + `ShadowPushConstants` are now GUARDED blocks
  (`cmake/check_shader_blocks.cmake`) — the push block had been hand-copied into three shadow stages
  and would have become four.
- **Dynamic cull mode on every shadow pipeline**, set per draw from an explicit per-family
  `ShadowFaceCull` policy (`PerCaster` / `AllFaces` / `BackFacesOnly`). `PerCaster` is what fixes the
  sheet: double-sided casters cull nothing, single-sided still cull front faces. Because the policy
  is now recorded state, the self-shadow FIRST pipeline disappeared — it had differed from the main
  shadow pipeline in cull mode alone.
- **Bindless is always set 2.** A shadow pipeline wants the bindless set without forward globals, and
  descriptor sets must be contiguous, so it would have received it at set 1 while forward pipelines
  have it at 2 — making "which set is bindless?" a per-pass fact for `material.glsl` and every
  recorder. `Pipeline`'s constructor declares an EMPTY set-1 layout in that case instead.
- **Alpha-masked shadow LOD begins at level 0**, reported as
  `ShadowLodReason::AlphaMaskedFallback` with an INFINITE projected error (not 0, which would rank a
  cutout as the frame's most accurate caster). Precedence: `InvalidCaster` -> `LodDisabled` ->
  `DeformableFallback` -> **`AlphaMaskedFallback`** -> `SingleLevel` -> selection. Below deformation
  because a mesh that moves after measurement has no valid error model at all, which is the stronger
  statement; above the chain length so a single-level cutout still reports why it may not select.
  Like every forced fallback it stages no hysteresis history.

**The silhouette policy is still open, and this reason is what keeps it visible.** No channel the
simplifier records measures a cutout boundary: a collapse can hold the surface inside the shadow
budget while moving the alpha edge anywhere (a shifted wedge UV redraws the leaf's edge). VDPM's
UV-deviation channel is a useful input but is not a proof that a binary alpha boundary is preserved,
so a coarser masked policy needs a silhouette-error argument first.

**The SH-03 budget sweep was re-run twice** (`tools/shadow_lod_sweep.sh`, idle machine): once on this
branch, because SH-05 changes which casters reach the shadow mask and every relative percentage in
that calibration is measured against the shadowed area — and again **on merged `main`**, because
SH-06 had landed meanwhile and the two items move the same figures in opposite directions, so
neither single-item run described the shipped engine. The merged table is the one in
`render/constants.hpp`; the outcome:

- SH-05 alone grew the measured shadowed area **10.4% -> 12.08%** of the frame — the predicted
  effect, and a bigger denominator, so every relative error fell slightly (budget 2 0.243% ->
  0.210%). SH-06 alone pushed the same figure the other way (0.289%). **Merged: 0.249%**, with the
  area settling at 12.07%;
- the **0.1% acceptance threshold re-applied unchanged still selects budget 1** (0.003%) in every
  one of those runs, so the constant never depended on which pair of changes was in the build;
- the dead-band half is statistically unchanged in both runs (3 / 1 / 1 transitions, **zero
  reversals** at every ratio), so ratio 1.0 stands;
- the merged triangle column is much lower than SH-05's run — 50.9% of full detail at budget 1
  against 68.2% — but that is **SH-06's** doing, not shadow LOD's: its caster-aware depth range ends
  at the receiver volume, so the cascade group now draws 30 of 52 candidate draws where it drew 39.
  Worth stating explicitly so the LOD budget is not credited with a cull win;
- SH-05's LOD pin costs ~nothing on this scene, and the reason is recorded rather than inferred: the
  masked caster is a two-triangle quad, so it carried a single level and drew its whole mesh anyway.
  It reported `SingleLevel` before and reports `AlphaMaskedFallback` now. **The pin is therefore
  untested by cost** — pricing it needs a cutout with a real LOD chain, which is a scene the
  acceptance set does not yet have.

The acceptance capture for the cutout and the sheet is a single `ShadowLodDemo` frame, and both
reference images were regenerated. The two fixes interact (making the pass respect `doubleSided`
changes which faces record depth, which changes what the alpha test then discards), so they were
landed and signed off together rather than in two passes. Expect the sheet's camera-facing side to
read DARK in the new capture: the sun is behind it, so once it records depth it correctly shadows
itself — before SH-05 it was bright because nothing of it reached the shadow map at all.

#### SH-06: Receiver/caster-aware cascade depth fitting — ✅ landed (branch `shadow-cascade-caster-depth-fit`)

The cascade XY fit is stable, but its light-space depth currently relies on the fixed
`kShadowDepthBackExtend`. A caster farther behind the receiver slice than that constant can be
clipped even though its shadow reaches the slice.

**RESOLVED, 2026-08-01 — SH-06 keeps its depth-fit scope, and now has a valid red test. The
historical half-ellipse is a SEPARATE open question.**

Two things came out of building the reproduction.

*The scene was mis-lit.* `ShadowLodMotionDemo` rendered under the engine's FALLBACK sun, because the
glTF loader dropped lights on animated nodes (fixed; see [`onboarding.md`](onboarding.md)
§ Cross-File Invariants). Every observation on this scene, including the half-ellipse, was made
under lighting the asset did not author.

*The fixed back-extension does clip real casters, and the cost is now measured.* The first probe
placement — a sphere centred ON cascade 2's legacy near plane — produced a complete ellipse, which
is correct and not a null result: the shadow pass culls FRONT faces
(`Pipeline::shadowConfig`), so the surface a caster records is its far side, and a sphere centred on
the plane has that whole surface downstream of it. Moving the caster one metre UPSTREAM along the
sun — which does not move its shadow, since it stays on the same light ray — makes the recorded
surface straddle the plane. The trace then reports `clippedNear=true` (`W [-45.723, -38.953]` against
`depth W [-41.340, 33.535]`) and the shadow measurably shrinks:

| probe placement | shadow pixels | bounding box |
|---|---|---|
| centre on the near plane | 35253 | 306 x 152 |
| recorded surface straddling | 26166 | 263 x 131 |

That is 14% smaller linearly and 26% by area, against an analytic prediction of 13.4% / 25% for the
cap the plane removes from a sphere. **This is SH-06's acceptance gate**: the caster-aware depth fit
must restore the full-size ellipse, and the numbers above are the pre-fix baseline.

Note the scope of the shape result. A plane perpendicular to the light removes a cap symmetric about
the light axis, so THIS fixture's sphere shrinks concentrically rather than acquiring a straight
edge. That is a statement about a sphere: asymmetric geometry can certainly present a straight
projected boundary under the same clip. What generalises is the mechanism, not the silhouette.

*The half-ellipse itself is unexplained and is not SH-06's gate.* Sweeping the caster's whole
animation range through `CascadeReceiverFit::fit` -> `fitLegacyCascadeDepth` -> `placeCaster` finds
no pose where the moving caster is depth-clipped (closest approach 20.7 m under the fallback sun,
30.8 m under the authored one), and a 676-row live trace over a ~25 s run reports zero `clippedNear`
events for any caster. The sweep reproduces the engine's own logged cascade-0 fit to the printed
digit and the trace flags the probe scene, so neither null is vacuous. Diagnosing it needs, and does
not yet have:

- the symptom CONFIRMED to still occur under the repaired authored sun (four sampled frames did not
  show it, which is not a search);
- the pass's own per-cascade filter/drawn verdict beside the placement — the trace runs before
  `ShadowDrawFilter`, so it proves where a caster is, not whether that cascade rasterised it;
- at the affected receiver pixels: the selected cascade, the blend factor, and the projected shadow
  U/V, since a receiver sampling outside a map, or two cascades disagreeing across a blend, can
  produce a straight boundary.

A footprint classification (`CascadeFootprintRelation`: Invalid / Outside / Inside / Straddles, with
edge-touching conservatively Straddles) now rides on the placement to make those cases inspectable.
It is diagnostics, NOT an accusation: light rays in an orthographic directional map preserve U and V,
so the part of a straddling caster outside the rectangle cannot shadow any receiver inside it. 82 of
those 676 rows straddle, and that is ordinary.

**Agreed structure (2026-07-31).** The fit splits into TWO carriers, because the pipeline is
`receiver slice → stable XY fit → candidate query → depth fit → render matrix` and the candidate
query consumes the first. Baking Z into it would make the query depend on its own output:

- **`CascadeReceiverFit`** — slice near/far, normalised light basis, frustum centre and snapped
  centre, radius, snapped U/V bounds, **exact receiver min/max W taken from the eight corners** (not
  `centreW ± radius`: the far plane must reach the receiver VOLUME, not the looser bounding sphere),
  and world units per texel.
- **`CascadeDepthFit`** — near/far W, light position, view-depth span, and the final view-projection
  matrix.

`fitLegacyCascadeDepth(receiver, kShadowDepthBackExtend)` reproduces today's behaviour first, so the
receiver fit and its tests land unchanged; SH-06 then swaps that one function for the caster-aware
policy. Pins: invalid or non-finite camera / aspect / slice / extent / light direction returns
FAILURE rather than letting `makeViewBasis` manufacture a plausible basis from corrupt input; tests
assert projection INVARIANTS (snapped U/V bounds map to clip edges, near/far W map to Vulkan depth
0 and 1) as well as the literal legacy numbers; diagnostics log only values read back from the two
carriers, including normalised light direction and aspect, never re-derived ones.

**Landed (slice 2, `render/cascade_fit.{hpp,cpp}`).** The `fitCascade` lambda is gone; the renderer
calls `fitCascadeReceiver` then `fitLegacyCascadeDepth`, and takes `worldPerTexel` back OUT of the
receiver fit rather than recomputing it. `tests/render/test_cascade_fit.cpp` carries a verbatim copy
of the pre-extraction lambda and asserts the resulting matrices are **bit-identical** across four
camera/sun poses × the four shipped splits, with a second case proving that reference still responds
to its own inputs (so the equality cannot pass vacuously).

One contract fell out of that equality rather than being designed in: **`lightDirection` must arrive
unit length and is REJECTED, not normalised, otherwise.** `Vec3::normalise` of an already-unit vector
moves it by an ulp whenever its squared length landed just under one — enough to change two of the
four poses' matrices. Re-normalising defensively would therefore have silently altered every shipped
cascade, and repairing a scaled direction would hide the producer that scaled it, so the fit refuses
it the same way it refuses a NaN aspect. The tolerance is `8 * FLT_EPSILON` on squared length —
sized in float rounding, not a round decimal, so a 1.0001 scale is caught while real `normalise`
output is not. (The engine normalises the sun twice on the way in: `Light::toLighting`, then
`Renderer::updateLightData` into `directionalLightDir_`, which is the authority for what the fit
receives.)

**The receiver fit is a CLASS, not an aggregate** (`CascadeReceiverFit::fit` is the only
constructor), for the same reason `ShadowView` is one. As a public struct it had a hole no
field-wise validator closes cheaply: a `lightUp` set equal to `lightDirection` — or to zero — is
finite, passes finiteness/ordering/positivity checks, and sends `Mat4::lookAt` to its own fallback
up, manufacturing precisely the plausible basis the API says it refuses. A shared full-carrier
validator (basis finiteness, unit lengths, orthogonality, handedness, snapped-centre and U/V-width
consistency) would have worked only while every future consumer remembered to call it, which is the
drift this extraction exists to remove. Encapsulation makes the invariant unexpressible instead;
`tests/render/test_cascade_fit.cpp` pins it with `STATIC_REQUIRE_FALSE(is_default_constructible)` /
`is_aggregate`, since there is no longer a runtime state to test.

`CascadeDepthFit` stays a plain aggregate deliberately — nothing CONSUMES one, so no policy has to
trust it. Encapsulate what is an input to something else.

`fitLegacyCascadeDepth` therefore validates only `backExtend` (the one input still arriving from
outside) and its own output. That output check is the interesting one, and it was verified against
the raw expressions rather than assumed: the view set rejects only non-finite matrices, and a
negative extension is always finite — a small one pulls both planes inside the sphere the cascade
was fitted to (clipping its own contents), and one past `-radius` reverses the range so every depth
comparison in the map inverts. Both look healthy from outside. The renderer makes either rejection
terminal via `rejectedCascadeFit`.

Per-cascade fit diagnostics log at `FE_LOG=render:debug`: every 120 frames starting with the first,
**plus unconditionally on the frame `--capture-frame` selects**, reading only carrier fields. The
periodic sample alone cannot describe a capture — at a 120-frame stride, frame 300's image would be
explained by the fit from frame 241 — so every later caster-bound / cascade-blend diagnostic should
use the same capture-triggered condition and describe one submitted frame.

**Fixture status.** `ShadowDepthClipDemo` is a **probe**, not the acceptance fixture, until a pre-fix
capture is genuinely red — the first placement produced a complete ellipse, which is consistent with
the prediction being exact: the shadow pass front-culls, so a near plane through a sphere's centre
removes the upstream hemisphere that never contributed, and the downstream hemisphere still projects
the full silhouette. A null result there says nothing about the arithmetic.

**The fit log settles the arithmetic half (2026-07-31).** With slice 2's diagnostics on,
`ShadowDepthClipDemo` reports cascade 2 at `depth W [-41.340, 33.535]` — the near plane the placement
predicted, to the printed digit, and the caster centre sits on it (its light-space W is -41.34 by
construction). So the calculation is exact and the probe is placed where it was meant to be; what it
is not is DISCRIMINATING, for the front-culling reason above. The probe therefore stays a probe, and
the next step is the one already agreed: freeze the ORIGINAL failing animated pose into a static
fixture rather than redesigning geometry, and record with it: receiver view depth, selected cascade
and blend factor, caster U/V/W bounds against both cascades involved, and whether the caster was
offered and rasterised in each — enough to separate depth clipping from candidate rejection or
cascade blending.

**Landed so far.** The two fit carriers and `placeCaster` (slice 2 + evidence tooling), and the
caster prepass: `RenderableScene::gatherShadowCasters` walks the scene before the fit and fills a
`ShadowCasterBoundsFrame` with each shadow-casting binding's world bounds in the current pose, with
cloth marked `Stale` because a compute pass rewrites its vertices. That record is the frame's ONLY
authority — `buildDrawCommands` receives it and every shadow command looks its own binding up rather
than anything recomputing, which both removes a second per-frame skinning walk and keeps the
per-binding precision the depth fit needs (the old path built an object-wide union). Two rules ride
with it: cloth is no longer coarse-culled by its bind-pose box
(`Object::localBoundsCoverDrawnGeometry`, kept separate from `deformable()` so a rigid sibling
binding is not misclassified), and a `Stale` bound may never EXCLUDE a caster — `ShadowDrawFilter`
passes it through, and the caster-aware candidate test must do the same until storage geometry
carries a conservative envelope of its own. It is a prepass and not a read of the draw list for an ordering
reason that cannot be worked around — the fitted matrices decide the shadow frustums, and those
frustums are what the draw walk culls against, so a cascade finalised after draw collection would
leave the frame's matrices describing a different fit than its draws were selected for.

**The `Stale` rule for the depth policy, agreed before it is written.** "Stale cannot tighten" is not
the same as "ignore stale entries": fitting only the `Exact` casters can produce a range NARROWER
than one that covers the cloth, which clips it — the defect, arrived at from the other side. Stale
XY cannot even establish which cascade a cloth affects, so it cannot be excluded per view either.
Until storage geometry carries a conservative simulation or authored envelope, the honest interim
rule is: **if any stale caster exists in the frame, every directional cascade falls back to the
legacy depth fit**, marked as an unresolved correctness fallback rather than a policy. That mark
belongs in the pure depth-fit RESULT — a mode on `CascadeDepthFit`, not something the log or the
overlay reconstructs — so the displayed reason is tied to the matrix that was actually selected and
cannot drift from it. THREE values, not two: `fitLegacyCascadeDepth` still exists as its own
function and its direct result is neither caster-aware nor a fallback from anything, so it reports
`LegacyFixedExtension`. The policy reports `CasterAware` when it fitted the casters, and converts a
legacy result to `LegacyStaleFallback` only where stale geometry forced that choice. The probe scene contains only `Exact` casters, so its 26166 -> 35253 gate still validates the
new policy independently of that fallback.

**LANDED (2026-08-03): the caster-aware depth fit.** `fitCasterAwareCascadeDepth` takes the receiver
fit and the frame's caster record and places the planes where the geometry is: the near plane reaches
the furthest-upstream CANDIDATE caster (footprint not `Outside`, since light rays preserve U/V), the
far plane covers the receiver volume and no further (geometry behind every receiver in the slice
cannot shadow one), and one texel of the fit's own `worldPerTexel` is allowed as slack rather than an
invented epsilon. The matrix is built with the same `lookAt` / `ortho` calls as the legacy fit, so
the ONLY difference between the policies is where the planes sit.

The result carries its own mode — `LegacyFixedExtension` / `CasterAware` / `LegacyStaleFallback` —
so the log and the panel report the policy that produced the matrix rather than reconstructing it.

**Gate met.** `ShadowDepthClipDemo`, whose caster the legacy fit clipped:

| | shadow pixels | bounding box |
|---|---|---|
| legacy fixed extension | 26166 | 263 x 131 |
| caster-aware | 35324 | 305 x 152 |

Restored, and checked rather than asserted. Against the geometrically unclipped baseline (35253 px,
306 x 152 — the earlier probe placement, whose shadow sits at the same floor point because the caster
only moved along the light ray) the two differ by 253 pixels, grouped into 185 HORIZONTAL SCANLINE
RUNS (maximal spans of differing pixels within one image row) whose longest is 5 px and whose median
is 1. Every difference is therefore a one-pixel-wide fringe following the silhouette; there is no
clip-sized interior region, which is what residual clipping would leave — the legacy row above is
exactly that, a concentric 26% loss of area. The remaining fringe is soft-edge rasterisation under a
different depth range. Cascade 2's range on that scene goes from
`[-41.340, 33.535]` (span 74.9) to `[-45.740, 8.292]` (span 54.0): it reaches further back to catch
the caster while giving up the empty space behind the receivers.

The SH-03 budget calibration reproduces to every printed digit, as it must — the depth range does not
enter shadow-LOD selection.

Keep the stable receiver XY fit, then determine the Z range from candidate caster bounds:

1. Build/extract the receiver slice volume.
2. Query casters that can project into that volume along the directional light.
3. Fit conservative light-space near/far bounds over those casters and receivers.
4. Use the same candidate set for culling and diagnostics.

This removes a scene-scale magic distance and gives the cache workstream a well-defined caster set.
If an unbounded scene makes the query impractical, expose an authored directional-shadow distance
or scene shadow bounds rather than silently returning to a fixed hidden extension.

Likely branch: `shadow-cascade-caster-fit`.

#### SH-07: Derive bias and filtering from each fitted view

The receiver shader currently scales directional bias with `exp2(cascade)`, which assumes each
cascade's relevant scale doubles. Practical splits and bounding-sphere fits do not guarantee that.

Retain and upload per-view metrics produced while fitting:

- world units per texel;
- light-space depth span / world-to-depth scale; and
- effective filter radius.

Derive raster bias, receiver bias, and normal offset from those values. Apply the same principle to
spot/point maps, whose perspective depth precision and texel footprint vary with distance. Keep
units explicit (world units, texels, or normalised depth) so tuning one stage cannot unknowingly
double-apply a scale in another.

Only after the scale model is correct should filtering change. The Poisson comparison path exists
but is currently configured with zero radius. Evaluate a small, scale-consistent PCF kernel first;
then consider temporal/rotated sampling or PCSS only if the validation scene demonstrates a need.
Contact shadows should remain a short-range complement, not compensation for an under-resolved or
over-biased CSM.

Likely branch: `shadow-scale-derived-bias`.

### Milestone 3 — smooth/progressive shadow LOD, only if justified

#### SH-08: Shadow VIPM transitions

If SH-01 shows visible whole-mesh LOD transitions after per-view selection and hysteresis, extend
VIPM to the shadow vertex path before attempting VDPM:

- select the VIPM band from shadow-view texel error;
- render the finer cut while morphing toward the next cut;
- carry shadow-specific target level/factor per map iteration rather than reusing the forward
  `MorphUBO`; and
- use the same factor in both self-shadow depth passes.

This directly addresses popping with much less state and compute than a per-light active front.
Measure the added shadow vertex work, especially on skinned/morphed draws.

Likely branch: `shadow-vipm`.

#### SH-09: Shadow VDPM feasibility checkpoint

Do not reuse the camera front. Generalise VDPM only if diagnostics show that discrete/VIPM shadow
geometry remains a meaningful cost or quality limitation.

The scoring API would need to support a general raster view:

- orthographic directional/self-shadow projections as well as perspective spot/point projections;
- an eye direction for directional cone/silhouette tests rather than a finite camera position;
- geometry and silhouette channels by default;
- UV refinement only where an alpha mask makes UV placement part of the shadow silhouette; and
- shadow-view repair using the exact light `viewProj` and map extent.

Start with one **shadow-union front per instance** scored by the maximum requirement of every active
shadow view that intersects it. This deliberately uses one topology in all maps: it avoids a front
per cascade/face, bounds persistent state, and prevents cascade-to-cascade topology disagreement.
It may over-refine for far views, so compare it against per-view discrete/VIPM using SH-01's
triangle and timing data.

A per-view front is a later optimisation only if the union front is demonstrably too expensive. It
multiplies persistent state and compute by cascades/faces and complicates slot reuse, hysteresis,
cache identity, and self-shadow pairing.

Shadow-front compute must complete before the shadow draw consumes its emitted buffers, unlike the
current forward-front compute that overlaps the shadow pass. Treat the lost overlap as a measured
cost in the go/no-go decision; do not hide it with a guessed scheduling win.

Likely checkpoint branch: `shadow-vdpm-spike`. Land it only if it beats the discrete/VIPM baseline
without weakening the visual/error acceptance criteria.

## Interaction with the separate architectural-review actions

The §6 actions in [`architecturalreview.md`](architecturalreview.md) should consume the following
contracts from this work:

- **Shadow caching:** a map's content signature includes the shadow view descriptor, stable draw
  ids, caster transform/deformation/material revisions, proxy identity, and every selected
  LOD/front generation. A camera epoch alone is insufficient.
- **Compute pre-skinning:** expose the pre-deformed vertex buffer, exact deformed bounds, and a
  deformation revision through the shadow draw description. It can then replace the LOD0
  deformable fallback and stop rerunning skin/morph work in every pass.
- **Barrier batching:** group transitions by shadow-map family/layers without changing the
  per-view LOD decision.
- **Static/dynamic separation:** if caching later splits static and dynamic casters, both layers
  must use the same view and LOD/error contract; moved dynamic depth cannot simply remain in a
  persistent map.

Already-landed fixes—skipping the redundant world CSM when no skinned draw exists and skipping
unassigned self-shadow slots—remain valid under this design.

## Independent shadow hygiene

These small items are useful but need not block the LOD milestones:

- Make `noShadows` suppress shadow pass recording, not only sampling in the forward shader.
  Re-enabling is safe because the maps are rendered before their next sample.
- Do not render directional/world/self maps when there is no active primary directional light.
  Likewise, avoid clearing an unused punctual family merely because its texture exists.
- Remove manually repeated shader limits such as `SHADOW_TOTAL_MATRIX_COUNT = 32`,
  `SHADOW_POINT_MATRIX_BASE = 8`, and self-shadow slot count `4`. Generate a GLSL limits include
  from the C++ authority or make shader compilation validate both sides. A one-sided count change
  can otherwise compile cleanly and index the wrong UBO region.
- Preserve explicit map validity when skipping a map family. Sampling code should either know that
  the map is valid for this frame or return fully lit; it should never rely on stale depth being
  harmless by accident.

## Verification gates

Every implementation branch should run the normal build/fast suite. Branches that change rendering
must additionally:

- add headless tests for all new projection/selection math;
- run `clang-format` on edited C++ files;
- run the relevant Linux/macOS CI parity path;
- smoke-test with validation layers and confirm zero VUIDs;
- exercise rigid, skinned, morphed, cloth, alpha-mask, double-sided, spot, and point cases as
  applicable;
- compare fixed-pose full-detail and selected-LOD shadow masks;
- hold camera/light/object static long enough to prove the selected levels and shadow image do not
  flicker; and
- update `README.md`, `docs/onboarding.md`, `docs/review-order.md`, `docs/lod.md`,
  `docs/roadmap.md`, and `docs/acceptance-testing.md` according to the repository's documentation
  rules.

The key success criteria for Milestone 1 are:

1. The same caster may select different levels for different shadow views.
2. A fixed light/caster setup does not change shadow LOD when only an irrelevant camera quantity
   changes.
3. The selected level's declared projected shadow-deviation metric stays within the configured
   shadow-texel budget; it is called a guarantee only if the underlying cut metric is formally
   conservative.
4. Deformable or otherwise unbounded cases fall back visibly and safely to full detail.
5. No transition creates a silhouette hole, self-shadow first/second topology mismatch, or
   cascade-boundary flicker.

## Suggested priority

| Order | Item | Why |
|---:|---|---|
| 1 | ~~SH-01 diagnostics/scene~~ ✅ | Establishes evidence and prevents quality-by-anecdote. |
| 2 | ~~SH-02 pure projection model~~ ✅ | Makes the central policy testable before renderer plumbing. |
| 3 | ~~SH-03 per-view discrete LOD~~ ✅ | Fixes the requested architectural mismatch. |
| 4 | SH-04 deformation/proxy policy — **deformation half ✅**, proxy half open | Removes invalid error claims and defines safe extension points. |
| 5 | ~~SH-05 material-aware casters~~ ✅ | Fixed the cutout and two-sided silhouettes; the coarser masked-LOD policy stays open behind `AlphaMaskedFallback`. |
| 6 | ~~SH-06 cascade caster fit~~ ✅ | Removed fixed-depth clipping; candidate alignment (per-view filtering from the same record) is what remains, and SH-07 consumes it. |
| 7 | SH-07 scale-derived bias/filtering | Makes quality controls physically tied to each map. |
| 8 | SH-08 shadow VIPM | Add only if measured popping remains. |
| 9 | SH-09 shadow VDPM checkpoint | Highest complexity; require evidence before committing. |

This ordering makes “correct LOD” a small, independently reviewable foundation rather than coupling
it immediately to GPU-front scheduling, shadow caching, or a new filtering technique.

**Next up (2026-08-04): SH-07.** SH-05 and SH-06 both landed, so milestone 2 is complete except for
the follow-ups indexed in [`roadmap.md`](roadmap.md), and SH-07 is better positioned than the
ordering above implies: the per-view metrics it needs already come back out of SH-06's fit, and the
depth span is no longer a fixed constant. Four things stay open behind it and are indexed in the
roadmap rather than blocking it: the historical half-ellipse (unexplained, and NOT a depth clip —
measured), the cloth `LegacyStaleFallback` (needs a conservative envelope for storage geometry),
SH-04's proxy half, and SH-05's masked-LOD policy (which needs a cutout carrying a real LOD chain
before its pin can even be priced). The GPU-timestamp diagnostics SH-07's cost claims need are now
WORKING (branch `gpu-timestamp-diagnostics`, 2026-08-05): the readback treated `VK_NOT_READY` as a
failed read, so the panel said "unavailable" on every device — our bug, not MoltenVK's. Per-pass GPU
milliseconds are live in the overlay, which SH-07 should use rather than re-deriving cost.

**The post-merge sweep is done** (2026-08-04): SH-05 and SH-06 were each measured without the other
and move the same figures in opposite directions, so the table in `render/constants.hpp` is now the
MERGED measurement, and the notes above it keep both single-item runs as the reason the numbers moved
twice. Budget 1 and ratio 1.0 came through all three runs unchanged. The one number to read carefully
is the triangle column: SH-06's tighter depth range culls 9 of 52 candidate draws per frame, which is
a cull win and not a shadow-LOD one.
