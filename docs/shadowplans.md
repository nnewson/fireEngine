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

#### SH-01: Shadow diagnostics and a purpose-built validation scene

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
| The alpha-masked cutout casts a solid silhouette — `shadow.frag` samples no texture, so the mask has no effect on its shadow | **SH-05** |
| The double-sided sheet casts nothing at all: it is authored face-on to the sun, and the shadow pipeline fixes `cullMode = eFront` while the forward pass draws both sides of a double-sided material, so the shadow pass culls the only faces it has. The verdict is per-light — a punctual light on the quad's *back* side keeps exactly the faces the sun's view culls, so the same quad would cast a grazing sliver from it; the scene places both flat quads out of punctual reach so this second effect doesn't sit on top of the first | **SH-05** |
| Node scale (including the non-uniform caster) does not participate in a projected-texel policy, because there is no such policy yet | **SH-02** |

Likely branch: `shadow-lod-diagnostics`.

### Milestone 1 — correct discrete shadow LOD

#### SH-02: Pure shadow-view projection model

Extract/test the projection math before threading it through the renderer:

- a `ShadowView` value describing orthographic or perspective map layers;
- a per-cut shadow deviation metric distinct from the existing QEM RMS selection error;
- conservative world-error scaling under uniform, non-uniform, reflected, and singular transforms;
- nearest-depth calculation from world bounds for perspective views;
- `projectShadowErrorTexels`;
- `selectShadowLod`; and
- hysteresis state transitions.

Keep this in the Vulkan-free layer so it is headless-testable. Unit tests should prove:

- camera movement alone cannot alter a fixed light-view selection;
- cuts with a large local/cumulative deviation cannot hide behind a small RMS QEM value;
- doubling orthographic world-units-per-texel permits the expected coarser level;
- doubling object scale doubles projected error;
- increasing spot/point distance reduces projected error;
- near-plane intersection and invalid inputs choose LOD0;
- selection differs correctly between near and far cascades; and
- refine/coarsen thresholds do not chatter.

Likely branch: `shadow-view-lod-model`.

#### SH-03: Resolve LOD per shadow iteration

Change the command seam so `Object` emits an unresolved shadow geometry description. Resolve its
index buffer/count inside each directional cascade, world-only cascade, self-shadow slot, spot map,
and point face.

Important invariants:

- culling and LOD use the same `ShadowView` descriptor;
- full and world-only CSM use the same choice for a given rigid draw/cascade;
- both passes of one self-shadow slot use one choice;
- point-face selection uses the face actually being rendered;
- forward `lodLevel`, indirect buffers, and `vdpmGpuFront` are irrelevant to shadow recording; and
- the shadow LOD chain belongs to `shadowGeometry`, not necessarily the visible geometry.

The draw loop should resolve a buffer view without copying the whole command per pass. This is also
the right seam for future cache signatures: the selected geometry identity and level become explicit
inputs to shadow content.

Likely branch: `shadow-per-view-discrete-lod`.

#### SH-04: Conservative deformation and proxy policy

Until a deformation-aware error bound exists, force skinned and morphed shadow casters to LOD0.
Make the fallback visible in diagnostics.

Then define authored shadow proxies deliberately:

- a rigid proxy may have and select its own LOD chain;
- a skinned proxy must either share a compatible vertex/joint mapping or be rejected;
- a morphing proxy needs an explicit compatible morph contract; and
- proxy bounds, not visible-geometry bounds, drive per-view LOD and culling.

The existing `Object::shadowGeometry` setter is currently unused and does not encode these
compatibility rules. Do not start using it for deformable proxies until the rules are enforced at
load time.

Likely branch: `shadow-deformation-policy`.

### Milestone 2 — shadow silhouette correctness

#### SH-05: Material-aware caster pipelines

The shadow fragment path currently cannot apply the visible material's alpha mask, and the generic
shadow pipeline front-culls even double-sided materials. Add explicit shadow material modes:

- opaque single-sided;
- opaque double-sided;
- alpha-masked single-sided; and
- alpha-masked double-sided.

The mask path must sample base-colour alpha with the material's selected UV set,
`KHR_texture_transform`, sampler, factor, and `alphaCutoff`, then discard consistently with the
forward shader. Reuse the bindless material authority rather than constructing a parallel shadow
material format.

For alpha-masked LOD, begin at LOD0. A later coarser policy must consider UV/cutout silhouette error,
not only geometry error; VDPM's UV-deviation channel is a useful input, but it is not by itself a
proof that a binary alpha boundary is preserved. BLEND shadow semantics should be a separate design
decision (opaque/dithered/transmittance), not accidentally treated as MASK.

Likely branch: `shadow-material-casters`.

#### SH-06: Receiver/caster-aware cascade depth fitting

The cascade XY fit is stable, but its light-space depth currently relies on the fixed
`kShadowDepthBackExtend`. A caster farther behind the receiver slice than that constant can be
clipped even though its shadow reaches the slice.

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
| 1 | SH-01 diagnostics/scene | Establishes evidence and prevents quality-by-anecdote. |
| 2 | SH-02 pure projection model | Makes the central policy testable before renderer plumbing. |
| 3 | SH-03 per-view discrete LOD | Fixes the requested architectural mismatch. |
| 4 | SH-04 deformation/proxy policy | Removes invalid error claims and defines safe extension points. |
| 5 | SH-05 material-aware casters | Fixes visibly wrong cutout and two-sided silhouettes. |
| 6 | SH-06 cascade caster fit | Removes fixed-depth clipping and aligns candidate sets. |
| 7 | SH-07 scale-derived bias/filtering | Makes quality controls physically tied to each map. |
| 8 | SH-08 shadow VIPM | Add only if measured popping remains. |
| 9 | SH-09 shadow VDPM checkpoint | Highest complexity; require evidence before committing. |

This ordering makes “correct LOD” a small, independently reviewable foundation rather than coupling
it immediately to GPU-front scheduling, shadow caching, or a new filtering technique.
