#!/usr/bin/env python3
"""Generator for the shadow-LOD acceptance scenes (SH-01, docs/shadowplans.md).

Sample assets can't answer the questions the shadow-LOD arc asks. They rarely contain a
caster whose shadow is the only evidence of it, a mesh under non-uniform scale, casters
spread across cascade bands, and spot/point lights close enough to magnify a silhouette —
let alone all of those in one frame, at fixed poses, so two runs are comparable. So the
arc owns its scene.

Two files, deliberately separate:

- `ShadowLodDemo.gltf` — everything STATIC. This is the measurement baseline: the poses
  are authored, so overlay numbers and screenshots are reproducible run to run.
- `ShadowLodMotionDemo.gltf` — the same content plus animation (a moving caster, a
  swinging sun, a swinging skinned limb, a pulsing morph). This is for the qualitative
  "no chatter, no flicker" loop. It is NOT a screenshot reference: an animated frame has
  no reproducible timestamp, so a captured image proves nothing.

Both are checked structurally before they are written (see `validate`) — this asset is
dense enough that a later edit could quietly drop a whole exposure, and a scene silently
missing its alpha-mask cutout would make an SH-05 regression invisible rather than loud.

Run from anywhere:  python3 assets/shadow_lod/generate.py
"""

import math
import os
import sys
from pathlib import Path

# The shared glTF machinery lives in the repository's tools/ directory. Resolve it from
# this file, not the working directory, so the script keeps running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from assetgen import (  # noqa: E402  (path bootstrap must run first)
    Scene,
    checker_cutout_png,
    cylinder_geometry,
    quad_geometry,
    normalise,
    quat_axis_angle,
    quat_from_to,
    rotate_by_quat,
    sphere_geometry,
    spiky_sphere_geometry,
    subdivided_box_geometry,
    write_gltf,
)

GENERATOR = "fireEngine shadow_lod generate.py"

# --- authored constants ----------------------------------------------------
# The floor is wide enough to catch the off-camera caster's shadow and to span the
# cascade bands the far casters sit in.
FLOOR_HALF = 24.0
FLOOR_THICKNESS = 0.25
DETAIL_RADIUS = 1.0
# The comparison pose. Everything below is placed to be simultaneously visible from it,
# because a shadow you have to fly the camera to see is a shadow nobody checks.
CAMERA_EYE = (13.0, 7.5, 17.0)
CAMERA_TARGET = (0.0, 1.0, -1.0)

# glTF lights emit along their node's local -Z, so every light below is aimed with
# quat_from_to(-Z, direction) — the same construction the engine uses for its own default
# sun (Quaternion::fromVectors), rather than a hand-worked axis-angle that is easy to get
# a sign wrong in.
LIGHT_FORWARD = (0.0, 0.0, -1.0)
# The sun travels roughly WITH the view direction (dot(view, sun) ≈ +0.84), i.e. it sits
# over the camera's shoulder at about 46 degrees elevation. This is not a cosmetic choice:
# a sun shining back toward the camera leaves every camera-facing surface on its own dark
# side, which makes single-sided geometry look broken and makes lit-vs-shadowed impossible
# to judge by eye. Shadows fall away from the viewer, across open floor.
SUN_DIRECTION = (-0.75, -0.95, -0.55)
# The engine's own default sun (kDirectionalLightIntensity); an authored scene that lights
# itself differently from every other scene would just be misleading.
SUN_INTENSITY = 1.35

GREY = (0.62, 0.62, 0.64)
BLUE = (0.35, 0.55, 0.85)
ORANGE = (0.85, 0.5, 0.25)
GREEN = (0.4, 0.7, 0.4)
CREAM = (0.85, 0.82, 0.7)


def _translation_ibm(position):
    """Column-major inverse bind matrix for a joint bound at `position` with no rotation
    — i.e. the inverse of a pure translation."""
    x, y, z = position
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -x, -y, -z, 1.0]


def _skinned_limb_geometry(radius, height):
    """A capped cylinder standing at the origin, with per-vertex weights that bind the
    lower half to the root joint and the upper half to the tip joint.

    A cylinder rather than a box because a 12-triangle box is below the engine's
    512-triangle LOD threshold: it would report `SingleLevel` forever and could never show
    the missing deformable fallback SH-04 owns."""
    positions, normals, indices = cylinder_geometry(radius, height)
    joints, weights = [], []
    for p in positions:
        # A soft transition around the midpoint so the bend deforms rather than shears.
        t = max(0.0, min(1.0, (p[1] / height - 0.25) / 0.5))
        joints.append((0, 1, 0, 0))
        weights.append((1.0 - t, t, 0.0, 0.0))
    return positions, normals, indices, joints, weights


def _morph_bulge(positions, axis=0, amount=0.6):
    """Morph deltas that push the +axis face outward — a shape change big enough to see
    in a shadow silhouette, not a subtle wobble."""
    extent = max(p[axis] for p in positions)
    deltas, normal_deltas = [], []
    for p in positions:
        push = amount if p[axis] > extent * 0.5 else 0.0
        deltas.append(tuple(push if k == axis else 0.0 for k in range(3)))
        normal_deltas.append((0.0, 0.0, 0.0))
    return deltas, normal_deltas


def build_scene(animated):
    """The shared content. `animated` adds the motion channels and nothing else, so the
    two files describe the same geometry at the same poses."""
    s = Scene(GENERATOR)
    named = {}

    # --- receiver ---------------------------------------------------------
    # Plain renderable, no physics: this scene is about shadows, and an authored
    # extras.Physics here would drag the solver into an unrelated acceptance run.
    #
    # RECEIVE-ONLY. A wide flat caster writes its own depth into every cascade, so every
    # fragment of it fails its own depth comparison and the whole surface greys out —
    # taking every real shadow with it. The engine's built-in `-f` plane sets
    # Geometry::castsShadow(false) for exactly this reason; extras.Shadow.Casts is how an
    # authored scene says the same thing.
    named["Receiver_Floor"] = s.box(
        "Receiver_Floor", (FLOOR_HALF, FLOOR_THICKNESS, FLOOR_HALF),
        (0.0, -FLOOR_THICKNESS, 0.0), GREY, extras={"Shadow": {"Casts": False}})

    # --- a caster the camera cannot see ------------------------------------
    # ~74 degrees off the view axis (horizontal half-FOV is ~33), so it is never on screen,
    # yet tall enough that the sun throws its shadow across the visible floor near x 14,
    # z 11. Its shadow is the ONLY evidence it exists — exactly what a camera-derived
    # shadow-LOD rule gets wrong, since the camera never sees the caster to size it.
    named["OffCamera_Caster"] = s.mesh_node(
        "OffCamera_Caster", subdivided_box_geometry((0.6, 4.0, 2.5)), (17.0, 4.0, 13.0), ORANGE)

    # --- the same mesh at identity, non-uniform scale, and several distances -
    # ONE mesh, five nodes. Sharing it is the point: every difference between these
    # casters is a transform or a distance, never different geometry — so a per-view LOD
    # histogram that splits them apart is reporting the selection rule, not the content.
    # (It also keeps the embedded buffer small enough to commit.)
    detail_mesh = s.mesh(*spiky_sphere_geometry(DETAIL_RADIUS), s.material(BLUE), "DetailSphere")
    named["Detail_Identity"] = s.add_node("Detail_Identity", mesh=detail_mesh,
                                          translation=(-4.0, 1.3, 1.0))
    # Node scale (not a rescaled mesh), so the projected-error policy has to cope with a
    # non-uniform transform — SH-02 territory.
    named["Detail_NonUniform"] = s.add_node("Detail_NonUniform", mesh=detail_mesh,
                                            translation=(-1.0, 1.3, 1.0),
                                            scale=(2.2, 0.6, 1.0))
    for name, position in (("Cascade_Near", (2.5, 1.3, 5.0)),
                           ("Cascade_Mid", (5.0, 1.3, -9.0)),
                           ("Cascade_Far", (9.0, 1.3, -26.0))):
        named[name] = s.add_node(name, mesh=detail_mesh, translation=position)

    # --- punctual lights close enough to magnify a caster -------------------
    # One shared 768-triangle sphere for all three small casters. A box would be 12
    # triangles — below the engine's 512-triangle LOD threshold — so these casters could
    # never select a level, and the spot/point/motion cases would be measuring nothing.
    small_caster_mesh = s.mesh(*sphere_geometry(0.5), s.material(CREAM), "SmallCaster")
    named["Spot_Caster"] = s.add_node("Spot_Caster", mesh=small_caster_mesh,
                                      translation=(-8.0, 1.1, 6.0))
    named["Spot_Light"] = s.add_node(
        "Spot_Light",
        light=s.light("Spot", "spot", colour=(1.0, 0.95, 0.85), intensity=60.0, range_=20.0,
                      inner_cone=0.25, outer_cone=0.6),
        translation=(-8.0, 4.2, 6.0),
        rotation=quat_from_to(LIGHT_FORWARD, (0.0, -1.0, 0.0)))
    named["Point_Caster"] = s.add_node("Point_Caster", mesh=small_caster_mesh,
                                       translation=(7.0, 1.6, 7.0))
    named["Point_Light"] = s.add_node(
        "Point_Light",
        light=s.light("Point", "point", colour=(0.9, 0.9, 1.0), intensity=40.0, range_=14.0),
        translation=(7.0, 2.9, 7.0))

    # --- deformable casters (skinned + morphed) -----------------------------
    limb_positions, limb_normals, limb_indices, limb_joints, limb_weights = (
        _skinned_limb_geometry(0.35, 2.4))
    limb_root = (-11.0, 0.0, -1.0)
    joint_tip = s.add_node("Skinned_Joint_Tip", translation=(0.0, 1.2, 0.0))
    joint_root = s.add_node("Skinned_Joint_Root", translation=limb_root, children=[joint_tip])
    limb_mesh = s.mesh(limb_positions, limb_normals, limb_indices, s.material(ORANGE),
                       "Skinned_Caster", joints=limb_joints, weights=limb_weights)
    limb_skin = s.skin("Skinned_Caster_Skin", [joint_root, joint_tip],
                       [_translation_ibm(limb_root),
                        _translation_ibm((limb_root[0], limb_root[1] + 1.2, limb_root[2]))])
    named["Skinned_Caster"] = s.add_node("Skinned_Caster", mesh=limb_mesh, skin=limb_skin)

    morph_positions, morph_normals, morph_indices = subdivided_box_geometry((0.8, 0.8, 0.8))
    morph_mesh = s.mesh(morph_positions, morph_normals, morph_indices, s.material(BLUE),
                        "Morph_Caster", morph_targets=[_morph_bulge(morph_positions)],
                        morph_weights=[0.5])
    named["Morph_Caster"] = s.add_node("Morph_Caster", mesh=morph_mesh,
                                       translation=(-7.0, 1.1, -3.0))

    # --- alpha-masked cutout, with real UVs and a real alpha texture --------
    # alphaMode alone would prove nothing: without holes in the texture there is no
    # difference between a mask-aware and a mask-blind shadow pass.
    cutout_texture = s.texture(checker_cutout_png(), "CutoutAlpha")
    cutout_positions, cutout_normals, cutout_indices, cutout_uvs = quad_geometry(1.4, 1.4, 3.0)
    # Double-sided, and that is forced by the engine rather than a preference. For the quad
    # to CAST (the whole point — its shadow should show holes and doesn't), the sun must be
    # on its back side, since the shadow pass culls front faces. That puts the camera, which
    # shares the sun's side, on the back face too — and a single-sided quad would simply be
    # invisible from there. So a flat single-sided caster cannot be both visible and casting
    # in this engine today; that is itself an SH-05 note.
    cutout_material = s.material(CREAM, double_sided=True, alpha_mode="MASK", alpha_cutoff=0.5,
                                 base_colour_texture=cutout_texture)
    # Placed beyond every punctual light's reach (asserted in `validate`), so the sun is the
    # only light that can cast from it — see the sheet below for why that matters.
    named["Mask_Cutout"] = s.add_node(
        "Mask_Cutout",
        mesh=s.mesh(cutout_positions, cutout_normals, cutout_indices, cutout_material,
                    "Mask_Cutout", uvs=cutout_uvs),
        translation=(-1.0, 1.5, -8.0),
        # Upright, yawed so its normal points along the sun's travel — the light is then on
        # its back side, which is what survives the shadow pass's front-face cull. The
        # camera sees the flipped back face, lit.
        rotation=quat_from_to((0.0, 0.0, 1.0),
                              normalise((SUN_DIRECTION[0], 0.0, SUN_DIRECTION[2]))))

    # --- double-sided sheet -------------------------------------------------
    # A zero-thickness plane, aimed FACE-ON AT THE SUN. Two reasons the orientation is
    # computed rather than eyeballed:
    #
    # 1. A quad near edge-on to the light projects to a line, and a line-shaped shadow is
    #    indistinguishable from a missing one — the test would prove nothing either way.
    # 2. It makes the SH-05 exposure decisive. The shadow pipeline fixes
    #    `cullMode = eFront` (Pipeline::shadowConfig) while a double-sided material's
    #    forward pass draws both faces. With the quad's front face turned toward the
    #    light, the shadow pass culls the only faces it has, so today the sheet casts
    #    NOTHING while being plainly lit. When SH-05 makes the passes agree, the same
    #    node starts casting a full quad — the fix is visible without re-authoring.
    #
    # It is also placed OUT OF RANGE of every punctual light (asserted in `validate`).
    # Within range, the same cull rule flips: a punctual light on the quad's back side
    # keeps the faces the sun's view culls, so the sheet threw a grazing, hugely stretched
    # sliver away from the sun — a second, unrelated finding sitting on top of the one this
    # caster exists to show. One caster, one lesson.
    sheet_positions, sheet_normals, sheet_indices, sheet_uvs = quad_geometry(1.8, 1.2)
    named["DoubleSided_Sheet"] = s.add_node(
        "DoubleSided_Sheet",
        mesh=s.mesh(sheet_positions, sheet_normals, sheet_indices,
                    s.material(GREEN, double_sided=True), "DoubleSided_Sheet", uvs=sheet_uvs),
        translation=(3.5, 1.6, -12.0),
        # The quad's normal is +Z; point it back along the sun's travel direction.
        rotation=quat_from_to((0.0, 0.0, 1.0), tuple(-c for c in SUN_DIRECTION)))

    # --- the caster and light that move (animated file only) ----------------
    named["Moving_Caster"] = s.mesh_node("Moving_Caster", sphere_geometry(0.7), (0.0, 1.4, 8.0),
                                         ORANGE)
    named["Sun"] = s.add_node(
        "Sun",
        light=s.light("Sun", "directional", colour=(1.0, 0.97, 0.9), intensity=SUN_INTENSITY),
        rotation=quat_from_to(LIGHT_FORWARD, SUN_DIRECTION))

    named["Camera"] = s.camera(CAMERA_EYE, CAMERA_TARGET)

    if animated:
        # The sun swings by rotating its direction about Y, so it sweeps across the scene
        # rather than dipping under the floor.
        sun_swing = [
            quat_from_to(LIGHT_FORWARD,
                         (SUN_DIRECTION[0] * math.cos(a) - SUN_DIRECTION[2] * math.sin(a),
                          SUN_DIRECTION[1],
                          SUN_DIRECTION[0] * math.sin(a) + SUN_DIRECTION[2] * math.cos(a)))
            for a in (-0.5, 0.0, 0.5)
        ]
        s.animation(
            "ShadowStability",
            [
                # A caster crossing cascade boundaries: the classic level-chatter trigger.
                (named["Moving_Caster"], "translation", [0.0, 4.0, 8.0],
                 [(-9.0, 1.4, 8.0), (9.0, 1.4, 8.0), (-9.0, 1.4, 8.0)], "LINEAR"),
                # A swinging sun sweeps every cascade fit; shadows must not pop.
                (named["Sun"], "rotation", [0.0, 3.0, 6.0],
                 [sun_swing[0], sun_swing[1], sun_swing[2]], "LINEAR"),
                (joint_tip, "rotation", [0.0, 1.5, 3.0],
                 [quat_axis_angle((0.0, 0.0, 1.0), a) for a in (-0.6, 0.6, -0.6)], "LINEAR"),
                (named["Morph_Caster"], "weights", [0.0, 1.5, 3.0], [0.0, 1.0, 0.0], "LINEAR"),
            ])

    return s, named


# Nodes every variant must carry. Named, because a positional check would pass on a scene
# whose casters had all silently become the same box.
# The engine's `kMinLodTriangles` (include/fire_engine/graphics/lod.hpp). Meshes at or below
# it never get coarser levels built, so a caster made of one cannot exercise LOD selection.
MIN_LOD_TRIANGLES = 512

# Casters whose whole purpose is to show shadow-LOD SELECTION. Each must clear the threshold
# above; the floor, the lights and the flat SH-05 quads are deliberately not in this list —
# they are testing something else.
LOD_EXPOSURES = (
    "OffCamera_Caster", "Detail_Identity", "Detail_NonUniform", "Cascade_Near", "Cascade_Mid",
    "Cascade_Far", "Spot_Caster", "Point_Caster", "Skinned_Caster", "Morph_Caster",
    "Moving_Caster",
)

REQUIRED_NODES = (
    "Receiver_Floor", "OffCamera_Caster", "Detail_Identity", "Detail_NonUniform",
    "Cascade_Near", "Cascade_Mid", "Cascade_Far", "Spot_Caster", "Spot_Light",
    "Point_Caster", "Point_Light", "Skinned_Caster", "Morph_Caster", "Mask_Cutout",
    "DoubleSided_Sheet", "Moving_Caster", "Sun", "Camera",
)


def validate(doc, animated):
    """Structural checks on the finished document.

    Every one of these guards a specific coverage claim the acceptance runbook makes. A
    generated asset that quietly loses its cutout, its skin, or its point light would
    still load and still look plausible — and would stop testing the thing it exists for.
    Raises AssertionError with the missing item named."""
    nodes = doc["nodes"]
    names = {n.get("name") for n in nodes}
    for required in REQUIRED_NODES:
        assert required in names, f"missing required node '{required}'"

    lights = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    types = {light["type"] for light in lights}
    for wanted in ("directional", "spot", "point"):
        assert wanted in types, f"missing a {wanted} light"
    assert "KHR_lights_punctual" in doc.get("extensionsUsed", []), "lights used but not declared"

    # Skinning: a skin with at least a parent/child pair, bound to a mesh that really
    # carries per-vertex joints and weights.
    assert doc.get("skins"), "no skin — the deformable-caster exposure is gone"
    skin = doc["skins"][0]
    assert len(skin["joints"]) >= 2, "skin needs at least two joints to deform"
    assert "inverseBindMatrices" in skin, "skin has no inverse bind matrices"
    skinned = [n for n in nodes if "skin" in n]
    assert skinned, "no node uses the skin"
    skinned_attrs = doc["meshes"][skinned[0]["mesh"]]["primitives"][0]["attributes"]
    assert "JOINTS_0" in skinned_attrs and "WEIGHTS_0" in skinned_attrs, \
        "skinned mesh has no joints/weights"

    # Morph targets, with a non-zero default weight so the static file is deformed too.
    morphed = [m for m in doc["meshes"] if "weights" in m]
    assert morphed, "no morph-target mesh"
    assert any(w > 0.0 for w in morphed[0]["weights"]), \
        "morph weights are all zero — the static file would show the base shape"

    # Alpha mask: MASK + cutoff + a real texture + real UVs on the mesh that uses it.
    masks = [(i, m) for i, m in enumerate(doc["materials"]) if m.get("alphaMode") == "MASK"]
    assert masks, "no alpha-masked material"
    mask_index, mask_material = masks[0]
    assert "alphaCutoff" in mask_material, "MASK material without an explicit cutoff"
    assert "baseColorTexture" in mask_material["pbrMetallicRoughness"], \
        "MASK material without a texture — nothing would actually be discarded"
    assert doc.get("textures") and doc.get("images"), "no embedded texture"
    mask_meshes = [m for m in doc["meshes"]
                   if m["primitives"][0].get("material") == mask_index]
    assert mask_meshes, "the MASK material is unused"
    assert "TEXCOORD_0" in mask_meshes[0]["primitives"][0]["attributes"], \
        "MASK mesh has no UVs — the mask could not be sampled"

    # The double-sided sheet must be its OWN material: a colour-keyed material cache
    # would have merged it with an ordinary surface and erased the exposure.
    sheet_materials = [i for i, m in enumerate(doc["materials"])
                       if m.get("doubleSided") and m.get("alphaMode") is None]
    assert sheet_materials, "no double-sided opaque material"
    assert mask_index not in sheet_materials, "the cutout and the sheet share a material"

    by_name = {n.get("name"): n for n in nodes}

    # Every caster that is supposed to demonstrate LOD SELECTION must be able to. The engine
    # only builds coarser levels for meshes above `kMinLodTriangles` (512,
    # graphics/lod.hpp), and only for connected geometry — a fully split mesh is
    # boundary-locked everywhere and collapses to nothing. Below either bar the caster
    # reports `SingleLevel` forever, so the per-view, off-screen, motion-chatter and
    # deformable-fallback cases would all silently measure nothing. This is exactly the
    # failure a build and a clean VUID count cannot see.
    for exposure in LOD_EXPOSURES:
        mesh = doc["meshes"][by_name[exposure]["mesh"]]
        primitive = mesh["primitives"][0]
        triangles = doc["accessors"][primitive["indices"]]["count"] // 3
        vertices = doc["accessors"][primitive["attributes"]["POSITION"]]["count"]
        assert triangles > MIN_LOD_TRIANGLES, (
            f"'{exposure}' has {triangles} triangles, at or below the engine's "
            f"{MIN_LOD_TRIANGLES}-triangle LOD threshold; it can never select a level")
        assert vertices < triangles * 3, (
            f"'{exposure}' has {vertices} vertices for {triangles} triangles — fully split, so "
            "the simplifier is boundary-locked and builds no coarser levels")

    # The scene must be FRONT-LIT: the sun has to travel roughly with the view direction,
    # not toward the camera. Back-lit, every camera-facing surface shows its own dark side —
    # single-sided geometry reads as broken, and lit-vs-shadowed can't be judged by eye,
    # which defeats a visual acceptance scene.
    view = normalise(tuple(CAMERA_TARGET[i] - CAMERA_EYE[i] for i in range(3)))
    sun_travel = normalise(SUN_DIRECTION)
    front_lit = sum(a * b for a, b in zip(view, sun_travel))
    assert front_lit > 0.5, \
        f"sun travels against the view (dot {front_lit:+.2f}); the scene would be back-lit"

    # The off-camera caster must actually be off camera. It was not: an earlier placement sat
    # 22 degrees off the view axis, inside the ~33-degree horizontal half-FOV at 16:9, so a
    # large orange slab filled frame-left and the "its shadow is the only evidence it exists"
    # test never ran at all. The threshold is deliberately well past 33: the window's aspect
    # ratio is a runtime property (an ultrawide reaches ~40 degrees), and this caster is 5
    # units across, so its angular radius counts too.
    off_camera = by_name["OffCamera_Caster"]["translation"]
    to_caster = normalise(tuple(off_camera[i] - CAMERA_EYE[i] for i in range(3)))
    off_axis = math.degrees(
        math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(view, to_caster))))))
    assert off_axis > 60.0, \
        f"'OffCamera_Caster' is only {off_axis:.0f} deg off the view axis; it would be on screen"

    # Flat single-quad casters must not sit near edge-on to the sun. Such a quad projects
    # to a line, and a line-shaped shadow can't be told apart from a missing one — the
    # cutout and sheet would both look "tested" while proving nothing. 0.35 ≈ 20 degrees
    # off edge-on.
    sun = sun_travel
    for flat in ("Mask_Cutout", "DoubleSided_Sheet"):
        node = by_name[flat]
        normal = rotate_by_quat(node["rotation"], (0.0, 0.0, 1.0))
        facing = sum(a * b for a, b in zip(normal, sun))
        assert abs(facing) >= 0.35, \
            f"'{flat}' is {abs(facing):.3f} off edge-on to the sun; its shadow would be a line"

    # The two flat quads must be lit by the SUN ALONE. Inside a punctual light's reach the
    # shadow pass's fixed front-face cull flips its verdict — a light on the quad's back
    # side keeps exactly the faces the sun's view culls — so the quad throws an extra
    # grazing sliver away from the sun. That is a real finding, but a different one from
    # what these casters exist to show, and stacked on top of it, it reads as a bug.
    lights = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    punctual = [(n, lights[n["extensions"]["KHR_lights_punctual"]["light"]])
                for n in nodes
                if "KHR_lights_punctual" in n.get("extensions", {})
                and lights[n["extensions"]["KHR_lights_punctual"]["light"]]["type"] != "directional"]
    for flat in ("Mask_Cutout", "DoubleSided_Sheet"):
        position = by_name[flat]["translation"]
        # Half-diagonal of the larger quad, so "outside the cone" means the whole quad is,
        # not just its centre.
        extent = math.hypot(1.8, 1.4)
        for light_node, light in punctual:
            separation = math.dist(position, light_node["translation"])
            reach = light.get("range")
            assert reach is not None, f"punctual light '{light['name']}' has no range to check"
            if separation > reach + extent:
                continue  # beyond the light's reach entirely
            # A spot reaches only inside its cone, so distance alone would reject placements
            # that are in fact unlit. Compare the direction to the quad against the spot's
            # axis (glTF lights shine down local -Z), widened by the quad's angular size.
            assert light["type"] == "spot", (
                f"'{flat}' is {separation:.1f} from point light '{light['name']}' "
                f"(range {reach}); it would cast a second, unrelated shadow")
            axis = normalise(rotate_by_quat(light_node["rotation"], (0.0, 0.0, -1.0)))
            to_quad = normalise(tuple(position[i] - light_node["translation"][i]
                                      for i in range(3)))
            angle = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(axis, to_quad)))))
            half_width = math.asin(min(1.0, extent / max(separation, 1e-6)))
            outer = light["spot"]["outerConeAngle"]
            assert angle - half_width > outer, (
                f"'{flat}' sits {math.degrees(angle):.0f} deg off the axis of spot "
                f"'{light['name']}' (cone {math.degrees(outer):.0f} deg, distance "
                f"{separation:.1f}); it would cast a second, unrelated shadow")

    # Plain casters: this scene must not drag the physics solver in.
    for node in nodes:
        assert "Physics" not in node.get("extras", {}), \
            f"node '{node.get('name')}' carries extras.Physics"

    # The floor must be RECEIVE-ONLY, and it must be the only one. A casting floor greys
    # its whole surface out (it fails its own depth test) and swallows every shadow the
    # scene exists to show; a second receive-only node would be a caster silently lost.
    receive_only = [n.get("name") for n in nodes
                    if n.get("extras", {}).get("Shadow", {}).get("Casts") is False]
    assert receive_only == ["Receiver_Floor"], \
        f"expected exactly the floor to be receive-only, got {receive_only}"

    animations = doc.get("animations", [])
    if animated:
        assert animations, "the motion variant has no animation"
        paths = {c["target"]["path"] for a in animations for c in a["channels"]}
        for wanted in ("translation", "rotation", "weights"):
            assert wanted in paths, f"motion variant is missing a {wanted} channel"
    else:
        assert not animations, "the static variant must not animate — its poses are the baseline"

    # Every node reachable from the scene root list (directly or as a child).
    reachable = set(doc["scenes"][0]["nodes"])
    frontier = list(reachable)
    while frontier:
        for child in nodes[frontier.pop()].get("children", []):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    assert len(reachable) == len(nodes), \
        f"{len(nodes) - len(reachable)} node(s) unreachable from the scene"


VARIANTS = {
    "ShadowLodDemo": False,
    "ShadowLodMotionDemo": True,
}


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, animated in VARIANTS.items():
        scene, _ = build_scene(animated)
        doc = scene.to_gltf()
        validate(doc, animated)
        path = os.path.join(out_dir, f"{name}.gltf")
        kind = "animated" if animated else "static baseline"
        print(f"wrote {write_gltf(path, doc)}  ({len(scene.bin)} buffer bytes, {kind})")


if __name__ == "__main__":
    main()
