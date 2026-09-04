#!/usr/bin/env python3
"""Generator for the shadow-residency gate scene (arc 2 #4, docs/shadowplans.md).

`ShadowResidencyTest.gltf` exists to answer ONE question: does a shadow view whose content
has not changed stop rasterising, and does the image it keeps still shade correctly? Every
choice below is in service of that, and most of them are about removing evidence that would
muddy the answer rather than about looking good.

- **One static point light, and no authored directional or spot.** A point light's fit does
  not depend on the camera, so its six faces are the family that reuses reliably — the case
  worth measuring.

  **The scene still renders cascades, and that is not a defect.** When an asset authors no
  directional light the engine seeds a default sun (`src/fire_engine.cpp`), so this scene
  gets four cascades whether or not it asks for them; authoring nothing does NOT make the
  cascade family ineligible. It does no harm to the measurement — each family carries its
  own timestamp span, so the point family's cost is never mixed with the cascades' — and
  with the camera parked the cascades reuse too, which is extra evidence rather than noise.
  What the scene DOES avoid is a second punctual light: a spot would add a seventh view
  competing for the same GPU with no separate story to tell.

  If the fallback sun ever becomes suppressible, this scene should suppress it — not because
  the numbers are wrong today, but because a gate is easier to trust when the frame contains
  only what the gate is about.
- **A closed room, so all six cube faces have a receiver.** Six slabs rather than six
  quads: a slab's inner surface is real front-facing geometry, so nothing here depends on
  double-sided shading or on winding the reader has to check.
- **One caster per axis, between the light and the wall it shades.** All six faces
  therefore do real work, and each casts a hard-edged rectangle a capture comparison can
  actually see. A face with nothing in it would reuse trivially and prove nothing.
- **Nothing moves.** No animation, no skin, no morph target — a deformable caster can
  never be cached (SH-04), and a moving one changes the content every frame, which is the
  opposite of what this scene is for. Run it with `--no-taa`: temporal accumulation makes
  two captures of the same state differ.

The gate procedure that uses this lives in docs/acceptance-testing.md.

Run from anywhere:  python3 assets/shadow_residency/generate.py
"""

import sys
from pathlib import Path

# The shared glTF machinery lives in the repository's tools/ directory. Resolve it from
# this file, not the working directory, so the script keeps running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from assetgen import Scene, write_gltf  # noqa: E402  (path bootstrap must run first)

GENERATOR = "fireEngine shadow_residency generate.py"

# --- authored constants ----------------------------------------------------
# The room's interior half-extent. Large enough that the light's six faces each see a wall
# at a useful distance, small enough that one point light of a sane range lights all of it.
ROOM_HALF = 8.0
WALL_THICKNESS = 0.4
# The light sits at the origin, so the six cube faces are symmetric and each gets the same
# amount of work — which is what makes the per-face timings comparable to each other.
LIGHT_POSITION = (0.0, 0.0, 0.0)
LIGHT_RANGE = 22.0
LIGHT_INTENSITY = 26.0
# Casters sit between the light and the wall they shade, closer to the light than to the
# wall so the shadow is magnified and its edges are unmistakable in a capture.
CASTER_DISTANCE = 3.0
CASTER_HALF = 0.5
# Inside the room, off-axis, looking back through the origin: three walls, the floor and
# four of the six casters are in frame at once. Deliberately NOT on an axis — a camera on
# one would hide the caster in front of it behind its own shadow.
CAMERA_EYE = (6.2, 3.6, 6.6)
CAMERA_TARGET = (-2.2, -1.4, -2.4)

WALL_COLOUR = (0.52, 0.51, 0.49)
FLOOR_COLOUR = (0.40, 0.40, 0.43)
CASTER_COLOUR = (0.85, 0.35, 0.25)

# The six axis directions, named. Each contributes one wall and one caster, which is what
# makes "every cube face has both a receiver and an occluder" true by construction rather
# than by a placement someone has to re-check.
AXES = (
    ("PosX", (1.0, 0.0, 0.0)),
    ("NegX", (-1.0, 0.0, 0.0)),
    ("PosY", (0.0, 1.0, 0.0)),
    ("NegY", (0.0, -1.0, 0.0)),
    ("PosZ", (0.0, 0.0, 1.0)),
    ("NegZ", (0.0, 0.0, -1.0)),
)


def wall_half_extent(axis):
    """A slab spanning the room in the two axes it is not normal to."""
    return tuple(
        WALL_THICKNESS * 0.5 if component else ROOM_HALF + WALL_THICKNESS
        for component in axis
    )


def scaled(axis, distance):
    return tuple(component * distance for component in axis)


def build():
    s = Scene(GENERATOR)

    # The light first, so it is light 0 and the scene's only one.
    s.add_node(
        "PointLight",
        light=s.light(
            "Point",
            "point",
            colour=(1.0, 0.96, 0.9),
            intensity=LIGHT_INTENSITY,
            range_=LIGHT_RANGE,
        ),
        translation=LIGHT_POSITION,
    )

    for name, axis in AXES:
        # The wall sits a half-thickness OUTSIDE the interior, so the interior surface is
        # exactly at ROOM_HALF and the geometry never intrudes on the casters.
        wall_centre = scaled(axis, ROOM_HALF + WALL_THICKNESS * 0.5)
        colour = FLOOR_COLOUR if name == "NegY" else WALL_COLOUR
        s.box(f"Wall{name}", wall_half_extent(axis), wall_centre, colour)
        # One occluder per face, on the axis between the light and that wall.
        s.box(
            f"Caster{name}",
            (CASTER_HALF, CASTER_HALF, CASTER_HALF),
            scaled(axis, CASTER_DISTANCE),
            CASTER_COLOUR,
        )

    s.camera(CAMERA_EYE, CAMERA_TARGET)
    return s


def validate(doc):
    """Structural checks, because this scene's whole value is what it does NOT contain.

    A later edit that adds a sun, or animates a caster to make a screenshot livelier, would
    not break anything visibly — it would quietly turn the gate into a measurement of
    something else. Each assertion below corresponds to a claim the gate makes.
    """
    lights = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    assert len(lights) == 1, f"the gate needs exactly one light, found {len(lights)}"
    assert lights[0]["type"] == "point", (
        f"the gate measures the POINT family; found a {lights[0]['type']} light"
    )
    assert "range" in lights[0], "a point light without a range has no radial depth to store"

    # Temporal or deforming content would change the content descriptor every frame, so a
    # reused view could never happen and the gate would fail for the wrong reason.
    assert not doc.get("animations"), "the residency gate scene must not animate"
    assert not doc.get("skins"), "the residency gate scene must not contain skinned casters"
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            assert "targets" not in primitive, (
                f"'{mesh['name']}' carries morph targets; a morph-capable caster is "
                "Deformable (SH-04) and can never be reused"
            )

    names = [node["name"] for node in doc["nodes"]]
    for axis_name, _ in AXES:
        assert f"Wall{axis_name}" in names, f"no receiver for the {axis_name} cube face"
        assert f"Caster{axis_name}" in names, f"no occluder for the {axis_name} cube face"

    # The camera must be INSIDE the room, or the capture is of six slabs from outside and
    # every shadow the gate is about is hidden.
    assert all(abs(component) < ROOM_HALF for component in CAMERA_EYE), (
        f"camera {CAMERA_EYE} is outside the room (interior half-extent {ROOM_HALF})"
    )
    # And it must not be inside a caster, which would fill the frame with one box.
    for _, axis in AXES:
        centre = scaled(axis, CASTER_DISTANCE)
        inside = all(
            abs(eye - c) <= CASTER_HALF for eye, c in zip(CAMERA_EYE, centre)
        )
        assert not inside, f"camera {CAMERA_EYE} is inside the caster at {centre}"

    # Every caster has to be strictly between the light and its wall: touching either would
    # make its shadow degenerate (no penumbra to see) or clip into the receiver.
    assert CASTER_HALF < CASTER_DISTANCE < ROOM_HALF - CASTER_HALF, (
        "casters must sit clear of both the light and the walls"
    )
    # The furthest interior corner must be inside the light's range, or the faces nearest it
    # store a clamped ratio and the capture shows an unlit wedge.
    corner_distance = (3.0 ** 0.5) * ROOM_HALF
    assert LIGHT_RANGE > corner_distance, (
        f"light range {LIGHT_RANGE} does not reach the room's corner at {corner_distance:.1f}"
    )


def main():
    scene = build()
    doc = scene.to_gltf()
    validate(doc)
    out = Path(__file__).resolve().parent / "ShadowResidencyTest.gltf"
    write_gltf(out, doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
