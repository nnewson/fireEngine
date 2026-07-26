#!/usr/bin/env python3
"""Generator for the P8 physics demonstration scenes.

Each physics capability (restitution, friction, stacking, toppling, convex hulls,
sleeping, static-mesh/compound colliders) gets its own minimal, self-contained
`.gltf`: simple untextured box/sphere geometry whose dimensions match its collider,
plus the `extras.Physics` authoring the engine reads (see collision.md
"glTF Authoring"). No textures, no external `.bin` — geometry is embedded as a
base64 data-URI buffer so every demo is a single file.

Run from anywhere:  python3 assets/physics_demos/generate.py
It (re)writes assets/physics_demos/*.gltf in place. New files are picked up by the
build after re-running `cmake ..` (copy_assets globs *.gltf at configure time).

This script owns only the SCENES. The geometry primitives, quaternion helpers and the
self-contained-glTF `Scene` assembler live in `tools/assetgen/`, shared with the other
generated-asset scripts; adding a demo here should not need to touch them.

Design notes:
- Geometry is authored at its true size and every node keeps scale = 1, with the
  collider given as an explicit `Shape` (Box/Sphere/...) whose params match the mesh.
  This keeps visual == collider == authored and sidesteps any node-scale vs collider
  interaction.
- Materials are double-sided (winding-agnostic) and colour-only.
- Headless replay tests in tests/physics/test_demos.cpp assert the resulting
  behaviour; the authored numbers below are the shared source of truth.
"""

import base64
import math
import os
import struct
import sys
from pathlib import Path

# The shared glTF machinery lives in the repository's tools/ directory. Resolve it from
# this file, not the working directory, so the script keeps running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from assetgen import (  # noqa: E402  (path bootstrap must run first)
    Scene,
    box_geometry,
    combine_geometry,
    quat_axis_angle,
    rotate_by_quat,
    tetrahedron_geometry,
    write_gltf,
)

# Recorded in every generated file's asset.generator. The committed .gltf carry this
# exact string, so changing it rewrites all of them.
GENERATOR = "fireEngine physics_demos generate.py"


def new_scene():
    """A Scene tagged with this script as the glTF generator."""
    return Scene(GENERATOR)


def _append_static_floor_box(doc, half_xz=4.0, thickness=0.25):
    """Append a Static box-floor (own data-URI buffer + mesh + node) to an existing glTF
    document; returns the new floor node index. Used by the ragdoll demos, which reuse an
    external skinned asset and needs ground to land on."""
    half = (half_xz, thickness, half_xz)
    positions, normals, indices = box_geometry(half)
    pos_b = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_b = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_b = b"".join(struct.pack("<H", i) for i in indices)
    blob = bytearray()

    def put(data):
        while len(blob) % 4 != 0:
            blob.append(0)
        off = len(blob)
        blob.extend(data)
        return off

    pos_off, nrm_off, idx_off = put(pos_b), put(nrm_b), put(idx_b)
    buf = len(doc["buffers"])
    doc["buffers"].append(
        {"byteLength": len(blob),
         "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()})
    bv = len(doc["bufferViews"])
    doc["bufferViews"] += [
        {"buffer": buf, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
        {"buffer": buf, "byteOffset": nrm_off, "byteLength": len(nrm_b), "target": 34962},
        {"buffer": buf, "byteOffset": idx_off, "byteLength": len(idx_b), "target": 34963}]
    mn = [min(p[k] for p in positions) for k in range(3)]
    mx = [max(p[k] for p in positions) for k in range(3)]
    ac = len(doc["accessors"])
    doc["accessors"] += [
        {"bufferView": bv, "componentType": 5126, "count": len(positions), "type": "VEC3",
         "min": mn, "max": mx},
        {"bufferView": bv + 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        {"bufferView": bv + 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"}]
    mat = len(doc["materials"])
    doc["materials"].append(
        {"name": "FloorMat", "doubleSided": True,
         "pbrMetallicRoughness": {"baseColorFactor": [0.45, 0.45, 0.48, 1.0],
                                  "metallicFactor": 0.0, "roughnessFactor": 0.7}})
    mesh = len(doc["meshes"])
    doc["meshes"].append(
        {"name": "Floor",
         "primitives": [{"attributes": {"POSITION": ac, "NORMAL": ac + 1}, "indices": ac + 2,
                         "material": mat}]})
    node = len(doc["nodes"])
    doc["nodes"].append(
        {"name": "Floor", "mesh": mesh, "translation": [0.0, -thickness, 0.0],
         "extras": {"Physics": {"BodyType": "Static", "Shape": "Box",
                                "HalfExtents": list(half), "Restitution": 0.0, "Friction": 0.6}}})
    return node


def build_single_joint_ragdoll_demo():
    """A one-joint skin tagged with extras.Ragdoll.

    This is the smallest legal ragdoll-authored scene: the loader builds one
    capsule body from the single skin joint and no parent-child constraints.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tilt = math.radians(10.0)
    tilt_z = math.sin(0.5 * tilt)
    tilt_w = math.cos(0.5 * tilt)
    d = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "SingleJointRagdoll", "skin": 0,
             "extras": {"Ragdoll": {"Mass": 1.0, "Radius": 0.05, "BoneLength": 0.4,
                                     "ConeTwist": False}}},
            {"name": "root_joint", "translation": [0.0, 1.6, 0.0],
             "rotation": [0.0, 0.0, tilt_z, tilt_w]},
        ],
        "skins": [{"name": "SingleJointRagdollSkin", "joints": [1]}],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "materials": [],
        "meshes": [],
    }

    d["nodes"][0]["children"] = [1]
    d["scenes"][0]["nodes"] = [0, _append_static_floor_box(d, half_xz=4.0)]
    out = os.path.join(here, "SingleJointRagdollDemo.gltf")
    print(f"wrote {write_gltf(out, d)}  (single-joint ragdoll + floor)")


def build_two_joint_ragdoll_demo():
    """A minimal two-joint skin tagged with extras.Ragdoll.

    This intentionally strips the ragdoll graph down to one parent/child constraint,
    so visual debugging can separate the basic joint-anchor behavior from the
    larger humanoid graph's branching and floor contacts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tilt = math.radians(3.0)
    tilt_z = math.sin(0.5 * tilt)
    tilt_w = math.cos(0.5 * tilt)
    d = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "TwoJointRagdoll", "skin": 0,
             "extras": {"Ragdoll": {"Mass": 1.0, "Radius": 0.05, "BoneLength": 0.4,
                                     "ConeTwist": False, "SwingLimit": 0.7,
                                     "TwistLimit": 0.5}}},
            {"name": "root_joint", "translation": [0.0, 1.6, 0.0],
             "rotation": [0.0, 0.0, tilt_z, tilt_w], "children": [2]},
            {"name": "tip_joint", "translation": [0.0, -0.4, 0.0]},
        ],
        "skins": [{"name": "TwoJointRagdollSkin", "joints": [1, 2]}],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "materials": [],
        "meshes": [],
    }

    d["nodes"][0]["children"] = [1]
    d["scenes"][0]["nodes"] = [0, _append_static_floor_box(d, half_xz=4.0)]
    out = os.path.join(here, "TwoJointRagdollDemo.gltf")
    print(f"wrote {write_gltf(out, d)}  (two-joint ragdoll + floor)")


def write_demo(name, scene):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, f"{name}.gltf")
    print(f"wrote {write_gltf(path, scene.to_gltf())}  ({len(scene.bin)} buffer bytes)")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_fall_rest():
    """Phase 0 smoke: a single Dynamic box falls and comes to rest on a Static floor."""
    s = new_scene()
    s.static_floor(half_xz=5.0)
    s.box_body(
        "FallingBox",
        (0.5, 0.5, 0.5),
        (0.0, 2.0, 0.0),
        (0.85, 0.3, 0.25),
        {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
         "Restitution": 0.0, "Friction": 0.5},
    )
    s.camera(eye=(5.0, 4.0, 7.0), target=(0.0, 1.0, 0.0))
    return s


def demo_restitution():
    """P2: three spheres with restitution 0.0 / 0.5 / 0.9 dropped from the same
    height bounce to visibly different rebound heights (rebound ~ restitution^2)."""
    s = new_scene()
    s.static_floor(half_xz=6.0)
    # Drop from y = 2 (1.5 m above the rest height): a clean restitution regime.
    # A much taller drop saturates because the speculative-margin CCD brakes the
    # fast approach (anti-tunnelling), which also suppresses the bounce.
    drops = [
        (-2.5, 0.0, (0.30, 0.30, 0.34)),  # dead — no bounce
        (0.0, 0.5, (0.30, 0.55, 0.85)),  # medium bounce
        (2.5, 0.9, (0.85, 0.55, 0.20)),  # lively bounce
    ]
    for x, restitution, colour in drops:
        s.sphere_body(
            f"Sphere_r{int(restitution * 100):02d}",
            0.5,
            (x, 2.0, 0.0),
            colour,
            {"BodyType": "Dynamic", "Shape": "Sphere", "Mass": 1.0,
             "Restitution": restitution, "Friction": 0.3},
        )
    s.camera(eye=(0.0, 3.5, 9.0), target=(0.0, 1.2, 0.0))
    return s


def demo_friction_ramp():
    """P2: two boxes on a tilted ramp — a high-friction box stays put while a
    low-friction box slides down (combined friction = sqrt(a*b), slides when the
    slope angle exceeds atan(mu))."""
    s = new_scene()
    # A rough floor so the slippery box grinds to a halt after sliding off the ramp
    # (combined friction is sqrt(a*b), so the floor must be rough to grip a low-
    # friction box) rather than sliding off the floor's far edge.
    s.static_floor(half_xz=8.0, friction=0.9)
    angle = math.radians(25.0)
    ramp_rot = quat_axis_angle((0.0, 0.0, 1.0), angle)  # tilt in the x-y plane
    ramp_half = (4.0, 0.15, 2.0)
    # Raise the ramp so its lower edge clears the floor — the slippery box slides off
    # the bottom and lands on the floor (rather than falling into the void).
    ramp_pos = (0.0, 2.5, 0.0)

    # Ramp: a Static slab, high friction so the per-box friction is the variable.
    s.box_body(
        "Ramp",
        ramp_half,
        ramp_pos,
        (0.40, 0.42, 0.48),
        {"BodyType": "Static", "Shape": "Box", "Restitution": 0.0, "Friction": 1.0},
        rotation=ramp_rot,
    )

    box_half = (0.4, 0.4, 0.4)
    # Local-space resting spot on the ramp's top face, offset up-slope (+local x) so
    # the boxes have room to slide toward down-slope (-local x).
    def surface_world(local):
        offset = rotate_by_quat(ramp_rot, local)
        return (ramp_pos[0] + offset[0], ramp_pos[1] + offset[1], ramp_pos[2] + offset[2])

    top_y = ramp_half[1] + box_half[1] + 0.02
    boxes = [
        ("StickyBox", (2.0, top_y, -1.0), 1.0, (0.30, 0.65, 0.35)),  # high friction → stays
        ("SlipperyBox", (2.0, top_y, 1.0), 0.08, (0.80, 0.30, 0.30)),  # low friction → slides
    ]
    for name, local, friction, colour in boxes:
        s.box_body(
            name,
            box_half,
            surface_world(local),
            colour,
            {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
             "Restitution": 0.0, "Friction": friction},
            rotation=ramp_rot,
        )
    s.camera(eye=(9.0, 5.0, 9.0), target=(-1.0, 1.8, 0.0))
    return s


def demo_stack():
    """P3: a 5-high tower of boxes dropped with small gaps settles into a resting stack
    and sleeps, instead of buzzing apart. The TGS soft-step solver (P9.2) propagates the
    settle down the stack through its substeps + relax pass, so a tall tower comes fully
    to rest in well under a second — the old fixed-iteration sequential-impulse solver
    quiesced a 5-high tower only after several seconds of shuffling (and diverged taller),
    which is why this demo was capped at three until P9."""
    s = new_scene()
    s.static_floor(half_xz=6.0)
    palette = [(0.80, 0.35, 0.30), (0.35, 0.55, 0.80), (0.80, 0.65, 0.30),
               (0.45, 0.75, 0.45), (0.70, 0.45, 0.75)]
    # Half 0.5 boxes; centres start at 0.55, 1.60, 2.65, ... (a small gap above the floor
    # top at y=0) so they fall a little and settle into contact at 0.5, 1.5, 2.5, ...
    for i in range(5):
        s.box_body(
            f"Box{i}",
            (0.5, 0.5, 0.5),
            (0.0, 0.55 + i * 1.05, 0.0),
            palette[i],
            {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
             "Restitution": 0.0, "Friction": 0.5},
        )
    s.camera(eye=(7.0, 5.0, 9.0), target=(0.0, 2.5, 0.0))
    return s


def demo_topple():
    """P3 headline: a tall box tilted past its balance angle topples onto its long
    side and comes to rest (full rotational dynamics — inertia + lever-arm torque)."""
    s = new_scene()
    # Friction 0.6 so the toppling box pivots on its edge rather than sliding.
    s.static_floor(half_xz=6.0, friction=0.6)
    box_half = (0.3, 1.0, 0.3)
    # Topple angle ~ atan(0.3/1.0) ~ 16.7 deg; a 30 deg tilt is safely past it.
    tilt = quat_axis_angle((0.0, 0.0, 1.0), math.radians(30.0))
    s.box_body(
        "TallBox",
        box_half,
        (0.0, 2.0, 0.0),
        (0.80, 0.50, 0.25),
        {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
         "Restitution": 0.0, "Friction": 0.6},
        rotation=tilt,
    )
    s.camera(eye=(5.0, 3.0, 6.0), target=(0.0, 1.0, 0.0))
    return s


def demo_convex_hull():
    """P3.5: tetrahedra (clearly not primitives) dropped onto the floor exercise the
    GJK/EPA convex narrowphase + face-clip manifold — they tumble, land on a face, and
    settle into a loose pile. Spread out in x so they rest mostly side by side rather
    than in a precarious tower (tall piles sit near the solver's stability margin —
    see demo_stack)."""
    s = new_scene()
    s.static_floor(half_xz=6.0)
    geo = tetrahedron_geometry(0.6)
    phys = {"BodyType": "Dynamic", "Mass": 1.0, "Restitution": 0.0, "Friction": 0.5}
    drops = [
        ("Tetra0", (-1.6, 2.0, 0.2), quat_axis_angle((1.0, 0.0, 0.0), 0.3), (0.80, 0.45, 0.30)),
        ("Tetra1", (0.0, 3.0, -0.2), quat_axis_angle((0.0, 0.0, 1.0), 0.5), (0.35, 0.60, 0.80)),
        ("Tetra2", (1.6, 2.4, 0.3), quat_axis_angle((1.0, 1.0, 0.0), 0.4), (0.55, 0.75, 0.40)),
    ]
    for name, pos, rot, colour in drops:
        s.convex_body(name, geo, pos, colour, phys, rotation=rot)
    s.camera(eye=(5.0, 3.5, 7.0), target=(0.0, 0.7, 0.0))
    return s


def demo_sleep():
    """P5: a small stack settles and goes to sleep (frozen — with --debug-physics its
    colliders dim to the asleep colour), then a striker flies in horizontally (no
    gravity) and wakes the island on impact, scattering it. Three boxes so the stack
    sleeps quickly and cleanly (see demo_stack on taller towers)."""
    s = new_scene()
    s.static_floor(half_xz=8.0)
    palette = [(0.80, 0.35, 0.30), (0.35, 0.55, 0.80), (0.80, 0.65, 0.30)]
    for i in range(3):
        s.box_body(
            f"Box{i}",
            (0.5, 0.5, 0.5),
            (0.0, 0.55 + i * 1.05, 0.0),
            palette[i],
            {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
             "Restitution": 0.0, "Friction": 0.5},
        )
    # Striker: a low-friction box sliding in along the floor. It reaches the stack
    # after the stack has slept, bumps it awake, then friction brings the striker to
    # rest against the stack and it too sleeps — so the whole scene ends asleep on the
    # floor (a gravity-free striker would just coast off-screen forever).
    s.box_body(
        "Striker",
        (0.5, 0.5, 0.5),
        (-8.0, 0.5, 0.0),
        (0.95, 0.55, 0.15),
        {"BodyType": "Dynamic", "Shape": "Box", "Mass": 2.0, "Restitution": 0.1,
         "Friction": 0.1, "Velocity": [6.0, 0.0, 0.0]},
    )
    s.camera(eye=(5.0, 3.0, 11.0), target=(-2.0, 0.8, 0.0))
    return s


# Trapezoidal valley: a wide flat bottom (z in [-3, 3] at y=0) with gentle slopes rising
# to walls at z=+-5. One-sided triangles wound so every normal faces up into the valley.
# Bodies are dropped low onto the flat bottom (a steep narrow trough makes them slosh /
# a sphere roll for many seconds; a wide flat bottom settles cleanly).
_VALLEY_VERTS = [
    (-7.0, 1.5, -5.0), (7.0, 1.5, -5.0),  # 0,1 back wall top
    (-7.0, 0.0, -3.0), (7.0, 0.0, -3.0),  # 2,3 bottom near
    (-7.0, 0.0, 3.0), (7.0, 0.0, 3.0),    # 4,5 bottom far
    (-7.0, 1.5, 5.0), (7.0, 1.5, 5.0),    # 6,7 front wall top
]
_VALLEY_TRIS = [
    (2, 3, 0), (0, 3, 1),  # left slope (normal up +z)
    (4, 5, 2), (2, 5, 3),  # flat bottom (normal +y)
    (5, 4, 6), (5, 6, 7),  # right slope (normal up -z)
]


def demo_static_mesh():
    """P6: a triangulated valley (Static Shape:"Mesh" — a triangle-mesh collider, not a
    box) catches dropped boxes and a sphere; they land on the mesh surface and settle.
    Proves contacts are generated against the mesh's actual triangles."""
    s = new_scene()
    s.static_mesh_body("Valley", _VALLEY_VERTS, _VALLEY_TRIS, (0.40, 0.42, 0.48), friction=0.6)
    box = {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0, "Restitution": 0.0, "Friction": 0.5}
    s.box_body("Box0", (0.4, 0.4, 0.4), (-3.0, 1.3, -1.5), (0.80, 0.40, 0.30), box)
    s.box_body("Box1", (0.4, 0.4, 0.4), (3.0, 1.3, 1.5), (0.35, 0.55, 0.80), box)
    s.sphere_body(
        "Ball", 0.4, (0.0, 1.3, 0.0), (0.80, 0.65, 0.30),
        {"BodyType": "Dynamic", "Shape": "Sphere", "Mass": 1.0, "Restitution": 0.1,
         "Friction": 0.4})
    s.camera(eye=(9.0, 5.0, 9.0), target=(0.0, 0.4, 0.0))
    return s


# L-shaped compound: a horizontal bar + a vertical upright at its left end. The two
# children and the matching visual mesh share these dimensions/offsets.
_L_BAR_HALF = (1.2, 0.4, 0.4)
_L_BAR_POS = (0.0, 0.0, 0.0)
_L_UPRIGHT_HALF = (0.4, 1.0, 0.4)
_L_UPRIGHT_POS = (-0.8, 1.0, 0.0)


def demo_compound():
    """P6: an L-shaped Dynamic body whose collider is a Compound of two boxes. Its true
    centre of mass is offset toward the corner (volume-weighted), computed by the engine
    — so it rests stably on its bar instead of tipping (a naive origin-as-COM would get
    the balance wrong)."""
    s = new_scene()
    s.static_floor(half_xz=6.0)
    geo = combine_geometry([
        (box_geometry(_L_BAR_HALF), _L_BAR_POS),
        (box_geometry(_L_UPRIGHT_HALF), _L_UPRIGHT_POS),
    ])
    children = [
        {"Shape": "Box", "HalfExtents": list(_L_BAR_HALF), "Position": list(_L_BAR_POS)},
        {"Shape": "Box", "HalfExtents": list(_L_UPRIGHT_HALF), "Position": list(_L_UPRIGHT_POS)},
    ]
    s.compound_body(
        "LBlock", geo, (0.0, 2.0, 0.0), (0.75, 0.55, 0.35), children,
        {"BodyType": "Dynamic", "Mass": 3.0, "Restitution": 0.0, "Friction": 0.5})
    s.camera(eye=(5.0, 3.0, 6.0), target=(-0.3, 0.8, 0.0))
    return s


DEMOS = {
    "FallRestDemo": demo_fall_rest,
    "RestitutionDemo": demo_restitution,
    "FrictionRampDemo": demo_friction_ramp,
    "StackDemo": demo_stack,
    "ToppleDemo": demo_topple,
    "ConvexHullDemo": demo_convex_hull,
    "SleepDemo": demo_sleep,
    "StaticMeshDemo": demo_static_mesh,
    "CompoundDemo": demo_compound,
}


def main():
    for name, builder in DEMOS.items():
        write_demo(name, builder())
    build_single_joint_ragdoll_demo()
    build_two_joint_ragdoll_demo()


if __name__ == "__main__":
    main()
