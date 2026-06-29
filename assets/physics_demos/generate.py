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
import json
import math
import os
import struct

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

# Unit cube faces: (outward normal, [4 corners CCW when viewed from outside]).
# Corner coords are in [-1, 1]; scaled by the half-extents per box.
_CUBE_FACES = [
    ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
    ((-1, 0, 0), [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)]),
    ((0, 1, 0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
    ((0, -1, 0), [(-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1)]),
    ((0, 0, 1), [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]),
    ((0, 0, -1), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
]


def box_geometry(half):
    """24-vertex box (per-face normals) centred at the origin. `half` = (hx, hy, hz)."""
    hx, hy, hz = half
    positions, normals, indices = [], [], []
    for normal, corners in _CUBE_FACES:
        base = len(positions)
        for cx, cy, cz in corners:
            positions.append((cx * hx, cy * hy, cz * hz))
            normals.append(normal)
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return positions, normals, indices


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalise(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


# Regular tetrahedron at alternating cube corners (centroid at origin); the four
# outward CCW faces. Built flat-shaded (3 verts per face, per-face normal).
_TETRA_CORNERS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
_TETRA_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]


def tetrahedron_geometry(scale):
    corners = [(x * scale, y * scale, z * scale) for x, y, z in _TETRA_CORNERS]
    positions, normals, indices = [], [], []
    for tri in _TETRA_FACES:
        p0, p1, p2 = (corners[tri[0]], corners[tri[1]], corners[tri[2]])
        n = _normalise(_cross(_sub(p1, p0), _sub(p2, p0)))
        base = len(positions)
        for p in (p0, p1, p2):
            positions.append(p)
            normals.append(n)
        indices += [base, base + 1, base + 2]
    return positions, normals, indices


def mesh_from_triangles(vertices, triangles):
    """Flat-shaded geometry (3 verts per triangle, per-face normal) from a vertex
    list + triangle index triples. Used for static triangle-mesh colliders."""
    positions, normals, indices = [], [], []
    for tri in triangles:
        p = [vertices[i] for i in tri]
        n = _normalise(_cross(_sub(p[1], p[0]), _sub(p[2], p[0])))
        base = len(positions)
        for q in p:
            positions.append(q)
            normals.append(n)
        indices += [base, base + 1, base + 2]
    return positions, normals, indices


def combine_geometry(parts):
    """Concatenate (positions, normals, indices) geometries, each translated by its
    offset — builds one mesh for a multi-box compound's visual."""
    positions, normals, indices = [], [], []
    for geo, offset in parts:
        gp, gn, gi = geo
        base = len(positions)
        for p in gp:
            positions.append((p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]))
        normals += gn
        indices += [base + i for i in gi]
    return positions, normals, indices


def sphere_geometry(radius, stacks=16, slices=24):
    """UV sphere centred at the origin (normals = normalised position)."""
    positions, normals, indices = [], [], []
    for i in range(stacks + 1):
        v = i / stacks
        phi = v * math.pi
        for j in range(slices + 1):
            u = j / slices
            theta = u * 2.0 * math.pi
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            normals.append((nx, ny, nz))
            positions.append((nx * radius, ny * radius, nz * radius))
    ring = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * ring + j
            b = a + ring
            indices += [a, b, a + 1, a + 1, b, b + 1]
    return positions, normals, indices


# ---------------------------------------------------------------------------
# glTF assembly
# ---------------------------------------------------------------------------


class Scene:
    """Accumulates meshes/materials/nodes into a single self-contained glTF dict."""

    def __init__(self):
        self.bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.materials = []
        self.nodes = []
        self.cameras = []
        self._material_cache = {}

    def _align(self):
        while len(self.bin) % 4 != 0:
            self.bin.append(0)

    def _add_view(self, data, target):
        self._align()
        offset = len(self.bin)
        self.bin += data
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
        )
        return len(self.buffer_views) - 1

    def _add_accessor(self, view, comp_type, count, acc_type, mn=None, mx=None):
        acc = {"bufferView": view, "componentType": comp_type, "count": count, "type": acc_type}
        if mn is not None:
            acc["min"] = mn
            acc["max"] = mx
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def material(self, colour):
        """Colour-only double-sided material; deduped by RGBA."""
        key = tuple(round(c, 4) for c in colour)
        if key in self._material_cache:
            return self._material_cache[key]
        r, g, b = colour
        idx = len(self.materials)
        self.materials.append(
            {
                "name": f"Mat{idx}",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [r, g, b, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
            }
        )
        self._material_cache[key] = idx
        return idx

    def mesh(self, positions, normals, indices, material_index, name="Mesh"):
        pos_bytes = bytearray()
        for p in positions:
            pos_bytes += struct.pack("<3f", *p)
        nrm_bytes = bytearray()
        for n in normals:
            nrm_bytes += struct.pack("<3f", *n)
        idx_bytes = bytearray()
        for i in indices:
            idx_bytes += struct.pack("<H", i)

        pos_view = self._add_view(pos_bytes, 34962)
        nrm_view = self._add_view(nrm_bytes, 34962)
        idx_view = self._add_view(idx_bytes, 34963)

        mn = [min(p[k] for p in positions) for k in range(3)]
        mx = [max(p[k] for p in positions) for k in range(3)]
        pos_acc = self._add_accessor(pos_view, 5126, len(positions), "VEC3", mn, mx)
        nrm_acc = self._add_accessor(nrm_view, 5126, len(normals), "VEC3")
        idx_acc = self._add_accessor(idx_view, 5123, len(indices), "SCALAR")

        idx = len(self.meshes)
        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": material_index,
                    }
                ],
            }
        )
        return idx

    def node(self, name, mesh_index, position, physics=None, rotation=None):
        node = {"name": name, "mesh": mesh_index, "translation": list(position)}
        if rotation is not None:
            node["rotation"] = list(rotation)
        if physics is not None:
            node["extras"] = {"Physics": physics}
        self.nodes.append(node)
        return len(self.nodes) - 1

    # --- convenience: a box/sphere body with matching geometry + collider ---

    def box_body(self, name, half, position, colour, physics, rotation=None):
        positions, normals, indices = box_geometry(half)
        mesh = self.mesh(positions, normals, indices, self.material(colour), name)
        cfg = dict(physics)
        if cfg.get("Shape", "Box") == "Box" and "HalfExtents" not in cfg:
            cfg["HalfExtents"] = list(half)
        return self.node(name, mesh, position, cfg, rotation)

    def sphere_body(self, name, radius, position, colour, physics):
        positions, normals, indices = sphere_geometry(radius)
        mesh = self.mesh(positions, normals, indices, self.material(colour), name)
        cfg = dict(physics)
        if cfg.get("Shape", "Sphere") == "Sphere" and "Radius" not in cfg:
            cfg["Radius"] = radius
        return self.node(name, mesh, position, cfg)

    def convex_body(self, name, geometry, position, colour, physics, rotation=None):
        """A Dynamic body whose collider is a ConvexHull built from its own mesh
        (the loader runs buildConvexHull on the node geometry)."""
        positions, normals, indices = geometry
        mesh = self.mesh(positions, normals, indices, self.material(colour), name)
        cfg = dict(physics)
        cfg["Shape"] = "ConvexHull"
        return self.node(name, mesh, position, cfg, rotation)

    def static_mesh_body(self, name, vertices, triangles, colour, friction=0.6):
        """A Static body whose collider is a triangle mesh built from its own geometry
        (Shape: "Mesh"). Vertices are in world space; the node sits at the origin."""
        geo = mesh_from_triangles(vertices, triangles)
        mesh = self.mesh(*geo, self.material(colour), name)
        return self.node(
            name, mesh, (0.0, 0.0, 0.0),
            {"BodyType": "Static", "Shape": "Mesh", "Restitution": 0.0, "Friction": friction})

    def compound_body(self, name, geometry, position, colour, children, physics, rotation=None):
        """A Dynamic body whose collider is a Compound of child boxes; `geometry` is the
        matching visual mesh and `children` the authored Compound child list."""
        positions, normals, indices = geometry
        mesh = self.mesh(positions, normals, indices, self.material(colour), name)
        cfg = dict(physics)
        cfg["Shape"] = "Compound"
        cfg["Children"] = children
        return self.node(name, mesh, position, cfg, rotation)

    def static_floor(self, name="Floor", half_xz=6.0, thickness=0.25,
                     colour=(0.45, 0.45, 0.48), friction=0.5):
        """A wide thin Static box whose top face sits at y = 0."""
        # Restitution 0 so the floor never imposes bounce: contact restitution is
        # combined as max(a, b), so the per-body restitution stays the controlling
        # value (the default would be 1.0 and make everything bouncy).
        half = (half_xz, thickness, half_xz)
        return self.box_body(
            name,
            half,
            (0.0, -thickness, 0.0),
            colour,
            {"BodyType": "Static", "Shape": "Box", "Restitution": 0.0, "Friction": friction},
        )

    def camera(self, eye, target, up=(0.0, 1.0, 0.0), yfov=0.7, name="Camera"):
        cam_index = len(self.cameras)
        self.cameras.append(
            {
                "name": name,
                "type": "perspective",
                "perspective": {"yfov": yfov, "znear": 0.05, "zfar": 500.0},
            }
        )
        rot = _look_at_quat(eye, target, up)
        self.nodes.append(
            {"name": name, "camera": cam_index, "translation": list(eye), "rotation": list(rot)}
        )
        return len(self.nodes) - 1

    def to_gltf(self):
        uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(self.bin)).decode(
            "ascii"
        )
        doc = {
            "asset": {"version": "2.0", "generator": "fireEngine physics_demos generate.py"},
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.bin), "uri": uri}],
        }
        if self.cameras:
            doc["cameras"] = self.cameras
        return doc


def quat_axis_angle(axis, angle):
    """Quaternion [x,y,z,w] for a rotation of `angle` radians about a (unit-ish) axis."""
    m = math.sqrt(sum(c * c for c in axis)) or 1.0
    ax, ay, az = (axis[0] / m, axis[1] / m, axis[2] / m)
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))


def rotate_by_quat(q, v):
    """Rotate vec3 v by quaternion q = [x,y,z,w]."""
    qx, qy, qz, qw = q
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    # v + qw * t + cross(q.xyz, t)
    return (
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    )


def _look_at_quat(eye, target, up):
    """Quaternion [x,y,z,w] orienting a glTF camera (looks down -Z) from eye to target."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def normalise(v):
        m = math.sqrt(sum(c * c for c in v)) or 1.0
        return (v[0] / m, v[1] / m, v[2] / m)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    z = normalise(sub(eye, target))  # camera +Z points away from the target
    x = normalise(cross(up, z))
    y = cross(z, x)
    # Rotation matrix columns (x, y, z) -> quaternion.
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return (qx, qy, qz, w)


def _append_static_floor_box(doc, half_xz=4.0, thickness=0.25):
    """Append a Static box-floor (own data-URI buffer + mesh + node) to an existing glTF
    document; returns the new floor node index. Used by the ragdoll demo, which reuses an
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


def build_ragdoll_demo():
    """P4: reuse the skinned CesiumMan humanoid, author extras.Ragdoll on it, lift it
    above a floor, and let the ragdoll (capsule body + cone-twist joint per bone)
    activate at load — the figure drops and crumples onto the floor. The generator can't
    emit a skinned mesh, so this rewrites the existing asset rather than building one.

    DEFERRED (not called from main()): a complex ragdoll never settles with the current
    solver — it sits in a joint-driven limit cycle (see roadmap P9 B2). Re-enable this
    (add the main() call back) once P9's stability state machine lets it come to rest."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "CesiumMan", "CesiumMan.gltf")) as f:
        d = json.load(f)
    # Character data/image stay external, resolved relative to physics_demos/.
    d["buffers"][0]["uri"] = "../CesiumMan/CesiumMan_data.bin"
    if d.get("images"):
        d["images"][0]["uri"] = "../CesiumMan/CesiumMan_img0.jpg"
    # Drop the walk animation so it doesn't fight the ragdoll's world-override drive.
    d.pop("animations", None)
    # Ragdoll on the skinned node (built from skin 0's joints at load).
    skinned = next(i for i, n in enumerate(d["nodes"]) if "skin" in n)
    d["nodes"][skinned].setdefault("extras", {})["Ragdoll"] = {
        "Mass": 1.0, "Radius": 0.06, "BoneLength": 0.15,
        "ConeTwist": True, "SwingLimit": 0.7, "TwistLimit": 0.5}
    # Lift the whole figure above the floor (node 0 has a matrix, so wrap it).
    scene = d["scenes"][d.get("scene", 0)]
    root = scene["nodes"][0]
    lift = len(d["nodes"])
    d["nodes"].append({"name": "RagdollLift", "translation": [0.0, 1.8, 0.0], "children": [root]})
    scene["nodes"] = [lift, _append_static_floor_box(d)]
    out = os.path.join(here, "RagdollDemo.gltf")
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(out)}  (from CesiumMan + ragdoll + floor)")


def write_demo(name, scene):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, f"{name}.gltf")
    with open(path, "w") as f:
        json.dump(scene.to_gltf(), f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(path)}  ({len(scene.bin)} buffer bytes)")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_fall_rest():
    """Phase 0 smoke: a single Dynamic box falls and comes to rest on a Static floor."""
    s = Scene()
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
    s = Scene()
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
    s = Scene()
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
    """P3: a tower of boxes dropped with small gaps settles into a resting stack
    and stays still (warm-started impulse solver, then sleeps) instead of buzzing
    apart. Three boxes — a taller tower quiesces much more slowly because the
    fixed-iteration sequential-impulse solver propagates the settle down the stack
    gradually (the bottom box bears the whole load), so a 5-high tower shuffles for
    several seconds before sleeping; three settles in ~1 s."""
    s = Scene()
    s.static_floor(half_xz=6.0)
    palette = [(0.80, 0.35, 0.30), (0.35, 0.55, 0.80), (0.80, 0.65, 0.30)]
    # Half 0.5 boxes; centres start at 0.55, 1.60, 2.65 (a small gap above the floor
    # top at y=0) so they fall a little and settle into contact at 0.5, 1.5, 2.5.
    for i in range(3):
        s.box_body(
            f"Box{i}",
            (0.5, 0.5, 0.5),
            (0.0, 0.55 + i * 1.05, 0.0),
            palette[i],
            {"BodyType": "Dynamic", "Shape": "Box", "Mass": 1.0,
             "Restitution": 0.0, "Friction": 0.5},
        )
    s.camera(eye=(6.0, 4.0, 8.0), target=(0.0, 2.5, 0.0))
    return s


def demo_topple():
    """P3 headline: a tall box tilted past its balance angle topples onto its long
    side and comes to rest (full rotational dynamics — inertia + lever-arm torque)."""
    s = Scene()
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
    s = Scene()
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
    s = Scene()
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
    s = Scene()
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
    s = Scene()
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
    # build_ragdoll_demo()  # deferred until roadmap P9 item 2 (TGS substeps) — single-step
    # soft joints settle a representative ragdoll but not the real CesiumMan skeleton robustly.


if __name__ == "__main__":
    main()
