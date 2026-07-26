"""Primitive geometry builders for generated glTF assets.

Every builder returns the same `(positions, normals, indices)` triple the `Scene`
assembler consumes, with positions/normals as 3-tuples of floats and indices as a flat
list of `uint16` triangle indices. Meshes are authored at their true size with node
scale left at 1, so a visual mesh and its collider (or its shadow silhouette) describe
the same shape.

Extracted verbatim from `assets/physics_demos/generate.py` so a second generator can
reuse it; the physics demos' output is byte-for-byte unchanged by the move.
"""

import math

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalise(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


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


# Regular tetrahedron at alternating cube corners (centroid at origin); the four
# outward CCW faces. Built flat-shaded (3 verts per face, per-face normal).
_TETRA_CORNERS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
_TETRA_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]


def tetrahedron_geometry(scale):
    corners = [(x * scale, y * scale, z * scale) for x, y, z in _TETRA_CORNERS]
    positions, normals, indices = [], [], []
    for tri in _TETRA_FACES:
        p0, p1, p2 = (corners[tri[0]], corners[tri[1]], corners[tri[2]])
        n = normalise(vec_cross(vec_sub(p1, p0), vec_sub(p2, p0)))
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
        n = normalise(vec_cross(vec_sub(p[1], p[0]), vec_sub(p[2], p[0])))
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


def check_winding(positions, normals, indices, label="geometry"):
    """Assert every triangle winds consistently with its authored vertex normals.

    A face wound backwards still SHADES correctly — the forward pass lights it from the
    authored normal — so this defect is invisible until the shadow pass, which culls by
    winding (`cullMode = eFront`, Pipeline::shadowConfig). An inverted face then survives
    culling where it should be dropped and vice versa, which shows up as self-shadow acne,
    visible triangle edges, and holes punched out of ground shadows. It cost a full debugging
    round on `subdivided_box_geometry`, where four of six faces were inverted.

    Degenerate triangles (zero area) are skipped rather than failed: their normal is
    undefined, and cones/fans legitimately produce them at poles."""
    for tri in range(len(indices) // 3):
        i0, i1, i2 = indices[tri * 3], indices[tri * 3 + 1], indices[tri * 3 + 2]
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        face = vec_cross(vec_sub(p1, p0), vec_sub(p2, p0))
        if sum(c * c for c in face) < 1.0e-18:
            continue
        # Compare against the average of the three vertex normals: a smooth-shaded mesh has
        # per-vertex normals that only approximate the face, so the test is a sign check, not
        # an equality.
        avg = tuple((normals[i0][k] + normals[i1][k] + normals[i2][k]) / 3.0 for k in range(3))
        if sum(c * c for c in avg) < 1.0e-18:
            continue
        if sum(a * b for a, b in zip(normalise(face), normalise(avg))) <= 0.0:
            raise AssertionError(
                f"{label}: triangle {tri} (verts {i0},{i1},{i2}) winds against its authored "
                f"normal — the face is inside-out and the shadow pass will cull the wrong side")


def subdivided_box_geometry(half, segments=8):
    """A box whose faces are `segments`x`segments` grids of shared vertices.

    `box_geometry` returns 12 triangles, which is below the engine's `kMinLodTriangles`
    (512) threshold — a caster built from it can never select a level, so it silently
    reports `SingleLevel` and measures nothing. At the default 8 segments this is 768
    triangles with connected interiors the simplifier can actually collapse.

    Vertices are shared within a face but not across face boundaries, which is what keeps
    the per-face normals hard: the six faces meet at genuine creases."""
    hx, hy, hz = half
    positions, normals, indices = [], [], []
    # (origin corner, edge-u vector, edge-v vector, outward normal) per face. The triangles
    # below wind u-then-v, so cross(edge_u, edge_v) MUST equal the authored normal — get the
    # order backwards and the face is inside-out while still claiming to face outward. That
    # reads as correct in the forward pass (which shades from the authored normal) and goes
    # wrong only in the shadow pass, which culls by winding: `check_winding` now enforces it.
    faces = [
        ((hx, -hy, -hz), (0, 2 * hy, 0), (0, 0, 2 * hz), (1, 0, 0)),
        ((-hx, -hy, hz), (0, 2 * hy, 0), (0, 0, -2 * hz), (-1, 0, 0)),
        ((-hx, hy, -hz), (0, 0, 2 * hz), (2 * hx, 0, 0), (0, 1, 0)),
        ((-hx, -hy, hz), (0, 0, -2 * hz), (2 * hx, 0, 0), (0, -1, 0)),
        ((-hx, -hy, hz), (2 * hx, 0, 0), (0, 2 * hy, 0), (0, 0, 1)),
        ((hx, -hy, -hz), (-2 * hx, 0, 0), (0, 2 * hy, 0), (0, 0, -1)),
    ]
    for origin, edge_u, edge_v, normal in faces:
        base = len(positions)
        for row in range(segments + 1):
            v = row / segments
            for col in range(segments + 1):
                u = col / segments
                positions.append(tuple(origin[k] + edge_u[k] * u + edge_v[k] * v
                                       for k in range(3)))
                normals.append(normal)
        stride = segments + 1
        for row in range(segments):
            for col in range(segments):
                a = base + row * stride + col
                b = a + 1
                c = a + stride
                d = c + 1
                indices += [a, b, d, a, d, c]
    return positions, normals, indices


def cylinder_geometry(radius, height, radial_segments=24, height_segments=12):
    """A capped cylinder standing on the origin, growing along +Y.

    Shared vertices along and around the shaft (the caps are separate fans, so the rim
    stays a hard edge). At the defaults this is 720 triangles — above the engine's
    512-triangle LOD threshold, unlike a 12-triangle box — and its height segments give a
    skinned limb something to deform smoothly."""
    positions, normals, indices = [], [], []
    for row in range(height_segments + 1):
        y = height * row / height_segments
        for col in range(radial_segments + 1):
            theta = 2.0 * math.pi * col / radial_segments
            nx, nz = math.cos(theta), math.sin(theta)
            positions.append((nx * radius, y, nz * radius))
            normals.append((nx, 0.0, nz))
    stride = radial_segments + 1
    for row in range(height_segments):
        for col in range(radial_segments):
            a = row * stride + col
            b = a + 1
            c = a + stride
            d = c + 1
            indices += [a, c, d, a, d, b]

    # `flip` reverses the fan winding for the DOWNWARD-facing cap. Going round the rim in
    # increasing theta gives a +Y triangle normal, which is correct for the top cap and
    # inside-out for the bottom one.
    for y, normal, flip in ((0.0, (0.0, -1.0, 0.0), False), (height, (0.0, 1.0, 0.0), True)):
        centre = len(positions)
        positions.append((0.0, y, 0.0))
        normals.append(normal)
        rim = len(positions)
        for col in range(radial_segments):
            theta = 2.0 * math.pi * col / radial_segments
            positions.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
            normals.append(normal)
        for col in range(radial_segments):
            first = rim + col
            second = rim + (col + 1) % radial_segments
            indices += [centre, second, first] if flip else [centre, first, second]
    return positions, normals, indices


def quad_geometry(half_width, half_height, uv_repeat=1.0):
    """A single flat quad in the XY plane facing +Z, with UVs. Returns
    `(positions, normals, indices, uvs)` — the extra channel is why this doesn't share
    the 3-tuple shape: a textured or alpha-masked surface is meaningless without UVs."""
    positions = [(-half_width, -half_height, 0.0), (half_width, -half_height, 0.0),
                 (half_width, half_height, 0.0), (-half_width, half_height, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    uvs = [(0.0, uv_repeat), (uv_repeat, uv_repeat), (uv_repeat, 0.0), (0.0, 0.0)]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, indices, uvs


# Icosahedron, used as the seed for the subdivided spiky sphere below.
_ICO_T = (1.0 + math.sqrt(5.0)) / 2.0
_ICO_VERTS = [
    (-1, _ICO_T, 0), (1, _ICO_T, 0), (-1, -_ICO_T, 0), (1, -_ICO_T, 0),
    (0, -1, _ICO_T), (0, 1, _ICO_T), (0, -1, -_ICO_T), (0, 1, -_ICO_T),
    (_ICO_T, 0, -1), (_ICO_T, 0, 1), (-_ICO_T, 0, -1), (-_ICO_T, 0, 1),
]
_ICO_FACES = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


def spiky_sphere_geometry(radius, subdivisions=3, spike=0.35):
    """A subdivided icosahedron with every third vertex pushed outward — a dense mesh
    with a RECOGNISABLE silhouette, so a shadow drawn at the wrong level shows up as a
    changed outline rather than a slightly different blob.

    Vertices are SHARED between triangles (normals are the radial direction), not split
    per face. That matters for LOD: the engine's simplifier collapses edges, and a
    fully-split flat-shaded mesh is boundary-locked everywhere, so it would build no
    coarser levels at all — the caster would report `SingleLevel` and measure nothing.

    The default 3 subdivisions gives 1280 triangles / 642 vertices, comfortably above the
    engine's `kMinLodTriangles` (512) threshold and inside the uint16 index range.
    `spike` is the fraction of `radius` added to the displaced vertices; 0 gives a plain
    sphere."""
    verts = [normalise(v) for v in _ICO_VERTS]
    faces = list(_ICO_FACES)
    midpoints = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoints:
            va, vb = verts[a], verts[b]
            verts.append(normalise(((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2,
                                    (va[2] + vb[2]) / 2)))
            midpoints[key] = len(verts) - 1
        return midpoints[key]

    for _ in range(subdivisions):
        next_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            next_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = next_faces

    # Deterministic spikes: every third vertex in creation order, so the shape is stable
    # across runs (a random or position-hashed choice would not be).
    positions, normals = [], []
    for i, v in enumerate(verts):
        r = radius * (1.0 + spike) if i % 3 == 0 else radius
        positions.append((v[0] * r, v[1] * r, v[2] * r))
        normals.append(v)  # radial: already unit length
    indices = [i for face in faces for i in face]
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
            # Wound so the face normal agrees with the outward vertex normals. This was
            # inverted for as long as the sphere existed — invisible in the forward pass,
            # which shades from the authored normals, and wrong only in the shadow pass,
            # which culls by winding. `check_winding` now catches this class of defect.
            indices += [a, a + 1, b, a + 1, b + 1, b]
    return positions, normals, indices
