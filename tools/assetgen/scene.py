"""Self-contained glTF 2.0 scene assembler.

`Scene` accumulates meshes/materials/nodes/cameras into one glTF document whose binary
data is embedded as a base64 data-URI, so a generated asset is a single `.gltf` file
with no sidecar `.bin` and no textures. `write_gltf` is the matching writer.

The API has two layers. `node`/`mesh_node`/`box`/`sphere` are generic: they build a
renderable node and pass `extras` through verbatim, which is how a generator authors the
engine-specific blocks (`extras.Physics`, `extras.Ragdoll`, `extras.Cloth` — see
`docs/collision.md`) without this module knowing what any of them mean. The `*_body`
helpers are a thin physics layer on top: they inject `extras.Physics` and default the
collider to match the mesh they just built. A caster that needs no physics uses the
generic layer and carries no `extras` at all.

Extracted verbatim from `assets/physics_demos/generate.py`, except that the `generator`
string is now a constructor argument instead of a hard-coded physics-demo literal — a
shared module must not claim to be one specific script.
"""

import base64
import json
import os
import struct

from .geometry import box_geometry, check_winding, mesh_from_triangles, sphere_geometry
from .quaternions import look_at_quat


class Scene:
    """Accumulates meshes/materials/nodes into a single self-contained glTF dict."""

    def __init__(self, generator):
        self.generator = generator
        self.bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.materials = []
        self.nodes = []
        self.cameras = []
        self.skins = []
        self.animations = []
        self.images = []
        self.samplers = []
        self.textures = []
        self.lights = []
        self._material_cache = {}
        # Nodes claimed as someone's child. The scene's root list is every node MINUS
        # these, so a joint hierarchy isn't also parented to the scene (which would
        # double-transform it).
        self._child_nodes = set()

    def _align(self):
        while len(self.bin) % 4 != 0:
            self.bin.append(0)

    def _add_view(self, data, target=None):
        self._align()
        offset = len(self.bin)
        self.bin += data
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            # Only vertex/index data carries a target; animation and inverse-bind data
            # must not (the spec reserves the hint for buffer bindings).
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _add_accessor(self, view, comp_type, count, acc_type, mn=None, mx=None):
        acc = {"bufferView": view, "componentType": comp_type, "count": count, "type": acc_type}
        if mn is not None:
            acc["min"] = mn
            acc["max"] = mx
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def material(self, colour, *, double_sided=True, alpha_mode=None, alpha_cutoff=None,
                 base_colour_texture=None, alpha=1.0):
        """A colour (optionally textured) material, deduped.

        The cache key covers EVERY property, not just the colour: an alpha-masked cutout
        and an ordinary opaque surface can share an RGB triple, and a colour-only key
        would silently hand the second one the first one's material — quietly deleting
        the alpha-mask coverage from a scene built to exercise it."""
        key = (
            tuple(round(c, 4) for c in colour),
            round(alpha, 4),
            bool(double_sided),
            alpha_mode,
            None if alpha_cutoff is None else round(alpha_cutoff, 4),
            base_colour_texture,
        )
        if key in self._material_cache:
            return self._material_cache[key]
        r, g, b = colour
        idx = len(self.materials)
        pbr = {
            "baseColorFactor": [r, g, b, alpha],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.6,
        }
        if base_colour_texture is not None:
            pbr["baseColorTexture"] = {"index": base_colour_texture, "texCoord": 0}
        mat = {"name": f"Mat{idx}", "doubleSided": bool(double_sided), "pbrMetallicRoughness": pbr}
        if alpha_mode is not None:
            mat["alphaMode"] = alpha_mode
        if alpha_cutoff is not None:
            mat["alphaCutoff"] = alpha_cutoff
        self.materials.append(mat)
        self._material_cache[key] = idx
        return idx

    def texture(self, png_bytes, name="Texture"):
        """Embed a PNG as a data-URI image + sampler + texture; returns the texture index."""
        image_index = len(self.images)
        self.images.append(
            {
                "name": name,
                "mimeType": "image/png",
                "uri": "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii"),
            }
        )
        if not self.samplers:
            # One shared sampler: linear min/mag, repeat wrap. Nothing generated so far
            # needs a second filtering mode, and a per-texture sampler would just be noise.
            self.samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497,
                                  "wrapT": 10497})
        self.textures.append({"name": name, "sampler": 0, "source": image_index})
        return len(self.textures) - 1

    def mesh(self, positions, normals, indices, material_index, name="Mesh", *, uvs=None,
             joints=None, weights=None, morph_targets=None, morph_weights=None):
        """One primitive from parallel attribute lists.

        `uvs` are TEXCOORD_0 pairs (required for any textured or alpha-masked material —
        an alphaMode without real UVs exercises nothing). `joints`/`weights` are 4-wide
        tuples for a skinned mesh. `morph_targets` is a list of `(position_deltas,
        normal_deltas)` per target, with `morph_weights` the mesh's default weights."""
        # Checked for EVERY mesh, from every generator, before a byte is written: a face wound
        # against its normal is invisible in the forward pass and only corrupts shadows, so it
        # must be caught here rather than in someone's screenshot.
        check_winding(positions, normals, indices, name)

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

        attributes = {"POSITION": pos_acc, "NORMAL": nrm_acc}
        if uvs is not None:
            uv_bytes = bytearray()
            for uv in uvs:
                uv_bytes += struct.pack("<2f", *uv)
            attributes["TEXCOORD_0"] = self._add_accessor(
                self._add_view(uv_bytes, 34962), 5126, len(uvs), "VEC2")
        if joints is not None:
            joint_bytes = bytearray()
            for j in joints:
                joint_bytes += struct.pack("<4H", *j)
            attributes["JOINTS_0"] = self._add_accessor(
                self._add_view(joint_bytes, 34962), 5123, len(joints), "VEC4")
        if weights is not None:
            weight_bytes = bytearray()
            for w in weights:
                weight_bytes += struct.pack("<4f", *w)
            attributes["WEIGHTS_0"] = self._add_accessor(
                self._add_view(weight_bytes, 34962), 5126, len(weights), "VEC4")

        primitive = {"attributes": attributes, "indices": idx_acc, "material": material_index}
        if morph_targets:
            targets = []
            for target_positions, target_normals in morph_targets:
                delta_pos = bytearray()
                for p in target_positions:
                    delta_pos += struct.pack("<3f", *p)
                delta_nrm = bytearray()
                for n in target_normals:
                    delta_nrm += struct.pack("<3f", *n)
                # Morph POSITION deltas need min/max like any position accessor.
                tmn = [min(p[k] for p in target_positions) for k in range(3)]
                tmx = [max(p[k] for p in target_positions) for k in range(3)]
                targets.append(
                    {
                        "POSITION": self._add_accessor(self._add_view(delta_pos, 34962), 5126,
                                                       len(target_positions), "VEC3", tmn, tmx),
                        "NORMAL": self._add_accessor(self._add_view(delta_nrm, 34962), 5126,
                                                     len(target_normals), "VEC3"),
                    }
                )
            primitive["targets"] = targets

        idx = len(self.meshes)
        mesh = {"name": name, "primitives": [primitive]}
        if morph_targets:
            mesh["weights"] = list(morph_weights if morph_weights is not None
                                   else [0.0] * len(morph_targets))
        self.meshes.append(mesh)
        return idx

    def skin(self, name, joint_nodes, inverse_bind_matrices=None):
        """A skin over `joint_nodes` (node indices). `inverse_bind_matrices` is one
        column-major 16-float list per joint; omitted means identity binds."""
        skin = {"name": name, "joints": list(joint_nodes)}
        if inverse_bind_matrices is not None:
            data = bytearray()
            for matrix in inverse_bind_matrices:
                data += struct.pack("<16f", *matrix)
            skin["inverseBindMatrices"] = self._add_accessor(
                self._add_view(data), 5126, len(inverse_bind_matrices), "MAT4")
        self.skins.append(skin)
        return len(self.skins) - 1

    def light(self, name, light_type, colour=(1.0, 1.0, 1.0), intensity=1.0, range_=None,
              inner_cone=None, outer_cone=None):
        """A KHR_lights_punctual light definition (attach it with `add_node(light=...)`).
        `light_type` is "directional", "point", or "spot"."""
        light = {"name": name, "type": light_type, "color": list(colour), "intensity": intensity}
        if range_ is not None:
            light["range"] = range_
        if light_type == "spot":
            spot = {}
            if inner_cone is not None:
                spot["innerConeAngle"] = inner_cone
            if outer_cone is not None:
                spot["outerConeAngle"] = outer_cone
            light["spot"] = spot
        self.lights.append(light)
        return len(self.lights) - 1

    def animation(self, name, channels):
        """An animation from `(node, path, times, values, interpolation)` channels.

        `path` is "translation"/"rotation"/"scale"/"weights"; `values` are per-keyframe
        tuples (or flat weight values). Input accessors get the min/max the spec
        requires — without them a strict importer rejects the sampler."""
        samplers, out_channels = [], []
        for channel in channels:
            node_index, path, times, values, interpolation = channel
            time_bytes = bytearray()
            for t in times:
                time_bytes += struct.pack("<f", t)
            input_acc = self._add_accessor(self._add_view(time_bytes), 5126, len(times), "SCALAR",
                                           [min(times)], [max(times)])
            components = 1 if path == "weights" else len(values[0])
            acc_type = {1: "SCALAR", 3: "VEC3", 4: "VEC4"}[components]
            value_bytes = bytearray()
            for value in values:
                value_bytes += (struct.pack("<f", value) if components == 1
                                else struct.pack(f"<{components}f", *value))
            output_acc = self._add_accessor(self._add_view(value_bytes), 5126, len(values),
                                            acc_type)
            samplers.append({"input": input_acc, "output": output_acc,
                             "interpolation": interpolation})
            out_channels.append({"sampler": len(samplers) - 1,
                                 "target": {"node": node_index, "path": path}})
        self.animations.append({"name": name, "samplers": samplers, "channels": out_channels})
        return len(self.animations) - 1

    # --- generic renderable nodes -----------------------------------------
    #
    # These know nothing about physics. A scene that just needs something to look at (a
    # shadow caster, a receiver) uses them directly; the *_body wrappers below are the
    # physics layer on top.

    def add_node(self, name, *, mesh=None, camera=None, skin=None, light=None, translation=None,
                 rotation=None, scale=None, children=None, extras=None):
        """The one node builder — every other helper funnels through it.

        `extras` is passed through VERBATIM: that is the seam the engine's authoring
        blocks (`Physics`, `Ragdoll`, `Cloth`, …) travel through, and nothing here
        interprets them. An empty/None `extras` writes no `extras` key at all, so a plain
        caster stays plain. Nodes named in `children` are removed from the scene's root
        list, so a hierarchy is transformed once, not twice."""
        node = {"name": name}
        if mesh is not None:
            node["mesh"] = mesh
        if camera is not None:
            node["camera"] = camera
        if skin is not None:
            node["skin"] = skin
        if translation is not None:
            node["translation"] = list(translation)
        if rotation is not None:
            node["rotation"] = list(rotation)
        if scale is not None:
            node["scale"] = list(scale)
        if children:
            node["children"] = list(children)
            self._child_nodes.update(children)
        if light is not None:
            node["extensions"] = {"KHR_lights_punctual": {"light": light}}
        if extras:
            node["extras"] = dict(extras)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def node(self, name, mesh_index, position, extras=None, rotation=None):
        """A renderable node at `position`."""
        return self.add_node(name, mesh=mesh_index, translation=position, rotation=rotation,
                             extras=extras)

    def mesh_node(self, name, geometry, position, colour, rotation=None, extras=None):
        """Node for an already-built `(positions, normals, indices)` geometry."""
        positions, normals, indices = geometry
        mesh = self.mesh(positions, normals, indices, self.material(colour), name)
        return self.node(name, mesh, position, extras, rotation)

    def box(self, name, half, position, colour, rotation=None, extras=None):
        return self.mesh_node(name, box_geometry(half), position, colour, rotation, extras)

    def sphere(self, name, radius, position, colour, rotation=None, extras=None):
        return self.mesh_node(name, sphere_geometry(radius), position, colour, rotation, extras)

    # --- physics bodies: geometry + a matching authored collider ------------
    #
    # Each of these injects `extras.Physics` and defaults the collider to the mesh it
    # just built, so visual == collider == authored. Use the generic builders above when
    # a node should carry no physics at all.

    def box_body(self, name, half, position, colour, physics, rotation=None):
        cfg = dict(physics)
        if cfg.get("Shape", "Box") == "Box" and "HalfExtents" not in cfg:
            cfg["HalfExtents"] = list(half)
        return self.box(name, half, position, colour, rotation, {"Physics": cfg})

    def sphere_body(self, name, radius, position, colour, physics):
        cfg = dict(physics)
        if cfg.get("Shape", "Sphere") == "Sphere" and "Radius" not in cfg:
            cfg["Radius"] = radius
        return self.sphere(name, radius, position, colour, None, {"Physics": cfg})

    def convex_body(self, name, geometry, position, colour, physics, rotation=None):
        """A Dynamic body whose collider is a ConvexHull built from its own mesh
        (the loader runs buildConvexHull on the node geometry)."""
        cfg = dict(physics)
        cfg["Shape"] = "ConvexHull"
        return self.mesh_node(name, geometry, position, colour, rotation, {"Physics": cfg})

    def static_mesh_body(self, name, vertices, triangles, colour, friction=0.6):
        """A Static body whose collider is a triangle mesh built from its own geometry
        (Shape: "Mesh"). Vertices are in world space; the node sits at the origin."""
        cfg = {"BodyType": "Static", "Shape": "Mesh", "Restitution": 0.0, "Friction": friction}
        return self.mesh_node(
            name, mesh_from_triangles(vertices, triangles), (0.0, 0.0, 0.0), colour, None,
            {"Physics": cfg})

    def compound_body(self, name, geometry, position, colour, children, physics, rotation=None):
        """A Dynamic body whose collider is a Compound of child boxes; `geometry` is the
        matching visual mesh and `children` the authored Compound child list."""
        cfg = dict(physics)
        cfg["Shape"] = "Compound"
        cfg["Children"] = children
        return self.mesh_node(name, geometry, position, colour, rotation, {"Physics": cfg})

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
        return self.add_node(name, camera=cam_index, translation=eye,
                             rotation=look_at_quat(eye, target, up))

    def to_gltf(self):
        uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(self.bin)).decode(
            "ascii"
        )
        roots = [i for i in range(len(self.nodes)) if i not in self._child_nodes]
        doc = {
            "asset": {"version": "2.0", "generator": self.generator},
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": roots}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.bin), "uri": uri}],
        }
        # Optional blocks, emitted only when used — an unused empty array would be churn
        # in every previously generated file.
        if self.cameras:
            doc["cameras"] = self.cameras
        if self.skins:
            doc["skins"] = self.skins
        if self.animations:
            doc["animations"] = self.animations
        if self.textures:
            doc["images"] = self.images
            doc["samplers"] = self.samplers
            doc["textures"] = self.textures
        if self.lights:
            doc["extensionsUsed"] = ["KHR_lights_punctual"]
            doc["extensions"] = {"KHR_lights_punctual": {"lights": self.lights}}
        return doc


def write_gltf(path, doc):
    """Write a glTF document to `path` — 2-space JSON plus a trailing newline.

    The formatting is part of the contract: the committed assets are regenerated by the
    build, so any change here shows up as churn in every generated file.
    """
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return os.path.relpath(path)
