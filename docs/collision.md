# Collision and Physics

The engine now separates low-level collision primitives from runtime physics ownership.

- `collision/` contains the broadphase (a `BroadPhase` interface implemented by both a dynamic AABB tree and sweep-and-prune), the shape-specific narrowphase (`NarrowPhase::collide` → `ContactManifold`, with a speculative-margin path for separated-but-approaching pairs), and the neutral world-space shapes.
- `physics/` owns bodies, colliders, body/collider handles, stepping, response, and scene sync.
- `scene::Node` no longer owns `Collider` or `PhysicsBody` directly. Nodes carry opaque physics handles and are synchronized with `PhysicsWorld`.

The current system is a custom rigid-body path. The narrowphase is **shape-specific** — it composes each authored box/sphere/capsule into a world-space primitive and produces a real `ContactManifold` (analytic pairs + box/box SAT). Contact *response* is a **TGS soft-step solver** (`ContactSolver`, P9.2) with **full rotational dynamics** (P3): per-shape inertia tensors, quaternion orientation integration, and angular (lever-arm) terms in the normal + friction impulses — boxes topple, spin, and come to rest on a face. The fixed step is split into substeps; penetration is handled by a **soft (compliant) constraint** plus a per-substep **relax pass** rather than a hard split-impulse pass, and restitution is a single end-of-step pass at the true impact velocity. **Joints, limits, and ragdolls** (P4) reuse the same solver as generic constraint rows: a `JointSolver` for distance/ball-socket/hinge constraints (with hinge-angle and cone-twist limits), interleaved with contacts, plus a skinned-skeleton `Ragdoll` driven through a `Node` world-override. **Scale** (P5): the dynamic bodies are partitioned into **simulation islands** and solved per-island; settled islands **sleep** (skip integration + solving until disturbed); and the broadphase is a **dynamic AABB tree** behind a `BroadPhase` interface (sweep-and-prune still selectable). **Real level geometry** (P6): a reusable `AabbBvh<T>` core, a **true centre-of-mass offset** (bodies spin about their real COM), **compound** colliders (one body, many offset child shapes, with aggregate mass properties), and **static triangle meshes** (a body collides against the mesh's actual triangles via a per-collider triangle BVH).

---

## Runtime Overview

Main loop order:

```cpp
scene.update(input_state);
scene.submitPhysics(physics_);

while (accumulator >= fixedDt)
{
    physics_.step(fixedDt);
    accumulator -= fixedDt;
}

scene.applyPhysics(physics_);
renderer->drawFrame(...);
```

`FireEngine` uses a fixed physics timestep of `1.0f / 60.0f`, with frame delta clamped to `0.25f`.

`PhysicsWorld::step()` currently:

1. Integrates `Dynamic` body **velocity** only (gravity); positions advance later.
2. Updates collider swept AABBs.
3. Uses the `BroadPhase` (default `DynamicAabbTreeBroadPhase`; sweep-and-prune selectable) to gather broadphase candidate pairs.
4. Uses `NarrowPhase::collide()` to build a shape-specific `ContactManifold` per pair.
5. Runs the **TGS soft-step solve** (`ContactSolver` + `JointSolver`): prepare once at the
   substep `h`, then per substep warm-start → bias solve (soft normal + friction, lever-arm
   torque) → integrate positions/orientations over `h` → relax solve (no bias).
6. Runs a single end-of-step **restitution** pass, then a **kinematic-only** split-impulse
   position pass (Dynamic bodies resolve penetration through the soft contact bias).
7. Resets resolved collider bounds and rebuilds broadphase pair state when anything moved.

---

## PhysicsWorld Runtime Model

`PhysicsWorld` is the owner of runtime physics state. It stores bodies, colliders, collision shapes, material data, broadphase state, and narrowphase state. The scene graph stores only opaque handles:

```cpp
PhysicsBodyHandle bodyHandle = node.physicsBodyHandle();
PhysicsColliderHandle colliderHandle = node.physicsColliderHandle();
```

This keeps `scene::Node` out of collision ownership. A node can be moved, animated, controlled, rendered, and transformed without owning a `Collider` or `PhysicsBody` directly. When physics is enabled for a node, the node and `PhysicsWorld` are linked by handles.

### Internal Ownership

`PhysicsWorld` owns:

- `BodyEntry` records: `PhysicsBody`, authoritative physics `Transform`, previous position, active flag, and child collider handles.
- `ColliderEntry` records: `Collider`, `ColliderShape`, material, owning body handle, and active flag.
- A `BroadPhase` (defaults to `DynamicAabbTreeBroadPhase`; a `SweepAndPruneBroadPhase` can be injected through the `PhysicsWorld(unique_ptr<BroadPhase>)` constructor): broadphase candidate generation.
- `NarrowPhase`: shape-specific `ContactManifold` generation (overlap + speculative-margin gap contacts).

The public API stays in `include/fire_engine/physics/physics_world.hpp`, but the implementation is
split by responsibility:

- `src/physics/physics_world.cpp`: body/collider/joint creation, destruction, fixed-step solve,
  rigid-body contact generation, island solving, and sleep.
- `src/physics/physics_world_shapes.cpp`: owner-pose resolution, shape→world composition, local
  bounds, and collider AABB refresh/reset.
- `src/physics/physics_world_queries.cpp`: `raycast`, `raycastAll`, `shapecast`, and overlap
  queries.
- `src/physics/physics_world_articulation.cpp`: articulation handles, link colliders,
  self/static contact gather, and articulation stepping/sleep.
- `src/physics/physics_world_debug.cpp`: debug contacts, collider bounds/sleep flags, and joint
  anchor extraction.
- `src/physics/physics_world_events.cpp`: cloth-collider export plus trigger/collision overlap
  event bookkeeping.

Bodies and colliders are created through:

```cpp
PhysicsBodyHandle body = physics.createBody(bodyDesc);
PhysicsColliderHandle collider = physics.createCollider(body, colliderDesc);
```

`createBody()` initializes the body state and its physics transform. `createCollider()` converts the public `ColliderShape` into the local AABB used by the broadphase (the shape-specific narrowphase reads the authored `ColliderShape` directly via `worldShape`), resets the collider frame from the owning body transform, and registers the collider with the broadphase.

Destroying a body unregisters its colliders from the broadphase, invalidates the body and collider handles, deactivates any joints connected to that body, and removes destroyed-collider pairs from trigger/collision overlap tracking. Backing storage remains stable for solver indices and broadphase pointer safety, but the lookup side-tables contain only live handles/pointers; inactive handles no longer resolve through `valid()`, `body()`, or `bodyTransform()`.

### Step Internals

`PhysicsWorld::step(fixedDt)` is independent of `SceneGraph`. It only reads and writes physics-owned state.

The current step order is a **TGS soft-step** (temporal Gauss-Seidel, P9.2): collision
detection runs once per fixed step, then the *solve* is substepped:

1. Ignore non-positive timesteps.
2. Refresh all active collider AABBs from their owning body transforms. (Gravity is no
   longer integrated here — it moves into the per-substep solve below.)
3. Refresh the broadphase from the colliders' current swept bounds.
4. Convert broadphase candidate pairs into solver contacts: pick a moving/target
   body (dynamic preferred, else kinematic), compose both colliders into world
   shapes (`worldShape`), and build a `ContactManifold` with `NarrowPhase::collide()`.
5. Run the **TGS solve** (`solveAndIntegrate` → per island `solveIsland`): build a flat
   `SolverBody` view (Static/Kinematic carry `invMass` and inverse inertia 0; each body's
   world inverse inertia `R·diag(invI_local)·Rᵀ` is built here). Prepare contacts + joints
   **once** at the substep `h = dt / kSubstepCount` (soft-constraint coefficients from
   `b2MakeSoft`; anchors stored body-local for analytic separation tracking). Then for each
   of `kSubstepCount` substeps: integrate gravity (× `gravityScale` × h) → warm-start →
   `kVelocityIterations` **bias** sweeps (friction + soft-normal impulses; separation/joint
   error recomputed from the current pose; each impulse updates both linear and angular
   velocity via lever-arm torque `I⁻¹(r×P)`) → integrate positions + orientations over h →
   `kVelocityIterations` **relax** sweeps (same solve, no bias) that remove the soft bias
   velocity so the correction can't pump energy. Contacts/joints are iterated in a canonical
   pair order (sorted by `(firstId, secondId)`), so the solve is deterministic.
   **Mid-step manifold refresh (P9.6):** at substep `kSubstepCount/2`, any non-mesh contact
   whose Dynamic body is sweeping more than `kSubstepRefreshRotation` this step (|ω|·dt at the
   current solver velocity) is re-collided at the in-flight `SolverBody` poses
   (`refreshIslandContacts` → `worldShapeAt`) and its solver rows re-prepared
   (`ContactSolver::refresh`, a row splice that carries the accumulated normal impulse,
   `relVelN0`, and the restitution engagement flag by tracked-anchor proximity). A step-start
   manifold is stale under fast rotation and concentrates impact impulses on wrongly-placed
   points — the "settle snap" / rotational-tunnelling family; the refresh retires it through
   ~20 rad/s (guarded by `Demos.RotationalTunnellingBoundedTo20RadPerSec` and
   `Demos.ConvexHull.TetraSettleTwistBounded`).
6. Run a single end-of-step **restitution** pass at the true impact velocity (gated below
   `kRestitutionThreshold`), then a **kinematic-only split-impulse position pass** — Dynamic
   bodies resolve penetration through the soft contact bias, but scene-driven Kinematic
   bodies (positionWeight 1, no inverse mass) still slide out of static penetration.
7. Advance every **reduced-coordinate articulation** in its own solve pass
   (`stepArticulations`): link-vs-static contacts and same-articulation self-collision pairs
   are gathered from this step's broadphase (real narrowphase manifolds, tracked link-local
   across the substeps), then each articulation runs its own TGS substep loop — ABA free
   dynamics, contact + velocity-level cone-twist joint-limit sweeps through the
   `ConstraintBody` / pair-impulse seams, and a settle assist on the floating base's linear
   velocity. Articulations do not yet join rigid islands (link-vs-dynamic-rigid is deferred).
8. Write final velocities back; if anything moved, reset collider frame bounds and rebuild
   broadphase pair state.
9. Capture each active body's current position as `previousPosition`.

**Speculative-margin CCD.** Step 2 expands each dynamic collider's swept AABB by its
predicted displacement (`velocity × dt`) so the broadphase pairs a fast mover with what
it is about to reach; step 4 passes a per-pair margin `(|v_moving| + |v_target|)·dt +
kSpeculativeDistance` to `NarrowPhase::collide`, which then emits a contact for a
*separated* pair within that margin with a **negative penetration** (`= -gap`). The
solver gives such a gap contact a normal-bias target of `-separation/dt`, so the body
may close the gap this step but the non-negative impulse clamp brakes any overshoot —
fast bodies stop at the surface instead of tunnelling, with no rewind or substepping.

`previousPosition` is important for both dynamic and kinematic bodies. Dynamic bodies move during `step()`. Kinematic bodies usually move before `step()` when scene/input code changes their node transform and `submitPhysics()` pushes that transform into physics.

Before the solver runs (between steps 5 and 6), `step()` snapshots the frame's contacts into `debugContacts_` — one entry per manifold point (the world contact point plus the manifold normal). This is captured *before* the solver moves bodies, and is read-only debug data: it does not affect the simulation.

### Debug + Determinism API

`PhysicsWorld` exposes read-only, Vulkan-free data for the renderer's debug-draw subsystem and for regression testing:

- `gatherColliders()` — world-space collider primitives (plane / sphere / box / capsule), already used by the cloth solver; reused to draw authored collider shapes.
- `debugColliderBounds()` — world AABBs of every active collider (broadphase bounds).
- `debugContacts()` — the most recent step's manifold contact points + normals.

`step(fixedDt)` is **deterministic**: a pure function of its initial state with no RNG and no hash-map iteration affecting results. The solver iterates constraints in broadphase-pair order (a total-order sweep-and-prune endpoint comparator tie-broken by `ColliderId`), and the warm-start cache is only ever *looked up* by collider-pair key — never iterated to produce results — so its `unordered_map` ordering cannot leak into the simulation. This is guarded by `tests/physics/test_physics_determinism.cpp`, which replays a scripted scene and asserts two runs hash bit-identically, free-fall matches the closed form, and the end-state matches a recorded golden hash (a behaviour tripwire — update it intentionally when the solver math changes).

Long-running physics demo and ragdoll settle tests are tagged `[slow]`. The `test_fire_engine`
CTest entry runs the fast subset (`~[slow]`), so plain CTest and `ctest --preset fast` stay
fast. Run `cmake --build --preset full` from the source root, or
`cmake --build build --target tests-full`, before merging solver, sleep, broadphase,
articulation, or ragdoll changes.

### SceneGraph Synchronization

Physics sync is deliberately explicit and one-directional at each stage of the frame.

Before stepping physics:

```cpp
scene.update(input_state);
scene.submitPhysics(physics_);
```

`SceneGraph::submitPhysics()` walks every node. For each node with a valid physics body handle, it looks up the body in `PhysicsWorld`. If the body exists and is not `Dynamic`, the node transform is pushed into the physics body:

```cpp
if (body != nullptr && body->type() != PhysicsBodyType::Dynamic)
{
    physics.setBodyTransform(handle, node.transform());
}
```

This means:

- `Static` bodies are scene-authored. Their transforms are copied from scene to physics before simulation.
- `Kinematic` bodies are scene/gameplay-authored. Input or animation moves the node first, then physics receives that target transform and can resolve collision against it.
- `Dynamic` bodies are not pushed from scene to physics. Physics remains authoritative for them.

After stepping physics:

```cpp
scene.applyPhysics(physics_);
renderer->drawFrame(...);
```

`SceneGraph::applyPhysics()` walks every node again. For each node with a valid body handle, it reads `PhysicsWorld::bodyTransform(handle)`. If the body exists and is not `Static`, the physics transform is copied back onto the node:

```cpp
if (body != nullptr && transform.has_value() && body->type() != PhysicsBodyType::Static)
{
    node.transform().position(transform->position());
    node.transform().rotation(transform->rotation());
    node.transform().scale(transform->scale());
}
```

This means:

- `Dynamic` bodies pull their simulated transforms back onto scene nodes for rendering.
- `Kinematic` bodies also pull back from physics, so collision response can correct or slide the gameplay-authored target transform.
- `Static` bodies do not pull from physics. The scene remains authoritative for static placement.

After applying physics, `SceneGraph::resolve()` is called so node world transforms and cached composed matrices are updated before rendering.

The current main loop syncs kinematic/static scene transforms once per rendered frame before the fixed-step loop. If more than one physics substep runs in a frame, all substeps use the same scene-authored kinematic target for that rendered frame.

### Authority by Body Type

| Body type | Before `step()` | During `step()` | After `step()` |
|---|---|---|---|
| `Static` | Scene transform is pushed into physics. | Can be collided against, but does not move. | Not copied back to scene. |
| `Kinematic` | Scene transform is pushed into physics. | Can slide/respond if its frame movement hits another collider. | Physics transform is copied back to scene. |
| `Dynamic` | Scene transform is not pushed into physics. | Physics integrates velocity, gravity, contacts, and response. | Physics transform is copied back to scene. |

This authority split is the main rule to preserve when adding richer physics. Scene code can request movement through kinematic transforms or body velocities, but dynamic simulation state should remain owned by `PhysicsWorld`.

---

## Source Layout

```text
include/fire_engine/collision/
  collider_id.hpp
  end_point.hpp
  collider.hpp
  aabb_bvh.hpp                       # generic payload-templated fat-AABB BVH core
  broad_phase.hpp                    # BroadPhase interface + CollisionPair
  sweep_and_prune_broad_phase.hpp
  dynamic_aabb_tree_broad_phase.hpp  # fat-AABB BVH (default), wraps AabbBvh
  narrow_phase.hpp
  contact_manifold.hpp
  world_shape.hpp
  geometry.hpp
  support.hpp
  gjk_epa.hpp
  ray.hpp                   # Ray + analytic ray/shape intersections (queries)
  shape_cast.hpp            # GJK conservative-advancement shape sweep (queries)

include/fire_engine/physics/
  physics_handle.hpp
  physics_body.hpp
  collider_shape.hpp
  contact.hpp
  solver_math.hpp           # shared per-body solver primitives (contact + joint)
  contact_solver.hpp
  joint.hpp                 # JointDesc / JointType / JointLimits / JointInput
  joint_solver.hpp
  island.hpp                # union-find island partition (per-island solve + sleep)
  physics_query.hpp         # QueryFilter / Raycast|Shapecast|OverlapHit
  collision_event.hpp       # EventPhase + ContactEvent (trigger/collision events)
  character_controller.hpp  # kinematic-capsule collide-and-slide controller
  physics_constants.hpp
  physics_world.hpp

include/fire_engine/scene/
  ragdoll.hpp               # skinned-skeleton → bodies + joints, world-override drive

src/collision/
  collider.cpp
  sweep_and_prune_broad_phase.cpp
  dynamic_aabb_tree_broad_phase.cpp
  narrow_phase.cpp
  geometry.cpp
  gjk_epa.cpp
  ray.cpp
  shape_cast.cpp

src/physics/
  contact_solver.cpp
  joint_solver.cpp
  island.cpp
  character_controller.cpp
  physics_body.cpp
  physics_world.cpp
  physics_world_articulation.cpp
  physics_world_debug.cpp
  physics_world_events.cpp
  physics_world_queries.cpp
  physics_world_shapes.cpp

src/scene/
  ragdoll.cpp

src/core/
  convex_hull_builder.cpp   # mesh → ConvexHullShape (welded verts + coplanar faces)
```

Both broadphases (`DynamicAabbTreeBroadPhase`, `SweepAndPruneBroadPhase`) implement the `BroadPhase` interface and are broadphase-only — they do not know about `SceneGraph`, `Node`, or gameplay response. `PhysicsWorld` owns one via `unique_ptr<BroadPhase>`, defaulting to the tree.

---

## Physics Types

### Body Types

`PhysicsBodyType` supports:

| Type | Meaning |
|---|---|
| `Static` | Non-simulated body. Useful for level geometry and walls. Synced from scene to physics. |
| `Kinematic` | Scene/gameplay driven body. Participates in collision and slides on impact. Use this for controllable objects. |
| `Dynamic` | Physics-driven body. Integrated by `PhysicsWorld`; synced back to the scene after stepping. |

`Dynamic` nodes cannot also be `Controllable` in glTF import. Use `Kinematic` for controllable collision objects.

### Body State

`PhysicsBody` currently stores:

- `type`
- `linearVelocity`
- `angularVelocity`
- `mass` / `inverseMass`
- `inverseInertiaLocal` (diagonal, principal frame; set from shape + mass in
  `createCollider`, zero ⇒ infinite inertia for Static/Kinematic)
- `gravityScale`
- `PhysicsMaterial { restitution, friction }`

Friction is now live (Coulomb friction as a clamped tangent impulse in the
`ContactSolver`). Angular velocity is now live: the solver applies torque from
off-centre contacts via per-shape inertia tensors, and `step()` integrates body
orientation each frame (P3).

### Collider Shapes

`ColliderShape` supports:

| Shape | Fields | Current behavior |
|---|---|---|
| `AabbShape` | `bounds` | Used directly as local AABB. |
| `BoxShape` | `halfExtents`, `center` | Converted to a local AABB. |
| `SphereShape` | `radius`, `center` | Converted to a local AABB. |
| `CapsuleShape` | `radius`, `halfHeight`, `center` | Converted to a local AABB. |
| `ConvexHullShape` | `vertices`, `faces` (ordered loops + normals) | AABB for broadphase; GJK/EPA + face-clip for contacts. |

The AABB column above is the **broadphase** proxy. The narrowphase now composes
each authored shape into a world-space primitive (`PhysicsWorld::worldShape` →
`WorldSphere`/`WorldBox`/`WorldCapsule`) and produces a shape-specific
`ContactManifold` (`NarrowPhase::collide`); contacts are no longer swept-AABB.

---

## Programmatic Setup

```cpp
#include <fire_engine/physics/physics_world.hpp>

using namespace fire_engine;

PhysicsWorld physics;

PhysicsBodyDesc body;
body.type = PhysicsBodyType::Dynamic;
body.position = {0.0f, 4.0f, 0.0f};
body.linearVelocity = {2.0f, 0.0f, 0.0f};
body.mass = 1.0f;
body.gravityScale = 1.0f;
body.material = PhysicsMaterial{.restitution = 0.5f, .friction = 0.0f};

PhysicsBodyHandle bodyHandle = physics.createBody(body);

ColliderDesc collider;
collider.shape = BoxShape{.halfExtents = {0.5f, 0.5f, 0.5f}, .center = {}};
collider.collisionLayer = 1U << 0U;
collider.collisionMask = 1U << 1U;

PhysicsColliderHandle colliderHandle = physics.createCollider(bodyHandle, collider);

node.physicsBodyHandle(bodyHandle);
node.physicsColliderHandle(colliderHandle);
```

In normal engine loading, you usually do not write this manually. The glTF loader creates the body/collider from node `extras.Physics`.

---

## glTF Authoring

Physics is authored on a glTF node using custom `extras` data:

```json
{
  "nodes": [
    {
      "name": "Crate",
      "mesh": 0,
      "extras": {
        "Physics": {
          "BodyType": "Dynamic",
          "Layer": 4,
          "Mask": 2,
          "Velocity": [0.0, 0.0, 0.0],
          "Mass": 1.0,
          "Restitution": 0.5,
          "Friction": 0.0,
          "GravityScale": 1.0,
          "Shape": "Box",
          "HalfExtents": [0.5, 0.5, 0.5],
          "Center": [0.0, 0.0, 0.0]
        }
      }
    }
  ]
}
```

The loader only reads physics from `extras.Physics`. The old top-level custom extras `CollisionLayer`, `CollisionMask`, `Dynamic`, and `Velocity` are no longer supported.

Physics extras currently require the node to have a mesh. If `Shape` is omitted, the loader computes an `AabbShape` from the mesh POSITION bounds.

### Supported Fields

| Field | Type | Default | Notes |
|---|---:|---:|---|
| `BodyType` | string | `"Static"` | One of `"Static"`, `"Kinematic"`, `"Dynamic"`. |
| `Layer` | uint32 | `1` | The layer this collider belongs to. |
| `Mask` | uint32 | `4294967295` | Layers this collider wants to collide with. |
| `IsTrigger` | bool | `false` | A trigger generates overlap **events** (`triggerEvents()`) instead of a solver response. |
| `Velocity` | vec3 | `[0, 0, 0]` | Initial linear velocity. Mainly useful for `Dynamic`. |
| `Mass` | number | `1.0` | Only gives non-zero inverse mass for `Dynamic`. |
| `Restitution` | number | `1.0` | Current dynamic response uses this for bounce. |
| `Friction` | number | `0.0` | Coulomb friction coefficient; combined per pair as `sqrt(a*b)` by the solver. |
| `GravityScale` | number | `1.0` | Set `0.0` to disable gravity for a body. |
| `Shape` | string | mesh AABB | One of `"Box"`, `"Sphere"`, `"Capsule"`, `"ConvexHull"`, `"Mesh"`, `"Compound"`. `"ConvexHull"` builds the hull from the node mesh (welded vertices + coplanar-merged faces); `"Mesh"` builds a static triangle-mesh collider from the node geometry (Static bodies only); `"Compound"` reads a `Children` array. |
| `Center` | vec3 | `[0, 0, 0]` | Shape-local center for explicit shapes. |
| `HalfExtents` | vec3 | `[0.5, 0.5, 0.5]` | Box only. |
| `Radius` | number | `0.5` | Sphere and capsule. |
| `HalfHeight` | number | `0.5` | Capsule only. |
| `Children` | array | — | `"Compound"` only — a list of child objects, each `{ "Shape": "Box"\|"Sphere"\|"Capsule", shape params, "Position": [...], "Rotation": [x,y,z,w], "Friction", "Restitution" }`. One child collider is created per entry and the body's mass/inertia is aggregated from them. |

JSON numbers for `Layer` and `Mask` must be unsigned 32-bit integers. Use decimal values in `.gltf` JSON; for example `4294967295` is `~0U`.

A static triangle mesh is **one-sided** (it collides from the CCW-outward face only) and assumes a fixed transform; a compound's child shapes are primitives (compounds do not nest).

### Ragdoll Authoring

`extras.Ragdoll` on a **skinned** node auto-builds a ragdoll from that node's skin
joints when the scene is loaded with a ragdoll out-parameter (the app passes one, so
authored ragdolls activate at runtime). One capsule body + a ball-socket(+cone-twist)
joint per bone, seeded from the bind pose:

```json
{
  "name": "Character",
  "mesh": 0,
  "skin": 0,
  "extras": {
    "Ragdoll": {
      "Mass": 1.0,
      "Radius": 0.05,
      "BoneLength": 0.2,
      "ConeTwist": true,
      "SwingLimit": 0.7,
      "TwistLimit": 0.5
    }
  }
}
```

| Field | Type | Default | Notes |
|---|---:|---:|---|
| `Mass` | number | `1.0` | Per-bone body mass. |
| `Radius` | number | `0.05` | Capsule radius for each bone. |
| `BoneLength` | number | `0.2` | Capsule length for a root/leaf bone (others span the bone-to-parent gap). |
| `ConeTwist` | bool | `true` | Enables the swing-cone + twist angular limit on each joint. |
| `SwingLimit` | number | `0.7` | Cone half-angle in radians. |
| `TwistLimit` | number | `0.5` | ± twist in radians. |
| `Articulated` | bool | `false` | Build a **reduced-coordinate articulation** (`Ragdoll::makeArticulated`) instead of maximal-coordinate bodies + joints. |

Presence of the `Ragdoll` object is what flags the node; all fields are optional. The
bones are driven via a `Node` world-override (the skinning path renders the simulated
pose unchanged); an **animated** skeleton's joints are reset to their bind TRS before the
ragdoll is seeded (at load time the animation hasn't been evaluated, so composing the raw
nodes would collapse every bone onto the armature origin).

**`Articulated: true` (the recommended path for full skeletons)** builds one Featherstone
articulation from the bone tree: the single root bone becomes a floating 6-DOF base, every
child bone a spherical joint seeded from the bind pose, with a bone-axis-aligned capsule
collider per link, velocity-level cone-twist limits, same-articulation self-collision
(bind-pose-overlapping pairs excluded), and a settle assist on the base's linear velocity.
Joint error is zero by construction, so a many-bone skeleton settles where the
maximal-coordinate chain limit-cycles — the 19-bone CesiumMan
(`assets/CesiumMan/CesiumManRagdoll.gltf`) is the showcase. Articulated bones are not
body-bound; `Ragdoll::syncNodes()` pushes link forward-kinematics to the bone nodes'
world-overrides each frame (the app's main loop does this after `applyPhysics`). The
maximal path remains for small joint counts and general constraints.

### Static World Collider

```json
"extras": {
  "Physics": {
    "BodyType": "Static",
    "Layer": 2,
    "Mask": 4294967295
  }
}
```

Because `Shape` is omitted, this uses the mesh POSITION bounds as an AABB.

### Controllable / Kinematic Player

Use the existing `Controllable` custom property plus a `Kinematic` physics body:

```json
"extras": {
  "Controllable": true,
  "Physics": {
    "BodyType": "Kinematic",
    "Layer": 1,
    "Mask": 2,
    "Shape": "Box",
    "HalfExtents": [0.5, 0.9, 0.5],
    "Center": [0.0, 0.9, 0.0]
  }
}
```

This lets scene/input code move the node, then `submitPhysics()` pushes that transform into `PhysicsWorld`. During `step()`, kinematic collision response slides the body against contacts.

Do not set `BodyType` to `"Dynamic"` on a node with `"Controllable": true`; the loader rejects that combination.

### Dynamic Crate

```json
"extras": {
  "Physics": {
    "BodyType": "Dynamic",
    "Layer": 4,
    "Mask": 2,
    "Mass": 2.0,
    "Velocity": [1.0, 0.0, 0.0],
    "Restitution": 0.25,
    "GravityScale": 1.0,
    "Shape": "Box",
    "HalfExtents": [0.5, 0.5, 0.5]
  }
}
```

After each physics step, `applyPhysics()` copies dynamic body transforms back onto their bound scene nodes.

### Physics demos — worked authoring examples

`assets/physics_demos/` holds one minimal glTF scene per capability — the reference for how
each feature is authored. They are emitted by `assets/physics_demos/generate.py` (run
automatically at build) and each is mirrored by a headless behaviour test in
`tests/physics/test_demos.cpp` (the `[Demos]` tag). Run a demo from `build/`:
`./fireEngineApp physics_demos/<Name>.gltf skybox.hdr --debug-physics`.

| Demo | Authoring feature it exercises |
|---|---|
| `FallRestDemo` | `Shape:"Box"` Dynamic body + Static floor (the baseline) |
| `RestitutionDemo` | `Restitution` (combined `max(a,b)`) on `Shape:"Sphere"` |
| `FrictionRampDemo` | `Friction` (combined `sqrt(a·b)`) + a rotated Static ramp |
| `StackDemo` | resting/sleeping stack of Dynamic boxes |
| `ToppleDemo` | rotational dynamics — a tall box authored tilted |
| `ConvexHullDemo` | `Shape:"ConvexHull"` (hull built from the node mesh) |
| `SleepDemo` | sleeping + wake-on-impact (asleep collider colour under `--debug-physics`) |
| `StaticMeshDemo` | `Shape:"Mesh"` (Static triangle-mesh collider from node geometry) |
| `CompoundDemo` | `Shape:"Compound"` with a `Children` array + aggregated COM |

Two query/character demos are driven from the main loop (not authored in glTF), behind CLI
flags: **`-k`** (the kinematic `CharacterController` walking a step-pyramid course) and **`-q`**
(a query probe — a rotating raycast fan + overlap on a ring of bodies, drawn via `DebugDraw`).
The full table with expected behaviours lives in [`README.md`](../README.md) § Physics Demos. The **ragdoll**
(`extras.Ragdoll`) demo is deferred until the solver can settle a complex joint network (see
[`roadmap.md`](../roadmap.md) P9).

---

## Layer / Mask Filtering

Every collider has:

```cpp
collisionLayer  // which layer this collider is on
collisionMask   // which layers this collider wants to test against
```

Two colliders can form a pair only when both checks pass:

```cpp
(a.mask & b.layer) != 0
(b.mask & a.layer) != 0
```

Recommended layer scheme:

| Name | Bit | Decimal |
|---|---:|---:|
| Player | `1 << 0` | `1` |
| World | `1 << 1` | `2` |
| Dynamic props | `1 << 2` | `4` |
| Triggers | `1 << 3` | `8` |
| Projectiles | `1 << 4` | `16` |

Examples:

- Player collides with world and props: `Layer = 1`, `Mask = 2 | 4 = 6`.
- World collides with everything: `Layer = 2`, `Mask = 4294967295`.
- Props collide with world and player: `Layer = 4`, `Mask = 1 | 2 = 3`.
- Projectiles collide with player and world: `Layer = 16`, `Mask = 1 | 2 = 3`.

Layer/mask filtering happens before narrowphase and is the cheapest way to keep broadphase pairs small.

---

## Low-Level Collision API

You can still use `SweepAndPruneBroadPhase` directly when you only need candidate AABB pairs.

```cpp
SweepAndPruneBroadPhase broadphase;

Collider a;
a.localBounds({.min = {-0.5f, -0.5f, -0.5f}, .max = {0.5f, 0.5f, 0.5f}});
a.update(aWorld);
broadphase.addCollider(a);

Collider b;
b.localBounds({.min = {-1.0f, -1.0f, -1.0f}, .max = {1.0f, 1.0f, 1.0f}});
b.update(bWorld);
broadphase.addCollider(b);

// Per frame:
a.update(newAWorld);
b.update(newBWorld);
broadphase.update();

for (const CollisionPair& pair : broadphase.possiblePairs())
{
    // pair.first / pair.second are const Collider*
}
```

`SweepAndPruneBroadPhase` methods:

| Method | Purpose |
|---|---|
| `addCollider(Collider&) -> ColliderId` | Register a collider. Idempotent for the same object. |
| `removeCollider(ColliderId) -> bool` | Unregister by id. |
| `removeCollider(Collider&) -> bool` | Unregister by collider address. |
| `update()` | Incrementally refresh all registered endpoints. |
| `updateCollider(Collider&)` | Refresh one registered collider. |
| `rebuild()` | Full sort and pair-state rebuild. |
| `clear()` | Remove all colliders. |
| `possiblePairs()` | Current broadphase candidate pairs. |
| `validate()` | O(N^2) debug/test self-check. |

`Collider` still has the important lifetime rule: do not move or destroy a registered collider. The broadphase stores pointers to endpoints owned inside each `Collider`.

---

## Current Limitations

- Narrowphase is shape-specific (sphere/box/capsule analytic + box/box SAT) plus a
  general **GJK + EPA** convex path for any pair involving a `ConvexHullShape`
  (convex/convex, convex/box, convex/sphere, convex/capsule), all producing a real
  `ContactManifold`. Convex inertia is approximated by the hull's AABB box inertia,
  and `ConvexHull` authoring assumes a convex input mesh (coplanar-merge, not quickhull).
- The broadphase proxies every shape as an AABB (it only needs bounds). The default is
  a **dynamic AABB tree** (`DynamicAabbTreeBroadPhase`); sweep-and-prune is still
  available behind the `BroadPhase` interface. Pairs are regenerated in full each step
  (a moved-leaf-only incremental pass is a future optimisation).
- Contact response is a sequential-impulse solver (warm-started normal + friction
  impulses, gated restitution, split-impulse positional correction) with **full
  rotational dynamics** — inertia tensors, orientation integration, lever-arm torque.
- Continuous collision is handled by **speculative-margin CCD** (motion-expanded swept
  broadphase + negative-penetration gap contacts braked by the solver), so fast movers
  don't tunnel. The speculative normal is the closest-feature normal at the start pose,
  so a steep corner graze can be slightly off (full conservative advancement is future).
- Bodies integrate about a **true centre of mass** (P6): `PhysicsBody::centerOfMassLocal`
  (the shape/compound centroid) — zero for a centred single collider, so those are
  unchanged. Gyroscopic torque (`ω × Iω`) is omitted, as is standard for stable game
  physics. Compound inertia keeps only the tensor diagonal (exact for compounds symmetric
  about the body axes, approximate otherwise).
- **Joints, limits & ragdolls are implemented** (P4 — see *Joints & Ragdolls* below):
  distance / ball-socket / hinge constraints with hinge-angle and cone-twist limits,
  and a skinned-skeleton `Ragdoll`. Joint position error is corrected through a
  Baumgarte velocity bias (not a separate split-impulse pass); ragdoll capsules align
  to the body local-y and anchors assume unit-scale bodies; ragdoll→animation *recovery*
  (blend back) is not wired (`deactivate()` just releases the override).
- **Sleeping + simulation islands are implemented** (P5 — see *Sleeping and Islands*
  below): the dynamic bodies are partitioned into islands and solved per-island, and a
  settled island sleeps (zeroed velocities, skipped solve) until disturbed. Broadphase
  and narrowphase still run for sleeping bodies — only the solve is skipped.
- **Compound colliders + static triangle meshes are implemented** (P6 — see *Static Mesh
  & Compound Colliders* below): a compound is one body with many offset child colliders
  (`createCompoundCollider`), and a static mesh collides against its actual triangles via a
  per-collider triangle BVH (`createMeshCollider`, Static-only, one-sided). Both authored
  via `extras.Physics` `Shape: "Compound"`/`"Mesh"`. Mesh contacts use a planar
  reconstruction (face normal + plane-measured penetration) for stability; EPA
  contact-point quality degrades for triangles enormous relative to the body.
- Spatial queries — `raycast` / `raycastAll` / `shapecast` / `overlapShape` / `overlapSphere`
  with a layer/mask `QueryFilter` — are implemented (P7), brute-force over active colliders
  (mesh colliders dispatch into their triangle BVH; BVH-accelerated broadphase queries are a
  noted follow-up). The **`-q` query-probe demo** visualises them: a ring of static bodies with a
  rotating fan of raycasts (green to hit + marker, faint on miss) drawn each frame from
  `PhysicsWorld::raycast`. The **`-k` demo** drives the kinematic `CharacterController` over the
  same queries. Both are headlessly covered (`test_physics_query.cpp`, `test_character_controller.cpp`).

The architecture keeps physics ownership behind `PhysicsWorld`, never back in `scene::Node`.

---

## Subsystem Detail

How each major subsystem works, with its provenance (the `Pn` labels are the historical build
order — see [`roadmap.md`](../roadmap.md) for the full physics-track history). All of this is **current**; genuinely
outstanding work is in [Future Directions](#future-directions) at the end. Everything below lives
behind the `PhysicsWorld` boundary — physics state never goes back onto `scene::Node`.

### Shape-Specific Narrowphase

Every authored shape is composed into a world-space primitive and tested shape-specifically (there is
no AABB contact shape):

- sphere vs sphere
- sphere vs box
- sphere vs capsule
- capsule vs box
- box vs box using SAT (with face clipping → up to 4 contact points)
- capsule vs capsule for character-like bodies
- **convex hull** vs convex/box/sphere/capsule via **GJK + EPA** (P3.5)

`NarrowPhase::collide(WorldShape, WorldShape, margin)` is the dispatch layer
(2-arg `std::visit`): primitive pairs go to analytic `collidePair` overloads (with
closest-point primitives in `collision/geometry.{hpp,cpp}`); any convex-involving pair
goes through `collideConvex` → `gjkEpaContact` (`collision/gjk_epa.{hpp,cpp}` over
`collision/support.hpp` support functions) → polytope face-clip or single witness point.

### Contact Manifolds and Persistent Contacts

The single swept-AABB contact is replaced by a real `ContactManifold`
(`collision/contact_manifold.hpp`): a normal plus up to four `ManifoldPoint`s
(`position` + `penetration`). **Persistent** contacts are live too — the
`ContactSolver` keeps a warm-start cache keyed by collider pair and proximity-
matches points across frames to inherit impulses, which removes the jitter on
resting stacks.

### Impulse Solver (full rigid body)

`ContactSolver` (`physics/contact_solver.{hpp,cpp}`) is a sequential-impulse solver
with full rotational dynamics. It handles:

- restitution as a (resting-threshold-gated) normal bias impulse
- Coulomb friction as a clamped tangent impulse (`PhysicsMaterial::friction`)
- mass ratios between two dynamic bodies (inverse-mass weighting)
- split-impulse positional/orientation correction for penetration (slop-tolerant)
- several velocity iterations per fixed step, warm-started across frames
- **angular dynamics (P3)** — per-shape inertia tensors, world inverse inertia
  `R·diag(invI)·Rᵀ`, lever-arm torque `I⁻¹(r×P)` in every impulse, and quaternion
  orientation integration; boxes topple, spin, and rest on a face.

`PhysicsMaterial::friction`, body mass, and `PhysicsBody::angularVelocity` are all
materially useful. **Speculative-margin CCD (P2.5)** reuses the solver's
normal-impulse clamp: gap contacts (negative penetration) get a `-separation/dt`
normal bias so fast movers stop at the surface.

### Joints & Ragdolls (P4)

Joints reuse the solver as **generic constraint rows** rather than a second solver.
The shared per-body math (world inverse inertia, relative velocity, effective mass,
apply-impulse) lives in `physics/solver_math.hpp`; both `ContactSolver` and the new
`JointSolver` (`physics/joint_solver.{hpp,cpp}`) call it.

- **`JointSolver`** mirrors `ContactSolver` (`prepare` / `warmStart` / `solveVelocity` /
  `store`). Each joint expands into rows carrying a full Jacobian
  (`linearA`/`angularA`/`linearB`/`angularB`); a row solves `λ = -effMass·(J·v − bias)`,
  clamps the *accumulated* impulse to `[lower, upper]` (`[-∞,∞]` bilateral, one-sided for
  a limit), and applies the delta to both bodies. Position error feeds a Baumgarte
  velocity bias (`kJointBaumgarte`) — there's no separate split-impulse pass for joints.
- **Types** (`physics/joint.hpp`): **Distance** (1 row along the anchor axis),
  **BallSocket** (3 rows holding the world anchors coincident), **Hinge** (ball-socket +
  2 rows aligning the hinge axes). `PhysicsWorld::createJoint` / `destroyJoint` store
  joints tombstoned like bodies; `buildJointInputs` composes local anchors/axes with the
  live body transforms each step.
- **Limits**: hinge `[lower, upper]` angle clamp and ball-socket **cone-twist** (swing
  cone half-angle + ±twist), derived from a **swing-twist quaternion decomposition** of
  the relative orientation (measured from the rest frame captured at creation). A limit
  adds a one-sided row only while violated, so an in-range joint adds no energy.
- **Solve order**: `solveAndIntegrate` interleaves the two solvers — warm-start both,
  then each velocity iteration solves joints first, then contacts, over the same
  `SolverBody` array. Joint impulses warm-start across frames (cached per joint, slot-
  indexed, lookup-only — determinism-safe).
- **Ragdoll** (`scene/ragdoll.{hpp,cpp}`): `Ragdoll::make(physics, boneNodes, params)`
  builds a capsule body + ball-socket(+cone-twist) joint per bone (parent resolved
  through the `Node` hierarchy; bones share a collision layer masked out of itself so
  overlapping capsules don't fight the joints), seeded from each bone's bind-pose world.
  The **drive** is a new **`Node` world-override**: `activate()` sets each bone node's
  `worldOverride_`, which `update()`/`resolve()` treat as the authoritative
  `composedWorld` (bypassing the parent chain + local transform). Since `Skin` reads
  `composedWorld`, the existing skinning path renders the simulated pose;
  `SceneGraph::applyPhysics` writes each body's world transform into the override every
  step. `deactivate()` clears the overrides to release the bones back to animation.
- Lives in `scene/` (not `physics/`) so the scene→physics dependency direction holds
  (the physics layer stays scene-free).

### Static Mesh & Compound Colliders (P6)

Level geometry no longer reduces to a mesh-sized AABB.

- **Reusable BVH core** (`collision/aabb_bvh.hpp`): a payload-templated `AabbBvh<T>` —
  fat-AABB proxies + `createProxy`/`destroyProxy`/`moveProxy`/`query` — lifted out of the
  P5 dynamic tree. `DynamicAabbTreeBroadPhase` now owns an `AabbBvh<Collider*>`
  (behaviour-preserving); the static mesh uses an `AabbBvh<int>` of triangle indices.
- **True COM offset**: `PhysicsBody::centerOfMassLocal` (the shape/compound centroid).
  `solveAndIntegrate` builds the solver body at the world COM (`origin + R·comLocal`) and
  writes the transform origin back (`COM − R·comLocal`). Zero ⇒ identical to before.
- **Compound** (`PhysicsWorld::createCompoundCollider`, `CompoundChild`): one child
  collider per primitive at its `localPosition`/`localRotation` (`ColliderEntry` composes
  the offset in `worldShape`; bounds offset too). Mass properties aggregate — COM by
  volume-weighted child centroid, inertia by parallel-axis sum (diagonalised).
- **Static triangle mesh** (`createMeshCollider`, `StaticMeshShape`, Static-only): a
  per-collider triangle `AabbBvh<int>` built once in world space; `contacts()` expands a
  mesh pair into one `SolverContact` per overlapping triangle (triangle-indexed warm-start
  sub-key). Per-triangle narrowphase reuses GJK/EPA, then **reconstructs a planar
  contact** — the normal is snapped to the triangle's CCW face normal and each point's
  penetration is re-measured against the face plane, which removes the edge-normal /
  flat-triangle EPA artifacts that otherwise rock a resting box to a NaN; non-finite or
  implausibly-deep witness points are dropped. Meshes are one-sided (front face only).
- **glTF** (`extras.Physics`): `Shape: "Mesh"` (static triangle mesh) and `Shape:
  "Compound"` with a `Children` array.
- Limits: compound inertia keeps only the diagonal; mesh EPA points degrade for triangles
  enormous relative to the body (a specialised primitive-vs-triangle test is future
  hardening); static meshes assume a fixed transform.

### Character Controller (P7)

`physics/CharacterController` is a kinematic-capsule controller built entirely on the P7
spatial queries (no rigid-body simulation). `move(world, displacement) →
{position, grounded, groundNormal}` resolves a desired displacement by **collide-and-slide**:

- The vertical (gravity/jump) and horizontal components are swept separately; blocked
  motion is projected along the contact plane (`v − (v·n)n`).
- **Slope limit** (`maxSlopeCosine`): sliding off a face steeper than the limit drops the
  upward component, so walls / steep slopes can't be climbed; gentler ramps are.
- **Step up/down**: a lift → walk → **sweep-and-rest** drop mounts low ledges (≤ `stepOffset`) —
  resting on the step's top edge so a rounded capsule climbs at walk speed instead of sliding off
  the edge; accepted only when the lifted walk *clears* the obstacle (which rejects walls/steep
  slopes). The horizontal slide ignores walkable ground/edge contacts (only walls block lateral
  travel) so the capsule doesn't wedge descending an edge, and a `move()` output guard drops any
  net-backward horizontal step (a rare riser depenetration) to a no-progress frame. A short
  downward probe keeps the character grounded stepping down.
- **Grounding**: a capsule sweep gives the true (slope-correct) feet-to-ground distance for
  the snap; a downward raycast gives the analytic surface normal (capsule-sweep normals are
  unreliable on box faces; the raycast normal sidesteps that). The GJK ground contact itself
  is now robust on large boxes/edges (P7.5: the loop reports the lower-bound separation through
  the closest feature, exact even when the support tie-breaks on a flat face).

It's a headless engine class (`tests/physics/test_character_controller.cpp`), driven from
`FireEngine::mainLoop` rather than a scene component — see **Design reviews** in
[`roadmap.md`](../roadmap.md) for the architectural reasoning. The `-k` CLI flag runs a step-pyramid patrol demo
(bounds-based turnaround at the flat ends; advanced at the real per-frame dt for smooth motion at
any refresh rate).

### Trigger and Query API (P7)

First-class spatial queries on `PhysicsWorld` (brute-force over active colliders with an
AABB reject; mesh colliders dispatch into their triangle BVH; all take a layer/mask
`QueryFilter`):

- `raycast` / `raycastAll` — `Ray{origin, direction, maxDistance}` → `RaycastHit`
  (`collision/ray.hpp`: analytic ray vs sphere/OBB/capsule/convex + Möller–Trumbore
  triangle).
- `shapecast(shape, pose, direction, maxDistance)` → `ShapecastHit` — sweep via GJK
  conservative advancement (`collision/shape_cast.hpp`).
- `overlapSphere(center, radius)` / `overlapShape(shape, pose)` → `OverlapHit`s (reuse
  `gjkEpaContact`).

**Trigger / collision events**: a collider flagged `isTrigger` (on `ColliderDesc`, or glTF
`extras.Physics` `"IsTrigger": true`) is tracked + tested but generates **no solver
response** — overlaps surface as events instead. Each step the overlapping pairs are diffed
against the previous step into enter/stay/exit `ContactEvent`s (carrying the public collider
handles), read after `step()` via `triggerEvents()` (trigger pairs) and `collisionEvents()`
(touching solid pairs). This replaces gameplay's reliance on raw `possiblePairs()`.

### Sleeping and Islands (P5)

Settled dynamic bodies stop simulating until disturbed, and the solve runs per island.

- **Islands** (`physics/island.{hpp,cpp}`): `buildIslands` is a union-find over the
  movable bodies (Dynamic + Kinematic nodes; Static is a boundary), merging the two
  endpoints of every movable-movable contact/joint. Built each step in deterministic
  (body-index) order. `solveAndIntegrate` partitions the contact/joint inputs by island
  and calls `solveIsland` on each — the contact + joint solvers run over just that
  island's subset (they index the shared global `SolverBody` array). Independent islands
  → equivalent to the old global solve, and the unit a future threaded solver splits on.
- **Sleeping**: per `BodyEntry`, a `sleepTimer` accumulates while the body is below
  `kLinearSleepThreshold` / `kAngularSleepThreshold`; `PhysicsBody::allowSleeping` opts a
  body out. `islandShouldSleep` returns true once every dynamic member is eligible
  (timer ≥ `kSleepTime`) and no kinematic member moved this step. A sleeping island has
  its velocities zeroed and skips integration + the solve. Islands are rebuilt from
  current contacts every step, so any awake member — a new contact, a moving kinematic
  platform — wakes the whole island; `PhysicsWorld::wake`/`sleeping`/`sleepingEnabled`
  and wake-on-mutation (`setBodyVelocity`/`setBodyTransform`) complete the API.
- The warm-start cache lifecycle is split into `beginStore` (once) / `store` (per island,
  appends) / `commitStore` (once) so the per-island solves share one frame's cache.
- A debug-draw colour for asleep colliders (`debugColliderSleeping()` →
  `PhysicsDebugData::shapesAsleep`; sleeping bodies dim in the `--debug-physics` overlay).

### Debug Visualization

Immediate-mode wireframe debug drawing (renderer-owned `DebugDraw` line pass, toggled via
`--debug-physics` or the overlay "Physics debug" panel, x-ray or depth-tested):

- broadphase AABBs (`debugColliderBounds()`)
- authored collider shapes (`gatherColliders()` — sphere / box / capsule)
- contact normals and points (`debugContacts()` — real `ContactManifold` points from the
  shape-specific narrowphase)
- sleep state — asleep colliders dim (`debugColliderSleeping()`)

This makes narrowphase and solver work far easier to validate than visual meshes alone.

---

## Future Directions

Genuinely outstanding work, all to build behind the `PhysicsWorld` boundary (never back onto
`scene::Node`):

- **Broadphase incrementalism** — pairs are regenerated in full each step; a moved-leaf-only
  incremental pass would cut broadphase cost on large scenes.
- **Full conservative-advancement CCD** — the speculative contact normal is the closest-feature
  normal at the *start* pose, so a steep corner graze can be slightly off; a full sweep would fix it.
- **BVH-accelerated queries** — `raycast` / `shapecast` / `overlap*` are brute-force over active
  colliders today (mesh colliders already dispatch into their own triangle BVH); routing the outer
  loop through the broadphase BVH would scale them.
- **Quickhull authoring** — `Shape: "ConvexHull"` assumes a convex input mesh (coplanar-merge, not a
  true hull); a quickhull pass would accept arbitrary meshes.
- **Better inertia** — convex inertia is the hull's AABB-box approximation, and compound inertia keeps
  only the tensor diagonal (exact only when symmetric about the body axes). Full off-diagonal tensors
  are a refinement.
- **Primitive-vs-triangle narrowphase** — mesh contacts reuse GJK/EPA + a planar reconstruction; EPA
  point quality degrades for triangles enormous relative to the body. A specialised
  primitive-vs-triangle test would harden it.
- **Gyroscopic torque** (`ω × Iω`) — omitted by design (standard for stable game physics); could be
  added for tumbling accuracy.
- **Layer-coloured debug draw** — colour collider wireframes by collision layer.
- **Ragdoll → animation recovery** — `Ragdoll::deactivate()` just releases the world-override; a blend
  back from the simulated pose into animation isn't wired.
- **Batched physics particles** — a particle-collision system (dense CPU/GPU arrays + emitter batches +
  query-based collision against `PhysicsWorld` shapes), *not* one `Node` / rigid body per particle.
  The rendering-side GPU `ParticleSystem` already exists (see [`roadmap.md`](../roadmap.md) / [`onboarding.md`](onboarding.md)); this is
  the gameplay-collision counterpart.
- **Threaded solver** — islands are the independent unit a parallel solve would split on.
