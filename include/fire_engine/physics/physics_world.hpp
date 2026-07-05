#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fire_engine/collision/aabb_bvh.hpp>
#include <fire_engine/collision/collider.hpp>
#include <fire_engine/collision/contact_manifold.hpp>
#include <fire_engine/collision/ray.hpp>
#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/generational_slot_pool.hpp>
#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/collision_event.hpp>
#include <fire_engine/physics/contact.hpp>
#include <fire_engine/physics/joint.hpp>
#include <fire_engine/physics/physics_body.hpp>
#include <fire_engine/physics/physics_query.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

class BroadPhase;
class ContactSolver;
struct CollisionPair;
struct Island;
class JointSolver;
class NarrowPhase;
struct SolverBody;
struct SolverContactInput;

struct DebugJointAnchor
{
    Vec3 originA{};
    Vec3 originB{};
    Vec3 anchorA{};
    Vec3 anchorB{};
};

class PhysicsWorld
{
public:
    PhysicsWorld();

    // Inject a broadphase implementation (testing / alternative strategies). The
    // default ctor uses a DynamicAabbTreeBroadPhase; this lets a caller substitute, say,
    // a SweepAndPruneBroadPhase without changing any call site.
    explicit PhysicsWorld(std::unique_ptr<BroadPhase> broadPhase);

    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;
    PhysicsWorld(PhysicsWorld&&) noexcept;
    PhysicsWorld& operator=(PhysicsWorld&&) noexcept;

    [[nodiscard]]
    PhysicsBodyHandle createBody(const PhysicsBodyDesc& desc);

    [[nodiscard]]
    PhysicsColliderHandle createCollider(PhysicsBodyHandle bodyHandle, const ColliderDesc& desc);

    // Create a compound collider: one child collider per CompoundChild (each placed at
    // its local offset and registered with the broadphase). For a Dynamic body the
    // children's mass properties are aggregated into the body's centre of mass +
    // inertia (mass split by child volume; parallel-axis sum, diagonalised). Returns
    // the first child's handle (or a null handle if `children` is empty / body invalid).
    [[nodiscard]]
    PhysicsColliderHandle
    createCompoundCollider(PhysicsBodyHandle bodyHandle, std::span<const CompoundChild> children,
                           std::uint32_t collisionLayer = 1U, std::uint32_t collisionMask = ~0U);

    // Create a static triangle-mesh collider. A moving body is resolved against the
    // mesh's actual triangles (via a per-collider triangle BVH), not its bounding box.
    // The body must be Static (returns a null handle otherwise) and is assumed to keep
    // a fixed transform (the triangle BVH is built once in world space).
    [[nodiscard]]
    PhysicsColliderHandle
    createMeshCollider(PhysicsBodyHandle bodyHandle, const StaticMeshShape& mesh,
                       const PhysicsMaterial& material = {}, std::uint32_t collisionLayer = 1U,
                       std::uint32_t collisionMask = ~0U);

    [[nodiscard]]
    bool destroyBody(PhysicsBodyHandle handle);

    // Create a constraint (distance / ball-socket / hinge) between two bodies. The
    // descriptor's anchors/axes are in each body's local frame; both bodies must be
    // valid. Returns a null handle if either body is missing.
    [[nodiscard]]
    PhysicsConstraintHandle createJoint(const JointDesc& desc);

    [[nodiscard]]
    bool destroyJoint(PhysicsConstraintHandle handle);

    // --- Reduced-coordinate articulations (P9 item 5) ---
    //
    // Create an empty articulation and build it through the returned handle's
    // articulation() accessor (addRootLink / addLink), then attachLinkCollider to give a
    // link a broadphase collider. Phase A is kinematic only: link colliders track their
    // forward-kinematics pose and pair in the broadphase / answer spatial queries, but do
    // not yet generate contact response — that is the Phase C ConstraintBody coupling.
    [[nodiscard]]
    PhysicsArticulationHandle createArticulation();

    // Destroy an articulation and all of its link colliders, recycling their slots. Returns
    // false for an unknown/already-destroyed handle. Enables runtime articulation churn
    // (spawning/despawning ragdolls) with bounded storage + stale-handle detection (CR-11/12).
    bool destroyArticulation(PhysicsArticulationHandle handle);

    [[nodiscard]]
    Articulation* articulation(PhysicsArticulationHandle handle) noexcept;

    [[nodiscard]]
    const Articulation* articulation(PhysicsArticulationHandle handle) const noexcept;

    [[nodiscard]]
    std::size_t articulationCount() const noexcept;

    // Attach a collider to articulation link `link`. Its world bounds track the link's
    // forward-kinematics pose each step. Returns a null handle if the articulation or link
    // index is invalid.
    [[nodiscard]]
    PhysicsColliderHandle attachLinkCollider(PhysicsArticulationHandle handle, int link,
                                             const ColliderDesc& desc);

    void clear();
    void step(float fixedDt);

    [[nodiscard]]
    bool valid(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    bool valid(PhysicsColliderHandle handle) const noexcept;

    [[nodiscard]]
    bool valid(PhysicsConstraintHandle handle) const noexcept;

    [[nodiscard]]
    std::size_t bodyCount() const noexcept;

    [[nodiscard]]
    std::size_t colliderCount() const noexcept;

    [[nodiscard]]
    std::size_t jointCount() const noexcept;

    // World-space collision primitives for the cloth/soft-body solver — each
    // active collider's shape composed with its body transform. Vulkan-free
    // output (mirrors the gatherLights / gatherEmitters pattern).
    [[nodiscard]]
    std::vector<ClothCollider> gatherColliders() const;

    // --- Debug visualisation (read-only, Vulkan-free) ---
    //
    // World-space AABBs of every active collider (broadphase bounds), and the
    // contacts generated by the most recent step() (approximate point + normal).
    // The renderer's debug-draw subsystem consumes these; gatherColliders()
    // supplies the authored collider shapes.
    [[nodiscard]]
    std::vector<AABB> debugColliderBounds() const;

    // Sleep flag (1 = asleep) per shape emitted by gatherColliders(), in the SAME
    // order, so the debug draw can colour sleeping bodies distinctly.
    [[nodiscard]]
    std::vector<std::uint8_t> debugColliderSleeping() const;

    [[nodiscard]]
    const std::vector<DebugContact>& debugContacts() const noexcept
    {
        return debugContacts_;
    }

    [[nodiscard]]
    std::vector<DebugJointAnchor> debugJointAnchors() const;

    // Overlap-lifecycle events from the most recent step(). triggerEvents covers pairs
    // where at least one collider isTrigger (no solver response); collisionEvents covers
    // touching solid pairs. Both are rebuilt each step (enter/stay/exit) — read them
    // after step().
    [[nodiscard]]
    const std::vector<ContactEvent>& triggerEvents() const noexcept
    {
        return triggerEvents_;
    }

    [[nodiscard]]
    const std::vector<ContactEvent>& collisionEvents() const noexcept
    {
        return collisionEvents_;
    }

    [[nodiscard]]
    const PhysicsBody* body(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    PhysicsBody* body(PhysicsBodyHandle handle) noexcept;

    [[nodiscard]]
    std::optional<Transform> bodyTransform(PhysicsBodyHandle handle) const noexcept;

    // Render-interpolated transform: blends the body's pose at the start of the most recent
    // step() towards its current pose by `alpha` (position lerp, orientation slerp). With a
    // fixed 60 Hz sim driving a faster display, `alpha = accumulator / fixedDt` renders the
    // in-between frames smoothly instead of snapping to the last simulated state. `alpha` is
    // clamped to [0, 1]; alpha == 1 reproduces bodyTransform().
    [[nodiscard]]
    std::optional<Transform> interpolatedBodyTransform(PhysicsBodyHandle handle,
                                                       float alpha) const noexcept;

    void setBodyTransform(PhysicsBodyHandle handle, const Transform& transform) noexcept;
    void setBodyVelocity(PhysicsBodyHandle handle, Vec3 velocity) noexcept;

    // Wake a sleeping body (and reset its sleep timer). A no-op for a body that is
    // already awake or invalid. Mutating a body's velocity/transform wakes it too.
    void wake(PhysicsBodyHandle handle) noexcept;

    // Whether a body is currently asleep (skipped by the solver until disturbed).
    [[nodiscard]]
    bool sleeping(PhysicsBodyHandle handle) const noexcept;

    // Global on/off for the sleeping system (default on). Disabling wakes nothing
    // already asleep but stops new islands from sleeping.
    void sleepingEnabled(bool enabled) noexcept
    {
        sleepingEnabled_ = enabled;
    }

    [[nodiscard]]
    bool sleepingEnabled() const noexcept
    {
        return sleepingEnabled_;
    }

    // --- Spatial queries (read-only; brute-force over active colliders) ---
    //
    // All filter colliders by layer/mask (see QueryFilter) and test the exact world
    // shape after an AABB reject. Mesh colliders are tested against their triangle BVH.

    // Nearest collider hit along `ray` (within ray.maxDistance), or nullopt.
    [[nodiscard]]
    std::optional<RaycastHit> raycast(const Ray& ray, QueryFilter filter = {}) const;

    // Every collider hit along `ray` (nearest hit per collider), unsorted.
    [[nodiscard]]
    std::vector<RaycastHit> raycastAll(const Ray& ray, QueryFilter filter = {}) const;

    // Sweep `shape` (posed by `pose`) along unit `direction` up to `maxDistance` and
    // return the first collider reached, or nullopt.
    [[nodiscard]]
    std::optional<ShapecastHit> shapecast(const ColliderShape& shape, const Transform& pose,
                                          Vec3 direction, float maxDistance,
                                          QueryFilter filter = {}) const;

    // Colliders overlapping `shape` (posed by `pose`) / a sphere.
    [[nodiscard]]
    std::vector<OverlapHit> overlapShape(const ColliderShape& shape, const Transform& pose,
                                         QueryFilter filter = {}) const;

    [[nodiscard]]
    std::vector<OverlapHit> overlapSphere(Vec3 center, float radius, QueryFilter filter = {}) const;

    [[nodiscard]]
    const std::vector<CollisionPair>& possiblePairs() const noexcept;

    [[nodiscard]]
    bool validateBroadPhase() const;

private:
    struct BodyEntry
    {
        PhysicsBodyHandle handle;
        PhysicsBody body;
        Transform transform;
        // Pose at the start of the most recent step(), kept so rendering can interpolate
        // between it and `transform` (CR-20). Seeded to `transform` at creation and reseeded
        // on any external reposition so no interpolation spans a teleport.
        Transform previousRenderTransform;
        Vec3 previousPosition{};
        bool active{true};
        std::vector<PhysicsColliderHandle> colliders;
        // Sleeping (P5): `sleeping` skips this body from integration + solving; the
        // timer accumulates while the body is below the sleep thresholds and resets
        // when it moves or is woken. Driven per-island in solveAndIntegrate.
        bool sleeping{false};
        float sleepTimer{0.0f};
    };

    // Per-collider triangle-mesh data (static mesh colliders only): the triangles in
    // world space + a BVH over them. Built once at createMeshCollider (the mesh body is
    // assumed not to move). Shared so ColliderEntry stays copyable.
    struct MeshCollisionData
    {
        std::vector<Vec3> worldVertices;
        std::vector<std::uint32_t> indices;
        AabbBvh<int> bvh{0.0f}; // triangle index → world AABB
    };

    struct ColliderEntry
    {
        PhysicsColliderHandle handle;
        // Owner is exactly one of: a rigid body (`body` valid) or an articulation link
        // (`articulation` valid + `link` ≥ 0). The two are mutually exclusive; the owner's
        // world pose (body transform or link forward-kinematics) drives worldShape and the
        // swept-bound update. A link collider's `body` is the null handle, which the
        // narrowphase pair guard treats as "no rigid response" (Phase C couples it instead).
        PhysicsBodyHandle body;
        PhysicsArticulationHandle articulation;
        int link{-1};
        Collider collider;
        ColliderShape shape;
        PhysicsMaterial material;
        // Offset of this collider within the body (a compound child); identity for a
        // plain single collider. worldShape composes it after the body transform.
        Vec3 localPosition{};
        Quaternion localRotation{Quaternion::identity()};
        bool active{true};
        // Non-null for a static triangle-mesh collider; null otherwise.
        std::shared_ptr<MeshCollisionData> mesh;

        [[nodiscard]]
        bool isLinkCollider() const noexcept
        {
            return articulation.valid();
        }
    };

    struct JointEntry
    {
        PhysicsConstraintHandle handle;
        JointDesc desc;
        // Relative orientation of B in A's frame at creation time — the rest frame
        // the angular limits are measured from.
        Quaternion restRelative{};
        bool active{true};
    };

    struct ContactCandidate
    {
        BodyEntry* moving{nullptr};
        BodyEntry* target{nullptr};
        ColliderEntry* movingCollider{nullptr};
        ColliderEntry* targetCollider{nullptr};
    };

    struct SolverContact
    {
        ContactManifold manifold; // normal points target -> moving; push moving out
        BodyEntry* moving{nullptr};
        BodyEntry* target{nullptr};
        ColliderEntry* movingCollider{nullptr};
        ColliderEntry* targetCollider{nullptr};
        // Distinguishes several contacts from one collider pair (a mesh triangle index
        // + 1); 0 for an ordinary single-manifold pair. Folded into the warm-start key.
        std::uint32_t subKey{0};
    };

    std::vector<BodyEntry> bodies_;
    std::deque<ColliderEntry> colliders_;
    std::vector<JointEntry> joints_;
    // Slot lifecycle (index + generation) for the entry containers. acquire() recycles a
    // released slot or grows; release() bumps the slot's generation so a stale handle to a
    // recycled slot is detectably invalid (CR-11 bounded storage + CR-12 generations). The
    // handle *encodes* its slot index + generation, so no handle→index side-table is needed.
    GenerationalSlotPool bodySlots_;
    GenerationalSlotPool colliderSlots_;
    GenerationalSlotPool jointSlots_;
    GenerationalSlotPool articulationSlots_;
    // Articulations live in a deque so the Articulation references handed out via
    // articulation() stay stable as slots are added/recycled (links/colliders cache the handle,
    // not a pointer). Indexed directly by the handle's slot index (articulationSlots_); a freed
    // slot holds a default (0-link) Articulation until reused. Sleep state is kept in the two
    // parallel arrays below, sized 1:1 with articulations_ and indexed by the same slot.
    std::deque<Articulation> articulations_;
    std::vector<float> articulationSleepTimers_;
    std::vector<std::uint8_t> articulationSleeping_;
    // Collider address → slot index, so a broadphase pair's `Collider*` resolves back to its
    // entry. (Bodies/colliders/joints no longer need a handle→index map — the handle is the
    // index; this side-table stays because the broadphase reports colliders by pointer.)
    std::unordered_map<const Collider*, std::size_t> colliderIndexByPointer_;
    // Owned via the BroadPhase interface so the implementation is swappable. Defaults to
    // the dynamic AABB tree; inject an alternative (e.g. SweepAndPruneBroadPhase) through
    // the unique_ptr constructor.
    std::unique_ptr<BroadPhase> broadPhase_;
    std::unique_ptr<NarrowPhase> narrowPhase_;
    std::unique_ptr<ContactSolver> solver_;
    std::unique_ptr<JointSolver> jointSolver_;
    Vec3 gravity_{0.0f, -9.81f, 0.0f};
    bool sleepingEnabled_{true};
    // Contacts from the most recent step(), kept for debug visualisation only.
    std::vector<DebugContact> debugContacts_;

    // Overlap tracking for trigger/collision events: the set of overlapping collider
    // pairs (ordered-id key) this step and last, diffed into enter/stay/exit events.
    std::unordered_set<std::uint64_t> triggerOverlaps_;
    std::unordered_set<std::uint64_t> previousTriggerOverlaps_;
    std::unordered_set<std::uint64_t> collisionOverlaps_;
    std::unordered_set<std::uint64_t> previousCollisionOverlaps_;
    std::vector<ContactEvent> triggerEvents_;
    std::vector<ContactEvent> collisionEvents_;

    // Record an overlapping collider pair (ordered key) into the current trigger /
    // collision set.
    void recordOverlap(PhysicsColliderHandle first, PhysicsColliderHandle second, bool trigger);
    void removeOverlapPairsForCollider(PhysicsColliderHandle collider);
    // Diff current vs previous overlap sets into enter/stay/exit events, then roll
    // current → previous for the next step.
    void updateOverlapEvents();
    void deactivateCollider(ColliderEntry& collider);
    void deactivateJoint(std::size_t jointIndex);
    void deactivateJointsForBody(PhysicsBodyHandle body);

    // Create a Collider + ColliderEntry for `shape` at a local offset within `owner`,
    // register it with the broadphase, and return its handle. Shared by createCollider
    // (identity offset) and createCompoundCollider (per-child offsets). Does not touch
    // the body's mass properties — the caller aggregates those.
    [[nodiscard]]
    PhysicsColliderHandle addColliderEntry(BodyEntry& owner, const ColliderShape& shape,
                                           const PhysicsMaterial& material,
                                           std::uint32_t collisionLayer,
                                           std::uint32_t collisionMask, const Vec3& localPosition,
                                           const Quaternion& localRotation, bool isTrigger = false);

    // Articulation-link variant of addColliderEntry: the owner is link `link` of
    // `articulation` rather than a rigid body. Registers with the broadphase the same way;
    // the collider's bounds seed from the link's current forward-kinematics pose.
    [[nodiscard]]
    PhysicsColliderHandle addLinkColliderEntry(
        PhysicsArticulationHandle articulation, int link, const ColliderShape& shape,
        const PhysicsMaterial& material, std::uint32_t collisionLayer, std::uint32_t collisionMask,
        const Vec3& localPosition, const Quaternion& localRotation, bool isTrigger);

    [[nodiscard]]
    Articulation* findArticulation(PhysicsArticulationHandle handle) noexcept;

    [[nodiscard]]
    const Articulation* findArticulation(PhysicsArticulationHandle handle) const noexcept;

    // World pose of a collider's owner — the rigid-body transform, or the articulation
    // link's forward-kinematics transform — resolved uniformly so worldShape / the
    // swept-bound update don't branch on owner kind at every call site. `scale` is the
    // body scale (1 for a link collider, which carries no scale).
    struct OwnerPose
    {
        Mat4 world{Mat4::identity()};
        Quaternion rotation{Quaternion::identity()};
        Vec3 scale{1.0f, 1.0f, 1.0f};
        bool valid{false};
    };

    [[nodiscard]]
    OwnerPose colliderOwnerPose(const ColliderEntry& entry) const;

    [[nodiscard]]
    BodyEntry* findBody(PhysicsBodyHandle handle) noexcept;

    [[nodiscard]]
    const BodyEntry* findBody(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    ColliderEntry* findCollider(PhysicsColliderHandle handle) noexcept;

    [[nodiscard]]
    const ColliderEntry* findCollider(PhysicsColliderHandle handle) const noexcept;

    [[nodiscard]]
    ColliderEntry* findCollider(const Collider* collider) noexcept;

    // Compose each active joint's local anchors/axes with the current body
    // transforms into the world-space JointInputs the solver consumes (skipping any
    // joint whose bodies are no longer valid). `bodyIndex` maps a body handle to its
    // solver-body index.
    [[nodiscard]]
    std::vector<JointInput> buildJointInputs() const;

    [[nodiscard]]
    AABB localBounds(const ColliderShape& shape) const noexcept;

    [[nodiscard]]
    static WorldShape composeWorldShape(const ColliderShape& shape, const Mat4& world,
                                        const Quaternion& rot, const Vec3& scale);

    [[nodiscard]]
    static AABB aabbOfWorldShape(const WorldShape& shape) noexcept;

    // Compose a collider's authored shape with its body's world transform into a
    // neutral world-space shape for the narrowphase (and gatherColliders).
    [[nodiscard]]
    WorldShape worldShape(const ColliderEntry& entry) const;

    // As worldShape, but against an explicit owner pose rather than the stored body
    // transform — the mid-step manifold refresh composes shapes at the in-flight
    // SolverBody pose (P9.6).
    [[nodiscard]]
    WorldShape worldShapeAt(const ColliderEntry& entry, const OwnerPose& owner) const;

    // Shared overlap core for overlapShape / overlapSphere: every active collider whose
    // bounds intersect `queryAabb` and which actually overlaps `query`.
    [[nodiscard]]
    std::vector<OverlapHit> overlapWorldShape(const WorldShape& query, const AABB& queryAabb,
                                              QueryFilter filter) const;

    void updateCollider(ColliderEntry& collider, float dt);
    void resetCollider(ColliderEntry& collider);
    void updateColliders(float dt);
    void resetResolvedColliders();
    void capturePreviousPositions() noexcept;

    // Snapshot each body's (and articulation's) current pose as the render-interpolation
    // baseline, called at the start of step() before anything advances (CR-20).
    void captureRenderBaseline() noexcept;

    // Reduced-coordinate articulation dynamics (Phase F1). Steps every articulation one
    // fixed step under gravity + joint damping, resolving link-collider contacts against
    // *static* rigid colliders through the ConstraintBody seam (link-vs-dynamic and
    // self-collision are deferred). Contacts come from the same broadphase as the rigid
    // path; each manifold point becomes a per-link plane contact tracked across substeps.
    // Runs in its own solve pass, separate from the rigid islands.
    void stepArticulations(float dt);

    [[nodiscard]]
    std::vector<SolverContact> contacts(float dt);

    // Rebuild debugContacts_ from this step's solver contacts (real manifold
    // points + normal), before the solver mutates positions.
    void captureDebugContacts(std::span<const SolverContact> contacts);

    // Append the solver contact(s) for one broadphase pair to `out`: a single manifold
    // for a primitive/convex pair, or one per overlapping triangle for a static-mesh
    // pair (mesh as the target). Nothing when there's no contact.
    void appendContactsForPair(const CollisionPair& pair, float dt,
                               std::vector<SolverContact>& out);

    [[nodiscard]]
    std::optional<SolverContact> singleContact(const ContactCandidate& candidate, float dt);

    // Collide the moving collider against the target static-mesh triangles. Pushes one
    // contact per overlapping triangle into `out` when non-null (a trigger pair passes
    // null to suppress the solver response). Returns whether any real overlap occurred
    // (for the overlap-event sets).
    bool appendMeshContacts(const ContactCandidate& candidate, float dt,
                            std::vector<SolverContact>* out);

    [[nodiscard]]
    std::optional<ContactCandidate> contactCandidateForPair(const CollisionPair& pair);

    // Integrate constrained velocities + positions for this step: build the global
    // SolverBody view + contact/joint inputs from the frame's contacts, partition the
    // dynamic bodies into islands, solve + integrate each island independently, then
    // write the results back onto the bodies. Returns true if any body moved.
    bool solveAndIntegrate(std::span<const SolverContact> contacts, float dt);

    // Solve one island: run the contact + joint solvers over the island's input
    // subset (indexing the shared `solverBodies` array), then integrate the island's
    // dynamic bodies' velocities into positions/orientations and position-correct.
    // `solverContacts` is the step's full SolverContact list (same indexing as
    // `contactInputs`) — the mid-step manifold refresh re-collides through it.
    void solveIsland(const Island& island, std::vector<SolverBody>& solverBodies,
                     std::span<const SolverContactInput> contactInputs,
                     std::span<const SolverContact> solverContacts,
                     std::span<const JointInput> jointInputs, float dt);

    // Mid-step manifold refresh (P9.6 stage 1). Once per step, at substep
    // kSubstepCount/2: for each of the island's non-mesh contacts whose either body is
    // Dynamic and sweeping more than kSubstepRefreshRotation this step (|ω|·dt at the
    // current solver velocity), re-collide the pair at the in-flight SolverBody poses
    // and overwrite that entry's manifold in `islandContacts` (empty result ⇒
    // pointCount 0, the rows drop). Sets `flags[k] = 1` per refreshed entry; returns
    // whether anything was refreshed (the caller then re-prepares the flagged solver
    // rows via ContactSolver::refresh). A stale step-start manifold under fast rotation
    // concentrates impact impulses on wrongly-placed points — the "settle snap" /
    // rotational-tunnelling family.
    bool refreshIslandContacts(const Island& island, std::span<const SolverBody> solverBodies,
                               std::span<const SolverContact> solverContacts,
                               std::vector<SolverContactInput>& islandContacts,
                               std::vector<std::uint8_t>& flags, float dt, float remaining);

    // Whether the whole island may sleep this step: sleeping enabled, every dynamic
    // member allows sleeping and has been below the thresholds for kSleepTime, and no
    // kinematic member moved (a moving platform keeps its riders awake). An already-
    // asleep island stays asleep because its members keep their elapsed timers.
    [[nodiscard]]
    bool islandShouldSleep(const Island& island) const;

    [[nodiscard]]
    static bool movable(const BodyEntry& body) noexcept;

    // Velocity-pass inverse mass: the body's inverse mass for Dynamic, 0 for
    // Static/Kinematic (contact impulses never shove a scene-driven body).
    [[nodiscard]]
    static float velocityInvMass(const BodyEntry& body) noexcept;

    // Split-impulse positional-correction weight: 0 for static (immovable), the
    // inverse mass for dynamic, a nominal 1 for kinematic (so it slides out of
    // penetration).
    [[nodiscard]]
    static float positionWeight(const BodyEntry& body) noexcept;
};

} // namespace fire_engine
