#include <catch2/catch_test_macros.hpp>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/physics/spatial.hpp>

using namespace fire_engine;

namespace
{

// Build a planar two-link revolute chain in the XY plane, each joint about +Z:
//   base (root) → link1 (joint 1m along +X, body 1m further) → link2 (same offsets).
// Returns the articulation with q all zero (call forwardKinematics / set q in the test).
Articulation makeTwoLinkChain()
{
    Articulation arm;
    arm.addRootLink(ArticulationLinkDesc{});

    ArticulationLinkDesc link;
    link.parent = 0;
    link.joint = ArticulationJointType::Revolute;
    link.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    link.parentToJoint = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    link.jointToChild = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    arm.addLink(link); // index 1

    link.parent = 1;
    arm.addLink(link); // index 2
    return arm;
}

} // namespace

TEST_CASE("Articulation.DofCountTracksRevoluteJoints", "[Articulation]")
{
    Articulation arm;
    arm.addRootLink(ArticulationLinkDesc{});
    CHECK(arm.linkCount() == 1);
    CHECK(arm.dofCount() == 0); // a floating base adds no generalized coordinates here

    ArticulationLinkDesc revolute;
    revolute.parent = 0;
    revolute.joint = ArticulationJointType::Revolute;
    arm.addLink(revolute);
    CHECK(arm.dofCount() == 1);
    CHECK(arm.linkDofOffset(1) == 0);

    ArticulationLinkDesc welded;
    welded.parent = 1;
    welded.joint = ArticulationJointType::Fixed;
    arm.addLink(welded); // a Fixed joint adds no DOF
    CHECK(arm.linkCount() == 3);
    CHECK(arm.dofCount() == 1);
    CHECK(arm.linkDofOffset(2) == -1);
}

TEST_CASE("Articulation.ForwardKinematicsMatchesHandComputedPose", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain();

    // q all zero: the chain lies straight along +X. Link1 body at (2,0,0), link2 at (4,0,0).
    arm.forwardKinematics();
    CHECK(arm.linkWorld(0).translation.approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{2.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{4.0f, 0.0f, 0.0f}, 1e-5f));

    // Bend the first joint 90° about +Z; leave the second straight. Hand-computed:
    //   link1 swings to (1,1,0) rotated +90°; link2 to (1,3,0) carrying that rotation.
    arm.q(0, pi / 2.0f);
    arm.forwardKinematics();
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{1.0f, 1.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{1.0f, 3.0f, 0.0f}, 1e-5f));

    // The link's orientation is the accumulated +90° about Z: it maps local +X onto world +Y.
    const Vec3 link1X = arm.linkWorld(1).transformDirection(Vec3{1.0f, 0.0f, 0.0f});
    CHECK(link1X.approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-5f));
}

TEST_CASE("Articulation.FloatingBaseOffsetsEveryLink", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain();
    arm.q(0, pi / 2.0f);

    // Translate the floating base by (10,0,0): every link's world pose shifts by it.
    arm.baseTransform(RigidTransform{Quaternion::identity(), Vec3{10.0f, 0.0f, 0.0f}});
    arm.forwardKinematics();
    CHECK(arm.linkWorld(0).translation.approxEqual(Vec3{10.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{11.0f, 1.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{11.0f, 3.0f, 0.0f}, 1e-5f));
}

TEST_CASE("Articulation.FloatingBaseRotationCarriesThroughChain", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain(); // q all zero ⇒ straight along local +X

    // Rotate the base +90° about Z: the straight chain now lies along world +Y.
    arm.baseTransform(
        RigidTransform{Quaternion::fromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, pi / 2.0f), Vec3{}});
    arm.forwardKinematics();
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{0.0f, 2.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{0.0f, 4.0f, 0.0f}, 1e-5f));
}

TEST_CASE("RigidTransform.ComposeAndInverseRoundTrip", "[Articulation]")
{
    const RigidTransform t{Quaternion::fromAxisAngle(Vec3{0.0f, 1.0f, 0.0f}, 0.7f),
                           Vec3{3.0f, -2.0f, 1.0f}};
    const Vec3 p{1.0f, 2.0f, 3.0f};

    // inverse() undoes the transform; (t * t.inverse()) is the identity map.
    const Vec3 roundTrip = t.inverse().transformPoint(t.transformPoint(p));
    CHECK(roundTrip.approxEqual(p, 1e-5f));

    const RigidTransform composed = t * t.inverse();
    CHECK(composed.transformPoint(p).approxEqual(p, 1e-5f));
}

TEST_CASE("PhysicsWorld.LinkColliderTracksForwardKinematicsInBroadphase", "[Articulation]")
{
    // A collider owned by an articulation link must register in the broadphase and answer
    // spatial queries at the link's forward-kinematics pose — the Phase A gate. The chain:
    // floating base at (0,5,0) → revolute-Z link, joint 1m along +X, body 1m beyond. With
    // q = 0 the link body sits at (2,5,0); bending the joint +90° swings it to (1,6,0).
    PhysicsWorld world;
    const PhysicsArticulationHandle arm = world.createArticulation();
    REQUIRE(world.articulationCount() == 1);

    Articulation* art = world.articulation(arm);
    REQUIRE(art != nullptr);
    art->addRootLink(ArticulationLinkDesc{});

    ArticulationLinkDesc link;
    link.parent = 0;
    link.joint = ArticulationJointType::Revolute;
    link.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    link.parentToJoint = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    link.jointToChild = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    const int linkIndex = art->addLink(link);
    art->baseTransform(RigidTransform{Quaternion::identity(), Vec3{0.0f, 5.0f, 0.0f}});

    ColliderDesc box;
    box.shape = BoxShape{Vec3{0.4f, 0.4f, 0.4f}};
    const PhysicsColliderHandle collider = world.attachLinkCollider(arm, linkIndex, box);
    REQUIRE(collider.valid());

    // step() runs forward kinematics, then refreshes the link collider's swept bounds.
    world.step(1.0f / 60.0f);

    const auto atRest = world.overlapSphere(Vec3{2.0f, 5.0f, 0.0f}, 0.2f);
    REQUIRE(atRest.size() == 1);
    CHECK(atRest[0].collider == collider);
    // A link collider has no rigid body — its owner is the articulation, so body is null.
    CHECK(!atRest[0].body.valid());
    // Nothing where the link is *not*.
    CHECK(world.overlapSphere(Vec3{1.0f, 6.0f, 0.0f}, 0.2f).empty());

    // Bend the joint +90°: the link (and its collider) swing to (1,6,0); the broadphase
    // tracks it on the next step.
    art->q(0, pi / 2.0f);
    world.step(1.0f / 60.0f);

    const auto bent = world.overlapSphere(Vec3{1.0f, 6.0f, 0.0f}, 0.2f);
    REQUIRE(bent.size() == 1);
    CHECK(bent[0].collider == collider);
    CHECK(world.overlapSphere(Vec3{2.0f, 5.0f, 0.0f}, 0.2f).empty());
}

TEST_CASE("SpatialVector.ArithmeticAndDot", "[Articulation]")
{
    const SpatialVector a{Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 2.0f, 0.0f}};
    const SpatialVector b{Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 0.0f, 3.0f}};

    CHECK((a + b).angular.approxEqual(Vec3{1.0f, 1.0f, 0.0f}, 1e-6f));
    CHECK((a + b).linear.approxEqual(Vec3{0.0f, 2.0f, 3.0f}, 1e-6f));
    CHECK((a * 2.0f).linear.approxEqual(Vec3{0.0f, 4.0f, 0.0f}, 1e-6f));

    // Spatial dot pairs angular·angular + linear·linear: here 0 + 0 = 0 (orthogonal parts).
    CHECK(a.dot(b) == 0.0f);
    // Self-dot is the sum of squared parts: 1 + 4 = 5.
    CHECK(a.dot(a) == 5.0f);
}
