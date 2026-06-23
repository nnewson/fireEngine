#include <fire_engine/collision/gjk_epa.hpp>

#include <algorithm>
#include <cmath>
#include <random>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/collision/world_shape.hpp>

using fire_engine::ConvexContact;
using fire_engine::gjkEpaContact;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldCapsule;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

namespace
{

WorldBox unitBox(Vec3 center)
{
    return WorldBox{center, {1.0f, 1.0f, 1.0f}, Quaternion::identity()};
}

// Closest point on an axis-aligned box (centre/half) to p, per-axis clamp.
Vec3 closestOnAabb(Vec3 p, Vec3 center, Vec3 half)
{
    return {std::clamp(p.x(), center.x() - half.x(), center.x() + half.x()),
            std::clamp(p.y(), center.y() - half.y(), center.y() + half.y()),
            std::clamp(p.z(), center.z() - half.z(), center.z() + half.z())};
}

} // namespace

TEST_CASE("GjkEpa.SeparatedBoxesReportGapAndWitnesses", "[GjkEpa]")
{
    // Two unit boxes 1.0 apart along x (faces at x=1 and x=2).
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({3.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    CHECK_FALSE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-4f)); // gap distance
    // Witnesses on the facing faces; normal points B -> A = -x.
    CHECK(c.pointA.x() == Catch::Approx(1.0f).margin(1e-3f));
    CHECK(c.pointB.x() == Catch::Approx(2.0f).margin(1e-3f));
    CHECK(c.normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-3f));
}

TEST_CASE("GjkEpa.SeparatedSphereBoxDistance", "[GjkEpa]")
{
    // Sphere radius 0.5 at x=2.5; box spans [-1,1]. Closest points: box face x=1,
    // sphere surface x=2.0 → gap 1.0.
    const WorldShape box{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape sphere{WorldSphere{{2.5f, 0.0f, 0.0f}, 0.5f}};

    const ConvexContact c = gjkEpaContact(box, sphere);
    CHECK_FALSE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-3f));
}

TEST_CASE("GjkEpa.OverlappingBoxesReportEpaDepthAndNormal", "[GjkEpa]")
{
    // Overlapping unit boxes (centres 1.0 apart along x, each half-extent 1.0):
    // x overlap is 1.0 (the minimum axis), y/z overlap 2.0. EPA → depth 1, normal
    // along x. Normal points b -> a: a is at -x of b, so -x.
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({1.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    REQUIRE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-3f));
    CHECK(c.normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-3f));
}

TEST_CASE("GjkEpa.OverlappingBoxesDeepYAxis", "[GjkEpa]")
{
    // Boxes overlapping most along x but the minimum-penetration axis is y here:
    // a at origin, b shifted (0.2, 1.6, 0) → y overlap 0.4 (min), normal +y? b is
    // above a, so b -> a normal is -y.
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({0.2f, 1.6f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    REQUIRE(c.colliding);
    CHECK(c.depth == Catch::Approx(0.4f).margin(1e-2f));
    CHECK(c.normal.approxEqual(Vec3{0.0f, -1.0f, 0.0f}, 1e-2f));
}

TEST_CASE("GjkEpa.TouchingReportsColliding", "[GjkEpa]")
{
    // Boxes exactly touching (faces coincident at x=1).
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({2.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    // Touching is the boundary; either a ~0 gap or colliding is acceptable.
    if (!c.colliding)
    {
        CHECK(c.depth == Catch::Approx(0.0f).margin(1e-3f));
    }
}

// --- Scale-invariance property/fuzz suite ----------------------------------------------
//
// gjkEpaContact must be correct across shape scales. Large shapes (half-extent >= 3) are
// the regime where it previously failed (false collisions, inverted normals) — these
// randomized cases at scales 0.01..100 cross-check against analytic ground truth.

TEST_CASE("GjkEpa.Fuzz.SphereSphereAcrossScales", "[GjkEpa]")
{
    std::mt19937 rng{0xC0FFEEu};
    std::uniform_real_distribution<float> pos{-2.5f, 2.5f};
    std::uniform_real_distribution<float> rad{0.2f, 1.0f};

    for (const float scale : {0.01f, 0.1f, 1.0f, 10.0f, 100.0f})
    {
        const float tol = 2e-3f * scale; // relative to the problem size
        for (int i = 0; i < 200; ++i)
        {
            const Vec3 c1{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const Vec3 c2{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const float r1 = rad(rng) * scale;
            const float r2 = rad(rng) * scale;

            const ConvexContact c =
                gjkEpaContact(WorldShape{WorldSphere{c1, r1}}, WorldShape{WorldSphere{c2, r2}});

            const float dist = (c1 - c2).magnitude();
            const float gap = dist - r1 - r2; // <0 overlapping
            INFO("scale=" << scale << " dist=" << dist << " gap=" << gap << " depth=" << c.depth);

            // Away from the touching boundary the collide flag is unambiguous.
            if (std::abs(gap) > tol)
            {
                CHECK(c.colliding == (gap < 0.0f));
            }
            if (gap > tol)
            {
                // Separated: GJK distance is exact for any convex → assert gap + normal
                // (B -> A = (c1 - c2) normalised). This is where the scale bug lives.
                CHECK(c.depth == Catch::Approx(gap).margin(tol));
                CHECK(c.normal.approxEqual((c1 - c2) * (1.0f / dist), 5e-3f));
            }
            else if (gap < -tol)
            {
                // Overlapping: EPA depth on a sphere is a polytope approximation, so only
                // the sign is asserted here (depth accuracy for curved shapes is separate).
                CHECK(c.depth > 0.0f);
            }
        }
    }
}

TEST_CASE("GjkEpa.Fuzz.SphereBoxAcrossScales", "[GjkEpa]")
{
    std::mt19937 rng{0xBADF00Du};
    std::uniform_real_distribution<float> pos{-2.5f, 2.5f};
    std::uniform_real_distribution<float> half{0.5f, 2.0f};
    std::uniform_real_distribution<float> rad{0.2f, 1.0f};

    for (const float scale : {0.01f, 0.1f, 1.0f, 10.0f, 100.0f})
    {
        const float tol = 3e-3f * scale;
        for (int i = 0; i < 200; ++i)
        {
            const Vec3 sc{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const float r = rad(rng) * scale;
            const Vec3 bc{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const Vec3 bh{half(rng) * scale, half(rng) * scale, half(rng) * scale};

            // a = sphere, b = box → normal points box -> sphere.
            const ConvexContact c =
                gjkEpaContact(WorldShape{WorldSphere{sc, r}},
                              WorldShape{WorldBox{bc, bh, Quaternion::identity()}});

            const Vec3 cp = closestOnAabb(sc, bc, bh);
            const float dist = (sc - cp).magnitude();
            const bool inside = dist < 1e-6f * scale; // sphere centre inside the box
            const float gap = dist - r;
            // Codimension of the closest box feature: 1 clamped axis = face, 2 = edge,
            // 3 = corner. GJK is exact on faces; edges/corners hit a known robustness
            // limitation (see the [!shouldfail] case below + roadmap P7.5).
            const int clamped = (std::abs(cp.x() - sc.x()) > 1e-6f * scale ? 1 : 0) +
                                (std::abs(cp.y() - sc.y()) > 1e-6f * scale ? 1 : 0) +
                                (std::abs(cp.z() - sc.z()) > 1e-6f * scale ? 1 : 0);
            INFO("scale=" << scale << " gap=" << gap << " depth=" << c.depth
                          << " clamped=" << clamped);

            if (!inside && std::abs(gap) > tol)
            {
                CHECK(c.colliding == (gap < 0.0f));
            }
            if (!inside && gap > tol)
            {
                if (clamped <= 1)
                {
                    // Face-closest: exact closest-distance + normal (box -> sphere).
                    CHECK(c.depth == Catch::Approx(gap).margin(tol));
                    CHECK(c.normal.approxEqual((sc - cp) * (1.0f / dist), 5e-3f));
                }
                else
                {
                    // Edge/corner-closest: GJK currently over-estimates the gap (converges
                    // to a corner, never under-estimates dangerously). Sanity bound only;
                    // the exact assertion lives in the [!shouldfail] case below.
                    CHECK(c.depth >= gap - tol);
                }
            }
            CHECK(c.depth >= -tol);
        }
    }
}

TEST_CASE("GjkEpa.LargeFloorNormalRegression", "[GjkEpa]")
{
    // The character-controller bug: a capsule just *above* a large floor box must report a
    // SEPARATED contact with an upward (+y) normal and the correct gap. Previously the
    // witnesses/normal went garbage for large floors → an inverted (-y) normal that froze
    // the controller. The capsule bottom sits 0.1 above the floor top (gap 0.1).
    for (const float floorHalf : {0.5f, 2.0f, 5.0f, 10.0f, 20.0f})
    {
        const WorldBox floor{{0.0f, -floorHalf, 0.0f},
                             {floorHalf, floorHalf, floorHalf},
                             Quaternion::identity()};                             // top at y = 0
        const WorldCapsule capsule{{0.0f, 0.4f, 0.0f}, {0.0f, 1.4f, 0.0f}, 0.3f}; // bottom y=0.1

        const ConvexContact c = gjkEpaContact(WorldShape{capsule}, WorldShape{floor});
        INFO("floorHalf=" << floorHalf << " colliding=" << c.colliding << " depth=" << c.depth
                          << " n=(" << c.normal.x() << "," << c.normal.y() << "," << c.normal.z()
                          << ")");
        CHECK_FALSE(c.colliding);
        CHECK(c.depth == Catch::Approx(0.1f).margin(2e-3f));
        CHECK(c.normal.approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 5e-3f)); // floor -> capsule = +y
    }
}

// KNOWN BUG — roadmap P7.5 (blocks P8). When a sphere is closest to a box *edge*, the GJK
// distance sub-algorithm converges to a box *corner* instead of the edge, so the reported
// gap is over-estimated and the normal carries a spurious component. This case fuzzes
// edge-closest configs and asserts the *correct* (exact) result, which GJK currently gets
// wrong on a subset of them — so it is tagged `[!shouldfail]` (Catch2 inverts it: the case
// PASSES because some assertions fail). The signed-volumes rewrite (P7.5) makes every
// assertion pass → this case then *fails*, the signal to delete the `[!shouldfail]` tag and
// fold the check into the main suite. A standalone config isn't reliable: GJK's error is
// razor-sensitive to the exact floats, so the fuzz is what dependably surfaces it.
TEST_CASE("GjkEpa.SphereBoxEdgeAccuracy", "[GjkEpa][!shouldfail]")
{
    std::mt19937 rng{0xED9Eu};
    std::uniform_real_distribution<float> pos{-2.5f, 2.5f};
    std::uniform_real_distribution<float> half{0.5f, 2.0f};
    std::uniform_real_distribution<float> rad{0.2f, 1.0f};

    for (const float scale : {0.1f, 1.0f, 10.0f})
    {
        const float tol = 3e-3f * scale;
        for (int i = 0; i < 400; ++i)
        {
            const Vec3 sc{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const float r = rad(rng) * scale;
            const Vec3 bc{pos(rng) * scale, pos(rng) * scale, pos(rng) * scale};
            const Vec3 bh{half(rng) * scale, half(rng) * scale, half(rng) * scale};

            const Vec3 cp = closestOnAabb(sc, bc, bh);
            const float dist = (sc - cp).magnitude();
            const float gap = dist - r;
            const int clamped = (std::abs(cp.x() - sc.x()) > 1e-6f * scale ? 1 : 0) +
                                (std::abs(cp.y() - sc.y()) > 1e-6f * scale ? 1 : 0) +
                                (std::abs(cp.z() - sc.z()) > 1e-6f * scale ? 1 : 0);
            if (clamped < 2 || gap <= tol) // only edge/corner-closest separated cases
            {
                continue;
            }
            const ConvexContact c =
                gjkEpaContact(WorldShape{WorldSphere{sc, r}},
                              WorldShape{WorldBox{bc, bh, Quaternion::identity()}});
            CHECK(c.depth == Catch::Approx(gap).margin(tol));
            CHECK(c.normal.approxEqual((sc - cp) * (1.0f / dist), 5e-3f));
        }
    }
}
