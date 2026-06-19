#include <cmath>

#include <fire_engine/collision/geometry.hpp>
#include <fire_engine/collision/world_shape.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::closestPointOnObb;
using fire_engine::closestPointOnSegment;
using fire_engine::closestPointsBetweenSegments;
using fire_engine::Quaternion;
using fire_engine::SegmentClosest;
using fire_engine::Vec3;
using fire_engine::WorldBox;

TEST_CASE("Geometry.ClosestPointOnSegment", "[Geometry]")
{
    const Vec3 a{0.0f, 0.0f, 0.0f};
    const Vec3 b{2.0f, 0.0f, 0.0f};

    // Projects onto the interior.
    CHECK(closestPointOnSegment({1.0f, 1.0f, 0.0f}, a, b).approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f));
    // Clamps past the ends.
    CHECK(closestPointOnSegment({-3.0f, 5.0f, 0.0f}, a, b).approxEqual(a, 1e-5f));
    CHECK(closestPointOnSegment({9.0f, -5.0f, 0.0f}, a, b).approxEqual(b, 1e-5f));
    // Degenerate segment returns the point.
    CHECK(closestPointOnSegment({1.0f, 0.0f, 0.0f}, a, a).approxEqual(a, 1e-5f));
}

TEST_CASE("Geometry.ClosestPointOnObb", "[Geometry]")
{
    WorldBox box;
    box.center = {0.0f, 0.0f, 0.0f};
    box.halfExtents = {1.0f, 1.0f, 1.0f};

    // Outside on +x → clamps to the face.
    CHECK(closestPointOnObb({3.0f, 0.0f, 0.0f}, box).approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f));
    // Inside → returns itself.
    CHECK(closestPointOnObb({0.4f, 0.2f, -0.1f}, box)
              .approxEqual(Vec3{0.4f, 0.2f, -0.1f}, 1e-5f));

    // Oriented box: 45° about Z. A point along world +x maps to a face.
    box.orientation = Quaternion{0.0f, 0.0f, std::sin(0.39269908f), std::cos(0.39269908f)}; // 45°
    const Vec3 cp = closestPointOnObb({5.0f, 0.0f, 0.0f}, box);
    // Closest point must lie on the box surface (distance from center ~ projected).
    CHECK(cp.magnitude() == Catch::Approx(std::sqrt(2.0f)).margin(1e-4f));
}

TEST_CASE("Geometry.ClosestPointsBetweenSegments", "[Geometry]")
{
    // Perpendicular crossing segments meet at (1,0,0).
    const SegmentClosest cross = closestPointsBetweenSegments(
        {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f});
    CHECK(cross.c1.approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(cross.c2.approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f));

    // Parallel segments offset in y: closest distance is the offset.
    const SegmentClosest par = closestPointsBetweenSegments(
        {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f});
    CHECK((par.c1 - par.c2).magnitude() == Catch::Approx(1.0f).margin(1e-5f));
}
