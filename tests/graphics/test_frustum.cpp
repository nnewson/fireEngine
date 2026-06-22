#include <fire_engine/graphics/frustum.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fire_engine/math/constants.hpp>

using fire_engine::Bounds3;
using fire_engine::Frustum;
using fire_engine::Mat4;
using fire_engine::pi;
using fire_engine::Vec3;

namespace
{

Bounds3 box(Vec3 center, float halfSize)
{
    Bounds3 b;
    b.expand(center - Vec3{halfSize, halfSize, halfSize});
    b.expand(center + Vec3{halfSize, halfSize, halfSize});
    return b;
}

// A 90° camera at the origin looking down -z (view = identity), near 0.1, far 100. The
// frustum half-extent at depth d is d (tan 45° = 1).
Frustum cameraFrustum()
{
    return Frustum::fromViewProj(Mat4::perspective(0.5f * pi, 1.0f, 0.1f, 100.0f));
}

} // namespace

TEST_CASE("Frustum.BoxInFrontIsVisible", "[Frustum]")
{
    CHECK(cameraFrustum().intersects(box({0.0f, 0.0f, -5.0f}, 1.0f)));
}

TEST_CASE("Frustum.BoxBehindCameraIsCulled", "[Frustum]")
{
    CHECK_FALSE(cameraFrustum().intersects(box({0.0f, 0.0f, 5.0f}, 1.0f)));
}

TEST_CASE("Frustum.BoxFarToTheSideIsCulled", "[Frustum]")
{
    // At z=-5 the frustum spans x∈[-5,5]; a box at x=100 is well outside.
    CHECK_FALSE(cameraFrustum().intersects(box({100.0f, 0.0f, -5.0f}, 1.0f)));
    CHECK_FALSE(cameraFrustum().intersects(box({0.0f, 100.0f, -5.0f}, 1.0f)));
}

TEST_CASE("Frustum.BoxBeyondFarPlaneIsCulled", "[Frustum]")
{
    CHECK_FALSE(cameraFrustum().intersects(box({0.0f, 0.0f, -200.0f}, 1.0f)));
}

TEST_CASE("Frustum.BoxStraddlingAPlaneIsVisible", "[Frustum]")
{
    // Half in front of the near plane, half behind — conservatively visible.
    CHECK(cameraFrustum().intersects(box({0.0f, 0.0f, -0.2f}, 0.5f)));
    // Straddling the right edge at z=-5 (frustum x-extent ±5).
    CHECK(cameraFrustum().intersects(box({5.0f, 0.0f, -5.0f}, 1.0f)));
}

TEST_CASE("Frustum.InvalidBoundsIsAlwaysVisible", "[Frustum]")
{
    const Bounds3 unbounded; // default: valid == false (e.g. the skybox)
    CHECK(cameraFrustum().intersects(unbounded));
}
