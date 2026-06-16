#include <fire_engine/input/controller_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/controllable.hpp>
#include <fire_engine/scene/transform.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Controllable;
using fire_engine::ControllerState;
using fire_engine::Mat4;
using fire_engine::Transform;
using fire_engine::Vec3;

TEST_CASE("ControllableUpdate.AppliesControllerDeltaToTransformPosition", "[ControllableUpdate]")
{
    Controllable controllable;
    Transform transform;
    ControllerState state;
    state.deltaPosition({0.25f, 0.0f, 0.0f});

    controllable.update(state, transform, Mat4::identity());

    CHECK(transform.position().x() == Catch::Approx(2.5f).margin(1e-5f));
    CHECK(transform.position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(transform.position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((transform.world()[0, 3]) == Catch::Approx(2.5f).margin(1e-5f));
}

TEST_CASE("ControllableUpdate.UsesParentWorldWhenUpdatingTransform", "[ControllableUpdate]")
{
    Controllable controllable;
    Transform transform;
    ControllerState state;
    state.deltaPosition({0.5f, 0.0f, 0.0f});

    controllable.update(state, transform, Mat4::translate(Vec3{1.0f, 2.0f, 3.0f}));

    CHECK(transform.position().x() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((transform.world()[0, 3]) == Catch::Approx(6.0f).margin(1e-5f));
    CHECK((transform.world()[1, 3]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((transform.world()[2, 3]) == Catch::Approx(3.0f).margin(1e-5f));
}
