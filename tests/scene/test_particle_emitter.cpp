#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/particle_emitter.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using fire_engine::Colour3;
using fire_engine::componentName;
using fire_engine::Components;
using fire_engine::EmitterState;
using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::ParticleEmitter;
using fire_engine::SceneGraph;
using fire_engine::Vec3;

TEST_CASE("ParticleEmitter.ComponentNameIsParticleEmitter", "[ParticleEmitter]")
{
    Components c = ParticleEmitter{};
    CHECK(componentName(c) == "ParticleEmitter");
}

TEST_CASE("ParticleEmitter.SettersRoundTrip", "[ParticleEmitter]")
{
    ParticleEmitter e;
    e.spawnRate(1234.0f);
    e.lifetime(3.5f);
    e.size(0.25f);
    e.gravity(-9.8f);
    e.intensity(7.0f);
    e.coneAngle(0.5f);
    e.colour(Colour3{0.2f, 0.4f, 0.6f});
    CHECK(e.spawnRate() == Catch::Approx(1234.0f).margin(1e-5f));
    CHECK(e.lifetime() == Catch::Approx(3.5f).margin(1e-5f));
    CHECK(e.size() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(e.gravity() == Catch::Approx(-9.8f).margin(1e-5f));
    CHECK(e.intensity() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(e.coneAngle() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(e.colour() == Colour3(0.2f, 0.4f, 0.6f));
}

TEST_CASE("ParticleEmitter.ToEmitterStateResolvesWorldPosition", "[ParticleEmitter]")
{
    ParticleEmitter e;
    auto world = Mat4::translate(Vec3{3.0f, 4.0f, 5.0f});
    EmitterState s = ParticleEmitter::toEmitterState(e, world);
    CHECK(s.worldPosition.x() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(s.worldPosition.y() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(s.worldPosition.z() == Catch::Approx(5.0f).margin(1e-5f));
    // Pure translation leaves the velocity direction unrotated.
    CHECK(s.baseVelocity.y() == Catch::Approx(e.baseVelocity().y()).margin(1e-5f));
}

TEST_CASE("ParticleEmitter.GatherEmittersFindsWorldPosition", "[ParticleEmitter]")
{
    SceneGraph sg;
    auto node = std::make_unique<Node>("Fountain");
    node->transform().position(Vec3{1.0f, 2.0f, -3.0f});
    node->component().emplace<ParticleEmitter>();
    sg.addNode(std::move(node));

    sg.resolve(); // populate composedWorld_ without needing an InputState

    std::vector<EmitterState> emitters = sg.gatherEmitters();
    REQUIRE(emitters.size() == 1u);
    CHECK(emitters[0].worldPosition.x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(emitters[0].worldPosition.y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(emitters[0].worldPosition.z() == Catch::Approx(-3.0f).margin(1e-5f));
}

TEST_CASE("ParticleEmitter.GatherEmittersOutputVectorIsClearedAndReused", "[ParticleEmitter]")
{
    SceneGraph sg;
    auto node = std::make_unique<Node>("Fountain");
    node->component().emplace<ParticleEmitter>().spawnRate(64.0f);
    sg.addNode(std::move(node));

    sg.resolve();

    std::vector<EmitterState> emitters(2);
    const auto previousCapacity = emitters.capacity();
    sg.gatherEmitters(emitters);

    REQUIRE(emitters.size() == 1u);
    CHECK(emitters.capacity() >= previousCapacity);
    CHECK(emitters[0].spawnRate == Catch::Approx(64.0f).margin(1e-5f));
}

TEST_CASE("ParticleEmitter.GatherEmittersIgnoresNonEmitters", "[ParticleEmitter]")
{
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("Empty"));
    sg.resolve();
    CHECK(sg.gatherEmitters().empty());
}
