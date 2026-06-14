#include <gtest/gtest.h>

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

TEST(ParticleEmitter, ComponentNameIsParticleEmitter)
{
    Components c = ParticleEmitter{};
    EXPECT_EQ(componentName(c), "ParticleEmitter");
}

TEST(ParticleEmitter, SettersRoundTrip)
{
    ParticleEmitter e;
    e.spawnRate(1234.0f);
    e.lifetime(3.5f);
    e.size(0.25f);
    e.gravity(-9.8f);
    e.intensity(7.0f);
    e.coneAngle(0.5f);
    e.colour(Colour3{0.2f, 0.4f, 0.6f});
    EXPECT_FLOAT_EQ(e.spawnRate(), 1234.0f);
    EXPECT_FLOAT_EQ(e.lifetime(), 3.5f);
    EXPECT_FLOAT_EQ(e.size(), 0.25f);
    EXPECT_FLOAT_EQ(e.gravity(), -9.8f);
    EXPECT_FLOAT_EQ(e.intensity(), 7.0f);
    EXPECT_FLOAT_EQ(e.coneAngle(), 0.5f);
    EXPECT_EQ(e.colour(), Colour3(0.2f, 0.4f, 0.6f));
}

TEST(ParticleEmitter, ToEmitterStateResolvesWorldPosition)
{
    ParticleEmitter e;
    auto world = Mat4::translate(Vec3{3.0f, 4.0f, 5.0f});
    EmitterState s = ParticleEmitter::toEmitterState(e, world);
    EXPECT_FLOAT_EQ(s.worldPosition.x(), 3.0f);
    EXPECT_FLOAT_EQ(s.worldPosition.y(), 4.0f);
    EXPECT_FLOAT_EQ(s.worldPosition.z(), 5.0f);
    // Pure translation leaves the velocity direction unrotated.
    EXPECT_FLOAT_EQ(s.baseVelocity.y(), e.baseVelocity().y());
}

TEST(ParticleEmitter, GatherEmittersFindsWorldPosition)
{
    SceneGraph sg;
    auto node = std::make_unique<Node>("Fountain");
    node->transform().position(Vec3{1.0f, 2.0f, -3.0f});
    node->component().emplace<ParticleEmitter>();
    sg.addNode(std::move(node));

    sg.resolve(); // populate composedWorld_ without needing an InputState

    std::vector<EmitterState> emitters = sg.gatherEmitters();
    ASSERT_EQ(emitters.size(), 1u);
    EXPECT_FLOAT_EQ(emitters[0].worldPosition.x(), 1.0f);
    EXPECT_FLOAT_EQ(emitters[0].worldPosition.y(), 2.0f);
    EXPECT_FLOAT_EQ(emitters[0].worldPosition.z(), -3.0f);
}

TEST(ParticleEmitter, GatherEmittersIgnoresNonEmitters)
{
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("Empty"));
    sg.resolve();
    EXPECT_TRUE(sg.gatherEmitters().empty());
}
