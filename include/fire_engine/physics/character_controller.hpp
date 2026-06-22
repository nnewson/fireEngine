#pragma once

#include <optional>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_query.hpp>

namespace fire_engine
{

class PhysicsWorld;

// Tunables for a kinematic capsule character. `height` is the full capsule height
// (caps included); the internal segment half-length is height/2 − radius. `maxSlopeCosine`
// is the cosine of the steepest walkable incline measured from straight up (a ground
// normal counts as walkable when normal.y >= maxSlopeCosine); the default ~0.64 is ~50°.
struct CharacterControllerConfig
{
    float radius{0.35f};
    float height{1.8f};
    float skinWidth{0.02f}; // gap kept between the capsule and geometry
    float maxSlopeCosine{0.64f};
    float stepOffset{0.35f}; // max ledge height the character steps up
    int maxIterations{4};    // collide-and-slide passes per move
    QueryFilter filter{};
};

// Result of a move: the resolved capsule-centre position, whether the character ended the
// move on walkable ground, and that ground's normal (up when airborne).
struct CharacterMoveResult
{
    Vec3 position{};
    bool grounded{false};
    Vec3 groundNormal{0.0f, 1.0f, 0.0f};
};

// Kinematic capsule character controller built entirely on PhysicsWorld shape queries
// (no rigid-body simulation). `move` resolves a desired displacement against the world by
// collide-and-slide: the vertical component (gravity/jump) and the horizontal component
// are swept separately, blocked motion is projected along contact planes, low ledges are
// stepped over, and a downward probe snaps to and reports walkable ground. Headless —
// the scene layer drives it and writes the result back onto a kinematic node.
class CharacterController
{
public:
    CharacterController() = default;
    CharacterController(const CharacterControllerConfig& config, Vec3 position) noexcept
        : config_{config},
          position_{position}
    {
    }

    [[nodiscard]] CharacterMoveResult move(const PhysicsWorld& world, Vec3 displacement);

    [[nodiscard]] Vec3 position() const noexcept
    {
        return position_;
    }
    void position(Vec3 position) noexcept
    {
        position_ = position;
    }

    [[nodiscard]] const CharacterControllerConfig& config() const noexcept
    {
        return config_;
    }
    void config(const CharacterControllerConfig& config) noexcept
    {
        config_ = config;
    }

private:
    // Sweep the capsule from `from` along unit `direction` up to `distance`; returns the
    // time-of-impact hit (collider + normal) or nullopt for a clear sweep.
    [[nodiscard]] std::optional<ShapecastHit> sweep(const PhysicsWorld& world, Vec3 from,
                                                    Vec3 direction, float distance) const;
    // Collide-and-slide `motion` from `start`, returning the resolved position. When
    // `keepGroundClimb` is false, upward slide off too-steep faces is removed (so the
    // character can't climb walls / steep slopes).
    [[nodiscard]] Vec3 slide(const PhysicsWorld& world, Vec3 start, Vec3 motion,
                             bool keepGroundClimb) const;

    CharacterControllerConfig config_{};
    Vec3 position_{};
};

} // namespace fire_engine
