#pragma once

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

enum class PhysicsBodyType
{
    Static,
    Kinematic,
    Dynamic,
};

struct PhysicsMaterial
{
    float restitution{1.0f};
    float friction{0.0f};
};

class PhysicsBody
{
public:
    PhysicsBody() = default;
    ~PhysicsBody() = default;

    PhysicsBody(const PhysicsBody&) = default;
    PhysicsBody& operator=(const PhysicsBody&) = default;
    PhysicsBody(PhysicsBody&&) noexcept = default;
    PhysicsBody& operator=(PhysicsBody&&) noexcept = default;

    [[nodiscard]]
    PhysicsBodyType type() const noexcept
    {
        return type_;
    }
    void type(PhysicsBodyType type) noexcept;

    [[nodiscard]]
    Vec3 linearVelocity() const noexcept
    {
        return linearVelocity_;
    }
    void linearVelocity(Vec3 velocity) noexcept
    {
        linearVelocity_ = velocity;
    }

    [[nodiscard]]
    Vec3 angularVelocity() const noexcept
    {
        return angularVelocity_;
    }
    void angularVelocity(Vec3 velocity) noexcept
    {
        angularVelocity_ = velocity;
    }

    [[nodiscard]]
    float mass() const noexcept
    {
        return mass_;
    }
    void mass(float mass) noexcept;

    [[nodiscard]]
    float inverseMass() const noexcept
    {
        return inverseMass_;
    }

    [[nodiscard]]
    float gravityScale() const noexcept
    {
        return gravityScale_;
    }
    void gravityScale(float gravityScale) noexcept
    {
        gravityScale_ = gravityScale;
    }

    [[nodiscard]]
    PhysicsMaterial material() const noexcept
    {
        return material_;
    }
    void material(PhysicsMaterial material) noexcept
    {
        material_ = material;
    }

private:
    PhysicsBodyType type_{PhysicsBodyType::Static};
    Vec3 linearVelocity_{};
    Vec3 angularVelocity_{};
    float mass_{1.0f};
    float inverseMass_{0.0f};
    float gravityScale_{1.0f};
    PhysicsMaterial material_{};
};

struct PhysicsBodyDesc
{
    PhysicsBodyType type{PhysicsBodyType::Static};
    Vec3 position{};
    Quaternion rotation{Quaternion::identity()};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    Vec3 linearVelocity{};
    Vec3 angularVelocity{};
    float mass{1.0f};
    float gravityScale{1.0f};
    PhysicsMaterial material{};
};

} // namespace fire_engine
