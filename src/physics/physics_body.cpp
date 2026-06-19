#include <fire_engine/physics/physics_body.hpp>

#include <algorithm>

namespace fire_engine
{

void PhysicsBody::type(PhysicsBodyType type) noexcept
{
    type_ = type;
    if (type_ != PhysicsBodyType::Dynamic)
    {
        inverseMass_ = 0.0f;
    }
    else
    {
        mass(mass_);
    }
}

void PhysicsBody::mass(float mass) noexcept
{
    mass_ = std::max(mass, 0.0f);
    inverseMass_ = type_ == PhysicsBodyType::Dynamic && mass_ > 0.0f ? 1.0f / mass_ : 0.0f;
}

} // namespace fire_engine
