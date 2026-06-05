#pragma once

#include <cstdint>

namespace fire_engine
{

// Opaque, type-safe handle into PhysicsWorld storage. The Tag parameter keeps
// otherwise-identical handle types (body, collider, constraint) from being
// implicitly interchangeable. A value of 0 is the null/invalid handle.
template <typename Tag>
class PhysicsHandle
{
public:
    constexpr PhysicsHandle() noexcept = default;

    constexpr explicit PhysicsHandle(std::uint32_t value) noexcept
        : value_{value}
    {
    }

    ~PhysicsHandle() = default;

    PhysicsHandle(const PhysicsHandle&) = default;
    PhysicsHandle& operator=(const PhysicsHandle&) = default;
    PhysicsHandle(PhysicsHandle&&) noexcept = default;
    PhysicsHandle& operator=(PhysicsHandle&&) noexcept = default;

    [[nodiscard]]
    constexpr std::uint32_t value() const noexcept
    {
        return value_;
    }

    [[nodiscard]]
    constexpr bool valid() const noexcept
    {
        return value_ != 0U;
    }

    [[nodiscard]]
    friend constexpr bool operator==(PhysicsHandle lhs, PhysicsHandle rhs) noexcept
    {
        return lhs.value_ == rhs.value_;
    }

    [[nodiscard]]
    friend constexpr bool operator!=(PhysicsHandle lhs, PhysicsHandle rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    std::uint32_t value_{0U};
};

struct PhysicsBodyTag
{
};

struct PhysicsColliderTag
{
};

struct PhysicsConstraintTag
{
};

using PhysicsBodyHandle = PhysicsHandle<PhysicsBodyTag>;
using PhysicsColliderHandle = PhysicsHandle<PhysicsColliderTag>;
using PhysicsConstraintHandle = PhysicsHandle<PhysicsConstraintTag>;

} // namespace fire_engine
