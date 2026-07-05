#pragma once

#include <cstdint>

#include <fire_engine/graphics/gpu_handle.hpp>

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

    // Handles pack a slot index (low 24 bits) + a generation (high 8 bits), mirroring the GPU
    // handle scheme (graphics/gpu_handle.hpp). The index directly addresses PhysicsWorld's slot
    // storage; the generation is validated against the slot's live generation so a stale handle
    // to a recycled slot is detectably invalid. Live handles always carry a nonzero (1-based)
    // generation, so a live handle's value is never 0 — it stays distinct from the null handle.
    [[nodiscard]]
    constexpr std::uint32_t index() const noexcept
    {
        return value_ & kHandleIndexMask;
    }

    [[nodiscard]]
    constexpr std::uint32_t generation() const noexcept
    {
        return (value_ >> kHandleIndexBits) & kHandleGenerationMask;
    }

    [[nodiscard]]
    static constexpr PhysicsHandle make(std::uint32_t index, std::uint32_t generation) noexcept
    {
        return PhysicsHandle{((generation & kHandleGenerationMask) << kHandleIndexBits) |
                             (index & kHandleIndexMask)};
    }

    [[nodiscard]]
    friend constexpr bool operator==(const PhysicsHandle&, const PhysicsHandle&) noexcept = default;

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

struct PhysicsArticulationTag
{
};

using PhysicsBodyHandle = PhysicsHandle<PhysicsBodyTag>;
using PhysicsColliderHandle = PhysicsHandle<PhysicsColliderTag>;
using PhysicsConstraintHandle = PhysicsHandle<PhysicsConstraintTag>;
using PhysicsArticulationHandle = PhysicsHandle<PhysicsArticulationTag>;

} // namespace fire_engine
