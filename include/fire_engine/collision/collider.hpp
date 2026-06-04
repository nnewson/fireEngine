#pragma once

#include <array>
#include <cstdint>

#include <fire_engine/collision/aabb.hpp>
#include <fire_engine/collision/end_point.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

class SweepAndPruneBroadPhase;

class Collider
{
public:
    Collider() noexcept;
    ~Collider() = default;

    Collider(const Collider&) = delete;
    Collider& operator=(const Collider&) = delete;

    // Move ops are safe ONLY when neither the source nor the destination is
    // currently registered with a SweepAndPruneBroadPhase instance. The endpoints
    // carry back-pointers to `this`, and the broadphase axis vectors hold pointers
    // to the endpoints inside the Collider — moving a registered collider would
    // invalidate those. Debug builds assert; release builds reset the source's id
    // so the source is at least cleanly unregistered after the move.
    Collider(Collider&& rhs) noexcept;
    Collider& operator=(Collider&& rhs) noexcept;

    [[nodiscard]] AABB localBounds() const noexcept
    {
        return localBounds_;
    }
    void localBounds(AABB localBounds) noexcept
    {
        localBounds_ = localBounds;
    }

    [[nodiscard]] AABB worldBounds() const noexcept
    {
        return worldBounds_;
    }

    [[nodiscard]] AABB previousWorldBounds() const noexcept
    {
        return previousWorldBounds_;
    }

    [[nodiscard]] AABB sweptWorldBounds() const noexcept
    {
        return sweptWorldBounds_;
    }

    [[nodiscard]]
    ColliderId colliderId() const noexcept
    {
        return colliderId_;
    }

    [[nodiscard]]
    std::uint32_t collisionLayer() const noexcept
    {
        return collisionLayer_;
    }

    void collisionLayer(std::uint32_t collisionLayer) noexcept
    {
        collisionLayer_ = collisionLayer;
    }

    [[nodiscard]]
    std::uint32_t collisionMask() const noexcept
    {
        return collisionMask_;
    }

    void collisionMask(std::uint32_t collisionMask) noexcept
    {
        collisionMask_ = collisionMask;
    }

    void update(Mat4 world);
    void resetFrame(Mat4 world);

    [[nodiscard]]
    EndPoint& minEndPoint(CollisionAxis axis) noexcept;

    [[nodiscard]]
    const EndPoint& minEndPoint(CollisionAxis axis) const noexcept;

    [[nodiscard]]
    EndPoint& maxEndPoint(CollisionAxis axis) noexcept;

    [[nodiscard]]
    const EndPoint& maxEndPoint(CollisionAxis axis) const noexcept;

    [[nodiscard]]
    std::array<EndPoint*, 6> endPoints() noexcept;

private:
    friend class SweepAndPruneBroadPhase;

    static constexpr std::uint32_t defaultCollisionLayer = 1U;
    static constexpr std::uint32_t defaultCollisionMask = ~0U;

    AABB localBounds_;
    AABB previousWorldBounds_;
    AABB worldBounds_;
    AABB sweptWorldBounds_;
    bool hasWorldBounds_{false};
    ColliderId colliderId_;
    std::uint32_t collisionLayer_{defaultCollisionLayer};
    std::uint32_t collisionMask_{defaultCollisionMask};
    EndPoint xMin_;
    EndPoint xMax_;
    EndPoint yMin_;
    EndPoint yMax_;
    EndPoint zMin_;
    EndPoint zMax_;

    void initialiseEndPoints() noexcept;
    void updateEndPointValues() noexcept;
    void colliderId(ColliderId colliderId) noexcept;
};

} // namespace fire_engine
