#include <fire_engine/collision/collider.hpp>

#include <algorithm>
#include <array>
#include <cassert>

#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]]
AABB mergeBounds(const AABB& lhs, const AABB& rhs) noexcept
{
    return AABB{
        Vec3{std::min(lhs.min.x(), rhs.min.x()), std::min(lhs.min.y(), rhs.min.y()),
             std::min(lhs.min.z(), rhs.min.z())},
        Vec3{std::max(lhs.max.x(), rhs.max.x()), std::max(lhs.max.y(), rhs.max.y()),
             std::max(lhs.max.z(), rhs.max.z())},
    };
}

[[nodiscard]]
AABB transformBounds(const AABB& localBounds, Mat4 world)
{
    const Vec3 localMin = localBounds.min;
    const Vec3 localMax = localBounds.max;
    const std::array<Vec3, 8> corners{
        Vec3{localMin.x(), localMin.y(), localMin.z()},
        Vec3{localMax.x(), localMin.y(), localMin.z()},
        Vec3{localMin.x(), localMax.y(), localMin.z()},
        Vec3{localMax.x(), localMax.y(), localMin.z()},
        Vec3{localMin.x(), localMin.y(), localMax.z()},
        Vec3{localMax.x(), localMin.y(), localMax.z()},
        Vec3{localMin.x(), localMax.y(), localMax.z()},
        Vec3{localMax.x(), localMax.y(), localMax.z()},
    };

    Vec3 transformed = Vec3{world * Vec4{corners[0]}};
    Vec3 min = transformed;
    Vec3 max = transformed;

    for (std::size_t i = 1; i < corners.size(); ++i)
    {
        transformed = Vec3{world * Vec4{corners[i]}};
        min.x(std::min(min.x(), transformed.x()));
        min.y(std::min(min.y(), transformed.y()));
        min.z(std::min(min.z(), transformed.z()));
        max.x(std::max(max.x(), transformed.x()));
        max.y(std::max(max.y(), transformed.y()));
        max.z(std::max(max.z(), transformed.z()));
    }

    return AABB{min, max};
}

} // namespace

Collider::Collider() noexcept
{
    initialiseEndPoints();
    updateEndPointValues();
}

Collider::Collider(Collider&& rhs) noexcept
    : localBounds_{rhs.localBounds_},
      previousWorldBounds_{rhs.previousWorldBounds_},
      worldBounds_{rhs.worldBounds_},
      sweptWorldBounds_{rhs.sweptWorldBounds_},
      hasWorldBounds_{rhs.hasWorldBounds_},
      collisionLayer_{rhs.collisionLayer_},
      collisionMask_{rhs.collisionMask_}
{
    assert(!rhs.colliderId_.valid() &&
           "Cannot move a Collider that is currently registered with a "
           "SweepAndPruneBroadPhase instance — its endpoints are tracked by address. "
           "Call SweepAndPruneBroadPhase::removeCollider(rhs) first.");
    initialiseEndPoints();
    updateEndPointValues();
    rhs.colliderId_ = ColliderId{};
}

Collider& Collider::operator=(Collider&& rhs) noexcept
{
    if (this == &rhs)
    {
        return *this;
    }

    assert(!rhs.colliderId_.valid() &&
           "Cannot move-assign from a registered Collider — see move-ctor.");
    assert(!colliderId_.valid() &&
           "Cannot move-assign into a registered Collider — see move-ctor.");

    localBounds_ = rhs.localBounds_;
    previousWorldBounds_ = rhs.previousWorldBounds_;
    worldBounds_ = rhs.worldBounds_;
    sweptWorldBounds_ = rhs.sweptWorldBounds_;
    hasWorldBounds_ = rhs.hasWorldBounds_;
    collisionLayer_ = rhs.collisionLayer_;
    collisionMask_ = rhs.collisionMask_;
    initialiseEndPoints();
    updateEndPointValues();
    rhs.colliderId_ = ColliderId{};
    return *this;
}

void Collider::update(Mat4 world)
{
    const AABB nextBounds = transformBounds(localBounds_, world);
    if (!hasWorldBounds_)
    {
        previousWorldBounds_ = nextBounds;
    }
    else
    {
        previousWorldBounds_ = worldBounds_;
    }

    worldBounds_ = nextBounds;
    sweptWorldBounds_ = mergeBounds(previousWorldBounds_, worldBounds_);
    hasWorldBounds_ = true;
    updateEndPointValues();
}

void Collider::resetFrame(Mat4 world)
{
    const AABB bounds = transformBounds(localBounds_, world);
    previousWorldBounds_ = bounds;
    worldBounds_ = bounds;
    sweptWorldBounds_ = bounds;
    hasWorldBounds_ = true;
    updateEndPointValues();
}

EndPoint& Collider::minEndPoint(CollisionAxis axis) noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xMin_;
    case CollisionAxis::Y:
        return yMin_;
    case CollisionAxis::Z:
        return zMin_;
    }

    return xMin_;
}

const EndPoint& Collider::minEndPoint(CollisionAxis axis) const noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xMin_;
    case CollisionAxis::Y:
        return yMin_;
    case CollisionAxis::Z:
        return zMin_;
    }

    return xMin_;
}

EndPoint& Collider::maxEndPoint(CollisionAxis axis) noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xMax_;
    case CollisionAxis::Y:
        return yMax_;
    case CollisionAxis::Z:
        return zMax_;
    }

    return xMax_;
}

const EndPoint& Collider::maxEndPoint(CollisionAxis axis) const noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xMax_;
    case CollisionAxis::Y:
        return yMax_;
    case CollisionAxis::Z:
        return zMax_;
    }

    return xMax_;
}

std::array<EndPoint*, 6> Collider::endPoints() noexcept
{
    return {&xMin_, &xMax_, &yMin_, &yMax_, &zMin_, &zMax_};
}

void Collider::initialiseEndPoints() noexcept
{
    xMin_.collider(this);
    xMin_.colliderId(colliderId_);
    xMin_.axis(CollisionAxis::X);
    xMin_.isMin(true);
    xMin_.index(EndPoint::invalidIndex);

    xMax_.collider(this);
    xMax_.colliderId(colliderId_);
    xMax_.axis(CollisionAxis::X);
    xMax_.isMin(false);
    xMax_.index(EndPoint::invalidIndex);

    yMin_.collider(this);
    yMin_.colliderId(colliderId_);
    yMin_.axis(CollisionAxis::Y);
    yMin_.isMin(true);
    yMin_.index(EndPoint::invalidIndex);

    yMax_.collider(this);
    yMax_.colliderId(colliderId_);
    yMax_.axis(CollisionAxis::Y);
    yMax_.isMin(false);
    yMax_.index(EndPoint::invalidIndex);

    zMin_.collider(this);
    zMin_.colliderId(colliderId_);
    zMin_.axis(CollisionAxis::Z);
    zMin_.isMin(true);
    zMin_.index(EndPoint::invalidIndex);

    zMax_.collider(this);
    zMax_.colliderId(colliderId_);
    zMax_.axis(CollisionAxis::Z);
    zMax_.isMin(false);
    zMax_.index(EndPoint::invalidIndex);
}

void Collider::updateEndPointValues() noexcept
{
    for (EndPoint* endPoint : endPoints())
    {
        if (endPoint->isMin())
        {
            endPoint->value(sweptWorldBounds_.axisMin(endPoint->axis()));
        }
        else
        {
            endPoint->value(sweptWorldBounds_.axisMax(endPoint->axis()));
        }
    }
}

void Collider::colliderId(ColliderId colliderId) noexcept
{
    colliderId_ = colliderId;
    for (EndPoint* endPoint : endPoints())
    {
        endPoint->colliderId(colliderId_);
    }
}

} // namespace fire_engine
