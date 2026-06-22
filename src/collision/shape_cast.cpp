#include <fire_engine/collision/shape_cast.hpp>

#include <variant>

#include <fire_engine/collision/gjk_epa.hpp>

namespace fire_engine
{

namespace
{

constexpr int kMaxIterations = 32;
constexpr float kDistanceTolerance = 1e-4f;
constexpr float kClosingEpsilon = 1e-8f;

// `shape` translated by `offset` (the sweep displacement). The target stays fixed, so
// only the moving shape is advanced each iteration.
[[nodiscard]] WorldShape translated(const WorldShape& shape, const Vec3& offset)
{
    return std::visit(
        [&offset](const auto& s) -> WorldShape
        {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, WorldSphere>)
            {
                return WorldSphere{s.center + offset, s.radius};
            }
            else if constexpr (std::is_same_v<T, WorldBox>)
            {
                return WorldBox{s.center + offset, s.halfExtents, s.orientation};
            }
            else if constexpr (std::is_same_v<T, WorldCapsule>)
            {
                return WorldCapsule{s.p0 + offset, s.p1 + offset, s.radius};
            }
            else
            {
                WorldConvex moved;
                moved.vertices.reserve(s.vertices.size());
                for (const Vec3& v : s.vertices)
                {
                    moved.vertices.push_back(v + offset);
                }
                moved.faces = s.faces; // loops are index-based, so the span still applies
                return moved;
            }
        },
        shape);
}

} // namespace

std::optional<ToiHit> shapeCast(const WorldShape& moving, const Vec3& direction, float maxDistance,
                                const WorldShape& target) noexcept
{
    float t = 0.0f;
    for (int iter = 0; iter < kMaxIterations; ++iter)
    {
        const WorldShape swept = translated(moving, direction * t);
        const ConvexContact contact = gjkEpaContact(swept, target);

        if (contact.colliding)
        {
            // Already touching/penetrating at t (t == 0 means the sweep started overlapped).
            return ToiHit{t, contact.pointB, contact.normal};
        }

        const float gap = contact.depth; // separation distance when not colliding
        if (gap < kDistanceTolerance)
        {
            return ToiHit{t, contact.pointB, contact.normal};
        }

        // `normal` points from the target toward the moving shape; the gap shrinks at
        // rate -(direction · normal) as the moving shape advances along +direction.
        const float closing = -Vec3::dotProduct(direction, contact.normal);
        if (closing <= kClosingEpsilon)
        {
            return std::nullopt; // sweeping parallel to or away from the target
        }

        t += gap / closing; // conservative advancement never overshoots the impact
        if (t > maxDistance)
        {
            return std::nullopt;
        }
    }
    return std::nullopt; // did not converge within the iteration budget
}

} // namespace fire_engine
