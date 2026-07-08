#include <fire_engine/collision/shape_cast.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <variant>

#include <fire_engine/collision/gjk_epa.hpp>
#include <fire_engine/collision/narrow_phase.hpp>

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

[[nodiscard]] bool involvesConvex(const WorldShape& a, const WorldShape& b) noexcept
{
    return std::holds_alternative<WorldConvex>(a) || std::holds_alternative<WorldConvex>(b);
}

[[nodiscard]] ConvexContact contactForSweep(const WorldShape& moving, const WorldShape& target,
                                            float maxDistance) noexcept
{
    if (!involvesConvex(moving, target))
    {
        NarrowPhase np;
        if (const auto manifold = np.collide(moving, target, maxDistance + kDistanceTolerance))
        {
            int best = 0;
            float signedDepth = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < manifold->count; ++i)
            {
                if (manifold->points[static_cast<std::size_t>(i)].penetration > signedDepth)
                {
                    signedDepth = manifold->points[static_cast<std::size_t>(i)].penetration;
                    best = i;
                }
            }

            return ConvexContact{signedDepth >= -kDistanceTolerance, manifold->normal,
                                 std::abs(signedDepth),
                                 manifold->points[static_cast<std::size_t>(best)].position,
                                 manifold->points[static_cast<std::size_t>(best)].position};
        }
        return ConvexContact{false, {}, maxDistance + kDistanceTolerance, {}, {}};
    }

    return gjkEpaContact(moving, target);
}

} // namespace

// shapeCast's only throwing path is translated()'s std::visit over WorldShape, which throws
// bad_variant_access only on a valueless variant; the assert guarantees that never happens.
static_assert(std::is_nothrow_move_constructible_v<WorldShape>);
// NOLINTNEXTLINE(bugprone-exception-escape): WorldShape is never valueless (asserted above).
std::optional<ToiHit> shapeCast(const WorldShape& moving, const Vec3& direction, float maxDistance,
                                const WorldShape& target) noexcept
{
    float t = 0.0f;
    for (int iter = 0; iter < kMaxIterations; ++iter)
    {
        const WorldShape swept = translated(moving, direction * t);
        const ConvexContact contact = contactForSweep(swept, target, maxDistance - t);

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
