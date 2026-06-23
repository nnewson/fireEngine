#include <fire_engine/physics/character_controller.hpp>

#include <algorithm>
#include <cmath>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

namespace
{

constexpr float kMinMotion = 1e-5f;

} // namespace

std::optional<ShapecastHit> CharacterController::sweep(const PhysicsWorld& world, Vec3 from,
                                                       Vec3 direction, float distance) const
{
    const float halfHeight = std::max(0.0f, config_.height * 0.5f - config_.radius);
    Transform pose;
    pose.position(from);
    pose.update(Mat4::identity());
    return world.shapecast(CapsuleShape{config_.radius, halfHeight, Vec3{}}, pose, direction,
                           distance, config_.filter);
}

Vec3 CharacterController::slide(const PhysicsWorld& world, Vec3 start, Vec3 motion,
                                bool keepGroundClimb) const
{
    Vec3 pos = start;
    for (int i = 0; i < config_.maxIterations; ++i)
    {
        const float dist = motion.magnitude();
        if (dist < kMinMotion)
        {
            break;
        }
        const Vec3 dir = motion * (1.0f / dist);
        const auto hit = sweep(world, pos, dir, dist + config_.skinWidth);
        if (!hit.has_value())
        {
            pos += motion;
            break;
        }

        // Advance to just short of the impact (keep the skin gap), then project the
        // blocked remainder along the contact plane and continue.
        const float advance = std::max(0.0f, hit->distance - config_.skinWidth);
        pos += dir * advance;
        const Vec3 n = hit->normal;

        // Walkable ground/edge underfoot must not block *horizontal* travel: height is owned by
        // the ground snap, so the capsule walks freely across it and only walls (low n.y) stop
        // it. Without this, descending a step wedges the rounded capsule against the step's top
        // edge — it touches the edge (zero advance → the nudge below), the snap rests it back on
        // the edge, and it jitters in place for many frames before escaping. (Horizontal slides
        // pass keepGroundClimb=false; the deliberate vertical/lift slides pass true and are
        // unaffected.)
        if (!keepGroundClimb && n.y() >= config_.maxSlopeCosine)
        {
            pos += motion - dir * advance; // complete the horizontal travel over the ground
            break;
        }

        // Grazing a surface (no real advance) stalls collide-and-slide: every sweep along
        // the surface re-hits it at distance ~0. Nudge out along the contact normal so the
        // next sweep clears it (a cheap depenetration; the ground snap re-settles height).
        if (advance < kMinMotion)
        {
            pos += n * config_.skinWidth;
        }

        const Vec3 blocked = motion - dir * advance;
        Vec3 slid = blocked - n * Vec3::dotProduct(blocked, n);

        // Sliding off a wall / too-steep slope must not lift the character. Walkable
        // ground (steep enough only by a little) keeps its climb so ramps are climbable.
        if (!keepGroundClimb && n.y() < config_.maxSlopeCosine && slid.y() > 0.0f)
        {
            slid = Vec3{slid.x(), 0.0f, slid.z()};
        }
        motion = slid;
    }
    return pos;
}

CharacterMoveResult CharacterController::move(const PhysicsWorld& world, Vec3 displacement)
{
    const Vec3 horizontal{displacement.x(), 0.0f, displacement.z()};
    const float vertical = displacement.y();

    Vec3 pos = position_;

    // Vertical first (gravity / jump) — vertical motion isn't "climbing", so allow it.
    if (std::abs(vertical) > 0.0f)
    {
        pos = slide(world, pos, Vec3{0.0f, vertical, 0.0f}, true);
    }

    // Horizontal collide-and-slide.
    Vec3 walked = slide(world, pos, horizontal, false);

    // Step-up: lift by the step offset, walk, drop back down. Keep it only if it makes
    // more forward progress than the flat slide (so the character mounts low ledges).
    if (config_.stepOffset > 0.0f && horizontal.magnitudeSquared() > kMinMotion * kMinMotion)
    {
        const Vec3 lifted = slide(world, pos, Vec3{0.0f, config_.stepOffset, 0.0f}, true);
        const Vec3 steppedWalk = slide(world, lifted, horizontal, false);

        // A low step is the only obstacle the *lifted* walk clears: above a step's flat top
        // there's open space, so the raised capsule advances ~fully, whereas a wall or a slope
        // keeps blocking it. So a near-full lifted advance is the signal that this is a
        // mountable step (and not a wall or a too-steep face, which the flat slide should keep
        // blocking). Gating on the lifted advance — rather than the surface normal under the
        // rounded capsule, whose ambiguous step-top-edge value reads "too steep" on approach —
        // both avoids crawling up steep slopes and removes the frame-after-frame stutter where
        // the capsule stalled at a riser for a beat before mounting.
        const Vec3 liftedStep{steppedWalk.x() - lifted.x(), 0.0f, steppedWalk.z() - lifted.z()};
        const bool clearedStep = liftedStep.magnitude() >= 0.8f * horizontal.magnitude();

        // Drop onto the step by a downward *sweep-and-rest*, not a collide-and-slide: rest at
        // first contact (a skin above it) with no lateral projection. A slide would push the
        // rounded capsule off the step's top edge back to the floor whenever its centre hasn't
        // yet cleared the edge — which, at walk speed (~0.05/frame ≪ radius), is every frame,
        // so it could never accumulate height. Resting on the edge lets it gain a little each
        // frame and mount over a run of frames.
        Vec3 dropped = steppedWalk;
        if (const auto down = sweep(world, steppedWalk, Vec3{0.0f, -1.0f, 0.0f},
                                    config_.stepOffset + config_.skinWidth))
        {
            dropped = steppedWalk +
                      Vec3{0.0f, -1.0f, 0.0f} * std::max(0.0f, down->distance - config_.skinWidth);
        }

        const auto progress = [&](Vec3 p)
        {
            const Vec3 delta{p.x() - pos.x(), 0.0f, p.z() - pos.z()};
            return Vec3::dotProduct(delta, horizontal);
        };
        // Accept the step only if the lifted walk cleared the obstacle AND it made more forward
        // progress than the flat slide (so it mounts low steps but not walls or steep slopes).
        if (clearedStep && progress(dropped) > progress(walked) + 1e-4f)
        {
            walked = dropped;
        }
    }
    pos = walked;

    CharacterMoveResult result;
    result.position = pos;

    // Ground probe + snap (skipped while moving up so a jump isn't cancelled). A straight
    // downward raycast from the capsule centre gives a clean analytic surface normal
    // (the capsule-sweep normal is unreliable on box faces). The feet sit `height/2` below
    // the centre; the character is grounded when the ground is at most a step offset below
    // the feet, and is then snapped to rest a skin-width above it (covers stepping down off
    // a low ledge while staying grounded).
    if (vertical <= 0.0f)
    {
        // Distance from a capsule sweep (the true capsule-to-ground gap, correct on slopes
        // where a centre raycast would read past the rounded contact) + a centre raycast
        // for the analytic surface normal (a capsule-sweep normal is unreliable on faces).
        const float reach = config_.stepOffset + 2.0f * config_.skinWidth;
        const auto sweepHit = sweep(world, pos, Vec3{0.0f, -1.0f, 0.0f}, reach);
        const Ray ray{pos, Vec3{0.0f, -1.0f, 0.0f},
                      config_.height * 0.5f + config_.stepOffset + config_.skinWidth + 0.1f};
        const auto normalHit = world.raycast(ray, config_.filter);
        if (sweepHit.has_value() && normalHit.has_value() &&
            normalHit->normal.y() >= config_.maxSlopeCosine)
        {
            result.grounded = true;
            result.groundNormal = normalHit->normal;
            const float drop = sweepHit->distance - config_.skinWidth; // rest a skin above ground
            result.position = pos + Vec3{0.0f, -drop, 0.0f};
        }
    }

    position_ = result.position;
    return result;
}

} // namespace fire_engine
