#pragma once

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/solver_math.hpp>

namespace fire_engine
{

// The contact-solver seam (P9 item 5, Phase D): the one thing a constraint row needs from
// whatever it acts on, whether that is a rigid body or an articulation link. A row asks a
// point's world velocity, how strongly the point resists an impulse along a direction (the
// inverse effective mass), and then applies an impulse — the body routes each to the right
// place (a flat SolverBody, or the articulated impulse response through a whole linkage).
//
// This is the abstraction the reduced-coordinate work rides on: "add an articulated
// constraint-body type to the contact solver," not "add ABA". A lightweight value type so
// the solver can hold one per contact side without heap churn.
class ConstraintBody
{
public:
    // A static / immovable side (a fixed floor): infinite mass, zero velocity, no response.
    ConstraintBody() = default;

    // A rigid body: mutates the SolverBody's linear/angular velocity directly, exactly as the
    // flat contact solver does — behaviour-preserving for rigid-vs-rigid.
    static ConstraintBody rigid(SolverBody& body) noexcept
    {
        ConstraintBody b;
        b.kind_ = Kind::Rigid;
        b.body_ = &body;
        return b;
    }

    // An articulation link: routes through the articulated impulse response. The articulation
    // must have been factorized (factorizeArticulatedInertia + computeLinkVelocities) this step.
    static ConstraintBody link(Articulation& articulation, std::size_t linkIndex) noexcept
    {
        ConstraintBody b;
        b.kind_ = Kind::Link;
        b.articulation_ = &articulation;
        b.link_ = linkIndex;
        return b;
    }

    [[nodiscard]]
    bool movable() const noexcept
    {
        return kind_ != Kind::Static;
    }

    // World velocity of the material point currently at `worldPoint`.
    [[nodiscard]]
    Vec3 velocityAt(const Vec3& worldPoint) const
    {
        switch (kind_)
        {
        case Kind::Rigid:
            return body_->velocity +
                   Vec3::crossProduct(body_->angularVelocity, worldPoint - body_->position);
        case Kind::Link:
            return articulation_->pointVelocity(link_, worldPoint);
        case Kind::Static:
            break;
        }
        return Vec3{};
    }

    // Inverse effective mass along `dir` (unit) at `worldPoint`: dᵀ M⁻¹ d, the point's
    // response to a unit impulse. 0 for a static side.
    [[nodiscard]]
    float inverseEffectiveMassAlong(const Vec3& worldPoint, const Vec3& dir,
                                    const Mat3& invInertia) const
    {
        switch (kind_)
        {
        case Kind::Rigid:
        {
            const Vec3 r = worldPoint - body_->position;
            const Vec3 rxd = Vec3::crossProduct(r, dir);
            return body_->invMass + Vec3::dotProduct(rxd, invInertia * rxd);
        }
        case Kind::Link:
            return articulation_->inverseEffectiveMass(link_, worldPoint, dir);
        case Kind::Static:
            break;
        }
        return 0.0f;
    }

    // Apply a world impulse at `worldPoint`.
    void applyImpulse(const Vec3& worldPoint, const Vec3& impulse, const Mat3& invInertia)
    {
        switch (kind_)
        {
        case Kind::Rigid:
            body_->velocity += impulse * body_->invMass;
            body_->angularVelocity +=
                invInertia * Vec3::crossProduct(worldPoint - body_->position, impulse);
            break;
        case Kind::Link:
            articulation_->applyImpulse(link_, worldPoint, impulse);
            break;
        case Kind::Static:
            break;
        }
    }

private:
    enum class Kind
    {
        Static,
        Rigid,
        Link,
    };

    Kind kind_{Kind::Static};
    SolverBody* body_{nullptr};
    Articulation* articulation_{nullptr};
    std::size_t link_{0};
};

} // namespace fire_engine
