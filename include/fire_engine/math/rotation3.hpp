#pragma once

#include <cmath>
#include <optional>

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/scalar.hpp>
#include <fire_engine/math/vec3.hpp>

// A 3D ROTATION — a value that is always a rotation, as opposed to four numbers that usually are
// (tier-0 review, finding 3).
//
// `Quaternion` is freely constructible and mutable while most of its API assumes unit length:
// `rotate`, `slerp`, `toMat4`, `toEulerXYZ` and `Mat3::fromQuaternion` all produce nonsense from a
// non-unit value, quietly. `fromAxisAngle({0, 0, 0}, angle)` is the clearest case — there is no
// rotation being described, and what comes back is not one either.
//
// WHY THIS EXISTS, given that nothing violates the invariant today. It was measured before it was
// designed: about 24 million observations across the whole test suite and three scenes (including a
// 900-frame ragdoll run) found ZERO values off unit by more than 1e-4, with the worst deviation
// anywhere at 7.15e-07 — roughly six ulps, and not growing. So this type is not fixing a live bug.
// It removes a representable invalid state, which is a different and longer-lived kind of value:
// the reason nothing violates the invariant today is that a handful of scattered `normalise()`
// calls happen to be in the right places, and nothing makes that true tomorrow.
//
// THE INVARIANT IS MAINTAINED BY EVERY OPERATION, not by a periodic repair, and it is maintained
// DEFINITIVELY. That distinction is what the measurement actually supports: `Quaternion::integrate`
// normalises its result and `slerp` normalises its near-linear path, so the bounded drift observed
// is evidence that THOSE repairs are sufficient — not that a rotation type could skip them. Every
// operation here that does floating-point algebra therefore ends at one private choke point, and
// that choke point cannot complete with a non-unit or non-finite value: it terminates instead.
// Normalising a NaN would otherwise store NaNs in a type whose whole claim is that it holds a
// rotation.
//
// NAMED FOR THE VALUE, not the representation: a caller wants a rotation, and that it is stored as
// a unit quaternion is this type's business. The quaternion is reachable (`quaternion()`) for the
// places that genuinely need four-component algebra — swing-twist decomposition in the joint
// solver, glTF CUBICSPLINE tangents, which are derivatives rather than orientations — and those
// stay on `Quaternion` deliberately.

namespace fire_engine
{

// TWO TOLERANCES, because they answer two different questions and one number cannot.
//
// ADMISSION: how far a raw four-component value may sit from unit length and still be understood as
// a rotation that merely needs normalising. Sized for AUTHORED DATA, not for the engine's own
// 7e-07 internal drift, and specifically for the COARSEST encoding glTF permits for rotation
// animation outputs — normalised signed bytes, where each component decodes as round(c·127)/127.
//
// The bound is derived, not guessed. A component's quantisation error is at most 0.5/127, so
//     |‖q‖² − 1| ≤ 2·Σ|cᵢ|·(0.5/127) + 4·(0.5/127)²
// and Σ|cᵢ| is maximal at 2 for a unit quaternion (all four components ±0.5), giving 0.01581. That
// worst case is real rather than theoretical: {0.5, 0.5, 0.5, 0.5} encodes as 64/127 per component
// and decodes to a squared norm of 1.015810, which a 1e-2 tolerance would have rejected — a
// conforming asset refused by the loader. 0.05 clears it by about three times.
//
// Anything beyond this is not an imprecise rotation but a different kind of value (a derivative, an
// uninitialised field, a scaled quaternion someone forgot to normalise), and it is refused rather
// than silently rescued: {0, 0, 10, 10} deviates by 199.
inline constexpr float kRotationAdmissionToleranceSquared = 0.05f;

// INVARIANT: how far a value this type has already accepted may sit from unit length before
// something is wrong with this type. Sized from the reconnaissance — the worst deviation observed
// anywhere in the engine was 7.15e-07, about 1.4e-06 squared — so 1e-4 leaves two decades of
// headroom while still catching a genuine failure of the normalisation path.
//
// BOTH ARE MEASURED ON |‖q‖² − 1|, the SQUARED deviation, and that is pinned here because the two
// spellings differ by a factor of two (|‖q‖² − 1| ≈ 2·|‖q‖ − 1| for small deviations): a linear
// tolerance transcribed in unchanged would be twice as strict as intended, and vice versa.
inline constexpr float kRotationUnitToleranceSquared = 1.0e-4f;

class Rotation3
{
public:
    // The identity rotation. A default-constructed `Rotation3` IS a rotation — there is no empty or
    // invalid state to check for, which is the whole point of the type.
    constexpr Rotation3() noexcept = default;
    constexpr Rotation3(const Rotation3&) noexcept = default;
    constexpr Rotation3(Rotation3&&) noexcept = default;
    constexpr Rotation3& operator=(const Rotation3&) noexcept = default;
    constexpr Rotation3& operator=(Rotation3&&) noexcept = default;
    ~Rotation3() = default;

    [[nodiscard]] static constexpr Rotation3 identity() noexcept
    {
        return Rotation3{};
    }

    // ADMISSION from four raw components. Normalises what it accepts, so an orientation that has
    // drifted a few ulps — or a quantised glTF keyframe — becomes an exact rotation rather than
    // being rejected for imprecision.
    //
    // It refuses what is not a rotation: a non-finite component, a magnitude too small to carry a
    // direction, or a deviation beyond `kRotationAdmissionToleranceSquared`. That last one is the
    // difference between "imprecise" and "a different kind of value": {0, 0, 10, 10} normalises to
    // a perfectly good rotation, and accepting it would mean this factory could not tell a rotation
    // from a derivative. Answering the identity for any of them would launder a producer bug into a
    // plausible value that rotates nothing.
    [[nodiscard]] static std::optional<Rotation3> tryFromQuaternion(const Quaternion& q) noexcept
    {
        if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
            !std::isfinite(q.w()))
        {
            return std::nullopt;
        }
        const float squared = q.magnitudeSquared();
        if (!std::isfinite(squared) ||
            std::fabs(squared - 1.0f) > kRotationAdmissionToleranceSquared)
        {
            return std::nullopt;
        }
        // NOT `Rotation3{Quaternion::normalise(q)}`: the choke point normalises, so pre-normalising
        // here would round twice for one value.
        return Rotation3{q};
    }

    // A rotation of `angle` radians about `axis`. The AXIS IS NORMALISED — a caller with a
    // direction of length 0.9998, or of length 7, means the direction — so the failure condition is
    // degenerate geometry (zero-length or non-finite), never "you did not hand me a unit vector".
    // `angle` itself must be finite; there is no rotation by NaN radians.
    [[nodiscard]] static std::optional<Rotation3> tryFromAxisAngle(const Vec3& axis,
                                                                   float angle) noexcept
    {
        if (!std::isfinite(angle) || !isFinite(axis) || axis.magnitude() < float_normalise_cutoff)
        {
            return std::nullopt;
        }
        const Vec3 unitAxis = Vec3::normalise(axis);
        const float half = angle * 0.5f;
        const float s = std::sin(half);
        return Rotation3{
            Quaternion{unitAxis.x() * s, unitAxis.y() * s, unitAxis.z() * s, std::cos(half)}};
    }

    // The shortest rotation taking `from` to `to`. Both are normalised, so lengths are irrelevant
    // and only directions matter; degenerate or non-finite input is refused. Antiparallel input is
    // NOT degenerate — it is a well-defined 180° rotation about some perpendicular axis.
    [[nodiscard]] static std::optional<Rotation3> tryFromVectors(const Vec3& from,
                                                                 const Vec3& to) noexcept
    {
        if (!isFinite(from) || !isFinite(to) || from.magnitude() < float_normalise_cutoff ||
            to.magnitude() < float_normalise_cutoff)
        {
            return std::nullopt;
        }
        const Vec3 f = Vec3::normalise(from);
        const Vec3 t = Vec3::normalise(to);
        const float d = Vec3::dotProduct(f, t);

        // ANTIPARALLEL: the half-angle form below degenerates to the zero quaternion, because there
        // is no shortest arc — every axis perpendicular to `f` turns it into `t` through 180°. One
        // is chosen deterministically, from whichever cardinal axis `f` is least aligned with, so
        // the result is stable rather than dependent on rounding.
        constexpr float kAntiparallel = -1.0f + 1.0e-6f;
        if (d <= kAntiparallel)
        {
            const Vec3 reference =
                std::fabs(f.x()) < 0.9f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
            const Vec3 axis = Vec3::normalise(Vec3::crossProduct(f, reference));
            return Rotation3{Quaternion{axis.x(), axis.y(), axis.z(), 0.0f}};
        }

        // The half-angle form, left UNNORMALISED for the choke point to finish — one rounding for
        // the whole factory rather than one here and another on construction.
        const Vec3 axis = Vec3::crossProduct(f, t);
        return Rotation3{Quaternion{axis.x(), axis.y(), axis.z(), 1.0f + d}};
    }

    // The unit quaternion behind this rotation, for the places that genuinely need four-component
    // algebra (swing-twist decomposition, spline tangents) and for conversion. Always unit, so a
    // caller can rely on that without checking.
    [[nodiscard]] constexpr const Quaternion& quaternion() const noexcept
    {
        return q_;
    }

    [[nodiscard]] constexpr Vec3 rotate(const Vec3& v) const noexcept
    {
        return q_.rotate(v);
    }

    // The inverse rotation. Exact for a unit quaternion — conjugation flips three signs and touches
    // no magnitudes — so this is the one algebraic member that cannot perturb the invariant.
    [[nodiscard]] constexpr Rotation3 inverse() const noexcept
    {
        return Rotation3{q_.conjugate(), AlreadyUnit{}};
    }

    // Composition: `a * b` applies b, then a. Renormalised, because a product of two unit
    // quaternions is unit only in exact arithmetic — and a chain of them (a skeleton, an
    // articulation) compounds the error link by link.
    [[nodiscard]] Rotation3 operator*(const Rotation3& rhs) const noexcept
    {
        return Rotation3{q_ * rhs.q_};
    }

    [[nodiscard]] Vec3 operator*(const Vec3& v) const noexcept
    {
        return rotate(v);
    }

    // Advance by an angular velocity over `dt` (exponential map): build the incremental rotation
    // from the rotation vector ω·dt and compose. NORMALISED EXACTLY ONCE, at the choke point — the
    // formula lives here rather than delegating to `Quaternion::integrate`, which normalises its
    // own result and would leave this rounding twice for no benefit.
    //
    // Non-finite ω or dt is a corrupt simulation state, not a case to interpolate through, and the
    // choke point terminates on it rather than storing NaNs.
    [[nodiscard]] Rotation3 integrate(const Vec3& omega, float dt) const noexcept
    {
        const Vec3 rotationVector = omega * dt;
        const float angle = rotationVector.magnitude();
        Quaternion delta;
        if (angle < float_normalise_cutoff)
        {
            // Small angle: Δq ≈ {ω·dt/2, 1}, normalised by the choke point below.
            delta = Quaternion{rotationVector.x() * 0.5f, rotationVector.y() * 0.5f,
                               rotationVector.z() * 0.5f, 1.0f};
        }
        else
        {
            const float half = angle * 0.5f;
            const float s = std::sin(half) / angle;
            delta = Quaternion{rotationVector.x() * s, rotationVector.y() * s,
                               rotationVector.z() * s, std::cos(half)};
        }
        return Rotation3{delta * q_};
    }

    // Spherical interpolation. Also normalised exactly once: the shortest-arc flip and the
    // near-parallel linear fallback both feed the same choke point.
    [[nodiscard]] static Rotation3 slerp(const Rotation3& a, const Rotation3& b, float t) noexcept
    {
        float dot = Quaternion::dotProduct(a.q_, b.q_);
        Quaternion target = b.q_;
        if (dot < 0.0f) // take the shortest arc across the double cover
        {
            target = Quaternion{-target.x(), -target.y(), -target.z(), -target.w()};
            dot = -dot;
        }

        // Near-parallel: the general form divides by sin(θ), which vanishes. Linear blending of two
        // nearly equal rotations is accurate to well within the normalisation that follows.
        constexpr float kLinearBlendThreshold = 0.9995f;
        if (dot > kLinearBlendThreshold)
        {
            return Rotation3{Quaternion{
                a.q_.x() + (target.x() - a.q_.x()) * t, a.q_.y() + (target.y() - a.q_.y()) * t,
                a.q_.z() + (target.z() - a.q_.z()) * t, a.q_.w() + (target.w() - a.q_.w()) * t}};
        }

        const float theta = std::acos(dot);
        const float sinTheta = std::sin(theta);
        const float wa = std::sin((1.0f - t) * theta) / sinTheta;
        const float wb = std::sin(t * theta) / sinTheta;
        return Rotation3{
            Quaternion{a.q_.x() * wa + target.x() * wb, a.q_.y() * wa + target.y() * wb,
                       a.q_.z() * wa + target.z() * wb, a.q_.w() * wa + target.w() * wb}};
    }

    // HEMISPHERE SELECTION, and deliberately not spelled `-q`. Negating a quaternion does not
    // produce a different rotation, so an `operator-` on this type would read as an inverse and
    // silently be a no-op — the kind of name that lies. Interpolation and error metrics need the
    // representative on the same side of the double cover as a reference, and that is what this
    // says out loud. Unary negation stays on `Quaternion`, where it means what it looks like.
    [[nodiscard]] constexpr Rotation3 alignedTo(const Rotation3& reference) const noexcept
    {
        const float dot = Quaternion::dotProduct(q_, reference.q_);
        return dot < 0.0f ? Rotation3{Quaternion{-q_.x(), -q_.y(), -q_.z(), -q_.w()}, AlreadyUnit{}}
                          : *this;
    }

    // The angle of the shortest rotation between the two, in radians, in [0, π].
    //
    // COMPUTED FROM THE RELATIVE ROTATION, in double, and not as `2·acos(|dot|)`. That form loses
    // the answer exactly where it matters: for a small angle θ the dot product is cos(θ/2), which
    // rounds to 1.0f below about θ = 5e-4, so every difference finer than that reports as zero —
    // coarser than the default tolerance of this type's own approximate comparison. Taking
    // `atan2(‖vec‖, |w|)` of the relative quaternion reads the angle off the vector part, which is
    // ≈ θ/2 for small θ and suffers no cancellation at all.
    [[nodiscard]] float angleTo(const Rotation3& other) const noexcept
    {
        const auto ax = static_cast<double>(q_.x());
        const auto ay = static_cast<double>(q_.y());
        const auto az = static_cast<double>(q_.z());
        const auto aw = static_cast<double>(q_.w());
        const auto bx = static_cast<double>(other.q_.x());
        const auto by = static_cast<double>(other.q_.y());
        const auto bz = static_cast<double>(other.q_.z());
        const auto bw = static_cast<double>(other.q_.w());

        // conj(this) * other — the rotation taking this to other.
        const double rx = aw * bx - ax * bw - ay * bz + az * by;
        const double ry = aw * by + ax * bz - ay * bw - az * bx;
        const double rz = aw * bz - ax * by + ay * bx - az * bw;
        const double rw = aw * bw + ax * bx + ay * by + az * bz;

        const double vectorPart = std::sqrt(rx * rx + ry * ry + rz * rz);
        return static_cast<float>(2.0 * std::atan2(vectorPart, std::fabs(rw)));
    }

    // SAME ROTATION, not same representation: `q` and `-q` are equal here because they rotate every
    // vector identically. A rotation type whose equality said otherwise would be answering about
    // its storage, and every caller comparing orientations would have to know about the double
    // cover.
    //
    // EXACT up to that double cover — it is equality, not approximation. Two independently computed
    // orientations will rarely satisfy it (a rotation composed with its own inverse is the identity
    // only in exact arithmetic); `approxEqual` is for those, and it answers in radians.
    [[nodiscard]] friend constexpr bool operator==(const Rotation3& lhs,
                                                   const Rotation3& rhs) noexcept
    {
        return lhs.q_ == rhs.q_ ||
               lhs.q_ == Quaternion{-rhs.q_.x(), -rhs.q_.y(), -rhs.q_.z(), -rhs.q_.w()};
    }

    // Exact component equality, for the rare caller that means the REPRESENTATION — a serialiser
    // checking round-trip fidelity, a test pinning which hemisphere a factory chose. Named so that
    // reaching for it is a decision rather than an accident.
    [[nodiscard]] constexpr bool sameComponents(const Rotation3& other) const noexcept
    {
        return q_ == other.q_;
    }

    // Approximate equality as an ANGLE, not as four component tolerances. Comparing components
    // independently answers a question nobody asks: two rotations can differ in every component and
    // be a thousandth of a degree apart, or agree closely in three and be far apart.
    //
    // `toleranceRadians` is validated the way `almostEqual`'s tolerances are: negative, NaN or
    // infinite is a caller defect, and the answer to one is false rather than "everything matches".
    [[nodiscard]] bool approxEqual(const Rotation3& other,
                                   float toleranceRadians = 1.0e-4f) const noexcept
    {
        if (!std::isfinite(toleranceRadians) || toleranceRadians < 0.0f)
        {
            return false;
        }
        return angleTo(other) <= toleranceRadians;
    }

    // Whether a quaternion is unit to within `kRotationUnitToleranceSquared`, on the SQUARED
    // deviation. Exposed because tests and assertions both need to ask it in the same units.
    [[nodiscard]] static bool isUnit(const Quaternion& q) noexcept
    {
        const float squared = q.magnitudeSquared();
        return std::isfinite(squared) && std::fabs(squared - 1.0f) <= kRotationUnitToleranceSquared;
    }

private:
    // The failure path for a violated invariant — logged and terminal. A PRIVATE member, defined
    // out of line: it is this type's own business, and a free function in the `fire_engine`
    // namespace would be an externally callable "terminate the process" that nobody should have.
    // Out of line so this header does not pull the logger into every translation unit that needs a
    // rotation, for a path that never runs.
    [[noreturn]] static void invariantViolated(const char* what) noexcept;

    [[nodiscard]] static bool isFinite(const Vec3& v) noexcept
    {
        return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
    }

    // Tag for the constructions that provably preserve unit length (conjugation, negation), so they
    // can skip a `sqrt` without anybody being able to skip it by accident: the tag is private, so
    // only members of this class can make that claim.
    struct AlreadyUnit
    {
    };

    // THE CHOKE POINT — private, so there is no way to build a `Rotation3` that is not one, and
    // TERMINAL, so there is no way for one to finish construction holding something that is not a
    // rotation.
    //
    // `Quaternion::normalise` answers NaNs for non-finite input (deliberately: visibly invalid
    // beats a laundered zero) and the identity for a degenerate one. Storing either would defeat
    // the entire type — the first with values that poison everything downstream, the second with a
    // confident "no rotation" that came from a bug. Public fallible boundaries (`tryFrom*`) filter
    // those cases into `nullopt` before they reach here, so arriving with one means an internal
    // operation was handed corrupt input: a NaN angular velocity, an infinite timestep. There is no
    // useful way to continue a simulation from that, and continuing quietly is how it surfaces
    // three seconds later somewhere unrelated.
    explicit Rotation3(const Quaternion& q) noexcept
        : q_{normalisedOrTerminate(q)}
    {
    }

    constexpr Rotation3(const Quaternion& q, AlreadyUnit) noexcept
        : q_{q}
    {
    }

    [[nodiscard]] static Quaternion normalisedOrTerminate(const Quaternion& q) noexcept
    {
        if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
            !std::isfinite(q.w()))
        {
            invariantViolated("a rotation was computed from a non-finite quaternion");
        }
        if (q.magnitude() < float_normalise_cutoff)
        {
            invariantViolated("a rotation was computed from a degenerate quaternion");
        }
        const Quaternion normalised = Quaternion::normalise(q);
        if (!isUnit(normalised))
        {
            invariantViolated("normalisation failed to produce a unit rotation");
        }
        return normalised;
    }

    Quaternion q_{Quaternion::identity()};
};

} // namespace fire_engine
