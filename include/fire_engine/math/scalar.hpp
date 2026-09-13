#pragma once

#include <cmath>

#include <fire_engine/math/constants.hpp>

// The ONE approximate scalar comparison, and the only thing in this library that decides whether
// two numbers are close enough (tier-0 review, finding 2).
//
// It exists because the pattern it replaces was wrong in a way that hid exactly the failures tests
// are for. Every vector, matrix and quaternion `approxEqual` computed
//
//     const float diff = a - b;
//     if (diff > eps || diff < -eps) { return false; }
//
// and BOTH comparisons are false when `diff` is NaN, so a NaN compared approximately equal to
// everything, itself included. Two equal infinities subtract to NaN and passed the same way — which
// reads as correct until you notice that opposite infinities do too. A test asserting a transform
// stayed finite could pass on a transform that was entirely NaN.

namespace fire_engine
{

// The RELATIVE term's default. Deliberately its own constant rather than `float_epsilon`: that one
// is the absolute tolerance, and a single number serving as absolute tolerance, relative tolerance
// and degeneracy threshold is three unrelated policies wearing one name.
//
// 1e-6 is roughly ten times float's 1.19e-7 epsilon — close enough to machine precision to reject
// real error, loose enough to absorb the last couple of bits after a few operations.
inline constexpr float float_relative_epsilon = 1.0e-6f;

// THREE OVERLOADS, because the tolerance a caller SPELLS OUT must mean what it says.
//
//   almostEqual(a, b)                       default absolute AND default relative
//   almostEqual(a, b, absolute)             absolute only — exactly the historical comparison
//   almostEqual(a, b, absolute, relative)   both, stated
//
// A single function with a defaulted relative term would have made `almostEqual(a, b, 1e-9f)` admit
// a 1e-7 difference at magnitude 1.0: an explicit tolerance silently overridden by an implicit one.
// Overload selection decides the policy instead, so ordinary comparisons get the large-magnitude
// fix while an explicit tolerance remains the whole answer.
//
// TOLERANCES ARE VALIDATED, not reinterpreted. A negative, NaN or infinite tolerance is a caller
// defect — a misconfigured constant, an uninitialised field, a division that went wrong — and the
// comparison refuses it by returning FALSE, even for operands that are equal. Treating a negative
// as "a term that cannot be satisfied" is mathematically defensible and diagnostically useless: it
// converts a broken configuration into a different valid policy and lets the run continue. The
// check happens BEFORE the `a == b` shortcut precisely so equal operands cannot hide it.
//
// The comparison itself, once the tolerances are known good:
//
//  1. `a == b`, so identical values are equal whatever they are — including two equal infinities,
//     which is the case the old subtraction form got right only by accident (inf - inf is NaN, and
//     NaN passed its test). This also makes +0.0 and -0.0 equal, which is correct: they are the
//     same number.
//  2. Any remaining non-finite OPERAND is not equal. After step 1 that means a NaN anywhere, or two
//     infinities that differ, or an infinity against any finite value. NaN is never equal to
//     anything, itself included — the property the old form inverted.
//  3. Finite values compare against the larger of the absolute and relative tolerances. The
//     absolute term is what makes values near zero comparable at all, where relative error is
//     meaningless; the relative term is what keeps large values comparable without a bespoke
//     tolerance per call site.
//
// DOUBLE intermediates, and not for accuracy: `a - b` in float overflows to infinity for values
// near FLT_MAX of opposite sign, so the difference itself could be non-finite for two perfectly
// finite inputs, and the comparison would be answering about a number neither caller mentioned. In
// double the subtraction of any two floats is exact.
[[nodiscard]] constexpr bool almostEqual(float a, float b, float absoluteTolerance,
                                         float relativeTolerance) noexcept
{
    // `std::isfinite`, not `(x - x) == 0`. The subtraction form classifies correctly but computes
    // `inf - inf` on the way, which is an INVALID operation: it raises FE_INVALID and would trap
    // where floating-point exceptions are enabled. A comparison must not alter exception state to
    // answer a question about its arguments. (C++23 made these constexpr, so the function stays
    // usable in constant expressions.)
    const auto validTolerance = [](float tolerance)
    { return std::isfinite(tolerance) && tolerance >= 0.0f; };
    if (!validTolerance(absoluteTolerance) || !validTolerance(relativeTolerance))
    {
        return false;
    }

    if (a == b)
    {
        return true;
    }
    if (!std::isfinite(a) || !std::isfinite(b))
    {
        return false;
    }

    const double difference = static_cast<double>(a) - static_cast<double>(b);
    const double magnitude = difference < 0.0 ? -difference : difference;
    if (magnitude <= static_cast<double>(absoluteTolerance))
    {
        return true;
    }
    const double scaleA = a < 0.0f ? -static_cast<double>(a) : static_cast<double>(a);
    const double scaleB = b < 0.0f ? -static_cast<double>(b) : static_cast<double>(b);
    const double scale = scaleA > scaleB ? scaleA : scaleB;
    return magnitude <= static_cast<double>(relativeTolerance) * scale;
}

// ABSOLUTE ONLY. `almostEqual(a, b, 1e-9f)` rejects anything more than 1e-9 apart, at any
// magnitude, which is what it looks like it does.
[[nodiscard]] constexpr bool almostEqual(float a, float b, float absoluteTolerance) noexcept
{
    return almostEqual(a, b, absoluteTolerance, 0.0f);
}

// The ordinary comparison: both defaults, so values near zero and values near FLT_MAX are each
// compared by the term that means something at their scale.
[[nodiscard]] constexpr bool almostEqual(float a, float b) noexcept
{
    return almostEqual(a, b, float_epsilon, float_relative_epsilon);
}

} // namespace fire_engine
