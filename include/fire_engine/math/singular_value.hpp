#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

// A CONSERVATIVE bound on the largest singular value of a 3x3 (its spectral norm): the greatest
// factor by which the matrix stretches any direction. σ_max(m) = sqrt(λ_max(mᵀm)); bound
// λ_max(mᵀm) by the Gershgorin / induced-∞-norm bound ‖mᵀm‖∞ (the max absolute row sum of the
// symmetric Gram matrix), which for any symmetric PSD matrix is a GUARANTEED upper bound on its
// spectral radius. So σ_max is never UNDER-estimated — an object-space radius bounded into world
// space stays conservative (an instance never under-refines) and a conditioning test never wrongly
// trusts a near-singular transform. EXACT for orthogonal (mᵀm = I) and diagonal (scale) transforms;
// a modest over-estimate for a sheared/rotated mix (off-diagonal mass inflates the row sum), which
// only over-refines — the safe direction.
//
// The Gram matrix + row sums are computed in DOUBLE and the float result is then stepped one ULP
// toward +∞, so float rounding of the products can't push the answer below the true bound (the
// bound is only conservative in exact arithmetic otherwise). The step is UNCONDITIONAL: when the
// double→float conversion already rounded upward this spends one extra ULP it did not have to.
// That is deliberate — a fixed one-ULP margin is cheaper to reason about than a conditional
// round-direction test, and erring upward is the safe direction for every caller.
//
// A finite power iteration is NOT usable here: a fixed start vector orthogonal to the dominant
// eigenvector converges to the WRONG (smaller) eigenvalue no matter how many iterations run — e.g.
// the in-plane shear [[5.5,-4.5,0],[-4.5,5.5,0],[0,0,1]] has σ_max = 10 but its dominant mᵀm
// eigenvector (1,-1,0) is orthogonal to (1,1,1), so power iteration returns 1 (a 10x
// under-estimate).
//
// A SINGULAR matrix is ordinary here, not a special case: its largest singular value is finite (a
// zero-scale axis only removes stretch). A NON-FINITE element is meaningless, and returns positive
// infinity so the caller rejects the transform outright — this function never invents a plausible
// number for one.
//
// Shared between VDPM's per-instance refinement and the shadow-LOD projection (SH-02): both bound
// an object-space deviation radius into world space, and two copies of a subtle numeric bound would
// eventually disagree.
[[nodiscard]] inline float largestSingularValue(const Mat3& m) noexcept
{
    // Reject non-finite input UP FRONT. Left to the arithmetic, a NaN would be swallowed:
    // std::max(finite, NaN) returns the finite operand on every implementation that compiles to a
    // single comparison, so a NaN column could silently yield a plausible scale — the opposite of
    // the propagation contract above, and a way for one bad transform to under-refine an instance.
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            if (!std::isfinite(m[row, col]))
            {
                return std::numeric_limits<float>::infinity();
            }
        }
    }

    double maxRowSum = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        double rowSum = 0.0;
        for (int j = 0; j < 3; ++j)
        {
            double gram = 0.0; // (mᵀm)_ij = Σ_k m_ki · m_kj
            for (int k = 0; k < 3; ++k)
            {
                gram += static_cast<double>(m[k, i]) * static_cast<double>(m[k, j]);
            }
            rowSum += std::abs(gram);
        }
        maxRowSum = std::max(maxRowSum, rowSum);
    }
    const double sigma = std::sqrt(std::max(0.0, maxRowSum));
    // Outward step: the result is >= the double bound >= true σ_max. Not the SMALLEST such float —
    // the step is unconditional, so when the conversion already rounded up this is one ULP beyond
    // it (see the note above).
    return std::nextafter(static_cast<float>(sigma), std::numeric_limits<float>::infinity());
}

// The linear (upper-left 3x3) part of a world transform — the part that stretches a direction.
// Translation cannot scale a deviation radius, so it is dropped.
[[nodiscard]] inline Mat3 linearPart(const Mat4& world) noexcept
{
    return Mat3::fromColumns({world[0, 0], world[1, 0], world[2, 0]},
                             {world[0, 1], world[1, 1], world[2, 1]},
                             {world[0, 2], world[1, 2], world[2, 2]});
}

} // namespace fire_engine
