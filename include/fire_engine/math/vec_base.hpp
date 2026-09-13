#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/scalar.hpp>

namespace fire_engine
{

template <typename Derived, std::size_t N>
class VecBase
{
public:
    [[nodiscard]]
    constexpr Derived operator-(const Derived& rhs) const noexcept
    {
        Derived result{self()};
        result -= rhs;
        return result;
    }

    constexpr Derived& operator-=(const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] -= rhs.data_[i];
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator+(const Derived& rhs) const noexcept
    {
        Derived result{self()};
        result += rhs;
        return result;
    }

    constexpr Derived& operator+=(const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] += rhs.data_[i];
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator*(const float rhs) const noexcept
    {
        Derived result{self()};
        result *= rhs;
        return result;
    }

    constexpr Derived& operator*=(const float rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] *= rhs;
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator/(const float rhs) const noexcept
    {
        Derived result{self()};
        result /= rhs;
        return result;
    }

    constexpr Derived& operator/=(const float rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] /= rhs;
        }
        return self();
    }

    [[nodiscard]]
    static constexpr float dotProduct(const Derived& lhs, const Derived& rhs) noexcept
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            sum += lhs.data_[i] * rhs.data_[i];
        }
        return sum;
    }

    [[nodiscard]]
    constexpr float dotProduct(const Derived& rhs) const noexcept
    {
        return Derived::dotProduct(self(), rhs);
    }

    // A FAST PATH THAT IS THE OLD ARITHMETIC, and a scaled fallback for the cases where the old
    // arithmetic was wrong.
    //
    // `sqrt(dot(v, v))` fails at both ends of float's range: components above ~1.8e19 square to
    // infinity, so a vector whose norm is perfectly representable reports infinity; components
    // below ~1e-22 square to zero, so a small but ordinary vector reports zero and then normalises
    // away to nothing. Both failures ANNOUNCE THEMSELVES in the sum — it comes back infinite, NaN,
    // or exactly zero — so the naive sum can be computed first and trusted whenever it is finite
    // and positive, which is every vector a frame of this engine actually contains.
    //
    // That matters for two reasons beyond speed (the scaled form measured ~3x the cost of this
    // one). It keeps ordinary results BIT-IDENTICAL to what the engine computed before, so the
    // physics goldens do not move for a change that was supposed to be about extreme values. And
    // it confines the robust path to inputs where there was no correct answer before it.
    //
    // Special values are decided rather than inherited: any NaN component makes the norm NaN
    // (checked first, so a vector holding both a NaN and an infinity is NaN rather than depending
    // on iteration order); any infinity with no NaN makes it infinite; an all-zero vector is zero.
    //
    // std::sqrt is not constexpr before C++26, so this and the normalise helpers cannot be either.
    [[nodiscard]] float magnitude() const noexcept
    {
        const float sumOfSquares = magnitudeSquared();
        // `isfinite` rather than a bare `< infinity` comparison: the two classify identically here
        // (a NaN fails every comparison, an infinity fails the bound, zero fails the first test)
        // and they measured identically too, so the one that says what it means wins.
        // NORMAL, not merely positive. A sum that has gone SUBNORMAL has already lost most of its
        // precision without reaching zero: (3e-23, 3e-23, 0) sums to 2.8e-45, which carries about
        // two significant bits, and `sqrt` of it answers 5.29e-23 against a true 4.24e-23 — a 24.8%
        // error from a fast path that thought it was fine because the sum was finite and positive.
        // Requiring the sum to be at least `float`'s smallest NORMAL value routes that whole region
        // to the scaled form, where the components are rescaled before they are squared and no
        // precision is lost at all. `isfinite` then rules out the top end; a NaN fails both.
        if (sumOfSquares >= std::numeric_limits<float>::min() && std::isfinite(sumOfSquares))
        {
            return std::sqrt(sumOfSquares);
        }
        return scaledMagnitude();
    }

    // The raw sum of squares, which OVERFLOWS where `magnitude()` does not — for large components
    // it is infinity and for tiny ones zero. That is honest for what it is (a squared quantity has
    // half the exponent range available to it), and it stays because comparisons of squared lengths
    // are a legitimate and cheaper thing to want. Reach for `magnitude()` when the answer is the
    // length itself.
    [[nodiscard]] constexpr float magnitudeSquared() const noexcept
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            sum += data_[i] * data_[i];
        }
        return sum;
    }

    // Normalised through the same fast path, for the same reasons: while the sum of squares is
    // finite and positive, this is exactly the division the engine did before — one rounding per
    // component, bit-identical results. Rounding twice instead (dividing by the largest component
    // and then by a scaled norm) costs an ulp per component, which sounds like nothing and delayed
    // a settling box stack from step 169 to 425 on macOS and 1309 on Linux against a 600-step
    // budget. A solver notices an ulp; that is what solvers are.
    //
    // Three answers, and each is deliberate:
    //   * a non-finite component yields a NaN vector — VISIBLY invalid. Returning the zero vector
    //     or some identity would launder a corrupt input into a plausible value.
    //   * a magnitude below `float_normalise_cutoff` yields the zero vector, as it always has:
    //     there is no direction to report.
    //   * anything else is normalised — including a vector whose LENGTH is unrepresentable but
    //     whose direction is ordinary, which is the case that needs the scaled form.
    [[nodiscard]]
    static Derived normalise(const Derived& v) noexcept
    {
        const float sumOfSquares = v.magnitudeSquared();
        // A NORMAL sum, for the subnormal-precision reason given on `magnitude`.
        if (sumOfSquares >= std::numeric_limits<float>::min() && std::isfinite(sumOfSquares))
        {
            const float length = std::sqrt(sumOfSquares);
            if (length < float_normalise_cutoff)
            {
                return Derived{};
            }
            Derived result{v};
            for (std::size_t i = 0; i < N; ++i)
            {
                result.data_[i] /= length;
            }
            return result;
        }
        return scaledNormalise(v);
    }

    Derived& normalise() noexcept
    {
        self() = normalise(self());
        return self();
    }

    // EXACT component-wise IEEE equality — not bitwise, despite what this used to claim. Two
    // differences matter and both are the float `==` operator's, not ours: `-0.0f` equals `+0.0f`
    // though their bit patterns differ, and a NaN equals nothing at all though its bit pattern is
    // identical to itself. Use `approxEqual` when you want tolerance; if a determinism diagnostic
    // ever needs REAL bit comparison, it has to say so with `std::bit_cast`.
    [[nodiscard]]
    friend constexpr bool operator==(const Derived& lhs, const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (lhs.data_[i] != rhs.data_[i])
            {
                return false;
            }
        }
        return true;
    }

    // Approximate equality, component by component, through the ONE scalar authority
    // (`math/scalar.hpp`), and in its three forms — no argument means both defaults, an explicit
    // tolerance means ABSOLUTE ONLY (so `approxEqual(rhs, 1e-9f)` rejects anything further apart
    // than 1e-9, exactly as it always did), and both arguments mean both terms. An invalid
    // tolerance — negative, NaN or infinite — makes the comparison FALSE rather than being
    // reinterpreted. NaNs compare unequal now, which is the defect this replaced.
    [[nodiscard]]
    constexpr bool approxEqual(const Derived& rhs, float eps, float relativeEps) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (!almostEqual(data_[i], rhs.data_[i], eps, relativeEps))
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Derived& rhs, float eps) const noexcept
    {
        // ABSOLUTE ONLY — a stated tolerance is the whole answer.
        return approxEqual(rhs, eps, 0.0f);
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Derived& rhs) const noexcept
    {
        return approxEqual(rhs, float_epsilon, float_relative_epsilon);
    }

protected:
    // THE ROBUST PATH, reached only when the sum of squares was zero, infinite or NaN — i.e. when
    // the naive computation had no answer to give. Kept out of line from the hot path above.
    [[nodiscard]] float scaledMagnitude() const noexcept
    {
        float largest = 0.0f;
        bool anyInfinite = false;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (std::isnan(data_[i]))
            {
                return data_[i]; // a NaN outranks an infinity elsewhere in the vector
            }
            if (std::isinf(data_[i]))
            {
                anyInfinite = true;
                continue;
            }
            const float componentMagnitude = std::fabs(data_[i]);
            largest = componentMagnitude > largest ? componentMagnitude : largest;
        }
        if (anyInfinite)
        {
            return std::numeric_limits<float>::infinity();
        }
        if (largest == 0.0f)
        {
            return 0.0f;
        }

        float sumOfScaledSquares = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            const float scaled = data_[i] / largest;
            sumOfScaledSquares += scaled * scaled;
        }
        return largest * std::sqrt(sumOfScaledSquares);
    }

    [[nodiscard]] static Derived scaledNormalise(const Derived& v) noexcept
    {
        float largest = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (!std::isfinite(v.data_[i]))
            {
                Derived invalid{};
                for (std::size_t j = 0; j < N; ++j)
                {
                    invalid.data_[j] = std::numeric_limits<float>::quiet_NaN();
                }
                return invalid;
            }
            const float componentMagnitude = std::fabs(v.data_[i]);
            largest = componentMagnitude > largest ? componentMagnitude : largest;
        }
        if (largest == 0.0f)
        {
            return Derived{};
        }

        float sumOfScaledSquares = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            const float scaled = v.data_[i] / largest;
            sumOfScaledSquares += scaled * scaled;
        }
        const float scaledNorm = std::sqrt(sumOfScaledSquares); // in [1, sqrt(N)]
        const float length = largest * scaledNorm;
        if (length < float_normalise_cutoff)
        {
            return Derived{};
        }
        if (std::isfinite(length))
        {
            Derived result{v};
            for (std::size_t i = 0; i < N; ++i)
            {
                result.data_[i] = v.data_[i] / length;
            }
            return result;
        }
        // The length itself is unrepresentable — a vector of finite components whose norm exceeds
        // float's range. The direction is still well defined, and dividing by an infinite length
        // would answer zero, so the two-step scaled form is the only way to keep it.
        Derived result{v};
        for (std::size_t i = 0; i < N; ++i)
        {
            result.data_[i] = (v.data_[i] / largest) / scaledNorm;
        }
        return result;
    }

protected:
    float data_[N]{};

private:
    [[nodiscard]] constexpr Derived& self() noexcept
    {
        return static_cast<Derived&>(*this);
    }

    [[nodiscard]] constexpr const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace fire_engine
