#include "fire_engine/math/rotation3.hpp"

#include <cstdlib>

#include <fire_engine/core/log.hpp>

namespace fire_engine
{

// Out of line so `rotation3.hpp` — included by scene, animation, physics and render — does not pull
// the logger into every one of their translation units for a path that never runs.
//
// TERMINAL, and deliberately so. Reaching here means an operation that promises to produce a
// rotation was handed something that cannot become one: a non-finite angular velocity, an infinite
// timestep, a corrupt orientation from upstream. The alternatives are worse in the ways this branch
// exists to prevent — storing NaNs gives every later operation a plausible-looking value that
// poisons whatever it touches, and substituting the identity turns a corrupt orientation into a
// confident "no rotation" that surfaces three seconds later somewhere unrelated.
void Rotation3::invariantViolated(const char* what) noexcept
{
    log::error(log::category::general, "Rotation3 invariant violated: {}", what);
    std::abort();
}

} // namespace fire_engine
