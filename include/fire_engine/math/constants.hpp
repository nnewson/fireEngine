#pragma once

namespace fire_engine
{

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float deg_to_rad = pi / 180.0f;
inline constexpr float rad_to_deg = 180.0f / pi;
inline constexpr float float_epsilon = 1e-8f;

// Below this magnitude a vector or quaternion has no reliable DIRECTION, so `normalise` returns its
// documented degenerate answer (the zero vector, or the identity rotation) rather than dividing.
//
// Its own constant, deliberately. `float_epsilon` was doing this job as well as being the absolute
// comparison tolerance, which meant one number carried two unrelated policies and neither could be
// tuned without disturbing the other. The VALUE is unchanged, so this commit moves no behaviour
// with the rename; the point is that the next person changing a comparison tolerance does not
// silently change what counts as a degenerate direction.
inline constexpr float float_normalise_cutoff = 1e-8f;

// Soft pitch clamp for first-person cameras. Just under π/2 (≈85.94°) — keeps
// the lookAt basis well-conditioned at the poles by avoiding the degenerate
// straight-up / straight-down case while still letting the player look near
// vertical.
inline constexpr float kCameraMaxPitch = 1.5f;

} // namespace fire_engine
