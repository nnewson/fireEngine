#pragma once

namespace fire_engine
{

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float deg_to_rad = pi / 180.0f;
inline constexpr float rad_to_deg = 180.0f / pi;
inline constexpr float float_epsilon = 1e-8f;

// Soft pitch clamp for first-person cameras. Just under π/2 (≈85.94°) — keeps
// the lookAt basis well-conditioned at the poles by avoiding the degenerate
// straight-up / straight-down case while still letting the player look near
// vertical.
inline constexpr float kCameraMaxPitch = 1.5f;

} // namespace fire_engine
