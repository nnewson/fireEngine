#pragma once

#include <cstdint>
#include <string_view>

// What a shadow view DOES in a frame, and whether this frame takes reuse at all (arc 2 #4).
//
// Its own header because two files that cannot include each other both need it: the plan
// (`shadow_pass_plan.hpp`) produces a disposition per view, and the diagnostics
// (`shadow_diagnostics.hpp`, which the plan includes) report the one each row ended up with. The
// LAW that produces it — what counts as identical content — stays with the plan, beside the
// residency and prepared-view types it reasons about.

namespace fire_engine
{

// What a view does this frame. Three states, and deliberately NOT folded into `ShadowMapValidity` —
// that type answers the shader's question ("is this map safe to sample"), which stays Boolean. This
// answers the pass's question, which is a different one: a CSM with two cascades recorded and two
// reused is entirely valid and half the work.
enum class ShadowViewDisposition : std::uint8_t
{
    // Not engaged this frame, or engaged with nothing sampleable behind it. Nothing to record and
    // nothing to sample.
    Invalid,
    // Resident content matches; the image already holds the right depth. Records nothing.
    Reused,
    // Records this frame, either because the content changed or because it can never be cached.
    Recorded,
};

[[nodiscard]] std::string_view toString(ShadowViewDisposition disposition) noexcept;

// Whether this frame TAKES reuse at all (`RenderTunables::shadowResidencyReuseEnabled`).
//
// SCHEDULING, NOT PIXELS, which is why it is an argument to the law rather than a field of the
// content descriptor: it decides whether identical content is re-rasterised, never what that
// content is. A frame recorded with reuse disabled is therefore perfectly reusable by a later frame
// with it enabled — recording commits residency either way — so the toggle can be flipped mid-run
// and the next frame picks up from the newest content rather than from whatever was resident when
// it was switched off. That is what makes it an honest A/B for the whole item.
enum class ShadowReusePolicy : std::uint8_t
{
    Enabled,
    Disabled,
};

// The two derived questions. Everything downstream asks one of these rather than testing the
// enumerator, so "reused counts as sampleable" is stated once.
[[nodiscard]] constexpr bool shadowViewSampleable(ShadowViewDisposition disposition) noexcept
{
    return disposition == ShadowViewDisposition::Reused ||
           disposition == ShadowViewDisposition::Recorded;
}
[[nodiscard]] constexpr bool shadowViewRecords(ShadowViewDisposition disposition) noexcept
{
    return disposition == ShadowViewDisposition::Recorded;
}

} // namespace fire_engine
