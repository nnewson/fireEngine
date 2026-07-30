#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>
#include <string_view>

#include <fire_engine/render/renderer.hpp>

namespace fire_engine
{

struct ApplicationArgs
{
    std::string_view scenePath{};
    std::string_view skyboxPath{};
    // -f flag: drop a 100×100 white plane at y=0 into the loaded scene so
    // shadow casters from punctual lights have something to occlude.
    bool addFloor{false};
    // -p flag: seed the demo GPU particle fountain. Off by default so normal
    // scenes are unaffected.
    bool addParticles{false};
    // -c flag: drop a demo cloth (pinned grid) into the scene.
    bool addCloth{false};
    // -k flag: drop a kinematic character-controller demo (a capsule that auto-walks an
    // obstacle course — floor, ramp, step, wall — sliding, climbing, and stepping).
    bool addCharacter{false};
    // -q flag: drop a query-probe demo — a ring of static bodies with a rotating fan of
    // raycasts + a pulsing overlap sphere, drawn each frame from PhysicsWorld queries.
    bool addQueryProbe{false};
    // --maximized / --maximised: ask the platform window to start maximized when supported.
    bool startMaximized{false};
    // Forwarded straight to the Renderer. Multiple --debug-* flags collapse to
    // the last one parsed (single debug view at a time); --no-shadows is
    // independent and combines with any view.
    RendererDebug debug{};
};

[[nodiscard]] inline bool isEnvironmentPath(std::string_view path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string_view::npos)
    {
        return false;
    }

    std::string extension(path.substr(dot));
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".hdr" || extension == ".exr";
}

[[nodiscard]] inline ApplicationArgs parseApplicationArgs(int argc, char* argv[]) noexcept
{
    ApplicationArgs args;

    // Collect positional (non-flag) args and consume known flags inline.
    std::string_view positional[2]{};
    int positionalCount = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];
        if (arg == "-f")
        {
            args.addFloor = true;
            continue;
        }
        if (arg == "-p")
        {
            args.addParticles = true;
            continue;
        }
        if (arg == "-c")
        {
            args.addCloth = true;
            continue;
        }
        if (arg == "-q")
        {
            args.addQueryProbe = true;
            continue;
        }
        if (arg == "-k")
        {
            args.addCharacter = true;
            continue;
        }
        if (arg == "--debug-normals")
        {
            args.debug.view = DebugView::Normals;
            continue;
        }
        if (arg == "--debug-ndotl")
        {
            args.debug.view = DebugView::NdotL;
            continue;
        }
        if (arg == "--debug-shadow")
        {
            args.debug.view = DebugView::Shadow;
            continue;
        }
        if (arg == "--debug-shadow-depth")
        {
            args.debug.view = DebugView::ShadowDepth;
            continue;
        }
        if (arg == "--debug-velocity")
        {
            args.debug.view = DebugView::Velocity;
            continue;
        }
        if (arg == "--debug-ssao")
        {
            args.debug.view = DebugView::Ssao;
            continue;
        }
        if (arg == "--debug-lod")
        {
            args.debug.view = DebugView::Lod;
            continue;
        }
        if (arg == "--debug-shadow-lod")
        {
            // SH-03: tints each mesh by the level ONE shadow view chose for it — the focused view
            // in the Shadows panel, or cascade 0 by default. A flag rather than overlay-only
            // because a reference capture of this view is how per-view selection is checked without
            // a human at the keyboard.
            args.debug.view = DebugView::ShadowLod;
            continue;
        }
        if (arg == "--debug-joints")
        {
            // Replaces the scene meshes with the ragdoll articulation gizmo + index:name labels.
            args.debug.view = DebugView::Joints;
            continue;
        }
        if (arg == "--no-shadows")
        {
            args.debug.noShadows = true;
            continue;
        }
        if (arg == "--no-lod")
        {
            // FULL DETAIL EVERYWHERE — the "before" half of an LOD acceptance A/B, and the contract
            // the runbook's existing commands rely on. It disables the shadow selection explicitly
            // now that the two have separate switches: before SH-03 slice 6 both read one flag, and
            // leaving this forward-only would have silently narrowed a documented promise.
            //
            // `--no-shadow-lod` is the shadow-ONLY control (forward geometry keeps selecting),
            // which is what an A/B of shadow LOD needs.
            args.debug.lod = false;
            args.debug.shadowLod = false;
            continue;
        }
        if (arg == "--capture")
        {
            // Write the captured frame to this path and exit. Same value-consuming discipline as
            // --lod-mode: a trailing `--capture`, or one followed by another flag, must not swallow
            // that flag or leave the path to be misread as a positional scene argument. With no
            // usable value the flag is simply inert (capturePath stays empty ⇒ no capture).
            if (i + 1 < argc)
            {
                const std::string_view value = argv[i + 1];
                if (!value.empty() && value.front() != '-')
                {
                    ++i;
                    args.debug.capturePath = value;
                }
            }
            continue;
        }
        if (arg == "--capture-frame")
        {
            // Explicit FRAME NUMBER, never a duration: the same frame must be captured on a slow
            // machine and a fast one, or the reference image isn't reproducible. A missing,
            // non-numeric, or non-positive value leaves the default rather than capturing frame 0.
            if (i + 1 < argc)
            {
                // Parsed from argv's own null-terminated buffer rather than a string_view's
                // data(), which clang-tidy flags as a potentially unterminated pointer.
                const char* raw = argv[i + 1];
                const std::string_view value = raw;
                if (!value.empty() && value.front() != '-')
                {
                    ++i;
                    int frame = 0;
                    const auto result = std::from_chars(raw, raw + value.size(), frame);
                    if (result.ec == std::errc{} && result.ptr == raw + value.size() && frame > 0)
                    {
                        args.debug.captureFrame = frame;
                    }
                }
            }
            continue;
        }
        if (arg == "--no-shadow-lod")
        {
            // Forces every shadow caster to LOD0 while the FORWARD selection is untouched — the
            // reference an A/B of shadow LOD needs. `--no-lod` is not that reference: it also
            // disables forward LOD, so the comparison picks up visible-geometry differences. Nor is
            // a tiny budget: the selector still runs, and a cut whose estimated deviation is 0 is
            // still eligible.
            args.debug.shadowLod = false;
            continue;
        }
        if (arg == "--shadow-budget" || arg == "--shadow-ratio")
        {
            // SH-03 slice 6's calibration handles, so a sweep is a shell loop rather than a person
            // dragging a slider and remembering where it was.
            //
            // The VALUE is recorded verbatim and validated in the renderer, where a bad one is
            // TERMINAL. Keeping the default on a malformed value — the discipline --capture-frame
            // uses — is wrong for a calibration input: a mistyped sweep step would silently
            // re-measure the default budget and produce a plausible table with one duplicated row.
            // Presence is recorded even when nothing is consumed, so a valueless flag fails too.
            const std::string_view value =
                (i + 1 < argc && argv[i + 1][0] != '\0' && argv[i + 1][0] != '-')
                    ? std::string_view{argv[++i]}
                    : std::string_view{};
            if (arg == "--shadow-budget")
            {
                args.debug.shadowLodBudget = value;
            }
            else
            {
                args.debug.shadowLodCoarsenRatio = value;
            }
            continue;
        }
        if (arg == "--shadow-focus")
        {
            // Which shadow view the diagnostics and the ShadowLod tint should follow, as
            // `<group>:<slot>` (`cascade:3`, `spot:0`, `point:0:4`). The SLOT is only how the
            // request is written — it is resolved once at startup into the identity that slot
            // holds, and the identity is what is kept.
            //
            // Unlike --capture, a MISSING value is not inert. The flag records its presence
            // (engaged optional) even when it consumes nothing, so the renderer can refuse to
            // start: silently ignoring it would run with the default focus and produce a capture of
            // cascade 0 that looks exactly like a correct capture of whatever was meant.
            args.debug.shadowFocus = std::string_view{};
            if (i + 1 < argc)
            {
                const std::string_view value = argv[i + 1];
                if (!value.empty() && value.front() != '-')
                {
                    ++i;
                    args.debug.shadowFocus = value;
                }
            }
            continue;
        }
        if (arg == "--no-taa")
        {
            args.debug.taa = false;
            continue;
        }
        if (arg == "--overlay")
        {
            args.debug.overlayVisible = true;
            continue;
        }
        if (arg == "--maximized" || arg == "--maximised")
        {
            args.startMaximized = true;
            continue;
        }
        if (arg == "--debug-physics")
        {
            args.debug.physicsDebug = true;
            continue;
        }
        if (arg == "--require-validation")
        {
            // Fail startup unless the Vulkan validation layer is actually active, so an automated
            // run can't report a clean frame when nothing was validating it (a Dev build on a
            // machine with no SDK, or an NDEBUG build). Used by the render smoke.
            args.debug.requireValidation = true;
            continue;
        }
        if (arg == "--vdpm-gpu")
        {
            // Force the GPU-driven VDPM backend ON (rendering-spine #3). Effective only with
            // `--lod-mode view-dependent` on a compute/scan-capable device. Since B5c-4 the backend
            // defaults ON where supported, so this is now an explicit override (and lets the render
            // smoke test pin the compute path); --no-vdpm-gpu forces it OFF. Last flag wins.
            args.debug.vdpmGpuBackend = true;
            continue;
        }
        if (arg == "--no-vdpm-gpu")
        {
            // Force the GPU-driven VDPM backend OFF (use the CPU view-dependent front). The A/B and
            // fallback escape hatch against the B5c-4 default-on. Last flag wins.
            args.debug.vdpmGpuBackend = false;
            continue;
        }
        if (arg == "--lod-mode")
        {
            // Select the initial LOD mode. `view-dependent` launches straight into VDPM (indirect
            // draws) so the path is reachable without the overlay — handy for the render smoke
            // test. Consume the following token as the value ONLY when it is present and not itself
            // a flag: a trailing `--lod-mode` or one followed by another flag (`--lod-mode
            // --overlay`) must NOT swallow that flag or fall through to become a positional — it
            // just defaults. An unrecognised value defaults to Discrete but is still consumed (not
            // left positional).
            if (i + 1 < argc)
            {
                const std::string_view value = argv[i + 1];
                if (!value.empty() && value.front() != '-')
                {
                    ++i;
                    if (value == "view-dependent")
                    {
                        args.debug.lodMode = LodMode::ViewDependent;
                    }
                    else if (value == "continuous")
                    {
                        args.debug.lodMode = LodMode::Continuous;
                    }
                    else
                    {
                        args.debug.lodMode = LodMode::Discrete;
                    }
                }
            }
            continue;
        }
        if (positionalCount < 2)
        {
            positional[positionalCount++] = arg;
        }
    }

    if (positionalCount >= 1)
    {
        if (isEnvironmentPath(positional[0]))
        {
            args.skyboxPath = positional[0];
        }
        else
        {
            args.scenePath = positional[0];
        }
    }
    if (positionalCount >= 2)
    {
        args.skyboxPath = positional[1];
    }

    return args;
}

} // namespace fire_engine
