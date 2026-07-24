#pragma once

#include <algorithm>
#include <cctype>
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
