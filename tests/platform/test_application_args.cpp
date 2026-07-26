#include <fire_engine/platform/application_args.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

using fire_engine::ApplicationArgs;
using fire_engine::DebugView;
using fire_engine::LodMode;
using fire_engine::parseApplicationArgs;

namespace
{

struct ParsedArgs
{
    std::vector<std::string> storage;
    std::vector<char*> argv;
    ApplicationArgs args;
};

ParsedArgs parseArgs(std::initializer_list<std::string_view> args)
{
    ParsedArgs parsed;
    parsed.storage.reserve(args.size());
    parsed.argv.reserve(args.size());

    for (std::string_view arg : args)
    {
        parsed.storage.emplace_back(arg);
    }

    for (std::string& arg : parsed.storage)
    {
        parsed.argv.push_back(arg.data());
    }

    parsed.args = parseApplicationArgs(static_cast<int>(parsed.argv.size()), parsed.argv.data());
    return parsed;
}

} // namespace

TEST_CASE("ApplicationArgs.EmptyArgsUseDefaults", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp"});
    const auto& args = parsed.args;

    CHECK(args.scenePath.empty());
    CHECK(args.skyboxPath.empty());
    CHECK(args.debug.view == DebugView::None);
    CHECK_FALSE(args.debug.noShadows);
    CHECK(args.debug.taa);
    CHECK_FALSE(args.debug.overlayVisible);
    CHECK_FALSE(args.addFloor);
    CHECK_FALSE(args.addParticles);
    CHECK_FALSE(args.addCloth);
    CHECK_FALSE(args.startMaximized);
    CHECK(args.debug.lodMode == LodMode::Discrete); // default matches RenderTunables
    // The GPU-VDPM backend request is a tri-state; unset by default so the Renderer resolves it
    // against device capability (B5c-4 default flip).
    CHECK_FALSE(args.debug.vdpmGpuBackend.has_value());
}

TEST_CASE("ApplicationArgs.VdpmGpuBackendTriStateFlags", "[ApplicationArgs]")
{
    // Unset by default (Renderer resolves to on-if-capable).
    CHECK_FALSE(parseArgs({"fireEngineApp"}).args.debug.vdpmGpuBackend.has_value());

    // --vdpm-gpu forces on, --no-vdpm-gpu forces off.
    {
        const auto b = parseArgs({"fireEngineApp", "--vdpm-gpu"}).args.debug.vdpmGpuBackend;
        REQUIRE(b.has_value());
        CHECK(*b == true);
    }
    {
        const auto b = parseArgs({"fireEngineApp", "--no-vdpm-gpu"}).args.debug.vdpmGpuBackend;
        REQUIRE(b.has_value());
        CHECK(*b == false);
    }

    // Repeated / conflicting flags are last-one-wins.
    {
        const auto b =
            parseArgs({"fireEngineApp", "--vdpm-gpu", "--no-vdpm-gpu"}).args.debug.vdpmGpuBackend;
        REQUIRE(b.has_value());
        CHECK(*b == false);
    }
    {
        const auto b =
            parseArgs({"fireEngineApp", "--no-vdpm-gpu", "--vdpm-gpu"}).args.debug.vdpmGpuBackend;
        REQUIRE(b.has_value());
        CHECK(*b == true);
    }
}

TEST_CASE("ApplicationArgs.RequireValidationFlag", "[ApplicationArgs]")
{
    // Off by default: a machine without the Vulkan SDK must still be able to run the app.
    CHECK_FALSE(parseArgs({"fireEngineApp"}).args.debug.requireValidation);
    // On demand it becomes a startup precondition (Device::createInstance throws) so an automated
    // run cannot report a clean frame that nothing validated.
    CHECK(parseArgs({"fireEngineApp", "--require-validation"}).args.debug.requireValidation);
    // Composes with the positional scene path rather than swallowing it.
    const auto parsed = parseArgs({"fireEngineApp", "--require-validation", "scene.gltf"});
    CHECK(parsed.args.debug.requireValidation);
    CHECK(parsed.args.scenePath == "scene.gltf");
}

TEST_CASE("ApplicationArgs.LodModeFlagSelectsMode", "[ApplicationArgs]")
{
    struct LodModeCase
    {
        std::string_view value;
        LodMode mode;
    };
    const auto testCase = GENERATE(values<LodModeCase>({
        {"discrete", LodMode::Discrete},
        {"continuous", LodMode::Continuous},
        {"view-dependent", LodMode::ViewDependent},
        {"nonsense", LodMode::Discrete}, // unknown value falls back to Discrete
    }));
    CAPTURE(testCase.value);

    const auto parsed = parseArgs({"fireEngineApp", "--lod-mode", testCase.value});
    CHECK(parsed.args.debug.lodMode == testCase.mode);
}

TEST_CASE("ApplicationArgs.LodModeTrailingLeavesPositionalsEmpty", "[ApplicationArgs]")
{
    // A trailing --lod-mode must default AND not read past argv or become a scene path.
    const auto parsed = parseArgs({"fireEngineApp", "--lod-mode"});
    CHECK(parsed.args.debug.lodMode == LodMode::Discrete);
    CHECK(parsed.args.scenePath.empty());
    CHECK(parsed.args.skyboxPath.empty());
}

TEST_CASE("ApplicationArgs.LodModeDoesNotSwallowAFollowingFlag", "[ApplicationArgs]")
{
    // --lod-mode --overlay must default the mode and STILL process --overlay (a flag is never a
    // mode value).
    const auto parsed = parseArgs({"fireEngineApp", "--lod-mode", "--overlay"});
    CHECK(parsed.args.debug.lodMode == LodMode::Discrete);
    CHECK(parsed.args.debug.overlayVisible);
    CHECK(parsed.args.scenePath.empty());
}

TEST_CASE("ApplicationArgs.LodModeInvalidValueIsConsumedNotPositional", "[ApplicationArgs]")
{
    // An unrecognised (non-flag) value defaults to Discrete but is consumed, so it never leaks into
    // the scene/skybox positionals; a real scene path after it still lands.
    const auto parsed =
        parseArgs({"fireEngineApp", "--lod-mode", "nonsense", "DamagedHelmet/DamagedHelmet.gltf"});
    CHECK(parsed.args.debug.lodMode == LodMode::Discrete);
    CHECK(parsed.args.scenePath == "DamagedHelmet/DamagedHelmet.gltf");
}

TEST_CASE("ApplicationArgs.SingleSceneArgumentSetsScenePath", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "Fox/Fox.gltf"});
    const auto& args = parsed.args;

    CHECK(args.scenePath == "Fox/Fox.gltf");
    CHECK(args.skyboxPath.empty());
    CHECK(args.debug.view == DebugView::None);
    CHECK_FALSE(args.debug.noShadows);
}

TEST_CASE("ApplicationArgs.SingleHdrArgumentSetsSkyboxPath", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "nightbox.hdr"});
    const auto& args = parsed.args;

    CHECK(args.scenePath.empty());
    CHECK(args.skyboxPath == "nightbox.hdr");
    CHECK(args.debug.view == DebugView::None);
    CHECK_FALSE(args.debug.noShadows);
}

TEST_CASE("ApplicationArgs.SingleExrArgumentSetsSkyboxPath", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "studio.EXR"});
    const auto& args = parsed.args;

    CHECK(args.scenePath.empty());
    CHECK(args.skyboxPath == "studio.EXR");
    CHECK(args.debug.view == DebugView::None);
    CHECK_FALSE(args.debug.noShadows);
}

TEST_CASE("ApplicationArgs.TwoArgumentsKeepSceneThenSkyboxOrder", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "Fox/Fox.gltf", "nightbox.hdr"});
    const auto& args = parsed.args;

    CHECK(args.scenePath == "Fox/Fox.gltf");
    CHECK(args.skyboxPath == "nightbox.hdr");
    CHECK(args.debug.view == DebugView::None);
    CHECK_FALSE(args.debug.noShadows);
}

struct DebugFlagCase
{
    std::string_view flag;
    DebugView view;
};

TEST_CASE("ApplicationArgs.DebugFlagsSetView", "[ApplicationArgs]")
{
    const auto testCase = GENERATE(values<DebugFlagCase>({
        {"--debug-normals", DebugView::Normals},
        {"--debug-ndotl", DebugView::NdotL},
        {"--debug-shadow", DebugView::Shadow},
        {"--debug-shadow-depth", DebugView::ShadowDepth},
        {"--debug-velocity", DebugView::Velocity},
    }));
    CAPTURE(testCase.flag);

    const auto parsed = parseArgs({"fireEngineApp", testCase.flag});
    const auto& args = parsed.args;

    CHECK(args.debug.view == testCase.view);
    CHECK(args.scenePath.empty());
    CHECK(args.skyboxPath.empty());
}

TEST_CASE("ApplicationArgs.ParticlesFlagSetsToggle", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "-p", "RiggedSimple/RiggedSimple.gltf"});
    const auto& args = parsed.args;

    CHECK(args.addParticles);
    CHECK_FALSE(args.addFloor);
    CHECK(args.scenePath == "RiggedSimple/RiggedSimple.gltf");
}

TEST_CASE("ApplicationArgs.ClothFlagSetsToggle", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "-c", "RiggedSimple/RiggedSimple.gltf"});
    const auto& args = parsed.args;

    CHECK(args.addCloth);
    CHECK_FALSE(args.addParticles);
    CHECK_FALSE(args.addFloor);
    CHECK(args.scenePath == "RiggedSimple/RiggedSimple.gltf");
}

TEST_CASE("ApplicationArgs.DebugNormalsFlagCoexistsWithSceneAndFloor", "[ApplicationArgs]")
{
    const auto parsed =
        parseArgs({"fireEngineApp", "--debug-normals", "-f", "RiggedSimple/RiggedSimple.gltf"});
    const auto& args = parsed.args;

    CHECK(args.debug.view == DebugView::Normals);
    CHECK(args.addFloor);
    CHECK(args.scenePath == "RiggedSimple/RiggedSimple.gltf");
}

TEST_CASE("ApplicationArgs.LastDebugFlagWins", "[ApplicationArgs]")
{
    const auto parsed =
        parseArgs({"fireEngineApp", "--debug-normals", "--debug-shadow", "--debug-velocity"});
    const auto& args = parsed.args;

    CHECK(args.debug.view == DebugView::Velocity);
}

TEST_CASE("ApplicationArgs.NoShadowsFlagSetsToggle", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--no-shadows"});
    const auto& args = parsed.args;

    CHECK(args.debug.noShadows);
    CHECK(args.debug.view == DebugView::None);
}

TEST_CASE("ApplicationArgs.NoTaaFlagDisablesTemporalAntialiasing", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--no-taa"});
    const auto& args = parsed.args;

    CHECK_FALSE(args.debug.taa);
}

TEST_CASE("ApplicationArgs.OverlayFlagShowsDebugOverlay", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--overlay"});
    const auto& args = parsed.args;

    CHECK(args.debug.overlayVisible);
}

TEST_CASE("ApplicationArgs.MaximizedFlagSetsStartupWindowState", "[ApplicationArgs]")
{
    const auto flag = GENERATE(values<std::string_view>({"--maximized", "--maximised"}));
    CAPTURE(flag);

    const auto parsed = parseArgs({"fireEngineApp", flag, "RiggedSimple/RiggedSimple.gltf"});
    const auto& args = parsed.args;

    CHECK(args.startMaximized);
    CHECK(args.scenePath == "RiggedSimple/RiggedSimple.gltf");
}

TEST_CASE("ApplicationArgs.UnknownFlagIsTreatedAsPositional", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--unknown", "nightbox.hdr"});
    const auto& args = parsed.args;

    CHECK(args.scenePath == "--unknown");
    CHECK(args.skyboxPath == "nightbox.hdr");
}

// ---------------------------------------------------------------------------
// --no-lod / --capture / --capture-frame
//
// A capture command is meant to be scriptable and reproducible, so its parsing has to be
// boring in the specific ways that bite: a value-taking flag must never swallow the NEXT
// flag, a missing value must not turn into a scene path, and a bad frame number must not
// silently become frame 0 (which would capture before anything has settled).
// ---------------------------------------------------------------------------

TEST_CASE("ApplicationArgs.NoLodDisablesLodAtStartup", "[ApplicationArgs]")
{
    CHECK(parseArgs({"fireEngineApp"}).args.debug.lod);
    CHECK_FALSE(parseArgs({"fireEngineApp", "--no-lod"}).args.debug.lod);
}

TEST_CASE("ApplicationArgs.CaptureTakesAPathAndDefaultsTheFrame", "[ApplicationArgs]")
{
    const auto parsed =
        parseArgs({"fireEngineApp", "--capture", "out.png", "DamagedHelmet/DamagedHelmet.gltf"});
    const auto& args = parsed.args;

    CHECK(args.debug.capturePath == "out.png");
    CHECK(args.debug.captureFrame == 16);
    // The path was consumed as the flag's value, NOT left to be read as the scene.
    CHECK(args.scenePath == "DamagedHelmet/DamagedHelmet.gltf");
}

TEST_CASE("ApplicationArgs.CaptureWithoutAPathCapturesNothing", "[ApplicationArgs]")
{
    // Trailing flag: must not read past argv, and must not leave capture "requested" with an
    // empty path (which would add swapchain transfer usage for a capture that never happens).
    CHECK(parseArgs({"fireEngineApp", "--capture"}).args.debug.capturePath.empty());

    // Followed by another flag: that flag must still be processed, not eaten as the path.
    const auto parsed = parseArgs({"fireEngineApp", "--capture", "--overlay"});
    CHECK(parsed.args.debug.capturePath.empty());
    CHECK(parsed.args.debug.overlayVisible);
}

TEST_CASE("ApplicationArgs.CaptureFrameParsesAndValidates", "[ApplicationArgs]")
{
    CHECK(parseArgs({"fireEngineApp", "--capture-frame", "42"}).args.debug.captureFrame == 42);

    // Every rejected form keeps the default rather than capturing frame 0 or a negative frame.
    for (std::string_view bad : {"0", "-3", "abc", "12x", ""})
    {
        CAPTURE(bad);
        CHECK(parseArgs({"fireEngineApp", "--capture-frame", bad}).args.debug.captureFrame == 16);
    }
    CHECK(parseArgs({"fireEngineApp", "--capture-frame"}).args.debug.captureFrame == 16);
}

TEST_CASE("ApplicationArgs.CaptureFrameDoesNotSwallowAFollowingFlag", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--capture-frame", "--no-lod"});

    CHECK(parsed.args.debug.captureFrame == 16);
    CHECK_FALSE(parsed.args.debug.lod);
}

TEST_CASE("ApplicationArgs.RepeatedCaptureFlagsTakeTheLastValue", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--capture", "first.png", "--capture-frame",
                                   "4", "--capture", "second.png", "--capture-frame", "9"});
    const auto& args = parsed.args;

    CHECK(args.debug.capturePath == "second.png");
    CHECK(args.debug.captureFrame == 9);
}
