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

TEST_CASE("ApplicationArgs.UnknownFlagIsTreatedAsPositional", "[ApplicationArgs]")
{
    const auto parsed = parseArgs({"fireEngineApp", "--unknown", "nightbox.hdr"});
    const auto& args = parsed.args;

    CHECK(args.scenePath == "--unknown");
    CHECK(args.skyboxPath == "nightbox.hdr");
}
