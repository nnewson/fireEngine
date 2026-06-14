#include <fire_engine/platform/application_args.hpp>

#include <gtest/gtest.h>

using fire_engine::DebugView;
using fire_engine::parseApplicationArgs;

TEST(ApplicationArgs, EmptyArgsUseDefaults)
{
    char program[] = "fireEngineApp";
    char* argv[] = {program};

    const auto args = parseApplicationArgs(1, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_TRUE(args.skyboxPath.empty());
    EXPECT_EQ(args.debug.view, DebugView::None);
    EXPECT_FALSE(args.debug.noShadows);
    EXPECT_FALSE(args.addFloor);
    EXPECT_FALSE(args.addParticles);
}

TEST(ApplicationArgs, SingleSceneArgumentSetsScenePath)
{
    char program[] = "fireEngineApp";
    char scene[] = "Fox/Fox.gltf";
    char* argv[] = {program, scene};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.scenePath, "Fox/Fox.gltf");
    EXPECT_TRUE(args.skyboxPath.empty());
    EXPECT_EQ(args.debug.view, DebugView::None);
    EXPECT_FALSE(args.debug.noShadows);
}

TEST(ApplicationArgs, SingleHdrArgumentSetsSkyboxPath)
{
    char program[] = "fireEngineApp";
    char skybox[] = "nightbox.hdr";
    char* argv[] = {program, skybox};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_EQ(args.skyboxPath, "nightbox.hdr");
    EXPECT_EQ(args.debug.view, DebugView::None);
    EXPECT_FALSE(args.debug.noShadows);
}

TEST(ApplicationArgs, SingleExrArgumentSetsSkyboxPath)
{
    char program[] = "fireEngineApp";
    char skybox[] = "studio.EXR";
    char* argv[] = {program, skybox};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_EQ(args.skyboxPath, "studio.EXR");
    EXPECT_EQ(args.debug.view, DebugView::None);
    EXPECT_FALSE(args.debug.noShadows);
}

TEST(ApplicationArgs, TwoArgumentsKeepSceneThenSkyboxOrder)
{
    char program[] = "fireEngineApp";
    char scene[] = "Fox/Fox.gltf";
    char skybox[] = "nightbox.hdr";
    char* argv[] = {program, scene, skybox};

    const auto args = parseApplicationArgs(3, argv);

    EXPECT_EQ(args.scenePath, "Fox/Fox.gltf");
    EXPECT_EQ(args.skyboxPath, "nightbox.hdr");
    EXPECT_EQ(args.debug.view, DebugView::None);
    EXPECT_FALSE(args.debug.noShadows);
}

TEST(ApplicationArgs, DebugNormalsFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-normals";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.debug.view, DebugView::Normals);
    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_TRUE(args.skyboxPath.empty());
}

TEST(ApplicationArgs, ParticlesFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char particles[] = "-p";
    char scene[] = "RiggedSimple/RiggedSimple.gltf";
    char* argv[] = {program, particles, scene};

    const auto args = parseApplicationArgs(3, argv);

    EXPECT_TRUE(args.addParticles);
    EXPECT_FALSE(args.addFloor);
    EXPECT_EQ(args.scenePath, "RiggedSimple/RiggedSimple.gltf");
}

TEST(ApplicationArgs, DebugNormalsFlagCoexistsWithSceneAndFloor)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-normals";
    char floor[] = "-f";
    char scene[] = "RiggedSimple/RiggedSimple.gltf";
    char* argv[] = {program, flag, floor, scene};

    const auto args = parseApplicationArgs(4, argv);

    EXPECT_EQ(args.debug.view, DebugView::Normals);
    EXPECT_TRUE(args.addFloor);
    EXPECT_EQ(args.scenePath, "RiggedSimple/RiggedSimple.gltf");
}

TEST(ApplicationArgs, DebugNdotLFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-ndotl";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.debug.view, DebugView::NdotL);
}

TEST(ApplicationArgs, DebugShadowFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-shadow";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.debug.view, DebugView::Shadow);
}

TEST(ApplicationArgs, DebugShadowDepthFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-shadow-depth";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.debug.view, DebugView::ShadowDepth);
}

TEST(ApplicationArgs, NoShadowsFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--no-shadows";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.debug.noShadows);
    EXPECT_EQ(args.debug.view, DebugView::None);
}
