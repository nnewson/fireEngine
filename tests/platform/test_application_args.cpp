#include <fire_engine/platform/application_args.hpp>

#include <gtest/gtest.h>

using fire_engine::parseApplicationArgs;

TEST(ApplicationArgs, EmptyArgsUseDefaults)
{
    char program[] = "fireEngineApp";
    char* argv[] = {program};

    const auto args = parseApplicationArgs(1, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_TRUE(args.skyboxPath.empty());
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_FALSE(args.noShadows);
}

TEST(ApplicationArgs, SingleSceneArgumentSetsScenePath)
{
    char program[] = "fireEngineApp";
    char scene[] = "Fox/Fox.gltf";
    char* argv[] = {program, scene};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_EQ(args.scenePath, "Fox/Fox.gltf");
    EXPECT_TRUE(args.skyboxPath.empty());
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_FALSE(args.noShadows);
}

TEST(ApplicationArgs, SingleHdrArgumentSetsSkyboxPath)
{
    char program[] = "fireEngineApp";
    char skybox[] = "nightbox.hdr";
    char* argv[] = {program, skybox};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_EQ(args.skyboxPath, "nightbox.hdr");
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_FALSE(args.noShadows);
}

TEST(ApplicationArgs, SingleExrArgumentSetsSkyboxPath)
{
    char program[] = "fireEngineApp";
    char skybox[] = "studio.EXR";
    char* argv[] = {program, skybox};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_EQ(args.skyboxPath, "studio.EXR");
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_FALSE(args.noShadows);
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
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_FALSE(args.noShadows);
}

TEST(ApplicationArgs, DebugNormalsFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-normals";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
    EXPECT_TRUE(args.scenePath.empty());
    EXPECT_TRUE(args.skyboxPath.empty());
}

TEST(ApplicationArgs, DebugNormalsFlagCoexistsWithSceneAndFloor)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-normals";
    char floor[] = "-f";
    char scene[] = "RiggedSimple/RiggedSimple.gltf";
    char* argv[] = {program, flag, floor, scene};

    const auto args = parseApplicationArgs(4, argv);

    EXPECT_TRUE(args.debugNormals);
    EXPECT_TRUE(args.addFloor);
    EXPECT_EQ(args.scenePath, "RiggedSimple/RiggedSimple.gltf");
}

TEST(ApplicationArgs, DebugNdotLFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-ndotl";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.debugNdotL);
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
}

TEST(ApplicationArgs, DebugShadowFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-shadow";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.debugShadow);
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadowDepth);
}

TEST(ApplicationArgs, DebugShadowDepthFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--debug-shadow-depth";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.debugShadowDepth);
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
}

TEST(ApplicationArgs, NoShadowsFlagSetsToggle)
{
    char program[] = "fireEngineApp";
    char flag[] = "--no-shadows";
    char* argv[] = {program, flag};

    const auto args = parseApplicationArgs(2, argv);

    EXPECT_TRUE(args.noShadows);
    EXPECT_FALSE(args.debugNormals);
    EXPECT_FALSE(args.debugNdotL);
    EXPECT_FALSE(args.debugShadow);
    EXPECT_FALSE(args.debugShadowDepth);
}
