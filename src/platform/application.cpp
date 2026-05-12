// Vulkan rotating cube — matches shader.vert / shader.frag in this repo.
// Dependencies: Vulkan SDK, GLFW 3.3+, glslc (for SPIR-V compilation via CMake).

#include <fire_engine/fire_engine.hpp>
#include <fire_engine/platform/application_args.hpp>

#include <cstdlib>
#include <iostream>
#include <string_view>

using namespace fire_engine;

int main(int argc, char* argv[])
{
    const auto args = parseApplicationArgs(argc, argv);
    try
    {
        FireEngine app;
        app.run(800, 600, "FireEngine Demo", args.scenePath, args.skyboxPath, args.addFloor,
                args.debug);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
