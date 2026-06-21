cmake_minimum_required(VERSION 3.20)

project(fire_engine
    VERSION 0.1.0
    DESCRIPTION "Vulkan demo in C++"
    LANGUAGES CXX
)

# --- Dependencies -------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(Ktx CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

include(cmake/fireengine_imgui.cmake)

# --- Library ------------------------------------------------------------

add_library(fireengine SHARED
    src/animation/animation.cpp
    src/collision/collider.cpp
    src/collision/geometry.cpp
    src/collision/gjk_epa.cpp
    src/collision/sweep_and_prune_broad_phase.cpp
    src/collision/narrow_phase.cpp
    src/core/convex_hull_builder.cpp
    src/core/gltf_loader.cpp
    src/core/gltf_loader_assets.cpp
    src/core/gltf_loader_mesh.cpp
    src/core/gltf_loader_animation.cpp
    src/core/shader_loader.cpp
    src/core/system.cpp
    src/core/tangent_generator.cpp
    src/fire_engine.cpp
    src/graphics/cloth.cpp
    src/graphics/geometry.cpp
    src/graphics/image.cpp
    src/graphics/ktx_image.cpp
    src/graphics/material_binding.cpp
    src/graphics/object.cpp
    src/graphics/skin.cpp
    src/graphics/texture.cpp
    src/input/input.cpp
    src/platform/keyboard.cpp
    src/platform/mouse.cpp
    src/platform/window.cpp
    src/physics/contact_solver.cpp
    src/physics/joint_solver.cpp
    src/physics/physics_body.cpp
    src/physics/physics_world.cpp
    src/render/compute_pipeline.cpp
    src/render/debug_draw.cpp
    src/render/debug_overlay.cpp
    src/render/device.cpp
    src/render/descriptors.cpp
    src/render/gpu_profiler.cpp
    src/render/environment_precompute.cpp
    src/render/frame.cpp
    src/render/particle_system.cpp
    src/render/pipeline.cpp
    src/render/post_processing.cpp
    src/render/render_context.cpp
    src/render/render_target.cpp
    src/render/renderer.cpp
    src/render/resources.cpp
    src/render/shadows.cpp
    src/render/soft_body_system.cpp
    src/render/ssao.cpp
    src/render/swapchain.cpp
    src/render/taa.cpp
    src/render/transmission.cpp
    src/scene/animator.cpp
    src/scene/controllable.cpp
    src/scene/camera.cpp
    src/scene/empty.cpp
    src/scene/light.cpp
    src/scene/mesh.cpp
    src/scene/node.cpp
    src/scene/particle_emitter.cpp
    src/scene/scene_graph.cpp
    src/scene/transform.cpp
)

target_include_directories(fireengine PUBLIC ${PROJECT_SOURCE_DIR}/include ${Stb_INCLUDE_DIR})
target_compile_features(fireengine PUBLIC cxx_std_23)
target_compile_options(fireengine PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(fireengine PRIVATE
    VK_ENABLE_BETA_EXTENSIONS
)
target_compile_definitions(fireengine PUBLIC
    VULKAN_HPP_NO_CONSTRUCTORS
)
target_compile_definitions(fireengine PRIVATE
    FIRE_ENGINE_SOURCE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
    FIRE_ENGINE_BUILD_ASSET_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(fireengine
    PUBLIC
        Vulkan::Vulkan
        glfw
        fastgltf::fastgltf
        KTX::ktx
    PRIVATE
        fireengine_imgui
)

# --- Executable ---------------------------------------------------------

add_executable(fireEngineApp
    src/platform/application.cpp
)

target_link_libraries(fireEngineApp PRIVATE fireengine Vulkan::Vulkan glfw)
target_compile_options(fireEngineApp PRIVATE -Wall -Wextra -Wpedantic)

# --- Shaders ------------------------------------------------------------

set(SHADER_SOURCES
    shaders/shader.vert
    shaders/shader.frag
    shaders/skybox.vert
    shaders/skybox.frag
    shaders/environment_convert.frag
    shaders/irradiance_convolution.frag
    shaders/prefilter_environment.frag
    shaders/brdf_integration.frag
    shaders/shadow.vert
    shaders/shadow.frag
    shaders/self_shadow_second.frag
    shaders/depth_prepass.frag
    shaders/ssao.frag
    shaders/ssao_blur.frag
    shaders/debug_line.vert
    shaders/debug_line.frag
    shaders/postprocess.vert
    shaders/postprocess.frag
    shaders/taa.frag
    shaders/bloom_downsample.frag
    shaders/bloom_upsample.frag
    shaders/particle_simulate.comp
    shaders/particle.vert
    shaders/particle.frag
    shaders/cloth_predict.comp
    shaders/cloth_solve.comp
    shaders/cloth_collide.comp
    shaders/cloth_finalize.comp
)

foreach(shader IN LISTS SHADER_SOURCES)
  get_filename_component(SHADER_NAME ${shader} NAME)
  set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)

  add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND glslc ${PROJECT_SOURCE_DIR}/${shader} -o ${SHADER_SPV}
    DEPENDS ${PROJECT_SOURCE_DIR}/${shader}
    COMMENT "Compiling ${SHADER_NAME} -> ${SHADER_NAME}.spv"
    VERBATIM
  )

  list(APPEND SHADER_SPVS ${SHADER_SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_SPVS} COMMENT "Building shaders")

# --- Assets -------------------------------------------------------------

add_custom_target(assets ALL
  COMMAND ${CMAKE_COMMAND}
    -DASSET_DIR=${PROJECT_SOURCE_DIR}/assets
    -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/copy_assets.cmake
  COMMENT "Copying assets to build directory"
  VERBATIM
)

# --- Tests --------------------------------------------------------------

enable_testing()

add_executable(test_fire_engine
    tests/collision/test_aabb.cpp
    tests/collision/test_geometry.cpp
    tests/collision/test_sweep_and_prune_broad_phase.cpp
    tests/collision/test_narrow_phase.cpp
    tests/collision/test_support.cpp
    tests/collision/test_gjk_epa.cpp
    tests/physics/test_physics_world.cpp
    tests/physics/test_contact_solver.cpp
    tests/physics/test_joint.cpp
    tests/physics/test_physics_handle.cpp
    tests/physics/test_physics_determinism.cpp
    tests/math/test_mat3.cpp
    tests/math/test_mat4.cpp
    tests/math/test_quaternion.cpp
    tests/math/test_vec2.cpp
    tests/math/test_vec3.cpp
    tests/math/test_vec4.cpp
    tests/graphics/test_cloth.cpp
    tests/graphics/test_colour3.cpp
    tests/graphics/test_joints4.cpp
    tests/graphics/test_image.cpp
    tests/graphics/test_ktx_image.cpp
    tests/graphics/test_vertex.cpp
    tests/graphics/test_skin.cpp
    tests/graphics/test_assets.cpp
    tests/graphics/test_material.cpp
    tests/graphics/test_geometry.cpp
    tests/graphics/test_gpu_handle.cpp
    tests/graphics/test_bounds.cpp
    tests/graphics/test_draw_command.cpp
    tests/graphics/test_frame_info.cpp
    tests/graphics/test_texture.cpp
    tests/graphics/test_sampler_settings.cpp
    tests/render/test_ubo.cpp
    tests/render/test_pipeline_config.cpp
    tests/scene/test_particle_emitter.cpp
    tests/core/test_shader_loader.cpp
    tests/core/test_system.cpp
    tests/core/test_gltf_loader.cpp
    tests/core/test_convex_hull_builder.cpp
    tests/core/test_tangent_generator.cpp
    tests/input/test_camera_state.cpp
    tests/input/test_controller_state.cpp
    tests/input/test_animation_state.cpp
    tests/input/test_variant_state.cpp
    tests/input/test_input_state.cpp
    tests/platform/test_application_args.cpp
    tests/scene/test_animator.cpp
    tests/scene/test_controllable.cpp
    tests/scene/test_camera.cpp
    tests/scene/test_light.cpp
    tests/scene/test_node.cpp
    tests/scene/test_scene_graph.cpp
    tests/scene/test_formatter.cpp
    tests/animation/test_animation.cpp
    tests/scene/test_transform.cpp
)
target_include_directories(test_fire_engine PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_fire_engine PRIVATE fireengine Catch2::Catch2WithMain)
target_compile_options(test_fire_engine PRIVATE -Wall -Wextra -Wpedantic)

file(GLOB_RECURSE TEST_ASSETS "${PROJECT_SOURCE_DIR}/tests/assets/*")
foreach(f IN LISTS TEST_ASSETS)
  get_filename_component(name "${f}" NAME)
  configure_file("${f}" "${CMAKE_CURRENT_BINARY_DIR}/test_assets/${name}" COPYONLY)
endforeach()

include(Catch)
catch_discover_tests(test_fire_engine)

# Layering guard: the Vulkan-free graphics layer must not include render/ from
# its public headers. Runs as a CTest case so `ctest` catches regressions
# without a GPU.
add_test(
  NAME graphics_layer_no_render_includes
  COMMAND ${CMAKE_COMMAND}
    -DGRAPHICS_INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include/fire_engine/graphics
    -P ${PROJECT_SOURCE_DIR}/cmake/check_graphics_layering.cmake
)
