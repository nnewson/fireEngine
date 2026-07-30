cmake_minimum_required(VERSION 3.20)

project(fire_engine
    VERSION 0.1.0
    DESCRIPTION "Vulkan demo in C++"
    LANGUAGES CXX
)

# Dev is the default single-config build: optimized enough for physics settle tests, with debug
# info, asserts, and Vulkan validation still enabled (no NDEBUG).
set(CMAKE_CXX_FLAGS_DEV "-O2 -g" CACHE STRING "Flags used by Dev builds" FORCE)
set(CMAKE_C_FLAGS_DEV "-O2 -g" CACHE STRING "Flags used by Dev builds" FORCE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Dev CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Dev Debug Release RelWithDebInfo MinSizeRel)

option(FIRE_ENGINE_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors" ON)

# --- Dependencies -------------------------------------------------------

# The vcpkg include tree must beat /usr/local/include, and `-isystem` cannot do it.
#
# Clang places /usr/local/include AHEAD of every -isystem path, which is where CMake puts an
# imported target's includes. Install a Vulkan SDK and that directory holds a second, newer
# vulkan-hpp: our own translation units then compile against the SDK's headers while anything
# resolved RELATIVE to a vcpkg header (vulkan_raii.hpp including its sibling vulkan.hpp) still gets
# vcpkg's. Two vulkan-hpp versions in one binary, and the first RAII call aborts on
# `m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION` — a mismatch with no bad line of C++
# behind it. (Observed for real when SDK 1.4.357 landed beside our pinned 1.4.335.)
#
# A plain -I comes before ALL system directories, so this pins every dependency to the version the
# manifest names. Deliberately not SYSTEM: that would emit -isystem again and lose the race.
#
# Both spellings are checked because vcpkg has used both, and the shaderc lookup below already
# hedges the same way — a silent miss here would restore the mixed-header build with no signal.
set(_fireengine_vcpkg_root "")
if(DEFINED VCPKG_INSTALLED_DIR)
    set(_fireengine_vcpkg_root "${VCPKG_INSTALLED_DIR}")
elseif(DEFINED _VCPKG_INSTALLED_DIR)
    set(_fireengine_vcpkg_root "${_VCPKG_INSTALLED_DIR}")
endif()
if(_fireengine_vcpkg_root AND DEFINED VCPKG_TARGET_TRIPLET)
    set(_fireengine_vcpkg_include "${_fireengine_vcpkg_root}/${VCPKG_TARGET_TRIPLET}/include")
    if(NOT EXISTS "${_fireengine_vcpkg_include}")
        message(FATAL_ERROR
            "vcpkg include dir not found at ${_fireengine_vcpkg_include}. It must precede "
            "/usr/local/include or a system Vulkan SDK silently supplies a second vulkan-hpp.")
    endif()
    # include_directories(BEFORE) alone is not enough: CMake de-duplicates the path against the
    # imported targets' interface includes and keeps the -isystem classification, which loses the
    # race again. NO_SYSTEM_FROM_IMPORTED is the switch that actually demotes them to -I.
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
    include_directories(BEFORE "${_fireengine_vcpkg_include}")
    message(STATUS "Pinned dependency headers first: ${_fireengine_vcpkg_include}")
endif()

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
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
    src/collision/ray.cpp
    src/collision/shape_cast.cpp
    src/collision/sweep_and_prune_broad_phase.cpp
    src/collision/dynamic_aabb_tree_broad_phase.cpp
    src/collision/narrow_phase.cpp
    src/core/convex_hull_builder.cpp
    src/core/node_id.cpp
    src/core/gltf_loader_assets.cpp
    src/core/gltf_loader_extras.cpp
    src/core/gltf_loader_geometry.cpp
    src/core/gltf_loader_images.cpp
    src/core/gltf_loader_material.cpp
    src/core/gltf_loader_mesh.cpp
    src/core/gltf_loader_nodes.cpp
    src/core/gltf_loader_scene.cpp
    src/core/gltf_loader_animation.cpp
    src/core/shader_loader.cpp
    src/core/system.cpp
    src/core/tangent_generator.cpp
    src/fire_engine.cpp
    src/graphics/cloth.cpp
    src/graphics/frame_capture.cpp
    src/graphics/geometry.cpp
    src/graphics/mesh_simplifier.cpp
    src/graphics/mesh_topology.cpp
    src/graphics/shadow_identity.cpp
    src/graphics/shadow_lod_resolver.cpp
    src/graphics/shadow_render_view.cpp
    src/graphics/shadow_diagnostics.cpp
    src/graphics/shadow_view.cpp
    src/graphics/vipm.cpp
    src/graphics/vdpm.cpp
    src/graphics/vdpm_material.cpp
    src/graphics/vdpm_parallel.cpp
    src/graphics/vdpm_gpu_registry.cpp
    src/graphics/vdpm_wedge_choices.cpp
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
    src/physics/articulation.cpp
    src/physics/articulation_contact.cpp
    src/physics/character_controller.cpp
    src/physics/contact_solver.cpp
    src/physics/island.cpp
    src/physics/joint_solver.cpp
    src/physics/physics_body.cpp
    src/physics/physics_world.cpp
    src/physics/physics_world_articulation.cpp
    src/physics/physics_world_debug.cpp
    src/physics/physics_world_events.cpp
    src/physics/physics_world_queries.cpp
    src/physics/physics_world_shapes.cpp
    src/render/compute_pipeline.cpp
    src/render/debug_draw.cpp
    src/render/debug_overlay.cpp
    src/render/device.cpp
    src/render/device_plan.cpp
    src/render/descriptors.cpp
    src/graphics/generational_slot_pool.cpp
    src/render/vma_impl.cpp
    src/render/gpu_profiler.cpp
    src/render/environment_precompute.cpp
    src/render/frame.cpp
    src/render/particle_system.cpp
    src/render/pipeline.cpp
    src/render/post_processing.cpp
    src/render/render_target.cpp
    src/render/renderer.cpp
    src/render/resources.cpp
    src/render/shadows.cpp
    src/render/soft_body_system.cpp
    src/render/ssao.cpp
    src/render/swapchain.cpp
    src/render/taa.cpp
    src/render/transmission.cpp
    src/render/vdpm_gpu.cpp
    src/render/vdpm_gpu_manager.cpp
    src/render/vdpm_scan.cpp
    src/scene/animator.cpp
    src/scene/controllable.cpp
    src/scene/camera.cpp
    src/scene/empty.cpp
    src/scene/light.cpp
    src/scene/mesh.cpp
    src/scene/node.cpp
    src/scene/particle_emitter.cpp
    src/scene/ragdoll.cpp
    src/scene/scene_culler.cpp
    src/scene/scene_graph.cpp
    src/scene/transform.cpp
)

target_include_directories(fireengine PUBLIC ${PROJECT_SOURCE_DIR}/include ${Stb_INCLUDE_DIR})
target_compile_features(fireengine PUBLIC cxx_std_23)
target_compile_options(fireengine PRIVATE -Wall -Wextra -Wpedantic)
if(FIRE_ENGINE_WARNINGS_AS_ERRORS)
    target_compile_options(fireengine PRIVATE -Werror)
endif()
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
        GPUOpen::VulkanMemoryAllocator
        glfw
        fastgltf::fastgltf
        KTX::ktx
    PRIVATE
        fireengine_imgui
)

# VMA's amalgamated implementation TU is third-party C++ — exempt it from the engine's
# -Wall -Wextra -Wpedantic (-Werror) so our own code stays strict.
set_source_files_properties(src/render/vma_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# --- Executable ---------------------------------------------------------

add_executable(fireEngineApp
    src/platform/application.cpp
)

target_link_libraries(fireEngineApp PRIVATE fireengine Vulkan::Vulkan glfw)
target_compile_options(fireEngineApp PRIVATE -Wall -Wextra -Wpedantic)
if(FIRE_ENGINE_WARNINGS_AS_ERRORS)
    target_compile_options(fireEngineApp PRIVATE -Werror)
endif()

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
    shaders/vdpm_score.comp
    shaders/vdpm_scan_block.comp
    shaders/vdpm_scan_add.comp
    shaders/vdpm_ancestor.comp
    shaders/vdpm_survival.comp
    shaders/vdpm_scatter.comp
    shaders/vdpm_emit_finalize.comp
    shaders/vdpm_mark.comp
    shaders/vdpm_close.comp
    shaders/vdpm_refine.comp
    shaders/vdpm_coarsen.comp
    shaders/vdpm_repair_detect.comp
    shaders/vdpm_repair_fallback.comp
    shaders/vdpm_repair_kernel.comp
    shaders/vdpm_apply_kernel.comp
    shaders/vdpm_health_reduce.comp
)

# glslc from the vcpkg `shaderc` port (installed under <vcpkg-installed>/<triplet>/tools/shaderc via
# vcpkg_copy_tools). Sourcing it from vcpkg — not a system glslc / Vulkan SDK on PATH — means every
# platform compiles shaders with the same baseline-pinned compiler, so the emitted SPIR-V is
# reproducible and CI needs no system shader-compiler package.
find_program(GLSLC_EXECUTABLE
  NAMES glslc
  HINTS
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shaderc"
    "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shaderc"
  NO_DEFAULT_PATH
)
if(NOT GLSLC_EXECUTABLE)
  message(FATAL_ERROR "glslc not found in the vcpkg tools dir — is 'shaderc' in vcpkg.json and installed?")
endif()

# Shared GLSL includes (#include-d by shaders via the shaders/ dir on the include path). A change to
# any of them rebuilds every shader — includes are rare + shared, so this over-approximation is cheap
# and keeps the SPV outputs correct.
set(SHADER_INCLUDES
    ${PROJECT_SOURCE_DIR}/shaders/vdpm_repair_classify.glsl
    ${PROJECT_SOURCE_DIR}/shaders/vdpm_apply_classify.glsl
)

foreach(shader IN LISTS SHADER_SOURCES)
  get_filename_component(SHADER_NAME ${shader} NAME)
  set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)

  add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/shaders ${PROJECT_SOURCE_DIR}/${shader} -o ${SHADER_SPV}
    DEPENDS ${PROJECT_SOURCE_DIR}/${shader} ${SHADER_INCLUDES}
    COMMENT "Compiling ${SHADER_NAME} -> ${SHADER_NAME}.spv"
    VERBATIM
  )

  list(APPEND SHADER_SPVS ${SHADER_SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_SPVS} COMMENT "Building shaders")

# `shaders` is an ALL target but the runtime targets load the .spv from the build dir, so make the
# dependency EXPLICIT — otherwise `cmake --build --target test_fire_engine` (or fireEngineApp) can
# build against a stale shader (a manual `rm *.spv` must never be needed for correctness). The app
# target exists above; the test target adds its own dependency after it is defined below.
add_dependencies(fireEngineApp shaders)

# --- Assets -------------------------------------------------------------

# Regenerate the P8 physics demo glTF from their generator so editing the script
# can never leave a stale .gltf behind (the copy step below feeds off the result).
# The generator is deterministic, so this produces no spurious churn; skipped with a
# warning if Python is unavailable (the committed .gltf are then used as-is).
find_program(PYTHON3_EXECUTABLE NAMES python3 python)
if(PYTHON3_EXECUTABLE)
  add_custom_target(physics_demos ALL
    COMMAND ${PYTHON3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/assets/physics_demos/generate.py
    COMMENT "Generating physics demo glTF assets"
    VERBATIM
  )
  # The SH-01 shadow-LOD acceptance scenes (static baseline + motion variant). Its own
  # target, not a second command on the physics one: the two generators are independent,
  # and a failure should name the scene set that broke.
  add_custom_target(shadow_lod_demos ALL
    COMMAND ${PYTHON3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/assets/shadow_lod/generate.py
    COMMENT "Generating shadow-LOD acceptance glTF assets"
    VERBATIM
  )
else()
  message(WARNING "python3 not found — generated .gltf will not be regenerated")
endif()

add_custom_target(assets ALL
  COMMAND ${CMAKE_COMMAND}
    -DASSET_DIR=${PROJECT_SOURCE_DIR}/assets
    -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/copy_assets.cmake
  COMMENT "Copying assets to build directory"
  VERBATIM
)

# Generate before copying so the build dir always has the fresh .gltf.
if(PYTHON3_EXECUTABLE)
  add_dependencies(assets physics_demos shadow_lod_demos)
endif()

# --- Tests --------------------------------------------------------------

enable_testing()

add_executable(test_fire_engine
    tests/collision/test_aabb.cpp
    tests/collision/test_geometry.cpp
    tests/collision/test_sweep_and_prune_broad_phase.cpp
    tests/collision/test_dynamic_aabb_tree_broad_phase.cpp
    tests/collision/test_aabb_bvh.cpp
    tests/collision/test_narrow_phase.cpp
    tests/collision/test_support.cpp
    tests/collision/test_gjk_epa.cpp
    tests/collision/test_shape_cast.cpp
    tests/physics/test_physics_world.cpp
    tests/physics/test_physics_query.cpp
    tests/physics/test_collision_events.cpp
    tests/physics/test_character_controller.cpp
    tests/physics/test_contact_solver.cpp
    tests/physics/test_island.cpp
    tests/physics/test_compound.cpp
    tests/physics/test_static_mesh.cpp
    tests/physics/test_joint.cpp
    tests/physics/test_articulation.cpp
    tests/physics/test_physics_handle.cpp
    tests/physics/test_physics_determinism.cpp
    tests/physics/test_demos.cpp
    tests/math/test_mat3.cpp
    tests/math/test_singular_value.cpp
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
    tests/graphics/test_frustum.cpp
    tests/graphics/test_vertex.cpp
    tests/graphics/test_skin.cpp
    tests/graphics/test_assets.cpp
    tests/graphics/test_material.cpp
    tests/graphics/test_vdpm_material.cpp
    tests/graphics/test_geometry.cpp
    tests/graphics/test_gpu_handle.cpp
    tests/graphics/test_mapped_buffer.cpp
    tests/graphics/test_mesh_simplifier.cpp
    tests/graphics/test_vipm.cpp
    tests/graphics/test_vdpm.cpp
    tests/graphics/test_vdpm_gpu_registry.cpp
    tests/graphics/test_vdpm_parallel.cpp
    tests/graphics/test_mesh_topology.cpp
    tests/graphics/test_vdpm_wedge_choices.cpp
    tests/graphics/test_bounds.cpp
    tests/graphics/test_draw_command.cpp
    tests/graphics/test_object.cpp
    tests/graphics/test_frame_capture.cpp
    tests/graphics/test_frame_info.cpp
    tests/graphics/test_shadow_diagnostics.cpp
    tests/graphics/test_shadow_view.cpp
    tests/graphics/test_shadow_identity.cpp
    tests/graphics/test_shadow_lod_resolver.cpp
    tests/graphics/test_shadow_render_view.cpp
    tests/graphics/test_texture.cpp
    tests/graphics/test_sampler_settings.cpp
    tests/render/test_ubo.cpp
    tests/render/test_descriptors.cpp
    tests/render/test_device_plan.cpp
    tests/render/test_pipeline_config.cpp
    tests/render/test_vdpm_gpu.cpp
    tests/render/test_vdpm_gpu_front.cpp
    tests/render/test_vdpm_gpu_repair.cpp
    tests/render/test_vdpm_gpu_manager.cpp
    tests/render/test_vdpm_gpu_runtime.cpp
    tests/render/test_vdpm_repair_kernel.cpp
    tests/render/test_vdpm_apply_kernel.cpp
    tests/render/test_vdpm_health_reduce.cpp
    tests/render/test_vdpm_helmet_evidence.cpp
    tests/render/test_vdpm_scan.cpp
    tests/graphics/test_generational_slot_pool.cpp
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
    tests/scene/test_ragdoll.cpp
    tests/scene/test_scene_culler.cpp
    tests/scene/test_scene_graph.cpp
    tests/scene/test_formatter.cpp
    tests/animation/test_animation.cpp
    tests/scene/test_transform.cpp
)
target_include_directories(test_fire_engine PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_fire_engine PRIVATE fireengine Catch2::Catch2WithMain)
target_compile_options(test_fire_engine PRIVATE -Wall -Wextra -Wpedantic)
if(FIRE_ENGINE_WARNINGS_AS_ERRORS)
    target_compile_options(test_fire_engine PRIVATE -Werror)
endif()
# The B5c-2 helmet-evidence test loads the copied asset by absolute path (CWD-independent).
target_compile_definitions(test_fire_engine PRIVATE
    FIRE_ENGINE_BUILD_ASSET_DIR="${CMAKE_CURRENT_BINARY_DIR}")
# The [.][gpu] tests load .spv from the build dir at runtime — depend on `shaders` so building the
# test target alone can never leave a stale shader (tests-full inherits this via test_fire_engine).
add_dependencies(test_fire_engine shaders)

file(GLOB_RECURSE TEST_ASSETS "${PROJECT_SOURCE_DIR}/tests/assets/*")
foreach(f IN LISTS TEST_ASSETS)
  get_filename_component(name "${f}" NAME)
  configure_file("${f}" "${CMAKE_CURRENT_BINARY_DIR}/test_assets/${name}" COPYONLY)
endforeach()

add_test(NAME test_fire_engine COMMAND test_fire_engine "~[slow]")

add_custom_target(tests-full
  COMMAND $<TARGET_FILE:test_fire_engine>
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R layering_guards
  DEPENDS test_fire_engine
  COMMENT "Running the full test suite, including [slow] tests"
)

# Layering guards over the public headers: graphics/ and scene/ must not include render/, and
# render/ must not include scene/ (CR-09). Runs as a CTest case so `ctest` catches regressions
# without a GPU.
add_test(
  NAME layering_guards
  COMMAND ${CMAKE_COMMAND}
    -DINCLUDE_DIR=${PROJECT_SOURCE_DIR}/include/fire_engine
    -P ${PROJECT_SOURCE_DIR}/cmake/check_layering.cmake
)

find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
    get_target_property(FIRE_ENGINE_TIDY_SOURCES fireengine SOURCES)
    set(FIRE_ENGINE_TIDY_TARGETS)

    foreach(tidy_source IN LISTS FIRE_ENGINE_TIDY_SOURCES)
        if(tidy_source MATCHES "\\.(cc|cpp|cxx)$")
            get_filename_component(tidy_source_abs "${tidy_source}" ABSOLUTE
                BASE_DIR "${PROJECT_SOURCE_DIR}"
            )
            file(RELATIVE_PATH tidy_source_rel "${PROJECT_SOURCE_DIR}" "${tidy_source_abs}")
            string(MAKE_C_IDENTIFIER "${tidy_source_rel}" tidy_source_id)

            add_custom_target(run-clang-tidy-${tidy_source_id}
                COMMAND ${CLANG_TIDY_EXE}
                    -p ${CMAKE_BINARY_DIR}
                    ${tidy_source_abs}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMENT "Running clang-tidy on ${tidy_source_rel}"
                VERBATIM
            )
            list(APPEND FIRE_ENGINE_TIDY_TARGETS run-clang-tidy-${tidy_source_id})
        endif()
    endforeach()

    add_custom_target(run-clang-tidy
        DEPENDS ${FIRE_ENGINE_TIDY_TARGETS}
        COMMENT "Running clang-tidy on fireengine sources"
    )
endif()
