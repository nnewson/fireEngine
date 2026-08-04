# Guard: a uniform block shared by more than one shader must be DECLARED once, in a shared include.
#
# Why this is worth a test. A uniform block's field offsets depend on every field declared before
# them, so a block written out by hand in two shaders is a latent layout bug: add a field to one copy
# and the other silently reads every later field at the wrong offset. That happened — `LightUBO`
# gained `selfShadowViewProj` in shader.frag and in the C++ struct but not in skybox.frag, which then
# read `environmentParams` 256 bytes early and multiplied the sky by a shadow matrix element. Nothing
# failed: no validation error, no crash, and the wrong value happened to be 1.0 until a scene
# supplied two skinned self-shadow casters. Only the shared declaration makes that unrepresentable,
# and only this check keeps a future shader from hand-rolling a copy again.
#
# Rule: for each guarded block name, exactly one file (the shared include) may declare it. Any other
# shader must reach it via #include. Invoked as a CTest case; needs SHADER_DIR.

if(NOT DEFINED SHADER_DIR)
  message(FATAL_ERROR "SHADER_DIR must be set (path to shaders/)")
endif()

# <block name>|<the one file allowed to declare it>
#
# SH-05 added `Materials` and `ShadowPushConstants`; the cutout-aware depth prepass added
# `ForwardPushConstants`. The two push blocks are the worst case of the kinds guarded here: a raw byte
# range with no reflection at all — strictly worse than a UBO, which at least has a declared size —
# and `ShadowPushConstants` had already been hand-copied into three shadow stages before a fourth
# needed it. `Materials` and `MaterialData` became shared the moment a depth-only pass had to apply
# the VISIBLE material's alpha cutout: a second copy of that struct is a second cutoff, a second
# UV-set choice and a second transform, and a pass disagreeing with its own surface reads as a bias
# bug.
set(guarded_blocks
  "LightUBO|light_ubo.glsl"
  "Materials|material.glsl"
  "ShadowPushConstants|shadow_push.glsl"
  "ForwardPushConstants|forward_push.glsl"
)

set(offenders "")
foreach(entry IN LISTS guarded_blocks)
  string(REPLACE "|" ";" parts "${entry}")
  list(GET parts 0 block)
  list(GET parts 1 owner)

  if(NOT EXISTS "${SHADER_DIR}/${owner}")
    list(APPEND offenders "${block}: shared declaration ${owner} is missing from ${SHADER_DIR}")
    continue()
  endif()

  file(GLOB shader_files
       "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp" "${SHADER_DIR}/*.glsl")
  set(declaring "")
  foreach(shader IN LISTS shader_files)
    # A declaration opens the block body; a mere reference (light.environmentParams) does not.
    # `buffer` as well as `uniform`: the bindless materials[] SSBO is a storage block, and its field
    # offsets drift exactly like a UBO's.
    file(STRINGS "${shader}" matches REGEX "(uniform|buffer)[ \t]+${block}[ \t]*\\{")
    if(matches)
      get_filename_component(shader_name "${shader}" NAME)
      list(APPEND declaring "${shader_name}")
    endif()
  endforeach()

  foreach(shader_name IN LISTS declaring)
    if(NOT shader_name STREQUAL owner)
      list(APPEND offenders
           "${shader_name} declares '${block}' inline; include ${owner} instead so the layout cannot drift")
    endif()
  endforeach()

  # list(FIND), not IN_LIST: this runs as `cmake -P`, where no project()/cmake_minimum_required has
  # set CMP0057, so IN_LIST is not an operator on older CMake and the script hard-errors there. It
  # passed locally and failed in the Ubuntu container on exactly that difference.
  list(FIND declaring "${owner}" owner_index)
  if(owner_index EQUAL -1)
    list(APPEND offenders "${owner} no longer declares '${block}' — the guard would pass vacuously")
  endif()
endforeach()

if(offenders)
  string(REPLACE ";" "\n  " offenders_text "${offenders}")
  message(FATAL_ERROR "shared shader block guard violated:\n  ${offenders_text}")
endif()

message(STATUS "shader block guard: shared uniform blocks have exactly one declaration")
