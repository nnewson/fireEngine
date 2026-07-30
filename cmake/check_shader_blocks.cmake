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
set(guarded_blocks
  "LightUBO|light_ubo.glsl"
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
    file(STRINGS "${shader}" matches REGEX "uniform[ \t]+${block}[ \t]*\\{")
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

  if(NOT "${owner}" IN_LIST declaring)
    list(APPEND offenders "${owner} no longer declares '${block}' — the guard would pass vacuously")
  endif()
endforeach()

if(offenders)
  string(REPLACE ";" "\n  " offenders_text "${offenders}")
  message(FATAL_ERROR "shared shader block guard violated:\n  ${offenders_text}")
endif()

message(STATUS "shader block guard: shared uniform blocks have exactly one declaration")
