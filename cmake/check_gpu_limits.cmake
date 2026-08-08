# Guard: the limits shared by C++ and GLSL are declared ONCE, in shaders/gpu_limits.glsl, and both
# sides actually read them.
#
# These numbers size UBO arrays and index the shadow matrix table. A one-sided change — a shader
# raising a caster count that the C++ struct still writes at the old size, or the reverse — compiles
# cleanly in both languages and then reads the wrong region of a bound buffer: every index stays in
# range, so there is no validation error and no crash, just a shadow matrix taken from another
# family's slot. `SHADOW_TOTAL_MATRIX_COUNT = 32`, `SHADOW_POINT_MATRIX_BASE = 8` and the caster
# counts were each hand-transcribed exactly that way before this guard existed.
#
# Two halves, and BOTH are needed. The GLSL half fails if a shader re-declares a shared name or
# stops including the file. The C++ half fails if graphics/gpu_limits.hpp stops including the shared
# file, or re-exports a name as anything other than the shared declaration — otherwise C++ could
# quietly return to hard-coded values while every shader-side check stayed green, which is the same
# drift with the sides swapped.
#
# COMMENTS ARE STRIPPED FIRST, both forms, in both languages: a check that a file "uses" a name is
# otherwise satisfied by the name appearing in prose, and this file is full of prose about it.
#
# Invoked as a CTest case; needs SHADER_DIR and INCLUDE_DIR.

if(NOT DEFINED SHADER_DIR)
  message(FATAL_ERROR "SHADER_DIR must be set (path to shaders/)")
endif()
if(NOT DEFINED INCLUDE_DIR)
  message(FATAL_ERROR "INCLUDE_DIR must be set (path to include/)")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/strip_glsl_comments.cmake")

set(offenders "")
set(shared "${SHADER_DIR}/gpu_limits.glsl")
set(cpp_authority "${INCLUDE_DIR}/fire_engine/graphics/gpu_limits.hpp")

# The shared names, paired with the C++ constant each must be re-exported as. Adding a limit that
# both languages need means adding it here too — deliberately, which is the point.
set(shared_names
    MAX_LIGHTS
    MAX_JOINTS
    MAX_MORPH_TARGETS
    MORPH_WEIGHT_VEC4_COUNT
    MAX_PARTICLE_EMITTERS
    SSAO_KERNEL_SIZE
    SHADOW_CASCADE_COUNT
    MAX_SKINNED_SELF_SHADOW_CASTERS
    MAX_SPOT_SHADOW_CASTERS
    MAX_POINT_SHADOW_CASTERS
    CUBE_FACE_COUNT
    SHADOW_CASCADE_MATRIX_BASE
    SHADOW_SPOT_MATRIX_BASE
    SHADOW_POINT_MATRIX_BASE
    SHADOW_TOTAL_MATRIX_COUNT
    SHADOW_MAP_VALID_CASCADES
    SHADOW_MAP_VALID_WORLD_ONLY
    SHADOW_MAP_VALID_SELF
    SHADOW_MAP_VALID_SPOT
    SHADOW_MAP_VALID_POINT)
set(cpp_names
    kMaxLights
    kMaxJoints
    kMaxMorphTargets
    kMorphWeightVec4Count
    kMaxParticleEmitters
    kSsaoKernelSize
    kShadowCascadeCount
    kMaxSkinnedSelfShadowCasters
    kMaxSpotShadowCasters
    kMaxPointShadowCasters
    kCubeFaceCount
    kShadowCascadeMatrixBase
    kShadowSpotMatrixBase
    kShadowPointMatrixBase
    kShadowTotalMatrixCount
    kShadowMapValidCascades
    kShadowMapValidWorldOnly
    kShadowMapValidSelf
    kShadowMapValidSpot
    kShadowMapValidPoint)

# Each consumer as `file:NAME:uses` — the shared name it must be READING, and HOW MANY TIMES. A file
# that keeps the #include but goes back to a literal at the point of use is the failure this pins;
# the include alone proves nothing, and neither does a single mention when the file uses the value in
# several places. (Mutation-testing found exactly that gap: reverting `ssao.frag`'s loop bound to 16
# passed a presence check, because the UBO array above it still named the constant.)
#
# A MINIMUM, unlike the bias guard's exact call count. There the count IS the contract — five
# receiver paths, no more — whereas here a new legitimate use of a size is unremarkable and losing
# one is the regression. Raise a number when a file gains a use; never lower one to make a red check
# pass, since that is the drift arriving.
set(consumers
    "shader.vert:MAX_JOINTS:2" # SkinUBO + PrevSkinUBO
    "shader.vert:MORPH_WEIGHT_VEC4_COUNT:1"
    "shadow.vert:MAX_JOINTS:1"
    "shadow.vert:MORPH_WEIGHT_VEC4_COUNT:1"
    "particle.vert:MAX_PARTICLE_EMITTERS:1"
    "particle_simulate.comp:MAX_PARTICLE_EMITTERS:1"
    "ssao.frag:SSAO_KERNEL_SIZE:3" # kernel[] + the loop bound + the occlusion divisor
    "light_ubo.glsl:SHADOW_CASCADE_COUNT:2" # cascadeViewProj[] + cascadeBiasMetrics[]
    "light_ubo.glsl:MAX_LIGHTS:1"
    "light_ubo.glsl:MAX_SPOT_SHADOW_CASTERS:2"
    "light_ubo.glsl:MAX_SKINNED_SELF_SHADOW_CASTERS:2"
    "light_ubo.glsl:MAX_POINT_SHADOW_CASTERS:1"
    "shadow.vert:SHADOW_TOTAL_MATRIX_COUNT:1"
    "shadow_depth.glsl:SHADOW_POINT_MATRIX_BASE:1"
    "self_shadow_second.glsl:MAX_SKINNED_SELF_SHADOW_CASTERS:1"
    "shader.frag:SHADOW_CASCADE_COUNT:4" # cascade search init + bound, blend factor, debug divisor
    "shader.frag:MAX_SKINNED_SELF_SHADOW_CASTERS:1"
    # EVERY family's validity bit is consulted by the receiver. A family the renderer skips has a
    # stale depth image, so a sampling path that stops asking reads last-frame's shadows with
    # nothing to report it — no error, no crash, just shadows from a frame that is gone. The cascade
    # bit is asked twice: by the sampler and by the raw-depth debug view, which reads lights[0] as
    # the sun and would have no valid one.
    "shader.frag:SHADOW_MAP_VALID_CASCADES:2"
    "shader.frag:SHADOW_MAP_VALID_WORLD_ONLY:1"
    "shader.frag:SHADOW_MAP_VALID_SELF:1"
    "shader.frag:SHADOW_MAP_VALID_SPOT:1"
    "shader.frag:SHADOW_MAP_VALID_POINT:1")

if(NOT EXISTS "${shared}")
  list(APPEND offenders "shaders/gpu_limits.glsl is missing — it is the single declaration both languages read")
else()
  file(READ "${shared}" shared_text)
  strip_glsl_comments("${shared_text}" shared_code)
  # Vacuity check first: if the shared file stopped declaring a name, the "nobody else declares it"
  # sweep below would pass while the value lived somewhere else entirely.
  foreach(name IN LISTS shared_names)
    if(NOT shared_code MATCHES "const[ \t]+int[ \t]+${name}[ \t]*=")
      list(APPEND offenders "gpu_limits.glsl no longer declares ${name} — the checks for it would pass vacuously")
    endif()
  endforeach()
  # The file has to stay in the GLSL/C++ common subset, and the C++-only keywords are what a C++
  # author reaches for first. A `constexpr` here breaks every shader compile in files that never
  # mention this one, so name the cause here instead.
  foreach(keyword constexpr inline namespace static_cast unsigned)
    if(shared_code MATCHES "(^|[^A-Za-z_])${keyword}[^A-Za-z_]")
      list(APPEND offenders "gpu_limits.glsl uses `${keyword}` — it must stay in the subset that is valid GLSL *and* valid C++")
    endif()
  endforeach()
endif()

# No shader may declare a shared name itself. The whole shaders/ tree is swept, not a list of known
# consumers: a NEW shader with its own `const int SHADOW_CASCADE_COUNT = 4;` is exactly the drift
# this exists to stop, and it would not be on any list.
file(GLOB shader_sources
     "${SHADER_DIR}/*.glsl" "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp")
foreach(shader IN LISTS shader_sources)
  if(shader STREQUAL "${shared}")
    continue()
  endif()
  file(READ "${shader}" shader_text)
  strip_glsl_comments("${shader_text}" shader_code)
  get_filename_component(shader_name "${shader}" NAME)
  foreach(name IN LISTS shared_names)
    if(shader_code MATCHES "const[ \t]+int[ \t]+${name}[ \t]*=")
      list(APPEND offenders
           "${shader_name} declares ${name} itself — include \"gpu_limits.glsl\" instead; a second declaration is how the two sides drift")
    endif()
  endforeach()
endforeach()

foreach(consumer IN LISTS consumers)
  string(REPLACE ":" ";" parts "${consumer}")
  list(GET parts 0 consumer_file)
  list(GET parts 1 consumer_name)
  list(GET parts 2 expected_uses)
  set(path "${SHADER_DIR}/${consumer_file}")
  if(NOT EXISTS "${path}")
    list(APPEND offenders "${consumer_file} is missing from ${SHADER_DIR}")
    continue()
  endif()
  file(READ "${path}" consumer_text)
  strip_glsl_comments("${consumer_text}" consumer_code)
  if(NOT consumer_code MATCHES "#include[ \t]+\"gpu_limits.glsl\"")
    list(APPEND offenders "${consumer_file} does not #include \"gpu_limits.glsl\"")
  endif()
  # The include line itself is code, not a use, and it does not contain the name — so every match
  # below is a real read of the value.
  #
  # SEMICOLONS ARE NEUTRALISED FIRST, and that is not cosmetic: `MATCHALL` returns a CMake LIST, and
  # a match that spans a `;` (`for (i < NAME; ++i)`) becomes TWO elements, so `list(LENGTH)` counted
  # one use as two and a lost use could hide behind the inflation. A `;` only ever terminates a GLSL
  # statement, so replacing it with a space cannot merge two identifiers.
  string(REPLACE ";" " " consumer_scan "${consumer_code}")
  string(REGEX MATCHALL "[^A-Za-z_0-9]${consumer_name}[^A-Za-z_0-9]" uses "${consumer_scan}")
  list(LENGTH uses use_count)
  if(use_count LESS expected_uses)
    list(APPEND offenders
         "${consumer_file} uses ${consumer_name} ${use_count} time(s); expected at least ${expected_uses}. A literal in its place is the drift this guard exists for")
  endif()
endforeach()

# The C++ half.
if(NOT EXISTS "${cpp_authority}")
  list(APPEND offenders "include/fire_engine/graphics/gpu_limits.hpp is missing — it is the C++ side of the shared limits")
else()
  file(READ "${cpp_authority}" cpp_text)
  strip_glsl_comments("${cpp_text}" cpp_code)
  if(NOT cpp_code MATCHES "#include[ \t]+\"gpu_limits.glsl\"")
    list(APPEND offenders
         "gpu_limits.hpp does not #include \"gpu_limits.glsl\" — the C++ side must READ the shared declarations, not restate them")
  endif()
  list(LENGTH shared_names shared_count)
  math(EXPR last_index "${shared_count} - 1")
  foreach(index RANGE ${last_index})
    list(GET shared_names ${index} shared_name)
    list(GET cpp_names ${index} cpp_name)
    # The INITIALISER, not merely a mention: `kMaxLights = 8;` beside an unused include is the exact
    # regression this half catches.
    if(NOT cpp_code MATCHES "${cpp_name}[ \t]*=[ \t]*shader_limits::${shared_name}[ \t]*;")
      list(APPEND offenders
           "gpu_limits.hpp does not define ${cpp_name} as shader_limits::${shared_name} — a hard-coded value here drifts from the shaders with nothing to catch it")
    endif()
  endforeach()
endif()

if(offenders)
  string(REPLACE ";" "\n  " report "${offenders}")
  message(FATAL_ERROR "gpu limits guard failed:\n  ${report}")
endif()

message(STATUS "gpu limits guard: one declaration in shaders/gpu_limits.glsl, read by every shader consumer and re-exported by gpu_limits.hpp")
