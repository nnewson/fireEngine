# Guard: ONE shadow transform reaches the GPU, and it is the recorded view's.
#
# The shadow pass used to push a 32-matrix table of every shadow view into every draw's per-object
# UBO and select a row with a push constant. That made the transform a per-DRAW lookup rather than a
# property of the view being recorded, and it left two descriptions of the same value: the table the
# vertex shader indexed, and the matrix everything else reasoned about. Arc 2 #4 needs those to be
# one thing — a cached shadow map may only be reused if the matrix compared is the matrix rasterised
# with — so `pc.lightViewProj` is now the only shadow transform, for every family.
#
# Reintroducing the table would compile and render correctly on the frame it was added; the damage
# is that the cache's comparison would silently stop describing what the GPU does. Nothing else can
# catch that, hence a build-time check.
#
# The same reasoning covers the depth discriminator: the point path branched on "is this matrix
# index at or past the point base", inferring a depth mode from where a matrix happened to live.
# It now reads `pc.radialDepth`, which is `PreparedShadowView::depthMode()` and nothing else.
#
# Invoked as a CTest case; needs SHADER_DIR.

if(NOT DEFINED SHADER_DIR)
  message(FATAL_ERROR "SHADER_DIR must be set (path to shaders/)")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/strip_glsl_comments.cmake")

set(offenders "")

# Every shader, so a NEW shadow stage cannot quietly reintroduce either pattern.
file(GLOB shader_sources
     "${SHADER_DIR}/*.glsl" "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp")
foreach(shader IN LISTS shader_sources)
  file(READ "${shader}" shader_text)
  strip_glsl_comments("${shader_text}" shader_code)
  get_filename_component(shader_name "${shader}" NAME)

  # An ARRAY of light matrices — the table itself, wherever it is declared. `mat4 lightViewProj[N]`
  # in a block, or an index into one.
  if(shader_code MATCHES "lightViewProj[ \t]*\\[")
    list(APPEND offenders
         "${shader_name} indexes or declares lightViewProj[] — the per-draw shadow matrix table is retired; use pc.lightViewProj, the matrix of the view being recorded")
  endif()
  # The ShadowUBO member it used to live in.
  if(shader_code MATCHES "shadow[ \t]*\\.[ \t]*lightViewProj")
    list(APPEND offenders
         "${shader_name} reads shadow.lightViewProj — ShadowUBO carries the object's model matrix only")
  endif()
  # The selector. Its absence is what forces a depth mode to be stated rather than inferred.
  if(shader_code MATCHES "matrixIndex")
    list(APPEND offenders
         "${shader_name} mentions matrixIndex — the push block carries radialDepth (the view's depth mode) in its place")
  endif()
endforeach()

# And the positive half: the paths that must consume the pushed values still do. Without these the
# checks above would pass on a shader that had stopped drawing anything at all.
set(required
    "shadow.vert:pc[ \t]*\\.[ \t]*lightViewProj:the shadow vertex stage must rasterise with the pushed view matrix"
    "shadow_push.glsl:int[ \t]+radialDepth:the shared push block must declare the depth-mode discriminator"
    "shadow_depth.glsl:pc[ \t]*\\.[ \t]*radialDepth:the point-face depth path must branch on the pushed depth mode")
foreach(entry IN LISTS required)
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 required_file)
  list(GET parts 1 required_pattern)
  list(GET parts 2 required_reason)
  set(path "${SHADER_DIR}/${required_file}")
  if(NOT EXISTS "${path}")
    list(APPEND offenders "${required_file} is missing from ${SHADER_DIR}")
    continue()
  endif()
  file(READ "${path}" required_text)
  strip_glsl_comments("${required_text}" required_code)
  if(NOT required_code MATCHES "${required_pattern}")
    list(APPEND offenders "${required_file}: ${required_reason}")
  endif()
endforeach()

if(offenders)
  string(REPLACE ";" "\n  " report "${offenders}")
  message(FATAL_ERROR "shadow matrix guard failed:\n  ${report}")
endif()

message(STATUS "shadow matrix guard: pc.lightViewProj is the only shadow transform; depth mode is stated, not inferred")
