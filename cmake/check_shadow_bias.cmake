# Guard: the receiver shader uses THE shared bias law in every path, and does not grow a private one.
#
# SH-07 replaced a local formula — `baseBias * exp2(cascade)`, which stood in for a texel footprint
# and a depth-range conversion at once — with `shaders/shadow_bias.glsl`, the single production
# implementation shared by every receiver path and mirrored by the unit-tested C++ in
# `graphics/shadow_bias.hpp`. That arrangement is only worth anything while it stays the ONLY path: a
# well-meant local `bias` expression in one sampler would be invisible (no validation error, no
# crash, just shadows subtly wrong in one family) and would silently un-fix the item.
#
# COMMENTS ARE STRIPPED FIRST — BOTH GLSL forms. Every check below asks whether the CODE does
# something, and a match is otherwise satisfied by commented-out text, which is exactly how a call
# gets disabled while appearing to survive. Line comments alone are not enough: wrapping a whole
# receiver path in /* ... */ keeps its call and its metrics read visible to a naive match, and can
# hide the law's own definition so every later check passes vacuously. Both gaps were found by
# mutation-testing this guard, which is the only way to know a textual check bites.
#
# The checks catch reintroduction, not every possible re-derivation; that is the honest limit of a
# textual build-time guard, and it is still the only thing standing between the fix and its quiet
# reversal.
#
# Invoked as a CTest case; needs SHADER_DIR.

if(NOT DEFINED SHADER_DIR)
  message(FATAL_ERROR "SHADER_DIR must be set (path to shaders/)")
endif()

# Comment stripping is shared with the other shader guards — see the file for why both forms matter.
include("${CMAKE_CURRENT_LIST_DIR}/strip_glsl_comments.cmake")

set(offenders "")
set(law "${SHADER_DIR}/shadow_bias.glsl")
set(receiver "${SHADER_DIR}/shader.frag")

if(NOT EXISTS "${law}")
  list(APPEND offenders "shadow_bias.glsl is missing — the shared bias law is the whole point of SH-07")
endif()
if(NOT EXISTS "${receiver}")
  list(APPEND offenders "shader.frag is missing from ${SHADER_DIR}")
endif()

if(EXISTS "${law}" AND EXISTS "${receiver}")
  # The law must actually be defined where it claims to live, or every check below passes vacuously.
  file(READ "${law}" law_text)
  strip_glsl_comments("${law_text}" law_code)
  if(NOT law_code MATCHES "ShadowBias[ \t]+shadowBiasFor[ \t]*\\(")
    list(APPEND offenders "shadow_bias.glsl no longer defines shadowBiasFor() — this guard would pass vacuously")
  endif()

  file(READ "${receiver}" receiver_text)
  strip_glsl_comments("${receiver_text}" receiver_code)

  if(NOT receiver_code MATCHES "#include[ \t]+\"shadow_bias.glsl\"")
    list(APPEND offenders "shader.frag does not #include \"shadow_bias.glsl\" — the receiver must use the shared law, not its own")
  endif()

  # EVERY receiver path, not just one. A single-call check passes while four paths still use the law
  # and the fifth has quietly grown a local formula — precisely the reversal worth guarding, since
  # the four healthy paths make the shader look right.
  #
  # Two independent pins, because either alone is weak: the CALL COUNT (five known paths — the
  # directional sampler, the self-shadow sampler, the debug depth readout, spot and point) and the
  # per-family METRICS ARRAY each path must consume. A path that stops calling the law changes the
  # count; a path that keeps the call but stops reading its own view's metrics loses its array.
  #
  # EXACT, not a floor. Adding a receiver path must fail this check: bump the expected count
  # deliberately, having confirmed the new path uses the shared law. A floor would let a sixth path
  # arrive with its own formula as long as the other five behaved.
  set(expected_law_calls 5)
  string(REGEX MATCHALL "shadowBiasFor[ \t]*\\(" law_calls "${receiver_code}")
  list(LENGTH law_calls law_call_count)
  if(NOT law_call_count EQUAL expected_law_calls)
    list(APPEND offenders
         "shader.frag calls shadowBiasFor() ${law_call_count} time(s); expected exactly ${expected_law_calls} (directional, self, debug readout, spot, point). Fewer means a path stopped using the shared law; more means a new path arrived and this count needs updating deliberately")
  endif()

  foreach(family cascadeBiasMetrics selfBiasMetrics spotBiasMetrics pointBiasMetrics)
    if(NOT receiver_code MATCHES "light\\.${family}\\[")
      list(APPEND offenders
           "shader.frag never reads light.${family}[] — that family's receiver is not using its own fitted metrics")
    endif()
  endforeach()

  # `exp2(cascade)` was the defect: one constant standing in for a per-view footprint AND a per-view
  # depth range. No other legitimate use of exp2 exists in this shader today; if one ever does,
  # narrow this check rather than deleting it.
  if(receiver_code MATCHES "exp2[ \t]*\\(")
    list(APPEND offenders "shader.frag uses exp2( — SH-07 removed per-cascade bias scaling; each view carries its own fitted metrics")
  endif()
endif()

if(offenders)
  string(REPLACE ";" "\n  " report "${offenders}")
  message(FATAL_ERROR "shadow bias guard failed:\n  ${report}")
endif()

message(STATUS "shadow bias guard: all five receiver paths use the shared law; no per-cascade exp2 scaling")
