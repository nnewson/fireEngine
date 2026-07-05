# Layering guards over the public headers of the engine's layers.
#
# Enforced (header-only — .cpp files may reach across for the documented bridges, e.g. a
# graphics/ .cpp including render/resources.hpp to allocate GPU resources):
#   * graphics/ must not include render/  — keeps the graphics layer Vulkan-free.
#   * scene/    must not include render/  — the scene layer stays backend-agnostic (CR-09).
#   * render/   must not include scene/   — the renderer depends on the graphics/ RenderableScene
#                                           seam, not the concrete scene graph (CR-09).
#
# Invoked as a CTest case (see CMakeLists.txt). Fails with a non-zero status and the offending
# lines if any rule is violated. Needs INCLUDE_DIR = <...>/include/fire_engine.

if(NOT DEFINED INCLUDE_DIR)
  message(FATAL_ERROR "INCLUDE_DIR must be set (path to include/fire_engine)")
endif()

# Rule triples: <layer subdir> must not include <forbidden layer prefix> (<why>).
set(rules
  "graphics|render|keeps the graphics layer Vulkan-free"
  "scene|render|keeps the scene layer backend-agnostic (CR-09)"
  "render|scene|the renderer uses the graphics/ RenderableScene seam, not the concrete scene (CR-09)"
)

set(offenders "")
foreach(rule IN LISTS rules)
  string(REPLACE "|" ";" parts "${rule}")
  list(GET parts 0 layer)
  list(GET parts 1 forbidden)
  list(GET parts 2 why)

  file(GLOB_RECURSE headers "${INCLUDE_DIR}/${layer}/*.hpp")
  foreach(header IN LISTS headers)
    file(STRINGS "${header}" matches
         REGEX "#[ \t]*include[ \t]*[<\"]fire_engine/${forbidden}/")
    foreach(line IN LISTS matches)
      list(APPEND offenders "[${layer}/ -> !${forbidden}/ : ${why}] ${header}: ${line}")
    endforeach()
  endforeach()
endforeach()

if(offenders)
  string(REPLACE ";" "\n  " offenders_text "${offenders}")
  message(FATAL_ERROR "layering guard violated:\n  ${offenders_text}")
endif()

message(STATUS "layering guards: OK (graphics/scene !-> render, render !-> scene)")
