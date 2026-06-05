# Layering guard: the Vulkan-free graphics layer must not depend on render/.
#
# Graphics *headers* form the layer's public interface and must stay clear of
# render/ entirely. Graphics .cpp files are allowed to include render/resources.hpp
# (the documented bridge for allocating GPU resources), so this guard only scans
# headers under include/fire_engine/graphics/.
#
# Invoked as a CTest case (see CMakeLists.txt). Fails with a non-zero status and
# a list of offending lines if any graphics header includes a render/ header.

if(NOT DEFINED GRAPHICS_INCLUDE_DIR)
  message(FATAL_ERROR "GRAPHICS_INCLUDE_DIR must be set")
endif()

file(GLOB_RECURSE graphics_headers "${GRAPHICS_INCLUDE_DIR}/*.hpp")

set(offenders "")
foreach(header IN LISTS graphics_headers)
  file(STRINGS "${header}" matches REGEX "#[ \t]*include[ \t]*[<\"]fire_engine/render/")
  foreach(line IN LISTS matches)
    list(APPEND offenders "${header}: ${line}")
  endforeach()
endforeach()

if(offenders)
  string(REPLACE ";" "\n  " offenders_text "${offenders}")
  message(FATAL_ERROR
    "graphics/ headers must not include render/ (keeps the graphics layer Vulkan-free):\n  ${offenders_text}")
endif()

message(STATUS "graphics layering guard: OK (no render/ includes in graphics headers)")
