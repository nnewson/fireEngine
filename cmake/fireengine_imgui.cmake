# Build a local ImGui imported target without inheriting vcpkg's transitive
# glfw/Vulkan link interface.
#
# vcpkg's imgui::imgui target for the glfw/vulkan feature build exports
# glfw and Vulkan::Vulkan through INTERFACE_LINK_LIBRARIES. Fire Engine owns
# those dependencies directly, so linking the imported target as-is duplicates
# static-library entries on Apple ld.

if(NOT TARGET imgui::imgui)
  message(FATAL_ERROR "imgui::imgui must exist before including fireengine_imgui.cmake")
endif()

add_library(fireengine_imgui STATIC IMPORTED)

get_target_property(FIREENGINE_IMGUI_INCLUDE_DIRS imgui::imgui INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(fireengine_imgui PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${FIREENGINE_IMGUI_INCLUDE_DIRS}"
)

foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  get_target_property(FIREENGINE_IMGUI_LOCATION imgui::imgui IMPORTED_LOCATION_${config})
  if(FIREENGINE_IMGUI_LOCATION)
    set_property(TARGET fireengine_imgui APPEND PROPERTY IMPORTED_CONFIGURATIONS ${config})
    set_target_properties(fireengine_imgui PROPERTIES
      IMPORTED_LOCATION_${config} "${FIREENGINE_IMGUI_LOCATION}"
      IMPORTED_LINK_INTERFACE_LANGUAGES_${config} CXX
    )
  endif()
endforeach()
