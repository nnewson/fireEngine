// The single translation unit that emits VMA's amalgamated implementation. Kept alone (and
// exempt from the engine's strict warning flags via CMake) so the third-party code compiles
// once, out of the way of our own sources.
//
// Static-function mode: the engine links the Vulkan loader directly (vulkan-hpp static
// dispatch), so VMA calls the core entry points straight through rather than fetching pointers.
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <fire_engine/render/vma.hpp>
