#include <algorithm>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <fire_engine/render/device_plan.hpp>

using namespace fire_engine;

namespace
{

// The planner takes what the loader reported; these fixtures stand in for that enumeration so the
// decisions are testable with no loader, no device, and no GPU.
const std::vector<std::string> kMoltenVkInstanceExtensions{"VK_KHR_surface", "VK_EXT_metal_surface",
                                                           kPortabilityEnumerationExtension};
const std::vector<std::string> kLinuxInstanceExtensions{"VK_KHR_surface", "VK_KHR_xcb_surface",
                                                        kPortabilityEnumerationExtension};
const std::vector<std::string> kWindowExtensions{"VK_KHR_surface", "VK_EXT_metal_surface"};

[[nodiscard]] bool contains(const std::vector<std::string>& names, const std::string& name)
{
    return std::ranges::find(names, name) != names.end();
}

[[nodiscard]] std::size_t count(const std::vector<std::string>& names, const std::string& name)
{
    return static_cast<std::size_t>(std::ranges::count(names, name));
}

} // namespace

TEST_CASE("instance plan enables validation only when the layer is installed", "[DevicePlan]")
{
    const std::vector<std::string> withLayer{"VK_LAYER_LUNARG_monitor", kValidationLayer};
    const std::vector<std::string> withoutLayer{"VK_LAYER_LUNARG_monitor"};

    SECTION("wanted and installed")
    {
        const auto plan = planInstanceCapabilities(kMoltenVkInstanceExtensions, withLayer,
                                                   kWindowExtensions, true);
        CHECK(plan.validation);
    }
    SECTION("wanted but absent — the Linux-without-SDK case that used to be fatal")
    {
        const auto plan = planInstanceCapabilities(kMoltenVkInstanceExtensions, withoutLayer,
                                                   kWindowExtensions, true);
        CHECK_FALSE(plan.validation);
        // The layer is never requested, so instance creation cannot fail with LAYER_NOT_PRESENT.
        CHECK_FALSE(contains(plan.extensions, kValidationLayer));
    }
    SECTION("installed but not wanted (NDEBUG build)")
    {
        const auto plan = planInstanceCapabilities(kMoltenVkInstanceExtensions, withLayer,
                                                   kWindowExtensions, false);
        CHECK_FALSE(plan.validation);
    }
}

TEST_CASE("instance plan requests portability enumeration only when the loader has it",
          "[DevicePlan]")
{
    const std::vector<std::string> layers{};

    SECTION("present")
    {
        const auto plan =
            planInstanceCapabilities(kMoltenVkInstanceExtensions, layers, kWindowExtensions, false);
        CHECK(plan.portabilityEnumeration);
        CHECK(contains(plan.extensions, kPortabilityEnumerationExtension));
    }
    SECTION("absent — the flag must not be set without the extension")
    {
        const std::vector<std::string> available{"VK_KHR_surface", "VK_KHR_xcb_surface"};
        const auto plan = planInstanceCapabilities(available, layers, kWindowExtensions, false);
        CHECK_FALSE(plan.portabilityEnumeration);
        CHECK_FALSE(contains(plan.extensions, kPortabilityEnumerationExtension));
    }
}

TEST_CASE("instance plan always carries the window's WSI extensions", "[DevicePlan]")
{
    const auto windowed =
        planInstanceCapabilities(kLinuxInstanceExtensions, {}, kWindowExtensions, false);
    for (const std::string& required : kWindowExtensions)
    {
        CHECK(contains(windowed.extensions, required));
    }

    // Headless compute passes none: no surface, so no WSI extensions (and it never touches GLFW).
    const auto headless = planInstanceCapabilities(kLinuxInstanceExtensions, {}, {}, false);
    CHECK_FALSE(contains(headless.extensions, "VK_EXT_metal_surface"));
}

TEST_CASE("device plan enables the portability subset exactly when advertised", "[DevicePlan]")
{
    const std::vector<std::string> required{"VK_KHR_swapchain"};

    SECTION("MoltenVK advertises it — mandatory, so it is enabled")
    {
        const std::vector<std::string> available{"VK_KHR_swapchain", kPortabilitySubsetExtension,
                                                 "VK_KHR_push_descriptor"};
        const auto plan = planDeviceCapabilities(available, required);
        CHECK(plan.portabilitySubset);
        CHECK(contains(plan.extensions, kPortabilitySubsetExtension));
        CHECK(plan.missingRequired.empty());
    }
    SECTION("a conformant driver does not — requesting it there is an error")
    {
        const std::vector<std::string> available{"VK_KHR_swapchain", "VK_KHR_push_descriptor"};
        const auto plan = planDeviceCapabilities(available, required);
        CHECK_FALSE(plan.portabilitySubset);
        CHECK_FALSE(contains(plan.extensions, kPortabilitySubsetExtension));
        CHECK(plan.missingRequired.empty());
    }
}

TEST_CASE("device plan reports missing required extensions by name", "[DevicePlan]")
{
    const std::vector<std::string> required{"VK_KHR_swapchain"};
    const std::vector<std::string> available{kPortabilitySubsetExtension};

    const auto plan = planDeviceCapabilities(available, required);
    REQUIRE(plan.missingRequired.size() == 1);
    CHECK(plan.missingRequired.front() == "VK_KHR_swapchain");
    // A missing required extension is never silently enabled...
    CHECK_FALSE(contains(plan.extensions, "VK_KHR_swapchain"));
    // ...but an advertised optional one is still planned, so the diagnostic isn't the only output.
    CHECK(plan.portabilitySubset);
}

TEST_CASE("device plan never repeats an extension name", "[DevicePlan]")
{
    // A duplicate in ppEnabledExtensionNames is legal but sloppy; more importantly this pins that
    // an optional extension which is ALSO listed as required is added once, not twice.
    const std::vector<std::string> required{"VK_KHR_swapchain", kPortabilitySubsetExtension};
    const std::vector<std::string> available{"VK_KHR_swapchain", kPortabilitySubsetExtension};

    const auto plan = planDeviceCapabilities(available, required);
    CHECK(count(plan.extensions, kPortabilitySubsetExtension) == 1);
    CHECK(count(plan.extensions, "VK_KHR_swapchain") == 1);
    CHECK(plan.portabilitySubset);
}

TEST_CASE("headless device plan requires no swapchain", "[DevicePlan]")
{
    // Device::requiredDeviceExtensions() returns an empty list headless; the plan must then accept
    // a device that has no swapchain support at all (offscreen compute never presents).
    const std::vector<std::string> available{"VK_KHR_maintenance5"};
    const auto plan = planDeviceCapabilities(available, {});
    CHECK(plan.missingRequired.empty());
    CHECK(plan.extensions.empty());
    CHECK_FALSE(plan.portabilitySubset);
}
