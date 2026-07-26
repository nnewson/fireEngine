#pragma once

#include <span>
#include <string>
#include <vector>

namespace fire_engine
{

// The Vulkan capability DECISIONS, separated from the Vulkan enumeration that feeds them.
//
// Everything platform-conditional about instance/device setup is resolved by asking the loader what
// it actually has, never by an #ifdef — one binary has to start on a MoltenVK Mac, a conformant
// Linux driver, and a machine with no Vulkan SDK installed. Enumeration needs a real loader, but
// the choices made from its answers are pure functions of (what's available, what we want), so they
// live here and are unit-tested headlessly (`tests/render/test_device_plan.cpp`).
//
// The plan is also the single authority that keeps the coupled decisions from drifting: device
// suitability and device creation call the SAME planner with the same inputs, so "is the
// portability subset enabled" cannot be answered differently in the two places (it used to be
// re-derived independently in each).

// Advertised only by a NON-CONFORMANT implementation (MoltenVK). Enabling it is MANDATORY when the
// device exposes it and an ERROR when it doesn't, so it is never a hard requirement — it is
// requested exactly when advertised.
inline constexpr const char* kPortabilitySubsetExtension = "VK_KHR_portability_subset";
// Lets the loader surface a non-conformant implementation at all. A loader extension, so it is
// present on any current loader; where it is absent there is also no portability driver to surface.
inline constexpr const char* kPortabilityEnumerationExtension = "VK_KHR_portability_enumeration";
inline constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

struct InstanceCapabilityPlan
{
    // Exactly the instance extensions to request: the caller's WSI extensions plus whatever
    // optional ones the loader advertises.
    std::vector<std::string> extensions;
    // The validation layer was wanted AND is installed. When false after wanting it, the caller
    // runs unvalidated (or refuses to start, if validation was made mandatory).
    bool validation{false};
    // Requesting VK_KHR_portability_enumeration; the eEnumeratePortabilityKHR instance flag is
    // legal only alongside it, so the two travel together.
    bool portabilityEnumeration{false};
};

struct DeviceCapabilityPlan
{
    // Required + every advertised optional extension — what device creation enables.
    std::vector<std::string> extensions;
    // Required extensions this device does NOT advertise. Non-empty ⇒ unsuitable, and these are
    // the names to report.
    std::vector<std::string> missingRequired;
    // The device advertises VK_KHR_portability_subset, so its extension is enabled, its feature
    // bits are meaningful (and must be checked), and its feature struct belongs in the device
    // pNext chain. All three follow from this one decision.
    bool portabilitySubset{false};
};

// `availableLayers` / `availableExtensions` are what the loader reports; `windowExtensions` are the
// WSI extensions the platform window requires (empty for a headless device).
[[nodiscard]] InstanceCapabilityPlan
planInstanceCapabilities(std::span<const std::string> availableExtensions,
                         std::span<const std::string> availableLayers,
                         std::span<const std::string> windowExtensions, bool wantValidation);

// `requiredExtensions` are hard rejection criteria; the optional set is fixed (the portability
// subset) and is added only when advertised.
[[nodiscard]] DeviceCapabilityPlan
planDeviceCapabilities(std::span<const std::string> availableExtensions,
                       std::span<const std::string> requiredExtensions);

// Borrowed views of `names` for the Vulkan call — `names` must outlive the returned vector.
[[nodiscard]] std::vector<const char*> toCStrings(std::span<const std::string> names);

} // namespace fire_engine
