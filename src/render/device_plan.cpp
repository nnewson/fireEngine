#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include <fire_engine/render/device_plan.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] bool contains(std::span<const std::string> haystack, std::string_view needle)
{
    return std::ranges::find(haystack, needle) != haystack.end();
}

// Device extensions enabled iff advertised. Deliberately tiny: an extension belongs here only when
// requesting it from an implementation that lacks it is wrong (rather than merely unsupported).
constexpr std::array kOptionalDeviceExtensions{kPortabilitySubsetExtension};

} // namespace

InstanceCapabilityPlan planInstanceCapabilities(std::span<const std::string> availableExtensions,
                                                std::span<const std::string> availableLayers,
                                                std::span<const std::string> windowExtensions,
                                                bool wantValidation)
{
    InstanceCapabilityPlan plan;
    plan.extensions.assign(windowExtensions.begin(), windowExtensions.end());

    plan.portabilityEnumeration = contains(availableExtensions, kPortabilityEnumerationExtension);
    if (plan.portabilityEnumeration)
    {
        plan.extensions.emplace_back(kPortabilityEnumerationExtension);
    }

    plan.validation = wantValidation && contains(availableLayers, kValidationLayer);
    return plan;
}

DeviceCapabilityPlan planDeviceCapabilities(std::span<const std::string> availableExtensions,
                                            std::span<const std::string> requiredExtensions)
{
    DeviceCapabilityPlan plan;
    for (const std::string& required : requiredExtensions)
    {
        if (contains(availableExtensions, required))
        {
            plan.extensions.push_back(required);
        }
        else
        {
            plan.missingRequired.push_back(required);
        }
    }

    for (const char* optional : kOptionalDeviceExtensions)
    {
        // `contains(plan.extensions, …)` guards the case where an optional extension is ALSO named
        // as required: enabling the same name twice is sloppy at best, and a caller reading the
        // list back (as the portability decision below does) must not see it as two decisions.
        if (contains(availableExtensions, optional) && !contains(plan.extensions, optional))
        {
            plan.extensions.emplace_back(optional);
        }
    }

    plan.portabilitySubset = contains(plan.extensions, kPortabilitySubsetExtension);
    return plan;
}

std::vector<const char*> toCStrings(std::span<const std::string> names)
{
    std::vector<const char*> pointers;
    pointers.reserve(names.size());
    for (const std::string& name : names)
    {
        pointers.push_back(name.c_str());
    }
    return pointers;
}

} // namespace fire_engine
