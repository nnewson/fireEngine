#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/render/device.hpp>

namespace fire_engine
{

#ifdef NDEBUG
constexpr bool enableValidation = false;
#else
constexpr bool enableValidation = true;
#endif

namespace
{

// Pipeline cache blob, kept next to the working directory so it survives between runs.
constexpr const char* kPipelineCacheFile = "pipeline_cache.bin";

// Whole-file read as bytes; empty on any error (missing file, etc.).
[[nodiscard]] std::vector<char> readBinaryFile(const char* path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        return {};
    }
    const std::streamsize size = in.tellg();
    if (size <= 0)
    {
        return {};
    }
    std::vector<char> data(static_cast<std::size_t>(size));
    in.seekg(0);
    if (!in.read(data.data(), size))
    {
        return {};
    }
    return data;
}

// A persisted pipeline cache is only valid on the same driver + GPU it was written on. Validate
// the VkPipelineCacheHeaderVersionOne header (version, vendor/device IDs, cache UUID) against the
// current device before trusting it; a driver or GPU change invalidates it and we start cold.
[[nodiscard]] bool cacheMatchesDevice(std::span<const char> data,
                                      const vk::PhysicalDeviceProperties& props)
{
    constexpr std::size_t headerSize = 16 + VK_UUID_SIZE; // 4×u32 + 16-byte UUID
    if (data.size() < headerSize)
    {
        return false;
    }
    std::uint32_t headerVersion = 0;
    std::uint32_t vendorID = 0;
    std::uint32_t deviceID = 0;
    std::memcpy(&headerVersion, data.data() + 4, sizeof(headerVersion));
    std::memcpy(&vendorID, data.data() + 8, sizeof(vendorID));
    std::memcpy(&deviceID, data.data() + 12, sizeof(deviceID));
    return headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE && vendorID == props.vendorID &&
           deviceID == props.deviceID &&
           std::memcmp(data.data() + 16, props.pipelineCacheUUID.data(), VK_UUID_SIZE) == 0;
}

} // namespace

// The device extensions the engine requires are built per-mode by Device::requiredDeviceExtensions
// (swapchain is windowed-only). Optional, enable-iff-advertised extensions and the layer/extension
// decisions live in the pure planner (`render/device_plan.hpp`), which suitability and device
// creation share so the coupled choices can't drift.

namespace
{

// THE baseline. The renderer calls promoted-to-core 1.4 entry points (vkCmdPushDescriptorSet), VMA
// is told the device is 1.4, and the 1.4 feature struct is queried — all of which require the
// PHYSICAL DEVICE, not just the instance, to be 1.4. Enforced in missingDeviceCapabilities.
constexpr std::uint32_t kMinimumDeviceApiVersion = VK_API_VERSION_1_4;

// Loader/driver enumeration -> plain names for the planner.
[[nodiscard]] std::vector<std::string>
extensionNames(const std::vector<vk::ExtensionProperties>& properties)
{
    std::vector<std::string> names;
    names.reserve(properties.size());
    for (const vk::ExtensionProperties& e : properties)
    {
        names.emplace_back(e.extensionName.data());
    }
    return names;
}

[[nodiscard]] std::vector<std::string>
layerNames(const std::vector<vk::LayerProperties>& properties)
{
    std::vector<std::string> names;
    names.reserve(properties.size());
    for (const vk::LayerProperties& l : properties)
    {
        names.emplace_back(l.layerName.data());
    }
    return names;
}

[[nodiscard]] std::string joinNames(std::span<const std::string> names)
{
    std::string joined;
    for (const std::string& name : names)
    {
        joined += joined.empty() ? "" : ", ";
        joined += name;
    }
    return joined;
}

[[nodiscard]] std::string versionString(std::uint32_t version)
{
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

// One required device feature: the `vk::Bool32` bit inside a Vulkan feature struct, plus a name for
// diagnostics. A pointer-to-member so the SAME entry drives both directions — enabling the bit on
// the device (createLogicalDevice) and verifying it is advertised (missingDeviceCapabilities) —
// which is the whole point: the "check" and the "enable" list can no longer drift apart.
template <typename FeatureStruct>
struct RequiredFeature
{
    vk::Bool32 FeatureStruct::* bit;
    const char* name;
};

// THE capability description — the single source of truth for the features this renderer requires,
// one table per feature struct. Add a feature here and both device selection and device creation
// pick it up. (Descriptor-indexing *limits* are a properties check, handled separately below; they
// have no enable counterpart.)
using Feature10 = RequiredFeature<vk::PhysicalDeviceFeatures>;
constexpr std::array kRequiredFeatures10{
    Feature10{&vk::PhysicalDeviceFeatures::samplerAnisotropy, "samplerAnisotropy"},
    Feature10{&vk::PhysicalDeviceFeatures::imageCubeArray,
              "imageCubeArray"}, // point-shadow cubemaps
    Feature10{&vk::PhysicalDeviceFeatures::independentBlend,
              "independentBlend"}, // per-attachment blend (colour vs velocity)
};

using Feature12 = RequiredFeature<vk::PhysicalDeviceVulkan12Features>;
constexpr std::array kRequiredFeatures12{
    Feature12{&vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
              "bufferDeviceAddress"}, // soft-body compute buffer pointers
    Feature12{&vk::PhysicalDeviceVulkan12Features::descriptorIndexing,
              "descriptorIndexing"}, // bindless materials
    Feature12{&vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
              "timelineSemaphore"}, // frame pacing
    Feature12{&vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray,
              "runtimeDescriptorArray"},
    Feature12{&vk::PhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing,
              "shaderSampledImageArrayNonUniformIndexing"},
    Feature12{&vk::PhysicalDeviceVulkan12Features::descriptorBindingSampledImageUpdateAfterBind,
              "descriptorBindingSampledImageUpdateAfterBind"},
    Feature12{&vk::PhysicalDeviceVulkan12Features::descriptorBindingPartiallyBound,
              "descriptorBindingPartiallyBound"},
    // (No descriptorBindingVariableDescriptorCount: the bindless array is a fixed
    // kMaxBindlessTextures slots with partiallyBound, not an eVariableDescriptorCount binding.)
};

using Feature13 = RequiredFeature<vk::PhysicalDeviceVulkan13Features>;
constexpr std::array kRequiredFeatures13{
    Feature13{&vk::PhysicalDeviceVulkan13Features::synchronization2, "synchronization2"},
    Feature13{&vk::PhysicalDeviceVulkan13Features::dynamicRendering, "dynamicRendering"},
};

using Feature14 = RequiredFeature<vk::PhysicalDeviceVulkan14Features>;
constexpr std::array kRequiredFeatures14{
    // Per-object set 0 is pushed inline (no per-frame descriptor set). Promoted to core in 1.4,
    // where it is an opt-in FEATURE rather than an extension — so this is the request that makes
    // vkCmdPushDescriptorSet legal, and VK_KHR_push_descriptor is deliberately NOT required
    // (a conformant 1.4 device need not still advertise the promoted extension).
    Feature14{&vk::PhysicalDeviceVulkan14Features::pushDescriptor, "pushDescriptor"},
};

using FeaturePortability = RequiredFeature<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>;
constexpr std::array kRequiredFeaturesPortability{
    FeaturePortability{&vk::PhysicalDevicePortabilitySubsetFeaturesKHR::mutableComparisonSamplers,
                       "mutableComparisonSamplers (portability subset)"}, // sampler2DShadow / PCF
};

// Set every bit in `table` on `target` (device-creation enable path).
template <typename FeatureStruct>
void enableFeatures(FeatureStruct& target, std::span<const RequiredFeature<FeatureStruct>> table)
{
    for (const RequiredFeature<FeatureStruct>& f : table)
    {
        target.*(f.bit) = vk::True;
    }
}

// Append the name of every bit in `table` that `supported` does not advertise (suitability check).
template <typename FeatureStruct>
void collectMissingFeatures(const FeatureStruct& supported,
                            std::span<const RequiredFeature<FeatureStruct>> table,
                            std::vector<std::string>& missing)
{
    for (const RequiredFeature<FeatureStruct>& f : table)
    {
        if (supported.*(f.bit) == vk::False)
        {
            missing.emplace_back(f.name);
        }
    }
}

// Verified during device selection so an unsupported GPU is rejected up front with a precise
// reason, instead of passing suitability and then failing an opaque Vulkan call inside device or
// pipeline creation. Returns nullopt when the device satisfies everything, or a comma-joined list
// of the missing capabilities. Driven by the kRequiredFeatures* tables above (shared with the
// enable path).
[[nodiscard]] std::optional<std::string>
missingDeviceCapabilities(const vk::raii::PhysicalDevice& d, bool portabilitySubset)
{
    // The version gate comes FIRST and returns early: every feature struct below is only written
    // by a driver that knows it, so querying a 1.4 struct on a 1.3 device silently reads back
    // zeros and would report a pile of "missing features" instead of the one real reason. (macOS
    // 26 makes this concrete — the same machine exposes MoltenVK at 1.4 and Apple's conformant
    // native driver at 1.3, and only the former can service this renderer.)
    const std::uint32_t apiVersion = d.getProperties().apiVersion;
    if (apiVersion < kMinimumDeviceApiVersion)
    {
        return "Vulkan " + versionString(kMinimumDeviceApiVersion) + " (device reports " +
               versionString(apiVersion) + ")";
    }

    const auto features =
        d.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceVulkan14Features>();
    const vk::PhysicalDeviceFeatures& f10 = features.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& f12 = features.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& f13 = features.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& f14 = features.get<vk::PhysicalDeviceVulkan14Features>();

    std::vector<std::string> missing;
    collectMissingFeatures<vk::PhysicalDeviceFeatures>(f10, kRequiredFeatures10, missing);
    collectMissingFeatures<vk::PhysicalDeviceVulkan12Features>(f12, kRequiredFeatures12, missing);
    collectMissingFeatures<vk::PhysicalDeviceVulkan13Features>(f13, kRequiredFeatures13, missing);
    collectMissingFeatures<vk::PhysicalDeviceVulkan14Features>(f14, kRequiredFeatures14, missing);

    // The portability-subset feature bits describe which parts of core Vulkan a NON-conformant
    // implementation actually supports, so they are only meaningful — and only written by the
    // driver — when the device advertises the extension. On a conformant device the extension is
    // absent and that functionality is unconditionally present, so there is nothing to check;
    // querying the struct anyway would read back the all-false struct the driver never touched
    // and reject every conformant GPU.
    if (portabilitySubset)
    {
        const auto portabilityFeatures =
            d.getFeatures2<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDevicePortabilitySubsetFeaturesKHR>();
        collectMissingFeatures<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>(
            portabilityFeatures.get<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>(),
            kRequiredFeaturesPortability, missing);
    }

    // The global bindless texture array (forward set 2) is kMaxBindlessTextures update-after-bind
    // combined-image-samplers; each counts against both the sampler and the sampled-image
    // update-after-bind limits, per stage and per set (set 1 adds a few more samplers on top).
    // Reject a device that can't hold the array rather than overflow the descriptor set at bind
    // time.
    const auto props = d.getProperties2<vk::PhysicalDeviceProperties2,
                                        vk::PhysicalDeviceDescriptorIndexingProperties>();
    const auto& di = props.get<vk::PhysicalDeviceDescriptorIndexingProperties>();
    auto requireLimit = [&missing](uint32_t have, uint32_t want, const char* name)
    {
        if (have < want)
        {
            missing.push_back(std::string(name) + " (" + std::to_string(have) + " < " +
                              std::to_string(want) + " required)");
        }
    };
    requireLimit(di.maxPerStageDescriptorUpdateAfterBindSamplers, kMaxBindlessTextures,
                 "maxPerStageDescriptorUpdateAfterBindSamplers");
    requireLimit(di.maxPerStageDescriptorUpdateAfterBindSampledImages, kMaxBindlessTextures,
                 "maxPerStageDescriptorUpdateAfterBindSampledImages");
    requireLimit(di.maxDescriptorSetUpdateAfterBindSamplers, kMaxBindlessTextures,
                 "maxDescriptorSetUpdateAfterBindSamplers");
    requireLimit(di.maxDescriptorSetUpdateAfterBindSampledImages, kMaxBindlessTextures,
                 "maxDescriptorSetUpdateAfterBindSampledImages");

    if (missing.empty())
    {
        return std::nullopt;
    }
    std::string reason = missing.front();
    for (std::size_t i = 1; i < missing.size(); ++i)
    {
        reason += ", ";
        reason += missing[i];
    }
    return reason;
}

} // namespace

Device::Device(const Window& window, bool requireValidation)
    : requireValidation_(requireValidation)
{
    createInstance();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
    createPipelineCache();
}

Device::Device(Mode mode)
    : mode_(mode)
{
    // Surface-free path: no createSurface, and createInstance / isDeviceSuitable /
    // findQueueFamilies / createLogicalDevice all branch on headless() to skip WSI extensions,
    // surface + swapchain checks, and the present queue.
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
    createPipelineCache();
}

Device Device::headlessCompute()
{
    return Device(Mode::HeadlessCompute);
}

std::vector<std::string> Device::requiredDeviceExtensions() const
{
    // Push descriptors are a 1.4 core FEATURE (kRequiredFeatures14), not an extension — the
    // promoted VK_KHR_push_descriptor is deliberately not required. The swapchain extension is
    // windowed-only: a headless compute device must not require or enable it.
    std::vector<std::string> exts;
    if (!headless())
    {
        exts.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    return exts;
}

DeviceCapabilityPlan Device::deviceCapabilityPlan(const vk::raii::PhysicalDevice& d) const
{
    return planDeviceCapabilities(extensionNames(d.enumerateDeviceExtensionProperties()),
                                  requiredDeviceExtensions());
}

void Device::createInstance()
{
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "FireEngine",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion = vk::ApiVersion14,
    };

    // Windowed needs GLFW's WSI instance extensions (surface creation); a headless compute device
    // creates no surface, so it enables none of them — only the portability enumeration MoltenVK
    // needs. (This also means the headless path never touches GLFW.)
    std::vector<std::string> windowExtensions;
    if (!headless())
    {
        for (const char* ext : Window::requiredVulkanExtensions())
        {
            windowExtensions.emplace_back(ext);
        }
    }

    if (enableValidation && log::enabled(log::Level::Debug, log::category::render))
    {
        printValidationInfo();
    }

    // Layers and platform extensions are enable-iff-available: requesting an absent one is a hard
    // VK_ERROR_LAYER_NOT_PRESENT / VK_ERROR_EXTENSION_NOT_PRESENT at instance creation, not a
    // degraded mode. The decisions are the pure planner's; this function only enumerates and obeys.
    const InstanceCapabilityPlan plan =
        planInstanceCapabilities(extensionNames(context_.enumerateInstanceExtensionProperties()),
                                 layerNames(context_.enumerateInstanceLayerProperties()),
                                 windowExtensions, enableValidation);

    // The validation layer ships with the Vulkan SDK (or a distro package), NOT with the loader
    // vcpkg provides. A miss is a warning by default (a fresh machine must still be able to run)
    // and a hard failure under --require-validation, for runs that must not silently degrade.
    if (requireValidation_ && !plan.validation)
    {
        // Not gated on enableValidation, so this also catches an NDEBUG build where the layer is
        // compiled out entirely and no warning would otherwise be emitted — the exact hole that
        // lets an unvalidated run be mistaken for a clean one.
        throw std::runtime_error(
            enableValidation
                ? std::string("--require-validation: validation layer ") + kValidationLayer +
                      " is not installed"
                : std::string("--require-validation: this build has validation compiled out "
                              "(NDEBUG); use a Dev build"));
    }
    if (enableValidation && !plan.validation)
    {
        log::warn(log::category::render,
                  "Vulkan validation NOT INSTALLED ({}) — running WITHOUT validation; install the "
                  "Vulkan SDK (on Linux the tarball, then source setup-env.sh) to restore VUID "
                  "checking",
                  kValidationLayer);
    }

    const std::vector<const char*> layers =
        plan.validation ? std::vector<const char*>{kValidationLayer} : std::vector<const char*>{};
    const std::vector<const char*> exts = toCStrings(plan.extensions);
    vk::InstanceCreateInfo ci{
        .flags = plan.portabilityEnumeration ? vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR
                                             : vk::InstanceCreateFlags{},
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(exts.size()),
        .ppEnabledExtensionNames = exts.data(),
    };

    instance_ = vk::raii::Instance(context_, ci);

    // Reported AFTER the instance exists, so the line means "validation is live on a created
    // instance" and not merely "we intended to ask for it". The render smoke asserts this token
    // (CLAUDE.md § Build) — logging it before construction would let a later startup failure still
    // produce the expected greps.
    if (plan.validation)
    {
        log::info(log::category::render, "Vulkan validation enabled ({})", kValidationLayer);
    }
}

void Device::printValidationInfo() const
{
    auto layersAvailable = context_.enumerateInstanceLayerProperties();
    std::string layers{"Available layers:"};
    for (const auto& layer : layersAvailable)
    {
        layers += "\n\t";
        layers += layer.layerName.data();
    }
    log::debug(log::category::render, "{}", layers);

    auto extensions = context_.enumerateInstanceExtensionProperties();
    std::string instanceExtensions{"Available instance extensions:"};
    for (const auto& ext : extensions)
    {
        instanceExtensions += "\n\t";
        instanceExtensions += ext.extensionName.data();
    }
    log::debug(log::category::render, "{}", instanceExtensions);
}

void Device::createSurface(const Window& window)
{
    surface_ = window.createVulkanSurface(instance_);
}

void Device::pickPhysicalDevice()
{
    auto devs = instance_.enumeratePhysicalDevices();
    for (auto& d : devs)
    {
        if (isDeviceSuitable(d))
        {
            physDevice_ = std::move(d);
            return;
        }
    }
    throw std::runtime_error("no suitable GPU found");
}

bool Device::isDeviceSuitable(const vk::raii::PhysicalDevice& d)
{
    auto [gf, pf] = findQueueFamilies(d);
    // Headless requires only a graphics+compute family; windowed needs a present family too.
    if (!gf.has_value() || (!headless() && !pf.has_value()))
    {
        return false;
    }

    auto avail = d.enumerateDeviceExtensionProperties();
    if (enableValidation && log::enabled(log::Level::Debug, log::category::render))
    {
        std::string deviceExtensionsList{"Available device extensions:"};
        for (const auto& extension : avail)
        {
            deviceExtensionsList += "\n\t";
            deviceExtensionsList += extension.extensionName.data();
        }
        log::debug(log::category::render, "{}", deviceExtensionsList);
    }

    // The SAME planner createLogicalDevice uses, on the same inputs — so what suitability accepts
    // and what device creation enables are one decision, not two that happen to agree.
    const DeviceCapabilityPlan plan = deviceCapabilityPlan(d);
    if (!plan.missingRequired.empty())
    {
        // Named, like the feature check below: "no suitable GPU found" with no reason is the
        // hardest failure to diagnose remotely, and a missing extension is the likeliest way a
        // machine with a perfectly capable GPU gets rejected.
        log::warn(log::category::render, "GPU '{}' unsuitable: missing device extension(s) {}",
                  d.getProperties().deviceName.data(), joinNames(plan.missingRequired));
        return false;
    }

    // Surface capability is a windowed-only concern; a headless compute device has no surface.
    if (!headless())
    {
        auto fmts = d.getSurfaceFormatsKHR(*surface_);
        auto modes = d.getSurfacePresentModesKHR(*surface_);
        if (fmts.empty() || modes.empty())
        {
            return false;
        }
    }

    // Verify every feature/limit createLogicalDevice enables is actually advertised, so a device
    // that clears queues+extensions+swapchain but lacks (say) update-after-bind is rejected here
    // with a named reason rather than crashing an opaque call during device/pipeline creation.
    if (auto reason = missingDeviceCapabilities(d, plan.portabilitySubset))
    {
        log::warn(log::category::render, "GPU '{}' unsuitable: missing {}",
                  d.getProperties().deviceName.data(), *reason);
        return false;
    }
    return true;
}

std::pair<std::optional<uint32_t>, std::optional<uint32_t>>
Device::findQueueFamilies(const vk::raii::PhysicalDevice& d)
{
    auto families = d.getQueueFamilyProperties();
    std::optional<uint32_t> gf, pf;
    for (uint32_t i = 0; i < families.size(); ++i)
    {
        const bool graphics =
            static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eGraphics);
        if (headless())
        {
            // One family that can do BOTH graphics and compute — the same-queue path the renderer
            // already dispatches compute on. NO present family: a headless device has no surface,
            // so `pf` stays absent (present is never created or used).
            const bool compute =
                static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eCompute);
            if (graphics && compute)
            {
                gf = i;
                break;
            }
            continue;
        }
        if (graphics)
        {
            gf = i;
        }
        if (d.getSurfaceSupportKHR(i, *surface_) != vk::False)
        {
            pf = i;
        }
        if (gf && pf)
        {
            break;
        }
    }
    return {gf, pf};
}

void Device::createLogicalDevice()
{
    auto [gf, pf] = findQueueFamilies(physDevice_);
    if (!gf.has_value() || (!headless() && !pf.has_value()))
    {
        throw std::runtime_error(
            headless() ? "GPU lacks a graphics+compute queue family (headless compute)"
                       : "GPU lacks a required graphics and/or present queue family");
    }
    graphicsFamily_ = *gf;

    // Headless creates ONLY the graphics+compute queue; there is no present family or queue (they
    // stay default/null — `presentQueue()` is meaningless and never called headless). Windowed
    // creates both (deduped when they share a family). `pf` is engaged exactly when windowed (see
    // findQueueFamilies), so gating on `pf.has_value()` is both equivalent to `!headless()` AND
    // makes the deref checked for the static analyser.
    std::set<uint32_t> uniqueFamilies = {graphicsFamily_};
    if (pf.has_value())
    {
        presentFamily_ = pf;
        uniqueFamilies.insert(*pf);
    }
    std::vector<vk::DeviceQueueCreateInfo> qcis;
    qcis.reserve(uniqueFamilies.size());
    float prio = 1.0f;
    for (uint32_t fam : uniqueFamilies)
    {
        qcis.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = fam,
            .queueCount = 1,
            .pQueuePriorities = &prio,
        });
    }

    // Enable exactly the features the kRequiredFeatures* tables describe — the same tables
    // missingDeviceCapabilities checked during selection, so this device is known to support the
    // whole set (no per-feature re-query/throw here) and the enable/check lists cannot drift. The
    // structs are chained by pNext below; the tables only decide which bits are set. (What each
    // feature is for lives in a comment beside its table entry.)
    vk::PhysicalDeviceFeatures features{};
    enableFeatures<vk::PhysicalDeviceFeatures>(features, kRequiredFeatures10);

    vk::PhysicalDeviceVulkan14Features features14{};
    enableFeatures<vk::PhysicalDeviceVulkan14Features>(features14, kRequiredFeatures14);

    vk::PhysicalDeviceVulkan13Features features13{};
    enableFeatures<vk::PhysicalDeviceVulkan13Features>(features13, kRequiredFeatures13);
    features14.pNext = &features13;

    vk::PhysicalDeviceVulkan12Features features12{};
    enableFeatures<vk::PhysicalDeviceVulkan12Features>(features12, kRequiredFeatures12);
    features13.pNext = &features12;

    // Same planner, same inputs as the suitability check — one decision. The portability struct is
    // chained ONLY when its extension is enabled: it is defined by that extension, so passing it to
    // a conformant driver that never advertised it is invalid (and the bits would be meaningless
    // there — the functionality is unconditionally present).
    const DeviceCapabilityPlan plan = deviceCapabilityPlan(physDevice_);
    const std::vector<const char*> enabledExtensions = toCStrings(plan.extensions);

    vk::PhysicalDevicePortabilitySubsetFeaturesKHR portability{};
    enableFeatures<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>(portability,
                                                                   kRequiredFeaturesPortability);
    portability.pNext = &features14;

    vk::DeviceCreateInfo ci{
        .pNext = plan.portabilitySubset ? static_cast<const void*>(&portability) : &features14,
        .queueCreateInfoCount = static_cast<uint32_t>(qcis.size()),
        .pQueueCreateInfos = qcis.data(),
        .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
        .pEnabledFeatures = &features,
    };

    device_ = vk::raii::Device(physDevice_, ci);
    graphicsQueue_ = device_.getQueue(graphicsFamily_, 0);
    // presentFamily_ is engaged exactly when windowed; gating on has_value() (rather than
    // !headless()) keeps the deref checked for the static analyser.
    if (presentFamily_.has_value())
    {
        presentQueue_ = device_.getQueue(*presentFamily_, 0);
    }
}

void Device::createAllocator()
{
    VmaAllocatorCreateInfo ci{};
    ci.instance = *instance_;
    ci.physicalDevice = *physDevice_;
    ci.device = *device_;
    ci.vulkanApiVersion = VK_API_VERSION_1_4; // matches the instance's requested apiVersion
    // The soft-body solver takes buffer device addresses; this lets VMA tag those allocations
    // with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT automatically (the device feature is already on).
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    VmaAllocator allocator = nullptr;
    if (vmaCreateAllocator(&ci, &allocator) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create VMA allocator");
    }
    allocator_ = VmaAllocatorHandle{allocator};
}

void Device::createPipelineCache()
{
    // Seed the cache from disk so the driver's shader compilation is paid once *across* runs,
    // not on every cold start. On MoltenVK the Metal compile is deferred to a pipeline's first
    // use, so a cold cache can stall mid-run the first time a pipeline draws; a warm cache from
    // a previous run avoids that. The blob is GPU/driver-specific — if it doesn't match this
    // device we discard it, start cold, and overwrite it on shutdown.
    std::vector<char> data = readBinaryFile(kPipelineCacheFile);
    if (!cacheMatchesDevice(data, physDevice_.getProperties()))
    {
        data.clear();
    }
    const vk::PipelineCacheCreateInfo ci{
        .initialDataSize = data.size(),
        .pInitialData = data.empty() ? nullptr : data.data(),
    };
    pipelineCache_ = vk::raii::PipelineCache(device_, ci);
}

void Device::savePipelineCache() const noexcept
{
    // Best-effort: a failed cache write must never crash shutdown.
    try
    {
        if (*pipelineCache_ == nullptr) // moved-from (or never created)
        {
            return;
        }
        const std::vector<uint8_t> data = pipelineCache_.getData();
        if (data.empty())
        {
            return;
        }
        std::ofstream out(kPipelineCacheFile, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }
    catch (...)
    {
        // Saving the pipeline cache is a best-effort optimisation; a failure just means the next
        // run recompiles. Swallow it (this runs from ~Device(), so nothing may escape) but note it.
        log::debug(log::category::render, "failed to save the pipeline cache (non-fatal)");
    }
}

Device::~Device()
{
    savePipelineCache();
}

UniqueVmaBuffer Device::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                     vk::MemoryPropertyFlags props) const
{
    const vk::BufferCreateInfo ci{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const VkBufferCreateInfo& cci = ci;

    const bool hostVisible = static_cast<bool>(props & vk::MemoryPropertyFlagBits::eHostVisible);
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    // Preserve the caller's exact memory requirement (host-coherent, device-local, …); AUTO
    // then picks a matching type. Host-visible buffers are kept persistently mapped — random
    // host access so arbitrary-offset writes (e.g. the materials SSBO) stay cache-coherent.
    aci.requiredFlags = static_cast<VkMemoryPropertyFlags>(props);
    if (hostVisible)
    {
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator_.get(), &cci, &aci, &buffer, &allocation, &info) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer via VMA");
    }
    return UniqueVmaBuffer{allocator_.get(), buffer, allocation,
                           hostVisible ? info.pMappedData : nullptr,
                           static_cast<std::size_t>(size)};
}

} // namespace fire_engine
