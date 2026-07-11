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

constexpr std::array validationLayers{"VK_LAYER_KHRONOS_validation"};
constexpr std::array deviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset", // required on macOS/MoltenVK
    // Per-object forward set 0 is pushed inline (vkCmdPushDescriptorSetKHR)
    // instead of allocating a descriptor set per object/frame. Core in 1.4;
    // enabled as an extension here. MoltenVK advertises it (pushDescriptor=true).
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
};

namespace
{

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
missingDeviceCapabilities(const vk::raii::PhysicalDevice& d)
{
    const auto features =
        d.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDevicePortabilitySubsetFeaturesKHR>();
    const vk::PhysicalDeviceFeatures& f10 = features.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& f12 = features.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& f13 = features.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& fp = features.get<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>();

    std::vector<std::string> missing;
    collectMissingFeatures<vk::PhysicalDeviceFeatures>(f10, kRequiredFeatures10, missing);
    collectMissingFeatures<vk::PhysicalDeviceVulkan12Features>(f12, kRequiredFeatures12, missing);
    collectMissingFeatures<vk::PhysicalDeviceVulkan13Features>(f13, kRequiredFeatures13, missing);
    collectMissingFeatures<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>(
        fp, kRequiredFeaturesPortability, missing);

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

Device::Device(const Window& window)
{
    createInstance();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
    createPipelineCache();
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

    auto exts = Window::requiredVulkanExtensions();
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    if (enableValidation && log::enabled(log::Level::Debug, log::category::render))
    {
        printValidationInfo();
    }

    vk::InstanceCreateInfo ci{
        .flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = enableValidation ? static_cast<uint32_t>(validationLayers.size()) : 0u,
        .ppEnabledLayerNames = enableValidation ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(exts.size()),
        .ppEnabledExtensionNames = exts.data(),
    };

    instance_ = vk::raii::Instance(context_, ci);
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
    if (!gf.has_value() || !pf.has_value())
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

    std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());
    for (auto& e : avail)
    {
        required.erase(e.extensionName);
    }
    if (!required.empty())
    {
        return false;
    }

    auto fmts = d.getSurfaceFormatsKHR(*surface_);
    auto modes = d.getSurfacePresentModesKHR(*surface_);
    if (fmts.empty() || modes.empty())
    {
        return false;
    }

    // Verify every feature/limit createLogicalDevice enables is actually advertised, so a device
    // that clears queues+extensions+swapchain but lacks (say) update-after-bind is rejected here
    // with a named reason rather than crashing an opaque call during device/pipeline creation.
    if (auto reason = missingDeviceCapabilities(d))
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
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
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
    if (!gf.has_value() || !pf.has_value())
    {
        throw std::runtime_error("GPU lacks a required graphics and/or present queue family");
    }
    graphicsFamily_ = *gf;
    presentFamily_ = *pf;

    std::set<uint32_t> uniqueFamilies = {graphicsFamily_, presentFamily_};
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

    vk::PhysicalDeviceVulkan13Features features13{};
    enableFeatures<vk::PhysicalDeviceVulkan13Features>(features13, kRequiredFeatures13);

    vk::PhysicalDeviceVulkan12Features features12{};
    enableFeatures<vk::PhysicalDeviceVulkan12Features>(features12, kRequiredFeatures12);
    features13.pNext = &features12;

    vk::PhysicalDevicePortabilitySubsetFeaturesKHR portability{};
    enableFeatures<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>(portability,
                                                                   kRequiredFeaturesPortability);
    portability.pNext = &features13;

    vk::DeviceCreateInfo ci{
        .pNext = &portability,
        .queueCreateInfoCount = static_cast<uint32_t>(qcis.size()),
        .pQueueCreateInfos = qcis.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &features,
    };

    device_ = vk::raii::Device(physDevice_, ci);
    graphicsQueue_ = device_.getQueue(graphicsFamily_, 0);
    presentQueue_ = device_.getQueue(presentFamily_, 0);
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
