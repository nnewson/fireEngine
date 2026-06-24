#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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
[[nodiscard]] bool cacheMatchesDevice(const std::vector<char>& data,
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

Device::Device(const Window& window)
{
    createInstance();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
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

    if (enableValidation)
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
    std::cout << "Available layers:\n";
    for (const auto& layer : layersAvailable)
    {
        std::cout << '\t' << layer.layerName << '\n';
    }
    std::cout << '\n';

    auto extensions = context_.enumerateInstanceExtensionProperties();
    std::cout << "Available extensions:\n";
    for (const auto& ext : extensions)
    {
        std::cout << '\t' << ext.extensionName << '\n';
    }
    std::cout << '\n';
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
    if (enableValidation)
    {
        std::cout << "Available extensions:\n";
        for (const auto& extension : avail)
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
        std::cout << '\n';
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
    return !fmts.empty() && !modes.empty();
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
        if (d.getSurfaceSupportKHR(i, *surface_))
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
    graphicsFamily_ = gf.value();
    presentFamily_ = pf.value();

    std::set<uint32_t> uniqueFamilies = {graphicsFamily_, presentFamily_};
    std::vector<vk::DeviceQueueCreateInfo> qcis;
    float prio = 1.0f;
    for (uint32_t fam : uniqueFamilies)
    {
        qcis.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = fam,
            .queueCount = 1,
            .pQueuePriorities = &prio,
        });
    }

    auto supported = physDevice_.getFeatures();
    if (!supported.imageCubeArray)
    {
        throw std::runtime_error(
            "GPU does not support imageCubeArray (required for point shadow maps)");
    }

    // synchronization2 and dynamicRendering are core in Vulkan 1.3 and mandatory
    // in 1.4, but must still be enabled explicitly before the matching APIs are
    // legal to use. Verify the driver advertises them before requesting them.
    auto supported13 =
        physDevice_.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>()
            .get<vk::PhysicalDeviceVulkan13Features>();
    if (!supported13.synchronization2)
    {
        throw std::runtime_error("GPU does not support synchronization2");
    }
    if (!supported13.dynamicRendering)
    {
        throw std::runtime_error("GPU does not support dynamicRendering");
    }

    vk::PhysicalDeviceFeatures features{};
    features.samplerAnisotropy = vk::True;
    // Point light shadow maps use samplerCubeArrayShadow over a cubemap-array
    // depth image. Requires the imageCubeArray feature.
    features.imageCubeArray = vk::True;
    // The forward/transmission passes render colour + a TAA velocity attachment
    // with different blend state per attachment (colour may alpha-blend; the
    // velocity attachment never blends). Per-attachment blend needs this.
    features.independentBlend = vk::True;

    // synchronization2: barriers/submits use the *2 APIs. dynamicRendering:
    // rendering without VkRenderPass/VkFramebuffer objects.
    vk::PhysicalDeviceVulkan13Features features13{};
    features13.synchronization2 = vk::True;
    features13.dynamicRendering = vk::True;

    // bufferDeviceAddress (core 1.2): the soft-body solver passes its buffers to
    // compute as 64-bit GPU pointers (GL_EXT_buffer_reference) instead of
    // descriptor sets.
    //
    // descriptorIndexing (core 1.2): bindless materials. The forward shader indexes
    // one global sampler2D[] texture array and a global materials[] SSBO, so the
    // array must allow runtime/non-uniform indexing, partially-bound slots (the
    // array is sparse — indexed by texture handle), and update-after-bind (textures
    // are written into the array as they load, after the set is first bound).
    vk::PhysicalDeviceVulkan12Features features12{};
    features12.bufferDeviceAddress = vk::True;
    features12.descriptorIndexing = vk::True;
    // timelineSemaphore (core 1.2): frame pacing uses one monotonic timeline
    // semaphore for CPU↔GPU sync instead of per-frame binary fences. The
    // swapchain acquire/present semaphores stay binary (WSI doesn't accept
    // timeline semaphores).
    features12.timelineSemaphore = vk::True;
    features12.runtimeDescriptorArray = vk::True;
    features12.shaderSampledImageArrayNonUniformIndexing = vk::True;
    features12.descriptorBindingSampledImageUpdateAfterBind = vk::True;
    features12.descriptorBindingPartiallyBound = vk::True;
    features12.descriptorBindingVariableDescriptorCount = vk::True;
    features13.pNext = &features12;

    // Shadow mapping uses sampler2DShadow (hardware PCF), which requires
    // compareEnable=VK_TRUE on the sampler. MoltenVK gates that behind the
    // portability-subset feature mutableComparisonSamplers — enable it here.
    vk::PhysicalDevicePortabilitySubsetFeaturesKHR portability{};
    portability.mutableComparisonSamplers = vk::True;
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
    }
}

Device::~Device()
{
    savePipelineCache();
}

uint32_t Device::findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) const
{
    auto mem = physDevice_.getMemoryProperties();
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
    {
        if ((filter & (1 << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
Device::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                     vk::MemoryPropertyFlags props) const
{
    vk::BufferCreateInfo ci{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    vk::raii::Buffer buf(device_, ci);

    auto req = buf.getMemoryRequirements();
    vk::MemoryAllocateInfo ai{
        .allocationSize = req.size,
        .memoryTypeIndex = findMemoryType(req.memoryTypeBits, props),
    };
    // Buffers whose addresses are taken (bufferDeviceAddress) must be allocated
    // with the matching memory-allocate flag.
    vk::MemoryAllocateFlagsInfo flagsInfo{.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
    {
        ai.pNext = &flagsInfo;
    }
    vk::raii::DeviceMemory mem(device_, ai);
    buf.bindMemory(*mem, 0);

    return {std::move(buf), std::move(mem)};
}

} // namespace fire_engine
