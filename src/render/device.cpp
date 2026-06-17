#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include <fire_engine/render/device.hpp>

namespace fire_engine
{

#ifdef NDEBUG
constexpr bool enableValidation = false;
#else
constexpr bool enableValidation = true;
#endif

constexpr std::array validationLayers{"VK_LAYER_KHRONOS_validation"};
constexpr std::array deviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset", // required on macOS/MoltenVK
};

Device::Device(const Window& window)
{
    createInstance();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
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
