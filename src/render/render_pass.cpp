#include <fire_engine/render/render_pass.hpp>

#include <array>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

RenderPass RenderPass::createForward(const Device& device)
{
    // Forward writes into an offscreen HDR target (post-process reads it).
    vk::AttachmentDescription colourAtt{
        .format = vk::Format::eR16G16B16A16Sfloat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::AttachmentDescription depthAtt{
        .format = vk::Format::eD32Sfloat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::AttachmentReference depthRef{
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
        .pDepthStencilAttachment = &depthRef,
    };

    std::array<vk::SubpassDependency, 2> deps = {
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                             vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        },
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        },
    };

    std::array<vk::AttachmentDescription, 2> attachments = {colourAtt, depthAtt};
    vk::RenderPassCreateInfo ci{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies = deps.data(),
    };

    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

RenderPass RenderPass::createForwardTransmission(const Device& device)
{
    // KHR_materials_transmission F3: second forward sub-pass that loads the
    // HDR target's existing contents (already populated by the first forward
    // pass) and renders transmissive draws on top. The renderer issues an
    // explicit barrier to bring the HDR target back to ColorAttachmentOptimal
    // before begin, after the sceneColor blit chain has run.
    vk::AttachmentDescription colourAtt{
        .format = vk::Format::eR16G16B16A16Sfloat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::AttachmentDescription depthAtt{
        .format = vk::Format::eD32Sfloat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::AttachmentReference depthRef{
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
        .pDepthStencilAttachment = &depthRef,
    };

    // Subpass dependencies MUST mirror `createForward` exactly so the same
    // forward pipelines remain render-pass-compatible across both passes.
    // The HDR target's TransferSrc → ColorAttachmentOptimal transition is
    // handled by an explicit pipelineBarrier in recordSceneColorCapture, not
    // by a dependency on this pass.
    std::array<vk::SubpassDependency, 2> deps = {
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                             vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        },
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        },
    };

    std::array<vk::AttachmentDescription, 2> attachments = {colourAtt, depthAtt};
    vk::RenderPassCreateInfo ci{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies = deps.data(),
    };

    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

RenderPass RenderPass::createShadow(const Device& device)
{
    // MoltenVK quirk: depth-only render passes don't commit storeOp on TBDR.
    // Attach a B8G8R8A8 colour target so Metal treats this as a real render
    // pass. We keep/store it so debug builds can sample the light-space depth
    // written into the colour attachment.
    std::array<vk::AttachmentDescription, 2> attachments = {
        vk::AttachmentDescription{
            .format = vk::Format::eB8G8R8A8Unorm,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        },
        vk::AttachmentDescription{
            .format = vk::Format::eD32Sfloat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        },
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::AttachmentReference depthRef{
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
        .pDepthStencilAttachment = &depthRef,
    };

    std::array<vk::SubpassDependency, 2> deps = {
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        },
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        },
    };
    vk::RenderPassCreateInfo ci{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies = deps.data(),
    };

    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

void RenderPass::createShadowFramebuffer(const Device& device,
                                         std::span<const vk::ImageView> colourViews,
                                         std::span<const vk::ImageView> depthViews, uint32_t extent)
{
    framebuffers_.clear();
    framebuffers_.reserve(depthViews.size());
    for (std::size_t i = 0; i < depthViews.size(); ++i)
    {
        vk::ImageView colourView = colourViews.size() == 1 ? colourViews[0] : colourViews[i];
        std::array<vk::ImageView, 2> views = {colourView, depthViews[i]};
        vk::FramebufferCreateInfo ci{
            .renderPass = *renderPass_,
            .attachmentCount = static_cast<uint32_t>(views.size()),
            .pAttachments = views.data(),
            .width = extent,
            .height = extent,
            .layers = 1,
        };
        framebuffers_.emplace_back(device.device(), ci);
    }
}

void RenderPass::createForwardFramebuffer(const Device& device, vk::ImageView colourView,
                                          vk::ImageView depthView, vk::Extent2D extent)
{
    framebuffers_.clear();
    std::array<vk::ImageView, 2> attachments = {colourView, depthView};
    vk::FramebufferCreateInfo ci{
        .renderPass = *renderPass_,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };
    framebuffers_.emplace_back(device.device(), ci);
}

RenderPass RenderPass::createBloomDown(const Device& device, vk::Format colourFormat)
{
    vk::AttachmentDescription colourAtt{
        .format = colourFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
    };

    vk::SubpassDependency dep{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
    };

    vk::RenderPassCreateInfo ci{
        .attachmentCount = 1,
        .pAttachments = &colourAtt,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dep,
    };
    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

RenderPass RenderPass::createBloomUp(const Device& device, vk::Format colourFormat)
{
    // initialLayout=eShaderReadOnlyOptimal because the downsample pass left
    // every mip in that layout. loadOp=Load preserves it; the upsample pipeline
    // uses additive blending on top.
    vk::AttachmentDescription colourAtt{
        .format = colourFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
    };

    vk::SubpassDependency dep{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask =
            vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
    };

    vk::RenderPassCreateInfo ci{
        .attachmentCount = 1,
        .pAttachments = &colourAtt,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dep,
    };
    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

void RenderPass::createColourFramebuffersPerMip(const Device& device,
                                                std::span<const vk::ImageView> colourViews,
                                                std::span<const vk::Extent2D> extents)
{
    framebuffers_.clear();
    framebuffers_.reserve(colourViews.size());
    for (std::size_t i = 0; i < colourViews.size(); ++i)
    {
        vk::ImageView v = colourViews[i];
        vk::FramebufferCreateInfo ci{
            .renderPass = *renderPass_,
            .attachmentCount = 1,
            .pAttachments = &v,
            .width = extents[i].width,
            .height = extents[i].height,
            .layers = 1,
        };
        framebuffers_.emplace_back(device.device(), ci);
    }
}

RenderPass RenderPass::createPostProcess(const Device& device, const Swapchain& swapchain)
{
    vk::AttachmentDescription colourAtt{
        .format = swapchain.format(),
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
    };

    // Wait for the forward pass to finish writing the HDR target before the
    // post-process fragment shader samples it.
    vk::SubpassDependency dep{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    vk::RenderPassCreateInfo ci{
        .attachmentCount = 1,
        .pAttachments = &colourAtt,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dep,
    };
    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

void RenderPass::createPostProcessFramebuffers(const Device& device, const Swapchain& swapchain)
{
    framebuffers_.clear();

    auto extent = swapchain.extent();
    const auto& colourViews = swapchain.imageViews();
    framebuffers_.reserve(colourViews.size());

    for (std::size_t i = 0; i < colourViews.size(); ++i)
    {
        vk::ImageView view = *colourViews[i];
        vk::FramebufferCreateInfo ci{
            .renderPass = *renderPass_,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        };
        framebuffers_.emplace_back(device.device(), ci);
    }
}

RenderPass RenderPass::createOffscreenColour(const Device& device, vk::Format colourFormat)
{
    vk::AttachmentDescription colourAtt{
        .format = colourFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::AttachmentReference colourRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colourRef,
    };

    std::array<vk::SubpassDependency, 2> deps = {
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        },
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        },
    };

    vk::RenderPassCreateInfo ci{
        .attachmentCount = 1,
        .pAttachments = &colourAtt,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies = deps.data(),
    };
    RenderPass pass;
    pass.renderPass_ = vk::raii::RenderPass(device.device(), ci);
    return pass;
}

void RenderPass::createColourFramebuffers(const Device& device,
                                          std::span<const vk::ImageView> colourViews,
                                          uint32_t extent)
{
    framebuffers_.clear();
    framebuffers_.reserve(colourViews.size());

    for (vk::ImageView colourView : colourViews)
    {
        vk::FramebufferCreateInfo ci{
            .renderPass = *renderPass_,
            .attachmentCount = 1,
            .pAttachments = &colourView,
            .width = extent,
            .height = extent,
            .layers = 1,
        };
        framebuffers_.emplace_back(device.device(), ci);
    }
}

} // namespace fire_engine
