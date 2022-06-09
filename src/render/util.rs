use std::slice;

use anyhow::Result;
use ash::{prelude::VkResult, vk, Device};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation
};

pub fn create_render_pass(device: &Device, color_format: vk::Format, depth_format: vk::Format) -> VkResult<vk::RenderPass> {
    let attachment_descriptions = [
        *vk::AttachmentDescription::builder()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        *vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    ];

    let color_attachment_reference = vk::AttachmentReference::builder().layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment_reference = vk::AttachmentReference::builder().attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass_description = vk::SubpassDescription::builder()
        .color_attachments(slice::from_ref(&color_attachment_reference))
        .depth_stencil_attachment(&depth_attachment_reference);

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descriptions)
        .subpasses(slice::from_ref(&subpass_description));

    unsafe { device.create_render_pass(&render_pass_create_info, None) }
}

pub fn create_depth_image(device: &Device, allocator: &mut Allocator, width: u32, height: u32, format: vk::Format) -> Result<(vk::Image, Allocation, vk::ImageView)> {
    let image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(*vk::Extent3D::builder().width(width).height(height).depth(1))
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_create_info, None) }?;
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator.allocate(&AllocationCreateDesc {
        name: "depth texture",
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false
    })?;

    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }?;

    let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(Default::default())
        .subresource_range(*vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1));

    let image_view = unsafe { device.create_image_view(&image_view_create_info, None) }?;

    Ok((image, allocation, image_view))
}
