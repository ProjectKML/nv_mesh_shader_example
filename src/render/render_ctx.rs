use std::{mem::ManuallyDrop, slice, sync::Arc};

use ash::{
    extensions::{
        khr::{Surface, Swapchain},
        nv::MeshShader
    },
    vk, Device, Entry, Instance
};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use winit::window::Window;

use crate::render::{frame, frame::Frame, util};

pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub struct RenderCtx {
    pub entry_loader: Entry,

    pub instance_loader: Instance,
    pub surface_loader: Surface,

    pub surface: vk::SurfaceKHR,

    pub device_loader: Arc<Device>,
    pub swapchain_loader: Swapchain,
    pub mesh_shader_loader: MeshShader,

    pub allocator: ManuallyDrop<Allocator>,

    pub direct_queue: vk::Queue,
    pub swapchain: vk::SwapchainKHR,

    pub render_pass: vk::RenderPass,

    pub depth_image: vk::Image,
    pub depth_image_allocation: Allocation,
    pub depth_image_view: vk::ImageView,

    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,

    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub frames: Vec<ManuallyDrop<Frame>>
}

impl RenderCtx {
    pub fn new(window: &Window) -> Self {
        unsafe {
            let entry_loader = Entry::load().unwrap();

            let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);

            let instance_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast()];

            let mut instance_extensions = vec![];
            ash_window::enumerate_required_extensions(&window).unwrap().iter().for_each(|e| instance_extensions.push(*e));

            let instance_create_info = vk::InstanceCreateInfo::builder()
                .enabled_layer_names(&instance_layers)
                .enabled_extension_names(&instance_extensions)
                .application_info(&application_info);

            let instance_loader = entry_loader.create_instance(&instance_create_info, None).unwrap();
            let surface_loader = Surface::new(&entry_loader, &instance_loader);

            let surface = ash_window::create_surface(&entry_loader, &instance_loader, &window, None).unwrap();

            let physical_devices = instance_loader.enumerate_physical_devices().unwrap();
            let physical_device = physical_devices[0];

            let queue_priority = 1.0;
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder().queue_priorities(slice::from_ref(&queue_priority));

            let device_extensions = [Swapchain::name().as_ptr(), MeshShader::name().as_ptr()];

            let physical_device_features = vk::PhysicalDeviceFeatures::builder();
            let mut physical_device_mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesNV::builder().task_shader(true).mesh_shader(true);

            let mut physical_device_features = vk::PhysicalDeviceFeatures2::builder()
                .features(*physical_device_features)
                .push_next(&mut physical_device_mesh_shader_features);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .push_next(&mut physical_device_features)
                .queue_create_infos(slice::from_ref(&device_queue_create_info))
                .enabled_extension_names(&device_extensions);
            let device_loader = Arc::new(instance_loader.create_device(physical_device, &device_create_info, None).unwrap());
            let swapchain_loader = Swapchain::new(&instance_loader, &device_loader);
            let mesh_shader_loader = MeshShader::new(&instance_loader, &device_loader);

            let mut allocator = ManuallyDrop::new(
                Allocator::new(&AllocatorCreateDesc {
                    instance: instance_loader.clone(),
                    device: (*device_loader).clone(),
                    physical_device,
                    debug_settings: Default::default(),
                    buffer_device_address: false
                })
                .unwrap()
            );

            let direct_queue = device_loader.get_device_queue(0, 0);

            let swapchian_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(2)
                .image_format(SWAPCHAIN_FORMAT)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(vk::Extent2D { width: WIDTH, height: HEIGHT })
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::IMMEDIATE);

            let swapchain = swapchain_loader.create_swapchain(&swapchian_create_info, None).unwrap();
            let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

            let render_pass = util::create_render_pass(&device_loader, SWAPCHAIN_FORMAT, DEPTH_FORMAT).unwrap();
            let (depth_image, depth_image_allocation, depth_image_view) = util::create_depth_image(&device_loader, &mut allocator, WIDTH, HEIGHT, DEPTH_FORMAT).unwrap();

            let swapchain_image_views = swapchain_images
                .iter()
                .map(|image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(SWAPCHAIN_FORMAT)
                        .components(Default::default())
                        .subresource_range(*vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

                    device_loader.create_image_view(&image_view_create_info, None)
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            let framebuffers = swapchain_image_views
                .iter()
                .map(|image_view| {
                    let attachments = [*image_view, depth_image_view];

                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(WIDTH)
                        .height(HEIGHT)
                        .layers(1);

                    device_loader.create_framebuffer(&framebuffer_create_info, None)
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            let mesh_shader = util::create_shader_module(&device_loader, "example.mesh.spv").unwrap();
            let fragment_shader = util::create_shader_module(&device_loader, "example.frag.spv").unwrap();

            let pipeline_layout = device_loader.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder(), None).unwrap();
            let pipeline = util::create_mesh_pipeline(&device_loader, mesh_shader, None, fragment_shader, render_pass, pipeline_layout).unwrap();

            device_loader.destroy_shader_module(fragment_shader, None);
            device_loader.destroy_shader_module(mesh_shader, None);

            let frames: Vec<_> = (0..frame::NUM_FRAMES).into_iter().map(|_| ManuallyDrop::new(Frame::new(device_loader.clone()))).collect();

            Self {
                entry_loader,

                instance_loader,
                surface_loader,

                surface,

                device_loader,
                swapchain_loader,
                mesh_shader_loader,

                allocator,

                direct_queue,
                swapchain,

                render_pass,

                depth_image,
                depth_image_allocation,
                depth_image_view,

                swapchain_images,
                swapchain_image_views,
                framebuffers,

                pipeline_layout,
                pipeline,

                frames
            }
        }
    }
}

impl Drop for RenderCtx {
    fn drop(&mut self) {
        unsafe {
            self.device_loader.device_wait_idle().unwrap();

            self.frames.iter_mut().for_each(|frame| ManuallyDrop::drop(frame));

            self.device_loader.destroy_pipeline(self.pipeline, None);
            self.device_loader.destroy_pipeline_layout(self.pipeline_layout, None);

            self.framebuffers.iter().for_each(|framebuffer| self.device_loader.destroy_framebuffer(*framebuffer, None));
            self.device_loader.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views.iter().for_each(|image_view| self.device_loader.destroy_image_view(*image_view, None));

            self.device_loader.destroy_image_view(self.depth_image_view, None);
            self.device_loader.destroy_image(self.depth_image, None);

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            ManuallyDrop::drop(&mut self.allocator);
            self.device_loader.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);

            self.instance_loader.destroy_instance(None);
        }
    }
}
