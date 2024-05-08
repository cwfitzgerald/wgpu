use std::sync::atomic::Ordering;

use arrayvec::ArrayVec;
use ash::vk;




impl crate::Queue for super::Queue {
    type A = super::Api;

    unsafe fn submit(
        &self,
        command_buffers: &[&super::CommandBuffer],
        surface_textures: &[&super::SurfaceTexture],
        signal_fence: Option<(&mut super::Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        let mut fence_raw = vk::Fence::null();

        let mut wait_stage_masks = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut signal_semaphores = ArrayVec::<_, 2>::new();
        let mut signal_values = ArrayVec::<_, 2>::new();

        for &surface_texture in surface_textures {
            wait_stage_masks.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            wait_semaphores.push(surface_texture.wait_semaphore);
        }

        let old_index = self.relay_index.load(Ordering::Relaxed);

        let sem_index = if old_index >= 0 {
            wait_stage_masks.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            wait_semaphores.push(self.relay_semaphores[old_index as usize]);
            (old_index as usize + 1) % self.relay_semaphores.len()
        } else {
            0
        };

        signal_semaphores.push(self.relay_semaphores[sem_index]);

        self.relay_index
            .store(sem_index as isize, Ordering::Relaxed);

        if let Some((fence, value)) = signal_fence {
            fence.maintain(&self.device.raw)?;
            match *fence {
                super::Fence::TimelineSemaphore(raw) => {
                    signal_semaphores.push(raw);
                    signal_values.push(!0);
                    signal_values.push(value);
                }
                super::Fence::FencePool {
                    ref mut active,
                    ref mut free,
                    ..
                } => {
                    fence_raw = match free.pop() {
                        Some(raw) => raw,
                        None => unsafe {
                            self.device
                                .raw
                                .create_fence(&vk::FenceCreateInfo::builder(), None)?
                        },
                    };
                    active.push((value, fence_raw));
                }
            }
        }

        let vk_cmd_buffers = command_buffers
            .iter()
            .map(|cmd| cmd.raw)
            .collect::<Vec<_>>();

        let mut vk_info = vk::SubmitInfo::builder().command_buffers(&vk_cmd_buffers);

        vk_info = vk_info
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks)
            .signal_semaphores(&signal_semaphores);

        let mut vk_timeline_info;

        if !signal_values.is_empty() {
            vk_timeline_info =
                vk::TimelineSemaphoreSubmitInfo::builder().signal_semaphore_values(&signal_values);
            vk_info = vk_info.push_next(&mut vk_timeline_info);
        }

        profiling::scope!("vkQueueSubmit");
        unsafe {
            self.device
                .raw
                .queue_submit(self.raw, &[vk_info.build()], fence_raw)?
        };
        Ok(())
    }

    unsafe fn present(
        &self,
        surface: &super::Surface,
        texture: super::SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let mut swapchain = surface.swapchain.write();
        let ssc = swapchain.as_mut().unwrap();

        let swapchains = [ssc.raw];
        let image_indices = [texture.index];
        let mut vk_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let old_index = self.relay_index.swap(-1, Ordering::Relaxed);
        if old_index >= 0 {
            vk_info = vk_info.wait_semaphores(
                &self.relay_semaphores[old_index as usize..old_index as usize + 1],
            );
        }

        let suboptimal = {
            profiling::scope!("vkQueuePresentKHR");
            unsafe { self.swapchain_fn.queue_present(self.raw, &vk_info) }.map_err(|error| {
                match error {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => crate::SurfaceError::Outdated,
                    vk::Result::ERROR_SURFACE_LOST_KHR => crate::SurfaceError::Lost,
                    _ => crate::DeviceError::from(error).into(),
                }
            })?
        };
        if suboptimal {
            // We treat `VK_SUBOPTIMAL_KHR` as `VK_SUCCESS` on Android.
            // On Android 10+, libvulkan's `vkQueuePresentKHR` implementation returns `VK_SUBOPTIMAL_KHR` if not doing pre-rotation
            // (i.e `VkSwapchainCreateInfoKHR::preTransform` not being equal to the current device orientation).
            // This is always the case when the device orientation is anything other than the identity one, as we unconditionally use `VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR`.
            #[cfg(not(target_os = "android"))]
            log::warn!("Suboptimal present of frame {}", texture.index);
        }
        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        self.device.timestamp_period
    }
}
