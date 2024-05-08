use ash::vk;



impl super::Swapchain {
    /// # Safety
    ///
    /// - The device must have been made idle before calling this function.
    unsafe fn release_resources(mut self, device: &ash::Device) -> Self {
        profiling::scope!("Swapchain::release_resources");
        {
            profiling::scope!("vkDeviceWaitIdle");
            // We need to also wait until all presentation work is done. Because there is no way to portably wait until
            // the presentation work is done, we are forced to wait until the device is idle.
            let _ = unsafe { device.device_wait_idle() };
        };

        for semaphore in self.surface_semaphores.drain(..) {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }

        self
    }
}

impl crate::Surface for super::Surface {
    type A = super::Api;

    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        // Safety: `configure`'s contract guarantees there are no resources derived from the swapchain in use.
        let mut swap_chain = self.swapchain.write();
        let old = swap_chain
            .take()
            .map(|sc| unsafe { sc.release_resources(&device.shared.raw) });

        let swapchain = unsafe { device.create_swapchain(self, config, old)? };
        *swap_chain = Some(swapchain);

        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        if let Some(sc) = self.swapchain.write().take() {
            // Safety: `unconfigure`'s contract guarantees there are no resources derived from the swapchain in use.
            let swapchain = unsafe { sc.release_resources(&device.shared.raw) };
            unsafe { swapchain.functor.destroy_swapchain(swapchain.raw, None) };
        }
    }

    unsafe fn acquire_texture(
        &self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let mut swapchain = self.swapchain.write();
        let sc = swapchain.as_mut().unwrap();

        let mut timeout_ns = match timeout {
            Some(duration) => duration.as_nanos() as u64,
            None => u64::MAX,
        };

        // AcquireNextImageKHR on Android (prior to Android 11) doesn't support timeouts
        // and will also log verbose warnings if tying to use a timeout.
        //
        // Android 10 implementation for reference:
        // https://android.googlesource.com/platform/frameworks/native/+/refs/tags/android-mainline-10.0.0_r13/vulkan/libvulkan/swapchain.cpp#1426
        // Android 11 implementation for reference:
        // https://android.googlesource.com/platform/frameworks/native/+/refs/tags/android-mainline-11.0.0_r45/vulkan/libvulkan/swapchain.cpp#1438
        //
        // Android 11 corresponds to an SDK_INT/ro.build.version.sdk of 30
        if cfg!(target_os = "android") && self.instance.android_sdk_version < 30 {
            timeout_ns = u64::MAX;
        }

        let wait_semaphore = sc.surface_semaphores[sc.next_surface_index];

        // will block if no image is available
        let (index, suboptimal) = match unsafe {
            sc.functor
                .acquire_next_image(sc.raw, timeout_ns, wait_semaphore, vk::Fence::null())
        } {
            // We treat `VK_SUBOPTIMAL_KHR` as `VK_SUCCESS` on Android.
            // See the comment in `Queue::present`.
            #[cfg(target_os = "android")]
            Ok((index, _)) => (index, false),
            #[cfg(not(target_os = "android"))]
            Ok(pair) => pair,
            Err(error) => {
                return match error {
                    vk::Result::TIMEOUT => Ok(None),
                    vk::Result::NOT_READY | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        Err(crate::SurfaceError::Outdated)
                    }
                    vk::Result::ERROR_SURFACE_LOST_KHR => Err(crate::SurfaceError::Lost),
                    other => Err(crate::DeviceError::from(other).into()),
                }
            }
        };

        sc.next_surface_index += 1;
        sc.next_surface_index %= sc.surface_semaphores.len();

        // special case for Intel Vulkan returning bizarre values (ugh)
        if sc.device.vendor_id == crate::auxil::db::intel::VENDOR && index > 0x100 {
            return Err(crate::SurfaceError::Outdated);
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-framebuffer-03209
        let raw_flags = if sc
            .raw_flags
            .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
        {
            vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
        } else {
            vk::ImageCreateFlags::empty()
        };

        let texture = super::SurfaceTexture {
            index,
            texture: super::Texture {
                raw: sc.images[index as usize],
                drop_guard: None,
                block: None,
                usage: sc.config.usage,
                format: sc.config.format,
                raw_flags,
                copy_size: crate::CopyExtent {
                    width: sc.config.extent.width,
                    height: sc.config.extent.height,
                    depth: 1,
                },
                view_formats: sc.view_formats.clone(),
            },
            wait_semaphore,
        };
        Ok(Some(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal,
        }))
    }

    unsafe fn discard_texture(&self, _texture: super::SurfaceTexture) {}
}
