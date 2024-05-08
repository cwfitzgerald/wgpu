use ash::{extensions::khr, vk};

use crate::vulkan::ExtensionFn;

/// The [`Api::Fence`] type for [`vulkan::Api`].
///
/// This is an `enum` because there are two possible implementations of
/// `wgpu-hal` fences on Vulkan: Vulkan fences, which work on any version of
/// Vulkan, and Vulkan timeline semaphores, which are easier and cheaper but
/// require non-1.0 features.
///
/// [`Device::create_fence`] returns a [`TimelineSemaphore`] if
/// [`VK_KHR_timeline_semaphore`] is available and enabled, and a [`FencePool`]
/// otherwise.
///
/// [`Api::Fence`]: crate::Api::Fence
/// [`vulkan::Api`]: Api
/// [`Device::create_fence`]: crate::Device::create_fence
/// [`TimelineSemaphore`]: Fence::TimelineSemaphore
/// [`VK_KHR_timeline_semaphore`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_timeline_semaphore
/// [`FencePool`]: Fence::FencePool
#[derive(Debug)]
pub enum Fence {
    /// A Vulkan [timeline semaphore].
    ///
    /// These are simpler to use than Vulkan fences, since timeline semaphores
    /// work exactly the way [`wpgu_hal::Api::Fence`] is specified to work.
    ///
    /// [timeline semaphore]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-semaphores
    /// [`wpgu_hal::Api::Fence`]: crate::Api::Fence
    TimelineSemaphore(vk::Semaphore),

    /// A collection of Vulkan [fence]s, each associated with a [`FenceValue`].
    ///
    /// The effective [`FenceValue`] of this variant is the greater of
    /// `last_completed` and the maximum value associated with a signalled fence
    /// in `active`.
    ///
    /// Fences are available in all versions of Vulkan, but since they only have
    /// two states, "signaled" and "unsignaled", we need to use a separate fence
    /// for each queue submission we might want to wait for, and remember which
    /// [`FenceValue`] each one represents.
    ///
    /// [fence]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-fences
    /// [`FenceValue`]: crate::FenceValue
    FencePool {
        last_completed: crate::FenceValue,
        /// The pending fence values have to be ascending.
        active: Vec<(crate::FenceValue, vk::Fence)>,
        free: Vec<vk::Fence>,
    },
}

impl Fence {
    /// Return the highest [`FenceValue`] among the signalled fences in `active`.
    ///
    /// As an optimization, assume that we already know that the fence has
    /// reached `last_completed`, and don't bother checking fences whose values
    /// are less than that: those fences remain in the `active` array only
    /// because we haven't called `maintain` yet to clean them up.
    ///
    /// [`FenceValue`]: crate::FenceValue
    fn check_active(
        device: &ash::Device,
        mut last_completed: crate::FenceValue,
        active: &[(crate::FenceValue, vk::Fence)],
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        for &(value, raw) in active.iter() {
            unsafe {
                if value > last_completed && device.get_fence_status(raw)? {
                    last_completed = value;
                }
            }
        }
        Ok(last_completed)
    }

    /// Return the highest signalled [`FenceValue`] for `self`.
    ///
    /// [`FenceValue`]: crate::FenceValue
    pub(super) fn get_latest(
        &self,
        device: &ash::Device,
        extension: Option<&ExtensionFn<khr::TimelineSemaphore>>,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        match *self {
            Self::TimelineSemaphore(raw) => unsafe {
                Ok(match *extension.unwrap() {
                    ExtensionFn::Extension(ref ext) => ext.get_semaphore_counter_value(raw)?,
                    ExtensionFn::Promoted => device.get_semaphore_counter_value(raw)?,
                })
            },
            Self::FencePool {
                last_completed,
                ref active,
                free: _,
            } => Self::check_active(device, last_completed, active),
        }
    }

    /// Trim the internal state of this [`Fence`].
    ///
    /// This function has no externally visible effect, but you should call it
    /// periodically to keep this fence's resource consumption under control.
    ///
    /// For fences using the [`FencePool`] implementation, this function
    /// recycles fences that have been signaled. If you don't call this,
    /// [`Queue::submit`] will just keep allocating a new Vulkan fence every
    /// time it's called.
    ///
    /// [`FencePool`]: Fence::FencePool
    /// [`Queue::submit`]: crate::Queue::submit
    pub(super) fn maintain(&mut self, device: &ash::Device) -> Result<(), crate::DeviceError> {
        match *self {
            Self::TimelineSemaphore(_) => {}
            Self::FencePool {
                ref mut last_completed,
                ref mut active,
                ref mut free,
            } => {
                let latest = Self::check_active(device, *last_completed, active)?;
                let base_free = free.len();
                for &(value, raw) in active.iter() {
                    if value <= latest {
                        free.push(raw);
                    }
                }
                if free.len() != base_free {
                    active.retain(|&(value, _)| value > latest);
                    unsafe {
                        device.reset_fences(&free[base_free..])?;
                    }
                }
                *last_completed = latest;
            }
        }
        Ok(())
    }
}
