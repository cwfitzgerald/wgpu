/*! This library safely implements WebGPU on native platforms.
 *  It is designed for integration into browsers, as well as wrapping
 *  into other language-specific user-friendly libraries.
 */

#![allow(
    // It is much clearer to assert negative conditions with eq! false
    clippy::bool_assert_comparison,
    // We use loops for getting early-out of scope without closures.
    clippy::never_loop,
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
    // Needless updates are more scaleable, easier to play with features.
    clippy::needless_update,
    // Need many arguments for some core functions to be able to re-use code in many situations.
    clippy::too_many_arguments,
    // For some reason `rustc` can warn about these in const generics even
    // though they are required.
    unused_braces,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

pub mod binding_model;
pub mod command;
mod conv;
mod destroy;
pub mod device;
pub mod error;
pub mod hub;
pub mod id;
mod init_tracker;
pub mod instance;
pub mod pipeline;
pub mod present;
pub mod registry;
pub mod resource;
pub mod sync;
mod track;
mod validation;

pub use hal::{api, MAX_BIND_GROUPS, MAX_COLOR_TARGETS, MAX_VERTEX_BUFFERS};

use atomic::{AtomicUsize, Ordering};

use std::{borrow::Cow, mem::ManuallyDrop, os::raw::c_char, ptr, sync::atomic};

/// The index of a queue submission.
///
/// These are the values stored in `Device::fence`.
type SubmissionIndex = hal::FenceValue;
type AtomicSubmissionIndex = atomic::AtomicU64;

type Index = u32;
type Epoch = u32;

pub type RawString = *const c_char;
pub type Label<'a> = Option<Cow<'a, str>>;

trait LabelHelpers<'a> {
    fn borrow_option(&'a self) -> Option<&'a str>;
    fn borrow_or_default(&'a self) -> &'a str;
}
impl<'a> LabelHelpers<'a> for Label<'a> {
    fn borrow_option(&'a self) -> Option<&'a str> {
        self.as_ref().map(|cow| cow.as_ref())
    }
    fn borrow_or_default(&'a self) -> &'a str {
        self.borrow_option().unwrap_or_default()
    }
}

#[derive(Debug)]
struct AtomicOptionalRefCount(atomic::AtomicPtr<AtomicUsize>);

impl AtomicOptionalRefCount {
    fn from_ref_count(ref_count: RefCount) -> Self {
        Self(atomic::AtomicPtr::new(ref_count.0.as_ptr()))
    }

    fn as_ref_count(&self) -> Option<ManuallyDrop<RefCount>> {
        let ptr = self.0.load(Ordering::Acquire);
        let non_null = ptr::NonNull::new(ptr)?;
        let ref_count = RefCount(non_null);

        Some(ManuallyDrop::new(ref_count))
    }

    fn is_some(&self) -> bool {
        self.as_ref_count().is_some()
    }

    fn is_none(&self) -> bool {
        !self.is_some()
    }

    fn take(&self) -> Option<RefCount> {
        let ptr = self.0.swap(ptr::null_mut(), Ordering::AcqRel);
        let non_null = ptr::NonNull::new(ptr)?;

        Some(RefCount(non_null))
    }
}

impl Drop for AtomicOptionalRefCount {
    fn drop(&mut self) {
        // Turn this into a real refcount, then drop it if it needs to drop.
        drop(self.take());
    }
}

/// Reference count object that is 1:1 with each reference.
///
/// All the clones of a given `RefCount` point to the same
/// heap-allocated atomic reference count. When the count drops to
/// zero, only the count is freed. No other automatic cleanup takes
/// place; this is just a reference count, not a smart pointer.
///
/// `RefCount` values are created only by [`LifeGuard::new`] and by
/// `Clone`, so every `RefCount` is implicitly tied to some
/// [`LifeGuard`].
#[derive(Debug)]
struct RefCount(ptr::NonNull<AtomicUsize>);

unsafe impl Send for RefCount {}
unsafe impl Sync for RefCount {}

impl RefCount {
    const MAX: usize = 1 << 24;

    /// Construct a new `RefCount`, with an initial count of 1.
    fn new() -> RefCount {
        let bx = Box::new(AtomicUsize::new(1));
        Self(unsafe { ptr::NonNull::new_unchecked(Box::into_raw(bx)) })
    }

    fn load(&self) -> usize {
        unsafe { self.0.as_ref() }.load(Ordering::Acquire)
    }
}

impl Clone for RefCount {
    fn clone(&self) -> Self {
        let old_size = unsafe { self.0.as_ref() }.fetch_add(1, Ordering::AcqRel);
        assert!(old_size < Self::MAX);
        Self(self.0)
    }
}

impl Drop for RefCount {
    fn drop(&mut self) {
        unsafe {
            if self.0.as_ref().fetch_sub(1, Ordering::AcqRel) == 1 {
                drop(Box::from_raw(self.0.as_ptr()));
            }
        }
    }
}

/// Reference count object that tracks multiple references.
/// Unlike `RefCount`, it's manually inc()/dec() called.
#[derive(Debug)]
struct MultiRefCount(AtomicUsize);

impl MultiRefCount {
    fn new() -> Self {
        Self(AtomicUsize::new(1))
    }

    fn inc(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    fn dec_and_check_empty(&self) -> bool {
        self.0.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// Information needed to decide when it's safe to free some wgpu-core
/// resource.
///
/// Each type representing a `wgpu-core` resource, like [`Device`],
/// [`Buffer`], etc., contains a `LifeGuard` which indicates whether
/// it is safe to free.
///
/// A resource may need to be retained for any of several reasons:
///
/// - The user may hold a reference to it (via a `wgpu::Buffer`, say).
///
/// - Other resources may depend on it (a texture view's backing
///   texture, for example).
///
/// - It may be used by commands sent to the GPU that have not yet
///   finished execution.
///
/// [`Device`]: device::Device
/// [`Buffer`]: resource::Buffer
#[derive(Debug)]
pub struct LifeGuard {
    /// `RefCount` for the user's reference to this resource.
    ///
    /// When the user first creates a `wgpu-core` resource, this `RefCount` is
    /// created along with the resource's `LifeGuard`. When the user drops the
    /// resource, we swap this out for `None`. Note that the resource may
    /// still be held alive by other resources.
    ///
    /// Any `Stored<T>` value holds a clone of this `RefCount` along with the id
    /// of a `T` resource.
    ref_count: AtomicOptionalRefCount,

    /// The index of the last queue submission in which the resource
    /// was used.
    ///
    /// Each queue submission is fenced and assigned an index number
    /// sequentially. Thus, when a queue submission completes, we know any
    /// resources used in that submission and any lower-numbered submissions are
    /// no longer in use by the GPU.
    submission_index: AtomicUsize,

    /// The `label` from the descriptor used to create the resource.
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl LifeGuard {
    #[allow(unused_variables)]
    fn new(label: &str) -> Self {
        Self {
            ref_count: AtomicOptionalRefCount::from_ref_count(RefCount::new()),
            submission_index: AtomicUsize::new(0),
            #[cfg(debug_assertions)]
            label: label.to_string(),
        }
    }

    fn add_ref(&self) -> RefCount {
        ManuallyDrop::into_inner(self.ref_count.as_ref_count().unwrap())
    }

    /// Record that this resource will be used by the queue submission with the
    /// given index.
    ///
    /// Returns `true` if the resource is still held by the user.
    fn use_at(&self, submit_index: SubmissionIndex) -> bool {
        self.submission_index
            .store(submit_index as _, Ordering::Release);
        self.ref_count.as_ref_count().is_some()
    }

    fn life_count(&self) -> SubmissionIndex {
        self.submission_index.load(Ordering::Acquire) as _
    }
}

#[derive(Clone, Debug)]
struct Stored<T> {
    value: id::Valid<T>,
    ref_count: RefCount,
}

const DOWNLEVEL_WARNING_MESSAGE: &str = "The underlying API or device in use does not \
support enough features to be a fully compliant implementation of WebGPU. A subset of the features can still be used. \
If you are running this program on native and not in a browser and wish to limit the features you use to the supported subset, \
call Adapter::downlevel_properties or Device::downlevel_properties to get a listing of the features the current \
platform supports.";
const DOWNLEVEL_ERROR_MESSAGE: &str = "This is not an invalid use of WebGPU: the underlying API or device does not \
support enough features to be a fully compliant implementation. A subset of the features can still be used. \
If you are running this program on native and not in a browser and wish to work around this issue, call \
Adapter::downlevel_properties or Device::downlevel_properties to get a listing of the features the current \
platform supports.";

/// Call a `Global` method, dispatching dynamically to the appropriate back end.
///
/// Uses of this macro have the form:
///
/// ```ignore
///
///     gfx_select!(id => global.method(args...))
///
/// ```
///
/// where `id` is some [`id::Id`] resource id, `global` is a [`hub::Global`],
/// and `method` is any method on [`Global`] that takes a single generic
/// parameter that implements [`hal::Api`] (for example,
/// [`Global::device_create_buffer`]).
///
/// The `wgpu-core` crate can support multiple back ends simultaneously (Vulkan,
/// Metal, etc.), depending on features and availability. Each [`Id`]'s value
/// indicates which back end its resource belongs to. This macro does a switch
/// on `id`'s back end, and calls the `Global` method specialized for that back
/// end.
///
/// Internally to `wgpu-core`, most types take the back end (some type that
/// implements `hal::Api`) as a generic parameter, so their methods are compiled
/// with full knowledge of which back end they're working with. This macro
/// serves as the boundary between dynamic `Id` values provided by `wgpu-core`'s
/// users and the crate's mostly-monomorphized implementation, selecting the
/// `hal::Api` implementation appropriate to the `Id` value's back end.
///
/// [`Global`]: hub::Global
/// [`Global::device_create_buffer`]: hub::Global::device_create_buffer
/// [`Id`]: id::Id
#[macro_export]
macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),* )) => {
        // Note: For some reason the cfg aliases defined in build.rs don't succesfully apply in this
        // macro so we must specify their equivalents manually
        match $id.backend() {
            #[cfg(any(
                all(not(target_arch = "wasm32"), not(target_os = "ios"), not(target_os = "macos")),
                feature = "vulkan-portability"
            ))]
            wgt::Backend::Vulkan => $global.$method::<$crate::api::Vulkan>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Metal => $global.$method::<$crate::api::Metal>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx12 => $global.$method::<$crate::api::Dx12>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx11 => $global.$method::<$crate::api::Dx11>( $($param),* ),
            #[cfg(any(
                all(unix, not(target_os = "macos"), not(target_os = "ios")),
                feature = "angle",
                target_arch = "wasm32"
            ))]
            wgt::Backend::Gl => $global.$method::<$crate::api::Gles>( $($param),+ ),
            other => panic!("Unexpected backend {:?}", other),

        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
/// Fast hash set used internally.
type FastHashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<fxhash::FxHasher>>;

#[inline]
pub(crate) fn get_lowest_common_denom(a: u32, b: u32) -> u32 {
    let gcd = if a >= b {
        get_greatest_common_divisor(a, b)
    } else {
        get_greatest_common_divisor(b, a)
    };
    a * b / gcd
}

#[inline]
pub(crate) fn get_greatest_common_divisor(mut a: u32, mut b: u32) -> u32 {
    assert!(a >= b);
    loop {
        let c = a % b;
        if c == 0 {
            return b;
        } else {
            a = b;
            b = c;
        }
    }
}

#[test]
fn test_lcd() {
    assert_eq!(get_lowest_common_denom(2, 2), 2);
    assert_eq!(get_lowest_common_denom(2, 3), 6);
    assert_eq!(get_lowest_common_denom(6, 4), 12);
}

#[test]
fn test_gcd() {
    assert_eq!(get_greatest_common_divisor(5, 1), 1);
    assert_eq!(get_greatest_common_divisor(4, 2), 2);
    assert_eq!(get_greatest_common_divisor(6, 4), 2);
    assert_eq!(get_greatest_common_divisor(7, 7), 7);
}
