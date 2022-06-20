use std::{
    borrow::Cow,
    cell::UnsafeCell,
    marker::PhantomData,
    mem,
    sync::atomic::{AtomicU32, Ordering},
};

use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    hub::{self, IdentityHandler},
    id::{self, TypedId},
    sync::{DebugMaybeUninit, DebugUnsafeCell},
    Epoch, LifeGuard, RefCount, SubmissionIndex,
};

#[derive(Clone, Debug, Error)]
pub enum InvalidId {
    #[error("Resource {index} is in the error state because {error}. This happens when resource creation fails with an error and the ID is re-used.")]
    ResourceInError {
        index: u32,
        error: Cow<'static, str>,
    },
    #[error("Resource {index} has been destroyed. The storage is currently empty.")]
    Vacant { index: u32 },
    #[error("Resource {index} has been destroyed. The storage is currently storing a new resource with epoch {new} (given epoch {old}).")]
    WrongEpoch { index: u32, old: Epoch, new: Epoch },
}

#[must_use]
pub(crate) struct FutureId<'a, T: hub::Resource> {
    id: T::Id,
    data: &'a Storage<T>,
}

impl<T> FutureId<'_, T>
where
    T: hub::Resource,
{
    #[cfg(feature = "trace")]
    pub fn id(&self) -> T::Id {
        self.id
    }

    pub fn into_id(self) -> T::Id {
        self.id
    }

    pub fn assign(self, value: T) -> id::Valid<T::Id> {
        use id::TypedId as _;
        let (index, epoch, _) = self.id.unzip();
        unsafe { self.data.fill(index, epoch, Ok(value)) };
        id::Valid(self.id)
    }

    pub fn assign_error(self, label: &str) -> T::Id {
        use id::TypedId as _;
        let (index, epoch, _) = self.id.unzip();
        unsafe {
            self.data
                .fill(index, epoch, Err(Cow::Owned(String::from(label))))
        };
        self.id
    }
}

pub struct Registry<A, T, F>
where
    T: hub::Resource,
    F: hub::IdentityHandlerFactory<T::Id>,
{
    storage: Storage<T>,
    identity_manager: Mutex<F::Filter>,

    _phantom: PhantomData<A>,
}

impl<A, T, F> Registry<A, T, F>
where
    A: hub::HalApi,
    T: hub::Resource,
    F: hub::IdentityHandlerFactory<T::Id>,
{
    pub fn new(factory: &F) -> Self {
        Self {
            storage: Storage::new(),
            identity_manager: Mutex::new(factory.spawn()),
            _phantom: PhantomData,
        }
    }

    /// Provides no authoritative value on the contents of the registry,
    /// just "there is no id greater than"
    pub fn max_index(&self) -> usize {
        self.storage.max_index.load(Ordering::Acquire) as usize
    }

    /// Safe to call concurrently.
    pub(crate) fn prepare(
        &self,
        input: <F::Filter as IdentityHandler<T::Id>>::Input,
    ) -> FutureId<'_, T> {
        let mut ident_guard = self.identity_manager.lock();
        let (index, id) = ident_guard.process(input, A::VARIANT);
        unsafe { self.storage.ensure_index(index) }
        drop(ident_guard);

        FutureId {
            id,
            data: &self.storage,
        }
    }

    /// # Safety
    ///
    /// - No calls to `contains`, `get`, `get_unchecked`, or `Index` may happen
    ///   while this call is executing.
    pub unsafe fn unregister(&self, id: T::Id) -> Result<T, InvalidId> {
        let mut ident_guard = self.identity_manager.lock();
        let (index, epoch) = ident_guard.free(id);
        let value = self.storage.free(index, epoch);
        drop(ident_guard);
        value
    }

    pub fn contains(&self, id: T::Id) -> bool {
        let (index, epoch, _) = id.unzip();
        self.storage.contains(index, epoch)
    }

    pub fn get(&self, id: T::Id) -> Result<&T, InvalidId> {
        let (index, epoch, _) = id.unzip();
        self.storage.get(index, epoch)
    }

    pub fn get_unchecked(&self, index: u32) -> &T {
        self.storage.get_unchecked(index).unwrap()
    }

    /// # Safety
    ///
    /// Inherets from [`Self::unregister`].
    ///
    /// - No calls to `contains`, `get`, `get_unchecked`, or `Index` may happen
    ///   while this call is executing.
    pub(crate) unsafe fn drop_no_life_guard(&self, id: T::Id) -> Option<id::Valid<id::DeviceId>> {
        Some(self.drop_inner(id)?.2)
    }

    /// # Safety
    ///
    /// Inherets from [`Self::unregister`].
    ///
    /// - No calls to `contains`, `get`, `get_unchecked`, or `Index` may happen
    ///   while this call is executing.
    pub(crate) unsafe fn drop_with_life_guard(
        &self,
        id: T::Id,
    ) -> Option<(&T, RefCount, SubmissionIndex, id::Valid<id::DeviceId>)> {
        let (resource, life_guard, device_id) = self.drop_inner(id)?;
        let life_guard = life_guard.unwrap();
        Some((
            resource,
            life_guard.ref_count.take().unwrap(),
            life_guard.life_count(),
            device_id,
        ))
    }

    /// # Safety
    ///
    /// Inherets from [`Self::unregister`].
    ///
    /// - No calls to `contains`, `get`, `get_unchecked`, or `Index` may happen
    ///   while this call is executing.
    unsafe fn drop_inner(
        &self,
        id: T::Id,
    ) -> Option<(&T, Option<&LifeGuard>, id::Valid<id::DeviceId>)> {
        match self.get(id) {
            Ok(resource) => Some((resource, resource.life_guard(), resource.device_id())),
            Err(InvalidId::ResourceInError { .. }) => {
                self.unregister(id).ok()?;
                None
            }
            Err(e) => {
                log::error!("Tried to drop invalid {} id: {}", T::TYPE, e);
                None
            }
        }
    }

    pub fn insert_error(&self, id: T::Id, implicit_failure: &'static str) {
        let (index, epoch, _) = id.unzip();
        unsafe {
            self.storage
                .fill(index, epoch, Err(Cow::Borrowed(implicit_failure)))
        }
    }

    pub fn force_replace(&self, id: T::Id, value: T) {
        let (index, epoch, _) = id.unzip();
        unsafe { self.storage.overwrite(index, epoch, value) }
    }

    pub fn label_for_resource(&self, id: T::Id) -> String {
        match self.get(id) {
            Ok(resource) => resource.label().to_owned(),
            Err(_) => format!("{:?}", id),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (T::Id, &T)> {
        IteratorDataAdapter {
            _data: self.identity_manager.lock(),
            // SAFETY: The identity manager lock is taken and will be kept
            // alive as long as the iterator is alive.
            iter: unsafe { self.storage.iter(A::VARIANT) },
        }
    }

    /// # Safety:
    ///
    /// This function must be called _as if_ it takes a &mut self argument.
    ///
    /// If this is called with anything other than exclusive access to the registry,
    /// all bets are off.
    pub unsafe fn iter_mut(&self) -> impl Iterator<Item = (T::Id, &mut T)> {
        self.storage.iter_mut(A::VARIANT)
    }

    /// # Safety:
    ///
    /// This function must be called _as if_ it takes a &mut self argument.
    ///
    /// If this is called with anything other than exclusive access to the registry,
    /// all bets are off.
    pub unsafe fn remove_all(&self) -> impl Iterator<Item = T> + '_ {
        self.storage.remove_all(A::VARIANT)
    }

    pub fn generate_report(&self) -> hub::StorageReport {
        todo!()
    }
}

impl<A, T, F> std::ops::Index<id::Valid<T::Id>> for Registry<A, T, F>
where
    A: hub::HalApi,
    T: hub::Resource,
    F: hub::IdentityHandlerFactory<T::Id>,
{
    type Output = T;

    fn index(&self, id: id::Valid<T::Id>) -> &Self::Output {
        self.get(id.0).unwrap()
    }
}

struct IteratorDataAdapter<D, I> {
    _data: D,
    iter: I,
}

impl<D, I> Iterator for IteratorDataAdapter<D, I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

const STORAGE_BLOCK_SIZE: usize = 32;

pub struct Storage<T> {
    blocks: [UnsafeCell<Option<Box<StorageBlock<STORAGE_BLOCK_SIZE, T>>>>; 256],
    max_index: AtomicU32,
}
unsafe impl<T> Send for Storage<T> where T: Send {}
unsafe impl<T> Sync for Storage<T> where T: Sync {}

impl<T> Storage<T>
where
    T: hub::Resource,
{
    fn new() -> Self {
        Self {
            blocks: [(); 256].map(|_| UnsafeCell::new(None)),
            max_index: AtomicU32::new(0),
        }
    }

    // SAFETY: Must be called inside the identity manager lock.
    unsafe fn ensure_index(&self, index: u32) {
        let max_index = self.max_index.load(Ordering::Relaxed);
        if index < max_index {
            return;
        }
        assert_eq!(index, max_index);

        let block = max_index / STORAGE_BLOCK_SIZE as u32;
        if block >= 256 {
            panic!("Too many resources allocated");
        }

        let block_size_minus_one = (STORAGE_BLOCK_SIZE - 1) as u32;
        let allocated_length = (max_index + block_size_minus_one) & !block_size_minus_one;
        if index >= allocated_length {
            // SAFETY: we're the first to ever allocate this block, so we're the only one
            // who could get this mutable reference.
            let block = self.blocks[block as usize].get();
            block.write(Some(Box::new(StorageBlock::new_uninit())));
        }
        self.max_index.store(max_index + 1, Ordering::Release);
    }

    fn raw_refs(
        &self,
        index: u32,
    ) -> Result<(&DebugUnsafeCell<DebugMaybeUninit<T>>, &Mutex<ElementStatus>), InvalidId> {
        let block = index as usize / STORAGE_BLOCK_SIZE;
        let data_index = index as usize % STORAGE_BLOCK_SIZE;

        let length = self.max_index.load(Ordering::Acquire);

        if index >= length {
            return Err(InvalidId::Vacant { index });
        }

        // SAFETY: We have just bounds checked the index and we always check the
        // status before accessing the data.
        let block_option_ref = unsafe { &*self.blocks[block].get() };
        let block_ref = block_option_ref.as_deref().unwrap();
        let data_ref = &block_ref.data[data_index];
        let status_ref = &block_ref.status[data_index];

        Ok((data_ref, status_ref))
    }

    pub unsafe fn fill(&self, index: u32, epoch: u32, value: Result<T, Cow<'static, str>>) {
        let (data_ref, status_ref) = self.raw_refs(index).unwrap_unchecked();

        match value {
            Ok(v) => {
                // SAFETY: We have reserved a slot in the block so this block is guarenteed to exist
                // and the slot will not be accessed by any other users.
                let data_mut_ref = &mut *data_ref.get_mut();
                data_mut_ref.write(v);

                *status_ref.lock() = ElementStatus::Occupied(epoch);
            }
            Err(e) => {
                *status_ref.lock() = ElementStatus::Error(epoch, e);
            }
        }
    }

    pub unsafe fn overwrite(&self, index: u32, epoch: u32, value: T) {
        let (data_ref, status_ref) = self.raw_refs(index).unwrap_unchecked();

        let mut status = status_ref.lock();
        let mut data_mut_ref = data_ref.get_mut();
        if let ElementStatus::Occupied(_) = *status {
            data_mut_ref.assume_init_drop();
        }

        data_mut_ref.write(value);

        *status = ElementStatus::Occupied(epoch);
    }

    fn contains(&self, index: u32, epoch: u32) -> bool {
        let (_data_ref, status_ref) = match self.raw_refs(index) {
            Ok(v) => v,
            Err(_) => return false,
        };

        if *status_ref.lock() != ElementStatus::Occupied(epoch) {
            return false;
        }

        true
    }

    fn get(&self, index: u32, epoch: u32) -> Result<&T, InvalidId> {
        let (data_ref, status_ref) = self.raw_refs(index)?;

        let status = &*status_ref.lock();
        match *status {
            ElementStatus::Occupied(stored_epoch) | ElementStatus::Error(stored_epoch, _)
                if epoch != stored_epoch =>
            {
                Err(InvalidId::WrongEpoch {
                    index,
                    old: stored_epoch,
                    new: epoch,
                })
            }
            ElementStatus::Occupied(_) => {
                Ok(unsafe { data_ref.get_debug_unchecked().assume_init_ref() })
            }
            ElementStatus::Error(_, ref error) => Err(InvalidId::ResourceInError {
                index,
                error: error.clone(),
            }),
            ElementStatus::Vacant => Err(InvalidId::Vacant { index }),
        }
    }

    fn get_unchecked(&self, index: u32) -> Result<&T, InvalidId> {
        let (data_ref, status_ref) = self.raw_refs(index)?;

        let status = status_ref.lock();
        match *status {
            ElementStatus::Occupied(_) => {
                // TODO SAFETY: it isn't
                Ok(unsafe { data_ref.get_debug_unchecked().assume_init_ref() })
            }
            ElementStatus::Error(_, ref error) => Err(InvalidId::ResourceInError {
                index,
                error: error.clone(),
            }),
            ElementStatus::Vacant => Err(InvalidId::Vacant { index }),
        }
    }

    // SAFETY: Must be called inside the identity manager lock.
    unsafe fn free(&self, index: u32, epoch: u32) -> Result<T, InvalidId> {
        let (data_ref, status_ref) = self.raw_refs(index)?;

        // We set this to vacant first before destroying it, so that if a stray index comes through
        // later, it will see that this is vacant and error and not try to grab a reference to
        // the actual data. We are also inside the lock, so creation can't come by and try to
        // create new data while we're still trying to drop things.
        let old_status = mem::replace(&mut *status_ref.lock(), ElementStatus::Vacant);
        assert_eq!(old_status, ElementStatus::Occupied(epoch));

        let mut data_mut_ref = data_ref.get_mut();
        let data = mem::replace(&mut *data_mut_ref, DebugMaybeUninit::uninit()).assume_init();

        Ok(data)
    }

    fn iter_inner(
        &self,
        backend: wgt::Backend,
    ) -> impl Iterator<
        Item = (
            T::Id,
            &DebugUnsafeCell<DebugMaybeUninit<T>>,
            &Mutex<ElementStatus>,
        ),
    > + '_ {
        let elements_allocated = self.max_index.load(Ordering::Acquire) as usize;
        let blocks_allocated = (elements_allocated + STORAGE_BLOCK_SIZE - 1) / STORAGE_BLOCK_SIZE;

        self.blocks[0..blocks_allocated]
            .iter()
            .enumerate()
            .flat_map(move |(block_idx, block_ref)| {
                let block = unsafe { &*block_ref.get() }.as_ref().unwrap();

                let starting_idx = block_idx * STORAGE_BLOCK_SIZE;
                let end_idx = ((block_idx + 1) * STORAGE_BLOCK_SIZE).min(elements_allocated);
                let count = end_idx - starting_idx;
                (0..count).filter_map(move |element_idx| {
                    let total_idx = starting_idx + element_idx;

                    let status_ref = &block.status[element_idx];
                    let data_ref = &block.data[element_idx];

                    let status = status_ref.lock();
                    if let ElementStatus::Occupied(epoch) = *status {
                        let id = T::Id::zip(total_idx as u32, epoch, backend);
                        Some((id, data_ref, status_ref))
                    } else {
                        None
                    }
                })
            })
    }

    /// # Safety
    ///
    /// - The this function must be called with the resource creation lock taken.
    /// - The iterator must die before the resource creation lock dies.
    unsafe fn iter(&self, backend: wgt::Backend) -> impl Iterator<Item = (T::Id, &T)> + '_ {
        self.iter_inner(backend)
            .map(|(id, cell, _)| (id, cell.get_debug_unchecked().assume_init_ref()))
    }

    /// # Safety
    ///
    /// - The this function must be called with exclusive access to self.
    unsafe fn iter_mut(&self, backend: wgt::Backend) -> impl Iterator<Item = (T::Id, &mut T)> + '_ {
        self.iter_inner(backend)
            .map(|(id, cell, _)| (id, cell.get_debug_unchecked_mut().assume_init_mut()))
    }

    /// # Safety
    ///
    /// - The this function must be called with exclusive access to self.
    unsafe fn remove_all(&self, backend: wgt::Backend) -> impl Iterator<Item = T> + '_ {
        self.iter_inner(backend).map(|(_, cell, status)| {
            *status.lock() = ElementStatus::Vacant;
            let value = mem::replace(&mut *cell.get_mut(), DebugMaybeUninit::uninit());
            value.assume_init()
        })
    }
}

struct StorageBlock<const COUNT: usize, T> {
    data: [DebugUnsafeCell<DebugMaybeUninit<T>>; COUNT],
    // TODO: this can be an atomic
    status: [Mutex<ElementStatus>; COUNT],
}

impl<const COUNT: usize, T> StorageBlock<COUNT, T> {
    fn new_uninit() -> Self {
        Self {
            data: [(); COUNT].map(|_| DebugUnsafeCell::new(DebugMaybeUninit::uninit())),
            status: [(); COUNT].map(|_| Mutex::new(ElementStatus::Vacant)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ElementStatus {
    Vacant,
    Occupied(Epoch),
    Error(Epoch, Cow<'static, str>),
}
