use parking_lot::Mutex;
use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, DynamicOffset,
};

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};
use core::{convert::Infallible, fmt, str};

use crate::{
    api_log,
    binding_model::{BindError, ImmediateUploadError, LateMinBufferBindingSizeMismatch},
    command::{
        bind::{Binder, BinderError},
        compute_command::ArcComputeCommand,
        encoder::EncodingState,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        pass::{self, flush_bindings_helper},
        pass_base, pass_try,
        query::{
            end_pipeline_statistics_query, record_pass_timestamp_writes,
            validate_and_begin_pipeline_statistics_query,
        },
        ArcCommand, ArcPassTimestampWrites, BasePass, BindGroupArenaIndex, BindGroupStateChange,
        CommandEncoder, CommandEncoderError, ComputeArenas, ComputeInternCaches,
        ComputePipelineArenaIndex, DebugGroupError, EncoderStateError, EncoderVecPool,
        InnerCommandEncoder, MapPassErr, PassErrorScope, PassStateError, PassTimestampWrites,
        QuerySetArenaIndex, QueryUseError, StateChange, TimestampWritesError,
        TransitionResourcesError,
    },
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures},
    global::Global,
    hal_label, id, impl_resource_type,
    init_tracker::MemoryInitKind,
    pipeline::ComputePipeline,
    resource::{
        self, Buffer, DestroyedResourceError, InvalidResourceError, Labeled,
        MissingBufferUsageError, ParentDevice, RawResourceAccess, TextureView, Trackable,
    },
    track::{ResourceUsageCompatibilityError, TextureViewBindGroupState, Tracker},
    Label,
};

pub type ComputeBasePass = BasePass<ArcComputeCommand, ComputePassError>;

/// A pass's [encoder state](https://www.w3.org/TR/webgpu/#encoder-state) and
/// its validity are two distinct conditions, i.e., the full matrix of
/// (open, ended) x (valid, invalid) is possible.
///
/// The presence or absence of the `parent` `Option` indicates the pass's state.
/// The presence or absence of an error in `base.error` indicates the pass's
/// validity.
pub struct ComputePass {
    /// All pass data & records is stored here.
    base: ComputeBasePass,

    /// Parent command encoder that this pass records commands into.
    ///
    /// If this is `Some`, then the pass is in WebGPU's "open" state. If it is
    /// `None`, then the pass is in the "ended" state.
    /// See <https://www.w3.org/TR/webgpu/#encoder-state>
    parent: Option<Arc<CommandEncoder>>,

    timestamp_writes: Option<ArcPassTimestampWrites>,

    // Resource binding dedupe state.
    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::ComputePipelineId>,

    /// Resources referenced by this pass, interned into per-type arenas. Moved
    /// out into the [`RunComputePass`](ArcCommand::RunComputePass) command when
    /// the pass ends.
    arenas: ComputeArenas,

    /// Per-slot last-resolved memos for interning; pure recording-time state.
    intern_caches: ComputeInternCaches,

    /// Whether this pass has acquired its pooled backing storage (`base` and
    /// `arenas`) from the encoder's [`EncoderVecPool`] yet. `false` until the
    /// first recorded command triggers [`Self::ensure_storage_acquired`]; a
    /// pass that records nothing never acquires and so never touches the pool.
    storage_acquired: bool,
}

impl_resource_type!(ComputePass);

impl crate::storage::StorageItem for ComputePass {
    type Marker = id::markers::ComputePassEncoder;
}

impl ComputePass {
    /// If the parent command encoder is invalid, the returned pass will be invalid.
    fn new(parent: Arc<CommandEncoder>, desc: ArcComputePassDescriptor) -> Self {
        let ArcComputePassDescriptor {
            label,
            timestamp_writes,
        } = desc;

        // Do not touch the encoder's pool here: the pass starts with fresh,
        // empty (zero-capacity) storage and lazily acquires pooled, warm
        // storage on its first recorded command (see
        // `ensure_storage_acquired`). A pass that records nothing never touches
        // the pool.
        Self {
            base: BasePass::new(&label),
            parent: Some(parent),
            timestamp_writes,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),

            arenas: ComputeArenas::default(),
            intern_caches: ComputeInternCaches::new(),
            storage_acquired: false,
        }
    }

    fn new_invalid(parent: Arc<CommandEncoder>, label: &Label, err: ComputePassError) -> Self {
        Self {
            base: BasePass::new_invalid(label, err),
            parent: Some(parent),
            timestamp_writes: None,
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),

            arenas: ComputeArenas::default(),
            intern_caches: ComputeInternCaches::new(),
            // An invalid pass never records, so it never acquires from the pool.
            storage_acquired: false,
        }
    }

    /// Lazily acquire this pass's pooled backing storage on its first recorded
    /// command. See [`RenderPass::acquire_storage`] for the full rationale;
    /// this is the compute twin.
    #[cold]
    fn acquire_storage(&mut self) {
        debug_assert!(self.base.commands.is_empty());
        debug_assert!(self.base.dynamic_offsets.is_empty());

        let parent = self
            .parent
            .as_ref()
            .expect("open pass must have a parent encoder");
        let device = &parent.device;
        let mut status = parent.data.lock();
        if let Some(pool) = status.vec_pool_mut() {
            let (commands, dynamic_offsets) = pool.acquire_compute(&device.compute_pass_size_hint);
            let arenas = pool.acquire_compute_arenas();
            drop(status);
            self.base.commands = commands;
            self.base.dynamic_offsets = dynamic_offsets;
            self.arenas = arenas;
        }
        // If the encoder has since been invalidated (no pool available), keep
        // the pass's fresh empty storage; its commands are discarded at finish.
        self.storage_acquired = true;
    }

    /// Fast-path wrapper around [`Self::acquire_storage`]: a cheap already-
    /// acquired check, deferring to the `#[cold]` acquire on the first command.
    #[inline]
    fn ensure_storage_acquired(&mut self) {
        if !self.storage_acquired {
            self.acquire_storage();
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.label.as_deref()
    }
}

impl fmt::Debug for ComputePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.parent {
            Some(ref cmd_enc) => write!(f, "ComputePass {{ parent: {} }}", cmd_enc.error_ident()),
            None => write!(f, "ComputePass {{ parent: None }}"),
        }
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputePassDescriptor<'a, PTW = PassTimestampWrites> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PTW>,
}

/// cbindgen:ignore
type ArcComputePassDescriptor<'a> = ComputePassDescriptor<'a, ArcPassTimestampWrites>;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DispatchError {
    #[error("Compute pipeline must be set")]
    MissingPipeline(pass::MissingPipeline),
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
    #[error(
        "Each current dispatch group size dimension ({current:?}) must be less or equal to {limit}"
    )]
    InvalidGroupSize { current: [u32; 3], limit: u32 },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
    #[error("Not all immediate data required by the pipeline has been set via set_immediates (missing byte ranges: {missing})")]
    MissingImmediateData {
        missing: naga::valid::ImmediateSlots,
    },
}

impl WebGpuError for DispatchError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

/// Error encountered when performing a compute pass.
#[derive(Clone, Debug, Error)]
pub enum ComputePassErrorInner {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    #[error("Parent encoder is invalid")]
    InvalidParentEncoder,
    #[error(transparent)]
    DebugGroupError(#[from] DebugGroupError),
    #[error(transparent)]
    BindGroupIndexOutOfRange(#[from] pass::BindGroupIndexOutOfRange),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Indirect buffer offset {0:?} is not a multiple of 4")]
    UnalignedIndirectBufferOffset(BufferAddress),
    #[error("Indirect buffer of {args_size} bytes starting at offset {offset} would overrun buffer of size {buffer_size}")]
    IndirectBufferOverrun {
        args_size: u64,
        offset: u64,
        buffer_size: u64,
    },
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
    #[error(transparent)]
    Bind(#[from] BindError),
    #[error(transparent)]
    ImmediateData(#[from] ImmediateUploadError),
    #[error(transparent)]
    QueryUse(#[from] QueryUseError),
    #[error(transparent)]
    TransitionResources(#[from] TransitionResourcesError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("The compute pass has already been ended and no further commands can be recorded")]
    PassEnded,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    TimestampWrites(#[from] TimestampWritesError),
    // This one is unreachable, but required for generic pass support
    #[error(transparent)]
    InvalidValuesOffset(#[from] pass::InvalidValuesOffset),
}

/// Error encountered when performing a compute pass, stored for later reporting
/// when encoding ends.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct ComputePassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: ComputePassErrorInner,
}

impl From<pass::MissingPipeline> for ComputePassErrorInner {
    fn from(value: pass::MissingPipeline) -> Self {
        Self::Dispatch(DispatchError::MissingPipeline(value))
    }
}

impl<E> MapPassErr<ComputePassError> for E
where
    E: Into<ComputePassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> ComputePassError {
        ComputePassError {
            scope,
            inner: self.into(),
        }
    }
}

impl WebGpuError for ComputePassError {
    fn webgpu_error_type(&self) -> ErrorType {
        let Self { scope: _, inner } = self;
        match inner {
            ComputePassErrorInner::Device(e) => e.webgpu_error_type(),
            ComputePassErrorInner::EncoderState(e) => e.webgpu_error_type(),
            ComputePassErrorInner::DebugGroupError(e) => e.webgpu_error_type(),
            ComputePassErrorInner::DestroyedResource(e) => e.webgpu_error_type(),
            ComputePassErrorInner::ResourceUsageCompatibility(e) => e.webgpu_error_type(),
            ComputePassErrorInner::MissingBufferUsage(e) => e.webgpu_error_type(),
            ComputePassErrorInner::Dispatch(e) => e.webgpu_error_type(),
            ComputePassErrorInner::Bind(e) => e.webgpu_error_type(),
            ComputePassErrorInner::ImmediateData(e) => e.webgpu_error_type(),
            ComputePassErrorInner::QueryUse(e) => e.webgpu_error_type(),
            ComputePassErrorInner::TransitionResources(e) => e.webgpu_error_type(),
            ComputePassErrorInner::MissingFeatures(e) => e.webgpu_error_type(),
            ComputePassErrorInner::MissingDownlevelFlags(e) => e.webgpu_error_type(),
            ComputePassErrorInner::InvalidResource(e) => e.webgpu_error_type(),
            ComputePassErrorInner::TimestampWrites(e) => e.webgpu_error_type(),
            ComputePassErrorInner::InvalidValuesOffset(e) => e.webgpu_error_type(),

            ComputePassErrorInner::InvalidParentEncoder
            | ComputePassErrorInner::BindGroupIndexOutOfRange { .. }
            | ComputePassErrorInner::UnalignedIndirectBufferOffset(_)
            | ComputePassErrorInner::IndirectBufferOverrun { .. }
            | ComputePassErrorInner::PassEnded => ErrorType::Validation,
        }
    }
}

struct State<'scope, 'snatch_guard, 'cmd_enc> {
    pipeline: Option<Arc<ComputePipeline>>,

    pass: pass::PassState<'scope, 'snatch_guard, 'cmd_enc>,

    active_query: Option<(Arc<resource::QuerySet>, u32)>,

    immediates: Vec<u32>,

    /// A bitmask, tracking which 4-byte slots have been written via `set_immediates`.
    /// Checked against the pipeline's required slots before each dispatch.
    immediate_slots_set: naga::valid::ImmediateSlots,

    intermediate_trackers: Tracker,
}

impl<'scope, 'snatch_guard, 'cmd_enc> State<'scope, 'snatch_guard, 'cmd_enc> {
    fn is_ready(&self) -> Result<(), DispatchError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.pass.binder.check_compatibility(pipeline.as_ref())?;
            self.pass.binder.check_late_buffer_bindings()?;
            if !self
                .immediate_slots_set
                .contains(pipeline.immediate_slots_required)
            {
                return Err(DispatchError::MissingImmediateData {
                    missing: pipeline
                        .immediate_slots_required
                        .difference(self.immediate_slots_set),
                });
            }
            Ok(())
        } else {
            Err(DispatchError::MissingPipeline(pass::MissingPipeline))
        }
    }

    /// Flush binding state in preparation for a dispatch.
    ///
    /// # Differences between render and compute passes
    ///
    /// There are differences between the `flush_bindings` implementations for
    /// render and compute passes, because render passes have a single usage
    /// scope for the entire pass, and compute passes have a separate usage
    /// scope for each dispatch.
    ///
    /// For compute passes, bind groups are merged into a fresh usage scope
    /// here, not into the pass usage scope within calls to `set_bind_group`. As
    /// specified by WebGPU, for compute passes, we merge only the bind groups
    /// that are actually used by the pipeline, unlike render passes, which
    /// merge every bind group that is ever set, even if it is not ultimately
    /// used by the pipeline.
    ///
    /// For compute passes, we call `drain_barriers` here, because barriers may
    /// be needed before each dispatch if a previous dispatch had a conflicting
    /// usage. For render passes, barriers are emitted once at the start of the
    /// render pass.
    ///
    /// # Indirect buffer handling
    ///
    /// The `indirect_buffer` argument should be passed for any indirect
    /// dispatch (with or without validation). It will be checked for
    /// conflicting usages according to WebGPU rules. For the purpose of
    /// these rules, the fact that we have actually processed the buffer in
    /// the validation pass is an implementation detail.
    ///
    /// The `track_indirect_buffer` argument should be set when doing indirect
    /// dispatch *without* validation. In this case, the indirect buffer will
    /// be added to the tracker in order to generate any necessary transitions
    /// for that usage.
    ///
    /// When doing indirect dispatch *with* validation, the indirect buffer is
    /// processed by the validation pass and is not used by the actual dispatch.
    /// The indirect validation code handles transitions for the validation
    /// pass.
    fn flush_bindings(
        &mut self,
        indirect_buffer: Option<&Arc<Buffer>>,
        track_indirect_buffer: bool,
    ) -> Result<(), ComputePassErrorInner> {
        for bind_group in self.pass.binder.list_active() {
            unsafe { self.pass.scope.merge_bind_group(&bind_group.used)? };
        }

        // Add the indirect buffer. Because usage scopes are per-dispatch, this
        // is the only place where INDIRECT usage could be added, and it is safe
        // for us to remove it below.
        if let Some(buffer) = indirect_buffer {
            self.pass
                .scope
                .buffers
                .merge_single(buffer, wgt::BufferUses::INDIRECT)?;
        }

        // For compute, usage scopes are associated with each dispatch and not
        // with the pass as a whole. However, because the cost of creating and
        // dropping `UsageScope`s is significant (even with the pool), we
        // add and then remove usage from a single usage scope.

        for bind_group in self.pass.binder.list_active() {
            self.intermediate_trackers
                .set_and_remove_from_usage_scope_sparse(&mut self.pass.scope, &bind_group.used);
        }

        if track_indirect_buffer {
            self.intermediate_trackers
                .buffers
                .set_and_remove_from_usage_scope_sparse(
                    &mut self.pass.scope.buffers,
                    indirect_buffer.map(|buf| buf.tracker_index()),
                );
        } else if let Some(buffer) = indirect_buffer {
            self.pass
                .scope
                .buffers
                .remove_usage(buffer, wgt::BufferUses::INDIRECT);
        }

        flush_bindings_helper(&mut self.pass)?;

        CommandEncoder::drain_barriers(
            self.pass.base.raw_encoder,
            &mut self.intermediate_trackers,
            self.pass.base.snatch_guard,
        );
        Ok(())
    }
}

/// Compute pass version of [`command::transition_resources`](crate::command::transition_resources).
/// See also `State::flush_bindings` for details on the implementation.
fn transition_resources(
    state: &mut State,
    buffer_transitions: Vec<wgt::BufferTransition<Arc<Buffer>>>,
    texture_transitions: Vec<wgt::TextureTransition<Arc<TextureView>>>,
) -> Result<(), TransitionResourcesError> {
    let indices = &state.pass.base.device.tracker_indices;
    state.pass.scope.buffers.set_size(indices.buffers.size());
    state.pass.scope.textures.set_size(indices.textures.size());

    let mut buffer_ids = Vec::with_capacity(buffer_transitions.len());
    let mut textures = TextureViewBindGroupState::new();

    // Process buffer transitions
    for buffer_transition in buffer_transitions {
        buffer_transition
            .buffer
            .same_device(state.pass.base.device)?;

        state
            .pass
            .scope
            .buffers
            .merge_single(&buffer_transition.buffer, buffer_transition.state)?;
        buffer_ids.push(buffer_transition.buffer.tracker_index());
    }

    state
        .intermediate_trackers
        .buffers
        .set_and_remove_from_usage_scope_sparse(&mut state.pass.scope.buffers, buffer_ids);

    // Process texture transitions
    for texture_transition in texture_transitions {
        texture_transition
            .texture
            .same_device(state.pass.base.device)?;

        unsafe {
            state.pass.scope.textures.merge_single(
                &texture_transition.texture.parent,
                texture_transition.selector,
                texture_transition.state,
            )
        }?;

        textures.insert_single(texture_transition.texture, texture_transition.state);
    }

    state
        .intermediate_trackers
        .textures
        .set_and_remove_from_usage_scope_sparse(&mut state.pass.scope.textures, &textures);

    // Record any needed barriers based on tracker data
    CommandEncoder::drain_barriers(
        state.pass.base.raw_encoder,
        &mut state.intermediate_trackers,
        state.pass.base.snatch_guard,
    );
    Ok(())
}

impl CommandEncoder {
    fn begin_compute_pass(
        self: &Arc<Self>,
        desc: &ComputePassDescriptor<'_, PassTimestampWrites<Arc<resource::QuerySet>>>,
    ) -> (ComputePass, Option<CommandEncoderError>) {
        use EncoderStateError as SErr;

        let scope = PassErrorScope::Pass;

        let label = desc.label.as_deref().map(Cow::Borrowed);

        let mut cmd_buf_data = self.data.lock();

        match cmd_buf_data.lock_encoder() {
            Ok(()) => {
                drop(cmd_buf_data);
                if let Err(err) = self.device.check_is_valid() {
                    return (
                        ComputePass::new_invalid(Arc::clone(self), &label, err.map_pass_err(scope)),
                        None,
                    );
                }

                match desc
                    .timestamp_writes
                    .as_ref()
                    .map(|tw| {
                        Self::validate_pass_timestamp_writes::<ComputePassErrorInner>(
                            &self.device,
                            tw,
                        )
                    })
                    .transpose()
                {
                    Ok(timestamp_writes) => {
                        let arc_desc = ArcComputePassDescriptor {
                            label,
                            timestamp_writes,
                        };
                        (ComputePass::new(Arc::clone(self), arc_desc), None)
                    }
                    Err(err) => (
                        ComputePass::new_invalid(Arc::clone(self), &label, err.map_pass_err(scope)),
                        None,
                    ),
                }
            }
            Err(err @ SErr::Locked) => {
                // Attempting to open a new pass while the encoder is locked
                // invalidates the encoder, but does not generate a validation
                // error.
                cmd_buf_data.invalidate(err.clone());
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(Arc::clone(self), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(err @ (SErr::Ended | SErr::Submitted)) => {
                // Attempting to open a new pass after the encode has ended
                // generates an immediate validation error.
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(
                        Arc::clone(self),
                        &label,
                        err.clone().map_pass_err(scope),
                    ),
                    Some(err.into()),
                )
            }
            Err(err @ SErr::Invalid) => {
                // Passes can be opened even on an invalid encoder. Such passes
                // are even valid, but since there's no visible side-effect of
                // the pass being valid and there's no point in storing recorded
                // commands that will ultimately be discarded, we open an
                // invalid pass to save that work.
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(Arc::clone(self), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(SErr::Unlocked) => {
                unreachable!("lock_encoder cannot fail due to the encoder being unlocked")
            }
        }
    }
}

// Running the compute pass.

impl Global {
    /// Creates a compute pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_begin_compute_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_>,
    ) -> (ComputePass, Option<CommandEncoderError>) {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);

        let desc = ComputePassDescriptor {
            label: desc.label.as_deref().map(Cow::Borrowed),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: hub.query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                }),
        };

        cmd_enc.begin_compute_pass(&desc)
    }

    pub fn command_encoder_begin_compute_pass_with_id(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_>,
        id_in: Option<id::ComputePassEncoderId>,
    ) -> (id::ComputePassEncoderId, Option<CommandEncoderError>) {
        let fid = self.hub.compute_passes.prepare(id_in);

        let (pass, err) = self.command_encoder_begin_compute_pass(encoder_id, desc);

        // no lock rank here because only one thread should be using compute pass
        // and it's only used by id variants of compute pass methods on global
        // so no deadlock (or concurrent lock) should happen in practise
        let id = fid.assign(Arc::new(Mutex::new(pass)));

        (id, err)
    }

    pub fn compute_pass_end(&self, pass: &mut ComputePass) -> Result<(), EncoderStateError> {
        profiling::scope!(
            "CommandEncoder::run_compute_pass {}",
            pass.base.label.as_deref().unwrap_or("")
        );

        let cmd_enc = pass.parent.take().ok_or(EncoderStateError::Ended)?;
        let mut cmd_buf_data = cmd_enc.data.lock();

        cmd_buf_data.unlock_encoder()?;

        // Fold this pass's final sizes into the device's capacity hint so that
        // subsequent passes can be pre-sized. Done before `take()` empties the
        // vectors. Cheap relaxed stores; races are harmless (see `PassSizeHint`).
        cmd_enc
            .device
            .compute_pass_size_hint
            .record_finished(pass.base.commands.len(), pass.base.dynamic_offsets.len());

        let base = pass.base.take();

        if let Err(ComputePassError {
            inner:
                ComputePassErrorInner::EncoderState(
                    err @ (EncoderStateError::Locked | EncoderStateError::Ended),
                ),
            scope: _,
        }) = base
        {
            // Most encoding errors are detected and raised within `finish()`.
            //
            // However, we raise a validation error here if the pass was opened
            // within another pass, or on a finished encoder. The latter is
            // particularly important, because in that case reporting errors via
            // `CommandEncoder::finish` is not possible.
            return Err(err.clone());
        }

        cmd_buf_data.push_with(|| -> Result<_, ComputePassError> {
            Ok(ArcCommand::RunComputePass {
                pass: base?,
                // Move the interned resources out to travel with the command
                // stream that indexes them (dropped with the pass on the error
                // path above).
                arenas: core::mem::take(&mut pass.arenas),
                timestamp_writes: pass.timestamp_writes.take(),
            })
        })
    }

    pub fn compute_pass_end_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), EncoderStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_end(&mut pass)
    }

    pub fn compute_pass_drop(&self, pass_id: id::ComputePassEncoderId) {
        self.hub.compute_passes.remove(pass_id);
    }
}

pub(super) fn encode_compute_pass(
    parent_state: &mut EncodingState<InnerCommandEncoder>,
    // The encoder's own vector pool, into which this pass's drained backing
    // vectors are recycled at the end of encoding. Passed as a separate
    // parameter (not through `EncodingState`) so it stays independent of the
    // field reborrows the inner `State` takes out of `parent_state`.
    pool: &mut EncoderVecPool,
    mut base: BasePass<ArcComputeCommand, Infallible>,
    arenas: ComputeArenas,
    mut timestamp_writes: Option<ArcPassTimestampWrites>,
) -> Result<(), ComputePassError> {
    let pass_scope = PassErrorScope::Pass;

    let device = parent_state.device;

    // We automatically keep extending command buffers over time, and because
    // we want to insert a command buffer _before_ what we're about to record,
    // we need to make sure to close the previous one.
    parent_state
        .raw_encoder
        .close_if_open()
        .map_pass_err(pass_scope)?;
    let raw_encoder = parent_state
        .raw_encoder
        .open_pass(base.label.as_deref())
        .map_pass_err(pass_scope)?;

    let mut debug_scope_depth = 0;

    let mut state = State {
        pipeline: None,

        pass: pass::PassState {
            base: EncodingState {
                device,
                raw_encoder,
                tracker: parent_state.tracker,
                buffer_memory_init_actions: parent_state.buffer_memory_init_actions,
                texture_memory_actions: parent_state.texture_memory_actions,
                as_actions: parent_state.as_actions,
                temp_resources: parent_state.temp_resources,
                indirect_draw_validation_resources: parent_state.indirect_draw_validation_resources,
                snatch_guard: parent_state.snatch_guard,
                debug_scope_depth: &mut debug_scope_depth,
                query_set_writes: parent_state.query_set_writes,
                deferred_query_set_resolves: parent_state.deferred_query_set_resolves,
            },
            binder: Binder::new(),
            temp_offsets: Vec::new(),
            dynamic_offset_count: 0,
            pending_discard_init_fixups: SurfacesInDiscardState::new(),
            scope: device.new_usage_scope(),
            string_offset: 0,
        },
        active_query: None,

        immediates: Vec::new(),

        immediate_slots_set: Default::default(),

        intermediate_trackers: Tracker::new(
            device.ordered_buffer_usages,
            device.ordered_texture_usages,
        ),
    };

    let indices = &device.tracker_indices;
    state
        .pass
        .base
        .tracker
        .buffers
        .set_size(indices.buffers.size());
    state
        .pass
        .base
        .tracker
        .textures
        .set_size(indices.textures.size());

    let timestamp_writes: Option<hal::PassTimestampWrites<'_, dyn hal::DynQuerySet>> =
        if let Some(tw) = timestamp_writes.take() {
            tw.query_set.same_device(device).map_pass_err(pass_scope)?;

            record_pass_timestamp_writes(&tw, state.pass.base.query_set_writes);

            let query_set = state
                .pass
                .base
                .tracker
                .query_sets
                .insert_single(tw.query_set);

            // Unlike in render passes we can't delay resetting the query sets since
            // there is no auxiliary pass.
            let range = if let (Some(index_a), Some(index_b)) =
                (tw.beginning_of_pass_write_index, tw.end_of_pass_write_index)
            {
                Some(index_a.min(index_b)..index_a.max(index_b) + 1)
            } else {
                tw.beginning_of_pass_write_index
                    .or(tw.end_of_pass_write_index)
                    .map(|i| i..i + 1)
            };
            let raw_query_set = query_set
                .try_raw(parent_state.snatch_guard)
                .map_pass_err(pass_scope)?;
            // Range should always be Some, both values being None should lead to a validation error.
            // But no point in erroring over that nuance here!
            if let Some(range) = range {
                unsafe {
                    state
                        .pass
                        .base
                        .raw_encoder
                        .reset_queries(raw_query_set, range);
                }
            }

            Some(hal::PassTimestampWrites {
                query_set: raw_query_set,
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            })
        } else {
            None
        };

    let hal_desc = hal::ComputePassDescriptor {
        label: hal_label(base.label.as_deref(), device.instance_flags),
        timestamp_writes,
    };

    unsafe {
        state.pass.base.raw_encoder.begin_compute_pass(&hal_desc);
    }

    for command in base.commands.drain(..) {
        match command {
            ArcComputeCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => {
                let scope = PassErrorScope::SetBindGroup;
                // Compute never merges bind-group usages into the pass scope at
                // set time (that happens per dispatch in `flush_bindings`), so
                // `merge_into_scope` is always false. The (4a) scope-merge
                // elision is render-only. The submit tracker still records the
                // bind group; compute passes are short, so it is inserted per
                // command as before.
                let bind_group = bind_group.map(|i| &arenas.bind_groups.entry(i).arc);
                pass::set_bind_group::<ComputePassErrorInner>(
                    &mut state.pass,
                    &base.dynamic_offsets,
                    index,
                    num_dynamic_offsets,
                    bind_group,
                    false,
                    true,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::SetPipeline(pipeline) => {
                let scope = PassErrorScope::SetPipelineCompute;
                set_pipeline(&mut state, device, arenas.pipelines.get(pipeline))
                    .map_pass_err(scope)?;
            }
            ArcComputeCommand::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => {
                let scope = PassErrorScope::SetImmediate;
                pass::set_immediates::<ComputePassErrorInner, _>(
                    &mut state.pass,
                    &base.immediates_data,
                    offset,
                    size_bytes,
                    Some(values_offset),
                    |data_slice| {
                        let offset_in_elements = (offset / wgt::IMMEDIATE_DATA_ALIGNMENT) as usize;
                        let size_in_elements =
                            (size_bytes / wgt::IMMEDIATE_DATA_ALIGNMENT) as usize;
                        state.immediates[offset_in_elements..][..size_in_elements]
                            .copy_from_slice(data_slice);
                    },
                )
                .map_pass_err(scope)?;
                state.immediate_slots_set |=
                    naga::valid::ImmediateSlots::from_range(offset, size_bytes);
            }
            ArcComputeCommand::DispatchWorkgroups(groups) => {
                let scope = PassErrorScope::Dispatch { indirect: false };
                dispatch_workgroups(&mut state, groups).map_pass_err(scope)?;
            }
            ArcComputeCommand::DispatchWorkgroupsIndirect { buffer, offset } => {
                let scope = PassErrorScope::Dispatch { indirect: true };
                dispatch_workgroups_indirect(&mut state, device, buffer, offset)
                    .map_pass_err(scope)?;
            }
            ArcComputeCommand::PushDebugGroup { color: _, len } => {
                pass::push_debug_group(&mut state.pass, &base.string_data, len);
            }
            ArcComputeCommand::PopDebugGroup => {
                let scope = PassErrorScope::PopDebugGroup;
                pass::pop_debug_group::<ComputePassErrorInner>(&mut state.pass)
                    .map_pass_err(scope)?;
            }
            ArcComputeCommand::InsertDebugMarker { color: _, len } => {
                pass::insert_debug_marker(&mut state.pass, &base.string_data, len);
            }
            ArcComputeCommand::WriteTimestamp {
                query_set,
                query_index,
            } => {
                let scope = PassErrorScope::WriteTimestamp;
                pass::write_timestamp::<ComputePassErrorInner>(
                    &mut state.pass,
                    None, // compute passes do not attempt to coalesce query resets
                    arenas.query_sets.get(query_set),
                    query_index,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => {
                let scope = PassErrorScope::BeginPipelineStatisticsQuery;
                validate_and_begin_pipeline_statistics_query(
                    arenas.query_sets.get(query_set).clone(),
                    state.pass.base.raw_encoder,
                    &mut state.pass.base.tracker.query_sets,
                    device,
                    query_index,
                    None,
                    &mut state.active_query,
                    state.pass.base.snatch_guard,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::EndPipelineStatisticsQuery => {
                let scope = PassErrorScope::EndPipelineStatisticsQuery;
                end_pipeline_statistics_query(
                    state.pass.base.raw_encoder,
                    &mut state.active_query,
                    state.pass.base.snatch_guard,
                    state.pass.base.query_set_writes,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::TransitionResources {
                buffer_transitions,
                texture_transitions,
            } => {
                let scope = PassErrorScope::TransitionResources;
                transition_resources(&mut state, buffer_transitions, texture_transitions)
                    .map_pass_err(scope)?;
            }
        }
    }

    // The command stream has been fully drained above, leaving `base.commands`
    // empty but still holding its grown capacity. `base.dynamic_offsets`, in
    // contrast, is consumed by reference (each `SetBindGroup` slices into it)
    // and so is still populated here; clear it (an O(1) `Vec<u32>` truncation,
    // no Drop glue) so only empty vectors reach the pool. Both then retain
    // their grown capacity, which we recycle into the encoder's pool so the
    // next pass on this encoder can reuse the storage instead of reallocating
    // and re-faulting it. (On the error paths below the pass simply drops,
    // freeing them, as before — not worth the complexity to recycle those rare
    // cases.)
    //
    // `base` and `arenas` are owned locals here, so both the command/offset
    // vectors and the arena backing vectors are *moved* into the pool rather
    // than reset with `mem::take` — the pass they came from is gone.
    base.dynamic_offsets.clear();
    let BasePass {
        commands,
        dynamic_offsets,
        ..
    } = base;
    pool.release_compute(commands, dynamic_offsets);
    // Recycle the arena backing vectors now that replay is complete; see the
    // render-pass equivalent for the lifetime argument.
    pool.release_compute_arenas(arenas);

    if *state.pass.base.debug_scope_depth > 0 {
        Err(
            ComputePassErrorInner::DebugGroupError(DebugGroupError::MissingPop)
                .map_pass_err(pass_scope),
        )?;
    }

    unsafe {
        state.pass.base.raw_encoder.end_compute_pass();
    }

    let State {
        pass: pass::PassState {
            pending_discard_init_fixups,
            ..
        },
        intermediate_trackers,
        ..
    } = state;

    // Stop the current command encoder.
    parent_state.raw_encoder.close().map_pass_err(pass_scope)?;

    // Create a new command encoder, which we will insert _before_ the body of the compute pass.
    //
    // Use that buffer to insert barriers and clear discarded images.
    let transit = parent_state
        .raw_encoder
        .open_pass(hal_label(
            Some("(wgpu internal) Pre Pass"),
            device.instance_flags,
        ))
        .map_pass_err(pass_scope)?;
    fixup_discarded_surfaces(
        pending_discard_init_fixups.into_iter(),
        transit,
        &mut parent_state.tracker.textures,
        device,
        parent_state.snatch_guard,
    );
    CommandEncoder::insert_barriers_from_tracker(
        transit,
        parent_state.tracker,
        &intermediate_trackers,
        parent_state.snatch_guard,
    );
    // Close the command encoder, and swap it with the previous.
    parent_state
        .raw_encoder
        .close_and_swap()
        .map_pass_err(pass_scope)?;

    Ok(())
}

fn set_pipeline(
    state: &mut State,
    device: &Arc<Device>,
    pipeline: &Arc<ComputePipeline>,
) -> Result<(), ComputePassErrorInner> {
    let _ = device;
    // `same_device` was already checked when the pipeline was interned.

    state.pipeline = Some(pipeline.clone());

    let pipeline = state
        .pass
        .base
        .tracker
        .compute_pipelines
        .insert_single(pipeline.clone())
        .clone();

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .set_compute_pipeline(pipeline.raw()?);
    }

    // Rebind resources
    let pipeline_layout = pipeline.layout()?;
    pass::change_pipeline_layout::<ComputePassErrorInner, _>(
        &mut state.pass,
        pipeline_layout,
        &pipeline.late_sized_buffer_groups,
        || {
            // This only needs to be here for compute pipelines because they use immediates for
            // validating indirect draws.
            state.immediates.clear();
            // Note that can only be one range for each stage. See the `MoreThanOneImmediateRangePerStage` error.
            if pipeline_layout.immediate_size != 0 {
                // Note that non-0 range start doesn't work anyway https://github.com/gfx-rs/wgpu/issues/4502
                let len = pipeline_layout.immediate_size as usize
                    / wgt::IMMEDIATE_DATA_ALIGNMENT as usize;
                state.immediates.extend(core::iter::repeat_n(0, len));
            }
        },
    )
}

fn dispatch_workgroups(state: &mut State, groups: [u32; 3]) -> Result<(), ComputePassErrorInner> {
    api_log!("ComputePass::dispatch {groups:?}");

    state.is_ready()?;

    state.flush_bindings(None, false)?;

    let groups_size_limit = state
        .pass
        .base
        .device
        .limits
        .max_compute_workgroups_per_dimension;

    if groups.iter().copied().any(|g| g > groups_size_limit) {
        return Err(ComputePassErrorInner::Dispatch(
            DispatchError::InvalidGroupSize {
                current: groups,
                limit: groups_size_limit,
            },
        ));
    }

    unsafe {
        state.pass.base.raw_encoder.dispatch_workgroups(groups);
    }
    Ok(())
}

fn dispatch_workgroups_indirect(
    state: &mut State,
    device: &Arc<Device>,
    buffer: Arc<Buffer>,
    offset: u64,
) -> Result<(), ComputePassErrorInner> {
    api_log!("ComputePass::dispatch_indirect");

    buffer.same_device(device)?;

    state.is_ready()?;

    state
        .pass
        .base
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    buffer.check_usage(wgt::BufferUsages::INDIRECT)?;

    if !offset.is_multiple_of(4) {
        return Err(ComputePassErrorInner::UnalignedIndirectBufferOffset(offset));
    }

    let args_size = size_of::<wgt::DispatchIndirectArgs>() as u64;
    if buffer.size < args_size || buffer.size - args_size < offset {
        return Err(ComputePassErrorInner::IndirectBufferOverrun {
            args_size,
            offset,
            buffer_size: buffer.size,
        });
    }

    buffer.check_destroyed(state.pass.base.snatch_guard)?;

    let stride = 3 * 4; // 3 integers, x/y/z group size
    if let Some(action) = buffer.create_init_action(
        offset..(offset + stride),
        MemoryInitKind::NeedsInitializedMemory,
    ) {
        state.pass.base.buffer_memory_init_actions.push(action);
    }

    if let Some(ref indirect_validation) = state.pass.base.device.indirect_validation {
        let params = indirect_validation.dispatch.params(
            &state.pass.base.device.limits,
            offset,
            buffer.size,
        );

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .set_compute_pipeline(params.pipeline);
        }

        unsafe {
            state.pass.base.raw_encoder.set_immediates(
                params.pipeline_layout,
                0,
                &[params.offset_remainder as u32 / 4],
            );
        }

        unsafe {
            state.pass.base.raw_encoder.set_bind_group(
                params.pipeline_layout,
                0,
                params.dst_bind_group,
                &[],
            );
        }
        unsafe {
            state.pass.base.raw_encoder.set_bind_group(
                params.pipeline_layout,
                1,
                buffer
                    .indirect_validation_bind_groups
                    .get(state.pass.base.snatch_guard)
                    .unwrap()
                    .dispatch
                    .as_ref(),
                &[params.aligned_offset as u32],
            );
        }

        let src_transition = state
            .intermediate_trackers
            .buffers
            .set_single(&buffer, wgt::BufferUses::STORAGE_READ_ONLY);
        let src_barrier = src_transition
            .map(|transition| transition.into_hal(&buffer, state.pass.base.snatch_guard));
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(src_barrier.as_slice());
        }

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params.dst_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::INDIRECT,
                        to: wgt::BufferUses::STORAGE_READ_WRITE,
                    },
                }]);
        }

        unsafe {
            state.pass.base.raw_encoder.dispatch_workgroups([1, 1, 1]);
        }

        // reset state
        {
            let pipeline = state.pipeline.as_ref().unwrap();

            unsafe {
                state
                    .pass
                    .base
                    .raw_encoder
                    .set_compute_pipeline(pipeline.raw()?);
            }

            let pipeline_layout = pipeline.layout()?;

            if !state.immediates.is_empty() {
                unsafe {
                    state.pass.base.raw_encoder.set_immediates(
                        pipeline_layout.raw()?,
                        0,
                        &state.immediates,
                    );
                }
            }

            for (i, group, dynamic_offsets) in state.pass.binder.list_valid() {
                let raw_bg = group.try_raw(state.pass.base.snatch_guard)?;
                unsafe {
                    state.pass.base.raw_encoder.set_bind_group(
                        pipeline_layout.raw()?,
                        i as u32,
                        raw_bg,
                        dynamic_offsets,
                    );
                }
            }
        }

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params.dst_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::STORAGE_READ_WRITE,
                        to: wgt::BufferUses::INDIRECT,
                    },
                }]);
        }

        state.flush_bindings(Some(&buffer), false)?;
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .dispatch_workgroups_indirect(params.dst_buffer, 0);
        }
    } else {
        state.flush_bindings(Some(&buffer), true)?;

        let buf_raw = buffer.try_raw(state.pass.base.snatch_guard)?;
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .dispatch_workgroups_indirect(buf_raw, offset);
        }
    }

    Ok(())
}

/// Intern an `Arc`-carrying compute [`BasePass`] (as produced by trace replay
/// in the `player`) into the arena form used at runtime. Inverse of
/// [`resolve_compute_base_pass_to_arc`].
#[cfg(feature = "replay")]
#[doc(hidden)]
pub fn intern_compute_base_pass_from_arc(
    base: BasePass<crate::command::ComputeCommand<crate::command::ArcReferences>, Infallible>,
) -> (BasePass<ArcComputeCommand, Infallible>, ComputeArenas) {
    use crate::command::ComputeCommand as C;
    let mut arenas = ComputeArenas::default();
    let commands = base
        .commands
        .into_iter()
        .map(|cmd| match cmd {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|bg| arenas.bind_groups.push(bg)),
            },
            C::SetPipeline(p) => C::SetPipeline(arenas.pipelines.push(p)),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::DispatchWorkgroups(g) => C::DispatchWorkgroups(g),
            // Buffers are not interned: the `Arc` passes straight through.
            C::DispatchWorkgroupsIndirect { buffer, offset } => {
                C::DispatchWorkgroupsIndirect { buffer, offset }
            }
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: arenas.query_sets.push(query_set),
                query_index,
            },
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: arenas.query_sets.push(query_set),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
            C::TransitionResources {
                buffer_transitions,
                texture_transitions,
            } => C::TransitionResources {
                buffer_transitions,
                texture_transitions,
            },
        })
        .collect();
    (
        BasePass {
            label: base.label,
            error: None,
            commands,
            dynamic_offsets: base.dynamic_offsets,
            string_data: base.string_data,
            immediates_data: base.immediates_data,
        },
        arenas,
    )
}

/// Resolve every command in an arena-indexed compute [`BasePass`] back into one
/// carrying resolved `Arc`s, for trace recording.
#[cfg(feature = "trace")]
pub(crate) fn resolve_compute_base_pass_to_arc(
    arenas: &ComputeArenas,
    base: &BasePass<ArcComputeCommand, Infallible>,
) -> BasePass<crate::command::ComputeCommand<crate::command::ArcReferences>, Infallible> {
    use crate::command::ComputeCommand as C;
    let commands = base
        .commands
        .iter()
        .map(|cmd| match *cmd {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                ref bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|i| arenas.bind_groups.get(i).clone()),
            },
            C::SetPipeline(p) => C::SetPipeline(arenas.pipelines.get(p).clone()),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::DispatchWorkgroups(g) => C::DispatchWorkgroups(g),
            // Buffers are not interned: clone the `Arc` straight off the command.
            C::DispatchWorkgroupsIndirect { ref buffer, offset } => C::DispatchWorkgroupsIndirect {
                buffer: buffer.clone(),
                offset,
            },
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: arenas.query_sets.get(query_set).clone(),
                query_index,
            },
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: arenas.query_sets.get(query_set).clone(),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
            C::TransitionResources {
                ref buffer_transitions,
                ref texture_transitions,
            } => C::TransitionResources {
                buffer_transitions: buffer_transitions.clone(),
                texture_transitions: texture_transitions.clone(),
            },
        })
        .collect();
    BasePass {
        label: base.label.clone(),
        error: None,
        commands,
        dynamic_offsets: base.dynamic_offsets.clone(),
        string_data: base.string_data.clone(),
        immediates_data: base.immediates_data.clone(),
    }
}

// Record-time interning helpers (compute). See the render-pass equivalents in
// `render.rs`; these differ only in the error type. Resources are resolved,
// validity- and `same_device`-checked once at record time and moved into the
// arena; a per-slot cache serves repeats with no resolve.

fn intern_bind_group_compute(
    hub: &crate::hub::Hub,
    arena: &mut crate::command::BindGroupArena,
    cache: Option<&mut crate::command::InternCache<id::BindGroupId, BindGroupArenaIndex>>,
    device: &Device,
    id: id::BindGroupId,
) -> Result<BindGroupArenaIndex, ComputePassErrorInner> {
    if let Some(cache) = cache.as_deref() {
        if let Some(index) = cache.get(id) {
            return Ok(index);
        }
    }
    let bind_group = hub.bind_groups.get(id).get()?;
    bind_group.same_device(device)?;
    let index = arena.push(bind_group);
    if let Some(cache) = cache {
        cache.store(id, index);
    }
    Ok(index)
}

fn intern_compute_pipeline(
    hub: &crate::hub::Hub,
    arena: &mut crate::command::ComputePipelineArena,
    cache: &mut crate::command::InternCache<id::ComputePipelineId, ComputePipelineArenaIndex>,
    device: &Device,
    id: id::ComputePipelineId,
) -> Result<ComputePipelineArenaIndex, ComputePassErrorInner> {
    if let Some(index) = cache.get(id) {
        return Ok(index);
    }
    let pipeline = hub.compute_pipelines.get(id);
    pipeline.check_valid()?;
    pipeline.same_device(device)?;
    let index = arena.push(pipeline);
    cache.store(id, index);
    Ok(index)
}

fn intern_query_set_compute(
    hub: &crate::hub::Hub,
    arena: &mut crate::command::QuerySetArena,
    device: &Device,
    id: id::QuerySetId,
) -> Result<QuerySetArenaIndex, ComputePassErrorInner> {
    let query_set = hub.query_sets.get(id);
    query_set.check_is_valid()?;
    query_set.same_device(device)?;
    Ok(arena.push(query_set))
}

// Recording a compute pass.
//
// The only error that should be returned from these methods is
// `EncoderStateError::Ended`, when the pass has already ended and an immediate
// validation error is raised.
//
// All other errors should be stored in the pass for later reporting when
// `CommandEncoder.finish()` is called.
//
// The `pass_try!` macro should be used to handle errors appropriately. Note
// that the `pass_try!` and `pass_base!` macros may return early from the
// function that invokes them, like the `?` operator.
impl Global {
    pub fn compute_pass_set_bind_group(
        &self,
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetBindGroup;

        // This statement will return an error if the pass is ended. It's
        // important the error check comes before the early-out for
        // `set_and_check_redundant`.
        let base = pass_base!(pass, scope);

        if pass.current_bind_groups.set_and_check_redundant(
            bind_group_id,
            index,
            &mut base.dynamic_offsets,
            offsets,
        ) {
            return Ok(());
        }

        // Borrow the device rather than cloning the `Arc` — this runs per
        // `set_bind_group` command, so a clone here would be per-command churn.
        // `pass.parent`, `pass.arenas` and `pass.intern_caches` are disjoint
        // fields, so the borrows do not conflict. `pass.parent` is `Some` here
        // (`pass_base!` returned early otherwise).
        let device = &pass.parent.as_ref().unwrap().device;
        let bind_group = match bind_group_id {
            Some(bind_group_id) => {
                let interned = intern_bind_group_compute(
                    &self.hub,
                    &mut pass.arenas.bind_groups,
                    pass.intern_caches.bind_groups.get_mut(index as usize),
                    device,
                    bind_group_id,
                );
                Some(pass_try!(base, scope, interned))
            }
            None => None,
        };

        base.commands.push(ArcComputeCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn compute_pass_set_bind_group_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_set_bind_group(&mut pass, index, bind_group_id, offsets)
    }

    pub fn compute_pass_set_pipeline(
        &self,
        pass: &mut ComputePass,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), PassStateError> {
        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);

        let scope = PassErrorScope::SetPipelineCompute;

        // This statement will return an error if the pass is ended.
        // Its important the error check comes before the early-out for `redundant`.
        let base = pass_base!(pass, scope);

        if redundant {
            return Ok(());
        }

        // Borrow (don't clone) the device: this runs per `set_pipeline` command.
        let device = &pass.parent.as_ref().unwrap().device;
        let interned = intern_compute_pipeline(
            &self.hub,
            &mut pass.arenas.pipelines,
            &mut pass.intern_caches.pipeline,
            device,
            pipeline_id,
        );
        let compute_pipeline = pass_try!(base, scope, interned);

        base.commands
            .push(ArcComputeCommand::SetPipeline(compute_pipeline));

        Ok(())
    }

    pub fn compute_pass_set_pipeline_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_set_pipeline(&mut pass, pipeline_id)
    }

    pub fn compute_pass_set_immediates(
        &self,
        pass: &mut ComputePass,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetImmediate;
        let base = pass_base!(pass, scope);

        let size_bytes = pass_try!(
            base,
            scope,
            u32::try_from(data.len()).map_err(|_| ImmediateUploadError::ImmediateOutOfMemory)
        );
        pass_try!(
            base,
            scope,
            pass::validate_immediates_alignment(offset, size_bytes)
        );

        let values_offset = base.immediates_data.len().try_into().unwrap();

        base.immediates_data.extend(
            data.chunks_exact(wgt::IMMEDIATE_DATA_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

        base.commands.push(ArcComputeCommand::SetImmediate {
            offset,
            size_bytes: data.len() as u32,
            values_offset,
        });

        Ok(())
    }

    pub fn compute_pass_set_immediates_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_set_immediates(&mut pass, offset, data)
    }

    pub fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut ComputePass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::Dispatch { indirect: false };

        pass_base!(pass, scope)
            .commands
            .push(ArcComputeCommand::DispatchWorkgroups([
                groups_x, groups_y, groups_z,
            ]));

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_dispatch_workgroups(&mut pass, groups_x, groups_y, groups_z)
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let hub = &self.hub;
        let scope = PassErrorScope::Dispatch { indirect: true };
        let base = pass_base!(pass, scope);

        // Buffers are not interned: resolve the id to its `Arc` and carry it
        // directly on the command (`same_device` is checked at replay).
        let buffer = pass_try!(base, scope, hub.buffers.get(buffer_id).get());

        base.commands
            .push(ArcComputeCommand::DispatchWorkgroupsIndirect { buffer, offset });

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups_indirect_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_dispatch_workgroups_indirect(&mut pass, buffer_id, offset)
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PushDebugGroup);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcComputeCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn compute_pass_push_debug_group_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_push_debug_group(&mut pass, label, color)
    }

    pub fn compute_pass_pop_debug_group(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PopDebugGroup);

        base.commands.push(ArcComputeCommand::PopDebugGroup);

        Ok(())
    }

    pub fn compute_pass_pop_debug_group_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_pop_debug_group(&mut pass)
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::InsertDebugMarker);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcComputeCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn compute_pass_insert_debug_marker_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_insert_debug_marker(&mut pass, label, color)
    }

    pub fn compute_pass_write_timestamp(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::WriteTimestamp;
        let base = pass_base!(pass, scope);

        let query_set = pass_try!(
            base,
            scope,
            intern_query_set_compute(
                &self.hub,
                &mut pass.arenas.query_sets,
                &pass.parent.as_ref().unwrap().device,
                query_set_id,
            ),
        );

        base.commands.push(ArcComputeCommand::WriteTimestamp {
            query_set,
            query_index,
        });

        Ok(())
    }

    pub fn compute_pass_write_timestamp_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_write_timestamp(&mut pass, query_set_id, query_index)
    }

    pub fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::BeginPipelineStatisticsQuery;
        let base = pass_base!(pass, scope);

        let query_set = pass_try!(
            base,
            scope,
            intern_query_set_compute(
                &self.hub,
                &mut pass.arenas.query_sets,
                &pass.parent.as_ref().unwrap().device,
                query_set_id,
            ),
        );

        base.commands
            .push(ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            });

        Ok(())
    }

    pub fn compute_pass_begin_pipeline_statistics_query_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_begin_pipeline_statistics_query(&mut pass, query_set_id, query_index)
    }

    pub fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), PassStateError> {
        pass_base!(pass, PassErrorScope::EndPipelineStatisticsQuery)
            .commands
            .push(ArcComputeCommand::EndPipelineStatisticsQuery);

        Ok(())
    }

    pub fn compute_pass_end_pipeline_statistics_query_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_end_pipeline_statistics_query(&mut pass)
    }

    pub fn compute_pass_transition_resources(
        &self,
        pass: &mut ComputePass,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<id::BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<id::TextureViewId>>,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::TransitionResources;
        let base = pass_base!(pass, scope);

        let hub = &self.hub;

        // Buffers are not interned: resolve each id to its `Arc` and carry it
        // directly on the command (`same_device` is checked at replay).
        let buffer_transitions = pass_try!(
            base,
            scope,
            buffer_transitions
                .map(|buffer_transition| -> Result<_, InvalidResourceError> {
                    Ok(wgt::BufferTransition {
                        buffer: hub.buffers.get(buffer_transition.buffer).get()?,
                        state: buffer_transition.state,
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        );

        let texture_transitions = pass_try!(
            base,
            scope,
            texture_transitions
                .map(|texture_transition| -> Result<_, InvalidResourceError> {
                    let texture_view = hub.texture_views.get(texture_transition.texture);
                    texture_view.check_valid()?;
                    Ok(wgt::TextureTransition {
                        texture: texture_view,
                        selector: texture_transition.selector,
                        state: texture_transition.state,
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        );

        base.commands.push(ArcComputeCommand::TransitionResources {
            buffer_transitions,
            texture_transitions,
        });

        Ok(())
    }
}
