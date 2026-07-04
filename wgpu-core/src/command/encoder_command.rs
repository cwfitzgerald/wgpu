use core::{convert::Infallible, num::NonZero};

use alloc::{string::String, sync::Arc, vec::Vec};
#[cfg(feature = "serde")]
use macro_rules_attribute::attribute_alias;

use crate::{
    command::ColorAttachments,
    id,
    instance::Surface,
    resource::{Buffer, QuerySet, Texture},
};

pub trait ReferenceType: Clone + core::fmt::Debug {
    type Buffer: Clone + core::fmt::Debug;
    type Surface: Clone; // Surface does not implement Debug, although it probably could.
    type Texture: Clone + core::fmt::Debug;
    type TextureView: Clone + core::fmt::Debug;
    type ExternalTexture: Clone + core::fmt::Debug;
    type QuerySet: Clone + core::fmt::Debug;
    type BindGroup: Clone + core::fmt::Debug;
    type RenderPipeline: Clone + core::fmt::Debug;
    type RenderBundle: Clone + core::fmt::Debug;
    type ComputePipeline: Clone + core::fmt::Debug;
    type Blas: Clone + core::fmt::Debug;
    type Tlas: Clone + core::fmt::Debug;

    /// The [`ReferenceType`] used for the *nested* render- and compute-pass
    /// command streams carried by [`Command::RunRenderPass`] and
    /// [`Command::RunComputePass`].
    ///
    /// A top-level [`Command`] stream and the pass command streams nested
    /// inside it can use *different* reference schemes. In particular
    /// [`ArcReferences`] (resources held as `Arc`s in top-level transfer
    /// commands) maps to [`ArenaReferences`] (resources interned into per-pass
    /// arenas and referenced by copyable index) for its pass streams. The
    /// [`IdReferences`] and `PointerReferences` schemes map to themselves.
    type PassReferences: ReferenceType;

    /// The arenas that keep a [`RunRenderPass`](Command::RunRenderPass)'s
    /// interned resources alive alongside its command stream.
    ///
    /// This is [`RenderArenas`](crate::command::RenderArenas) for the
    /// [`ArcReferences`] scheme, where pass resources are interned; it is the
    /// zero-sized [`NoArenas`] for the [`IdReferences`]/`PointerReferences`
    /// schemes, whose pass commands carry ids/pointers directly and need no
    /// arena.
    type RenderPassArenas: Clone + core::fmt::Debug + Default;

    /// The arenas for a [`RunComputePass`](Command::RunComputePass). See
    /// [`RenderPassArenas`](Self::RenderPassArenas).
    type ComputePassArenas: Clone + core::fmt::Debug + Default;
}

/// A zero-sized stand-in for [`ReferenceType::RenderPassArenas`] /
/// [`ReferenceType::ComputePassArenas`] under reference schemes whose pass
/// commands need no arena (ids and pointers carry their resources directly).
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoArenas;

/// Reference wgpu objects via numeric IDs assigned by [`crate::identity::IdentityManager`].
#[derive(Clone, Debug)]
pub struct IdReferences;

/// Reference wgpu objects via the integer value of pointers.
///
/// This is used for trace recording and playback. Recording stores the pointer
/// value of `Arc` references in the trace. Playback uses the integer values
/// as keys to a `HashMap`.
#[cfg(any(feature = "trace", feature = "replay"))]
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct PointerReferences;

/// Reference wgpu objects via `Arc`s.
#[derive(Clone, Debug)]
pub struct ArcReferences;

/// Reference wgpu objects via copyable indices into per-pass [arenas].
///
/// This is the [`ReferenceType`] used for the render- and compute-pass command
/// streams recorded inside an [`ArcReferences`] top-level command stream (see
/// [`ReferenceType::PassReferences`]). Resources are interned once into the
/// arenas that travel alongside the command stream, and each command records a
/// small index into the relevant arena instead of an `Arc`.
///
/// Several families of resources are *not* interned and stay as `Arc`s even
/// under this scheme:
///
/// * [`Buffer`](Self::Buffer)s (vertex/index/indirect/count buffers, and a
///   compute pass's `TransitionResources` buffers) — the reference workload
///   references thousands of distinct buffers per pass with no intra-pass reuse,
///   so a resolve memo would always miss and interning would only add overhead.
///   Buffer commands carry their `Arc`s directly.
/// * [`RenderBundle`](Self::RenderBundle)s referenced by `ExecuteBundle` — rare,
///   and interning them would create a chicken-and-egg between a bundle's arena
///   and bundles it references.
/// * [`TextureView`](Self::TextureView)s referenced by a compute pass's
///   `TransitionResources` — the only texture references in any pass command
///   stream, and rare.
///
/// The remaining associated types (`Surface`, `Texture`, `ExternalTexture`,
/// `Blas`, `Tlas`) never appear in a `RenderCommand` or `ComputeCommand`, so
/// their choice here is immaterial; they are `Arc`s for consistency.
///
/// [arenas]: crate::command::RenderArenas
#[derive(Clone, Debug)]
pub struct ArenaReferences;

impl ReferenceType for IdReferences {
    type Buffer = id::BufferId;
    type Surface = id::SurfaceId;
    type Texture = id::TextureId;
    type TextureView = id::TextureViewId;
    type ExternalTexture = id::ExternalTextureId;
    type QuerySet = id::QuerySetId;
    type BindGroup = id::BindGroupId;
    type RenderPipeline = id::RenderPipelineId;
    type RenderBundle = id::RenderBundleId;
    type ComputePipeline = id::ComputePipelineId;
    type Blas = id::BlasId;
    type Tlas = id::TlasId;

    type PassReferences = IdReferences;
    type RenderPassArenas = NoArenas;
    type ComputePassArenas = NoArenas;
}

impl ReferenceType for ArenaReferences {
    type Buffer = Arc<Buffer>;
    type Surface = Arc<Surface>;
    type Texture = Arc<Texture>;
    type TextureView = Arc<crate::resource::TextureView>;
    type ExternalTexture = Arc<crate::resource::ExternalTexture>;
    type QuerySet = crate::command::QuerySetArenaIndex;
    type BindGroup = crate::command::BindGroupArenaIndex;
    type RenderPipeline = crate::command::RenderPipelineArenaIndex;
    type RenderBundle = Arc<crate::command::RenderBundle>;
    type ComputePipeline = crate::command::ComputePipelineArenaIndex;
    type Blas = Arc<crate::resource::Blas>;
    type Tlas = Arc<crate::resource::Tlas>;

    // A pass command stream under `ArenaReferences` recurses no further: pass
    // streams don't nest, so this maps to itself.
    type PassReferences = ArenaReferences;
    type RenderPassArenas = crate::command::RenderArenas;
    type ComputePassArenas = crate::command::ComputeArenas;
}

#[cfg(any(feature = "trace", feature = "replay"))]
impl ReferenceType for PointerReferences {
    type Buffer = id::PointerId<id::markers::Buffer>;
    type Surface = id::PointerId<id::markers::Surface>;
    type Texture = id::PointerId<id::markers::Texture>;
    type TextureView = id::PointerId<id::markers::TextureView>;
    type ExternalTexture = id::PointerId<id::markers::ExternalTexture>;
    type QuerySet = id::PointerId<id::markers::QuerySet>;
    type BindGroup = id::PointerId<id::markers::BindGroup>;
    type RenderPipeline = id::PointerId<id::markers::RenderPipeline>;
    type RenderBundle = id::PointerId<id::markers::RenderBundle>;
    type ComputePipeline = id::PointerId<id::markers::ComputePipeline>;
    type Blas = id::PointerId<id::markers::Blas>;
    type Tlas = id::PointerId<id::markers::Tlas>;

    type PassReferences = PointerReferences;
    type RenderPassArenas = NoArenas;
    type ComputePassArenas = NoArenas;
}

impl ReferenceType for ArcReferences {
    type Buffer = Arc<Buffer>;
    type Surface = Arc<Surface>;
    type Texture = Arc<Texture>;
    type TextureView = Arc<crate::resource::TextureView>;
    type ExternalTexture = Arc<crate::resource::ExternalTexture>;
    type QuerySet = Arc<QuerySet>;
    type BindGroup = Arc<crate::binding_model::BindGroup>;
    type RenderPipeline = Arc<crate::pipeline::RenderPipeline>;
    type RenderBundle = Arc<crate::command::RenderBundle>;
    type ComputePipeline = Arc<crate::pipeline::ComputePipeline>;
    type Blas = Arc<crate::resource::Blas>;
    type Tlas = Arc<crate::resource::Tlas>;

    // Top-level transfer commands hold `Arc`s, but the render/compute pass
    // command streams nested inside them intern their resources into arenas.
    type PassReferences = ArenaReferences;
    type RenderPassArenas = crate::command::RenderArenas;
    type ComputePassArenas = crate::command::ComputeArenas;
}

#[cfg(feature = "serde")]
attribute_alias! {
    #[apply(serde_object_reference_struct)] =
    #[derive(serde::Serialize, serde::Deserialize)]
    #[serde(bound =
         "R::Buffer: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Surface: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Texture: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::TextureView: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::ExternalTexture: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::QuerySet: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::BindGroup: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::RenderPipeline: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::RenderBundle: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::ComputePipeline: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Blas: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Tlas: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          wgt::BufferTransition<R::Buffer>: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          wgt::TextureTransition<R::Texture>: serde::Serialize + for<'d> serde::Deserialize<'d>"
    )];
}

#[derive(Clone, Debug)]
// Like `serde_object_reference_struct`, but `Command` additionally nests
// `RenderCommand`/`ComputeCommand` streams parameterized by `R::PassReferences`
// (its arena field is `#[serde(skip)]`, so no bound is needed for it), so its
// serde bound must also cover those.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(
        bound = "R::Buffer: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Surface: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Texture: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::TextureView: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::ExternalTexture: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::QuerySet: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::BindGroup: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::RenderPipeline: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::RenderBundle: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::ComputePipeline: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Blas: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          R::Tlas: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          wgt::BufferTransition<R::Buffer>: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          wgt::TextureTransition<R::Texture>: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          crate::command::RenderCommand<R::PassReferences>: serde::Serialize + for<'d> serde::Deserialize<'d>,\
          crate::command::ComputeCommand<R::PassReferences>: serde::Serialize + for<'d> serde::Deserialize<'d>"
    )
)]
pub enum Command<R: ReferenceType> {
    CopyBufferToBuffer {
        src: R::Buffer,
        src_offset: wgt::BufferAddress,
        dst: R::Buffer,
        dst_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    CopyBufferToTexture {
        src: wgt::TexelCopyBufferInfo<R::Buffer>,
        dst: wgt::TexelCopyTextureInfo<R::Texture>,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: wgt::TexelCopyTextureInfo<R::Texture>,
        dst: wgt::TexelCopyBufferInfo<R::Buffer>,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: wgt::TexelCopyTextureInfo<R::Texture>,
        dst: wgt::TexelCopyTextureInfo<R::Texture>,
        size: wgt::Extent3d,
    },
    ClearBuffer {
        dst: R::Buffer,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    ClearTexture {
        dst: R::Texture,
        subresource_range: wgt::ImageSubresourceRange,
    },
    WriteTimestamp {
        query_set: R::QuerySet,
        query_index: u32,
    },
    ResolveQuerySet {
        query_set: R::QuerySet,
        start_query: u32,
        query_count: u32,
        destination: R::Buffer,
        destination_offset: wgt::BufferAddress,
    },
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
    RunComputePass {
        pass:
            crate::command::BasePass<crate::command::ComputeCommand<R::PassReferences>, Infallible>,
        /// Arenas keeping the pass's interned resources alive alongside `pass`.
        /// Skipped by serde: only pointer/id command streams (which have no
        /// arenas) are ever serialized.
        ///
        /// This is safe only by convention, not by types: on deserialize this
        /// field defaults to empty (see [`NoArenas`]/`Default`), which is
        /// correct because the only deserialized form is
        /// `Command<PointerReferences>` (player), whose `arenas` type is the
        /// zero-sized `NoArenas`. The arena-indexed `ArcReferences`/
        /// `ArenaReferences` forms must never be deserialized: doing so would
        /// leave `arenas` empty while the command stream still carries
        /// non-zero arena indices, causing a checked-indexing panic on
        /// resolve (never memory-unsafe, since indexing is bounds-checked).
        #[cfg_attr(feature = "serde", serde(skip))]
        arenas: R::ComputePassArenas,
        timestamp_writes: Option<crate::command::PassTimestampWrites<R::QuerySet>>,
    },
    RunRenderPass {
        pass:
            crate::command::BasePass<crate::command::RenderCommand<R::PassReferences>, Infallible>,
        /// Arenas keeping the pass's interned resources alive alongside `pass`.
        /// Skipped by serde: only pointer/id command streams (which have no
        /// arenas) are ever serialized.
        ///
        /// This is safe only by convention, not by types: on deserialize this
        /// field defaults to empty (see [`NoArenas`]/`Default`), which is
        /// correct because the only deserialized form is
        /// `Command<PointerReferences>` (player), whose `arenas` type is the
        /// zero-sized `NoArenas`. The arena-indexed `ArcReferences`/
        /// `ArenaReferences` forms must never be deserialized: doing so would
        /// leave `arenas` empty while the command stream still carries
        /// non-zero arena indices, causing a checked-indexing panic on
        /// resolve (never memory-unsafe, since indexing is bounds-checked).
        #[cfg_attr(feature = "serde", serde(skip))]
        arenas: R::RenderPassArenas,
        color_attachments: ColorAttachments<R::TextureView>,
        depth_stencil_attachment:
            Option<crate::command::ResolvedRenderPassDepthStencilAttachment<R::TextureView>>,
        timestamp_writes: Option<crate::command::PassTimestampWrites<R::QuerySet>>,
        occlusion_query_set: Option<R::QuerySet>,
        multiview_mask: Option<NonZero<u32>>,
    },
    BuildAccelerationStructures {
        blas: Vec<crate::ray_tracing::OwnedBlasBuildEntry<R>>,
        tlas: Vec<crate::ray_tracing::OwnedTlasPackage<R>>,
    },
    TransitionResources {
        buffer_transitions: Vec<wgt::BufferTransition<R::Buffer>>,
        texture_transitions: Vec<wgt::TextureTransition<R::Texture>>,
    },
}

pub type ArcCommand = Command<ArcReferences>;
