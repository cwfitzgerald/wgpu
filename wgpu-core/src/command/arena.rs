//! Per-pass resource arenas.
//!
//! Recorded render- and compute-pass command streams used to carry an
//! `Arc<T>` for every resource each command referenced. Resolving those
//! commands at replay time meant a `Storage::get` + `Fallible::clone` on the
//! record side and a paired `Arc::drop` on teardown, per command, even when
//! the same resource was referenced thousands of times in a row (the object
//! uniform bind group in the reference benchmark is re-bound on every one of
//! ~9,500 draws).
//!
//! Instead, each pass interns the *reused* resources it references (bind
//! groups, render/compute pipelines and query sets) into a small set of
//! per-type, append-only arenas (one arena per resource type). A resource is
//! resolved (`Storage::get`, validity-checked, `same_device`-checked) exactly
//! once, the first time it is referenced on a given slot; the resolved `Arc`
//! is *moved* into the arena and the command records a small copyable index
//! into it. Subsequent references to the same resource on the same slot are
//! served from a per-slot memo ([`super::StateChange`] and friends) and cost
//! nothing but an index push.
//!
//! Buffers are *not* interned: the reference workload references thousands of
//! distinct vertex/index buffers per pass with no intra-pass reuse, so a resolve
//! memo would always miss and interning would only add overhead. Buffer
//! commands therefore carry their `Arc`s directly, resolved and
//! `same_device`-checked once per command at replay as before.
//!
//! The arena is the sole keep-alive for its resources from the moment they are
//! interned (record time) until the pass's command stream is fully replayed
//! (encode time): it travels next to the command stream inside the
//! `RunRenderPass`/`RunComputePass` command, so the arena and the indices into
//! it share an owner and therefore an exact lifetime. Dropping the encoder,
//! abandoning the pass, or hitting a mid-replay error drops the arena together
//! with the commands that index it. Indices are only ever resolved while their
//! backing arena is a live local in the same frame.
//!
//! # Index safety
//!
//! Because two arenas can be live simultaneously (a pass's own arena and, while
//! executing a bundle, that bundle's arena), the index newtypes are per-type
//! (so a [`BindGroupArenaIndex`] can never resolve a pipeline arena) and, in
//! debug / `strict_asserts` builds, additionally carry the unique *generation*
//! of the arena instance they were minted from. Resolution asserts the
//! generation matches and that the index is in bounds; a mismatch is a
//! programming error (index confusion) that would otherwise silently resolve to
//! a valid-but-wrong resource. In release builds without `strict_asserts` the
//! generation is compiled out and resolution falls back to *checked* `Vec`
//! indexing — an out-of-bounds index panics rather than reading arbitrary
//! memory. There is deliberately no `get_unchecked` anywhere in this module.

use alloc::{sync::Arc, vec::Vec};
#[cfg(any(debug_assertions, feature = "strict_asserts"))]
use core::sync::atomic::{AtomicU32, Ordering};

#[allow(unused_imports)]
use wgt::{strict_assert, strict_assert_eq};

use crate::binding_model::BindGroup;
use crate::pipeline::{ComputePipeline, RenderPipeline};
use crate::resource::QuerySet;

/// A globally-unique tag identifying an arena instance, used in debug /
/// `strict_asserts` builds to catch resolving an index against the wrong
/// arena.
///
/// This counter is never read in release builds without `strict_asserts`; the
/// generation field it feeds is compiled out there.
#[cfg(any(debug_assertions, feature = "strict_asserts"))]
static ARENA_GENERATION: AtomicU32 = AtomicU32::new(1);

#[cfg(any(debug_assertions, feature = "strict_asserts"))]
fn next_arena_generation() -> u32 {
    // Wrapping is fine: the tag only needs to distinguish the handful of arenas
    // that can be simultaneously live, and a collision merely weakens (never
    // breaks) the debug assertion.
    ARENA_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Generate a `#[repr(transparent)]`-in-release index newtype plus its arena.
///
/// The index carries a plain `u32` in all builds and, additionally in
/// debug / `strict_asserts` builds, the generation of the arena it indexes.
/// The `serde` derives exist only to satisfy the `serde_object_reference_struct`
/// bounds on the command enums — arena-referenced commands are never actually
/// serialized (traces resolve indices back to pointers first).
macro_rules! arena {
    (
        $(#[$index_meta:meta])*
        index $Index:ident,
        arena $Arena:ident,
        entry $Entry:ident { arc: $Res:ty $(, $field:ident : $field_ty:ty = $field_init:expr)* $(,)? }
    ) => {
        $(#[$index_meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[cfg_attr(not(any(debug_assertions, feature = "strict_asserts")), repr(transparent))]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Index {
            index: u32,
            #[cfg(any(debug_assertions, feature = "strict_asserts"))]
            #[cfg_attr(feature = "serde", serde(skip))]
            generation: u32,
        }

        /// One interned resource plus its per-pass replay bookkeeping.
        ///
        /// `Clone` clones the underlying `Arc` (a refcount bump). It is only
        /// ever invoked on the (rare) trace-recording path, which clones the
        /// whole command list including its arenas.
        #[derive(Clone, Debug)]
        pub(crate) struct $Entry {
            pub(crate) arc: $Res,
            $(pub(crate) $field: $field_ty,)*
        }

        /// An append-only arena of interned resources of one type.
        #[derive(Clone, Debug)]
        pub(crate) struct $Arena {
            entries: Vec<$Entry>,
            #[cfg(any(debug_assertions, feature = "strict_asserts"))]
            generation: u32,
        }

        impl $Arena {
            /// Create an empty arena backed by `entries`, which must be empty
            /// (it may be a recycled, already-grown `Vec` from the pool).
            pub(crate) fn from_pooled(entries: Vec<$Entry>) -> Self {
                debug_assert!(entries.is_empty());
                Self {
                    entries,
                    #[cfg(any(debug_assertions, feature = "strict_asserts"))]
                    generation: next_arena_generation(),
                }
            }

            /// Append a resolved `Arc` and return its index. The `Arc` is
            /// *moved* in; this is the only place the arena grows.
            pub(crate) fn push(&mut self, arc: $Res) -> $Index {
                let index = self.entries.len() as u32;
                self.entries.push($Entry {
                    arc,
                    $($field: $field_init,)*
                });
                $Index {
                    index,
                    #[cfg(any(debug_assertions, feature = "strict_asserts"))]
                    generation: self.generation,
                }
            }

            #[inline]
            fn check(&self, i: $Index) -> usize {
                #[cfg(any(debug_assertions, feature = "strict_asserts"))]
                strict_assert_eq!(
                    i.generation,
                    self.generation,
                    concat!(stringify!($Index), " resolved against the wrong arena"),
                );
                let index = i.index as usize;
                strict_assert!(
                    index < self.entries.len(),
                    concat!(stringify!($Index), " out of bounds"),
                );
                index
            }

            /// Resolve an index to a borrow of the interned resource.
            ///
            /// Release builds without `strict_asserts` use checked `Vec`
            /// indexing, which panics on an out-of-bounds index rather than
            /// reading arbitrary memory.
            #[inline]
            pub(crate) fn get(&self, i: $Index) -> &$Res {
                let index = self.check(i);
                &self.entries[index].arc
            }

            /// Resolve an index to a borrow of its full [entry](`$Entry`),
            /// including replay bookkeeping. (Only the bind-group arena uses
            /// this, for the repeat-merge / tracker-insert elision; the pipeline
            /// and query-set arenas only ever call [`get`](Self::get).)
            #[inline]
            #[allow(dead_code)]
            pub(crate) fn entry(&self, i: $Index) -> &$Entry {
                let index = self.check(i);
                &self.entries[index]
            }

            /// Resolve an index to a mutable borrow of its full entry.
            ///
            /// Only valid for an arena owned by value (a pass's arena during
            /// replay); a bundle's arena is shared `&self` and never resolved
            /// this way.
            #[inline]
            #[allow(dead_code)]
            pub(crate) fn entry_mut(&mut self, i: $Index) -> &mut $Entry {
                let index = self.check(i);
                &mut self.entries[index]
            }

            /// Consume the arena, returning its backing `Vec` so its grown
            /// storage can be returned to the pool. The arena is moved by value
            /// (nothing indexes it anymore at release time), so no `mem::take`
            /// reset is needed.
            #[allow(dead_code)]
            pub(crate) fn into_entries(self) -> Vec<$Entry> {
                self.entries
            }

            /// Iterate the interned resources in insertion order.
            #[allow(dead_code)]
            pub(crate) fn iter(&self) -> impl Iterator<Item = &$Entry> {
                self.entries.iter()
            }

            #[allow(dead_code)]
            pub(crate) fn len(&self) -> usize {
                self.entries.len()
            }
        }
    };
}

arena! {
    /// Index into a pass's or bundle's [`BindGroupArena`].
    index BindGroupArenaIndex,
    arena BindGroupArena,
    entry BindGroupEntry {
        arc: Arc<BindGroup>,
        // (4a): whether this bind group's resources have already been merged
        // into the pass's usage scope, so the repeat merge can be elided.
        merged_into_scope: bool = false,
        // Whether this bind group has already been inserted into the submit
        // tracker's `bind_groups` list. Deduplicating this subsumes the
        // per-command clone + submit-time re-validation (issue #8510).
        merged_into_tracker: bool = false,
    }
}

arena! {
    /// Index into a pass's or bundle's [`RenderPipelineArena`].
    index RenderPipelineArenaIndex,
    arena RenderPipelineArena,
    entry RenderPipelineEntry { arc: Arc<RenderPipeline> }
}

arena! {
    /// Index into a pass's [`ComputePipelineArena`].
    index ComputePipelineArenaIndex,
    arena ComputePipelineArena,
    entry ComputePipelineEntry { arc: Arc<ComputePipeline> }
}

arena! {
    /// Index into a pass's or bundle's [`QuerySetArena`].
    index QuerySetArenaIndex,
    arena QuerySetArena,
    entry QuerySetEntry { arc: Arc<QuerySet> }
}

/// A per-slot last-resolved memo mapping an id to the arena index it was
/// interned at.
///
/// This is the *resolve* dedup, distinct from the *emission* dedup performed by
/// [`super::StateChange`]/[`super::BindGroupStateChange`]. When the same id is
/// set again on the same slot, the cached index is returned and no
/// `Storage::get` / `Fallible::clone` is paid. It deliberately persists across
/// offset-bearing bind-group sets (which reset the emission memo) so the
/// object-uniform bind group, re-bound with dynamic offsets on every draw, is
/// resolved exactly once per pass.
///
/// A cache miss (different id, or first use) is the caller's cue to resolve the
/// id, intern the resulting `Arc` into the arena, and record the returned index
/// via [`Self::store`].
#[derive(Debug)]
pub(crate) struct InternCache<Id: Copy + PartialEq, Index: Copy> {
    last: Option<(Id, Index)>,
}

impl<Id: Copy + PartialEq, Index: Copy> InternCache<Id, Index> {
    pub(crate) fn new() -> Self {
        Self { last: None }
    }

    /// Return the cached index if `id` matches the last id stored on this slot.
    #[inline]
    pub(crate) fn get(&self, id: Id) -> Option<Index> {
        match self.last {
            Some((last_id, index)) if last_id == id => Some(index),
            _ => None,
        }
    }

    /// Record that `id` was interned at `index` on this slot.
    #[inline]
    pub(crate) fn store(&mut self, id: Id, index: Index) {
        self.last = Some((id, index));
    }
}

impl<Id: Copy + PartialEq, Index: Copy> Default for InternCache<Id, Index> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-slot resolve caches for a render pass's or bundle's bind groups plus the
/// pipeline.
///
/// Buffers are *not* interned (the workload has zero intra-pass buffer reuse, so
/// a resolve memo would always miss), so there is no vertex/index-buffer cache
/// here; buffer commands carry their `Arc`s directly.
///
/// Kept separate from the arenas themselves so the arenas can be moved out at
/// pass end while these caches (which are pure recording-time state) are
/// dropped.
#[derive(Debug)]
pub(crate) struct RenderInternCaches {
    pub(crate) pipeline: InternCache<crate::id::RenderPipelineId, RenderPipelineArenaIndex>,
    pub(crate) bind_groups:
        [InternCache<crate::id::BindGroupId, BindGroupArenaIndex>; hal::MAX_BIND_GROUPS],
}

impl RenderInternCaches {
    pub(crate) fn new() -> Self {
        Self {
            pipeline: InternCache::new(),
            bind_groups: core::array::from_fn(|_| InternCache::new()),
        }
    }
}

impl Default for RenderInternCaches {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-slot resolve caches for a compute pass's bind groups and pipeline.
#[derive(Debug)]
pub(crate) struct ComputeInternCaches {
    pub(crate) pipeline: InternCache<crate::id::ComputePipelineId, ComputePipelineArenaIndex>,
    pub(crate) bind_groups:
        [InternCache<crate::id::BindGroupId, BindGroupArenaIndex>; hal::MAX_BIND_GROUPS],
}

impl ComputeInternCaches {
    pub(crate) fn new() -> Self {
        Self {
            pipeline: InternCache::new(),
            bind_groups: core::array::from_fn(|_| InternCache::new()),
        }
    }
}

impl Default for ComputeInternCaches {
    fn default() -> Self {
        Self::new()
    }
}

/// The set of arenas a *render* pass or render bundle needs.
///
/// Bind groups, render pipelines and query sets are interned; buffers are *not*
/// (they carry their `Arc`s directly in the command, since the workload has zero
/// intra-pass buffer reuse). Render bundles referenced by `ExecuteBundle` are
/// likewise kept as plain `Arc`s in the command (see [`super::ArenaReferences`])
/// to avoid a chicken-and-egg between a bundle's arena and bundles it
/// references.
///
/// Public only because it appears as an associated type of the public
/// [`ReferenceType`](super::ReferenceType) trait; its fields are crate-private
/// and it is not part of the stable API.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RenderArenas {
    pub(crate) bind_groups: BindGroupArena,
    pub(crate) pipelines: RenderPipelineArena,
    pub(crate) query_sets: QuerySetArena,
}

impl Default for RenderArenas {
    fn default() -> Self {
        Self::from_pooled(Vec::new(), Vec::new(), Vec::new())
    }
}

impl RenderArenas {
    /// Build a fresh set of render arenas from pooled (empty) backing vectors.
    pub(crate) fn from_pooled(
        bind_groups: Vec<BindGroupEntry>,
        pipelines: Vec<RenderPipelineEntry>,
        query_sets: Vec<QuerySetEntry>,
    ) -> Self {
        Self {
            bind_groups: BindGroupArena::from_pooled(bind_groups),
            pipelines: RenderPipelineArena::from_pooled(pipelines),
            query_sets: QuerySetArena::from_pooled(query_sets),
        }
    }
}

/// The set of arenas a *compute* pass needs.
///
/// Public only because it appears as an associated type of the public
/// [`ReferenceType`](super::ReferenceType) trait; not part of the stable API.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ComputeArenas {
    pub(crate) bind_groups: BindGroupArena,
    pub(crate) pipelines: ComputePipelineArena,
    pub(crate) query_sets: QuerySetArena,
}

impl Default for ComputeArenas {
    fn default() -> Self {
        Self::from_pooled(Vec::new(), Vec::new(), Vec::new())
    }
}

impl ComputeArenas {
    /// Build a fresh set of compute arenas from pooled (empty) backing vectors.
    pub(crate) fn from_pooled(
        bind_groups: Vec<BindGroupEntry>,
        pipelines: Vec<ComputePipelineEntry>,
        query_sets: Vec<QuerySetEntry>,
    ) -> Self {
        Self {
            bind_groups: BindGroupArena::from_pooled(bind_groups),
            pipelines: ComputePipelineArena::from_pooled(pipelines),
            query_sets: QuerySetArena::from_pooled(query_sets),
        }
    }
}

#[cfg(test)]
mod tests {
    // A separate arena instantiation over a trivial resource type lets us
    // exercise the generated push/get/generation/bounds machinery without
    // needing a `Device` to build real resources. It uses the exact same
    // `arena!` macro as the production arenas, so behavior is shared.
    use super::*;

    arena! {
        index TestIndex,
        arena TestArena,
        entry TestEntry { arc: u32 }
    }

    #[test]
    fn push_returns_incrementing_indices_and_get_resolves() {
        let mut arena = TestArena::from_pooled(Vec::new());
        let i0 = arena.push(10);
        let i1 = arena.push(20);
        let i2 = arena.push(30);
        assert_eq!(arena.len(), 3);
        assert_eq!(*arena.get(i0), 10);
        assert_eq!(*arena.get(i1), 20);
        assert_eq!(*arena.get(i2), 30);
        // Re-resolving is stable.
        assert_eq!(*arena.get(i0), 10);
    }

    #[test]
    fn indices_are_copy_and_eq() {
        let mut arena = TestArena::from_pooled(Vec::new());
        let a = arena.push(1);
        let b = a; // Copy
        assert_eq!(a, b);
        let c = arena.push(2);
        assert_ne!(a, c);
    }

    #[test]
    fn entry_mut_bookkeeping_round_trips() {
        // Use the production bind-group-shaped entry via a local arena that has
        // extra fields, to check `entry`/`entry_mut` reach the bookkeeping.
        arena! {
            index BkIndex,
            arena BkArena,
            entry BkEntry { arc: u32, flag: bool = false }
        }
        let mut arena = BkArena::from_pooled(Vec::new());
        let i = arena.push(5);
        assert!(!arena.entry(i).flag);
        arena.entry_mut(i).flag = true;
        assert!(arena.entry(i).flag);
        assert_eq!(arena.entry(i).arc, 5);
    }

    #[test]
    fn into_entries_yields_the_backing_vec() {
        let mut arena = TestArena::from_pooled(Vec::new());
        arena.push(1);
        arena.push(2);
        assert_eq!(arena.len(), 2);
        let taken = arena.into_entries();
        assert_eq!(taken.len(), 2);
    }

    #[test]
    #[cfg(any(debug_assertions, feature = "strict_asserts"))]
    fn distinct_arenas_get_distinct_generations() {
        let a = TestArena::from_pooled(Vec::new());
        let b = TestArena::from_pooled(Vec::new());
        assert_ne!(a.generation, b.generation);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_index_panics() {
        // Checked indexing (release) or bounds assert (debug/strict) both
        // panic; either way a bogus index never reads arbitrary memory.
        let arena = TestArena::from_pooled(Vec::new());
        let bogus = TestIndex {
            index: 0,
            #[cfg(any(debug_assertions, feature = "strict_asserts"))]
            generation: 0,
        };
        let _ = arena.get(bogus);
    }

    #[test]
    #[should_panic]
    #[cfg(any(debug_assertions, feature = "strict_asserts"))]
    fn wrong_arena_generation_panics() {
        let mut a = TestArena::from_pooled(Vec::new());
        let b = TestArena::from_pooled(Vec::new());
        let idx_from_a = a.push(1);
        // Resolving `a`'s index against `b` must trip the generation assert.
        let _ = b.get(idx_from_a);
    }
}
