# M3m — Metal explicit-synchronization track

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — re-plan at milestone start; anchors drift, symbol names are stable.

## Goal

Land the Metal explicit-synchronization rework the resource table needs, in three sub-phases: **M3a** tracker producer tokens + a pass-DAG-shaped hal fence-edge API + fence lowering (gated on perf parity with auto-tracking); **M3b** untracked/heap allocations; **M3c** the table itself (argument buffer of `gpuResourceID`s, per-table gap fences, residency sets, MSL lowering).

**Doc scope (:170, quote):** "M3a tracker producer tokens + hal edge API + fence lowering, perf-parity gate vs auto-tracking; M3b untracked/heap allocations; M3c tables (arg buffer, per-table gap fences, residency sets, MSL lowering). Optional bridge: tracked + conservative fence chain around table passes only."
**Accept:** the doc gives **no explicit `Accept:` line** for M3m; the **bolded perf-parity gate vs auto-tracking** (:170) is the load-bearing exit criterion for M3a, and behavioral parity with Vulkan/DX12 on the shared conformance suite (once M3c lands) is the table-level bar. Confirm the numeric parity bar with Connor (open question).

**Dependency / parallelism:** doc :141 says "Metal track independent after M0." M3a (tracker/fence rework) has no M1/M2 dependency and can run in parallel with M1/M2. M3c (tables) needs M1 metadata + M2 spec-exact visibility. [milestones.md](milestones.md) sequences M3m as a parallel track that becomes fully unblocked after M1/M2. **M3a requires a design review of the hal pass-edge DAG API shape BEFORE implementation** (doc open Q1, :180).

## Binding decisions & invariants

- **D1** (:23) — design must hold for Metal.
- **D14** (:36) — **explicit-synchronization rework is the committed direction; it must be pass-DAG-shaped** (per-pass fence edges from tracker producer tokens), because a linear fence chain would serialize passes that auto-tracking overlaps. Residency via queue-attached `MTLResidencySet` (macOS 15+ gate); attachment after encoding is valid.
- **D16** (:38) — confirmed facts (do not re-verify): Metal passes execute fully parallel absent explicit sync; `MTLResidencySet` may be attached after encoding; residency sets do not participate in hazard tracking; `useResource` feeds Metal hazard tracking for argument-buffer access.
- **Invariant 1** (:42) — submission-boundary observability holds via fence edges + residency-set commit at gaps.
- **Invariant 3** (:44) — on Metal "correctness comes from edges, not transitions": `transition_buffers/textures` are no-ops and **stay** no-ops (doc :111, `metal/command.rs:582-592`).

## Inherited state (from M0 — pointers, no duplication)

- **Metal hal stubs:** `pub struct ResourceTable;` + `unimplemented!()` table-method bodies (m0-notes 0.4 :57); `resource_table_memory_barrier` DX12/Metal/gles stubs (m0-notes wave 7 :149). M3c fills the table methods; the barrier maps to compute-encoder `memoryBarrier(scope:)`.
- **Metal storage pattern:** `MTLBuffer` of `gpuResourceID`s (`metal/device.rs:1035`, doc :111).
- **Metal no-op transitions:** `transition_buffers`/`transition_textures` are already no-ops (doc :111) — the tracker rework must not start relying on them.
- **Cross-backend core:** gap/splice, metadata (M1), mask ring + epoch immediate (M2) are in wgpu-core. M3a's hal fence-edge API is a **new hal surface** that Vulkan/DX12 ignore; M3c consumes it for per-table gap fences.
- **naga MSL:** runtime-array lengths via `binding_array_length_map` (`back/msl/mod.rs:431`, doc :121); heterogeneous MSL gets `[[clang::may_alias]]` (that is M4, not M3m).

## Carry-over items absorbed here

- Metal M3a (producer tokens, hal edge API, fence lowering, perf-parity gate) / M3b (untracked/heap allocations) / M3c (arg buffer, per-table gap fences, residency sets, MSL lowering) — **doc-stated**.
- **M3a hal pass-edge DAG API design review** — doc open Q1 (:180); a prerequisite, not a work item to start coding blind.

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

### M3a — tracker producer tokens + hal edge API + fence lowering (perf-parity gated)

- **3a.0 Design review (BLOCKING).** Settle the hal pass-edge (DAG) API shape and node identity for top-level transfer chunks (doc open Q1). Deliverable: an approved API sketch of `begin_*_pass(sync: { signal: NodeId, waits: &[NodeId] })` (doc :111) and how tracker producer tokens (u32 sync-node = pass/chunk index, doc :36) map to nodes. *Verify:* Connor sign-off before any impl.
- **3a.1 Tracker producer tokens.** Per-resource last-writer *producer token* (sync node) in the tracker (doc :36, :195 glossary). *Verify:* token updates at producer pass end.
- **3a.2 hal edge API.** Add `sync` to `begin_*_pass`; Vulkan/DX12 ignore it; Metal maps nodes to pooled `MTLFence`s (update at producer end, stage-scoped wait at consumer start, doc :36, :111). *Verify:* the API compiles cross-backend; Vulkan/DX12 unaffected.
- **3a.3 perf-parity gate (EXIT CRITERION).** Measure explicit-sync vs auto-tracking on real bindful workloads (doc :36 "Exit gate: perf parity with auto-tracking"). *Verify:* numbers meet the agreed bar (open question) on the tracking issue.

### M3b — untracked / heap allocations

- **3b.1** Untracked/heap allocation path for Metal (doc :170). Design shape is itself under-specified — scope it during M3a. *Verify:* heap-allocated resources participate correctly in the edge model.

### M3c — tables

- **3c.1 Argument buffer table.** Table = `MTLBuffer`/argument buffer of `gpuResourceID`s; slot writes mirror core deltas. *Verify:* getResource reads the right resource id.
- **3c.2 Per-table gap fences.** Consumer passes pre-encode `waitForFence(table_fence)` (identity encode-known); appended gap blit encoders `updateFence` after the actual producers (placement submit-known) — doc :111. Metal gaps = fresh blit encoder on the same `MTLCommandBuffer` (errata (a), no CB split). *Verify:* cross-pass table read waits on producers; no over-serialization.
- **3c.3 Residency sets.** Queue-attached `MTLResidencySet` per table (macOS 15+ gate); attach after encoding (D14/D16). *Verify:* members resident; residency-set does not perturb hazard tracking.
- **3c.4 MSL lowering.** Metadata load → compare → (mask test) → select → arg-buffer access; runtime-array lengths via `binding_array_length_map`. Compute `memoryBarrier(scope:)` for the dirty-bit scheme (Apple silicon). *Verify:* MSL snapshots compile; e2e parity with Vulkan.
- **Optional bridge** (doc :170): tracked + conservative fence chain around table passes only — a fallback if the full DAG rework slips. Confirm whether to build it.

## Open questions requiring a user decision

- **hal pass-edge DAG API shape + node identity for top-level transfer chunks** (doc Q1, :180): the 3a.0 design-review deliverable. Must be settled before M3a impl.
- **Numeric perf-parity bar** (doc :36 gate is qualitative): what regression tolerance vs auto-tracking counts as "parity" (e.g. ≤ X% on which bindful workloads)? Gates M3a exit.
- **M3b design:** the untracked/heap-allocation approach is only sketched (doc :170) — needs a shape decision.
- **Optional bridge:** build the conservative fence-chain-around-table-passes bridge as an interim, or go straight to the full DAG? Affects schedule risk.
- **Render-encoder post-fragment barriers don't exist on Apple silicon** (doc :111) — confirm the fences-only approach for render passes is acceptable (no compute `memoryBarrier` analogue there).

## Risks / landmines

- **A linear fence chain is explicitly wrong** (D14): it serializes passes auto-tracking overlaps. The DAG shape is non-negotiable — the whole reason the design review gates M3a.
- **`transition_buffers/textures` stay no-ops** (doc :111): do not "fix" them to carry layout; Metal correctness is edges-only (Invariant 3).
- **macOS 15+ residency gate:** `MTLResidencySet` needs the OS floor; feature exposure must gate on it, and there must be a story for older macOS (feature simply unavailable).
- **Perf-parity gate can fail late:** M3a is measured, not just built — budget iteration on the fence-lowering before assuming the gate passes.
- **Metal stubs:** `unimplemented!()` table bodies panic at runtime if left; grep the convention before M3c.

## Verification strategy

Harness/gates per m0-notes cheat sheet (:179-189); Metal is not exercisable on the Windows dev box — M3m needs a macOS runner (the M0 e2e suite ran Vulkan-only). New axes:
- **M3a perf-parity microbench** vs auto-tracking on bindful workloads (the exit gate) — numbers on the tracking issue.
- **M3c table conformance:** once M3c lands, run the backend-parametric suite (M3's conformance target) on Metal for behavioral parity with Vulkan/DX12.
- **Fence-edge correctness stress:** overlapping passes that auto-tracking parallelizes must stay parallel (no linear-chain regression) while table cross-pass reads correctly wait.
