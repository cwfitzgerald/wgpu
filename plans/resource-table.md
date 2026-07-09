# Resource Table (bindless) — implementation plan

Status: **design settled, implementation not started** · Last updated: 2026-07-08
Spec: [`docs/bindless.md`](../docs/bindless.md) (the gpuweb proposal) · Tracking: [wgpu#8557](https://github.com/gfx-rs/wgpu/issues/8557) → [gpuweb#5372](https://github.com/gpuweb/gpuweb/issues/5372). This file is the executable distillation of the design discussion and is the source of truth.

## How to use this document (for agents)

- Read this file first, then `docs/bindless.md` for API semantics. Do **not** re-derive design from scratch.
- **Decisions ledger entries are settled.** Do not relitigate them; if implementation reveals a conflict, stop and surface it rather than silently deviating.
- **Invariants must hold in every PR.** They are the safety argument.
- Each work item below is roughly one PR. Respect the dependency notes. Acceptance criteria are the definition of done.
- Terminology used throughout is defined in the Glossary. File anchors (`path:line`) are as of 2026-07 and may drift; the symbol names are the stable reference.
- Update the Progress log at the bottom when a work item lands or a decision changes.

## Background (one paragraph)

The gpuweb bindless proposal adds `GPUResourceTable`: a device-timeline-mutable, sparse descriptor array bound as encoder state, accessed from WGSL via `getResource<T>(index)` / `hasResource<T>(index)`. wgpu implements it native-first to feed implementation experience back to the WG. The core difficulty: wgpu records hal command buffers at `CommandEncoder::finish()`, but table contents, submission order, and the dynamically-accessed slot set are only known at `queue.submit()` (or shader runtime). Everything below is architecture for closing that gap safely (no driver UB reachable from the checked path — this is a hard requirement for eventual Firefox use) without giving up eager, multithreaded command recording.

## Decisions ledger (settled 2026-07-08, Connor)

| # | Decision |
|---|----------|
| D1 | Backend order: Vulkan first; design must hold for DX12/Metal (both fully designed below). |
| D2 | Sync architecture: **suspend & append** (strategy D). No submit-time replay of user commands. hal CBs are left open at table-relevant boundaries; exact barriers + metadata writes are appended at submit, then closed. |
| D3 | Stage **semantics, not synchronization**: v0 ships the final sync architecture with the strictest semantics; each milestone only relaxes behavior. v0 may reject spec-valid programs, never accepts spec-invalid ones, never differs silently. |
| D4 | Feature bits: `EXPERIMENTAL_SAMPLING_RESOURCE_TABLE` (checked, the primary path), `EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE` (checked, implies sampling), `EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED` (unsafe add-on; elides only the naga check emission). Checked and unchecked share one code path differing only in the naga policy. |
| D5 | Rust API mirrors the proposal's semantics with Rust-ified surface (`Result` errors, `SubmissionIndex` in `SlotInUse` errors). |
| D6 | hal API is **slot-oriented** (`update_table_slot`), never descriptor-write-oriented. |
| D7 | Samplers: **one deduplicated device-global sampler heap on every backend**; table sampler slots are metadata-only (heap index in the metadata payload). Dedup/refcount logic lives in wgpu-core. `update()` returns `Err(SamplerHeapFull)` at capacity. |
| D8 | `insert_binding` allocates the **lowest currently-available slot**. |
| D9 | Conflict/visibility checks at **whole-resource granularity** (per spec note), not per-subresource. |
| D10 | Heterogeneous descriptors: **`VK_EXT_mutable_descriptor_type` unconditionally** on Vulkan (feature not exposed without it); **overlapping root-signature ranges unconditionally** on DX12 (SM 5.1/FXC compatible; SM 6.6 explicitly off the path). |
| D11 | **No blanket `GENERAL` layouts.** Mixed steady-state policy: sampled-only table members hold `SHADER_READ_ONLY_OPTIMAL`; storage-capable members use `GENERAL` descriptors and are normalized to `GENERAL` at table-visible gaps and kept there inside table-bound compute passes. |
| D12 | **Visibility is spec-exact at per-dispatch granularity.** The API-side `isShaderVisible(slot, wgslType, usage_scope)` coupling in `docs/bindless.md` is normative for `getResource`/`hasResource`. Implemented via the **epoch immediate + mask ring** mechanism (below). Arbitrary visible/hidden/visible patterns must be represented exactly. |
| D13 | Mid-pass `compute_pass.set_resource_usage` is committed (M5) via **suspension-on-demand**; it exists to place a precise named barrier — its visibility effect rides the M2 masks. |
| D14 | Metal: **explicit-synchronization rework** is the committed direction; it must be **pass-DAG-shaped** at the hal level (per-pass fence edges from tracker producer tokens), because a linear fence chain would serialize passes that auto-tracking overlaps. Residency via queue-attached `MTLResidencySet` (macOS 15+ gate); attachment after encoding is valid. |
| D15 | Vulkan set index: require `maxBoundDescriptorSets ≥ 5` to expose the feature (table set binds at index = pipeline layout's bind-group count). Revisit bind-group stealing only if hardware data demands. |
| D16 | Confirmed facts, do not re-verify: aliased same-set/binding SPIR-V globals are valid and work on drivers (DXC ships the pattern); Metal passes execute fully parallel absent explicit sync; `MTLResidencySet` may be attached after encoding; residency sets do not participate in hazard tracking; `useResource` is what feeds Metal hazard tracking for argument-buffer access. |

## Invariants (every PR must preserve)

1. **Submission-boundary observability.** All observable table mutation (metadata visibility flips, steady-state layout transitions, residency-set membership) is applied by GPU-timeline writes at submission boundaries (submit-head fixup) or at gaps within a submission. In-flight work reads old state; work submitted later sees new state. CPU descriptor writes only ever target slots that no in-flight submission can *dynamically* use — this is what makes them legal under Vulkan's `UPDATE_UNUSED_WHILE_PENDING` + `PARTIALLY_BOUND` rules (VUID-vkUpdateDescriptorSets-pDescriptorWrites-03047) and its DX12/Metal analogues.
2. **Slot-reuse gating.** A slot may be rewritten only when `available_after_submit ≤ completed_submission_index` (maps onto `LifetimeTracker` / `SubmissionIndex`). Never relaxed, even in unchecked mode.
3. **Layout correctness never depends on conservative triggers.** Only memory *ordering* is conservative (dirty bits); image layouts are pinned by the mixed steady-state policy (D11) plus exact transitions in appended gap commands. A missed memory barrier ⇒ unspecified values (safe); a wrong layout ⇒ driver UB (must be impossible by construction).
4. **Checked/unchecked single code path.** The only difference is the naga bounds/visibility policy eliding the metadata compare + select. No separate lowering.
5. **Tables hold `Arc`s to their resources**; `destroy()` of a resource zeroes its slots' metadata at the next submit head; hal teardown stays deferred by submission index as for all resources.
6. **Every conservative bound is either exact or over-approximate in the safe direction** (hiding more than spec is forbidden after M2 — D12; barriering more than needed is always allowed).

## Architecture

### Object model (wgpu-core `ResourceTable`)

Per table: `Vec<Slot>` where `Slot = { resource: Option<Arc<TrackedResource>>, flavor: TypeClass, available_after: SubmissionIndex }`; a `resource → slots` multimap; the usage-state map `Map<ResourceId, TableUsage>` (`None | ReadOnly | Writable`, default ReadOnly per spec); a pending-ops queue drained into the submit-head fixup; hal handle (descriptor set / heap range / argument buffer); metadata buffer; mask ring buffer; default resources in `K` hidden tail slots (one per supported type class — no null-descriptor dependency). Registry/lifetime plumbing follows the Blas/Tlas template exactly: `ids!` (`wgpu-core/src/id.rs:326` area), `Hub` registry (`hub.rs:213` area), `Fallible` + invalid-object pattern (`device/ray_tracing.rs:27-51`), `Snatchable` raw + deferred destruction (`resource.rs:1110`, `snatch.rs`), `TrackingData` + lock rank + trace `Action`s + `HubReport`.

### Suspend & append (gaps)

- A wgpu CB is already a `Vec` of hal CBs with splice ops (`command/mod.rs:592-689`, `close_and_swap`, `close_and_push_front`); submit already records a per-CB "Transit" prologue (`device/queue.rs:1466-1498`). Generalize: at each **gap** position, the current hal segment is left **open** at `finish()` with a gap marker carrying that boundary's usage-scope snapshot; at submit, exact barriers/layout transitions/metadata writes are appended to the open tail, then the segment is closed. Vulkan: CB stays in recording state (pool exclusively owned by submit time — `BakedCommands` already carries the encoder). DX12: `Close()` at submit. Metal: append a fresh blit encoder to the same `MTLCommandBuffer`.
- Gap positions: (a) before each pass that binds a table (subsumes the pre-pass transit), (b) between submitted CBs (exists today), (c) at M5 mid-pass `set_resource_usage` calls (suspension-on-demand). **Never** per-dispatch.
- Submit-head fixup rides the existing `pending_writes` encoder (`queue.rs:388`): membership joins/leaves (steady-state layout normalization), metadata snapshot deltas, destroyed-slot zeroing, sampler-heap maintenance, residency-set commit (Metal).
- Gap snapshot contents (retained finish→submit): bound table id, pass kind, per-dispatch compact usage digests for table-bound compute passes (see visibility), the pass's merged writable set.

### Visibility: epoch immediate + mask ring (spec-exact, D12)

Semantics: `getResource`/`hasResource` behave as-if evaluating `isShaderVisible(slot, wgslType, usage_scope)` per **usage scope** — per dispatch in compute (each dispatch is one scope; a render pass is a single scope, `docs/index.bs:1236-1251`).

Inputs split by when they're known: conflict(R, E) per dispatch-epoch E is **encode-known** (depends only on E's bindful scope: *writable* usages hide `readonly` entries; *any-usage-other-than-storage* hides `writable` entries; table accesses are not part of usage scopes). Membership R→slots and the piecewise usage-state timeline are **submit-known**. The executing dispatch is **runtime-known** via a reserved 4-byte immediate (push constant / root constant / `setBytes`) rewritten before each user dispatch in table-bound compute passes (internal injected dispatches don't increment E).

Mechanism: metadata entry is a `u64`:

```
bits  0..11  type_class (dimension × sample-type × ms × depth × access; sampler classes)
bits 12..15  flags: has_mask, reserved
bits 16..31  payload: sampler slots → global sampler-heap index; else debug generation
bits 32..63  mask_ptr: word offset into the table's mask ring (valid when has_mask)
```

Each table owns a **mask ring buffer** (internal binding of the table's descriptor set, so baked CBs can address it; ring reuse gated on submission retirement; growth is a rare gated event). The pass-start gap writes, for each slot whose visibility *varies within the pass*, an exact hidden-bit vector over the pass's N dispatch epochs (⌈N/32⌉ words — the truth table of the visibility function; visible/hidden/visible is `0,1,0`; any pattern representable). Usage-state changes (including M5 mid-pass ones) fold into the bits at submit. Pass-constant slots (vast majority, and all render passes) use plain word writes, no mask. Pass-end gaps restore varied words. Shader check (checked path):

```
meta = metadata[slot]; ok = meta.type_class == EXPECTED
if ok && meta.has_mask { ok = !bit(mask_ring[meta.mask_ptr + E/32], E%32) }
idx = select(DEFAULT_SLOT_FOR_T, slot, ok)   // then NonUniform-decorated array access
```

Precision notes: `readonly` masks test writable bindful usages only; `writable` masks test any-usage-other-than-storage (bindful-storage + table-storage coexistence stays visible — matches `isShaderVisible` literally); multi-slot resources get per-slot mask pointers (shareable when equal); `hasResource` uses the identical predicate.

### Barriers

- **At gaps:** exact, appended at submit with full knowledge (membership, submission order, device tracker state). Vulkan `vkCmdPipelineBarrier` with real layouts; DX12 legacy state transitions; Metal fence edges (see Metal).
- **Inside table-bound compute passes:** conservative **two-dirty-bit** scheme, permanent unannotated fallback (spec-exact visibility makes write→table-read edges real):
  - `dirty_write` ← any dispatch with writable bindful bindings; (hetero) any table-using dispatch whose module requests writable table types (**naga reflection bit**).
  - `dirty_table_read` ← any table-using dispatch.
  - Before dispatch D: if (D uses table ∧ dirty_write) or (D has writable bindings ∧ dirty_table_read) → one global compute→compute memory barrier (plain `VkMemoryBarrier` / NULL UAV barrier / Metal compute `memoryBarrier(scope:)`), clear both bits. No image barriers ever at these points (Invariant 3; mixed layout policy makes them unnecessary).
  - Elision: `set_resource_usage(None)` on written resources (encode-visible) prevents their writes setting `dirty_write`; reflection bit handles the hetero write side; row-1 bindful-vs-bindful hazards keep today's precise per-dispatch treatment (`command/compute.rs:350 flush_bindings`).
- **Cross-encoder / cross-submission:** existing device-tracker reconciliation at submit, extended per-gap.

### Layout policy (D11)

Only storage writes can touch textures *inside* passes (copies/attachments are pass-level, i.e. at gap positions), and storage writes already require `GENERAL`. So: sampled-only members never leave `SHADER_READ_ONLY_OPTIMAL` mid-pass by construction (compression preserved); storage-capable members get `GENERAL` descriptors, are normalized to `GENERAL` at table-visible gaps, and the tracker keeps every storage-capable texture in `GENERAL` inside table-bound compute passes (superset policy, encode-decidable from `TextureUsages`, legal for non-members). Buffers: no layouts. `VK_KHR_unified_image_layouts` makes residual `GENERAL` free where present.

### Sampler heap (D7)

Move dx12-hal's dedup/refcount sampler cache (`wgpu-hal/src/dx12/sampler.rs`) concept into wgpu-core; per backend: DX12 = existing global sampler heap unchanged; Vulkan = device-global descriptor set with one `SAMPLER` UAB array sized `min(4000, limit)`; Metal = device-global argument buffer of sampler `gpuResourceID`s. Table sampler slots never touch descriptor memory: metadata carries the heap index; `getResource<sampler>` extracts it from the word it already loaded. Heap-slot reuse gated on submission indices. naga's `nagaSamplerHeap` codegen (`naga/src/back/hlsl/writer.rs:1265`, options `hlsl/mod.rs:517`) is the template for all three backends.

### Per-backend

**Vulkan** (first): table = one `VkDescriptorSet` from the existing UAB allocator (`vulkan/descriptor.rs:321`), shared device-wide layout: binding 0 = `SAMPLED_IMAGE` runtime array (sampling) / `MUTABLE_EXT` array (hetero, D10), variable count = size+K, flags `PARTIALLY_BOUND | UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING | VARIABLE_DESCRIPTOR_COUNT`; binding 1 = metadata SSBO; binding 2 = mask ring SSBO. Shader declares one aliased typed view per used type class at the same set/binding (confirmed valid, D16). Newly enable: `runtimeDescriptorArray`, `descriptorBindingVariableDescriptorCount`, `descriptorBindingUpdateUnusedWhilePending`, sampler UAB (see `vulkan/adapter.rs:367-380` for where the existing bits are set). Table set index = pipeline layout's group count (D15). Feature exposed only when all present.

**DX12**: table = contiguous `size+K` range in the existing shader-visible view heap (`dx12/descriptor.rs:33` free-list); `set_resource_table` = one root descriptor table param + root SRV for metadata + mask ring; slot writes = `CopyDescriptorsSimple` (gated by Invariant 2). Shader: overlapping unbounded typed ranges per type class over the same heap range (D10). Barriers: legacy states in appended commands. Tier ≥ 2 required (already the wgpu floor), tier 3 for full sizes.

**Metal** (after explicit-sync groundwork): storage = `MTLBuffer` of `gpuResourceID`s (pattern: `metal/device.rs:1035`). Sync = the explicit-sync rework (D14): (M3a) tracker gains per-resource *producer tokens* (u32 sync-node = pass/chunk index); hal `begin_*_pass` gains `sync: { signal: NodeId, waits: &[NodeId] }`; Vulkan/DX12 ignore; Metal maps nodes to pooled `MTLFence`s (update at producer end, stage-scoped wait at consumer start). Exit gate: perf parity with auto-tracking on real bindful workloads. (M3b) untracked/heap allocations. (M3c) tables: per-table fence per gap — consumer passes pre-encode `waitForFence(table_fence)` (identity encode-known), appended gap blit encoders `updateFence` after the actual producers (placement submit-known); queue-attached `MTLResidencySet` per table. Compute-encoder `memoryBarrier(scope:)` works on Apple silicon for the dirty-bit scheme; render-encoder barriers post-fragment don't exist there — fences only. Note `transition_buffers/textures` are no-ops today (`metal/command.rs:582-592`); they stay no-ops — Metal correctness comes from edges, not transitions.

### naga

- Enable `resource_table` in `front/wgsl/parse/directive/enable_extension.rs` (template: `WgpuBindingArray`, `:170/:193`), mapped to a new capability.
- Builtins in `call_builtin` (`front/wgsl/lower/mod.rs:3061`), template `T` via `TemplateListIter::ty` (like `bitcast<T>`, `:3062`). New IR expressions `ResourceTableGet/Has { ty_class, index }` + typifier/validator support.
- Lowering per backend = metadata load → compare → (mask test) → select → aliased-array access; `NonUniform` / `NonUniformResourceIndex` decoration driven by the existing uniformity analysis (`valid/analyzer.rs:551-614`; SPIR-V `back/spv/block.rs:878-883`, HLSL `back/hlsl/writer.rs:3983`) — decorate unconditionally in v0.
- Check emission via a new `BoundsCheckPolicies` field beside `binding_array` (`proc/index.rs:126`) — unchecked mode elides compare+select only (Invariant 4).
- Reflection: per-module "requests writable table types" bit in module info.
- Epoch immediate: reserved 4 bytes of immediate space on `uses_resource_table` pipelines (document the user-immediate reduction).
- Backend options: implicit table bind targets modeled on `sampler_heap_target` (`back/hlsl/mod.rs:517`); MSL runtime-array lengths via `binding_array_length_map` (`back/msl/mod.rs:431`); heterogeneous MSL declarations get `[[clang::may_alias]]`.

### Rust API sketch

```rust
let table = device.create_resource_table(&ResourceTableDescriptor { label, size })?; // size ≤ 65536
table.update(slot, BindingResource::TextureView(&view))?;   // Err(SlotInUse { available_after }) | Err(SamplerHeapFull) | ...
let slot = table.insert_binding(resource)?;                  // lowest-available-first (D8)
table.remove_binding(slot)?;
table.set_resource_usage(&texture, ResourceTableUsage::ReadOnly);          // device timeline
encoder.set_resource_usage(&table, &texture, ResourceTableUsage::None);    // queue timeline
compute_pass.set_resource_table(Some(&table));
RenderPassDescriptor { resource_table: Some(&table), .. };
PipelineLayoutDescriptor { uses_resource_table: true, .. };
```

WGSL: `enable resource_table;` + `getResource<T>(i)` / `hasResource<T>(i)` per `docs/bindless.md`.

## Work breakdown

Dependencies: 0.1 → {0.2, 0.4} → {0.3, 0.5} → 0.6 → 0.7 → 0.8 → {0.9, 0.10} → 0.11 → 0.12. S-bench parallel. M1+ sequential per milestone. Metal track independent after M0.

### S-bench (parallel, days)
Microbench: open-tail append vs separate CBs vs monolithic recording; segment-count scaling; all three backends. Prices the unscheduled per-dispatch-suspension option. **Accept:** numbers in a comment on the tracking issue.

### M0 — skeleton (unsafe-only, Vulkan, sampling)
- **0.1 wgpu-types**: three feature bits (naming rules `features.rs:30-50`; second u64 — `features.rs:1490`), `ResourceTableDescriptor`, `ResourceTableUsage`, error types. **Accept:** compiles everywhere, features documented as experimental+unsafe per [#8619](https://github.com/gfx-rs/wgpu/issues/8619).
- **0.2 naga front+IR**: enable, builtins, IR expressions, validation, typifier; WGSL error tests. **Accept:** naga snapshot tests for parse/validate.
- **0.3 naga SPIR-V**: unchecked lowering (aliased typed arrays at table set/binding, NonUniform always); reflection bit. **Accept:** spvasm snapshots + spirv-val clean.
- **0.4 wgpu-hal trait + noop**: `create/destroy_resource_table`, `update_table_slot`, `set_resource_table`, gap-aware encoder additions (`suspend_segment`/open-tail contract). **Accept:** noop + dyn wrappers compile; contract documented on the trait.
- **0.5 wgpu-hal Vulkan**: descriptor-set-backed table, new device features (gated exposure), set-index binding. **Accept:** hal-level smoke test renders with a table.
- **0.6 wgpu-core object**: registry/lifetime/slot-gating/resource→slots map/usage map/trace-replay; create/destroy/update/insert/remove with submission gating. **Accept:** lifetime tests incl. destroy-while-in-flight, double-destroy no-op, slot-reuse gating.
- **0.7 wgpu-core encoder state**: `set_resource_table` (compute pass + render pass descriptor), `uses_resource_table` pipeline-layout validation, draw/dispatch-time checks, render-bundle flag.
- **0.8 wgpu-core queue — the big one**: gap markers at table-bound pass starts; open-tail suspend at finish; submit-time append (exact barriers, membership normalization, mixed layout policy); submit-head fixup on `pending_writes`; layout-override tracker support. **Accept:** cross-encoder out-of-order-submit tests; VVL-clean under stress; add-to-table-after-finish tests.
- **0.9 conflict validation**: submit-time error when a table-visible resource is written in a scope of the submission (v0 semantics, D3). **Accept:** targeted validation tests.
- **0.10 compute dirty bits**: two-bit scheme, global compute barriers; layout policy inside table-bound passes. **Accept:** write→table-read interleaving stress test produces correct results under VVL.
- **0.11 wgpu API layer**: `ResourceTable` type, `dispatch.rs` methods, wgpu_core impl, webgpu/custom `unimplemented!` stubs, `as_hal`.
- **0.12 example + tests + CHANGELOG**: port a binding_array example to tables; gpu test suite skeleton (`tests/tests/wgpu-gpu/resource_table/`).

### M1 — checked path
Metadata buffer (u64 words) + defaults in K tail slots + naga checked lowering (policy-driven, Invariant 4) + destroy/remove → metadata zeroing + `hasResource` + checked bit becomes primary gate. **Accept:** type-confusion/OOB/stale-slot fuzz tests pass on checked path; unchecked path unchanged.

### M2 — usage states & spec-exact visibility
Epoch immediate plumbing; mask ring buffer + pass-start/end gap writes; per-dispatch scope digests in gap snapshots; `set_resource_usage` (table + encoder-at-boundary forms) with barrier elision; conflict errors replaced by exact hiding; reflection-bit consumption. **Accept:** visible/hidden/visible fuzz tests match a CPU reference implementation of `isShaderVisible` exactly; upstream write-up of the mechanism + measured costs.

### M3 — DX12
Heap-range tables, sampler-heap reuse, appended legacy-state barriers, naga HLSL lowering (overlapping ranges). **Accept:** backend-parametric conformance suite green on Vulkan+DX12.

### M3m — Metal explicit-sync track (parallel project)
M3a tracker producer tokens + hal edge API + fence lowering, **perf-parity gate vs auto-tracking**; M3b untracked/heap allocations; M3c tables (arg buffer, per-table gap fences, residency sets, MSL lowering). Optional bridge: tracked + conservative fence chain around table passes only.

### M4 — heterogeneous
Storage textures then buffers; mutable descriptor type (VK) / overlapping ranges (DX12) per D10; MSL `may_alias`; writable-entry dirty-bit triggers (reflection-narrowed); unsafe-mode audit; perf pass; upstream feedback digest.

### M5 — mid-pass usage changes
Suspension-on-demand at `compute_pass.set_resource_usage`: segment ends at the call; appended tail carries the precise named barrier (visibility already rides masks); Metal = encoder split + blit, no CB split. **Accept:** hide→write→set-readonly→sample in one pass, one precise barrier, VVL-clean.

## Open questions (do not silently decide)

1. Exact hal shape of the pass-edge (DAG) API and node identity for top-level transfer chunks (M3a design review).
2. Mask-ring default size / growth-gating policy; per-slot encoding compression if pathologies appear (exactness non-negotiable, representation negotiable).
3. Whether `wgpu` should expose a strict mode keeping v0's conflict-erroring after M2 (diagnostics value).
4. Spec TODOs we must track: type-compat rules ([#5470](https://github.com/gpuweb/gpuweb/issues/5470), [#5374](https://github.com/gpuweb/gpuweb/issues/5374)), default resources ([#5471](https://github.com/gpuweb/gpuweb/issues/5471)), insert ordering ([#5466](https://github.com/gpuweb/gpuweb/issues/5466)), uniformity ([#5582](https://github.com/gpuweb/gpuweb/issues/5582)), aliasing ([#5581](https://github.com/gpuweb/gpuweb/issues/5581)).
5. Upstream feedback backlog: sampler capacity vs 64k tables; the WGSL text's missing visibility coupling; Dawn/wgpu recorder-architecture asymmetry of the "update metadata before each usage scope" note; `writable` vs `storage-read-write` naming nit; `slot > size` off-by-one in `update()` steps.

## Glossary

- **Gap / suspension point**: a position where the hal CB's tail is left open at `finish()` and commands are appended at submit.
- **Submit-head fixup**: GPU work at the front of a submission (on `pending_writes`) applying device-timeline table deltas.
- **Epoch (E)**: index of a user dispatch within a table-bound compute pass, delivered via a reserved immediate.
- **Mask ring**: per-table GPU buffer of per-slot hidden-bit vectors over epochs; the truth table of `isShaderVisible` for slots whose visibility varies within a pass.
- **Type class**: the metadata enum fusing dimension/sample-type/multisampled/depth/access (+ sampler kinds); one compare validates a `getResource<T>`.
- **Mixed steady-state policy**: D11 layout rules.
- **Dirty bits**: the conservative intra-compute-pass barrier triggers (`dirty_write`, `dirty_table_read`).
- **Producer token / sync node**: tracker-recorded last-writer pass identity powering Metal fence edges.
- **Flavor**: the binding kind a slot was created with (sampled vs storage view, etc.); fixed per slot — a resource needing both occupies two slots.

## Progress log

- 2026-07-08 — Design settled (this document). No implementation started; no PRs exist.
