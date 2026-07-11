# M3 — DX12 backend

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — re-plan at milestone start; anchors drift, symbol names are stable.

## Goal

Bring the full checked + spec-exact-visibility resource table to DX12: heap-range tables in the existing shader-visible view heap, sampler-heap reuse, appended legacy-state barriers, and naga HLSL lowering with overlapping typed ranges. After M3 the feature is backend-parametric-conformance-green on Vulkan **and** DX12.

**Doc scope (:167, quote):** "Heap-range tables, sampler-heap reuse, appended legacy-state barriers, naga HLSL lowering (overlapping ranges)."
**Accept (:167, quote):** "backend-parametric conformance suite green on Vulkan+DX12."

Depends on **M1 + M2** (per doc :141 "M1+ sequential per milestone" and doc work-breakdown order M2→M3): DX12 must implement the checked path (M1) and the mask ring / spec-exact visibility (M2), not just the M0 skeleton.

## Binding decisions & invariants

- **D1** (:23) — design must hold for DX12 (fully designed in the doc).
- **D10** (:32) — **overlapping root-signature ranges unconditionally** on DX12 (SM 5.1 / FXC compatible; SM 6.6 explicitly off the path). Tier ≥ 2 required (wgpu floor); tier 3 for full sizes (doc :109).
- **D7** (:29) — DX12 keeps its existing global sampler heap unchanged; table sampler slots are metadata-only.
- **D16** (:38) — confirmed: aliased same-set/binding globals work (DXC ships the pattern) — the HLSL analogue of what Vulkan already runs.
- **Invariants 1–6** (:42-47) — all preserved on DX12; Invariant 2 (slot-reuse gating) maps onto DX12 submission indices, Invariant 3 (layout via legacy states, never conservative triggers).

## Inherited state (from M0/M1/M2 — pointers, no duplication)

- **hal DX12 stubs:** `pub struct ResourceTable;` + `unimplemented!()` bodies for `create/destroy_resource_table`, `update_table_slot`, `set_resource_table`, and `resource_table_memory_barrier` (m0-notes 0.4 :57, wave 7 :149 stub convention). M3 fills them.
- **DX12 free-list + heaps:** existing shader-visible view heap free-list (`dx12/descriptor.rs:33`, doc :109); existing global sampler heap + dedup/refcount cache (`dx12/sampler.rs`, doc :101-103) — the template D7 generalizes device-wide.
- **naga HLSL template:** `nagaSamplerHeap` codegen (`back/hlsl/writer.rs:1265`, options `hlsl/mod.rs:517`) — the implicit-bind-target model (doc :103, :121) M3 mirrors for the table image/metadata/mask-ring ranges.
- **Core is backend-agnostic:** the gap/splice, metadata buffer (M1), mask ring + epoch immediate (M2), `set_resource_usage`, slot gating are all in wgpu-core already. M3 is hal + naga only; core changes should be minimal (root-constant epoch immediate mapping, legacy-state barrier hooks).
- **Vulkan reference behavior** for every semantic — M3's conformance target is byte-for-byte behavioral parity with the Vulkan path on the shared suite.

## Carry-over items absorbed here

- DX12: heap-range tables, sampler-heap reuse, legacy-state barriers, naga HLSL overlapping-ranges lowering — **doc-stated**.

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

1. **Heap-range table object.** `create_resource_table` = contiguous `size+K` range in the shader-visible view heap (free-list gated by Invariant 2); `destroy` frees the range on submission retirement. *Verify:* hal smoke; free-list accounting under create/destroy churn.
2. **`set_resource_table` + root params.** One root descriptor table param + root SRV for metadata + root SRV/UAV for the mask ring; epoch immediate = a root constant (doc :109, :66). *Verify:* the table binds; epoch root-constant reaches the shader.
3. **Slot writes.** `update_table_slot` = `CopyDescriptorsSimple` into the range (gated by Invariant 2 slot gate). Metadata + mask writes mirror the core deltas. *Verify:* slot-reuse gate honored; VVL/DX debug-layer clean.
4. **Sampler-heap reuse.** Reuse the existing DX12 global sampler heap for D7; table sampler slots carry the heap index in metadata. Reconcile the dx12-hal dedup cache with the wgpu-core-owned refcount from M1. *Verify:* dedup across tables; capacity → `SamplerHeapFull`.
5. **Legacy-state barriers in appended gap commands.** The submit-time gap splice (core) emits DX12 legacy resource-state transitions instead of Vulkan pipeline barriers; `Close()` at submit per the splice model (errata (a)). Compute hazard barrier = NULL UAV barrier (doc :93). *Verify:* correct states for cross-pass table-sampled members; debug layer clean.
6. **naga HLSL lowering.** Overlapping unbounded typed ranges per type class over the same heap range (D10); metadata load → compare → (mask test) → select → NonUniform (via `NonUniformResourceIndex`, doc :117). Implicit bind targets modeled on `sampler_heap_target` (`hlsl/mod.rs:517`). *Verify:* hlsl snapshots; DXC + FXC compile; SM 5.1 target.
7. **Conformance parametrization.** Make the resource_table e2e suite backend-parametric (drop the Vulkan-only skip for the covered tests); run on DX12 tiers ≥ 2. *Verify:* full suite green on Vulkan + DX12 (Accept).

## Open questions requiring a user decision

- **Sampler-heap reuse mechanics** (doc :167): does the wgpu-core-owned sampler refcount (M1) subsume the dx12-hal cache, or do they coexist with the hal cache as a device-global backing? Determines whether M1's core sampler heap needs a DX12-specific backing hook.
- **HLSL lowering template specifics:** the exact overlapping-range root-signature layout and the aliased-typed-range declaration form for SM 5.1 (FXC) — confirm against a DXC-shipped reference before committing the template.
- **Root-constant budget for the epoch immediate:** confirm 4 bytes of root-constant space is acceptable given DX12 root-signature size limits on the tables path.

## Risks / landmines

- **DX12 stub convention:** `unimplemented!()` bodies exist for every table method (m0-notes convention). Grep for them before starting; each is a required fill, and a missed one panics only at runtime.
- **SM 6.6 is explicitly OFF the path** (D10) — do not reach for `ResourceDescriptorHeap`; the design commits to overlapping ranges for SM 5.1/FXC compatibility.
- **Tier gating:** tier 2 is the floor; full sizes need tier 3 (doc :109). Feature exposure must gate on the tier like Vulkan gates on descriptor-indexing bits.
- **Legacy-state model vs Vulkan layout model:** Invariant 3 still holds — layouts (states) are pinned by construction, never by conservative triggers. The mixed-layout policy's storage-member handling is M4; M3's sampled-only members stay in the read state.
- **Behavioral parity is the acceptance, not "it runs":** any DX12 divergence from the Vulkan reference on the shared suite is a bug, including visibility edge cases from M2.

## Verification strategy

Harness/gates per m0-notes cheat sheet (:179-189); the box already generates DX12 test variants (4 DX12 adapters noted at m0-notes :182). M3's headline axis is **backend-parametric conformance**: the same e2e tests (smoke, regression C1/C2/M1, render, lifecycle, negative, conflict/binding, M1 type-confusion/OOB/stale-slot, M2 visible/hidden/visible-vs-CPU-reference) run on both Vulkan and DX12 with identical expected outputs. Add DX12-debug-layer cleanliness as the DX12 analogue of the VVL canary.
