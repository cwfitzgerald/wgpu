# M1 — checked path

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — the executing orchestrator re-plans at milestone start against the then-current tree; anchors drift, symbol names are stable.

## Goal

Make the checked path real: introduce the metadata buffer, decode it in a naga **checked** lowering (type-class compare + `select` to a default slot), add `hasResource<T>`, add `getResource<sampler>` on a device-global sampler heap, and zero metadata on destroy/remove. After M1 the checked bit is the primary safety gate; the unchecked path stays byte-for-byte identical.

**Doc scope (:161, quote):** "Metadata buffer (u64 words) + defaults in K tail slots + naga checked lowering (policy-driven, Invariant 4) + destroy/remove → metadata zeroing + `hasResource` + checked bit becomes primary gate."
**Accept (:161, quote):** "type-confusion/OOB/stale-slot fuzz tests pass on checked path; unchecked path unchanged."

## Binding decisions & invariants

- **D3** (:25) — semantics-only staging: M1 may still reject spec-valid programs (visibility conflicts stay ERROR until M2), never accepts spec-invalid ones.
- **D4** (:26) — checked is the primary path; UNCHECKED elides only the naga compare+select. M1 makes the checked emission exist for the first time.
- **D7** (:29) — samplers are one deduplicated device-global sampler heap; table sampler slots are metadata-only (heap index in payload bits 16..31). `update()` returns `Err(SamplerHeapFull)` at capacity. Dedup/refcount in wgpu-core.
- **D9** (:31) — checks at whole-resource granularity.
- **Invariant 3** (:44) — layout correctness never depends on conservative triggers. Metadata is a memory-ordering/visibility concern, not a layout one; a missed metadata write ⇒ wrong-slot read (still memory-safe under bounds check), never driver UB.
- **Invariant 4** (:45) — checked/unchecked single code path: the ONLY difference is the bounds/visibility policy eliding compare+select. No separate lowering. M1's central correctness obligation.
- **Invariant 5** (:46) — tables hold `Arc`s; `destroy()` zeroes slots' metadata at the next submit head; hal teardown stays deferred by submission index.

## Inherited state (from M0 — do not duplicate m0-notes; pointers)

- **naga IR + front:** `Expression::ResourceTableGet { ty, index }`, `Capabilities::RESOURCE_TABLE = 1<<44`, enable ident `resource_table`, `ModuleInfo::uses_resource_table()` / `::requests_writable_table_types()` (m0-notes landed shapes :26). `hasResource` is **not** parsed yet (deviation 3 :18) — M1 adds it.
- **naga SPIR-V unchecked lowering:** `Options.resource_table_target: Option<ResourceTableBindTarget { descriptor_set, binding }>`; `BoundsCheckPolicies.resource_table: BoundsCheckPolicy` (default Unchecked). **Any non-Unchecked policy currently returns `Error::FeatureNotImplemented("checked resource-table lowering arrives with the metadata buffer (M1)")`** (m0-notes 0.3 :30) — M1 replaces this stub with the real checked emission. Aliased runtime-array globals, unconditional `NonUniform`, spv≥1.4 interface listing all landed (:31-35).
- **naga landmines (must extend for `hasResource` + checked):** `valid/function.rs` emittable list, `valid/expression.rs global_var_ty`, `back/spv/image.rs get_handle_id` (arm added in 0.3), `proc/constant_evaluator.rs`, `valid/analyzer.rs GlobalOrArgument::ResourceTable` (m0-notes :163-168). Any new IR expression (`ResourceTableHas`) touches all of these.
- **hal Vulkan:** device-wide shared UAB set layout; binding 0 = `SAMPLED_IMAGE` array. **binding 1 = metadata SSBO** and the sampler-heap set are NOT created yet — M1 adds them. `DeviceShared.resource_table: Option<ResourceTableShared>`, per-table pool+set (m0-notes 0.5 :60-67). Deferred naga compile for table shaders is load-bearing (`ShaderInput::Naga` only).
- **core object:** `wgpu_core::resource_table::ResourceTable { slots: Box<[Slot { available_after }]> , … }`, snatchable hal handle, deferred destroy via `TempResource::DestroyedResourceTable`, slot gate `check_slot_available` vs `Device::last_completed_submission_index`, `mark_all_slots_in_use` (m0-notes 0.6 :68-76, 0.8 :88-93). **No metadata buffer, no resource→slots multimap for usage yet** beyond `texture_to_slots`.
- **core queue splice:** `ResourceTableGap { table, insertion_point }`, two-phase `process_resource_table_gaps` (ascending compute / descending splice), submit-head fixup on `pending_writes` (m0-notes 0.8 :90). M1's destroy/remove zeroing rides the submit-head fixup.
- **public API:** `wgpu::ResourceTable` with `update(&TextureView)` / `insert_binding` / `remove_binding` / `as_hal`; `ResourceTableError` (m0-notes 0.11 :95-97). M1 keeps texture-only signatures (BindingResource widening is M4) but must add a sampler entry point for `getResource<sampler>` — see open questions.

## Carry-over items absorbed here

- `hasResource<T>`, `getResource<sampler>` + device-global sampler heap (D7), metadata buffer + checked lowering, K default tail slots, destroy/remove metadata zeroing — **doc-stated**.
- Recorded-but-unsubmitted-CB slots are not slot-gated (m0-notes wart :118) — **M1 risk** for metadata zeroing (see Risks).
- Error-scope integration for `ResourceTableError::Other(String)` (m0-notes :96) — **[suggested]** API-polish item.
- Port a `binding_array` example to tables — **[suggested]** (Connor deferred in M0).

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

1. **Metadata word layout (types/core).** Encode the `u64` per the doc (:70-75): bits 0..11 `type_class`, bits 12..15 flags (`has_mask` — stays 0 until M2), bits 16..31 payload (sampler heap index / debug generation), bits 32..63 `mask_ptr` (unused until M2). Add a `type_class` derivation from `TextureView`/sampler flavor + a metadata-word packer in core. *Verify:* unit tests on the packer round-trip; `has_mask` never set at M1.
2. **Core metadata buffer.** Per-table metadata buffer (hal buffer owned by the table); write a metadata word on every `update`/`insert_binding`; zero on `remove_binding`/`destroy` **at the next submit head** (Inv.5). Snapshot deltas fold into the existing submit-head fixup (m0-notes 0.8 :90). *Verify:* validation test that destroy zeroes the word visible to a later submission; slot-gate still blocks in-flight rewrite.
3. **hal metadata binding.** Vulkan: add binding 1 (metadata SSBO) to the shared set layout + a per-table metadata buffer bound into the table's set (baked CBs address it). New/extended hal `update_table_slot` payload carries the metadata word alongside the descriptor write. *Verify:* hal smoke renders reading metadata; VVL-clean.
4. **naga checked SPIR-V lowering.** Implement the non-Unchecked policy: emit `meta = metadata[slot]; ok = meta.type_class == EXPECTED; idx = select(DEFAULT_SLOT_FOR_T, slot, ok)` then the NonUniform array access (doc snippet :79-83, **minus** the `has_mask`/mask-ring branch which is M2 — see open questions). Requires a **metadata bind target** in `spv::Options` (mirror `resource_table_target`). Remove the `FeatureNotImplemented` stub (m0-notes :30). *Verify:* spvasm snapshot + spirv-val clean (`cargo xtask validate spv` from `naga/`); a snapshot exercising type-mismatch → default slot.
5. **`hasResource<T>` in naga.** Parse `hasResource<T>(i)` (front/wgsl), new IR `Expression::ResourceTableHas { ty, index }` (or reuse a predicate flag), typifier → bool, validator, all four landmine sites (:163-168). Lowering = identical predicate to getResource's `ok`, returned as bool (doc :85 "hasResource uses the identical predicate"). *Verify:* parse/validate snapshots; spvasm snapshot; on the checked path `hasResource` matches getResource's visibility exactly.
6. **`K` default tail slots + per-type-class defaults.** Allocate K hidden tail slots (one per supported type class), populate with a default resource per class (no null-descriptor dependency, doc :53). `DEFAULT_SLOT_FOR_T` in the lowering points here. **K value + default contents need a user decision (TODO #5471).** *Verify:* OOB/type-mismatch access lands on the default slot and reads defined values under VVL.
7. **Device-global sampler heap (D7) + `getResource<sampler>`.** Move the dx12-hal dedup/refcount sampler-cache concept into wgpu-core (doc :101-103); Vulkan = device-global `SAMPLER` UAB array sized `min(4000, limit)`. Sampler table slots are metadata-only (heap index in payload). Add a core sampler-update entry point + `SamplerHeapFull` error. naga: `getResource<sampler>` extracts the heap index from the metadata word it already loaded (front-end error removed). *Verify:* e2e sampling with a table-sourced sampler; heap dedup refcount test; `SamplerHeapFull` at capacity.
8. **Checked bit becomes primary gate.** Wire pipeline creation so `EXPERIMENTAL_SAMPLING_RESOURCE_TABLE` alone (without UNCHECKED) selects the checked policy; UNCHECKED continues to elide compare+select (Inv.4). M0's "UNCHECKED required" contract (m0-notes deviation 5 :20) is lifted. *Verify:* a table pipeline compiles + runs with only SAMPLING enabled; unchecked path snapshot unchanged (byte-diff the M0 spvasm).
9. **[suggested] Error-scope integration** for the rare `ResourceTableError::Other(String)` kinds (m0-notes :96). *Verify:* the collapsed kinds surface through the error sink where appropriate. Confirm scope with Connor.
10. **[suggested] Port a `binding_array` example** to tables + CHANGELOG. *Verify:* example runs on a real adapter.

## Open questions requiring a user decision

- **`K` and per-type-class default contents** (spec TODO #5470/#5471, doc :183): how many default tail slots, and what resource each holds (1×1 texture per dimension/sample-type? shared?). Blocks work item 6 — this is the OOB/type-mismatch fallback target.
- **`has_mask` semantics at M1** (mask ring doesn't exist until M2): confirm the checked lowering emits the type-class compare + select only, with `has_mask` hard-wired 0 and the mask-ring branch absent (added by M2). Is a dead `has_mask` bit in the word acceptable at M1, or should M1 gate the branch out at codegen?
- **Type-compat rules** (spec TODO #5470/#5374, doc :183): what exactly makes a stored resource's `type_class` "compatible" with a `getResource<T>` request (exact match only, or subtyping across sample-type/arrayed)? Determines the compare in work item 4.
- **Sampler-heap capacity behavior / acceptance** (doc :184 upstream backlog): is `min(4000, limit)` the right Vulkan cap, and is returning `SamplerHeapFull` (vs a device-lost) the accepted surface? Confirm the public sampler-update API shape (a new `update_sampler`? widen `update`?).
- **Error-scope [suggested]** and **example [suggested]**: confirm these belong in M1 (Connor deferred the example "for now" in M0).

## Risks / landmines

- **Recorded-but-unsubmitted-CB slots are not slot-gated** (m0-notes wart :118). Metadata zeroing on destroy/remove happens at submit head; a CB recorded-but-not-yet-submitted that references a since-zeroed slot on the checked path reads the default slot (memory-safe) rather than stale data. Confirm this is the intended checked-path behavior and add a regression test — it is the M0-flagged landmine for exactly this milestone.
- **Invariant 4 is easy to break:** the checked and unchecked emissions must diverge *only* at the compare+select. Byte-diff the unchecked spvasm against M0 as a snapshot guard (work item 8).
- **naga expression proliferation:** `ResourceTableHas` (if a new variant) must be added to every exhaustive match the m0-notes landmines enumerate (:163-168) — the emittable list, `global_var_ty`, `get_handle_id`, `constant_evaluator`, `analyzer`. Missing one is a compile error at best, a validation hole at worst.
- **Sampler heap dedup/refcount + submission gating** interacts with the slot gate: heap-slot reuse is gated on submission indices (doc :103). Do not free a heap slot while an in-flight submission's metadata payload still points at it.
- **Metadata write ordering vs descriptor write:** both must be visible at the same submission boundary; a torn update (descriptor visible, metadata stale, or vice-versa) is a wrong-type read. Fold both into one submit-head delta.

## Verification strategy

Base harness + gates are in the m0-notes Test/verify cheat sheet (:179-189): `.features(EXPERIMENTAL_SAMPLING_RESOURCE_TABLE)` (drop UNCHECKED once work item 8 lands — that is itself a test axis), Vulkan-only skip, VVL canary auto-fails, `cargo xtask test -E 'binary(wgpu-gpu) and test(resource_table)'`. Reuse `common.rs` helpers (m0-notes wave 6 :139).

New test axes M1 adds:
- **Type-confusion fuzz on the checked path:** `getResource<T>` against a slot holding a different type class → default slot, not a mis-typed read. Compare against the unchecked path (which must be unchanged).
- **OOB fuzz:** index ≥ size and index into an empty slot → default slot.
- **Stale-slot fuzz:** destroy/remove a member, then a later submission's checked access → default slot / metadata-zeroed; the recorded-but-unsubmitted-CB case (Risks) explicitly covered.
- **`hasResource` vs `getResource` agreement:** identical predicate on the checked path for every axis above.
- **Sampler heap:** dedup refcount, `SamplerHeapFull`, e2e sample-through-table.
- **Unchecked-path-unchanged guard:** naga spvasm snapshot byte-identical to M0 for the unchecked policy.
- naga snapshots: `cargo nextest run -p naga` (running blesses; review with `jj diff`). Ignore the pre-existing `recursion_depth_template` stack-overflow failure (m0-notes :189).
