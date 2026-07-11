# M2 — usage states & spec-exact visibility

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — re-plan at milestone start; anchors drift, symbol names are stable.

## Goal

Replace M0's submit-time incompatible-member ERROR with **spec-exact per-dispatch hiding**: build the epoch-immediate + mask-ring mechanism so `getResource`/`hasResource` behave as-if evaluating `isShaderVisible(slot, wgslType, usage_scope)`, add `set_resource_usage` in both forms with barrier elision, and consume the reflection bit. After M2, hiding is exact (never more than spec — Invariant 6).

**Doc scope (:164, quote):** "Epoch immediate plumbing; mask ring buffer + pass-start/end gap writes; per-dispatch scope digests in gap snapshots; `set_resource_usage` (table + encoder-at-boundary forms) with barrier elision; conflict errors replaced by exact hiding; reflection-bit consumption."
**Accept (:164, quote):** "visible/hidden/visible fuzz tests match a CPU reference implementation of `isShaderVisible` exactly; upstream write-up of the mechanism + measured costs."

## Binding decisions & invariants

- **D3** (:25) — each milestone only relaxes behavior: M2 is the milestone where spec-valid programs previously rejected start being accepted (via hiding), never the reverse.
- **D9** (:31) — whole-resource visibility granularity.
- **D12** (:34) — **visibility is spec-exact at per-dispatch granularity**; the `isShaderVisible` coupling in `docs/bindless.md` is normative; implemented via epoch immediate + mask ring; arbitrary visible/hidden/visible patterns representable exactly.
- **D13** (:35) — mid-pass `set_resource_usage` (M5) rides the M2 masks; M2 builds the mask machinery it will ride.
- **Invariant 1** (:42) — submission-boundary observability: mask/usage-state writes are GPU-timeline writes at gaps/submit head.
- **Invariant 3** (:44) — layout correctness never depends on conservative triggers; the mask ring is a *visibility* mechanism, never a layout one.
- **Invariant 6** (:47) — hiding MORE than spec is **forbidden after M2**; barriering more than needed stays allowed. This is M2's hard correctness line.

## Inherited state (from M0/M1 — pointers, no duplication)

- **M0 conflict machinery to be replaced/generalized:** `ResourceTableConflictError::IncompatibleMemberUsage` → `QueueSubmitError::ResourceTableConflict`; `ResourceTable::find_incompatible_member` (reads live membership); `TextureTracker::collect_table_incompatible_usages` + `TABLE_INCOMPATIBLE_USAGES` mask; CB field `resource_table_member_usages` (m0-notes wave 7 :146, wave 8 :158). track/texture.rs:433-592.
- **M0 dirty-bit barriers become load-bearing:** the two-bit `dirty_write`/`dirty_table_read` scheme in `command/compute.rs`, `flush_resource_table_barrier` after `flush_bindings`, hal `resource_table_memory_barrier` (Vulkan `VkMemoryBarrier` COMPUTE→COMPUTE) (m0-notes wave 7 :149-150). In M0 it was redundant (0.9 rejected the hazards); M2 removes the rejection so this barrier is the real thing.
- **M0 splice / gap machinery:** `ResourceTableGap { table, insertion_point }`, two-phase `process_resource_table_gaps` (m0-notes 0.8 :90). M2 extends the gap **snapshot** to carry per-dispatch scope digests + the merged writable set (doc :60) and adds pass-start/end mask-ring writes to the appended tail.
- **M1 metadata word:** `has_mask` (bit 12) + `mask_ptr` (bits 32..63) fields exist but are inert (M1 keeps `has_mask` 0). M2 turns them on. Metadata SSBO binding (binding 1) and per-table metadata buffer landed in M1.
- **Reflection bit:** `ModuleInfo::requests_writable_table_types()` (m0-notes :26) — consumed here to drive the hetero write side of `dirty_write`.

## Carry-over items absorbed here

- Replace submit-time incompatible-member ERROR with spec-exact hiding; mask ring, epoch immediate, `set_resource_usage`, per-dispatch scope digests — **doc-stated**.
- 0.10 compute barrier becomes load-bearing — **doc-implied** (m0-notes :150).
- **Sandwich-escape layout holes [suggested]:** the per-dispatch gap/mask machinery gives the precision M0's start∪end folding lacks. M2 must either **close** these escapes or **consciously extend** the rejection. Two shapes (residual doc at `TextureTracker::collect_table_incompatible_usages`, track/texture.rs:565-592, m0-notes :160):
  - *intra-pass* (sample → storage-write → sample in one compute pass): layout-unsafe, memory-safe via the 0.10 barrier — per-dispatch masks make the write→table-read edge precise.
  - *cross-pass w/ top-level transfer* (bindful sample; transfer write into member; table read in a LATER pass, no bindful use there): genuine memory hazard AND layout-unsafe today, unreached by any barrier — M2 must handle it explicitly.
- **Split with M4 (make explicit here and in m4-plan):** M2 owns per-dispatch visibility + intra-pass precision for sampled members. Full D11 mixed steady-state **GENERAL-pinning** has no subjects until storage-capable members exist → the pinning mechanism is **M4**, not M2. M2's masks are visibility-only; layout-pinning of storage members is out of M2 scope.

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

1. **Epoch immediate plumbing.** Reserve 4 bytes of immediate space on `uses_resource_table` pipelines (doc :120); rewrite it before each user dispatch in table-bound compute passes (internal injected dispatches don't increment E — doc :66). Document the user-immediate reduction. *Verify:* the immediate reaches the shader; injected indirect-validation dispatches don't perturb E.
2. **Mask ring buffer (hal + core).** Per-table mask-ring buffer, internal binding of the table's descriptor set (Vulkan binding 2, doc :107). Ring reuse gated on submission retirement; growth is a rare gated event. *Verify:* VVL-clean allocation/binding; ring wrap under stress.
3. **Per-dispatch scope digests in gap snapshots.** Extend the gap snapshot (doc :60) to retain per-dispatch compact usage digests for table-bound compute passes + the pass's merged writable set. `conflict(R, E)` is encode-known (doc :66). *Verify:* digest count matches dispatch count; render pass = single scope.
4. **Pass-start/end mask writes.** At the pass-start gap, for each slot whose visibility *varies within the pass*, write the exact hidden-bit vector over N epochs (⌈N/32⌉ words — the truth table; visible/hidden/visible = `0,1,0`, doc :77). Pass-constant slots (vast majority + all render passes) use plain word writes, no mask. Pass-end gaps restore varied words. *Verify:* CPU-reference match (see Accept).
5. **naga checked lowering: mask branch.** Turn on the `has_mask` branch from the doc snippet (:79-83): `if ok && meta.has_mask { ok = !bit(mask_ring[meta.mask_ptr + E/32], E%32) }`. Requires a mask-ring bind target in `spv::Options`. *Verify:* spvasm snapshot + spirv-val; the M1 no-mask path unchanged when `has_mask==0`.
6. **`set_resource_usage` (both forms).** `table.set_resource_usage(&resource, usage)` (device timeline) and `encoder.set_resource_usage(&table, &resource, usage)` (queue timeline, at a boundary) — doc Rust sketch :130-131. Usage `None | ReadOnly | Writable` folds into the mask bits at submit. *Verify:* usage state changes flip visibility per the CPU reference.
7. **Barrier elision.** `set_resource_usage(None)` on written resources (encode-visible) prevents their writes setting `dirty_write`; reflection bit handles the hetero write side; row-1 bindful-vs-bindful hazards keep today's precise per-dispatch treatment (`compute.rs flush_bindings`) — doc :94. *Verify:* elision drops the barrier where declared safe; VVL-clean.
8. **Replace the conflict ERROR with hiding.** Remove `find_incompatible_member`'s rejection for the *visibility* cases; a written-then-table-read resource is now hidden per-dispatch, not rejected. Decide the fate of the *layout* cases (sandwich-escape — carry-over above). *Verify:* the M0 conflict tests flip from "rejected" to "hidden and correct"; Invariant 6 (no over-hiding) fuzzed.
9. **Reflection-bit consumption.** Wire `requests_writable_table_types()` into the hetero `dirty_write` side (doc :92). *Verify:* a module requesting writable table types sets the bit; barrier emitted accordingly.
10. **Upstream write-up + measured costs** (Accept criterion). *Verify:* mechanism doc + numbers on the tracking issue.

## Open questions requiring a user decision

- **Mask-ring default size + growth-gating policy** (doc Q2, :181): "Mask-ring default size / growth-gating policy; per-slot encoding compression if pathologies appear (exactness non-negotiable, representation negotiable)." Blocks work item 2/4.
- **Per-slot mask-pointer encoding** (doc Q2): multi-slot resources get per-slot mask pointers, shareable when equal (doc :85). Confirm the sharing/compression scheme and the `mask_ptr` word-offset encoding.
- **Scope-digest data structure:** what compact form for per-dispatch usage digests (bitset of bindful writable usages per dispatch?) balances gap-snapshot size against the truth-table computation at submit.
- **Strict-mode retention** (doc Q3, :182): "Whether `wgpu` should expose a strict mode keeping v0's conflict-erroring after M2 (diagnostics value)." If yes, work item 8 keeps the erroring path behind a flag.
- **Cross-pass sandwich-escape:** close it via per-dispatch machinery, or consciously extend the M0 layout-incompatibility rejection? It is a genuine memory hazard today (not merely layout-unsafe) — this needs an explicit decision, not silent carry-forward.
- **Render bundles with tables** (currently `CreateRenderBundleError::ResourceTableUnsupported`, m0-notes 0.8 :93): candidate to unblock once replay visibility semantics settle. Record whether M2 lifts the rejection or defers.

## Risks / landmines

- **Invariant 6 (no over-hiding) is the hard line.** The mask truth table must exactly equal `isShaderVisible`; hiding a slot the spec would show is a conformance failure. The Accept criterion (CPU-reference fuzz) exists precisely to catch this.
- **Epoch counting vs injected dispatches:** internal indirect-validation dispatches must not increment E (doc :66) — M0 already special-cases them for the dirty bits (m0-notes :149); reuse that seam.
- **Mask-ring addressing from baked CBs:** the ring is an internal binding of the table's set so already-recorded CBs can address it — do not route it through a user bind group.
- **Two writers of the metadata word:** M1's `has_mask` bit and M2's `mask_ptr` must be written coherently with the descriptor; keep folding through the single submit-head delta.
- **Load-bearing 0.10 barrier:** with the conflict ERROR gone, a missed compute→compute barrier is now an actual data race (was masked by rejection in M0). Re-audit the two-bit scheme against the new accepted programs.

## Verification strategy

Harness/gates per m0-notes cheat sheet (:179-189). Reuse `conflict.rs` / `binding.rs` e2e modules (m0-notes wave 7 :151) — several M0 conflict tests **flip** from rejection to correct-hiding.

New test axes M2 adds:
- **Visible/hidden/visible fuzz vs a CPU reference of `isShaderVisible`** (the doc's Accept criterion) — random usage-state timelines + dispatch sequences, mask output compared exactly.
- **Over-hiding guard (Invariant 6):** assert no slot is hidden that the reference shows visible.
- **`set_resource_usage` both forms** driving visibility flips within and across passes.
- **Barrier elision correctness:** `set_resource_usage(None)` drops the barrier and results stay correct under VVL; the load-bearing barrier still fires where needed (data-race stress).
- **Sandwich-escape:** whichever decision is taken (close vs extend), a regression test pins it.
