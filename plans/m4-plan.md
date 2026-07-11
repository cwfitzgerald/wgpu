# M4 — heterogeneous resource tables

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — re-plan at milestone start; anchors drift, symbol names are stable.

## Goal

Admit non-sampled resources into tables (storage textures first, then buffers) behind `EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE`, using mutable descriptor types on Vulkan / overlapping ranges on DX12, `[[clang::may_alias]]` on Metal. This is where the D11 mixed steady-state **GENERAL-pinning** mechanism finally has subjects, where the writable dirty-bit triggers become real, and where update/insert signatures widen toward `BindingResource`.

**Doc scope (:173, quote):** "Storage textures then buffers; mutable descriptor type (VK) / overlapping ranges (DX12) per D10; MSL `may_alias`; writable-entry dirty-bit triggers (reflection-narrowed); unsafe-mode audit; perf pass; upstream feedback digest."
**Accept:** the doc gives **no explicit `Accept:` line** for M4. Implicit bar: storage-capable members work correctly (compute write→table-read) with layouts pinned by construction, on all shipped backends, unsafe-mode audited, with no perf regression on the sampled path. Confirm the acceptance bar with Connor (open question).

Depends on **M2** (spec-exact visibility + per-dispatch precision; the mask machinery the writable side rides). D11 GENERAL-pinning is deliberately deferred here from M2 — see the split note.

## Binding decisions & invariants

- **D4** (:26) — `EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE` (implies sampling); checked/unchecked share one path.
- **D10** (:32) — **`VK_EXT_mutable_descriptor_type` unconditionally** on Vulkan (feature not exposed without it); **overlapping root-signature ranges unconditionally** on DX12 (SM 5.1/FXC); SM 6.6 off the path.
- **D11** (:33) — **no blanket GENERAL layouts**; sampled-only members hold `SHADER_READ_ONLY_OPTIMAL`; storage-capable members use `GENERAL` descriptors, normalized to `GENERAL` at table-visible gaps and kept there inside table-bound compute passes. **This is the milestone that implements the pinning** — it had no subjects before storage members existed.
- **D16** (:38) — MSL `may_alias` / argument-buffer aliasing facts confirmed.
- **Invariant 3** (:44) — storage writes require `GENERAL`; layout stays correct by the mixed-steady-state construction, never by conservative triggers. The hard safety line here.
- **Invariant 6** (:47) — over-approximate only in the safe direction.

## Inherited state (from M0–M2 — pointers, no duplication)

- **Reflection bit:** `ModuleInfo::requests_writable_table_types()` (m0-notes :26) — M2 consumes it for the hetero `dirty_write` side (m2-plan work item 9); M4 makes writable table types actually expressible, so this bit starts firing for real.
- **Layout policy scaffolding:** M0's `TABLE_INCOMPATIBLE_USAGES` mask + `collect_table_incompatible_usages` (track/texture.rs:433-592) rejects storage/writable member usages today; M4 replaces the rejection for storage-capable members with the D11 GENERAL-pinning steady state. The residual sandwich-escape (m0-notes :160) partly resolves here.
- **hal per-backend heterogeneous hooks:** Vulkan shared set layout uses `MUTABLE_EXT` array for the hetero variant (doc :107); DX12 overlapping ranges (doc :109); the sampled variant is what M0–M3 built.
- **Public API:** M0/M1 kept `update`/`insert_binding` as `&TextureView`-only (m0-notes 0.11 :96, "enum widening lands with M4"). M4 widens toward `BindingResource`.
- **Dirty-bit scheme:** M0's two-bit compute barrier (`dirty_write`/`dirty_table_read`, m0-notes :149) + M2's reflection-bit consumption — M4 narrows the writable-entry triggers via reflection.

## Carry-over items absorbed here

- Heterogeneous: storage textures then buffers; VK mutable descriptor type / DX12 overlapping ranges (D10); MSL `may_alias`; writable-entry dirty-bit triggers (reflection-narrowed); unsafe-mode audit; perf pass; upstream feedback digest — **doc-stated**.
- **D11 GENERAL-pinning mechanism** — the M2/M4 split: M2 owns per-dispatch visibility + intra-pass precision for sampled members; **M4 owns the GENERAL-pinning of storage-capable members** (no subjects until now). Made explicit in [m2-plan.md](m2-plan.md) too.

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

1. **Storage-texture type classes (naga + core).** Extend `type_class` (metadata bits 0..11) to storage textures (access dimension); enable `getResource<T>` for storage texture types (the M0 "storage → error pointing at the heterogeneous milestone", m0-notes deviation 3 :18, is lifted). *Verify:* naga snapshots; type-class compare covers storage.
2. **Vulkan `MUTABLE_EXT` set layout.** Switch the shared set layout to `VK_EXT_mutable_descriptor_type` for the hetero feature; gate exposure on the extension (D10). *Verify:* hal smoke with mixed sampled + storage members; VVL-clean.
3. **DX12 overlapping ranges for storage** (D10). *Verify:* HLSL snapshots; DXC/FXC compile.
4. **D11 GENERAL-pinning.** Storage-capable members get `GENERAL` descriptors; normalize to `GENERAL` at table-visible gaps; the tracker keeps every storage-capable texture in `GENERAL` inside table-bound compute passes (superset policy, encode-decidable from `TextureUsages`, doc :99). Sampled-only members stay `SHADER_READ_ONLY_OPTIMAL`. *Verify:* layout correctness under VVL for sample→write→sample; the intra-pass sandwich-escape closes.
5. **Writable-entry dirty-bit triggers (reflection-narrowed).** `dirty_write` set by table-using dispatches whose module requests writable table types (the reflection bit), narrowing the conservative trigger (doc :92). *Verify:* barrier fires for writable table use, elided for read-only.
6. **Storage buffers in tables.** After storage textures: buffers (no layouts, doc :99). *Verify:* buffer getResource e2e.
7. **`BindingResource`-style signature widening.** Widen `update`/`insert_binding` from `&TextureView` toward `BindingResource` (m0-notes 0.11 :96). *Verify:* API accepts texture views, storage views, buffers; back-compat for existing texture callers.
8. **MSL `may_alias`.** Heterogeneous MSL declarations get `[[clang::may_alias]]` (doc :121). *Verify:* MSL snapshots; Metal e2e (needs M3c + macOS runner).
9. **Unsafe-mode audit.** Re-audit the UNCHECKED path against the wider type surface — the checked/unchecked single-path invariant (Inv.4) must still hold with storage types. *Verify:* unchecked spvasm diff is compare+select-only vs checked.
10. **Perf pass + upstream feedback digest** (doc :173). *Verify:* no regression on the sampled path; write-up.

## Open questions requiring a user decision

- **Admissible resource types vs the proposal's no-uniform-buffers stance** (doc :184 upstream backlog): which resource types are admitted (storage textures, storage buffers, read-only storage; uniform buffers?) and in what order beyond "storage textures then buffers"?
- **update/insert signature-widening shape:** full `BindingResource` mirror, or a narrower table-specific enum? Affects the public API surface committed at M4 (m0-notes flagged this as the M4 deliverable).
- **Writable dirty-bit narrowing rule:** the exact predicate for reflection-narrowed `dirty_write` — per-module bit only, or per-slot/per-type refinement? Determines barrier precision vs cost.
- **GENERAL-pinning scope on drivers without `VK_KHR_unified_image_layouts`** (doc :99 notes it makes residual `GENERAL` free where present): accept the compression cost of `GENERAL` storage members where the extension is absent?
- **Acceptance bar** (no doc `Accept:` line): what is the concrete definition of done for M4 (which backends must be green, what perf-regression tolerance)?

## Risks / landmines

- **Invariant 3 is the whole point here:** the GENERAL-pinning must make wrong-layout-by-omission impossible by construction. A storage member that escapes `GENERAL` at any gap or inside a pass is driver UB — the residual sandwich-escape (m0-notes :160) must be closed for storage members, not merely barriered.
- **Mutable-descriptor-type exposure gating:** the Vulkan hetero feature must be unexposed without `VK_EXT_mutable_descriptor_type` (D10) — no silent fallback.
- **Checked/unchecked single path (Inv.4)** must survive the wider type surface: the only divergence stays compare+select. The unsafe-mode audit (work item 9) exists to enforce this.
- **Cross-pass sandwich-escape** (m0-notes :160) is a genuine memory hazard whose full fix may need this milestone's GENERAL-pinning + M2 masks together — coordinate with the M2 decision (m2-plan open questions).
- **Metal `may_alias` correctness** depends on M3c landing first; on the dev box this axis is untestable.

## Verification strategy

Harness/gates per m0-notes cheat sheet (:179-189); gate on `EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE` (m0-notes :25). New axes:
- **Storage write→table-read** correctness under VVL on Vulkan (+ DX12/Metal as available), with GENERAL-pinning verified (no layout VVL errors).
- **Intra-pass sandwich** (sample → storage-write → sample) now correct, not merely memory-safe — regression pinning the M0 residual-hole closure.
- **Reflection-narrowed barrier** fires/elides correctly (data-race stress on writable members).
- **`BindingResource` widening** back-compat: existing texture-view callers unchanged.
- **Unsafe-mode audit** guard: checked vs unchecked spvasm differs only by compare+select across the new types.
- **Perf pass:** sampled-path performance unregressed by the hetero machinery.
