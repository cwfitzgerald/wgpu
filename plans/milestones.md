# Resource Table (bindless) — milestone roadmap & carry-over ledger

Index for the post-M0 milestone plans. **M0 is done** (unsafe-only Vulkan sampling path landed in 8 waves; see the Progress log in [resource-table.md](resource-table.md):198-208 and the execution notes in [m0-notes.md](m0-notes.md)). This file is the map; each `mN-plan.md` is a standalone brief for one milestone.

## Reading order (future agent)

1. [resource-table.md](resource-table.md) — the design source of truth (decisions D1–D16 :23-38, invariants 1–6 :42-47, architecture :49-137, work breakdown :139-176, open questions :180-184, glossary :186-196). **Do not relitigate settled decisions.**
2. This file — milestone sequence, carry-over ledger, errata, process notes.
3. The specific `mN-plan.md` for the milestone you are starting.
4. [m0-notes.md](m0-notes.md) — landed shapes to build against, the Landmines sections (:125-132, :163-168), the Accepted warts (:116-119), and the Test/verify cheat sheet (:179-189). Read the landmines before writing any test.

Each milestone plan carries a **draft** work-item breakdown. The executing orchestrator re-plans at milestone start against the then-current tree; anchors below are as of 2026-07-10 and drift — symbol names are the stable reference.

## Milestone table

| id  | theme | depends on | status |
|-----|-------|-----------|--------|
| M0  | skeleton (unsafe-only, Vulkan, sampling) | — | **done** (waves 1–8) |
| M1  | checked path (metadata + defaults + hasResource + samplers) | M0 | next |
| M2  | usage states & spec-exact visibility (mask ring, epoch immediate) | M1 | planned |
| M3  | DX12 backend | M1, M2 (per doc order :141, "M1+ sequential") | planned |
| M3m | Metal explicit-sync track (M3a→M3b→M3c) | parallel track (see note) | planned |
| M4  | heterogeneous (storage textures/buffers, D10) | M2 | planned |
| M5  | mid-pass usage changes (suspension-on-demand, D13) | M4 (last) | planned |

Doc dependency note (:141): "M1+ sequential per milestone. Metal track independent after M0." **M3m framing:** M3a (tracker producer tokens + hal fence-edge API + perf-parity gate) has no M1/M2 dependency and is startable after M0 as the doc states; M3c (tables on Metal) needs M1 metadata + M2 visibility. This plan sequences M3m as a parallel track that becomes fully unblocked after M1/M2. See Errata note and the M3m plan.

## Cross-milestone carry-over ledger

`[suggested]` = orchestrator proposal the milestone's planner must confirm with Connor, not a doc-stated fact.

| item | owner | source |
|------|-------|--------|
| `hasResource<T>` (parser + lowering + predicate) | **M1** | doc-stated (:161) |
| `getResource<sampler>` + device-global sampler heap (D7) | **M1** | m0-notes dev.3 (:18), progress log (:201); doc M1 scope line omits it → Errata (b) |
| Metadata buffer (u64 words) + naga **checked** lowering (policy-driven, Inv.4) | **M1** | doc-stated (:161) |
| `K` default tail slots + per-type-class default contents | **M1** | doc-stated (:161); K value is TODO #5471 |
| destroy/remove → metadata zeroing at submit head | **M1** | doc-stated (:161), Inv.5 (:46) |
| Recorded-but-unsubmitted-CB slots not slot-gated | **M1 risk** | m0-notes wart (:118) — metadata-zeroing must not race |
| Error-scope integration for `ResourceTableError::Other(String)` | **M1 [suggested]** | m0-notes 0.11 (:96) — API polish |
| Port a `binding_array` example to tables | **M1 [suggested]** | 0.12 deferred; Connor deferred "for now" in M0 |
| Replace submit-time incompatible-member ERROR with spec-exact hiding | **M2** | doc-stated (:164) |
| Mask ring buffer, epoch immediate, `set_resource_usage`, per-dispatch scope digests | **M2** | doc-stated (:163-164) |
| 0.10 compute barrier becomes **load-bearing** (was redundant in M0) | **M2** | m0-notes wave 7 (:150) |
| Sandwich-escape layout holes — intra-pass (layout-unsafe/memory-safe) | **M2 [suggested]** | residual doc at `TextureTracker::collect_table_incompatible_usages` (track/texture.rs:565-592); per-dispatch machinery gives the precision M0's start∪end folding lacks |
| Sandwich-escape — cross-pass w/ top-level transfer (genuine memory hazard) | **M2 [suggested]** | same; M2 must close or consciously extend the rejection |
| D11 mixed steady-state **GENERAL-pinning mechanism** | **M4** | doc-aligned — no subjects until storage-capable members exist (M4) |
| Retain a strict conflict-erroring mode after M2? | **M2 open question** | doc open Q3 (:182) |
| Render bundles with tables (rejected: `CreateRenderBundleError::ResourceTableUnsupported`) | **unassigned, candidate M2+ [suggested]** | m0-notes 0.8 (:93); needs settled replay visibility |
| DX12 (heap-range tables, sampler-heap reuse, legacy-state barriers, HLSL overlapping ranges) | **M3** | doc-stated (:166-167) |
| Metal M3a/M3b/M3c (producer tokens, fence-edge hal API, untracked allocs, tables/arg-buffer/residency/MSL) | **M3m** | doc-stated (:169-170); M3a needs a hal pass-edge DAG API **design review** (doc open Q1 :180) BEFORE impl |
| Heterogeneous (storage textures then buffers, VK mutable descriptor / DX12 overlapping ranges D10, MSL `may_alias`, `BindingResource` widening, writable dirty-bit reflection-narrowed, unsafe audit, perf) | **M4** | doc-stated (:172-173) |
| Mid-pass `compute_pass.set_resource_usage` via suspension-on-demand (D13) | **M5** | doc-stated (:175-176) |
| `i32` getResource index (M0 is u32-only; proposal says "i32 or u32") | **upstream-feedback backlog, unassigned** | m0-notes dev.4 (:19) |
| Conservative slot-gating granularity (`mark_all_slots_in_use` marks ALL slots) + maintain-only availability cache | **no owner; behavioral contract** | accepted warts (m0-notes :116-118, :141); asserted by `RESOURCE_TABLE_SLOT_IN_USE_THEN_POLL` (lifecycle.rs:158) — any granularity change MUST update that test |

## Errata / clarifications

The design doc is **not** being edited. Record clarifications here.

- **(a) D2 realization = insertion-point splice, not literally-open hal CBs.** D2's "hal CBs are left open … then closed" text (:24) and the "left open at `finish()`" language in the architecture section (:57) are **superseded** by the splice realization: `finish()` force-closes; a `GapMarker { insertion_point }` is recorded on `CommandBufferMutable`; at submit a fresh segment is appended to `baked.encoder` and spliced via `close_and_insert_at`. Template is `DeferredQuerySetResolve`. hal has **no** open-tail/suspend contract. Authority: [m0-notes.md](m0-notes.md) deviation 1 (:15). Later milestones assume the splice model — every new gap position (M5's mid-pass point) is a new GapMarker + splice, never a literal open CB.
- **(b) Samplers / D7 belong to M1** even though the doc's M1 scope line (:161) omits them. m0-notes deviation 3 (:18) and the progress log (:201) assign `getResource<sampler>` + the device-global sampler heap to M1.
- **(c) S-bench is cancelled.** Its subject (open-tail cost) dissolved with the splice realization (a). m0-notes deviation 6 (:21). Do not schedule it.
- **(d) Feature bits live in `FeaturesWGPU` word 0, which is now 100% full.** The doc's "second u64 / features.rs:1490" note (:147) was stale (that word is `FeaturesWebGPU`, reserved for spec-standard features). Any NEW feature bit must grow `bitflags_array!` / land in another word. m0-notes deviation 2 (:16).

## Process notes for future orchestrators

The M0 pipeline that worked — reuse it:

- **research → plan → user checkpoint → parallel execution → interim fresh-eyes review after big core waves → final fresh-eyes review.** M0's interim review (over waves 1–5) caught 2 criticals + 1 medium; the final review (wave 8) caught 1 medium + 1 low. Budget review waves; they paid off.
- **Parallel execution in jj workspaces under `.worktrees/<name>`** (add `.worktrees/` to `.git/info/exclude` first). **Clean them up when done** — `jj workspace forget` + delete the dir as soon as no phase needs them (M0 did this after wave 7).
- **Move e2e GPU tests early** (user preference): M0 pulled the e2e suite forward to wave 6 rather than deferring to the end.
- **Run e2e:** `$env:WGPU_BACKEND="vulkan"; cargo xtask test -E 'binary(wgpu-gpu) and test(resource_table)'`. Never put `-E` after `--`. See the cheat sheet (m0-notes :179-189).
- **`getResource` shaders must reach hal as `ShaderInput::Naga`, never pre-compiled SPIR-V** — the Vulkan target is injected per-pipeline at compile (m0-notes 0.5 landmine :65). This constraint carries through every milestone that touches naga lowering.
- **Read the m0-notes Landmines (:125-132, :163-168) before writing any test.** The Rust-2021 disjoint-closure-capture trap (`let ctx = &ctx;`) alone (m0-notes :140) will otherwise cost an afternoon.
- Match subagent model tier to work difficulty; re-specify `model` on any stall-and-resume.

## The three-to-five most consequential open questions (raise with Connor)

Compiled from the doc's open questions (:180-184) and per-milestone underspecification. Full lists live in each plan.

1. **Mask-ring default size / growth-gating policy** (doc Q2, :181). Blocks M2. "Mask-ring default size / growth-gating policy; per-slot encoding compression if pathologies appear (exactness non-negotiable, representation negotiable)."
2. **Value of `K` and per-type-class default-resource contents** (spec TODO #5471, :183). Blocks M1 — the default tail slots are the OOB/type-mismatch fallback target.
3. **hal pass-edge (DAG) API shape + node identity for top-level transfer chunks** (doc Q1, :180). Gates M3a; needs a design review BEFORE implementation. "Exact hal shape of the pass-edge (DAG) API and node identity for top-level transfer chunks (M3a design review)."
4. **Whether wgpu keeps a strict conflict-erroring mode after M2** (doc Q3, :182): "Whether `wgpu` should expose a strict mode keeping v0's conflict-erroring after M2 (diagnostics value)."
5. **Cross-pass sandwich-escape at M2**: close it with the per-dispatch machinery, or consciously extend the M0 layout-incompatibility rejection to cover it? (It is a genuine memory hazard today; the intra-pass shape is only layout-unsafe.)
