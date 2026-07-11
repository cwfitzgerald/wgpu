# M5 — mid-pass usage changes

Standalone brief; read with [resource-table.md](resource-table.md) + [m0-notes.md](m0-notes.md). See [milestones.md](milestones.md) for the ledger and errata. **Draft** — re-plan at milestone start; anchors drift, symbol names are stable.

## Goal

Commit mid-pass `compute_pass.set_resource_usage` via **suspension-on-demand**: the compute segment ends at the call, an appended tail carries the precise named barrier, and the pass resumes — the visibility effect already rides the M2 masks. This is the last milestone.

**Doc scope (:176, quote):** "Suspension-on-demand at `compute_pass.set_resource_usage`: segment ends at the call; appended tail carries the precise named barrier (visibility already rides masks); Metal = encoder split + blit, no CB split."
**Accept (:176, quote):** "hide→write→set-readonly→sample in one pass, one precise barrier, VVL-clean."

Sequenced **last** (per [milestones.md](milestones.md); doc work-breakdown order ends with M5). Depends on M2's masks (visibility) and — for the Metal path — M3m's fence-edge model.

## Binding decisions & invariants

- **D13** (:35) — mid-pass `compute_pass.set_resource_usage` is committed via **suspension-on-demand**; it exists to place a precise named barrier — its visibility effect rides the M2 masks. This milestone's entire mandate in one line.
- **Errata (a)** — the "suspension" is a new **GapMarker + `close_and_insert_at` splice** at the call site, not a literally-open CB (m0-notes deviation 1 :15). This is gap position (c) from the doc (:58).
- **Invariant 1** (:42) — the mid-pass barrier/metadata write is a GPU-timeline write at the (new) gap.
- **Invariant 3** (:44) — the named barrier is a memory-ordering barrier; layouts stay pinned by the D11 steady state (M4). A missed barrier ⇒ unspecified values (safe), never wrong layout.

## Inherited state (from M0–M4 — pointers, no duplication)

- **Gap position (c) is reserved but unbuilt:** the doc lists mid-pass `set_resource_usage` as gap position (c) (doc :58); M0 built (a) pass-start and (b) between-CB only. M5 adds (c).
- **Splice machinery:** `ResourceTableGap { table, insertion_point }` on `CommandBufferMutable`, two-phase `process_resource_table_gaps` (m0-notes 0.8 :90). M5 records a gap **inside** a compute pass at the `set_resource_usage` call and splices the tail there.
- **M2 masks + `set_resource_usage`:** the encoder-at-boundary and device-timeline forms of `set_resource_usage` landed at M2 (m2-plan work item 6); the visibility effect (hide/show slots per usage state) already rides the mask ring. M5 adds only the **mid-pass** call site + the precise named barrier it places.
- **Compute encoder state:** `compute.rs` pass encoding, `flush_bindings`, the pre-pass transit that M0 generalized into gap (a) (m0-notes 0.8 :90, :177). M5's suspension splits the segment at the call.
- **Metal:** M3m's fence-edge model + gaps-as-blit-encoders (m3m-plan; doc :111 "Metal = encoder split + blit, no CB split"). M5 on Metal is an encoder split, not a `MTLCommandBuffer` split.

## Carry-over items absorbed here

- Mid-pass `set_resource_usage` via suspension-on-demand (D13) — **doc-stated**. This is the sole carry-over; M5 is the smallest milestone.

## Draft breakdown — the executing orchestrator should re-plan at milestone start against the then-current tree

1. **Mid-pass gap position (c).** Record a `ResourceTableGap` at the `compute_pass.set_resource_usage` call's insertion point (end the current segment there); template is gap (a)'s pass-start marker (doc :58, m0-notes :90). *Verify:* the gap lands at the right insertion point; `adjust()` arithmetic still composes with query-resolve splices.
2. **Appended tail = precise named barrier.** At submit, the spliced tail carries exactly the barrier the `set_resource_usage` names (not a conservative one) + any metadata/mask delta. Visibility already handled by M2 masks — this places only the memory barrier. *Verify:* exactly one barrier for the hide→write→set-readonly→sample sequence.
3. **Resume the segment.** After the splice, the compute pass continues on a fresh segment; the epoch immediate + mask addressing survive the split. *Verify:* dispatches after the call still increment E correctly; masks still addressable.
4. **Metal encoder split (needs M3m).** On Metal, suspend = end the compute encoder, append a blit encoder for the barrier/fence edge, resume with a new compute encoder — no CB split (doc :111). *Verify:* Metal e2e parity; fence edge placed correctly.

## Open questions requiring a user decision

- **Composition of the Metal encoder-split with the M3m fence-edge model:** how does the mid-pass encoder split interact with per-table gap fences and producer tokens (does the split introduce a new sync node, or reuse the pass's)? Needs the M3m DAG API settled first.
- **Interaction of mid-pass gaps with the epoch immediate:** does splitting a compute pass mid-flight require re-emitting the epoch immediate on the resumed segment, and does E numbering continue or reset across the split? (Affects mask addressing correctness.)
- **Scope of `set_resource_usage(None)` mid-pass** vs the M2 barrier-elision path: confirm the mid-pass form composes with M2's elision rather than double-counting.

## Risks / landmines

- **Splice composition:** M5 adds a third source of splices (query-resolve, pass-start table gaps, now mid-pass gaps). The `adjust()` insertion-point remapping (m0-notes 0.8 :90) must remain correct across all three; the two-phase ascending-compute / descending-splice ordering is easy to break.
- **Epoch immediate across the split:** if the resumed segment mis-numbers dispatch epochs, the mask ring reads the wrong bit — a silent visibility bug. Cover with a dedicated test.
- **Precise vs conservative barrier:** the whole point of D13 is *one precise named barrier*; falling back to a conservative compute→compute barrier would defeat the milestone (still safe, but not the deliverable). The Accept criterion ("one precise barrier") pins this.
- **Metal CB-split temptation:** the doc is explicit — encoder split + blit, **no CB split** (doc :111). A CB split would break the single-submission model.

## Verification strategy

Harness/gates per m0-notes cheat sheet (:179-189). New axis (the Accept criterion): **hide→write→set-readonly→sample in one compute pass → exactly one precise barrier, VVL-clean**, with byte-exact readback. Add:
- **Epoch-across-split** test: dispatches before and after the mid-pass call read the correct mask bits.
- **Barrier-count assertion:** exactly one barrier emitted for the canonical sequence (not the conservative fallback).
- **Metal parity** (needs M3m + macOS runner): the same sequence via encoder split + blit, no CB split.
