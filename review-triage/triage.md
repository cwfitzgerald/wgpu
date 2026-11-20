# wgpu PR triage

Last update: 2026-11-20. All dates are UTC, as GitHub gives them.

This is a pre-review pass over the open PRs in gfx-rs/wgpu: 113 open PRs (64 ready for review, 49 drafts), and the 3 issues assigned to [@cwfitzgerald](https://github.com/cwfitzgerald). Merge queue PRs from Mergify are not counted. PRs that a member approved and sent to the merge queue are not in this file, merged or not. It is read only. No code was built or tested, and nothing was posted to GitHub. CI status comes from GitHub check runs. Findings that are reasoned from code, not run, are marked as such in the details.

This file shows the current state only. It is updated in place; `UPDATING.md` gives the procedure.

**Order of this file.** The summary is in priority order for the release in about one week:

1. Ready PRs that are assigned to you, oldest wait first.
2. Ready PRs by other people that are not assigned to you, oldest wait first.
3. All other items: your own PRs, drafts, issues.

In each priority, the PRs that wait on a maintainer come first, because a maintainer can move them. The PRs that wait on the author come after them. "Last maintainer reply" is the date of the last comment or review by a member or collaborator who is not the author. An assignment is not a reply. "never" means that no member replied after the PR was opened; the wait then starts at "Opened".

## Attention first

- **Release sweep: 22 ready PRs need one approval, one review or small fixes.** "Land before the release" below lists them. 6 of them were opened before 2026-11-01. Most wait on an approval from a member with write access.
- **Small fixes that an agent can make do not hold a PR back.** Release rows that start with "Agent fix" need only a changelog entry, a test, a rebase or a small code fix, then one review or approval.
- **The v30 branch cannot pass its MSRV Minimal Versions job.** v30's `Cargo.toml` patches `libtest-mimic` to your fork at rev `9979b3c`, which no branch of the fork holds, so `cargo` cannot fetch it. Push a branch that holds `9979b3c`, or drop the patch on v30. This blocks [#10985](https://github.com/gfx-rs/wgpu/issues/10985) and any v30.0.3.
- **Approvals that GitHub does not count.** [#10980](https://github.com/gfx-rs/wgpu/issues/10980), [#10901](https://github.com/gfx-rs/wgpu/issues/10901), [#10891](https://github.com/gfx-rs/wgpu/issues/10891), [#10821](https://github.com/gfx-rs/wgpu/issues/10821) and [#10837](https://github.com/gfx-rs/wgpu/issues/10837) have an approval from inner-daemons, who has read permission, and show `REVIEW_REQUIRED`. Each needs an approval from a member with write access.
- **[#10840](https://github.com/gfx-rs/wgpu/issues/10840) has an answer to its version question:** an Arm engineer says that driver r46 (most likely g21 on Rockchip names) fixes the timeline semaphore bug. The PR can limit the workaround to older drivers. [#10992](https://github.com/gfx-rs/wgpu/issues/10992) does the same for Imagination drivers, without a workaround bit.
- **Agent-written PRs opened on or after 2026-11-13:** [#10981](https://github.com/gfx-rs/wgpu/issues/10981) (code and description), [#10979](https://github.com/gfx-rs/wgpu/issues/10979), [#10990](https://github.com/gfx-rs/wgpu/issues/10990) and [#10991](https://github.com/gfx-rs/wgpu/issues/10991) (matthargett), [#11000](https://github.com/gfx-rs/wgpu/issues/11000) (code and description). See "Policy: agent-written PRs".

## Land before the release

These ready PRs have the verdict "Land as is", or need one small step, or need only small fixes that an agent can make ("Agent fix"). Sorted by Opened.

| PR | Title | Opened | Step |
|---|---|---|---|
| [#10212](https://github.com/gfx-rs/wgpu/issues/10212) (yours) | Pass first-use tracking | 2026-08-31 | Rebase (conflict), then andyleiserson review. |
| [#10600](https://github.com/gfx-rs/wgpu/issues/10600) | spv-out: wrapped function preamble helper | 2026-10-12 | jimblandy review. Refactor, CI green. Lands before [#10697](https://github.com/gfx-rs/wgpu/issues/10697). |
| [#10609](https://github.com/gfx-rs/wgpu/issues/10609) | naga GLSL matrix interfaces | 2026-10-13 | Agent fix: changelog attribution, and limit the sampling change to the GLSL path or write it in the description. Then a review (inner-daemons is requested). |
| [#10661](https://github.com/gfx-rs/wgpu/issues/10661) | naga: process_overrides panic on anonymous overrides | 2026-10-18 | Agent fix: add the requested snapshot test, rebase. Then review (about 50 lines). |
| [#10694](https://github.com/gfx-rs/wgpu/issues/10694) | wgsl-in: rayQueryGenerateIntersection scope fix | 2026-10-21 | Rerun the SPIR-V validate job (curl failure), review. |
| [#10784](https://github.com/gfx-rs/wgpu/issues/10784) | CTS: non-const `modf` f32 tests | 2026-10-30 | Review. teoxoy is assigned. |
| [#10837](https://github.com/gfx-rs/wgpu/issues/10837) | vk: pass-begin timestamp at TOP_OF_PIPE | 2026-11-04 | Agent fix: rebase (conflict). Then you approve and queue. |
| [#10866](https://github.com/gfx-rs/wgpu/issues/10866) | GLES: detach renderbuffers on framebuffer reuse | 2026-11-06 | Agent fix: changelog entry. Ask for the "LLM Use?" section. Then review. The red Mac job is a Metal poll timeout. |
| [#10869](https://github.com/gfx-rs/wgpu/issues/10869) (yours) | KosmicKrisp test expectations | 2026-11-06 | Approved by andyleiserson. Queue it. |
| [#10873](https://github.com/gfx-rs/wgpu/issues/10873) | spv-out: external texture plumbing | 2026-11-06 | Agent fix: rebase (small `writer.rs` conflict). Then a review; no reviewer is assigned. |
| [#10901](https://github.com/gfx-rs/wgpu/issues/10901) | naga cooperative matrix store order | 2026-11-10 | Agent fix: make an abstract stride `u32` in `coopLoad`/`coopStore` lowering, add a test with a literal stride. Then member approval (inner-daemons approved on 2026-11-20). Land it before [#10596](https://github.com/gfx-rs/wgpu/issues/10596) and [#10414](https://github.com/gfx-rs/wgpu/issues/10414). |
| [#10903](https://github.com/gfx-rs/wgpu/issues/10903) (yours) | Immediates fix (panic) | 2026-11-10 | Rebase (conflict), then andyleiserson review. |
| [#10957](https://github.com/gfx-rs/wgpu/issues/10957) | naga: bitcast in the constant evaluator | 2026-11-14 | Agent fix: address andyleiserson's comments of 2026-11-20 (`flatten_compose`, error contents, mixed-width compose test, CTS tests), NaN/infinity error tests, changelog entry. Then your review. |
| [#10959](https://github.com/gfx-rs/wgpu/issues/10959) (yours) | Fix indirect draw validation panic on noop | 2026-11-15 | Approved by andyleiserson. Queue it. |
| [#10961](https://github.com/gfx-rs/wgpu/issues/10961) | MSL: bitcast of a packed vec3 field | 2026-11-15 | The author adds one `typos.toml` line, then review. |
| [#10966](https://github.com/gfx-rs/wgpu/issues/10966) | Doc on `Device::from_core` | 2026-11-16 | The author fixes the title and description, then approve and squash. |
| [#10980](https://github.com/gfx-rs/wgpu/issues/10980) | wgsl-in: `continuing` block shadowing | 2026-11-18 | Member approval (inner-daemons approved). |
| [#10982](https://github.com/gfx-rs/wgpu/issues/10982) | naga: `wgpu_f64` enable extension | 2026-11-18 | Agent fix: breaking-change changelog entry, `SHADER_F64` docs line, negative tests. Answer the author's 4 questions. Then review. |
| [#10985](https://github.com/gfx-rs/wgpu/issues/10985) | v30: Vulkan crash with `poolSizeCount == 0` (backport) | 2026-11-18 | Decide if you want a v30.0.3; then approve. Same fix as trunk [#10124](https://github.com/gfx-rs/wgpu/issues/10124). The red MSRV job is the `libtest-mimic` fork problem (see Attention). |
| [#10992](https://github.com/gfx-rs/wgpu/issues/10992) | Vulkan: no timeline semaphores on Imagination | 2026-11-19 | Member approval. Conflicts with [#10840](https://github.com/gfx-rs/wgpu/issues/10840) in `adapter.rs`; the second to land resolves it. |
| [#11000](https://github.com/gfx-rs/wgpu/issues/11000) | Vulkan: acquire timeout returns `Timeout` | 2026-11-19 | Agent-policy decision (both agent boxes checked), then member approval. |
| [#11019](https://github.com/gfx-rs/wgpu/issues/11019) (yours) | wasm runner page timeout to 60 s | 2026-11-20 | Wait for CI, then one approval. |

---

# Priority 1: ready PRs assigned to you (25)

### Waiting on a maintainer (12)

| PR | Title | Opened | Last maintainer reply | Verdict | Next step |
|---|---|---|---|---|---|
| [#10506](https://github.com/gfx-rs/wgpu/issues/10506) | Metal: one command buffer for many passes **[Mac CI cancelled]** | 2026-09-30 | never | Needs a bit of work | Your design sign-off. **High, confirmed:** with indirect validation off, ICB setup for multi-draws is encoded after the pass that uses it. Has conflicts. |
| [#10438](https://github.com/gfx-rs/wgpu/issues/10438) | create_*_pipeline_async | 2026-09-23 | 2026-10-04 (sagudev approved) | Needs a bit of work | Sign off on the public API shape. The author asked you for review on 2026-10-24. Has conflicts. |
| [#10591](https://github.com/gfx-rs/wgpu/issues/10591) | core: SmallVec for single-region texture copies | 2026-10-12 | 2026-10-14 | Needs a bit of work | Decide if you want it. inner-daemons said "very low importance". |
| [#10609](https://github.com/gfx-rs/wgpu/issues/10609) | naga GLSL matrix interfaces | 2026-10-13 | 2026-10-14 | Needs a bit of work | Agent fix: changelog attribution, and limit the sampling change to the GLSL path or write it in the description. Then a review (inner-daemons is requested). |
| [#10778](https://github.com/gfx-rs/wgpu/issues/10778) | Make `wayland-sys` `dlopen` togglable | 2026-10-29 | never | Needs a lot of work | Decide if compile-time libwayland linking is wanted. As written it breaks macOS, iOS and Android builds (CI red, 15 jobs). If no, close. |
| [#10840](https://github.com/gfx-rs/wgpu/issues/10840) | Vulkan: no timeline semaphores on Arm Mali | 2026-11-05 | never | Needs a bit of work | Ask the author to limit the flag to drivers before r46, as the Arm engineer's comment of 2026-11-19 allows. Then review. The workaround bit `1 << 3` is taken on trunk. Has conflicts. |
| [#10862](https://github.com/gfx-rs/wgpu/issues/10862) | Pool command-encoder and pass trackers | 2026-11-06 | 2026-11-09 | Needs a bit of work | Decide the agent-disclosure question (both boxes checked), then inner-daemons reviews. rowanG077's `ResourceMetadata::clear` change is a performance follow-up. Safer after the release. |
| [#10837](https://github.com/gfx-rs/wgpu/issues/10837) | vk: pass-begin timestamp at TOP_OF_PIPE | 2026-11-04 | 2026-11-12 (inner-daemons approved) | Land as is | Resolve the conflict (agent fix), approve and queue. |
| [#10890](https://github.com/gfx-rs/wgpu/issues/10890) | deno: expose `chromium-experimental-multi-draw-indirect` | 2026-11-09 | 2026-11-18 | Needs a bit of work | Decide. **High:** wgpu-core does not validate count-draw arguments, and this makes them reachable from JS on Vulkan and DX12. inner-daemons objects to using Chrome's tests for an API that wgpu does not match, and suggests splitting out the new validation. Recommended: split and land the validation; hold the feature. |
| [#10979](https://github.com/gfx-rs/wgpu/issues/10979) | Quest 1 (Adreno 540) debug builds, GPU tests, CTS | 2026-11-18 | 2026-11-18 | Needs a bit of work | inner-daemons must say if the Vulkan 1.2 gate answers the OOM concern (the tests report unsupported, not a pass). A member reviews the `deno_webgpu` line. Agent-policy decision. Can land before the release. |
| [#10981](https://github.com/gfx-rs/wgpu/issues/10981) | GLES: prefer an 8-bit EGL config | 2026-11-18 | never | Needs a bit of work | First review and agent policy. Medium (from code): asking for 8-bit channel sizes changes the EGL sort, so 10-bit configs can come first on every EGL platform. |
| [#10982](https://github.com/gfx-rs/wgpu/issues/10982) | naga: `wgpu_f64` enable extension | 2026-11-18 | never | Needs a bit of work | Agent fix: breaking-change changelog entry, `SHADER_F64` docs line, negative tests. Answer the author's 4 questions. Then review. |

### Waiting on the author (13)

| PR | Title | Opened | Last maintainer reply | Verdict | Next step |
|---|---|---|---|---|---|
| [#10098](https://github.com/gfx-rs/wgpu/issues/10098) | Immediates on WebGPU, remove `Features::IMMEDIATES` | 2026-08-18 | 2026-09-18 (you) | Needs a lot of work | Inactive since 2026-09-18. Removes the feature that you asked to keep. Close. |
| [#10414](https://github.com/gfx-rs/wgpu/issues/10414) | naga non-square cooperative matrix tiles | 2026-09-21 | 2026-09-24 | Needs a bit of work | Rebase (conflicts; one with [#10901](https://github.com/gfx-rs/wgpu/issues/10901) in `naga/src/valid/expression.rs`). Shape validation ignores the scalar type. |
| [#10800](https://github.com/gfx-rs/wgpu/issues/10800) | Metal: remove the `wg_memory_sizes` logic | 2026-11-01 | never | Needs a bit of work | inner-daemons answers matthargett's A10X crash report. CI green. |
| [#10838](https://github.com/gfx-rs/wgpu/issues/10838) | Sparse TLAS instance indices | 2026-11-04 | never | Needs a bit of work | CI red: the new test segfaults lavapipe (a Mesa bug); skip or expect it. Medium: the empty BLAS reads the zero buffer, which has no BLAS-input usage and gets no barrier. Has conflicts. |
| [#9679](https://github.com/gfx-rs/wgpu/issues/9679) | Metal ICBs for multi-draw indirect count | 2026-06-15 | 2026-11-05 (you, changes requested) | Needs a bit of work | Rebase. Core does not validate count-draw arguments. The [#10506](https://github.com/gfx-rs/wgpu/issues/10506) ordering bug also affects it. |
| [#10821](https://github.com/gfx-rs/wgpu/issues/10821) | Release the surface texture on device loss | 2026-11-03 | 2026-11-09 | Needs a bit of work | sagudev's question about [#10296](https://github.com/gfx-rs/wgpu/issues/10296) has no answer since 2026-11-03. Trunk already releases the texture. If no answer, close. |
| [#9295](https://github.com/gfx-rs/wgpu/issues/9295) | naga/spv layoutless workgroup types | 2026-03-24 | 2026-11-12 | Needs a lot of work | kvark must say if he continues. Two confirmed bugs (pointer type order, mesh output wrapper). [#10928](https://github.com/gfx-rs/wgpu/issues/10928) (the agent-written rework) is closed, so its fixes are not available as a PR. |
| [#10596](https://github.com/gfx-rs/wgpu/issues/10596) | BF16 + cooperative matrix | 2026-10-12 | 2026-11-13 (inner-daemons, changes requested) | Needs a bit of work | kvark answers inner-daemons (does `subgroup_matrix` replace cooperative matrices?). Has conflicts. |
| [#10860](https://github.com/gfx-rs/wgpu/issues/10860) | Metal watchOS MSL profiles | 2026-11-05 | 2026-11-16 | Needs a bit of work | Rebase onto [#10927](https://github.com/gfx-rs/wgpu/issues/10927) (move the 32-bit watchOS skip into `render_icb_probe_allowed`), add the promised version comments. Indirect dispatch and base vertex are enabled with evidence for indirect draws only. No CI on the head. |
| [#10966](https://github.com/gfx-rs/wgpu/issues/10966) | Doc on `Device::from_core` | 2026-11-16 | 2026-11-16 | Needs a bit of work | Fix the title and description: the tests were removed, and the diff is a 5-line doc. Then approve and squash. |
| [#10880](https://github.com/gfx-rs/wgpu/issues/10880) | Vulkan: trim command pools over a window | 2026-11-08 | 2026-11-18 | Needs a bit of work | The trim uses a window of resets. The author promised a `VulkanBackendOptions` window field on 2026-11-18 (not pushed). Decide if you accept a new public backend option. matthargett's changes-requested review is open. |
| [#10901](https://github.com/gfx-rs/wgpu/issues/10901) | naga cooperative matrix store order | 2026-11-10 | 2026-11-20 (inner-daemons approved) | Needs a bit of work | Agent fix: make an abstract stride `u32` in `coopLoad`/`coopStore` lowering, add a test with a literal stride. Then member approval (inner-daemons approved on 2026-11-20). Land it before [#10596](https://github.com/gfx-rs/wgpu/issues/10596) and [#10414](https://github.com/gfx-rs/wgpu/issues/10414). |
| [#10957](https://github.com/gfx-rs/wgpu/issues/10957) | naga: bitcast in the constant evaluator | 2026-11-14 | 2026-11-20 (andyleiserson) | Needs a bit of work | Agent fix: address andyleiserson's comments of 2026-11-20 (`flatten_compose`, error contents, mixed-width compose test, CTS tests), NaN/infinity error tests, changelog entry. Then your review. |

---

# Priority 2: ready PRs by other people, not assigned to you (33)

### Waiting on a maintainer (23)

| PR | Title | Opened | Last maintainer reply | Assignee | Verdict | Next step |
|---|---|---|---|---|---|---|
| [#9876](https://github.com/gfx-rs/wgpu/issues/9876) | Metal: query subgroup size | 2026-07-15 | 2026-10-06 | jimblandy | Needs a bit of work | Decide if `ComputePipeline::get_subgroup_size` is wanted as non-WebGPU API. Has conflicts. |
| [#10600](https://github.com/gfx-rs/wgpu/issues/10600) | spv-out: wrapped function preamble helper | 2026-10-12 | 2026-10-13 | jimblandy | Land as is | Review. |
| [#10694](https://github.com/gfx-rs/wgpu/issues/10694) | wgsl-in: rayQueryGenerateIntersection scope fix | 2026-10-21 | never | jimblandy | Land as is | Rerun the SPIR-V validate job, review. |
| [#10731](https://github.com/gfx-rs/wgpu/issues/10731), [#10743](https://github.com/gfx-rs/wgpu/issues/10743) | Metal: workarounds for OCLP-only Intel compiler bugs | 2026-10-26 | never | andyleiserson | Needs a lot of work | Decide if OCLP-only workarounds are supported. If no, close both. If yes, fold [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731). |
| [#10750](https://github.com/gfx-rs/wgpu/issues/10750) | VK_EXT_hdr_metadata | 2026-10-27 | never | jimblandy | Needs a lot of work | Fully agent-written; the API returns a bare bool. Close under the agentic-PR policy. |
| [#10697](https://github.com/gfx-rs/wgpu/issues/10697) | `rayQueryGetCurrentTMax` builtin | 2026-10-22 | 2026-10-28 | jimblandy | Needs a bit of work | Decide if the non-standard builtin is accepted. After [#10600](https://github.com/gfx-rs/wgpu/issues/10600) lands, it must adapt to the new helper signature. |
| [#10775](https://github.com/gfx-rs/wgpu/issues/10775) | `Extensions<T>` for backend options | 2026-10-28 | never | jimblandy, teoxoy | Needs a bit of work | Decide the extension pattern. andyleiserson asked on 2026-11-03. Rebase only since the last review. |
| [#10784](https://github.com/gfx-rs/wgpu/issues/10784) | CTS: non-const `modf` f32 tests | 2026-10-30 | never | teoxoy | Land as is | Review. |
| [#10812](https://github.com/gfx-rs/wgpu/issues/10812) | Child textures in `Arc` | 2026-11-02 | never | teoxoy | Needs a lot of work | teoxoy reviews. **High:** child and parent textures share clear views, and every `Texture::drop` destroys them. Has conflicts. |
| [#10839](https://github.com/gfx-rs/wgpu/issues/10839) | Render bundle resource lifetimes | 2026-11-05 | never | teoxoy | Needs a bit of work | inner-daemons review (requested on 2026-11-09). The test literal may need the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field (from code). |
| [#10866](https://github.com/gfx-rs/wgpu/issues/10866) | GLES: detach renderbuffers on framebuffer reuse | 2026-11-06 | never | teoxoy | Needs a bit of work | Agent fix: changelog entry. Ask for the "LLM Use?" section. Then review. The red Mac job is a Metal poll timeout. |
| [#10873](https://github.com/gfx-rs/wgpu/issues/10873) | spv-out: external texture plumbing | 2026-11-06 | never | jimblandy | Needs a bit of work | Agent fix: rebase (small `writer.rs` conflict). Then a review; no reviewer is assigned. |
| [#10891](https://github.com/gfx-rs/wgpu/issues/10891) | Enforce 2047 elements in array constructors | 2026-11-09 | 2026-11-09 (inner-daemons approved) | teoxoy, inner-daemons | Needs a bit of work | Decide: the WGSL spec gives 2047 as a minimum to support, not a limit. The change rejects native shaders with 2048 or more elements. |
| [#10583](https://github.com/gfx-rs/wgpu/issues/10583) | msl-out: dedup array aliases | 2026-10-11 | 2026-11-10 | jimblandy, inner-daemons | Needs a bit of work | Choose the approach (kpreid suggests a shared `naga::proc` helper). Nested aliases are missed. |
| [#10725](https://github.com/gfx-rs/wgpu/issues/10725) | Metal: MSL version probing | 2026-10-26 | 2026-11-16 | andyleiserson | Needs a lot of work | Decide probing at the maintainers meeting of 2026-11-18. andyleiserson proposed opt-in probing (OCLP plist or an environment variable). Conflicts in four files. |
| [#10980](https://github.com/gfx-rs/wgpu/issues/10980) | wgsl-in: `continuing` block shadowing | 2026-11-18 | 2026-11-18 (inner-daemons approved) | teoxoy | Land as is | Member approval. |
| [#10985](https://github.com/gfx-rs/wgpu/issues/10985) | v30: Vulkan crash with `poolSizeCount == 0` (backport) | 2026-11-18 | never | none | Land as is | Decide on a v30.0.3, approve. |
| [#10991](https://github.com/gfx-rs/wgpu/issues/10991) | Vulkan: Adreno 5xx `OpArrayLength` and indirect dispatch workarounds | 2026-11-18 | never | none | Needs a bit of work | inner-daemons review (requested); naga and hal maintainers accept the new push-constant buffer-size path. CHANGELOG conflict. Agent-written. |
| [#10831](https://github.com/gfx-rs/wgpu/issues/10831) | Fix truncated trace playback | 2026-11-03 | 2026-11-19 (andyleiserson) | andyleiserson | Needs a bit of work | Agree or disagree with andyleiserson's record-separator proposal (keeps traces pretty-printed). Opened by an agent account; ask the named human to own it. Has conflicts. |
| [#10992](https://github.com/gfx-rs/wgpu/issues/10992) | Vulkan: no timeline semaphores on Imagination | 2026-11-19 | never | none | Land as is | Member approval. |
| [#11000](https://github.com/gfx-rs/wgpu/issues/11000) | Vulkan: acquire timeout returns `Timeout` | 2026-11-19 | never | none | Land as is | Agent-policy decision, then member approval. Low: the fence wait in `release_resources` has no bound. |
| [#10990](https://github.com/gfx-rs/wgpu/issues/10990) | spv-out: workgroup zero-init with distributed loops | 2026-11-18 | 2026-11-20 | none | Needs a bit of work | Decide: inner-daemons asks to wait for the shared zero-init pass ([#10456](https://github.com/gfx-rs/wgpu/issues/10456)). Generated SPIR-V is correct from code; CI never runs the new path. Agent-written. |

### Waiting on the author (10)

| PR | Title | Opened | Last maintainer reply | Next step |
|---|---|---|---|---|
| [#9645](https://github.com/gfx-rs/wgpu/issues/9645) | Metal AS build sync | 2026-06-04 | 2026-06-23 | Vecvec asked for a race fix on 2026-06-23, no answer. Land this before a split [#10292](https://github.com/gfx-rs/wgpu/issues/10292). Ping once, then close. |
| [#10155](https://github.com/gfx-rs/wgpu/issues/10155) | Reuse the AS build scratch buffer | 2026-08-23 | 2026-10-06 | Silent since 2026-09-03. Close. |
| [#10629](https://github.com/gfx-rs/wgpu/issues/10629) | Vulkan multi-planar DMA-buf import | 2026-10-15 | never | No tests; changelog `#TODO`; jakobkukla notes a `pub` item that is not reachable. Ask if it is ready for review. |
| [#9940](https://github.com/gfx-rs/wgpu/issues/9940) | naga: more vector flattening | 2026-07-24 | 2026-10-21 (jimblandy, changes requested) | andyleiserson fixes the exponential SPIR-V constant writer (**high, DoS**) and the matrix-column const-eval panic. |
| [#10661](https://github.com/gfx-rs/wgpu/issues/10661) | naga: process_overrides panic on anonymous overrides | 2026-10-18 | 2026-10-23 | Agent fix: add the requested snapshot test, rebase. Then review (about 50 lines). |
| [#10515](https://github.com/gfx-rs/wgpu/issues/10515) | Emscripten via `web-sys` | 2026-10-02 | 2026-10-27 (changes requested) | Silent since 2026-10-09. CI red from the PR. Close; DouglasDwyer has a prototype. |
| [#10673](https://github.com/gfx-rs/wgpu/issues/10673) | Metal: threadgroup memory for passthrough shaders | 2026-10-20 | 2026-11-01 | Superseded by [#10800](https://github.com/gfx-rs/wgpu/issues/10800). Close after [#10800](https://github.com/gfx-rs/wgpu/issues/10800) merges. |
| [#10875](https://github.com/gfx-rs/wgpu/issues/10875) | Metal: lose the device on command buffer failure | 2026-11-07 | 2026-11-13 | Answer andyleiserson's questions ([#9579](https://github.com/gfx-rs/wgpu/issues/9579), [#9278](https://github.com/gfx-rs/wgpu/issues/9278), split). Medium: the iOS background error also loses the device. Has conflicts. |
| [#10676](https://github.com/gfx-rs/wgpu/issues/10676) | naga: runtime check for out-of-bounds `let` indices | 2026-10-20 | 2026-11-14 | The author is silent since 2026-10-28. Medium: with an override in the module, valid WGSL can fail pipeline creation. |
| [#10961](https://github.com/gfx-rs/wgpu/issues/10961) | MSL: bitcast of a packed vec3 field | 2026-11-15 | never | The fix is correct. CI red from the PR: `typos` reads `@IsseW` as a misspelling; add the name to `typos.toml`. Then review. |

---

# Priority 3: everything else

## Your own PRs

| PR | State | Next step |
|---|---|---|
| [#11019](https://github.com/gfx-rs/wgpu/issues/11019) | wasm runner page timeout from 10 s to 60 s, for the `clear_buffer_range_respected` flake. CI pending | Get a review. Small CI change; can land before the release. |
| [#10959](https://github.com/gfx-rs/wgpu/issues/10959) | Indirect draw validation panic on noop. Approved by andyleiserson, CI green | Queue it. His question about where the test lives has no reply. |
| [#10212](https://github.com/gfx-rs/wgpu/issues/10212) | Pass first-use tracking. CI green with the D2Array test. Has conflicts | andyleiserson (assigned on 2026-11-18) reviews. Can land before the release. |
| [#10903](https://github.com/gfx-rs/wgpu/issues/10903) | Immediates fix ([#10902](https://github.com/gfx-rs/wgpu/issues/10902)). CI green. Has conflicts | andyleiserson (assigned on 2026-11-18) reviews. A panic fix; can land before the release. |
| [#8388](https://github.com/gfx-rs/wgpu/issues/8388) | DXGI swapchain on Vulkan. CI green. The `pending_present_count` fixup is correct from code and matches [#10511](https://github.com/gfx-rs/wgpu/issues/10511). 3 of 46 threads are unresolved | andyleiserson re-reviews (his only review is changes-requested from 2026-10-07; no pending request shows for him). Squash the fixups after approval. |
| [#10869](https://github.com/gfx-rs/wgpu/issues/10869) | KosmicKrisp test expectations. Approved by andyleiserson, CI green | Queue it. Tick the self-review box. |
| [#10983](https://github.com/gfx-rs/wgpu/issues/10983) (draft) | Changelog fragment tooling. CI green | Mark ready; inner-daemons is requested. Can land before v31. |
| [#10984](https://github.com/gfx-rs/wgpu/issues/10984) (draft) | Switch to changelog fragments after v31. Mac CI red from a Metal poll timeout (infrastructure) | After v31 and [#10983](https://github.com/gfx-rs/wgpu/issues/10983). |
| [#10354](https://github.com/gfx-rs/wgpu/issues/10354) (draft) | `debug_printf_output` times out on Mac aarch64 Metal | Skip or expect the test on Metal. Waits on [#9389](https://github.com/gfx-rs/wgpu/issues/9389). |
| [#9794](https://github.com/gfx-rs/wgpu/issues/9794), [#9853](https://github.com/gfx-rs/wgpu/issues/9853), [#10396](https://github.com/gfx-rs/wgpu/issues/10396) (drafts) | Stalled | [#10396](https://github.com/gfx-rs/wgpu/issues/10396) has the empty template description. |

## Drafts assigned to you (3)

| PR | State | Recommended action |
|---|---|---|
| [#10016](https://github.com/gfx-rs/wgpu/issues/10016) | Stalled since 2026-09-27 | Decide whether the Vulkan memory model is split out (inner-daemons wants it split, kvark does not). |
| [#9053](https://github.com/gfx-rs/wgpu/issues/9053) | Not close | None of your 12 findings from 2026-08-14 is addressed. |
| [#8653](https://github.com/gfx-rs/wgpu/issues/8653) | Not close | Not split as asked. Park it, and decide whether you want this suite in addition to the CTS. |

## Other drafts (40)

**Close candidates (7):** [#6584](https://github.com/gfx-rs/wgpu/issues/6584) (empty `todo!()` stub), [#9897](https://github.com/gfx-rs/wgpu/issues/9897) (DO NOT MERGE repro for a fixed issue), [#9128](https://github.com/gfx-rs/wgpu/issues/9128) (jimblandy does not need it), [#9659](https://github.com/gfx-rs/wgpu/issues/9659) (superseded by [#9679](https://github.com/gfx-rs/wgpu/issues/9679), close once [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands), [#10114](https://github.com/gfx-rs/wgpu/issues/10114) (stale experiment), [#9427](https://github.com/gfx-rs/wgpu/issues/9427) (ping or close), [#9523](https://github.com/gfx-rs/wgpu/issues/9523) (pinged twice, ping once more then close).

**Close to ready; ask the author to undraft (5):** [#9512](https://github.com/gfx-rs/wgpu/issues/9512) (approved by andyleiserson in May; has a conflict), [#9909](https://github.com/gfx-rs/wgpu/issues/9909) (two approvals; CI red from July; needs a rebase), [#10409](https://github.com/gfx-rs/wgpu/issues/10409) (naga compaction reorder; decide if the about 70% slowdown when it reorders is acceptable), [#10790](https://github.com/gfx-rs/wgpu/issues/10790) (dx12 driver-reported size; decide if the `texture_from_raw` signature change is acceptable), [#10811](https://github.com/gfx-rs/wgpu/issues/10811) (msl mesh clip distances; **CI shows a real failure**; also a conflict).

**Need your decision (9):**
- [#8480](https://github.com/gfx-rs/wgpu/issues/8480): is `set_web_canvas_size` the accepted API?
- [#8849](https://github.com/gfx-rs/wgpu/issues/8849): do you want an `AnyError`-style wrapper?
- [#8954](https://github.com/gfx-rs/wgpu/issues/8954): which Metal import fix? kvark was asked and never answered.
- [#9353](https://github.com/gfx-rs/wgpu/issues/9353): opt-in flag for explicit passthrough binding indices, or wait for precompiled shaders? 5andr0 hit [#9326](https://github.com/gfx-rs/wgpu/issues/9326) and offered an agent-written redraft on 2026-11-15.
- [#9809](https://github.com/gfx-rs/wgpu/issues/9809): `WriteOnly` volatile writes without `const fn`?
- [#9888](https://github.com/gfx-rs/wgpu/issues/9888): policy on repro examples that CI cannot run
- [#9975](https://github.com/gfx-rs/wgpu/issues/9975): does the CTS time budget allow it?
- [#10355](https://github.com/gfx-rs/wgpu/issues/10355): an inline pass in naga (stacked on [#10409](https://github.com/gfx-rs/wgpu/issues/10409))
- [#10852](https://github.com/gfx-rs/wgpu/issues/10852): accept a new download action and Mergify retry change, or ask for a smaller PR?

**Ping the author (10):** [#9023](https://github.com/gfx-rs/wgpu/issues/9023), [#9380](https://github.com/gfx-rs/wgpu/issues/9380), [#9485](https://github.com/gfx-rs/wgpu/issues/9485), [#9894](https://github.com/gfx-rs/wgpu/issues/9894), [#10008](https://github.com/gfx-rs/wgpu/issues/10008), [#10282](https://github.com/gfx-rs/wgpu/issues/10282), [#10804](https://github.com/gfx-rs/wgpu/issues/10804), [#10288](https://github.com/gfx-rs/wgpu/issues/10288), [#10289](https://github.com/gfx-rs/wgpu/issues/10289), [#10774](https://github.com/gfx-rs/wgpu/issues/10774).

**Metal ray tracing stack:** [#10292](https://github.com/gfx-rs/wgpu/issues/10292) is the base, then [#10287](https://github.com/gfx-rs/wgpu/issues/10287), [#10288](https://github.com/gfx-rs/wgpu/issues/10288) and [#10289](https://github.com/gfx-rs/wgpu/issues/10289). None of the four has workflow runs, and all four have conflicts. The author last pushed on 2026-09-09. [#10292](https://github.com/gfx-rs/wgpu/issues/10292) should be split (see [#9645](https://github.com/gfx-rs/wgpu/issues/9645)).

**Nothing to do (7):** [#9843](https://github.com/gfx-rs/wgpu/issues/9843), [#9863](https://github.com/gfx-rs/wgpu/issues/9863) (blocked upstream), [#10304](https://github.com/gfx-rs/wgpu/issues/10304) (author splitting it), [#10403](https://github.com/gfx-rs/wgpu/issues/10403) (waits on a glow release), [#10783](https://github.com/gfx-rs/wgpu/issues/10783) (waits on [gpuweb/cts#4723](https://github.com/gpuweb/cts/issues/4723); reduced to a one-line expectation change on 2026-11-18), [#10803](https://github.com/gfx-rs/wgpu/issues/10803), [#10806](https://github.com/gfx-rs/wgpu/issues/10806).

## Issues assigned to you (3)

| Issue | Recommended action |
|---|---|
| [#10622](https://github.com/gfx-rs/wgpu/issues/10622) Encourage PR rationales | Ask kpreid to send the PR, and link CONTRIBUTING "designing new features". |
| [#9125](https://github.com/gfx-rs/wgpu/issues/9125) Actions job limits | GitHub raised the limit to 40 on 2026-09-17. Reply with the result, then keep the issue open for 60 plus cache storage, or close it as partly resolved. |
| [#8354](https://github.com/gfx-rs/wgpu/issues/8354) DXGI swapchain for Vulkan | Keep it open. Tick [#8356](https://github.com/gfx-rs/wgpu/issues/8356) (merged). [#8388](https://github.com/gfx-rs/wgpu/issues/8388) waits on andyleiserson. |

---

# Cross-cutting notes

## Decisions that only a maintainer can make

- **API and design:** [#10438](https://github.com/gfx-rs/wgpu/issues/10438) (async pipeline API), [#10506](https://github.com/gfx-rs/wgpu/issues/10506) (Metal pass merging), [#10775](https://github.com/gfx-rs/wgpu/issues/10775) (extension pattern), [#10880](https://github.com/gfx-rs/wgpu/issues/10880) (public trim-window option), [#9876](https://github.com/gfx-rs/wgpu/issues/9876) (subgroup size API), [#10697](https://github.com/gfx-rs/wgpu/issues/10697) (non-standard builtin), [#10583](https://github.com/gfx-rs/wgpu/issues/10583) (alias dedup approach), [#10591](https://github.com/gfx-rs/wgpu/issues/10591) (wanted or not).
- **Release scope:** [#10985](https://github.com/gfx-rs/wgpu/issues/10985) (a v30.0.3).
- **Validation and spec:** [#10890](https://github.com/gfx-rs/wgpu/issues/10890) (count draws from JS), [#10891](https://github.com/gfx-rs/wgpu/issues/10891) (2047 limit as a hard error), [#10840](https://github.com/gfx-rs/wgpu/issues/10840) (limit to drivers before r46).
- **Scope:** [#10778](https://github.com/gfx-rs/wgpu/issues/10778) (compile-time libwayland), [#10731](https://github.com/gfx-rs/wgpu/issues/10731) and [#10743](https://github.com/gfx-rs/wgpu/issues/10743) (OCLP workarounds), [#10725](https://github.com/gfx-rs/wgpu/issues/10725) (probing; at the meeting of 2026-11-18).

## Highest-severity findings

- **[#10890](https://github.com/gfx-rs/wgpu/issues/10890) (high, untrusted input):** count draws reach wgpu-core from Deno JS without argument validation on Vulkan and DX12.
- **[#10812](https://github.com/gfx-rs/wgpu/issues/10812) (high, safe-API UB):** child and parent textures share clear views, but every `Texture::drop` destroys them.
- **[#10506](https://github.com/gfx-rs/wgpu/issues/10506) (high, UB; your queue):** ICB setup for multi-draws is encoded after the pass that uses it when indirect validation is off.
- **[#9940](https://github.com/gfx-rs/wgpu/issues/9940) (high, DoS):** the SPIR-V constant writer is exponential in nesting depth. Already found by jimblandy.
- **[#10289](https://github.com/gfx-rs/wgpu/issues/10289) (high, from code; draft):** raising the per-stage acceleration structure limit from 1 to 8 overflows Metal's 31-slot buffer budget in the vertex stage.
- **[#10901](https://github.com/gfx-rs/wgpu/issues/10901) (medium, from code):** the new stride check rejects a plain literal stride.
- **[#10981](https://github.com/gfx-rs/wgpu/issues/10981) (medium, from code):** 10-bit EGL configs can sort first on every EGL platform.
- **[#10990](https://github.com/gfx-rs/wgpu/issues/10990) (medium, from code):** no test runs the new zero-init loops; lavapipe takes the native path. A skipped element leaks old workgroup memory.
- **[#10838](https://github.com/gfx-rs/wgpu/issues/10838) (medium):** the empty BLAS reads the zero buffer without BLAS-input usage or a barrier.
- **[#10875](https://github.com/gfx-rs/wgpu/issues/10875) (medium):** every Metal command buffer error loses the device, including the iOS background error.
- **[#10812](https://github.com/gfx-rs/wgpu/issues/10812) (medium):** parent and children share one tracker index.
- **[#10676](https://github.com/gfx-rs/wgpu/issues/10676) (medium):** valid WGSL with an override can fail pipeline creation.
- **[#10731](https://github.com/gfx-rs/wgpu/issues/10731) (medium, performance):** a compile-and-pipeline probe on every Intel Mac at enumeration.

## Policy: agent-written PRs

AGENTS.md does not allow fully agentic PRs. These need a policy call, ideally one consistent answer:

- **Agent wrote the code and the description:** [#10981](https://github.com/gfx-rs/wgpu/issues/10981) (first-time contributor), [#10750](https://github.com/gfx-rs/wgpu/issues/10750), [#10831](https://github.com/gfx-rs/wgpu/issues/10831) (agent account), [#10862](https://github.com/gfx-rs/wgpu/issues/10862), [#11000](https://github.com/gfx-rs/wgpu/issues/11000).
- **Agent wrote the code, Claude drafted the description:** [#10979](https://github.com/gfx-rs/wgpu/issues/10979), [#10990](https://github.com/gfx-rs/wgpu/issues/10990), [#10991](https://github.com/gfx-rs/wgpu/issues/10991) (matthargett; the description box is unchecked or broken).
- **Claude Code wrote the description at the author's direction, no disclosure boxes:** [#10985](https://github.com/gfx-rs/wgpu/issues/10985).
- **Mostly LLM-written with unticked self-review or an open policy question:** [#10676](https://github.com/gfx-rs/wgpu/issues/10676), [#10725](https://github.com/gfx-rs/wgpu/issues/10725), [#10839](https://github.com/gfx-rs/wgpu/issues/10839), [#10880](https://github.com/gfx-rs/wgpu/issues/10880).
- [#9353](https://github.com/gfx-rs/wgpu/issues/9353): 5andr0 offered an agent-written redraft.
- Your own [#10869](https://github.com/gfx-rs/wgpu/issues/10869) and [#10983](https://github.com/gfx-rs/wgpu/issues/10983) have the self-review box unchecked.

## Dependencies and landing order

- **naga ray query:** [#10694](https://github.com/gfx-rs/wgpu/issues/10694) and [#10600](https://github.com/gfx-rs/wgpu/issues/10600) in any order, then [#10697](https://github.com/gfx-rs/wgpu/issues/10697) (adapt to the [#10600](https://github.com/gfx-rs/wgpu/issues/10600) helper signature).
- **naga snapshots:** [#10600](https://github.com/gfx-rs/wgpu/issues/10600) and [#10957](https://github.com/gfx-rs/wgpu/issues/10957) conflict in `naga/tests/out/spv/wgsl-operators.spvasm`; the PR that lands second regenerates it.
- **naga cooperative matrix:** [#10901](https://github.com/gfx-rs/wgpu/issues/10901) first. Then [#10596](https://github.com/gfx-rs/wgpu/issues/10596) rebases its capability arm onto `require_cooperative_matrix()`, and [#10414](https://github.com/gfx-rs/wgpu/issues/10414) resolves a conflict in `naga/src/valid/expression.rs` (both carry the same "report the failing operand" fix).
- **Vulkan semaphores:** [#10840](https://github.com/gfx-rs/wgpu/issues/10840) (Arm Mali, a workaround bit) and [#10992](https://github.com/gfx-rs/wgpu/issues/10992) (Imagination, a change to the `timeline_semaphores` expression) both send a driver to the FencePool path, whose race [#10960](https://github.com/gfx-rs/wgpu/issues/10960) (merged) fixed. They conflict in `wgpu-hal/src/vulkan/adapter.rs`. [#10840](https://github.com/gfx-rs/wgpu/issues/10840) uses bit `1 << 3`, which trunk uses, and [#10991](https://github.com/gfx-rs/wgpu/issues/10991) takes `1 << 4` and `1 << 5`.
- **Metal multi-draw:** land [#9679](https://github.com/gfx-rs/wgpu/issues/9679) before [#10890](https://github.com/gfx-rs/wgpu/issues/10890), then drop the duplicate alignment change from [#10890](https://github.com/gfx-rs/wgpu/issues/10890). The [#10506](https://github.com/gfx-rs/wgpu/issues/10506) ordering bug also affects [#9679](https://github.com/gfx-rs/wgpu/issues/9679).
- **Metal adapter:** [#10860](https://github.com/gfx-rs/wgpu/issues/10860) must rebase onto [#10927](https://github.com/gfx-rs/wgpu/issues/10927) (merged). [#10875](https://github.com/gfx-rs/wgpu/issues/10875) and [#9679](https://github.com/gfx-rs/wgpu/issues/9679) both change `wgpu-hal/src/metal/mod.rs` (not checked for a conflict).
- **Metal MSL / Intel:** [#10725](https://github.com/gfx-rs/wgpu/issues/10725), then [#10731](https://github.com/gfx-rs/wgpu/issues/10731), then [#10743](https://github.com/gfx-rs/wgpu/issues/10743) (or squash [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731)).
- **Metal workgroup memory:** [#10800](https://github.com/gfx-rs/wgpu/issues/10800) supersedes [#10673](https://github.com/gfx-rs/wgpu/issues/10673). [#10800](https://github.com/gfx-rs/wgpu/issues/10800) does not cover passthrough MSL with `[[threadgroup(n)]]` arguments, which [#10673](https://github.com/gfx-rs/wgpu/issues/10673) handles.
- **Metal AS sync:** [#10292](https://github.com/gfx-rs/wgpu/issues/10292) is a superset of [#9645](https://github.com/gfx-rs/wgpu/issues/9645). Land [#9645](https://github.com/gfx-rs/wgpu/issues/9645) first, then a split [#10292](https://github.com/gfx-rs/wgpu/issues/10292).
- **naga workgroup zero-init:** inner-daemons asks to hold [#10990](https://github.com/gfx-rs/wgpu/issues/10990) until the shared zero-init pass ([#10456](https://github.com/gfx-rs/wgpu/issues/10456), on draft [#10409](https://github.com/gfx-rs/wgpu/issues/10409)) is designed.
- **Core trackers:** [#10862](https://github.com/gfx-rs/wgpu/issues/10862) overlaps rowanG077's `ResourceMetadata::clear` commit on his fork (not a PR).
- **Changelog:** [#10983](https://github.com/gfx-rs/wgpu/issues/10983), then the v31 release, then [#10984](https://github.com/gfx-rs/wgpu/issues/10984). Changelog-only conflicts: [#10891](https://github.com/gfx-rs/wgpu/issues/10891), [#10991](https://github.com/gfx-rs/wgpu/issues/10991).

## Infrastructure problems seen in CI

- The v30 MSRV Minimal Versions job fails: `cargo` cannot fetch the `libtest-mimic` fork rev `9979b3c`, which no branch holds. It blocks [#10985](https://github.com/gfx-rs/wgpu/issues/10985).
- Cancelled macOS jobs are on the head of [#10506](https://github.com/gfx-rs/wgpu/issues/10506), and on drafts [#10409](https://github.com/gfx-rs/wgpu/issues/10409), [#10790](https://github.com/gfx-rs/wgpu/issues/10790), [#10803](https://github.com/gfx-rs/wgpu/issues/10803), [#9353](https://github.com/gfx-rs/wgpu/issues/9353) and [#9975](https://github.com/gfx-rs/wgpu/issues/9975).
- Metal `poll::wait_on_submission_with_timeout` times out on the paravirtual Mac runner: [#10866](https://github.com/gfx-rs/wgpu/issues/10866) and [#10984](https://github.com/gfx-rs/wgpu/issues/10984). Possible flake, not confirmed.
- HTTP 504 tool downloads and a `mach-dxcompiler-rs` build script panic on Windows: [#10890](https://github.com/gfx-rs/wgpu/issues/10890), [#9679](https://github.com/gfx-rs/wgpu/issues/9679).
- The Metal ray tracing stack ([#10287](https://github.com/gfx-rs/wgpu/issues/10287), [#10288](https://github.com/gfx-rs/wgpu/issues/10288), [#10289](https://github.com/gfx-rs/wgpu/issues/10289), [#10292](https://github.com/gfx-rs/wgpu/issues/10292)) has no workflow runs at all.

## Limits of this pass

- No builds or tests were run. High and medium findings that say "from code" were not reproduced.
- "Last maintainer reply" uses the GitHub author association of comments and reviews. A reply from someone without the member, owner or collaborator association does not count.
- [#10839](https://github.com/gfx-rs/wgpu/issues/10839): the missing [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field is reasoned from the struct definition and the diff.
- [#10875](https://github.com/gfx-rs/wgpu/issues/10875) iOS background behavior and [#10860](https://github.com/gfx-rs/wgpu/issues/10860) watchOS indirect dispatch were not checked on hardware.
- [#10891](https://github.com/gfx-rs/wgpu/issues/10891): the 2047 limit was read in the WGSL specification; Tint's behavior was not checked.
- [#10957](https://github.com/gfx-rs/wgpu/issues/10957): `cts_runner/fail.lst` may need an update for the bitcast validation tests; the CTS was not run.

---

# Details

Entries are grouped by area and sorted by number. **Reviewed** is the date of the last code review of the entry; day counts in an entry are as of that date. **State** is read again at each update.

## naga

#### [#9295](https://github.com/gfx-rs/wgpu/issues/9295): fix(naga/spv): emit layoutless types for Workgroup storage class variables ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Needs a lot of work
- **Waiting on:** kvark and maintainer. kvark must say if he will continue, and if he will adopt [#10928](https://github.com/gfx-rs/wgpu/issues/10928). inner-daemons said that he will look at the PR again.
- **Last activity:** 2026-11-13, 5andr0, comment. 0 days.
- **Approvals:** none (cwfitzgerald CHANGES_REQUESTED, asking for regenerated GLSL sidecars)
- **CI:** green (old run, 2026-09-21); has conflicts
- **Summary:** The SPIR-V backend stops adding `ArrayStride`/`Offset`/`MatrixStride`/`ColMajor` to types that only Workgroup or TaskPayload variables use (VUID-StandaloneSpirv-None-10684). Types that buffer variables also use get an undecorated copy on SPIR-V 1.4+. Loads and stores convert between the two types with `OpCopyLogical`.
- **Findings:**
  1. [high, reasoned from code] Arena `ptr<workgroup, S>` types point to the decorated `S`, because `write_workgroup_types` runs after the arena loop. A call `f(&wg)` with a shared `S` gives an argument/parameter type mismatch.
  2. [medium, reasoned from code] The mesh output wrapper (`mesh_shader.rs:91,114,127,546,686-708`) uses default type IDs for Workgroup pointers. This is wrong when the output array type is shared with a buffer.
  3. [medium] No test covers the copy path: the test toml sets no SPIR-V version, and no type is shared.
- **To land:** Either rebase and fix bugs 1 and 2 and add the shared-type 1.4 test here, or replace this PR with [#10928](https://github.com/gfx-rs/wgpu/issues/10928)'s commit, which already has kvark as co-author.
- **Move-along action:** ask kvark to take [#10928](https://github.com/gfx-rs/wgpu/issues/10928)'s commit into [#9295](https://github.com/gfx-rs/wgpu/issues/9295) and then fix the mesh wrapper and add a `ptr<workgroup, S>` test. Close [#10928](https://github.com/gfx-rs/wgpu/issues/10928) after that. If kvark does not answer, inner-daemons can adopt [#10928](https://github.com/gfx-rs/wgpu/issues/10928).
- **Depth:** status only for [#9295](https://github.com/gfx-rs/wgpu/issues/9295)'s code (unchanged). Full comparison with [#10928](https://github.com/gfx-rs/wgpu/issues/10928).

#### [#9940](https://github.com/gfx-rs/wgpu/issues/9940): fix(naga): Additional cases of vector flattening ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-10-21.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (andyleiserson). jimblandy requested changes on 2026-10-21.
- **Last activity:** 2026-10-21, jimblandy, CHANGES_REQUESTED review plus a follow-up comment with a 13 s repro. 16 days ago.
- **Approvals:** none
- **CI:** green (head 907bb9c, rebased by jimblandy on 2026-10-21)
- **Summary:** Flattens nested vector composes before SPIR-V `OpConstantComposite`, and before const-eval swizzle/index/select. This fixes wrong constant results ([#10463](https://github.com/gfx-rs/wgpu/issues/10463)) and a const `select` panic. It also refactors `flatten_compose` to emit `FlattenedComponent::Zero`, and dedups the SPIR-V constant writer.
- **Findings:**
  1. [high] `naga/src/back/spv/writer.rs:109` (open jimblandy thread) recursively calls `write_constant_expr` on operands. This is exponential in the nesting depth: a 28-deep `array<...,2>()` takes 13 s in release. That is a DoS on untrusted WGSL (Firefox).
  2. [high] `naga/src/proc/constant_evaluator.rs:2313` (open jimblandy thread) is a regression. `eval_constructors` no longer lowers matrix columns, so `const m = mat2x2f(vec2f(), vec2f(1.0)); const v = vec2f(1.0,2.0) * m;` now panics.
  3. [medium] There is no CHANGELOG entry, although the description checklist says it is "_Needed_". This is a user-visible correctness fix.
  4. [low] Commit hygiene (jimblandy): the commit "Extend eval_zero_value_and_splat to composes" does not pass tests alone. The PR is marked Rebase, so squash it with the next commit or switch the PR to squash.
  5. [low] Design question, not blocking: jimblandy suggested removing flattening from the IR, and later agreed to go ahead with this approach.
- **To land:** fix the exponential recursion (use cached `constant_ids`). Fix the matrix-column panic and add a snapshot or test for it. Add a CHANGELOG entry. Fix or squash the commit that does not build alone.
- **Move-along action:** ping author (andyleiserson) on jimblandy's two 2026-10-21 threads.
- **Depth:** targeted (timeline and threads in full, diff file list. I relied on jimblandy's review for the code and did not re-review the 27 files.)

**CI and status check ([#9940](https://github.com/gfx-rs/wgpu/issues/9940): fix(naga): Additional cases of vector flattening):**

> - **Open questions from the first pass:** full CI on current head (first pass saw 30 of about 67).
> - **Answers:**
>   1. [confirmed] All 69 check runs complete with no failure: 65 success, 2 neutral (Mergify Merge Queue, Mergify Merge Protections), 2 skipped ("Trigger head-approved check"). Head 907bb9c. `mergeable_state: blocked`, `mergeable: true`.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10016](https://github.com/gfx-rs/wgpu/issues/10016): Emit spec-compliant SPIR-V under the Vulkan memory model, volatile support in MSL ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-09-27.
- **Verdict:** Draft: stalled
- **Waiting on:** maintainer (cwfitzgerald must decide on scope) and then author
- **CI:** green on head `c0b23d1` (2026-09-27). The PR is `dirty` (merge conflicts with trunk).
- **Summary:** Makes the SPIR-V writer follow the Vulkan memory model when a module uses cooperative matrices, or when the new `spv::Options::use_vulkan_memory_model` is set. It adds NonPrivatePointer and MakeAvailable/Visible operands, QueueFamily scopes, and Coherent/Volatile as per-access operands. It adds an `enable wgpu_memory_fence;` extension (`storageFence()` / `workgroupFence()`), `@volatile` / `@coherent` support in MSL, and changes the Vulkan adapter in wgpu-hal. Closes [#8805](https://github.com/gfx-rs/wgpu/issues/8805). Status: kvark converted it to draft on 2026-09-27 after inner-daemons' second review. The fix commits were pushed the same day, and nothing has happened since (about 6 weeks). kvark cross-referenced it from other PRs on 2026-10-13/14 ([#10596](https://github.com/gfx-rs/wgpu/issues/10596)'s bf16 tests check VMM output for cooperative matrices, so the two PRs overlap).
- **Findings:**
  1. [high] Scope dispute is unresolved: inner-daemons wants VMM split from the MSL volatile/coherent and fence work ("very big step ... warrants much closer scrutiny") and left the decision to cwfitzgerald. kvark wants to land them together (and offered `doc(hidden)` on the option). This decision blocks everything else.
  2. [medium] Unresolved thread `naga/src/back/spv/image.rs:1403`: under VMM, `OpImageRead` / `OpImageWrite` have no `NonPrivateTexel` / `MakeTexel{Available,Visible}` operands. So storage-texture communication stays undefined, and "Closes [#8805](https://github.com/gfx-rs/wgpu/issues/8805)" overstates what the PR does.
  3. [medium] Unresolved threads on fence uniformity (`naga/src/valid/analyzer.rs:947`, `naga/src/back/wgsl/writer.rs:1133`): kvark agreed to remove the MSL fence and the WGSL-out path. The threads are still open, so check that the last push did this.
  4. [medium] The unresolved (outdated) thread on `naga/src/back/msl/writer.rs` says that SPIR-V-parsed shaders with any `OpMemoryBarrier` can no longer be compiled to MSL, which would be a regression. It is outdated but still open; confirm that it is fixed.
  5. [low] The change affects SPIR-V output for all existing cooperative-matrix modules and wgpu-hal Vulkan device setup, so it needs a GPU test run before it lands.
- **To land:** maintainer decision on splitting VMM; author rebases, resolves the open threads (image operands or downgrade "Closes", fence uniformity, MSL `OpMemoryBarrier` regression), and marks the PR ready.
- **Depth:** status note only (not reviewed). I read the description, commits, all comments and review threads, the file list and CI.

#### [#10414](https://github.com/gfx-rs/wgpu/issues/10414): feat(naga): support non-square cooperative matrix tiles ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-28.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (merge conflict). After the rebase, it waits on inner-daemons, whose 3 threads have answers but are not resolved. infosave2007 published a rebased branch and reported that the GPU test passes on an RTX 5090.
- **CI:** green (from 2026-09-25, before the conflict. The mergeable state is `dirty`.)
- **Summary:** Adds `coop_mat8x16`/`coop_mat16x8` (columns first, like `matCxR`). `coopMultiplyAdd` now checks the K×M / N×K / N×M shapes and requires that A and B have the same scalar type. wgpu-core gives naga the set of tile shapes that the adapter reports, and naga rejects types with other shapes. The PR also updates the documentation to use the columns-by-rows convention and adds a GPU example test.
- **Findings:**
  1. [medium] `wgpu-core/src/device/resource.rs:374-397`, `naga/src/valid/type.rs:461` The shape check takes the union of all A/B/C shapes from all configurations and ignores scalar type and role. A shader can mix shapes from different configurations and still pass. Example: (8,8,8,f32) and (M=8,N=16,K=16,f16) together allow the shapes A=8x8, B=16x8, C=16x8, which form an M=8,N=16,K=8 multiply that no configuration supports. On Vulkan this violates the `OpTypeCooperativeMatrixKHR`/MulAdd VUIDs. The feature is `EXPERIMENTAL_*`, so this is not a safe-API UB hole. But the docs bullet "Tile sizes (M, N, K) and scalar types match at least one entry" (cooperative_matrices.rs ~413) claims more than the code validates. Either validate full (M,N,K,ab,cr) configurations at `CooperativeMultiplyAdd`, or reword the docs.
  2. [low] `naga/src/valid/mod.rs:680` `Validator::new` with `COOPERATIVE_MATRIX` allows all 4 shapes. The MSL backend then emits `simdgroup_half16x8`, which Metal does not have, so standalone naga users (the CLI) get MSL that does not compile instead of a naga error. wgpu is protected because Metal reports only 8x8.
  3. [low] `wgpu-core/src/device/resource.rs:2619` `adapter.cooperative_matrix_properties()` and the shape list are computed again for every shader module creation, also when the feature is off. This is cheap, but it could be computed once per device.
  4. [low] The PR is 971 lines. 338 of them are the GPU test that inner-daemons requested. The documentation rewrite (columns-first, `u32`/`CooperativeScalarType` field types) is partly unrelated scope.
- **To land:** Rebase (the conflict is only in `naga/tests/naga/validation.rs`, and infosave2007's branch already resolves it). Fix the docs/validation mismatch in finding 1. Ask inner-daemons to resolve the threads.
- **Depth:** full for the naga, wgpu-core and docs changes. I did not read the GPU test (`examples/features/src/cooperative_matrix/tests.rs`) or the `wgsl_errors`/`validation.rs` test bodies. I checked that the SPIR-V snapshot emits Rows/Columns in the right order and with the right load strides.

Targeted pass on: naga validation of untrusted shaders, feature/capability gating, and test coverage. Backend codegen correctness got only a light read.

#### [#10583](https://github.com/gfx-rs/wgpu/issues/10583): [naga msl-out] preserve equivalent array aliases ([@pbkx](https://github.com/pbkx))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-10.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (jimblandy / inner-daemons must choose the approach)
- **Last activity:** 2026-11-10, kpreid, comment. 3 days.
- **Approvals:** none
- **CI:** green (66 pass); mergeable
- **Summary:** When two IR array types have the same `TypeInner`, the MSL writer now emits a `typedef` to the first wrapper struct and does not emit a second, incompatible struct. This fixes WGSL aliases such as `alias A = array<u32,5>; alias B = array<u32,5>;`.
- **Findings:**
  1. [medium] `naga/src/back/msl/writer.rs:4778` The key is `ty.inner`, so dedup compares `base` by handle. Nested aliases (`array<A,2>` vs `array<B,2>`) are still distinct, so the bug remains for them.
  2. [medium] Scope: the same bug class exists in SPIR-V, HLSL and the validator. This PR fixes only MSL. kpreid now suggests a shared `naga::proc` helper, which is a third option next to per-backend fixes (jimblandy) and IR dedup (inner-daemons).
  3. [low] The reporter confirmed on an M4 Max that the PR fixes the original repro.
- **To land:** a maintainer chooses the approach (per-backend, or a shared `naga::proc` helper). The author handles the nested-alias case. File issues for HLSL, SPIR-V and the validator.
- **Move-along action:** decide between per-backend dedup and a shared `naga::proc` helper, and tell the author.
- **Depth:** status only

#### [#10596](https://github.com/gfx-rs/wgpu/issues/10596): Support BF16 type storage and cooperative matrix operations ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (kvark). inner-daemons requested changes on 11-13 and asks whether the planned `subgroup_matrix` extension replaces cooperative matrices.
- **CI:** green on head `ed22a76` (all 66 non-Mergify checks pass). The PR is `dirty` (it has merge conflicts with trunk), so it needs a rebase.
- **Summary:** Adds `ScalarKind::BFloat` and `Literal::BF16` to naga IR, a `wgpu_bfloat16` WGSL enable extension, and `Capabilities::SHADER_BFLOAT16`. bf16 is allowed only for storage, `f32 <-> bf16` conversions, and cooperative-matrix operands. The SPIR-V (`SPV_KHR_bfloat16`) and MSL 3.1+ (`bfloat`) backends support it. GLSL and HLSL reject it.
- **Findings:**
  1. [medium] `naga/src/proc/constant_evaluator.rs:2882` (`binary_op`): the const evaluator folds `==`, `!=`, `<`, `<=`, `>`, `>=` on two `Literal::BF16` values into a `bool` literal, using the generic `Literal` comparison. The new runtime validator rule (`valid/expression.rs`, Equal/NotEqual/Less... reject `BFloat`) never sees these expressions, so `const b = bf16(1.0) < bf16(2.0);` is accepted even though the same comparison is rejected at runtime. Add `BF16` rejection to the const-eval comparison path, and add an error test for it.
  2. [medium] `naga/tests/naga/bf16.rs` (new, about 280 lines): all coverage is in a custom test module that checks the output with `assembly.contains(...)`. Nothing is added under `naga/tests/in/wgsl` with snapshots, so CI's `Validate: SPIR-V` (spirv-val) and `Validate: MSL` jobs never check the generated bf16 SPIR-V or MSL. Add a snapshot input (SPIR-V and MSL targets) so the output is validated. Also add the enable-extension test to `wgsl_errors.rs`, which is the convention for enable extensions.
  3. [low] `naga/src/back/spv/writer.rs:1993`: each bf16 type declaration inserts `StorageBuffer16BitAccess`, `UniformAndStorageBuffer16BitAccess` and `SPV_KHR_16bit_storage` directly into `capabilities_used`. This happens even when bf16 is only used in function-local or workgroup memory, and it skips the `require_any` check that other capabilities use. In `back/spv/block.rs:801`, a separate `request_type_capabilities` call for each cooperative-matrix expression is an ad hoc path next to the normal type-capability flow. Ask for the rationale, or make it the same as the f16 handling.
  4. [low] `CHANGELOG.md`: adds an empty `### Added/New Features` / `#### Naga` header at the top of Unreleased. The real entry has no `By @kvark in [#10596]`.
  5. [low] Scope/plumbing: no `wgpu-types` feature or `wgpu-naga-bridge` mapping is added, so `SHADER_BFLOAT16` is unreachable from `wgpu`. This is safe because the gating defaults to off. The description and changelog should say that this is naga-only for now. The PR description is thin (empty "Connections", "LLM used to generate code", and the "Tests demonstrate validation" box is unchecked).
- **To land:** rebase; reject bf16 comparisons in const eval and add a test for it; add snapshot tests that go through spirv-val and the MSL validation; fix the CHANGELOG (remove the stray header, add attribution); justify or fix the unconditional 16-bit-storage capabilities.
- **Depth:** targeted. I read all validator, front-end, const-eval and capability changes. I skimmed the SPIR-V and MSL writers. I did not check the MSL `simdgroup_bfloat8x8` codegen against the Metal spec, or the SPIR-V instructions against the full `SPV_KHR_bfloat16` allowed-instruction list (for example, whether `OpSelect` on bf16 is legal; the validator still accepts `select` on bf16).

**Deeper review ([#10596](https://github.com/gfx-rs/wgpu/issues/10596): Support BF16 type storage and cooperative matrix operations):**

> - **Open questions from the first pass:**
>   1. Review the parts the first pass skipped: the MSL and HLSL backends, and the frontends.
>   2. Can bf16 reach a backend that cannot express it, without a validation error?
> - **Answers:**
>   1. **[done]**
>      - **HLSL:** `Writer::write` returns `UnsupportedScalar(BF16)` at the start if `back::uses_bfloat16` is true (`naga/src/back/hlsl/writer.rs:330-332`). `write_literal` and `to_hlsl_str` also return errors. The `unreachable!()` in `hlsl/conv.rs:14` is reached only after that early return, so it is safe.
>      - **GLSL:** same pattern, in `Writer::new` (`glsl/writer.rs:65-67`).
>      - **MSL:**
>        - It gates on `lang_version >= (3,1)` (`msl/writer.rs:4577-4581`).
>        - Scalars map to `bfloat` (`msl/writer.rs:572-575`), vectors to `metal::bfloatN`, and cooperative matrices to `metal::simdgroup_bfloat8x8`.
>        - A literal is written as `bfloat(<f32>)`.
>        - Casts are written as `static_cast<bfloat…>`. The validator never allows a bitcast to or from bf16, so `as_type` is not used.
>        - A tight `vec3<bf16>` struct member goes through `should_pack_struct_member` (`msl/writer.rs:637-641`, width 2), which gives `packed_bfloat3`.
>        - I did not check `packed_bfloat3`, `simdgroup_bfloat8x8`, or a mixed-type `simdgroup_multiply_accumulate` (bf16 A/B with f32 C) against the Metal spec. This is unknown.
>      - **`uses_bfloat16`** (`back/mod.rs:47-68`) checks the type arena, plus `As{kind: BFloat}` and `Literal::BF16` in function and entry-point expressions. Every other bf16 value comes from an arena type, so this is complete for emitted code. It does not scan `global_expressions`, but backends emit only the named constants, and those have arena types.
>      - **WGSL frontend:**
>        - `bf16` is gated by `enable wgpu_bfloat16` (`front/wgsl/parse/conv.rs:577-579`).
>        - Abstract numbers convert automatically to bf16. No concrete type does (`front/wgsl/lower/conversion.rs:485-488`).
>        - `texture_2d<bf16>` is rejected, because sample types must have width 4 (`front/wgsl/lower/mod.rs:2922`).
>      - **GLSL frontend:** it has no bf16 syntax. Only `type_power` changed.
>      - **SPIR-V frontend:** it is not changed. A bf16 `OpTypeFloat` has 4 words, and `parse_type_float` calls `inst.expect(3)` (`front/spv/mod.rs:2198`), so it fails with an operand-count error. It is not misread as f16. The error message is not specific, but the behavior is safe.
>      - **Validator:** it rejects bf16 for unary negate, all arithmetic, bitwise and comparison binaries, Math functions (the overload lists contain no bf16), derivatives, relational functions, subgroup ops (`valid/function.rs` kind lists), atomics (`valid/type.rs:491+`), overrides (`valid/mod.rs:721-732`), varyings (`valid/interface.rs:820`), and immediates. `select` on bf16 is still allowed. That is legal: `SPV_KHR_bfloat16` lists `OpSelect` as allowed (I checked the registry text).
>      - **wgpu-core:** wgpu-core and `wgpu-naga-bridge` never set `SHADER_BFLOAT16`, so the validator rejects bf16 on every wgpu path. Only direct naga users (for example naga-cli, which uses `Capabilities::all()`) can reach the backends.
>   2. **[confirmed: one gap, IR only]** The `As` arm of the validator (`naga/src/valid/expression.rs:1331-1358`) accepts a **matrix** source. `As { expr: <mat2x2<f32>>, kind: BFloat, convert: Some(2) }` passes, because `(Sc::F32, Sc::BF16)` is allowed and `check_width(BF16)` passes. The result is a `mat2x2<bf16>`, which `valid/type.rs:445` rejects as an arena type, but here it is only a value resolution. The backends then do this:
>      - **MSL:** writes `metal::bfloat2x2(m)` (`msl/writer.rs:3013-3024` → `put_numeric_type`). Metal has no bfloat matrix types as far as I know (not verified), so the output does not compile.
>      - **WGSL:** writes `mat2x2<bf16>(m)`, and the WGSL frontend then rejects it.
>      - **SPIR-V:** declares an `OpTypeMatrix` of bf16 columns, then returns `Error::Validation("Matrices must be floats")` (`spv/block.rs`, `write_as_expression`). The error is clean, but it comes from the backend.
>      - **HLSL/GLSL:** reject it cleanly.
>
>      WGSL cannot produce this, because writing `mat2x2<bf16>` adds a rejected arena type. Only hand-built IR can, for example `ShaderSource::Naga` with the capability on, or other naga users. Fix: in the `As` arm, reject `Ti::Matrix` when either kind is `BFloat`, and add a test.
>
>      I found no other path from WGSL, GLSL, or SPIR-V input to a bf16 value in a backend that cannot express it without a validation or backend error.
> - **New findings:**
>   1. [medium] The matrix `As` validator gap from answer 2 (`valid/expression.rs:1331-1351`). This is a one-line fix.
>   2. [low] Image sampled `kind: BFloat` is not checked by the validator (`valid/type.rs:909-938` checks no sampled kind). This gap already exists for `Bool`/abstract kinds. Hand-built IR would hit `unreachable!()` in `wgpu-core/src/validation.rs:928` (layout derivation). WGSL cannot produce it. This is not new to the PR, but the PR widens it.
>   3. [low] AbstractFloat → bf16 const conversion goes through f32 (`proc/constant_evaluator.rs:2518-2526`). This rounds twice and can be wrong by 1 ULP at halfway points. A value above the bf16 maximum becomes `inf`, which `check_literal_value` reports as a validation error, not as a const-eval error at the source location.
>   4. [nit] The message of the new `unreachable!` in `spv/image.rs:326-328` says "sampled image", but the operand is the coordinate component type.
> - **Verdict change:** none. "Needs a bit of work" stays. Add the matrix-`As` rejection to the list of things to do before landing.
> - **Move-along action change:** none.
>
> ---

#### [#10600](https://github.com/gfx-rs/wgpu/issues/10600): refactor(naga spv-out): Add wrapped function preamble helper ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-17.
- **Verdict:** Land as is
- **Waiting on:** reviewer (jimblandy is assigned and has not reviewed). GitHub `reviewDecision` is `REVIEW_REQUIRED`.
- **Last activity:** 2026-11-17, andyleiserson, force push that rebases onto trunk and converts the `hit_object.rs` calls. No comment with it.
- **Approvals:** none
- **CI:** green (65 pass, 2 neutral, 1 skipped).
- **Summary:** Adds `write_function_signature` and `write_wrapped_function_signature` in `spv/helpers.rs`. They emit `OpFunction`, the `OpFunctionParameter`s and, with `WriterFlags::DEBUG`, `OpName`s for the function and its parameters. The div/mod, std140-conversion, `matCx2` column, ray-query, ray-pipeline and hit-object helpers use them, and the ray helper in `spv/ray/mod.rs` is removed. Most of the 43 changed files are SPIR-V snapshot churn from new debug names and a different ID order. The PR prepares four external-texture helpers that come later.
- **Findings:**
  1. [medium] `naga/src/back/spv/ray/query.rs` in [#10697](https://github.com/gfx-rs/wgpu/issues/10697) (`write_ray_query_get_current_t_max`) calls the old `write_function_signature(&[...], ret)`. This PR removes that signature, so [#10697](https://github.com/gfx-rs/wgpu/issues/10697) does not compile after this PR lands. This is a landing-order item, not a defect of this PR. (From code; the merge also has textual conflicts that come from [#10697](https://github.com/gfx-rs/wgpu/issues/10697) against trunk.)
  2. [low] `naga/src/back/spv/writer.rs:925`, `:1151` The `format!("{:?}_from_std140", ...)` and `format!("{:?}_get_column", ...)` debug names are built also when `WriterFlags::DEBUG` is off. The cost is small and occurs once for each wrapped type.
  3. [low] The function ID is allocated after the caller's type lookups, and the ray helpers get new `OpName`s. This gives ID-order churn in about 40 SPIR-V snapshots, with no behavior change. A textual conflict with [#10957](https://github.com/gfx-rs/wgpu/issues/10957) in `naga/tests/out/spv/wgsl-operators.spvasm` follows from this; the PR that lands second regenerates the snapshot.
  4. [low] PR template: all checklist boxes are unchecked. The agent-disclosure lines are left in the HTML comment and not filled in. "LLM Use?" says "No".
- **To land:** a review from jimblandy (or another member). Release (in about one week): can land before it; it is a refactor with no user-visible change and needs no changelog entry.
- **Move-along action:** ask jimblandy to review; the code compiles on trunk and CI is green.
- **Depth:** full for `helpers.rs`, `writer.rs` and `spv/ray/*`. I compared the new head with the old head with `git range-diff`: the only source change is the 5 `hit_object.rs` call conversions. I did not read the snapshot files line by line.

#### [#10609](https://github.com/gfx-rs/wgpu/issues/10609): naga: Support GLSL matrix interfaces and column component access ([@rleux](https://github.com/rleux))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** other reviewer (inner-daemons, requested since 10-14)
- **Last activity:** 2026-11-09, rleux, force push. 4 days.
- **Approvals:** none
- **CI:** green (66 pass); mergeable
- **Summary:** In the GLSL frontend, this PR allows component and swizzle access through a matrix column (`m[0].y`, `m[1].xz = ...`). It splits matrix stage inputs and outputs into column vectors. The rewritten `arg_type_walker` assigns locations recursively, so locations are correct for arrays and blocks, and dvec3/dvec4 take 2 locations.
- **Findings:**
  1. [medium] `naga/src/front/interpolator.rs:52` `apply_default_interpolation` is shared with the SPIR-V frontend. With `sampling.get_or_insert(Center)`, the SPIR-V frontend now keeps a `Centroid`/`Sample` decoration when there is no interpolation decoration. Before, it replaced the sampling with `Center`. The doc comment now describes the new behavior. The PR description and CHANGELOG do not mention the SPIR-V frontend, and no SPIR-V-input test covers it. (Reasoned from code, not run.)
  2. [low] `naga/src/front/glsl/functions.rs:1500-1506` The location is incremented with `checked_add` after every leaf, also after the last one. So `layout(location=4294967295) in float x;` fails with "interface location overflow", although one location fits. The new test uses `mat2` at that location, which correctly overflows, so the false positive has no test.
  3. [low] `naga/src/front/glsl/parser/declarations.rs:545` In/out blocks now default to `Std430` instead of `Std140`. The description does not explain this. It changes IR struct offsets and spans for interface blocks.
  4. [low] `CHANGELOG.md:246-247` The two new entries have no `By @rleux in #10609` attribution.
- **To land:** Add the CHANGELOG attribution. Mention the SPIR-V frontend sampling change, or limit it to the GLSL path. Optionally fix the last-location overflow. inner-daemons (or another reviewer) must review.
- **Move-along action:** ping inner-daemons for review, or reassign. The GLSL frontend has low priority for them.
- **Depth:** targeted (I read all `naga/src` changes again; I skipped the snapshot outputs and parser test bodies, except the overflow test)

#### [#10661](https://github.com/gfx-rs/wgpu/issues/10661): naga: Fix process_overrides panic on anonymous overrides ([@SAY-5](https://github.com/SAY-5))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-16.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (SAY-5). The author rebased on 2026-11-16 and asked for review again, but did not add the snapshot test that inner-daemons requested on 2026-10-22.
- **Last activity:** 2026-11-16, SAY-5, rebase and review request.
- **Approvals:** none (CHANGES_REQUESTED from inner-daemons)
- **CI:** green (head deaf606). There is a merge conflict (`dirty`, probably CHANGELOG).
- **Summary:** In `process_overrides`, the inner "catch up to override h" loop now skips `process_override` for anonymous overrides (only their `init` is adjusted), as the final loop already did. This fixes the `unreachable!()` panic ([#10660](https://github.com/gfx-rs/wgpu/issues/10660)). `override_map` changes from `HandleVec` to `FastHashMap`, because skipped entries leave gaps.
- **Findings:**
  1. [medium] A snapshot test was requested and is missing. Only `naga/tests/naga/validation.rs` `override_in_array_size_expression` was added.
  2. [low] `naga/src/back/pipeline_constants.rs` ~213: `new_h` is not reset per iteration. If the target override `h` were itself anonymous, `new_h.unwrap()` would silently return the previous named override's constant (or panic). Today no `Expression::Override` points at an anonymous override, so this is latent. Setting `new_h = None` in the anonymous arm, or adding a `debug_assert`, would make it robust.
  3. [low] The anonymous/named `match` is now copied in both loops. A small helper would remove the duplication.
  4. [low] `process_function` still indexes `override_map[&h]`. That keeps the old panic-on-missing behavior and is acceptable.
- **To land:** add a snapshot (`naga/tests/in/wgsl/...`) that covers an array size expression before another override, then rebase.
- **Move-along action:** ping author (SAY-5) about the requested snapshot test. If the author does not reply, take over (the change is about 50 lines).
- **Depth:** full

**CI and status check ([#10661](https://github.com/gfx-rs/wgpu/issues/10661): naga: Fix process_overrides panic on anonymous overrides):**

> - **Open questions from the first pass:** full CI.
> - **Answers:**
>   1. [confirmed] 69 check runs: 66 success, 2 neutral, 1 skipped. No failure. Head deaf606. `dirty` (conflict).
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10676](https://github.com/gfx-rs/wgpu/issues/10676): [naga] Check out-of-bounds `let` indices at runtime instead of rejecting them ([@ilyas-mallah](https://github.com/ilyas-mallah))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-14.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (ilyas-mallah). inner-daemons withdrew the policy question on 2026-11-14 but said that the substance stays. The medium finding about `let` folding is not raised on the PR.
- **Last activity:** 2026-11-14, inner-daemons, deleted the policy-question comment and apologized for its tone. The author has not answered. The hostile third-party comment of 2026-10-28 is still there.
- **Approvals:** none (review requested from inner-daemons; jimblandy assigned)
- **CI:** green (head 372abd2, 2026-10-27)
- **Summary:** The validator no longer rejects `Access` with a constant out-of-range or negative index, because a WGSL `let` is not a const-expression ([#4390](https://github.com/gfx-rs/wgpu/issues/4390), [#6396](https://github.com/gfx-rs/wgpu/issues/6396)). Back ends (SPIR-V, GLSL, HLSL, MSL, WGSL) now replace a known out-of-range index with a valid one, and the bounds-check policy decides the runtime behavior. Const or override errors move to the WGSL and GLSL front ends and `process_overrides`. CTS access tests move to test.lst.
- **Findings:**
  1. [high] This is a security-sensitive relaxation of a naga validation check. After the change, every back end and every indexing path must avoid emitting a literal out-of-range index. The diff covers `Access` on values in each writer, plus the SPIR-V `Unchecked` path (`naga/src/back/spv/index.rs` ~562 clamps to `length-1`). It needs a careful audit of the paths it does not touch, such as GLSL/HLSL pointer chains and binding arrays under `Unchecked` (the binding-array policy is often `Unchecked`), and of input from the SPIR-V and GLSL front ends. A teoxoy or jimblandy-level review is needed.
  2. [medium] Policy question: the description reads as LLM-written. The "LLM Use" section only says that Claude Code "helped". The question from inner-daemons has no answer.
  3. [medium] `naga/src/valid/expression.rs`: removing `ExpressionError::NegativeIndex` breaks the public API, and the CHANGELOG does not mention it.
  4. [low] `naga/src/proc/index.rs` `known_out_of_bounds_replacement` returns `Some(0)` for a negative index into a runtime-sized array. Under `Unchecked` this reads element 0 instead of being UB, which is acceptable, but the doc comment should say so.
- **To land:** the author answers the LLM-policy question. A maintainer audits all index-emission paths for the relaxed validator. Note the API removal in the CHANGELOG.
- **Move-along action:** ping author for an answer to inner-daemons' policy question. If the answer is acceptable, assign a naga back-end reviewer (teoxoy or jimblandy) for the security audit.
- **Depth:** targeted (I read the validator, `proc/index.rs`, the SPIR-V index/block changes, and the HLSL/MSL replacement sites. I skipped the GLSL/WGSL writers, the front-end error moves, `pipeline_constants.rs` and the snapshots.)

**Deeper review ([#10676](https://github.com/gfx-rs/wgpu/issues/10676): [naga] Check out-of-bounds `let` indices at runtime instead of rejecting them):**

> - **Open questions from the first pass:**
>   1. Does every back end and every bounds-check policy (`Restrict`, `ReadZeroSkipWrite`, `Unchecked`) still produce safe code for a constant out-of-bounds index on every indexing path? This includes the paths that the diff does not touch: GLSL/HLSL pointer chains, binding arrays under `Unchecked`, and input from the SPIR-V and GLSL front ends.
>   2. Does the change match the WGSL spec for `let` and `const` indices?
> - **Answers:**
>   1. [confirmed, with caveats] Each back end is safe for the paths that wgpu uses. I traced each path:
>      - **SPIR-V, value access** (`naga/src/back/spv/block.rs:887-930`): `OpCompositeExtract` is used only when the index is known and `access_needs_check` is `None`. In all other cases the base is spilled to a temporary and goes through `write_bounds_check`.
>      - **SPIR-V, pointer chains** (`naga/src/back/spv/index.rs:530-580`): a negative constant becomes `Known(u32::MAX)` (`:546`). `Restrict` with a known index and a known length clamps statically (`write_restricted_index`). Otherwise it emits `UMin`. `ReadZeroSkipWrite` falls through to a runtime `OpULessThan` guard. The guarded block still has an `OpAccessChain` with the literal out-of-bounds constant (for example `%32 = OpAccessChain %28 %27 %29` with `%29 = 9` in `wgsl-bounds-check-let-zero.spvasm`). This code is dead at runtime. CI validates the snapshots with spirv-val and is green, so spirv-val accepts it. `Unchecked` with a known index into a fixed length clamps to `length - 1` (`:572`). For a negative constant into a runtime-sized array, `Unchecked` emits index `0xFFFFFFFF`. This is acceptable because a runtime index gets the same treatment under `Unchecked`. On Vulkan, wgpu uses `Unchecked` for buffers only when `robustBufferAccess2` is present.
>      - **MSL** (`naga/src/back/msl/writer.rs:3436-3600`): `Access` always reaches `put_subscripted_access_chain` as `GuardedIndex::Expression`, so the replacement always applies. `Restrict` writes `metal::min(unsigned(9), N-1u)`. `ReadZeroSkipWrite` writes `uint(9) < N ? … : 0`.
>      - **HLSL** (`naga/src/back/hlsl/writer.rs:3883-4030`): non-storage `Access` uses `min(uint(i), N-1u)` when `restrict_indexing` is set, and the replacement otherwise. This includes the sampler-heap index buffer and binding arrays. Storage-buffer chains (`naga/src/back/hlsl/storage.rs:632`) emit `index * stride` byte offsets and never clamp, before or after the PR. An out-of-range constant reads other bytes of the same binding. A negative constant wraps to a huge offset, and D3D12 raw-buffer robustness returns 0 for it. WGSL allows both results for an invalid memory reference to the same originating variable. The matCx2 helpers (`__get_col_of_matCx2`) get the raw index, but they use a `switch`, so this is not a compile error.
>      - **GLSL** (`naga/src/back/glsl/writer.rs:2452`): GLSL has no pointers, so this one site covers value access and lvalue access. GL uses `Unchecked` for `index` and `buffer` (`wgpu-hal/src/gles/device.rs:412-416`), so the replacement is the only protection. It is enough to prevent the GLSL compile-time error.
>      - **Binding arrays:** every hal back end uses `binding_array: Unchecked` (`wgpu-hal/src/vulkan/adapter.rs:2947`, `metal/device.rs:246`, `gles/device.rs:416`). A runtime out-of-range binding-array index is already unchecked on trunk. So the PR adds no new class of UB here. See new finding 2 for an inconsistency.
>      - **Front ends:** the GLSL front end now reports "Constant index is out of range" (`naga/src/front/glsl/context.rs` hunk at ~576). The WGSL front end rejects a negative const index (`naga/src/front/wgsl/lower/mod.rs:2539`). A positive const index becomes an `AccessIndex`, which the validator still checks. SPIR-V input that carries a constant out-of-range `Access` now reaches the back ends, which handle it as described above.
>   2. [confirmed] The change matches the spec. WGSL [array access](https://gpuweb.github.io/gpuweb/wgsl/#array-access-expr) says that an out-of-range index is a shader-creation error only for a const-expression and a pipeline-creation error only for an override-expression. Otherwise the result is an indeterminate value or an invalid memory reference. A `let` identifier is not a const-expression, and the WGSL front end marks it non-const (`naga/src/front/wgsl/lower/mod.rs:1868`, `force_non_const`). A negative const index into a runtime-sized array is a shader-creation error, and the front end now reports it. There is one gap: `let i = o; a[i]`, where `o` is an override. `process_function` decides "override-expression" by structure (`naga/src/back/pipeline_constants.rs:435,453`, `ExpressionKindTracker::from_arena`), so it converts this access to `AccessIndex` and rejects it at pipeline creation. The spec says it is a runtime index. Trunk also rejects this case, so it is not a regression.
> - **New findings:**
>   1. [medium, correctness] **Valid shaders fail at pipeline creation when the module uses an override.** `process_function` (`naga/src/back/pipeline_constants.rs:476-485`) uses a fresh `ExpressionKindTracker`, which classifies the `let` initializer `9` and `array<f32,2>(1.0, 2.0)` as `Const`. `try_eval_and_append(Access{Compose, Literal 9})` then const-evaluates the access. `ConstantEvaluator::access` returns `InvalidAccessIndex` (`naga/src/proc/constant_evaluator.rs:2351`), or `InvalidAccessIndexTy` for `-1` (`:1279`). The runtime fallback (`:1180-1188`) handles only `NotImplemented` and `InvalidBinaryOpArgs`, so the error propagates. Repro, reasoned and not run: the PR's own `index_array_value` or `index_vector_value` function plus any override that the entry point uses. The PR's `override_index_bounds` test does not catch this because its `let_index` case indexes a storage global, which is not const-foldable. Fix: when the evaluator fails on an `Access`, append the access as a runtime expression. Add a snapshot or test that has an override.
>   2. [low, pre-existing class] **The binding-array size lookup uses the index before replacement.** When the replacement or the `Unchecked` clamp moves a binding-array index (for example `bufs[10]` to `bufs[3]`), these paths still use the original constant: the MSL runtime-array length (`put_dynamic_array_max_index` → `_buffer_sizes.sizeN[unsigned(10)]`, `naga/src/back/msl/writer.rs:1799`) and the SPIR-V `OpArrayLength` chain (`naga/src/back/spv/index.rs:143`, `self.cached[index_outer]`). The bounds of the inner array then come from a different slot than the one accessed. This adds no new capability, because a runtime index 10 is already unchecked UB on trunk. But the replacement makes the code look safe when it is not. Either apply the same replacement in those two places, or document it next to the binding-array TODO.
>   3. [low] **Constant 64-bit indices are no longer rejected.** The removed block mapped `ConstValueError::InvalidType` to `InvalidIndexType`. An `i64`/`u64` literal index therefore failed validation on trunk. Now it passes (`naga/src/valid/expression.rs:425`), and `known_out_of_bounds_replacement` returns `None` for it (`naga/src/proc/index.rs:350`), so the literal goes to the back end unchanged. Trunk already accepted runtime 64-bit indices, so this extends a gap that exists. I did not trace 64-bit index handling in every back end. Suggest one test with `SHADER_INT64`.
>   4. [nit] `known_out_of_bounds_replacement` returns `Some(0)` for a negative index into a runtime-sized array (`naga/src/proc/index.rs:365`). SPIR-V uses `u32::MAX` for the same case. Both are correct, but the doc comment should state the runtime-sized behavior. The CHANGELOG has a "Bug Fixes" line, but it does not mention the removed public `ExpressionError::NegativeIndex` variant.
>   5. [info] Test coverage: there are no snapshots for binding arrays, uniform matCx2, workgroup or private memory, or overrides. HLSL and GLSL are not covered for `ReadZeroSkipWrite`, because those back ends do not support that policy.
> - **Verdict change:** none ("Needs a bit of work"). The safety audit passes. New finding 1 is a real correctness bug and should be fixed before merge.
> - **Move-along action change:** add one step: ask the author to fix the `process_overrides` folding (finding 1) and add an override test. The author still must answer the policy question from inner-daemons, and a back-end reviewer still must sign off.
>
> ---

**CI and status check ([#10676](https://github.com/gfx-rs/wgpu/issues/10676): [naga] Check out-of-bounds `let` indices at runtime):**

> - **Open questions from the first pass:** full CI.
> - **Answers:**
>   1. [confirmed] 67 check runs: 65 success, 2 neutral. No failure. Head 372abd2. `blocked` (no approval), `mergeable: true`.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10694](https://github.com/gfx-rs/wgpu/issues/10694): fix(wgsl-in): NotInScope error for `rayQueryGenerateIntersection` argument ([@DouglasDwyer](https://github.com/DouglasDwyer))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI red (1 failed: Validate: SPIR-V/GLSL/DOT/WGSL); last update 2026-10-21.
- **Verdict:** Land as is
- **Waiting on:** maintainer (assigned jimblandy; no review yet)
- **Last activity:** 2026-10-21, DouglasDwyer, opened the PR (the 2026-10-28 update has no comment or review, so it is probably the assignment). 16 days.
- **Approvals:** none
- **CI:** red, not related. "Validate: SPIR-V/GLSL/DOT/WGSL" failed after 8 s in step "Install Vulkan SDK" with curl exit 56 (network). All other checks are green.
- **Summary:** Flushes the emitter before the `GenerateIntersection` `RayQuery` statement, so that a `hit_t` such as `candidate.t` is emitted before it is used. This uses the same pattern as `rayQueryInitialize`. Adds a validation regression test and a CHANGELOG entry.
- **Findings:**
  1. [low] `naga/tests/naga/wgsl_errors.rs:5238` The test doc comment says "Previously, ...", which describes history instead of the current behavior. This is a nit and does not block.
- **To land:** rerun the failed infra job, then approve.
- **Move-along action:** merge (rerun the SPIR-V validate job, then approve and queue)
- **Depth:** full

#### [#10697](https://github.com/gfx-rs/wgpu/issues/10697): feat(naga, wgsl): Add `rayQueryGetCurrentTMax` function for ray queries ([@DouglasDwyer](https://github.com/DouglasDwyer))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-28.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (jimblandy assigned: design decision and review), then author (rebase)
- **Last activity:** 2026-10-28, Vecvec, commented that the PR uses the less strict init tracking from [#10735](https://github.com/gfx-rs/wgpu/issues/10735), so it does not match the current tracking. No author reply. 9 days.
- **Approvals:** none
- **CI:** green (on the 2026-10-22 head). The PR has merge conflicts (`dirty`).
- **Summary:** Adds a new wgpu-only WGSL builtin, `rayQueryGetCurrentTMax(&rq) -> f32`, and a new IR expression `RayQueryGetCurrentTMax`. It returns the committed `t`, or the tracked `t_max` if there is no committed hit. It is lowered for SPIR-V (helper function), HLSL (`CommittedRayT()`) and MSL (helper and `t_max` tracker). The PR includes snapshot tests, validation tests, a GPU test (`ray_tracing/current_t_max`), docs and a CHANGELOG entry.
- **Findings:**
  1. [medium] Design: this adds a new non-standard builtin and IR variant to the `wgpu_ray_query` extension. A maintainer should decide whether to accept the API and its name. HLSL calls it `CommittedRayT`, and GLSL uses `rayQueryGetIntersectionTEXT(committed)`. The alternative, a return value from `rayQueryGenerateIntersection`, was considered and rejected in the description.
  2. [medium] `naga/src/back/mod.rs:298` `bake_ref_count` returns the default of 2 for the new expression, so the HLSL and MSL writers inline it at its use site. The expression reads mutable ray-query state. If an unnamed use comes after a statement that mutates `rq` (for example a `Call` that runs `rayQueryGenerateIntersection` between emit and use), HLSL/MSL read the new value and SPIR-V reads the old one. It should bake at ref count 1, like `Load`. `RayQueryGetIntersection` has the same problem on trunk.
  3. [medium] `naga/src/back/spv/ray/query.rs` (new `write_ray_query_get_current_t_max`) The PR does not compile after [#10600](https://github.com/gfx-rs/wgpu/issues/10600) lands, because the helper signature changes (see Ordering). It also repeats the committed-`t` branch from the GenerateIntersection helper and could share it.
  4. [low] Init-tracking semantics: Vecvec says the PR follows the [#10735](https://github.com/gfx-rs/wgpu/issues/10735) model, not the current one. Someone must confirm that the "returns 0.0 if not initialized" contract in `wgpu/src/documentation/extensions/ray_tracing.rs` and `wgpu-types/src/shader.rs` is the intended one. The author has not replied.
  5. [low] The GPU test `single_hit` generates a second hit after a commit. On Vulkan this needs the fix from [#10698](https://github.com/gfx-rs/wgpu/issues/10698), which is on trunk.
- **To land:** a maintainer decision on the API; a rebase; adaptation to [#10600](https://github.com/gfx-rs/wgpu/issues/10600) if that lands first; baking of the expression (or a reasoned answer on why it is not needed); an answer to Vecvec's init-tracking note.
- **Move-along action:** decide whether `rayQueryGetCurrentTMax` is an API we want in the `wgpu_ray_query` extension (and ping jimblandy, who is assigned)
- **Depth:** targeted (I read the IR, wgsl-in, validator, the spv/hlsl/msl lowering, the tests and the docs. I did not read the generated snapshot files line by line.)

**Deeper review:** see the block under [#10600](https://github.com/gfx-rs/wgpu/issues/10600).

#### [#10800](https://github.com/gfx-rs/wgpu/issues/10800): Fix [#10736](https://github.com/gfx-rs/wgpu/issues/10736) by removing the wg_memory_sizes logic ([@inner-daemons](https://github.com/inner-daemons))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (to answer matthargett's A10X regression report and proposed patch)
- **Last activity:** 2026-11-06, matthargett, linked a tested "aggregate-allocation" patch ([rebeckerspecialties/wgpu@611e7f4](https://github.com/rebeckerspecialties/wgpu/commit/611e7f4)) that passes on A10X/A12/A13/Watch. On 2026-11-05 matthargett reported an A10X/tvOS 26.6 regression: `MTLCompilerService` crashes at pipeline creation with this PR and passes on the parent. Neither the author nor a maintainer has replied. 0 days.
- **Approvals:** none
- **CI:** red, not caused by the code, but the relevant jobs did not run. All 10 macOS jobs (Test Mac, API Validation/Operation Mac, Clippy Mac/iOS/visionOS, MSRV Mac) were cancelled because of the "scheduled macOS 14 runner image brownout". This PR changes only MSL and the Metal HAL, so the important CI has not run. A rebase is needed so that CI uses the newer runner.
- **Summary:** The MSL backend declares workgroup variables as `threadgroup` locals in the entry point (or in the task/mesh wrapper) instead of as `[[threadgroup(n)]]` entry-point arguments. The Metal HAL drops `wg_memory_sizes` and the `setThreadgroupMemoryLength`/`setObjectThreadgroupMemoryLength` calls. Snapshots and a CHANGELOG entry are updated.
- **Findings:**
  1. [high] matthargett reports a hardware regression: the A10X Metal compiler crashes on the reduced WGSL in the 2026-11-05 comment. The parent commit passes. A proposed fix exists (one struct for all static workgroup globals, with aliased fields) that includes GPU and MSL snapshot coverage. The author must adopt it, or answer it.
  2. [medium] The PR includes the unrelated commit `13eacf13` ("ci: Update Vulkan SDK to 1.4.363"), which is [#10801](https://github.com/gfx-rs/wgpu/issues/10801) (still open). The description says [#10801](https://github.com/gfx-rs/wgpu/issues/10801) is probably needed first. Land [#10801](https://github.com/gfx-rs/wgpu/issues/10801) first, or drop the commit.
  3. [medium] There is no new GPU test for several workgroup variables, workgroup atomics and padded vector types on Metal. The existing coverage passed on the old path, and Mac CI did not run on this head.
  4. [low] `CHANGELOG.md` The entry is under naga only. It also changes the Metal HAL behavior: no `setThreadgroupMemoryLength`. This matters to users who feed external MSL through the passthrough path and rely on threadgroup arguments. Check whether a Metal note is needed.
- **To land:** a rebase (to get Mac CI); a decision on [#10801](https://github.com/gfx-rs/wgpu/issues/10801) ordering; a resolution of the A10X regression (probably by adopting matthargett's aggregate struct patch); Metal GPU coverage for mixed workgroup layouts.
- **Move-along action:** ping author (ask them to respond to matthargett's A10X report and patch, and to rebase so Mac CI runs)
- **Depth:** targeted (I read all non-snapshot code. I did not reproduce or inspect matthargett's patch.)

**Deeper review ([#10800](https://github.com/gfx-rs/wgpu/issues/10800): Fix [#10736](https://github.com/gfx-rs/wgpu/issues/10736) by removing the wg_memory_sizes logic):**

> - **Open questions from the first pass:**
>   1. Is the A10X compiler crash plausible from the code?
>   2. Is the linked fix `611e7f4` sound?
>   3. Does [#10800](https://github.com/gfx-rs/wgpu/issues/10800) fully cover [#10673](https://github.com/gfx-rs/wgpu/issues/10673)?
> - **Answers:**
>   1. [confirmed plausible; root cause unknown] For this shader, the only MSL difference between the parent and [#10800](https://github.com/gfx-rs/wgpu/issues/10800) is the declaration form. Each used workgroup global changes from a `threadgroup T& x [[threadgroup(n)]]` argument to a `threadgroup T x;` local in the kernel (`naga/src/back/msl/writer.rs:8354-8369` at the PR head). Both forms are valid MSL. A crash in `MTLCompilerService` is therefore an Apple compiler bug that the new form triggers, and not invalid output.
>
>      matthargett's data fits a layout bug with several static threadgroup objects that include `packed_uint3` and vec3 types:
>      - `@size(16)` on `Padded.xyz` avoids the crash, and so does `vec3` -> `vec4`.
>      - The old path rounded each allocation to 16 bytes (`next_multiple_of(16)` in the removed `command.rs` code).
>
>      I cannot reproduce it. It is A10X/tvOS only, and A12, A13 and M-series pass.
>   2. [confirmed sound as a workaround, by reading] `611e7f4` adds `write_workgroup_variable_declarations` (`naga/src/back/msl/writer.rs`, new fn near line 1042). It declares one function-local `struct nagaWorkgroupMemoryType { T0 a; T1 b; ... }` and one `threadgroup` instance of it. Then it binds `threadgroup Ti& name = mem.name;` references. The task/mesh wrapper in `mesh_shader.rs` uses the same function.
>
>      Checked:
>      - Local struct types are legal C++/MSL.
>      - Workgroup references never get `const`, because `needs_access_qualifier` is false for `WorkGroup` (`writer.rs:744` on trunk). So zero-init through the references compiles.
>      - Names come from `namer.call`.
>      - Workgroup globals have no binding, so the removed `else if WorkGroup` branch cannot fall into the binding branch.
>      - It adds a GPU test (`tests/tests/wgpu-gpu/shader/workgroup_memory_layout.rs`) and a `workgroup-memory-layout` snapshot.
>
>      Caveats:
>      - It changes MSL for every device to avoid a bug on one old GPU, and it has no root cause.
>      - The struct can add padding between members, so the total threadgroup size can be slightly larger than the sum that wgpu-core validates. At the limit, this could push a shader over the Metal threadgroup memory limit. This is not checked.
>      - CI cannot cover A10X.
>   3. [refuted] [#10800](https://github.com/gfx-rs/wgpu/issues/10800) does not fully cover [#10673](https://github.com/gfx-rs/wgpu/issues/10673).
>      - [#10673](https://github.com/gfx-rs/wgpu/issues/10673) lets callers of MSL or metallib passthrough give `workgroup_memory_sizes` for kernels that take `threadgroup T* x [[threadgroup(n)]]` arguments. Its test `tests/tests/wgpu-gpu/passthrough/shared_memory.metal` uses exactly two such arguments.
>      - [#10800](https://github.com/gfx-rs/wgpu/issues/10800) removes all `setThreadgroupMemoryLength` calls. It only makes naga-generated MSL independent of them.
>      - Passthrough MSL that uses threadgroup arguments (hand-written, older naga output, other compilers) still gets 0 bytes after [#10800](https://github.com/gfx-rs/wgpu/issues/10800), as on trunk.
>
>      It covers the case inner-daemons expected ("I doubt you are using non-naga generated shaders with dynamic memory"), but lexoliu has not confirmed it.
> - **New findings:**
>   1. [low] There is a stale comment at `naga/src/back/msl/writer.rs:8355-8357` (PR head): "So we receive them as an input reference here", but the code now declares them. The linked fix removes this branch.
>   2. [low] The CHANGELOG (naga section only) should say that passthrough MSL with `[[threadgroup(n)]]` arguments is still not supported.
> - **Verdict change:** none ("Needs a bit of work"). The A10X item now has a concrete, reviewed fix: adopt `611e7f4` (or an equivalent). It is reviewed by reading only, and A10X is not on CI.
> - **Move-along action change:**
>   - Ask inner-daemons to adopt or squash `611e7f4` (with credit to matthargett).
>   - Ask them to rebase so the Mac CI runs.
>   - Ask them to state in the CHANGELOG and the PR description that passthrough shaders with threadgroup arguments are out of scope. Then ask lexoliu whether that is acceptable before [#10673](https://github.com/gfx-rs/wgpu/issues/10673) is closed.

#### [#10873](https://github.com/gfx-rs/wgpu/issues/10873): Plumbing for external textures in SPIR-V backend ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-17.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviewer assigned or requested), then author (rebase).
- **Last activity:** 2026-11-07, andyleiserson, force push. 6 days.
- **Approvals:** none
- **CI:** green; has conflicts (`naga/src/back/spv/writer.rs`, a small textual conflict next to the HitObject helpers from [#10311](https://github.com/gfx-rs/wgpu/issues/10311))
- **Summary:** Part of the external texture work for Vulkan ([#4386](https://github.com/gfx-rs/wgpu/issues/4386), Firefox bug 2008705). One WGSL `texture_external` global becomes four SPIR-V globals (three planes and a params uniform buffer at bindings +0..+3, as the Vulkan HAL lays them out), and one function argument becomes four SPIR-V parameters. Commit 1 refactors binding lookup into `Writer::resolve_global` with a `ResolveGlobal` enum and moves the binding-array size override there. Commit 2 adds the external texture variants, declarations, prelude loads (params are loaded and converted from the std140 type), and call-argument expansion through a new `Function::parameter_offsets`. Image expressions on external textures are not supported yet.
- **Findings:**
  1. [low] `naga/tests/naga/snapshots.rs:26` The temporary `allowed_capabilities |= TEXTURE_EXTERNAL` runs after the GLSL and HLSL `&=`, so it also allows the capability for snapshots that target GLSL. It is disclosed as a temporary hack. A `TODO` with an issue link would help track removal.
  2. [low] `naga/src/back/spv/writer.rs:584` `assert!(b.external_texture)` panics if a binding map entry for an external texture global does not set `external_texture`. The Vulkan HAL sets it (`wgpu-hal/src/vulkan/device.rs:1537`), so wgpu is safe. A standalone naga user with a wrong binding map gets a panic, not an `Error`. Consider returning an error.
  3. [low] `naga/src/back/spv/mod.rs:408` External image expressions still reach `unreachable!` (before: `unimplemented!`). This is not reachable through wgpu: Vulkan does not advertise `Features::EXTERNAL_TEXTURE` and `spv::supported_capabilities()` does not include `TEXTURE_EXTERNAL`. (Reasoned from code.)
  4. [low] No CHANGELOG entry. This is acceptable, because the feature is not usable yet. The CHANGELOG box is unchecked. The PR template has no agent-disclosure lines. The "LLM Use?" section says an LLM generated some code and did refactoring.
- **To land:** assign a naga/spv-out reviewer; rebase on trunk (small `writer.rs` conflict); optionally address findings 1 and 2.
- **Move-along action:** assign a reviewer (for example, inner-daemons or the maintainer).
- **Depth:** full for `naga/src`; I skimmed the new snapshot outputs and relied on CI `spirv-val` for their validity.

#### [#10891](https://github.com/gfx-rs/wgpu/issues/10891): [naga wgsl-in] Enforce limit of 2047 elements in array constructor ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (decide if the new limit is acceptable for native users), then author (CHANGELOG wording and conflict).
- **Last activity:** 2026-11-13, sagudev, force push. 0 days.
- **Approvals:** inner-daemons (11-09, before the 11-13 force push)
- **CI:** green; has conflicts (`CHANGELOG.md` only)
- **Summary:** Adds an error in the WGSL front end when an array value constructor (`array(...)` or `array<T, N>(...)`) has more than 2047 components. The check runs in `Lowerer::construct` before the components are lowered, so it covers `const`, `override` and runtime constructors. The motivation is issue [#10877](https://github.com/gfx-rs/wgpu/issues/10877): a `const` array with more than 65532 elements makes the SPIR-V backend emit an `OpConstantComposite` that is too long (it now panics in `Instruction::to_words`).
- **Findings:**
  1. [medium] `CHANGELOG.md:303` The entry says "as required by the WGSL specification". This is not correct. I checked the spec (https://www.w3.org/TR/WGSL/#limits, section 2.4): 2047 is a "minimum supported value". The spec says an implementation "may support shaders that go beyond the specified limits", and the error is only a non-normative "should" note. Thus the PR is a policy choice, not a spec fix. Reword the entry, and consider listing it as a behavior change rather than a bug fix. (Reasoned from the spec text and the diff.)
  2. [medium] `naga/src/front/wgsl/lower/construction.rs:129` Behavior change for native users: shaders with large `const` lookup tables (2048 to about 65532 elements) compiled and ran before this PR. Now they fail to parse. The issue reporter had such a generator. The maintainer must decide if this break is acceptable or if the limit should be higher (the description already suggests that a higher limit is possible later). (Reasoned from code; I did not run a shader.)
  3. [low] The PR says "Fixes [#10877](https://github.com/gfx-rs/wgpu/issues/10877)", but the SPIR-V backend panic in [#10877](https://github.com/gfx-rs/wgpu/issues/10877) stays reachable through other front ends (SPIR-V in, GLSL in) and the IR validator does not check this limit. Consider "Part of [#10877](https://github.com/gfx-rs/wgpu/issues/10877)", or keep the issue open for an spv-out error instead of the `to_words` panic.
  4. [low] Test `array_constructor_element_limit` covers typed and inferred `array` at 2047 and 2048 in a function `let`. It does not cover a module-scope `const` (the case in the issue). This is optional, because the same `construct` path is used.
  5. [low] PR template: the agent-disclosure boxes are still inside the HTML comment (not filled). The "LLM Use?" section says GPT-6 Luna wrote the first version and the author polished it by hand.
- **To land:** maintainer decides that the 2047 limit is acceptable; fix the CHANGELOG wording; rebase to resolve the CHANGELOG conflict.
- **Move-along action:** decide whether naga may reject constructors above 2047 elements on native, then ask sagudev to fix the CHANGELOG wording.
- **Depth:** full

#### [#10901](https://github.com/gfx-rs/wgpu/issues/10901): Naga coop store emit ([@Sandvoxel](https://github.com/Sandvoxel))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-20.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (stride literal type, finding 1), then a member approval. inner-daemons approved on 2026-11-20, after the last commit, but GitHub `reviewDecision` is `REVIEW_REQUIRED`. cwfitzgerald and inner-daemons are assigned.
- **Last activity:** 2026-11-20, inner-daemons, approval. Last author activity: 2026-11-17, Sandvoxel, commit that uses `require_all`.
- **Approvals:** inner-daemons (2026-11-13, 2026-11-16 and 2026-11-20; read permission, so GitHub does not count them)
- **CI:** green (66 pass, 2 neutral, 1 skipped).
- **Summary:** Fixes two cooperative-matrix bugs in the SPIR-V path. (1) The WGSL front end pushed `Statement::CooperativeStore` without first flushing the emitter, so a non-constant stride was emitted after the store and spv-out panicked ("Expression [N] is not cached"). The fix flushes the emitter like `textureStore`. (2) A cooperative matrix type that appears only as an expression result is compacted out of the type arena, so spv-out did not request `SPV_KHR_cooperative_matrix` or the Vulkan memory model. The new `Writer::require_cooperative_matrix()` (`require_all` of `CooperativeMatrixKHR` and `VulkanMemoryModel`) is called from the type path and from the `CooperativeLoad` and `CooperativeMultiplyAdd` paths. The validator now checks that the store pointer and stride are in scope and that the stride is a `u32` scalar, and `CooperativeMultiplyAdd` reports the failing operand instead of always `a`. Adds a snapshot case, 4 validator tests and a GPU test.
- **Findings:**
  1. [medium] `naga/src/valid/function.rs:1769-1775` with `naga/src/front/wgsl/lower/mod.rs` (coopStore lowering): the WGSL front end lowers an explicit stride with `self.expression(...)`, which concretizes an abstract integer to `i32`. So `coopStore(m, &buf[0], 16)` (no `u` suffix) gets an `i32` stride, and the new check rejects it with `InvalidCooperativeDataStride`. The extension documentation (`wgpu/src/documentation/extensions/cooperative_matrices.rs`) gives the stride as `u32`, so the front end should convert an abstract stride to `u32`. All tests use `u32` values (`16u` or `u32` variables), so CI does not show this. (From code.)
  2. [low] `naga/src/valid/expression.rs:1447` `E::CooperativeLoad` checks the pointer but not the stride type. After finding 1, `coopLoad(p, 16)` and `coopStore(m, p, 16)` behave differently. Add the same `u32` check for loads.
  3. [low] `naga/src/back/spv/block.rs:2227`, `:2269` The capability request is added at each cooperative expression. A more general fix is to request capabilities when a local cooperative-matrix type ID is created. [#10596](https://github.com/gfx-rs/wgpu/issues/10596) fixes the same bug in a different way (see Dependencies).
  4. [low] `naga/src/valid/expression.rs:1468` The "report the failing operand" fix duplicates the same fix in [#10414](https://github.com/gfx-rs/wgpu/issues/10414) and gives a textual conflict with it (see Dependencies).
  5. [low] The description notes that Vulkan needs the stride to be a multiple of 16 bytes, and nothing validates this. The GPU test stays legal. This is a known follow-up, not a defect of this PR.
  6. [low] PR template: the agent-disclosure lines are not present. "LLM Use?" says Claude found the bugs and wrote the fixes and tests, and the author reviewed them.
- **To land:** convert an abstract stride to `u32` in the WGSL front end for `coopLoad`/`coopStore` (and add a test with a plain literal stride); optionally check the load stride type; a member approval. Coordinate with [#10596](https://github.com/gfx-rs/wgpu/issues/10596) and [#10414](https://github.com/gfx-rs/wgpu/issues/10414). Release (in about one week): can land before it if the stride fix comes soon.
- **Move-along action:** ask Sandvoxel to make abstract-integer strides `u32` (finding 1); after that, approve.
- **Depth:** full for the new commits (validator, `require_all`, validation tests). I compared the old and new heads with `git range-diff`: the first three commits are the same except for the CHANGELOG rebase. I did not run the new tests.

#### [#10957](https://github.com/gfx-rs/wgpu/issues/10957): [naga] Implement bitcast in constant evaluator ([@szymros](https://github.com/szymros))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-20.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (szymros): andyleiserson's comments of 2026-11-20 and the findings below. andyleiserson asks for a test that a const compose bitcast with mixed widths (for example `(u32, u16, u16)` to `vec4<u16>`) is rejected, for only the bitcast overloads in the spec, for `flatten_compose`, for a type handle (not the expression) in the error, and for the related CTS tests in `cts_runner/test.lst`. Then a review by cwfitzgerald (assigned). GitHub `reviewDecision` is `REVIEW_REQUIRED`.
- **Last activity:** 2026-11-20, andyleiserson, partial review with comments. Last author activity: 2026-11-15, szymros, commit "Add more tests, adjust error types".
- **Approvals:** none
- **CI:** green (65 pass, 2 neutral, 1 skipped).
- **Summary:** Implements const evaluation of `Expression::Bitcast`. `expression_to_bytes` flattens a literal or `Compose` to little-endian bytes, and `literal_from_bytes` builds the target scalars from them, so `vec2<f16>` <-> 32-bit and `vec4<f16>` <-> `vec2` of 32-bit casts follow the WGSL internal layout. A size mismatch returns a new `InvalidBitcastArgs` error. Bitcasts with const arguments in function bodies are now folded too. The `bitcast-f16.wgsl` runtime cases now use `let` values, so they stay runtime and keep their backend coverage. Part of [#7700](https://github.com/gfx-rs/wgpu/issues/7700).
- **Findings:**
  1. [medium] Behavior change with no test and no changelog entry: a const bitcast whose result is a NaN or infinity (for example `bitcast<f32>(0x7fc00000u)`, also inside a function body) now fails with the `check_literal_value` error from `register_evaluated_expr`. Before, it was emitted as a runtime bitcast. This is correct per WGSL ([Finite Math Assumption](https://gpuweb.github.io/gpuweb/wgsl/#floating-point-evaluation): infinities and NaNs made before shader execution are errors), but users can see it. Add a `wgsl_errors` test for it and for the f16 case, and a changelog entry. (From code.)
  2. [low] `naga/src/proc/constant_evaluator.rs:2808` `InvalidBitcastArgs { from: format!("{:?}", expr), ... }` formats the expression handle (for example `[12]`), not its type, and `to` is the `TypeInner` debug output. The error is also built before it is needed. From WGSL it is not reachable because the front end checks sizes, so a plainer message is enough.
  3. [low] `naga/src/proc/constant_evaluator.rs:2719`, `:2759`, `:2799` `expression_to_bytes`, `literal_from_bytes` and `bitcast` are `pub` on the public `naga::proc::ConstantEvaluator`. This adds public API by accident; make them private. `literal_from_bytes` does not need `&mut self`.
  4. [low] `cts_runner/fail.lst:71` lists `webgpu:shader,validation,expression,call,builtin,bitcast:*` with the note "bitcast const-eval unimplemented". The PR does not update it. Someone should run that CTS selector and update the list.
  5. [low] `naga/tests/in/wgsl/bitcast-f16.wgsl` uses `let u = 4294967295;`. In WGSL a `let` with an AbstractInt initializer is concretized to `i32`, so this value overflows and should be an error. Naga accepts it and passes the abstract value to `bitcast<u32>` (the snapshot shows `bitcast<u32>(4294967295u)`). The test depends on a naga deviation; use `4294967295u`. (From the snapshot.)
  6. [low] No changelog entry. PR template: the agent-disclosure lines are not present. "LLM Use?" says a Claude chatbot helped to understand the codebase and no code is AI-generated.
- **To land:** answer andyleiserson; add error tests for NaN/infinity results and for mixed-width compose bitcasts; add a changelog entry; make the new methods private and fix the error message; enable the related CTS tests. An agent can do most of this. Release (in about one week): can land before it if the fixes come soon.
- **Move-along action:** have an agent (or szymros) address andyleiserson's comments and the error tests, then review.
- **Depth:** full for `constant_evaluator.rs`, the test inputs and the WGSL snapshots. I checked the f16 layout values in the snapshot by hand (`vec2<f16>(1.0)` -> `0x3C003C00` = 1006648320). I skimmed the HLSL, MSL, GLSL and SPIR-V snapshots.

#### [#10961](https://github.com/gfx-rs/wgpu/issues/10961): Fix bitcasting packed vec3 struct field for MSL ([@IsseW](https://github.com/IsseW))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI red (1 failed: Format & Typos); last update 2026-11-15.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (typos CI failure), and reviewers (inner-daemons and andyleiserson, assigned 2026-11-18). GitHub `reviewDecision` is `REVIEW_REQUIRED`.
- **Last activity:** 2026-11-18, cwfitzgerald, assigned inner-daemons and andyleiserson. Last author activity: 2026-11-15, IsseW, opened the PR.
- **Approvals:** none
- **CI:** red (1 failed): "Format & Typos". The PR causes it: `typos` reads the user name in the new CHANGELOG line (`By @IsseW`, `CHANGELOG.md:315`) as a misspelling of "Issue". Fix: add the name to the "Usernames" list in `typos.toml` (as for `Healthire`), or change the entry.
- **Summary:** In MSL, a `vec3` struct member that is followed by a scalar is emitted as `packed_float3`. `as_type<uint3>(s.v)` then has a 12-byte source and a 16-byte target, and Metal rejects it. The fix wraps the bitcast operand with `put_wrapped_expression_for_packed_vec3_access`, so it becomes `as_type<metal::uint3>(metal::float3(s.v))`, as for swizzles and matrix multiplies. Adds an MSL snapshot test and a changelog entry.
- **Findings:**
  1. [low] CI typos failure (see CI).
  2. [low] `CHANGELOG.md:315` The entry text is missing a word: "fix generated for `bitcast`" should be "fix the generated code for `bitcast`".
  3. [low] The description says the test was added to `packed-vec3-bitcast.wgsl`. The PR adds a new file, `packed-vec3-struct-field-bitcast.wgsl`.
  4. [low] PR template: the agent-disclosure lines are not present. "LLM Use?" says an LLM made the minimal repro from the author's shader.
- **To land:** fix the typos failure; fix the changelog wording; a member review. Release (in about one week): can land before it.
- **Move-along action:** ask IsseW to add the user name to `typos.toml`; ask inner-daemons or andyleiserson to review.
- **Depth:** full. I checked `get_packed_vec_kind`, which finds a packed member through `AccessIndex` on a struct value or pointer. I did not check whether `As` (`static_cast`) of a packed member has the same problem.

#### [#10980](https://github.com/gfx-rs/wgpu/issues/10980): fix(naga): allow `continuing` block to shadow loop body declarations ([@ekarad1um](https://github.com/ekarad1um))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Land as is
- **Waiting on:** member approval. inner-daemons approved on 2026-11-18, but GitHub `reviewDecision` is `REVIEW_REQUIRED`. teoxoy is assigned.
- **Last activity:** 2026-11-18, ekarad1um, force push that removes the `wgsl_errors` test, as inner-daemons suggested in his nit. Before that, 2026-11-18, inner-daemons, approval ("LGTM with one nit").
- **Approvals:** inner-daemons (2026-11-18, before the last force push)
- **CI:** green (66 pass, 2 neutral, 1 skipped).
- **Summary:** The WGSL parser did not push a scope for the `continuing` block, so `loop { let x = a; continuing { let x = b; } }` was rejected as a redefinition. The WGSL grammar makes `continuing_compound_statement` its own compound statement in the loop body, and Tint accepts this. The fix adds `push_scope`/`pop_scope` around the continuing block, including `break if`. Adds a `continuingLexicalScope` case to the `lexical-scopes` snapshot and a changelog entry. This is item 4 of the closed [#10648](https://github.com/gfx-rs/wgpu/issues/10648).
- **Findings:**
  1. [low] `naga/tests/in/wgsl/lexical-scopes.wgsl:56` The snapshot output (`wgsl-lexical-scopes.wgsl`) folds the body away, so it shows only that the shader parses. The input checks the important cases (the continuing block reads the body's `a`, shadows it, and `break if` sees the shadowed `bool`), because a wrong scope gives a type error. No change needed.
  2. [low] PR template: the agent-disclosure lines are left in the HTML comment and not filled in. "LLM Use?" says Claude helped with the investigation, fixes, tests and checks, and the author supervised and reviewed.
- **To land:** a member approval (the code is the same as what inner-daemons approved, less one test). Release (in about one week): can land before it.
- **Move-along action:** approve and queue (or ask teoxoy, who is assigned).
- **Depth:** full. I read the parser's `loop` function: the body scope is pushed before the loop and popped after it, so the new scope nests correctly.

#### [#10982](https://github.com/gfx-rs/wgpu/issues/10982): [naga] Implement wgpu_f64 enable extension ([@stefnotch](https://github.com/stefnotch))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (answers to the author's questions), then author or an agent (changelog entry, docs, tests). cwfitzgerald and inner-daemons are assigned; review requested from inner-daemons. GitHub `reviewDecision` is `REVIEW_REQUIRED`.
- **Last activity:** 2026-11-18, stefnotch, opened the PR with questions for maintainers.
- **Approvals:** none
- **CI:** green (65 pass, 2 neutral, 1 skipped).
- **Summary:** Adds the WGSL enable extension `wgpu_f64` (mapped to `Capabilities::FLOAT64`). The `f64` type and the `lf` literal suffix need `enable wgpu_f64;`. The WGSL backend writes the directive when the type arena has an `f64` scalar, vector or matrix. Existing f64 test shaders get the directive. The PR also adds `spv::Capability::Float64` to the Vulkan list of available SPIR-V capabilities when `Features::SHADER_F64` is on. Fixes [#8975](https://github.com/gfx-rs/wgpu/issues/8975).
- **Findings:**
  1. [medium] Breaking change for `wgpu` users: every WGSL shader that uses `f64` fails until it adds `enable wgpu_f64;`. cwfitzgerald accepts the break. There is no changelog entry (the author lists it as missing), and the `Features::SHADER_F64` docs (`wgpu-types/src/features.rs:~780`) do not say that the directive is required, as `SHADER_I16` does for `wgpu_int16`.
  2. [medium] No negative test: nothing checks that `f64` or `1lf` without the directive is an error, or that `enable wgpu_f64;` without `Capabilities::FLOAT64` is an error. The author offers to add them. Add them to `wgsl_errors.rs`, as for the other enable extensions.
  3. [low] `wgpu-hal/src/vulkan/adapter.rs:2847` The new `Float64` entry is not mentioned in the description. On trunk the SPIR-V writer inserts `Float64` directly into `capabilities_used` (`spv/writer.rs:2077`) and does not call `require_any`, so f64 works on Vulkan without it. The line is harmless but is unrelated scope; ask why it is there.
  4. [low] `naga/src/back/wgsl/writer.rs` The writer checks only the type arena. An `f64` value that has no arena type (for example an `f64` literal or `As` result only in a function expression, from a non-WGSL front end) gives WGSL output without the directive. `f16` has the same gap on trunk. (From code.)
  5. [low] Author questions that need a maintainer answer: why the capability is `FLOAT64` and not `SHADER_FLOAT64`; that the WGSL writer does not write `enable wgpu_ray_query;` (true on trunk: `RequiredEnabled` has no ray-query flag); whether `li`/`lu` literals need an extension (there is no 64-bit integer enable extension on trunk); how to update inline snapshot tests.
  6. [low] PR template: the agent-disclosure lines are not present. "LLM Use?" says "Written lovingly by hand." The changelog and tests boxes are unchecked, and the author says that docs and a changelog entry are still missing.
- **To land:** answers to the questions; a changelog entry (breaking change); the `SHADER_F64` docs line and an extension docs page if wanted; negative tests; an explanation or removal of the Vulkan capability line. An agent can make the changelog entry, docs and tests. Release (in about one week): can land before it.
- **Move-along action:** answer the author's questions and say that the break is accepted (inner-daemons has the review request).
- **Depth:** full. I checked every site that `wgpu_int16` touches for a matching `wgpu_f64` change (parser, `map_predeclared_type`, number suffix, capability mapping, WGSL writer).

#### [#10990](https://github.com/gfx-rs/wgpu/issues/10990): [naga spv-out] Initialize workgroup arrays with distributed loops ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-20.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer decision: inner-daemons asks to hold this PR until a shared naga zero-initialization pass ([#10456](https://github.com/gfx-rs/wgpu/issues/10456), on top of [#10409](https://github.com/gfx-rs/wgpu/issues/10409)) is designed, so that each backend does not have its own copy of this logic. Then a member review. Maintainer policy decision on agent authorship.
- **Last activity:** 2026-11-20 (UTC), inner-daemons, comment: does not want per-backend zero-init logic (also raised on [#10314](https://github.com/gfx-rs/wgpu/issues/10314)); asks to stop work on this PR until the shared pass is decided; plans to work on it "in the next week or so".
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** In `ZeroInitializeWorkgroupMemoryMode::Polyfill`, the SPIR-V backend zeroes each workgroup array element in a loop that starts at the local invocation index and steps by the workgroup invocation count. Nested arrays become one flat loop with `UDiv`/`UMod` index math; structs that contain arrays are walked member by member; invocation 0 stores the values without arrays. One workgroup barrier follows. This is the SPIR-V version of [#10314](https://github.com/gfx-rs/wgpu/issues/10314) (HLSL), part of [#4592](https://github.com/gfx-rs/wgpu/issues/4592). Motivation: on an Oculus Quest 1 (Adreno 540, Vulkan 1.1, no `VK_KHR_zero_initialize_workgroup_memory`), the old single `OpConstantNull` store of an `array<atomic<u32>, 3584>` took 328 s to compile; the author reports 0.06 s with this PR. Snapshot churn is most of the 36 files; the PR adds SPIR-V snapshots for `workgroup-init-compute.wgsl`.
- **Findings:**
  1. [medium] Design conflict, not a code bug: inner-daemons (collaborator) objects to per-backend zero-init loops and wants a shared IR pass ([#10456](https://github.com/gfx-rs/wgpu/issues/10456), [#10409](https://github.com/gfx-rs/wgpu/issues/10409)). The code here duplicates the HLSL logic from [#10314](https://github.com/gfx-rs/wgpu/issues/10314) in a different form (`naga/src/back/spv/writer.rs:35-91`, `:3083-3248`). A maintainer must decide: land now as a fix for Adreno 5xx and devices without the extension, or wait for the shared pass.
  2. [medium] `naga/src/back/spv/writer.rs:3051-3248` The polyfill path has no runtime test on CI, from code. On Vulkan, `Polyfill` is used only when the device has neither Vulkan 1.3 nor `VK_KHR_zero_initialize_workgroup_memory` (`wgpu-hal/src/vulkan/adapter.rs:2970-2977`), or for mesh and task shaders on AMD proprietary drivers (`POLYFILL_MESH_TASK_WORKGROUP_ZERO_INIT`, `wgpu-hal/src/vulkan/adapter.rs:2378-2379`). Lavapipe is Vulkan 1.3, so the GPU test `ZERO_INIT_WORKGROUP_MEMORY_PARALLEL` (which runs `workgroup-init-compute.wgsl`) takes the native path. The new code is checked only by snapshots and `spirv-val`, and by the author's runs on the Quest. This is a trusted-translator path for untrusted shaders: a skipped element leaks old workgroup memory. A naga or hal test option that forces `Polyfill` on a CI Vulkan device and runs the existing self-checking shader would close this.
  3. [low] Generated code is correct, from code and from the `wgsl-workgroup-var-init.spvasm` snapshot. Loop header has `OpPhi` first and `OpLoopMerge` before the conditional branch; the `OpPhi` incoming value comes from the block before the loop (the entry block or the previous loop's merge block); the barrier is after all loops and after the invocation-0 `if`, in uniform control flow. The stride is the product of `entry_point.workgroup_size`, the same value written to `LocalSize` (`writer.rs:2007-2034`), and pipeline overrides are resolved before the backend runs, so every index in `[0, count)` is stored by exactly one invocation. Index math for `array<struct{array}>` paths is correct (stride divides out each `Element`, `UMod` on all but the outermost). Atomics are zeroed with `OpStore`, as on trunk.
  4. [low] `naga/src/back/spv/writer.rs:3051-3056`, `:3093-3102`, `:3229-3235` The overflow checks on invocation count and element count are good, but `flat_index + invocations` can wrap in 32 bits if `count + invocations > 2^32`, which gives a loop that does not end. wgpu-core limits workgroup storage and invocations before the backend runs (`wgpu-core/src/validation.rs:1867`), so `wgpu` cannot reach it. A direct naga user with extreme sizes can.
  5. [low] The mesh and task snapshots change (`wgsl-mesh-shader*.spvasm`, `wgsl-primitive-index-mesh-shader`). With the AMD workaround above, AMD proprietary drivers get the new loops in mesh and task shaders. The description does not mention this, and it is not tested on AMD.
  6. [low] `CHANGELOG.md:418` The entry has a typo ("Xiammi") and claims "Oculus Quest 1, Xiammi, and HTC phones" and a "Qualcomm shader compiler bug". The PR shows measurements only on the Quest 1. The entry should say what was measured. Trunk is moving to changelog fragments, so the changelog entry will need to move.
  7. [low] The "WebGPU implementations may be affected" box is not checked, but Firefox on Vulkan devices without the extension (Android, older drivers) uses this path.
  8. [low] Agent disclosure: "This PR was automatically created by an Agent" is checked. The "description was automatically written by an Agent" line is `- []`, which does not render as a checkbox, but "LLM Use?" says Claude Code (Opus 5.5) found the problem, wrote the code, ran the tests and drafted the description and commit message. "I self-reviewed and fully understand this PR" is checked. Same pattern as [#10979](https://github.com/gfx-rs/wgpu/issues/10979).
- **To land:** maintainer decides between this PR and the shared pass ([#10456](https://github.com/gfx-rs/wgpu/issues/10456)); if this PR goes ahead: a test that forces the polyfill on CI, fix the changelog entry, a member review. Not likely before the release, because of the open design objection.
- **Move-along action:** decide with inner-daemons if a per-backend SPIR-V polyfill can land now as a fix for devices without the extension, or if it waits for [#10456](https://github.com/gfx-rs/wgpu/issues/10456).
- **Depth:** full (snapshot outputs checked by sample: `wgsl-workgroup-var-init`; the other snapshot files were not read line by line)

## Metal

#### [#9645](https://github.com/gfx-rs/wgpu/issues/9645): [metal] synchronize acceleration structure builds using fences ([@jtbirdsell](https://github.com/jtbirdsell))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-06-23.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (jtbirdsell). On 2026-06-23, Vecvec reported "a race when splitting the encoder" and asked for a fix. There has been no author reply or push since then.
- **Last activity:** 2026-06-23, Vecvec, comment: the fix works on their phone, but a race exists when the encoder splits. 136 days ago. Draft [#10292](https://github.com/gfx-rs/wgpu/issues/10292) (2026-09-09) took over this branch.
- **Approvals:** none. Vecvec (assigned) left a COMMENTED review ("code looks good").
- **CI:** green, but old (2026-06-04, base `6e8d0bd`). A rebase is needed, and mergeability is unknown.
- **Summary:** Makes `place_acceleration_structure_barrier` split the AS encoder with `updateFence:`/`waitForFence:` on one lazily created `MTLFence` per encoder. It also splits before `read_acceleration_structure_compact_size` if the open encoder has builds. Fixes [#9215](https://github.com/gfx-rs/wgpu/issues/9215).
- **Findings:**
  1. [high] The race that Vecvec reported is not resolved or explained. This blocks the merge.
  2. [medium] No in-tree regression test. Validation used an external repo (`Vecvec/macos-ray-tracing-test`). [#10292](https://github.com/gfx-rs/wgpu/issues/10292) adds `issue_9215` GPU tests that could be cherry-picked here.
  3. [medium] `wgpu-hal/src/metal/command.rs` `read_acceleration_structure_compact_size`: the hal works around a missing barrier in wgpu-core. Vecvec's thread says this is probably a wgpu-core bug (and related to [#8825](https://github.com/gfx-rs/wgpu/issues/8825)). Decide whether to fix it in core instead, because other backends can have the same gap.
  4. [low] Ordering is only between AS encoders. Ordering from AS builds to later compute/render passes that consume the TLAS (ray query) depends on Metal's implicit hazard tracking. Vecvec's first comment raises this cross-encoder concern too.
  5. [low] `newFence().unwrap()` panics on allocation failure. Other paths in this file do the same.
- **Overlap with [#10292](https://github.com/gfx-rs/wgpu/issues/10292):** [#10292](https://github.com/gfx-rs/wgpu/issues/10292) (draft, [@acoliver](https://github.com/acoliver)) says it "Supersedes [#9645](https://github.com/gfx-rs/wgpu/issues/9645)". It contains jtbirdsell's commit plus a 2.5k-line rework: a seeded fence, retiring tail command buffers, `MAX_UNSUBMITTED_COMMAND_BUFFERS = MAX_COMMAND_BUFFERS - 1`, completion-handler poll wakeups, and per-bind-group AS usage. It has had no activity since it was opened (58 days), it has no reviewer, and it is still a draft. Its scope is much larger than the bug. In my view [#10292](https://github.com/gfx-rs/wgpu/issues/10292) does not replace [#9645](https://github.com/gfx-rs/wgpu/issues/9645) as a landable change. It is a design alternative that would need to be split.
- **To land:** rebase. Fix or explain the encoder-split race. Add a regression test (for example, port the `issue_9215` tests from [#10292](https://github.com/gfx-rs/wgpu/issues/10292)). Decide the core-vs-hal location for the compact-size barrier.
- **Move-along action:** decide which design to pursue: the small [#9645](https://github.com/gfx-rs/wgpu/issues/9645) fix or the [#10292](https://github.com/gfx-rs/wgpu/issues/10292) rework. Then ping the matching author. Recommendation: continue with [#9645](https://github.com/gfx-rs/wgpu/issues/9645) (ping jtbirdsell about the race and the rebase), and ask acoliver to split [#10292](https://github.com/gfx-rs/wgpu/issues/10292) into "tests" and "rework" PRs, or close it.
- **Depth:** full for [#9645](https://github.com/gfx-rs/wgpu/issues/9645) (68 lines). [#10292](https://github.com/gfx-rs/wgpu/issues/10292): description only.

**Deeper review ([#9645](https://github.com/gfx-rs/wgpu/issues/9645): [metal] synchronize acceleration structure builds using fences):**

> - **Open questions from the first pass:**
>   1. Does [#10292](https://github.com/gfx-rs/wgpu/issues/10292) contain this fix?
>   2. Does either PR fix the "race when splitting the encoder" that Vecvec reported?
>   3. Which PR should the maintainer pursue, and what is the smallest piece that can land?
> - **Answers:**
>   1. [confirmed] [#10292](https://github.com/gfx-rs/wgpu/issues/10292) contains this fix. `a9778b0b` is literally the first commit of [#10292](https://github.com/gfx-rs/wgpu/issues/10292), and the [#10292](https://github.com/gfx-rs/wgpu/issues/10292) head keeps the mechanism:
>      - `place_acceleration_structure_barrier` ends the open AS encoder with `updateFence` (`wgpu-hal/src/metal/command.rs:1981` at [#10292](https://github.com/gfx-rs/wgpu/issues/10292), through `end_acceleration_structure_builder(true)` at `:329`).
>      - The next AS encoder waits on the fence.
>      - The compact-size read splits a busy encoder (`:1999`, now also after AS copies).
>
>      The fence model is different:
>      - In [#9645](https://github.com/gfx-rs/wgpu/issues/9645), the fence is created lazily for each hal encoder with `newFence().unwrap()`, and the pending wait is cleared at `end_encoding` (`command.rs:562` at [#9645](https://github.com/gfx-rs/wgpu/issues/9645)).
>      - In [#10292](https://github.com/gfx-rs/wgpu/issues/10292), there is one fence for each queue (`wgpu-hal/src/metal/mod.rs:524`). It is created in fallible `create_command_encoder` (`device.rs:753`), and every AS encoder waits on it at start.
>
>      So [#10292](https://github.com/gfx-rs/wgpu/issues/10292) is a strict superset. Landing [#9645](https://github.com/gfx-rs/wgpu/issues/9645) first is compatible with the stack: a rebase of [#10292](https://github.com/gfx-rs/wgpu/issues/10292) then drops `a9778b0b`.
>   2. Reasoned from code, not run:
>      - [confirmed] [#9645](https://github.com/gfx-rs/wgpu/issues/9645) does not fix the race. [#9645](https://github.com/gfx-rs/wgpu/issues/9645) orders AS commands only inside one Metal command buffer. `end_encoding` and `discard_encoding` clear `pending_acceleration_structure_fence_wait` (`command.rs:533`, `:562`). Each pooled hal encoder also has its own fence. So a BLAS build in one wgpu `CommandEncoder` and a TLAS build in another `CommandEncoder` are not ordered. I read Vecvec's report as this case, for two reasons:
>        - Vecvec's first comment (2026-06-09) asks about "build commands ... in separate command encoders".
>        - jtbirdsell's own measurement only covered "the cases that build BLAS and TLAS in the same submission".
>
>        The [#10292](https://github.com/gfx-rs/wgpu/issues/10292) commit `9e7d8b19` says it directly: "builds in separate command buffers committed back to back were unordered".
>
>        Inside one command buffer, the split in [#9645](https://github.com/gfx-rs/wgpu/issues/9645) looks correct. wgpu-core always ends `build_acceleration_structures` with a barrier (`wgpu-core/src/command/ray_tracing.rs:444` and the end of `build_blas`, at trunk-era `6e8d0bd1`). The AS encoder is therefore closed with `updateFence` before any blit or pass that wgpu-core or the user records. The pending wait also survives `enter_blit`. I found no race inside one command buffer.
>
>        The repro repo is not readable here, so the meaning of "splitting the encoder" is still inferred.
>      - [confirmed by design, unverified on hardware] [#10292](https://github.com/gfx-rs/wgpu/issues/10292) addresses the race. The single queue fence is waited at the start of every AS encoder and updated at its end. Metal evaluates fence order in commit order, so a later command buffer's AS encoder waits for the latest committed AS update. A seed command buffer gives the first waiter an update (`mod.rs:587-600`). A tail `waitForFence` in the signal command buffer (`mod.rs:~628`) makes queue completion include AS work. Tests `blas_and_tlas_in_separate_command_buffers` and `compacted_blas_in_pending_writes` in `tests/tests/wgpu-gpu/regression/issue_9215.rs` cover this case.
>
>        Caveats:
>        - It was tested on an M4 Max only. Vecvec's race reproduced only on a phone.
>        - No CI has run on [#10292](https://github.com/gfx-rs/wgpu/issues/10292).
>   3. [answer] Pursue [#9645](https://github.com/gfx-rs/wgpu/issues/9645) first, as the smallest piece that can land:
>      - Rebase it.
>      - Add the same-command-buffer tests from `ac6935b1` (`blas_and_tlas_in_same_command_buffer`, `blit_between_builds_in_same_command_buffer`, `compute_pass_between_builds_in_same_command_buffer`). That commit only touches `tests/`.
>      - Fix [#9215](https://github.com/gfx-rs/wgpu/issues/9215) for the same-submission case. Track the cross-command-buffer case as a follow-up.
>      - Do not treat [#10292](https://github.com/gfx-rs/wgpu/issues/10292) as a drop-in replacement (see [#10292](https://github.com/gfx-rs/wgpu/issues/10292)).
>
>      The earlier agent that read the diffs was right. The agent that read only the descriptions was wrong about "landable replacement", but right that [#10292](https://github.com/gfx-rs/wgpu/issues/10292) contains [#9645](https://github.com/gfx-rs/wgpu/issues/9645).
> - **New findings:**
>   1. [medium] The race Vecvec reported is a known, designed-in gap: the pending wait is cleared at `end_encoding` on purpose (jtbirdsell 2026-06-04 comment), to avoid a cross-command-buffer wait without a committed update. Land the PR with an explicit statement that it covers only the same-command-buffer case, or do not land it.
>   2. [low] A ray-query pass that consumes the TLAS has no fence wait. It relies on Metal hazard tracking. [#10292](https://github.com/gfx-rs/wgpu/issues/10292) adds waits for these passes (see below). This is unchanged from the first pass.
> - **Verdict change:** none ("Needs a bit of work"). The blocking item is now precise: document or accept the cross-command-buffer gap, rebase, and add the same-command-buffer tests.
> - **Move-along action change:** ping jtbirdsell. Ask them to (a) rebase, (b) cherry-pick the same-command-buffer tests from `ac6935b1`, and (c) scope the PR and CHANGELOG to same-command-buffer ordering. Then ask Vecvec whether the phone race is the separate-encoder case.

#### [#9679](https://github.com/gfx-rs/wgpu/issues/9679): Use Metal ICBs for multi-draw indirect count ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI red (3 failed: Test Mac aarch64, Test Windows x86_64, Test Windows x86_64 (attempt 1)); last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (rebase onto trunk, open findings), then cwfitzgerald to re-review. The 2026-11-05 "changes requested" review still stands.
- **Last activity:** 2026-11-09, matthargett, inline reply pointing to [#10890](https://github.com/gfx-rs/wgpu/issues/10890). 4 days.
- **Approvals:** none (cwfitzgerald: changes requested, 2026-11-05)
- **CI:** red, infra. "Test Mac aarch64" failed on HTTP 504 tool downloads. "Test Windows x86_64 (attempt 1)" failed in `compile_fail` because the `mach-dxcompiler-rs` build script panicked, and no retry ran. Has conflicts: a textual conflict in `wgpu-hal/src/metal/mod.rs` (confirmed with `git merge-tree`).
- **Summary:** Exposes `MULTI_DRAW_INDIRECT_COUNT` on Metal. Count draws run through a GPU-generated ICB with a GPU-clamped execution range. Pipelines that cannot run in an ICB fall back to a GPU-clamped, zero-padded copy of the arguments and a CPU loop of `max_count` indirect draws. The PR also adds `wgt::INDIRECT_BUFFER_OFFSET_ALIGNMENT` and count-buffer offset validation on all backends. The Deno changes are now removed from this PR; the diff from the merge base touches no `deno_webgpu` file.
- **Findings:**
  1. [high, interaction] Semantic conflict with [#10506](https://github.com/gfx-rs/wgpu/issues/10506) (unchanged): [#10506](https://github.com/gfx-rs/wgpu/issues/10506) appends count draws to the pass's command buffer when indirect validation is off, so the ICB range and clamped args are produced after the pass that uses them. Fix it in [#10506](https://github.com/gfx-rs/wgpu/issues/10506).
  2. [medium] `wgpu-core/src/command/render.rs` (`multi_draw_indirect_count`) Count-draw arguments are not validated. Metal clamps only the count, and Metal does not guarantee bounds-checked vertex or index fetch. So count draws through the safe `wgpu` API can read out of bounds on Metal. Reasoned from code; not run.
  3. [medium] `wgpu-hal/src/metal/icb.rs` The fallback allocates a new private buffer of `max_count * stride` bytes on every count-draw call and issues `max_count` CPU indirect draws. This is a performance cliff.
  4. [low] `wgpu-types/src/features.rs:570-584` The doc still says the feature implies that multi-draws are not emulated. The new Metal paragraph contradicts it.
  5. [low] `wgpu-hal/src/metal/adapter.rs:133` If the ICB kernels fail to compile, device creation fails with `DeviceError::Unexpected` when the feature was requested.
  6. [low] `wgpu-hal/src/metal/icb.rs:1211` The `max_count * words_per_draw` `u32` multiply can overflow with a 16 GiB or larger indirect buffer.
  7. [low] There is no validation test for the unaligned count-buffer offset error in this PR ([#10890](https://github.com/gfx-rs/wgpu/issues/10890) adds one).
  8. [low] Overlap with [#10890](https://github.com/gfx-rs/wgpu/issues/10890): both add the same alignment constant, error variant and CHANGELOG line.
- **To land:** rebase onto trunk (conflict in `wgpu-hal/src/metal/mod.rs`). Decide on count-draw argument validation on Metal. Fix the feature doc. Pool or document the fallback buffer. Guard the multiply. Rerun CI. Coordinate with [#10506](https://github.com/gfx-rs/wgpu/issues/10506) and [#10890](https://github.com/gfx-rs/wgpu/issues/10890).
- **Move-along action:** ask the author to rebase, then re-review the Deno-free diff (cwfitzgerald).
- **Depth:** targeted. I reviewed the delta (Deno removal and trunk merge) and re-checked each old finding at the current head. I did not re-review `icb.rs` or the test file, which did not change since 11-05.

#### [#9876](https://github.com/gfx-rs/wgpu/issues/9876): Query Subgroup Size on Metal ([@Vollkornaffe](https://github.com/Vollkornaffe))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-30.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (jimblandy is assigned and has not reviewed). sagudev reviewed informally ("I am not actually reviewer here"), and the author addressed their last comment in `836b42f`. The author bumped the PR on 2026-09-23.
- **Last activity:** around 2026-10-30, Vollkornaffe, pushed the head (CI ran on 10-30; the last commit is from 10-26 and addresses sagudev's DynComputePipeline thread). 7 days ago.
- **Approvals:** none
- **CI:** green
- **Summary:** Adds a public `wgpu::ComputePipeline::get_subgroup_size() -> Option<u32>`, a new hal `ComputePipeline` trait with a blanket `DynComputePipeline` impl, and a wgpu-core accessor. Metal returns `threadExecutionWidth`. All other backends return `None`.
- **Findings:**
  1. [medium] API decision needed: this is a new non-WebGPU public method on a core type, with no feature gate. It returns `None` on all backends except Metal. The related gpuweb issue ([#6278](https://github.com/gfx-rs/wgpu/issues/6278)) is still open. The maintainers must agree on the shape and name before merge.
  2. [low] `CHANGELOG.md`: the entry is under the "Metal" section, but it adds cross-backend public API (`wgpu` and the custom-backend `ComputePipelineInterface`, which is breaking for custom backends). It belongs in General/New Features.
  3. [low] `wgpu-hal/src/vulkan/mod.rs`: always returns `None`. Vulkan could report a value when `min_subgroup_size == max_subgroup_size`, or when a required size was set. This is not required, but without it the API is Metal-only in practice.
  4. [low] `wgpu-hal/src/lib.rs`: the new `pub trait ComputePipeline` has no trait-level doc. sagudev's last thread is unresolved, although the author addressed it.
- **To land:** the maintainers accept the API. Move the CHANGELOG entry. Optionally add the Vulkan value.
- **Move-along action:** decide whether `ComputePipeline::get_subgroup_size` is wanted as public API. If yes, reassign the review (jimblandy has not engaged in 3+ months) or review it yourself.
- **Depth:** full for non-test code. Skimmed the test file (146 lines).

#### [#10506](https://github.com/gfx-rs/wgpu/issues/10506): Reduce Metal command-buffer memory for multi-pass encoders ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI 7 cancelled jobs; last update 2026-11-05.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviews yet; inner-daemons is requested). The maintainer must approve the design first: the PR adds a new `wgpu_hal::Capabilities::pass_barriers_are_noop` and changes the core pass-encoding structure in wgpu-core, which Firefox uses.
- **CI:** incomplete. Most Mac jobs were cancelled at about 19:25 UTC on 2026-11-05 (API Validation/Operation Mac, Other Mac, Clippy MacOS/IOS/VisionOS aarch64, Core MSRV MacOS). The same timestamp shows on [#9679](https://github.com/gfx-rs/wgpu/issues/9679), which suggests a runner-wide cancellation, not a PR failure. "Test Mac aarch64" passed. All other jobs are green. A rerun is necessary.
- **Summary:** When the backend says that barriers and query resets are no-ops (Metal, Noop), and a conservative preflight finds no discarded-texture fixups, no indirect validation and no bundles, render and compute passes append to the current native command buffer. Without this, each pass gets its own command buffer plus a pre-pass buffer. The PR reports that this cuts about 2.3 MiB of driver memory per pass (a 97% footprint reduction in the 800-pass repro) and removes the 4096-pass limit.
- **Findings:**
  1. [high] `wgpu-core/src/command/render.rs:2305` and `:2803` In append mode, `transit.encode_deferred_multi_draws()` runs after the pass in the same MTLCommandBuffer. This breaks the HAL contract from [#9640](https://github.com/gfx-rs/wgpu/issues/9640), which is already in trunk: the deferred work must go in a command buffer that executes *before* the pass. The preflight splits `DrawIndirect` and `MultiDrawIndirectCount` only when `device.indirect_validation.is_some()`. With `InstanceFlags::VALIDATION_INDIRECT_CALL` cleared, a `multi_draw_indirect` of 512 or more draws (the trunk ICB path) runs an ICB that has not been generated yet (stale commands from the pool, or garbage). That is wrong rendering, and possibly a GPU fault through a safe API. [#9679](https://github.com/gfx-rs/wgpu/issues/9679) adds more deferred work (the ICB count range and clamped args), which makes this worse. Fix: always split on `DrawIndirect { count > 1 }` and `MultiDrawIndirectCount`, or ask the HAL whether deferred multi-draws are pending. Add a test that clears `VALIDATION_INDIRECT_CALL`.
  2. [medium] `wgpu-core/src/command/render.rs:2754`, `wgpu-core/src/command/compute.rs:907` `assert!(!append || pending_discard_init_fixups.is_empty())` panics in wgpu-core if the preflight and the real init tracking ever disagree, and both depend on untrusted input. I could not find such a mismatch: attachment Load/D3 handling, read-only depth, bind groups and bundles all match. But the invariant depends on two separate code paths staying in sync. A non-panicking fallback is safer, for example recording the fixups into a new buffer and swapping, or at minimum a `debug_assert` plus a forced split.
  3. [medium] The design adds a new HAL-wide capability, and it weakens the `transition_buffers` contract in `wgpu-hal/src/lib.rs`. The alternative is a Metal-local solution, for example reusing one MTLCommandBuffer across `begin_encoding`/`end_encoding` inside the Metal HAL. That would not touch wgpu-core's pass structure. The maintainer should decide which approach is acceptable before reviewing details.
  4. [low] `CHANGELOG.md` The entry says that passes with indirect draws split only "with indirect validation enabled". This documents the bug in finding 1.
  5. [low] The description has a truncated sentence ("I have some adjustments in"). The full-suite run reports 2 failures that the author calls unrelated (`subgroup_operations` unexpectedly passing, `written_timestamp_resolves_to_nonzero`). They should be confirmed against a clean trunk run on the same machine.
- **To land:** always split for multi-draw indirect and count draws (or query the HAL for pending deferred work) and add a test with indirect validation off. Replace the asserts with a safe fallback. Get maintainer sign-off on the HAL-capability approach. Rerun the cancelled Mac CI.
- **Depth:** targeted. I read the wgpu-core preflight, the transit/append logic and the HAL capability in full. I skimmed the tests. I did not check every query-resolve or TLAS path for ordering assumptions.

**Deeper review ([#10506](https://github.com/gfx-rs/wgpu/issues/10506): Reduce Metal command-buffer memory for multi-pass encoders):**

> - **Open questions from the first pass:**
>   1. With `VALIDATION_INDIRECT_CALL` cleared, is ICB setup for 512+ draws encoded after the pass that uses it?
>   2. Can query resolves, timestamp writes, acceleration structure builds, or clear/copy work between passes be ordered wrong now that passes share one command buffer?
>   3. Can the new `assert!`s fire on any input?
> - **Answers:**
>   1. **[confirmed]** This is a code trace. I did not run it.
>      - `wgpu-core/src/device/resource.rs:655-676`: `device.indirect_validation` is `None` when the flag is cleared. The flag is set by default (`wgpu-types/src/instance.rs:278-286`). Any user can clear it through the safe API or with `WGPU_VALIDATION_INDIRECT_CALL=0`.
>      - `wgpu-core/src/command/render.rs:2302-2306`: the preflight returns `false` for `DrawIndirect`/`MultiDrawIndirectCount`, so `append = true`. `raw_encoder` is the current open buffer (`:2341-2349`).
>      - `render.rs:3436-3450`: `multi_draw_indirect` takes the no-validation branch and calls the HAL `draw_indirect`/`draw_indexed_indirect`/`draw_mesh_tasks_indirect` with the full `count`.
>      - `wgpu-hal/src/metal/command.rs:1615-1625` (and the indexed and mesh versions): this calls `defer_multi_draw_via_icb`. At `icb.rs:626-788`, with 512 or more draws, it records `executeCommandsInBuffer` into the *current render encoder* (`icb.rs:767`) and queues generation.
>      - After the pass ends, `render.rs:2755-2757` sets `transit = encoder.open_if_closed()`, which is the same MTLCommandBuffer. `render.rs:2803` calls `encode_deferred_multi_draws`, then `icb.rs:815` `raw.computeCommandEncoder()` adds the generation kernel *after* the ended render encoder.
>      - Metal runs encoders in order, and hazard tracking orders the compute write after the render read. So the pass executes the ICB before it is filled.
>      - A new ICB is `StorageModePrivate` with undefined contents. A pooled ICB (`take_pooled`) still holds commands from an earlier submission, with index-buffer GPU addresses baked in (`icb.rs:757-765` comment) that can point to freed buffers.
>      - This is undefined behavior even when all draw arguments are *valid*, so the `VALIDATION_INDIRECT_CALL` doc ("UB if ... rules are not satisfied") does not cover it.
>      - All three families are affected: draw, indexed and mesh. With validation on, the preflight splits on any `DrawIndirect`, mesh included, so that case is safe.
>   2. Each kind of work between passes:
>      - **Query resets [refuted]:** Metal `reset_queries` is a no-op (`command.rs:870`). Transitions are no-ops (`command.rs:599-609`).
>      - **Query resolves [refuted]:**
>        - `copy_query_results` is a blit, recorded in API order (`query.rs:594-616`). The occlusion `raw_buffer` is a tracked MTLBuffer.
>        - A deferred resolve calls `close_if_open()` before it takes `insertion_point = list.len()` (`query.rs:619-620`), so the insertion point stays on a buffer boundary.
>      - **Timestamps, pass-level `timestamp_writes` [refuted]:** these are sample-buffer attachments on the pass itself.
>      - **Timestamps, encoder-level `write_timestamp` [confirmed new behavior, impact unknown]:**
>        - A timestamp written with no open encoder goes to `pending_timer_queries` (`command.rs:858-866`).
>        - On trunk, wgpu-core always called `close_if_open` before a pass, so `end_encoding` flushed the queue through a blit (`command.rs:567-573`). The pass's `begin_encoding` debug-asserts that the queue is empty (`command.rs:535`).
>        - In append mode the buffer stays open, so the next appended pass drains the queue into its own `sampleBufferAttachments`: render `command.rs:1034-1039` (start-of-vertex sample), compute `command.rs:1774-1781`.
>        - wgpu-core could not reach this path before this PR.
>        - Risks: (a) more than 4 attachments (several encoder timestamps plus the pass's own `timestamp_writes`) indexes past Metal's sample-buffer attachment array; (b) a start-of-vertex sample on a pass with no draws may never be written, but `query_set_writes` marks it as written, so the resolve copies garbage or zero.
>        - It is reachable only where both `TIMESTAMP_QUERY` (STAGE_BOUNDARIES) and `TIMESTAMP_QUERY_INSIDE_ENCODERS` (ON_BLIT_ENCODER) are exposed (`metal/adapter.rs:1349-1357`). That is a narrow set of hardware. Not tested.
>      - **Same-buffer counter resolve [unknown]:**
>        - `MTLCounterSampleBuffer` is not an `MTLResource`, so hazard tracking does not order `resolveCounters` after the sampling encoder in the same command buffer.
>        - The author's failing `resolve_query_set_init::written_timestamp_resolves_to_nonzero` is exactly this shape: a compute pass with `timestamp_writes`, then `resolve_query_set`, in one encoder.
>        - The author says the failure persists with compute coalescing disabled, which suggests it is older than this PR. A clean trunk run on the same machine is still needed.
>      - **AS builds [unknown, possible regression]:**
>        - Builds are recorded in API order (`command.rs:1897-1936`). The TLAS is bound directly and is tracked.
>        - BLASes are reached only through the TLAS and are made resident with a residency set (`command.rs:1958-1977`), not `useResource`. So Metal does not hazard-track the BLAS build against a later ray query.
>        - Before this PR, a command-buffer boundary separated the build from the pass. Now only an encoder boundary does.
>        - Mac CI is a VM without ray tracing, and the PR adds no test for this.
>      - **Clear/copy [refuted]:** blit work on tracked buffers and textures (`device.rs:496`, `:2074` leave them tracked) stays in API order in one buffer, and Metal hazard-tracks across encoders.
>      - **Discard fixups [refuted]:** fixups split the pass, as before. The divergent depth/stencil clear in `RenderPassInfo::finish` was already in the pass's buffer before this PR.
>      - **Empty indirect-validation inject [refuted]:** with an empty batcher it returns early (`indirect_validation/draw.rs:281`).
>   3. **[refuted, reasoned]** The asserts are at `render.rs:2754` and `compute.rs:907`. Fixups come only from `register_init_action` in these places:
>      - Attachments (`render.rs:1175`, `:1187`, `:1377`). The preflight covers Load, D3 and depth/stencil Load with the same aspect rules as `start()`.
>      - Bind groups (`pass.rs:227`). The preflight checks every `SetBindGroup(Some)`, which is a superset of the groups actually flushed.
>      - Bundles (`render.rs:3635`). These always split.
>
>      `has_discarded_texture` is texture-wide, which is a superset of the real range check. Discards that `start()` adds during the pass are for this pass's own attachment subresources. Overlapping them in the same pass is a usage conflict. Read-only depth leaves store as Store, and a diverging stencil discard takes the no-`discards` path. Any validation error returns before the assert. So I found no input that fires it. The invariant still depends on two code paths that must stay in sync.
> - **New findings:**
>   1. [medium] `wgpu-hal/src/metal/command.rs:1034-1039`, `:1774-1781`: encoder-level `write_timestamp` is now attached to the next appended pass's sample-buffer attachments instead of being flushed by `end_encoding` (see answer 2). Fix: flush `pending_timer_queries` when a pass starts in append mode (`leave_blit(); enter_blit()` before `begin_*_pass`), or split when timestamps are pending.
>   2. [low, unknown] BLAS build to ray query now has only an encoder boundary between them, not a command-buffer boundary (see answer 2). Ask the author to run the ray-query tests on Apple silicon, or to split after AS builds.
>   3. [low] `tests/tests/wgpu-gpu/pass_coalescing.rs` has 3 tests. None covers multi-draw indirect, indirect validation off, timestamps, queries or AS builds. A test with `VALIDATION_INDIRECT_CALL` cleared and 512 or more draws would have caught answer 1.
> - **Verdict change:** none. It stays "Needs a bit of work". Answer 1 is confirmed as undefined behavior with valid input, so I would now call it a blocker for this design.
> - **Move-along action change:** add to the list: flush pending encoder timestamps before an appended pass, and get author confirmation on the ray-query ordering.

#### [#10673](https://github.com/gfx-rs/wgpu/issues/10673): fix(metal): allocate threadgroup memory for passthrough shaders ([@lexoliu](https://github.com/lexoliu))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-01.
- **Verdict:** Needs a lot of work (superseded)
- **Waiting on:** author, who should mark it draft or close it. [#10800](https://github.com/gfx-rs/wgpu/issues/10800) (inner-daemons) says it "Closes [#10673](https://github.com/gfx-rs/wgpu/issues/10673) as obselete".
- **Last activity:** 2026-11-01, inner-daemons, comment: [#10800](https://github.com/gfx-rs/wgpu/issues/10800) makes this obsolete. Please mark it as a draft. 5 days ago. The author has not replied since 2026-10-20.
- **Approvals:** inner-daemons (2026-10-20, before they proposed the alternative)
- **CI:** green (2026-10-20)
- **Summary:** Adds `workgroup_memory_sizes` to `PassthroughShaderEntryPoint` (breaking) and passes the values to `setThreadgroupMemoryLength` for MSL/metallib passthrough.
- **Findings:**
  1. [high] It is superseded by [#10800](https://github.com/gfx-rs/wgpu/issues/10800), which removes `wg_memory_sizes` and makes naga emit static threadgroup memory. With that change, passthrough shaders need no extra reflection field. inner-daemons (the reviewer who approved) prefers [#10800](https://github.com/gfx-rs/wgpu/issues/10800).
  2. [medium] It adds a breaking public field that the reviewer called a temporary hack ("Aagh, hate this!").
  3. [low] The CHANGELOG thread (it is listed under Metal only and prefixed "BREAKING") is unresolved.
- **To land:** not intended to land if [#10800](https://github.com/gfx-rs/wgpu/issues/10800) merges.
- **Move-along action:** close (superseded by [#10800](https://github.com/gfx-rs/wgpu/issues/10800)) after [#10800](https://github.com/gfx-rs/wgpu/issues/10800) merges, with thanks. Optionally review [#10800](https://github.com/gfx-rs/wgpu/issues/10800) now to unblock that.
- **Depth:** timeline only. The code was not reviewed because the PR is superseded.

**Deeper review ([#10673](https://github.com/gfx-rs/wgpu/issues/10673): fix(metal): allocate threadgroup memory for passthrough shaders):**

> - **Open questions from the first pass:** Does [#10800](https://github.com/gfx-rs/wgpu/issues/10800) fully cover what [#10673](https://github.com/gfx-rs/wgpu/issues/10673) did?
> - **Answers:**
>   1. [refuted] No (see [#10800](https://github.com/gfx-rs/wgpu/issues/10800), answer 3). [#10800](https://github.com/gfx-rs/wgpu/issues/10800) covers lexoliu's case only if the passthrough MSL was generated by naga at or after [#10800](https://github.com/gfx-rs/wgpu/issues/10800), or uses static `threadgroup` locals. Kernels with `[[threadgroup(n)]]` arguments, as in its own `shared_memory.metal` test, still have no way to get memory.
> - **New findings:** none.
> - **Verdict change:** none ("superseded"). It is only "superseded" if the maintainer accepts that hand-written passthrough MSL with threadgroup arguments stays unsupported until the passthrough API rework that inner-daemons mentioned.
> - **Move-along action change:** before closing, ask lexoliu whether their shaders use threadgroup arguments or are naga output. Close with thanks only if naga output (or static locals) covers them.

#### [#10725](https://github.com/gfx-rs/wgpu/issues/10725): fix(metal): Probe MSL support and add version controls ([@ITJamie](https://github.com/ITJamie))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-16.
- **Verdict:** Needs a lot of work
- **Waiting on:** author and maintainers. On 2026-11-16 andyleiserson proposed: probe only if the OCLP plist exists or `WGPU_METAL_PROBE_MSL_VERSION` is set, remove the maximum-version setting, and keep `WGPU_METAL_MSL_VERSION`. He put a consistent position on probing on the agenda of the maintainers meeting of 2026-11-18.
- **Last activity:** 2026-11-16, andyleiserson, proposal for opt-in probing.
- **Approvals:** none
- **CI:** green; has conflicts (confirmed with `git merge-tree`: `CHANGELOG.md`, `docs/testing.md`, `wgpu-hal/src/metal/adapter.rs`, `wgpu-hal/src/metal/mod.rs`).
- **Summary:** At adapter enumeration, compiles a tiny kernel for each MSL version from the OS maximum downward and selects the first that compiles. Adds `wgt::MetalBackendOptions { msl_version, max_msl_version }` and the env vars `WGPU_METAL_MSL_VERSION`/`WGPU_METAL_MAX_MSL`. Gates several features on the selected MSL version. Adapters with no compiling version are excluded.
- **Findings:**
  1. [high] `wgpu-hal/src/metal/mod.rs` `expose` probes for every Metal user (opt-out). Andy asked for opt-in. osimon9's 1.6 ms measurement on one OCLP machine suggests that the cost is small. There is no measurement on stock hardware.
  2. [high] `wgpu-hal/src/metal/adapter.rs` `features()` adds new MSL gates that can remove features on supported macOS versions (`SHADER_PER_VERTEX` at 4.0, `SHADER_BARYCENTRICS` 2.2 to 2.3, and others). No rationale is given.
  3. [medium] Process: agent-authored. The description says contributor review and an author-written description are pending, but the PR is not a draft.
  4. [medium] New public API surface (`MetalShadingLanguageVersion`, `MetalBackendOptions`, `BackendOptions::metal`, two env vars). Andy asks whether it is needed.
  5. [low] `CHANGELOG.md` entries have no "By @..." attribution. Merge conflicts.
- **To land:** Andy decides on opt-in probing and the version knobs. Then the author makes probing opt-in, removes or justifies the knobs, moves the feature-gate changes to a separate PR, writes the description, and rebases.
- **Move-along action:** take the probing question to the maintainers meeting of 2026-11-18, then tell the author the result. The proposal of 2026-11-16 also answers part of [#10731](https://github.com/gfx-rs/wgpu/issues/10731) and [#10743](https://github.com/gfx-rs/wgpu/issues/10743).
- **Depth:** status only (conversation changed; code not re-reviewed)

#### [#10731](https://github.com/gfx-rs/wgpu/issues/10731): fix(metal): Work around legacy Intel compute compiler failures ([@ITJamie](https://github.com/ITJamie))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-27.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer (andyleiserson is assigned and has not reviewed).
- **Last activity:** 2026-10-28, ITJamie, merged trunk into the branch. 9 days ago. No human comments.
- **Approvals:** none
- **CI:** green
- **Summary:** On Intel adapters, probes at adapter creation whether `[[max_total_threads_per_threadgroup]]` breaks pipeline creation. If it does, naga omits the attribute and hal sets `maxTotalThreadsPerThreadgroup` on the descriptor instead. On all Intel adapters, top-level runtime-sized storage arrays are emitted as `T*` instead of `T(&)[]`. Adds two `naga::back::msl::Options` fields.
- **Findings:**
  1. [high] `wgpu-hal/src/metal/adapter.rs` `PrivateDisabilities::new`: `use_runtime_array_pointers: is_intel` changes shader codegen for every Intel Mac, including stock macOS. The description says the failure was reproduced only on OCLP. This needs a probe or a much narrower condition, and a naga reviewer must confirm that pointer emission keeps the bounds-check and `arrayLength` semantics in all paths (only top-level arrays are handled; structs with a trailing runtime array are not).
  2. [high] `broken_workgroup_size_attribute` compiles and creates a pipeline at every adapter enumeration on every Intel Mac. [#10743](https://github.com/gfx-rs/wgpu/issues/10743) measures about 30 s for each rejected probe on OCLP. Without [#10743](https://github.com/gfx-rs/wgpu/issues/10743), this PR makes enumeration on affected machines very slow. A lazy probe on the first compute pipeline creation would avoid the cost for most users.
  3. [medium] `naga/src/back/msl/mod.rs`: new public `Options` fields (`emit_workgroup_size_attribute`, `use_runtime_array_pointers`). This is a naga API change with no naga CHANGELOG entry and no doc on the hal `PrivateDisabilities` fields.
  4. [medium] Agent-authored: "Contributor review and a contributor-written final description are pending." The workaround targets unsupported (OCLP/Hackintosh) systems. Decide whether wgpu wants to carry naga codegen variants for these.
  5. [low] `CHANGELOG.md` has an extra blank line, and the entry text says "related to [#5827](https://github.com/gfx-rs/wgpu/issues/5827)" instead of a PR link.
- **To land:** scope the pointer codegen to affected systems (probe-gated). Make the probe lazy or cache it (fold [#10743](https://github.com/gfx-rs/wgpu/issues/10743) in). Get a naga owner review. Update the naga changelog. Get a contributor-written description.
- **Move-along action:** decide whether wgpu supports OCLP-only Intel compiler workarounds in naga/hal at all. If not, close [#10731](https://github.com/gfx-rs/wgpu/issues/10731) and [#10743](https://github.com/gfx-rs/wgpu/issues/10743) with that reason. If yes, ping andyleiserson to review, with the scoping notes above.
- **Depth:** full for hal and naga writer changes. Skimmed the GPU regression test and the snapshot.

#### [#10743](https://github.com/gfx-rs/wgpu/issues/10743): fix(metal): Cache Intel workgroup-size compiler probes ([@ITJamie](https://github.com/ITJamie))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-27.
- **Verdict:** Needs a lot of work (blocked on [#10731](https://github.com/gfx-rs/wgpu/issues/10731))
- **Waiting on:** [#10731](https://github.com/gfx-rs/wgpu/issues/10731) (it stacks on it). Then maintainer (andyleiserson is assigned).
- **Last activity:** 2026-10-28, ITJamie, merged trunk into the branch. 9 days ago. No human comments.
- **Approvals:** none
- **CI:** green
- **Summary:** Adds a process-global cache, keyed by `registryID` and MSL version, for the [#10731](https://github.com/gfx-rs/wgpu/issues/10731) Intel workgroup probe result, so that repeated enumeration skips the 30 s probe. Fixes [#10742](https://github.com/gfx-rs/wgpu/issues/10742).
- **Findings:**
  1. [medium] It exists only to fix a cost that [#10731](https://github.com/gfx-rs/wgpu/issues/10731) introduces. It should be folded into [#10731](https://github.com/gfx-rs/wgpu/issues/10731), or replaced by a lazy per-device probe. A separate PR for a self-inflicted regression adds review overhead.
  2. [low] `wgpu-hal/src/metal/adapter.rs`: the first enumeration still pays the full probe cost. The global `static OnceCell` cache never clears. That is acceptable for its size, but it hides driver updates within a process.
  3. [low] `CHANGELOG.md`: the PR adds a new `#### Metal` heading at line ~317, probably in the wrong release section.
  4. [low] The branch contains the [#10731](https://github.com/gfx-rs/wgpu/issues/10731) commit `6cc39f8e`. It must be rebased after [#10731](https://github.com/gfx-rs/wgpu/issues/10731) lands.
- **Overlap and landing order:** [#10725](https://github.com/gfx-rs/wgpu/issues/10725) and [#10731](https://github.com/gfx-rs/wgpu/issues/10731) are independent on trunk, but both edit `wgpu-hal/src/metal/adapter.rs` (the import list, `CapabilitiesQuery`/`PrivateDisabilities::new`) and will have textual conflicts. [#10731](https://github.com/gfx-rs/wgpu/issues/10731) reads `private_caps.msl_version`, which [#10725](https://github.com/gfx-rs/wgpu/issues/10725) changes to the probed version, so on OCLP the two work together. Order: [#10725](https://github.com/gfx-rs/wgpu/issues/10725) (if accepted in an opt-in form) -> [#10731](https://github.com/gfx-rs/wgpu/issues/10731) -> [#10743](https://github.com/gfx-rs/wgpu/issues/10743), or squash [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731). [#10743](https://github.com/gfx-rs/wgpu/issues/10743) cannot land before [#10731](https://github.com/gfx-rs/wgpu/issues/10731).
- **To land:** [#10731](https://github.com/gfx-rs/wgpu/issues/10731) lands first. Then squash this PR into it, or rebase it.
- **Move-along action:** ask the author to fold [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731) (same decision gate as [#10731](https://github.com/gfx-rs/wgpu/issues/10731)).
- **Depth:** full for the cache commit.

#### [#10860](https://github.com/gfx-rs/wgpu/issues/10860): fix(metal): select modern shader language profiles on watchOS ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-16.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (rebase, and the per-check version comments that inner-daemons asked for and that the author promised on 2026-11-16). Then a member review.
- **Last activity:** 2026-11-16, matthargett, pushed two commits that raise four watchOS floors to 11.0, after a discussion with inner-daemons about watchOS support.
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`. inner-daemons left a non-blocking `COMMENTED` review on 2026-11-14.
- **CI:** no CI runs on the head `febb88519`. Only the Mergify checks and "Summary" ran. The merge conflict probably stops the `pull_request` workflows.
- **Summary:** Adds `watchos` keys to the Metal adapter `available!` checks. watchOS 26/11/10 select MSL 4.0/3.2/3.1. The GPU family, read-write texture tier, counter sampling and argument buffer checks run on watchOS 11+, and fragment read-write storage on watchOS 10+. Indirect draw/dispatch (`INDIRECT_EXECUTION`) and base vertex/first instance are on for Apple5+ watch GPUs on watchOS 11+. The render ICB probe floor goes from watchOS 6 to 11, and the probe is skipped on 32-bit watchOS. CHANGELOG entry added.
- **Findings:**
  1. [medium] `wgpu-hal/src/metal/adapter.rs:896-898`, `:931`, `:939` `watchos_indirect_drawing` sets `indirect_draw_dispatch`, which turns on indirect draws and indirect dispatch. It also sets `base_vertex_first_instance_drawing`. The evidence in the description covers indirect draws only (S8 on watchOS 11.6.2, Series 10 on watchOS 27.2). There is no evidence for indirect dispatch or base vertex/first instance. Not fixed.
  2. [medium] Conflict with trunk in `wgpu-hal/src/metal/adapter.rs` near `:861`. [#10927](https://github.com/gfx-rs/wgpu/issues/10927) (merged 2026-11-14) moved the probe exclusions into `render_icb_probe_allowed()`. The 32-bit watchOS skip must move into that function on rebase.
  3. [low] inner-daemons asked for a comment at each check that says which watchOS version was tested. The new commits only change the numbers. No comment says what was tested. The floors are mixed: `family_check`, read-write texture, counter sampling and argument buffers use 11.0; MSL 3.1 (`:810-816`) and `fragment_rw_storage` (`:911-917`) keep 10.0. The description still says all capability checks run on watchOS 10 and later.
  4. [low] `wgpu-hal/src/metal/adapter.rs:863` The 32-bit watchOS probe skip has no log or evidence. It covers all `arm64_32` watches (S4 to S8), including the S8 that is the indirect-drawing floor.
  5. [low] No CI builds or runs watchOS. No test.
  6. [low] The description has no "LLM Use?" section and no checklist, so there are no agent-disclosure boxes.
  7. [low, process] inner-daemons asked on 2026-11-16 to move the watchOS support policy (App Store rule 2.5.1, best-effort support, in-repo docs) to an issue. No issue exists yet in the PR thread.
- **To land:** rebase onto [#10927](https://github.com/gfx-rs/wgpu/issues/10927); add evidence for indirect dispatch and base vertex/first instance on Apple5 watchOS 11, or limit the change to draws; add the version comments; update the description; get CI to run. Not likely before the release unless the author rebases this week.
- **Move-along action:** ask the author to rebase and to answer the indirect-dispatch evidence question. Decide if the watchOS policy issue blocks this PR (inner-daemons said it does not).
- **Depth:** full (re-review of the full diff and of the two new commits)

#### [#10875](https://github.com/gfx-rs/wgpu/issues/10875): metal: Lose the device when a command buffer fails ([@lexoliu](https://github.com/lexoliu))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** author. andyleiserson asked on 11-13 how the device-loss change interacts with `release_gpu_resources`, [#9579](https://github.com/gfx-rs/wgpu/issues/9579) and [#9278](https://github.com/gfx-rs/wgpu/issues/9278), and suggested a split into two PRs. No approval yet.
- **Last activity:** 2026-11-07, lexoliu, pushed the CHANGELOG PR-number commit. 6 days.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** On Metal, a command buffer that ends in `MTLCommandBufferStatus::Error` (GPU watchdog, out of memory) was reported as a successful completion. The PR attaches a completed handler to every command buffer of a submission (internal wait/signal buffers included). The fence advances only when all buffers of a submission finish. Any error marks the fence lost, so `get_fence_value`/`wait` return `DeviceError::Lost`. Failed present and `wait_for_idle` buffers also report loss. In wgpu-core, `Device::maintain` no longer returns the hal error. It loses the device, fires pending `on_submitted_work_done` callbacks, fails pending maps and BLAS compaction read-backs, and returns `Ok`. This wgpu-core change applies to all backends. Fixes [#9545](https://github.com/gfx-rs/wgpu/issues/9545).
- **Findings:**
  1. [medium] `wgpu-hal/src/metal/mod.rs:1370` Every Metal error code is treated as device loss. This includes the iOS error for work submitted while the app is in the background. On trunk such a frame is dropped silently. With this PR, an iOS app that submits during a background transition loses its device and must recreate all resources. The author asks the maintainer to decide (Dawn does the same). Reasoned from code; not run.
  2. [low] `wgpu-core/src/device/resource.rs:1252` After a loss, in-flight submissions stay tracked and their resources are freed only when the queue drops. `poll(wait_indefinitely())` returns `WaitSucceeded`, but the queue never reports empty. A user loop that polls until `QueueEmpty` spins forever on a lost device. This is documented in the doc comment but not in the CHANGELOG.
  3. [low] `tests/tests/wgpu-gpu/poll.rs` The only test of the lost path depends on the GPU watchdog. If the watchdog does not fire (likely on CI runners), the test takes the "not lost" branch. The new wgpu-core path, which runs on all backends, has no deterministic test.
  4. [low] `wgpu-hal/src/metal/mod.rs:1355`, `:1360` `expect("submission is not in flight")` runs inside an Objective-C completion block. A panic there aborts the process. I found no input that triggers it (every value is pushed before any handler is attached).
  5. [low] Process: the checklist omits the "Validation and feature gates" line, and the agent-disclosure lines are absent. The LLM section says Claude Opus implemented the code under the author's direction and the author reviewed it in full.
- **To land:** decide on finding 1 (keep uniform, or ignore the iOS background error code). Get a review from andyleiserson or inner-daemons.
- **Move-along action:** decide whether the iOS background-execution error should lose the device, then ping andyleiserson to review.
- **Depth:** full

#### [#10890](https://github.com/gfx-rs/wgpu/issues/10890): [deno] Expose chromium-experimental-multi-draw-indirect ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI red (4 failed: API Validation Mac aarch64, Other Mac aarch64, Test Windows x86_64, Test Windows x86_64 (attempt 1)); last update 2026-11-18.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer decision. On 2026-11-18 inner-daemons said that the PR uses Chrome's CTS tests for a Chrome API that wgpu's API does not match (wgpu has a separate count-buffer function), and that the new validation is valuable on its own and could be split out. The count-draw validation question (finding 1) is not answered.
- **Last activity:** 2026-11-18, inner-daemons, comment against using the Chromium experimental feature tests.
- **Approvals:** none
- **CI:** red, infra. "API Validation Mac" and "Other Mac" failed on HTTP 504 tool downloads. "Test Windows x86_64 (attempt 1)" failed in `compile_fail` because the `mach-dxcompiler-rs` build script panicked, and no retry ran. [#9679](https://github.com/gfx-rs/wgpu/issues/9679) has the same failures on the same day. A rerun is necessary. Mergeable.
- **Summary:** The Deno split that cwfitzgerald asked for on [#9679](https://github.com/gfx-rs/wgpu/issues/9679). Exposes `MULTI_DRAW_INDIRECT_COUNT` to JS as `chromium-experimental-multi-draw-indirect`, with `multiDrawIndirect` and `multiDrawIndexedIndirect` (with and without a count buffer). wgpu-core no longer strips this feature under `STRICT_WEBGPU_COMPLIANCE`. The PR also carries the count-buffer offset alignment check and `wgt::INDIRECT_BUFFER_OFFSET_ALIGNMENT` from [#9679](https://github.com/gfx-rs/wgpu/issues/9679). It moves `indirect_multi_draw` from `skip.lst` to `test.lst` and adds validation and Deno integration tests.
- **Findings:**
  1. [high] `wgpu-core/src/instance.rs:1460` and `deno_webgpu/render_pass.rs` This exposes count draws to JS, and wgpu-core does not validate count-draw arguments. `multi_draw_indirect_count` (`wgpu-core/src/command/render.rs`) checks only the feature, usage, alignment and buffer bounds, then calls the hal `draw_*_indirect_count` with no `indirect_validation` branch. Non-count `multi_draw_indirect` goes through indirect validation. Vulkan and DX12 expose the feature on trunk, so this applies today, not only after [#9679](https://github.com/gfx-rs/wgpu/issues/9679). The `STRICT_WEBGPU_COMPLIANCE` change also weakens that flag for every embedder that sets it: the flag no longer means "WebGPU features only". Firefox cannot request the feature (`wgpu-core-remote-types` carries only `FeaturesWebGPU`; I confirmed this). Fix: extend indirect validation to count draws first, or get explicit maintainer sign-off. Reasoned from code; not run.
  2. [medium] `deno_webgpu/render_pass.rs:37-45` When the feature is missing, `require_multi_draw_indirect` reports the error to the device immediately with `device.handle_error` and returns. It does not invalidate the pass or the encoder. So `encoder.finish()` succeeds and the command buffer submits without the draw. A pass command with a missing feature should make the encoder invalid and fail at `finish()`. The new integration test wraps call, `end()` and `finish()` in one error scope, so it cannot detect this. Reasoned from code; not run.
  3. [low] Overlap with [#9679](https://github.com/gfx-rs/wgpu/issues/9679): both PRs add `INDIRECT_BUFFER_OFFSET_ALIGNMENT`, `UnalignedIndirectCountBufferOffset` and the same CHANGELOG line. The CHANGELOG line in this PR links [#9679](https://github.com/gfx-rs/wgpu/issues/9679), not [#10890](https://github.com/gfx-rs/wgpu/issues/10890). There is no CHANGELOG entry for the Deno feature. Whichever lands second must drop the duplicate.
  4. [low] Under strict compliance, the Metal hal in [#9679](https://github.com/gfx-rs/wgpu/issues/9679) removes `MULTI_DRAW_INDIRECT_COUNT` (`wgpu-hal/src/metal/mod.rs:473-478` at [#9679](https://github.com/gfx-rs/wgpu/issues/9679)'s head). The CTS runner sets strict compliance by default. So on Metal, Deno and the CTS never see this feature, and the description's "Depends on [#9679](https://github.com/gfx-rs/wgpu/issues/9679)" is not true for Vulkan/DX12. The integration test already skips Apple targets.
  5. [low] Process: the description says Claude Code cloud did the split. All checklist boxes are ticked. There are no agent-disclosure lines.
- **To land:** decide on finding 1. Record a pass/encoder error instead of an immediate device error (finding 2), and test that `finish()` fails. Fix the CHANGELOG link and add a Deno entry. Rebase after [#9679](https://github.com/gfx-rs/wgpu/issues/9679) to drop the duplicate wgpu-core change. Rerun CI.
- **Move-along action:** decide: split the encoder validation into its own PR and land it, and hold the Deno feature until wgpu's multi-draw API matches the proposal and core validates count draws.
- **Depth:** full

## Vulkan

#### [#10629](https://github.com/gfx-rs/wgpu/issues/10629): Vulkan multi-planar DMA-buf import support ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-15.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (description still says "Testing TBD"; CHANGELOG has `#TODO`), then reviewer inner-daemons (review requested; jimblandy assigned)
- **Last activity:** 2026-11-15, jakobkukla, review comment: a `pub` item in `wgpu-hal/src/vulkan/device.rs:20` is not reachable from outside `wgpu-hal`.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** Adds multi-plane DMA-buf import to the Vulkan hal. `texture_from_dmabuf_fd` takes a slice of plane layouts and an `alias` flag. The new `texture_from_dmabuf_fds` imports disjoint images. Adds `ImportTextureError` and replaces `TextureMemory::Dedicated` with `DeviceMemory(Vec<_>)`. For Firefox external textures; fixes [#9801](https://github.com/gfx-rs/wgpu/issues/9801).
- **Findings:**
  1. [medium] No tests. The multi-fd disjoint path is untested even locally.
  2. [low] `wgpu-hal/src/vulkan/device.rs:719-726` The check for `ext::external_memory_dma_buf` and `ext::image_drm_format_modifier` returns `MissingExtension(khr::external_memory_fd::NAME)`, the wrong extension name.
  3. [low] Optional hardening: in the disjoint shared-fd path, bound `allocation_size` by the DMA-buf size. The earlier undersized-allocation concern is refuted for conforming drivers.
  4. [low] `CHANGELOG.md:211` `[#TODO]` link; the checklist is unchecked; the fixup commits ("Windows build fix", "View format fix", "fmt") must be squashed before the rebase-merge (label `merge: rebase`).
  5. [low] The Vulkan 1.1 check covers only the disjoint path. Open question from the author.
- **To land:** add tests (at least single-fd multi-plane and the error paths); fix the CHANGELOG link; squash the fixups; fix the extension name in the error; answer the Vulkan 1.1 question.
- **Move-along action:** ping author (is it ready for review? add the tests).
- **Depth:** status only (confirmed the push is a pure rebase; old findings re-checked at their lines)

#### [#10750](https://github.com/gfx-rs/wgpu/issues/10750): Add opt-in static HDR metadata submission for Vulkan surfaces ([@ecke101](https://github.com/ecke101))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-09.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer (jimblandy assigned) for the agentic-PR policy and API design decisions; author for the merge conflict.
- **Last activity:** 2026-11-09, ecke101, pushed 2 commits. 4 days.
- **Approvals:** none
- **CI:** green on the 11-09 head; has conflicts (`wgpu-core/src/instance.rs`)
- **Summary:** Adds `wgt::HdrMetadata`, `Features::SURFACE_HDR_METADATA` (bit 16), `Surface::supports_hdr_metadata` and `Surface::set_hdr_metadata`, and new default methods on the `wgpu_hal::Surface` trait. On native Vulkan swapchains the metadata goes through `vkSetHdrMetadataEXT`.
- **Findings:**
  1. [high] Policy: the "LLM Use?" section says "An agent implemented and reviewed the code, executed the checks, and created this PR". Both agent-disclosure boxes are checked. The contributor wrote only the Description paragraph.
  2. [medium] API design (unchanged): `set_hdr_metadata` and `supports_hdr_metadata` return a bare `bool` at all layers (`wgpu/src/api/surface.rs:78`, `:92`; `wgpu-core/src/instance.rs:781`, `:786`; `wgpu-hal/src/lib.rs:714`, `:719`). There is no error type and nothing goes to the error sink. The shape is Vulkan-only and does not consider DX12 or Metal HDR metadata.
  3. [medium] Metadata is not kept across `configure` (unchanged). A failed `set_hdr_metadata` is silent.
  4. [low] No GPU tests. The new unit test covers the four extension combinations of the support predicate.
  5. [low] `wgpu-types/src/features.rs:596` Bit 16 is still free on trunk (trunk has 15 and 17). A trial merge shows no feature-bit collision.
- **To land:** maintainer decision on the agentic-PR policy and the API shape; then rebase, and a real design review.
- **Move-along action:** decide whether to accept a fully agentic PR with this API shape. If not, close it and point the author to a design issue.
- **Depth:** targeted (reviewed the 2 new commits and re-checked the API signatures and feature bit; did not re-review the unchanged code)

#### [#10837](https://github.com/gfx-rs/wgpu/issues/10837): vulkan: write the pass-begin timestamp at TOP_OF_PIPE ([@kikijiki](https://github.com/kikijiki))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-16.
- **Verdict:** Land as is
- **Waiting on:** you: approve. kikijiki rebased on 2026-11-16 with no change to the diff, and the PR is mergeable. You and inner-daemons are assignees.
- **Last activity:** 2026-11-16, kikijiki, rebase.
- **Approvals:** inner-daemons approved on 11-12, before the last push. GitHub now shows "review required", so the approval does not count for the current head.
- **CI:** green on the head of 11-13; has conflicts with trunk.
- **Summary:** Render and compute passes write `beginning_of_pass_write_index` at `TOP_OF_PIPE` through a new private `unsafe fn write_pass_begin_timestamp`, instead of at `BOTTOM_OF_PIPE`. The end-of-pass stamp and `write_timestamp` do not change. This fixes near-zero pass durations on RADV when the GPU is busy.
- **Findings:**
  1. [low] `wgpu-hal/src/vulkan/command.rs:42-54` The begin stamp can now latch before earlier passes finish, so a pass duration can include queue wait. This is the usual profiler trade-off and is better than zero durations. The new doc comment explains why the helper exists.
  2. [low] No test (the bug depends on timing). Acceptable.
- **To land:** rebase, then an approval from a member with write access.
- **Move-along action:** approve and queue (the push after inner-daemons' approval changed only a doc comment).
- **Depth:** full (2-file diff)

#### [#10840](https://github.com/gfx-rs/wgpu/issues/10840): Vulkan: Don't use timeline semaphores on Arm's proprietary Mali driver ([@tadeohepperle](https://github.com/tadeohepperle))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-19.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (cwfitzgerald, assigned). An Arm engineer gives the version where the bug is fixed, so the version question has an answer.
- **Last activity:** 2026-11-19, archmonduu (an Arm engineer, unofficial): the old timeline semaphore implementation on CSF GPUs (for example Mali-G610) has the bug. Arm replaced it in driver "r46". On Rockchip version names, the fix came between g18 and g24, most likely in g21.
- **Approvals:** none
- **CI:** green
- **Summary:** Adds `Workarounds::DISABLE_TIMELINE_SEMAPHORES` for `ARM_PROPRIETARY`, which makes wgpu fall back to the binary `FencePool`. Without it, the Mali driver uses 16 B of kernel memory per timeline signal and loses the device after about 4.2M submissions. The PR also adds a 297-line `examples/bug-repro/03_arm_timeline_semaphore_leak` crate.
- **Findings:**
  1. [medium] `CHANGELOG.md` has a placeholder link: `[#XXXXX](https://github.com/gfx-rs/wgpu/pull/XXXXX)`.
  2. [medium] `wgpu-hal/src/vulkan/adapter.rs:2372` The flag is on for every Arm proprietary driver version, but only `g18p0` was tested. The Arm proprietary driver runs most Mali Android devices, so the change affects a large population. An Arm engineer (archmonduu, 2026-11-19) says that driver r46 fixes the bug, so the flag can apply to older drivers only. Recreating the timeline semaphore periodically is a more complex alternative.
  3. [low] The repro crate is a workspace and default member, so CI builds it and it adds to `Cargo.lock`. This follows the existing `bug-repro/01`, `02` pattern. It is the maintainer's choice whether to keep it.
  4. [low] Bit `1 << 3` collides with [#10692](https://github.com/gfx-rs/wgpu/issues/10692).
- **To land:** fix the CHANGELOG PR number; limit the flag to driver versions before the r46 fix (the version names differ by vendor); keep or drop the repro crate; rebase (conflicts, and the workaround bit `1 << 3` is taken on trunk).
- **Move-along action:** ask the author to limit the flag to driver versions before r46, as the Arm engineer's comment allows. Then review.
- **Depth:** full for the hal change; skimmed the repro crate.

#### [#10880](https://github.com/gfx-rs/wgpu/issues/10880): [hal vulkan] Trim command pools to what the last submission used ([@purian23](https://github.com/purian23))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (the `VulkanBackendOptions` trim-window field promised on 2026-11-18 is not pushed) and matthargett (asked to benchmark window 16 or 8 on the Quest 1). A maintainer must accept a new public backend option.
- **Last activity:** 2026-11-18, purian23, comment: will add a `VulkanBackendOptions` field for the window, default 16.
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`. matthargett's `CHANGES_REQUESTED` (2026-11-17) is still the latest review from him; the author applied his suggestions in `7d183a500`.
- **CI:** green
- **Summary:** In `reset_all`, the Vulkan command encoder records the fewest unused command buffers at any reset over a window of 64 resets. At the end of the window it frees that many and calls `vkTrimCommandPool` (core 1.1, or `VK_KHR_maintenance1` on 1.0). `ALLOCATION_GRANULARITY` goes from 16 to 1. The cause is that wgpu-core pools hal encoders for the life of the device, and Mesa returns command buffer memory only on a trim. The windowed form comes from matthargett's Quest 1 benchmarks: "trim at every reset" cost 2 to 3.5 times the CPU time when recording sizes alternate; the window stays within 5% of base.
- **Findings:**
  1. [medium, design] The next push adds a public `VulkanBackendOptions` with a window size (inner-daemons proposed it on 2026-11-18, default 8 or 16). This is new public API in `wgpu-types`, so it needs docs and a changelog entry. The author's own measurement shows why it is needed: encoders reset only when a submission retires, so an idle app never finishes a window and keeps its peak (45 to 60 buffers with window 64 on Mesa anv). A maintainer should decide if a backend option is wanted, or if a fixed window is enough. Not pushed yet.
  2. [low] `wgpu-hal/src/vulkan/command.rs:203-207` The comment "Keep only what this encoding used" does not match the windowed code. The code keeps what the busiest encoding in the window used. The second half of the comment (it does not matter which buffers are freed) is correct and fixes the old wording finding.
  3. [low] The old alternating-size churn finding is fixed by the window, from code: `min_unused` is 0 in any window where an encoding drew a new buffer, so no trim happens then. matthargett measured this on the Quest 1.
  4. [low] `wgpu-hal/src/vulkan/adapter.rs:2725` The `maintenance1` function table loads when `device_api_version < 1.1`, which matches the required-extension condition (`adapter.rs:1242`) and the adapter filter (`adapter.rs:2412`). Correct.
  5. [low] No test.
  6. [low] PR template: "I self-reviewed and fully understand this PR" is not checked. The "LLM Use?" section says Claude helped to find the cause and write the patch, and the author reviewed and tested it. There are no agent-disclosure boxes. The description is stale: it describes trim-after-each-reset and "Single commit", but the PR has three commits and a window.
- **To land:** decide on the backend option; author pushes it with docs and a changelog entry, fixes the comment, updates the description and checks the self-review box; matthargett confirms the default window on the Quest 1. Not likely before the release, because the public option is not written yet.
- **Move-along action:** reply on the thread to accept or reject the `VulkanBackendOptions` window field before the author writes it.
- **Depth:** full

#### [#10979](https://github.com/gfx-rs/wgpu/issues/10979): Fix running debug builds, GPU tests and the CTS on Oculus Quest 1 (Adreno 540) ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-20 (UTC, trunk merge).
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer. inner-daemons must say if the version gate answers the OOM-test concern (no new review request on GitHub). A member must review the `deno_webgpu` timestamp line. The maintainer must decide on agent authorship. Then the author updates the description.
- **Last activity:** 2026-11-20 (UTC), matthargett, merged trunk (head `2e74d9c28`). On 2026-11-19 he pushed "Skip OOM tests on Vulkan drivers older than 1.2".
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** Found while running the CTS and GPU tests on an Oculus Quest 1. The net diff has three parts. (1) hal Vulkan: if `vkCreateInstance` returns `VK_ERROR_EXTENSION_NOT_PRESENT` and `VK_EXT_debug_utils` was requested, create the instance again without it. (2) Tests: the six OOM GPU tests need Vulkan 1.2 or higher. The test harness and `wgpu-info` record the driver's Vulkan API version in the adapter report for this. (3) `deno_webgpu`: set `InstanceFlags::AUTOMATIC_TIMESTAMP_NORMALIZATION`, so timestamps are in nanoseconds. Changelog entries for (1) and (3).
- **Findings:**
  1. [low] `tests/src/params.rs:179-188`, `tests/tests/wgpu-gpu/oom.rs` The OOM tests that return early and report a pass are gone. The harness marks them as unsupported ("Vulkan API Version") in the test name, the same as other skips (`tests/src/run.rs:59-63`). This answers the "looks like a pass" part of inner-daemons' concern. The new gate is not the real cause: Vulkan predicts OOM only with `VK_EXT_memory_budget` (`wgpu-hal/src/vulkan/device.rs:892-898`), and the API version is a proxy for it. A 1.2+ driver without the extension still runs the tests until memory runs out (the same as trunk). A 1.1 driver with the extension loses the tests. The comment at `oom.rs:31-32` does not say why 1.2 was chosen. inner-daemons asked for a per-device skip. A maintainer must accept the version proxy.
  2. [low] `tests/src/report.rs:74-108` and `wgpu-info/src/report.rs:77-111` hold the same 40-line `vulkan_api_version` helper. The JSON field has `serde(default)`, so older `.gpuconfig` files still load. Builds on macOS work because the workspace `wgpu` dependency enables `vulkan-portability`.
  3. [low] `wgpu-hal/src/vulkan/instance.rs:997-1071` The retry is correct, from code: each pass pushes the debug-utils create info and the validation features onto a new copy of the caller's `create_info`, so the retry does not chain the removed messenger. `debug_utils` and `extensions` are both cleared, so `from_raw` sees no debug utils. If a `CreateInstanceCallback` adds a different missing extension, the code retries once without debug utils and logs a misleading warning before the real error. The comment says `InstanceFlags::DEBUG`, as inner-daemons asked.
  4. [low] The V8 external-memory commit `0f8de6bb8` is still in the branch history, but the trunk merge replaced it with the merged [#10802](https://github.com/gfx-rs/wgpu/issues/10802) (by Sam), so it is not in the net diff. The description is stale: item 2 describes the removed `VK_EXT_memory_budget` check, and item 4 and the 160 x 16 MB measurement describe code that is not in the PR. "Squash or Rebase?" says rebase, but a rebase would put `0f8de6bb8` and the merge that undoes it on trunk. Squash, or the author removes the commit.
  5. [low] The `deno_webgpu` change is Deno-specific, as the project rules require: it is an embedder choice when `deno_webgpu` creates its own instance (`deno_webgpu/lib.rs:197-199`).
  6. [low] Agent disclosure: "This PR was automatically created by an Agent" is checked. "The description was automatically written by an Agent" is not checked, but "LLM Use?" says Claude Mythos wrote the code, ran the checks and drafted the description and commit messages. The author says they read, edited and verified on the device. "I self-reviewed and fully understand this PR" is checked. A maintainer must decide if this is within the policy on agentic PRs.
- **To land:** inner-daemons or a member accepts the Vulkan 1.2 gate; a member reviews the `deno_webgpu` line; decide on agent authorship; author updates the description and drops `0f8de6bb8` (or squash at merge). Can land before the release: the remaining diff is small.
- **Move-along action:** ask inner-daemons on the OOM thread if the 1.2 gate is acceptable, and ask the author to fix the description.
- **Depth:** targeted (reviewed commit `384825697` and the net diff against trunk; the hal retry logic has no new commits and was not reviewed again)

#### [#10985](https://github.com/gfx-rs/wgpu/issues/10985): [v30] Fix crash on older Vulkan drivers when `poolSizeCount == 0` (backport of [#10124](https://github.com/gfx-rs/wgpu/issues/10124)) ([@tomasmach](https://github.com/tomasmach))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; base `v30`; mergeable; CI red (1 failed); last update 2026-11-18.
- **Verdict:** Land as is
- **Waiting on:** maintainer (no reviewer or assignee). The maintainer must decide if a v30.0.3 is wanted.
- **Last activity:** 2026-11-18, tomasmach, opened the PR.
- **Approvals:** none. GitHub shows no review decision for this PR (base `v30`).
- **CI:** red (1 failed): "MSRV Minimal Versions". The infrastructure caused it, not the PR. The `v30` `Cargo.toml` patches `libtest-mimic` to `cwfitzgerald/libtest-mimic` rev `9979b3c`. Cargo fails with "revspec '9979b3c' not found". The GitHub API shows the commit, but `master` of the fork is 1 commit behind it, so no branch holds it and a git fetch cannot get it. The same job passed on [#10929](https://github.com/gfx-rs/wgpu/issues/10929) (v30.0.2) on 2026-11-12. Trunk does not use the fork. All v30 PRs and a v30.0.3 release will fail this job until cwfitzgerald pushes a branch that holds `9979b3c` to the fork, or `v30` drops the patch.
- **Summary:** Backport of [#10124](https://github.com/gfx-rs/wgpu/issues/10124) (merged on trunk 2026-09-03 as `49afd3e38`) to `v30`. If a descriptor pool would have no pool sizes, add one `SAMPLER` entry, because `poolSizeCount == 0` is valid only since Vulkan 1.3.215 and older drivers crash. The author found it with nine minidumps from Intel HD 610 machines on Windows that crash at device creation with wgpu 30.0.1. Fixes [#9845](https://github.com/gfx-rs/wgpu/issues/9845) for wgpu 30.
- **Findings:**
  1. [low] `wgpu-hal/src/vulkan/descriptor.rs:321-330` The hal hunk is the same as trunk `49afd3e38`. The commit is a `cherry-pick -x` that keeps lucasmerlin as author. The changelog entry is under `## Unreleased` of the `v30` `CHANGELOG.md`. The fix ships in v31 from trunk. A v30.0.3 helps users who cannot upgrade yet.
  2. [low] Testing: `cargo check` only. The description says the affected user is testing a patched 30.0.1 build and the result will be added. The PR has no result.
  3. [low] PR template: "LLM Use?" says Claude Code read the minidumps, found the cause, applied the change to `v30` and wrote the description under the author's direction. "I self-reviewed and fully understand this PR" is not in the checklist. There are no agent-disclosure boxes.
- **To land:** decide on a v30.0.3; approve; fix the `libtest-mimic` fork rev, or accept the red MSRV job. Can land before the trunk release; it does not touch trunk.
- **Move-along action:** restore a branch that holds `9979b3c` in the `libtest-mimic` fork, then approve and merge to `v30`.
- **Depth:** full

#### [#10991](https://github.com/gfx-rs/wgpu/issues/10991): [hal vulkan] Work around Adreno 5xx OpArrayLength and indirect dispatch bugs ([@matthargett](https://github.com/matthargett))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-19.
- **Verdict:** Needs a bit of work
- **Waiting on:** inner-daemons (review requested). A naga and hal maintainer must accept the design. Author must fix the `CHANGELOG.md` conflict.
- **Last activity:** 2026-11-19, matthargett, comment to inner-daemons: this PR shows why a device-ID and driver blocklist is better than the extension-flag check in [#10979](https://github.com/gfx-rs/wgpu/issues/10979).
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** Two workarounds for Qualcomm's proprietary driver on Adreno 5xx (driver ID `QUALCOMM_PROPRIETARY` and device ID top byte 5), found on the Oculus Quest 1. (1) `OpArrayLength` crashes `vkCmdDispatch` when the layout has a storage buffer that the shader does not use. naga gets `spv::Options::buffer_sizes`, which computes `(size - offset) / stride` from binding sizes in push constants (after any immediates). wgpu-hal gives every pipeline layout the full push-constant range, assigns slots to storage buffers, and pushes the sizes in `set_bind_group`. (2) An indirect dispatch whose arguments an earlier dispatch wrote does not run unless the shader declares `NumWorkgroups`. `spv::WriterFlags::FORCE_NUM_WORKGROUPS` declares it. New naga tests and two GPU tests. The author tested on the device: CTS `arrayLength` goes from 205 crashes to 205 passes.
- **Findings:**
  1. [medium, design] This is a large driver-specific path in naga's SPIR-V backend (new public `spv::Options::buffer_sizes`, `spv::BufferSizes` and `WriterFlags::FORCE_NUM_WORKGROUPS`), and on these devices every pipeline layout gets a push-constant range of `maxPushConstantsSize`. `Restrict` bounds checks use the computed length, so a wrong size lets untrusted shaders access out of bounds on these devices. I checked these points from code and found no error: the size is the resolved binding size (`wgpu-hal/src/vulkan/device.rs:2092-2115`); the slot order is the same in `buffer_sizes_layout` and `create_bind_group` (same BGL entry order, same `take` limit); wgpu-core rebinds from the first incompatible group and on a change of immediate size (`wgpu-core/src/command/bind.rs:217-250,377-378`), so the offsets of groups that are not rebound stay valid; after the indirect-dispatch validation dispatch, wgpu-core pushes the immediates and all bind groups again (`wgpu-core/src/command/compute.rs:1098-1118`); `size - offset` does not underflow while wgpu-core's late minimum-binding-size check holds (`bind.rs:499-506`). A naga and a hal maintainer must accept the design.
  2. [low] `naga/src/back/spv/mod.rs` `BufferSizes` doc If the immediates leave no room, or the module has more than one immediate variable, naga still emits `OpArrayLength`, and the crash stays on these devices. The doc and the description say this.
  3. [low] `naga/src/back/spv/writer.rs:1591-1614` The `NumWorkgroups` fix applies to every indirect dispatch on these devices, because wgpu-core's indirect validation writes the arguments with a dispatch in the same command buffer (`compute.rs:1022-1098`). That is why the new GPU test reproduces the bug with CPU-written arguments.
  4. [low] `naga/src/back/spv/mod.rs:1078` `spv::Options` is not `#[non_exhaustive]`, so the new field breaks callers that use a struct literal. The changelog entry is under Vulkan only and does not mention the new naga API.
  5. [low] Workaround bits `1 << 4` and `1 << 5` follow trunk's `1 << 3`. [#10840](https://github.com/gfx-rs/wgpu/issues/10840) also adds a bit (and uses `1 << 3`, which trunk already uses), so the PR that lands second must take new numbers.
  6. [low] Tests: the naga tests check codegen, with and without immediates. The two GPU tests run everywhere but use the workaround only on Adreno 5xx, which CI does not have. No GPU test covers a change of pipeline layout or immediates together with sizes.
  7. [low] Conflict: `CHANGELOG.md` only.
  8. [low] Agent disclosure: "This PR was automatically created by an Agent" is checked. "The description was automatically written by an Agent" is not checked, but "LLM Use?" says Claude Opus 5.5 found the bugs, wrote the code, ran the tests and drafted the description and commit message. "I self-reviewed and fully understand this PR" is checked. A maintainer must decide if this is within the policy on agentic PRs.
- **To land:** a naga and a hal maintainer accept the design; author fixes the `CHANGELOG.md` conflict, mentions the naga API in the changelog entry, and corrects the disclosure box; decide on agent authorship. Not likely before the release: it needs a full review of 773 lines in naga and hal.
- **Move-along action:** ask inner-daemons (requested reviewer) to say if a push-constant path in naga is acceptable before a line-by-line review.
- **Depth:** full

#### [#10992](https://github.com/gfx-rs/wgpu/issues/10992): fix(vulkan): don't use timeline semaphores on Imagination's proprietary driver ([@tronical](https://github.com/tronical))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-19.
- **Verdict:** Land as is
- **Waiting on:** maintainer (no reviewer or assignee).
- **Last activity:** 2026-11-19, tronical, opened the PR.
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** On TI's AM62P (PowerVR, DDK 25.3), `vkWaitSemaphores` sometimes returns `VK_SUCCESS` while `vkGetSemaphoreCounterValue` still reports a lower value. wgpu-core then turns `PollType::Wait` with no timeout into `PollError::Timeout`. The PR turns off the `timeline_semaphores` private capability for `DriverId::IMAGINATION_PROPRIETARY`, so wgpu uses the binary `FencePool`. Mesa's PowerVR driver has a different driver ID and keeps timeline semaphores. The author tested on the device, with and without [#10677](https://github.com/gfx-rs/wgpu/issues/10677).
- **Findings:**
  1. [low] `wgpu-hal/src/vulkan/adapter.rs:2476-2477` The PR adds a driver-ID check to the `timeline_semaphores` expression, as the VeriSilicon check does. It does not need a `Workarounds` bit. [#10840](https://github.com/gfx-rs/wgpu/issues/10840) turns off timeline semaphores for Arm Mali with a new `Workarounds::DISABLE_TIMELINE_SEMAPHORES` bit (`1 << 3`, which trunk already uses). Both PRs edit the same expression and conflict in `adapter.rs`. A maintainer should use one pattern for both. This PR's pattern needs no new bit.
  2. [low] This PR selects the `FencePool` path. [#10960](https://github.com/gfx-rs/wgpu/issues/10960), which fixes the fence race on that path, merged on 2026-11-19, after this PR's base (2026-11-18). The PR is mergeable, so the merge result has the fix and no rebase is necessary.
  3. [low] The analysis agrees with the code: after a successful wait, `wgpu-core/src/device/resource.rs:1196-1202` returns `Timeout` if the fence value it reads is below the target. The change applies to every version of the Imagination proprietary driver, and only DDK 25.3 was tested. [#10840](https://github.com/gfx-rs/wgpu/issues/10840) has the same scope question for Arm Mali. The cost is one fence for each submission on these devices.
  4. [low] No test, because the bug needs this driver. The changelog entry is present.
  5. [low] PR template: "I self-reviewed and fully understand this PR" is checked. "LLM Use?" says Claude Fable 5.2 helped to develop and test the fix. There are no agent-disclosure boxes.
- **To land:** a member approval; accept the all-versions scope; whichever of this PR and [#10840](https://github.com/gfx-rs/wgpu/issues/10840) lands second resolves the `adapter.rs` conflict. Can land before the release.
- **Move-along action:** approve and queue, and ask the [#10840](https://github.com/gfx-rs/wgpu/issues/10840) author to use the same pattern after a rebase.
- **Depth:** full

#### [#11000](https://github.com/gfx-rs/wgpu/issues/11000): Return a timeout instead of losing the device when the Vulkan acquire fence wait times out ([@mayocream](https://github.com/mayocream))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-19.
- **Verdict:** Land as is
- **Waiting on:** maintainer (no reviewer or assignee). The maintainer must decide on agent authorship: both agent-disclosure boxes are checked.
- **Last activity:** 2026-11-19, mayocream, opened the PR.
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** Fixes [#9029](https://github.com/gfx-rs/wgpu/issues/9029). On Windows with `Fifo` or `FifoRelaxed`, the native Vulkan swapchain waits on an acquire fence. A `VK_TIMEOUT` from that wait went through `map_host_device_oom_and_lost_err`, became `DeviceError::Unexpected` and lost the device. With the PR, the swapchain keeps the acquired image index in `pending_acquire` and returns `SurfaceError::Timeout`. The next acquire uses that image and waits on the fence again, and does not acquire a new image. `release_resources` waits on a pending fence before it destroys it. The author tested on an AMD iGPU on Windows 11, also with a reconfigure after each timeout, with the validation layer on.
- **Findings:**
  1. [low] The PR agrees with the surface error contract, from code. hal `acquire_texture` returns no texture on `Err`, and the held image stays inside the swapchain. The next acquire, `configure` or `unconfigure` handles it. wgpu's `CurrentSurfaceTexture::Timeout` ("skip the current frame and try again later") and `wgpu-types` `SurfaceStatus::Timeout` ("Try reconfiguring your surface") both work with it. WebGPU's `getCurrentTexture` has no timeout, so this does not change WebGPU behavior. On a timeout, `advance_acquire_semaphore` is not called (`wgpu-hal/src/vulkan/swapchain/native.rs:592`), so the next acquire gives the same acquire semaphore, which the held image signals. This is correct.
  2. [low] `native.rs:464-467` `release_resources` waits on the pending fence with `u64::MAX` after `vkDeviceWaitIdle`, before `configure` makes the new swapchain (`instance.rs:1228-1234`). The fence signals when the presentation engine releases the image. After the device is idle, the queued presents finish and release it, so the wait ends in the normal case. A driver that needs one more present to release the image would hang `configure` or `unconfigure`. A bounded wait would remove this risk. The author tested reconfigure on one GPU.
  3. [low] No automated test. A test needs more than 1 s of GPU work in one frame, which is not practical in CI. The changelog entry is present.
  4. [low] Agent disclosure: "This PR was automatically created by an Agent" and "The description was automatically written by an Agent" are both checked. "LLM Use?" says Claude Code wrote the change, the repro and the description. "I self-reviewed and fully understand this PR" is checked. The project does not allow fully agentic PRs, so a maintainer must decide.
- **To land:** decide on agent authorship; a member approval; optionally a bounded wait in `release_resources`. Can land before the release if the policy decision allows it.
- **Move-along action:** decide on the agent-authorship policy for this PR before a review.
- **Depth:** full

## GLES, DX12, present and platform

#### [#8388](https://github.com/gfx-rs/wgpu/issues/8388): Implement DXGI Swapchain on Vulkan ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-15.
- **Verdict:** Status note
- **Waiting on:** andyleiserson (re-review). His only review is CHANGES_REQUESTED from 2026-10-07. He is an assignee, but GitHub does not show a pending review request for him. inner-daemons has a pending review request.
- **Last activity:** 2026-11-15, cwfitzgerald, force-push with six new fixup commits (head `ac9137b13`).
- **Approvals:** none (review decision CHANGES_REQUESTED)
- **CI:** green (66 pass, 1 skipped).
- **Summary:** Adds a DXGI flip-model swapchain path to the Vulkan backend on Windows. The fixes for the review are in fixup commits, which the author will squash after approval.
- **Findings:**
  1. [low] `wgpu-hal/src/vulkan/swapchain/dxgi.rs:829,1078,1202` The fixup `ac9137b13` replaces `waited_for_frame: bool` with `pending_present_count`. The count starts at `maximum_frame_latency`, an acquire waits and decrements only if the count is above 0, and only a successful present increments it. This is the same logic as the DX12 code from merged [#10511](https://github.com/gfx-rs/wgpu/issues/10511) (`wgpu-hal/src/dx12/mod.rs:1156,1384,1502`). The Vulkan path makes a new swapchain and waitable on each configure (no `ResizeBuffers`), so resetting the count on configure is correct. On a wait timeout, the count does not change and the acquire returns `Timeout`, which is also correct. Checked from code.
  2. [low] Review threads: 3 of 46 are unresolved. In all 3, the last comment is from cwfitzgerald.
- **To land:** re-request a review from andyleiserson; get an approval; squash the fixups.
- **Move-along action:** Re-request the review from andyleiserson, so GitHub shows it in his queue.
- **Depth:** status only (plus a correctness check of fixup `ac9137b13`)

#### [#10515](https://github.com/gfx-rs/wgpu/issues/10515): Fix support for Emscripten (v6.0.10+) by making it use `web-sys` ([@almarklein](https://github.com/almarklein))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI red (4 failed: Clippy Emscripten, Test Linux x86_64, Test Mac aarch64, Test Windows x86_64); last update 2026-10-27.
- **Verdict:** Needs a lot of work
- **Waiting on:** author. grovesNL (2026-10-07) proposed option 1: make Emscripten identical to `wasm32-unknown-unknown` for both WebGL and WebGPU, which needs a small glow change. The author agreed on 2026-10-09 to "have a stab" and has not pushed since 2026-10-07.
- **Last activity:** 2026-10-27, ErichDonGubler, changed their approval to CHANGES_REQUESTED because CI is red. 10 days. (andyleiserson dequeued the PR from Mergify on 2026-10-16.)
- **Approvals:** ErichDonGubler approved on 2026-10-05 and replaced the approval with CHANGES_REQUESTED on 2026-10-27.
- **CI:** red: Clippy Emscripten, Test Linux x86_64, Test Windows x86_64, Test Mac aarch64. Related. Erich saw `UnwindSafe` errors after wasm-bindgen became a dependency, and the current diff removes the Emscripten wgpu-core and wgpu-hal deps.
- **Summary:** Makes `web` and `webgpu` cfgs true on Emscripten, and drops the Emscripten-specific dep block in `wgpu/Cargo.toml`.
- **Findings:**
  1. [high] `wgpu/Cargo.toml` The PR removes the `target_os = "emscripten"` block that pulls in `wgpu-core` and `wgpu-hal`. But `webgl` in `wgpu/build.rs` still excludes Emscripten, so the GLES-on-Emscripten path is left in an inconsistent state. CI confirms the breakage.
  2. [medium] The agreed direction (option 1) is a different and larger change: route Emscripten through the web glow backend and remove the `cfg(Emscripten)` branches across wgpu and the examples. The current diff does not do this.
  3. [low] `wgpu/Cargo.toml` `cfg(all(target_family = "wasm" ))` has a stray `all(` and space. It should be `cfg(target_family = "wasm")`.
  4. [low] There is no CI coverage that builds for Emscripten 6.0.10+ with wasm-bindgen. Only a link to an external PoC repo exists.
- **To land:** implement option 1 (with any needed glow change), make all CI green, and get a re-review.
- **Move-along action:** ping author. Ask whether they still plan option 1. If not, close it in favour of a maintainer or community PR (DouglasDwyer has a prototype branch; grovesNL prototyped option 1).
- **Depth:** full

#### [#10775](https://github.com/gfx-rs/wgpu/issues/10775): feat(hal/dx12): Extension to restrict enumeration to a specific adapter LUID ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-17.
- **Verdict:** Needs a bit of work (a design decision is open; the code is in good shape)
- **Waiting on:** maintainer (design decision on the `Extensions<T>` token and the separate `request_adapters` argument; andyleiserson asked on 2026-11-03), then jimblandy and teoxoy (assigned on 2026-11-18).
- **Last activity:** 2026-11-18, cwfitzgerald, assigned jimblandy and teoxoy. The last push was a rebase on 2026-11-17.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green
- **Summary:** Lands part of the [#10304](https://github.com/gfx-rs/wgpu/issues/10304) extension API. It adds `wgt::BackendMap`, `BackendAdapterOptionsMap`, an `Extensions<T>` token that is unsafe to construct, `hal::Api::AdapterOptions`, an `options` argument on `hal::Instance::enumerate_adapters`, `dx12::Dx12AdapterOptions { luid }`, and a public `wgpu::Instance::request_adapters(options, extensions)`. A DX12 GPU test covers select-by-LUID and `NotFound` for an unknown LUID. Fixes [#9932](https://github.com/gfx-rs/wgpu/issues/9932).
- **Findings:**
  1. [medium] Design question for the maintainer: one token type `Extensions<T>` gates all extensions. Some are safe (the `Dx12AdapterOptions` docs say "These options are safe"), but all need `unsafe`. andyleiserson says that the `request_adapter` / `request_adapters` split bothers him more than the token. This sets the pattern for all future extensions ([#10304](https://github.com/gfx-rs/wgpu/issues/10304), [#4067](https://github.com/gfx-rs/wgpu/issues/4067)), so decide it before merge.
  2. [low-medium] `wgpu/src/backend/wgpu_core.rs:882` `request_adapter` passes `options.apply_limit_buckets` where trunk passes `false` (trunk `wgpu_core.rs:850`). This fixes a separate bug: trunk ignores the user's `apply_limit_buckets`. It changes the adapter limits for users who set it, is not in the changelog entry, and is not related to the LUID filter. Make it a separate commit with its own changelog entry.
  3. [low] The changelog entry covers only the LUID filter. It does not mention the public `wgpu::Instance::request_adapters`, `wgpu::Extensions`, `BackendAdapterOptionsMap`, or the breaking `hal::Instance::enumerate_adapters` signature (this matters for out-of-tree hal backends).
  4. [low] `wgpu-types/src/backend.rs:298,306` `BackendMap::get` and `remove` call `downcast_ref::<T>().unwrap()` (and the `remove` equivalent). `__backend_map_private` is `pub` (`backend.rs:235`), so a downstream crate can implement `Sealed`/`BackendMapValue` for its own type with the DX12 backend key, and `request_adapters` then panics. This needs deliberate misuse and an `unsafe { Extensions::new(..) }`, and it is a panic, not UB. Safe code cannot make a non-empty `Extensions<T>` or change one.
  5. [low] `wgpu-hal/src/lib.rs:693` The doc on `Api::AdapterOptions` says "used with `request_adapter_ext`", but that function does not exist.
  6. [low] `wgpu-core/src/instance.rs:454,609` `request_adapter` and `request_adapters` both use `profiling::scope!` and `api_log!` with the name "Instance::request_adapter". `request_adapter` calls `request_adapters`, so each call logs twice under one name.
  7. [low] `wgpu-types/src/backend.rs:477` `backend_type!` is `#[macro_export]`ed (public API), but nothing uses it, and its doc example names `wgt::BackendDeviceOptions`, which does not exist. The `BackendMap` doc says that `SomeTrait` "is implemented by the value type for each backend", which is not true for `BackendAdapterOptions`.
  - The rebase of 2026-11-17 changed only the `wgpu/src/lib.rs` import lists (adds `Rect`) and the changelog context. All other patches are identical (checked with `git range-diff`).
- **To land:** a design decision on the token and API shape; a split of the `apply_limit_buckets` change; changelog entries for the new public API; a fix for the stale doc reference; a member review. Not reasonable before the release.
- **Move-along action:** Decide whether the `Extensions<T>` token plus separate-argument shape is the extension pattern, and tell jimblandy and teoxoy before they review.
- **Depth:** targeted (verified that the rebase changed only imports and changelog context; findings carried from the 2026-11-06 full and deeper review and spot-checked at the new head)

#### [#10778](https://github.com/gfx-rs/wgpu/issues/10778): Make `wayland-sys`'s `dlopen` feature togglable ([@HigherOrderLogic](https://github.com/HigherOrderLogic))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI red (15 failed: API Operation Linux x86_64, API Validation Linux x86_64, Clippy Android aarch64, Clippy IOS aarch64); last update 2026-10-29.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer: decide if the feature is wanted. No maintainer has replied. Then the author fixes CI.
- **Last activity:** 2026-10-29, HigherOrderLogic, opened the PR. 8 days.
- **Approvals:** none
- **CI:** red: Clippy Linux x86_64/aarch64, Clippy MacOS x86_64/aarch64, Clippy IOS, Clippy Android aarch64, Clippy OpenHarmony, Clippy cts_runner, Core MSRV Linux/MacOS, Test Linux x86_64, Test Mac aarch64, API Validation/Operation Linux, Other Linux. Related. When the feature is off, `wayland-sys` (a `cfg(unix)` dep) links to libwayland at compile time on every unix target.
- **Summary:** Moves `wayland-sys/dlopen` behind a new `wgpu-hal/wayland-dlopen` feature, which is on in `wgpu`'s default features only, and adds direct-link imports in `gles/egl.rs`.
- **Findings:**
  1. [high] `wgpu-hal/Cargo.toml` `wgpu-hal` without the new feature (wgpu-core and wgpu-hal users such as Firefox and Servo, and `wgpu` users with `default-features = false`, for example Bevy) now links libwayland-client and libwayland-egl at build time. That breaks builds on macOS, iOS and Android, and adds a hard runtime dep on X11-only Linux. Dynamic loading must stay the default for every crate, not only for `wgpu`'s defaults.
  2. [medium] Cargo features are additive, so "link at compile time" can only win when nobody in the graph enables `wayland-dlopen`. The design needs a maintainer decision. Options: an opt-in `wayland-link` feature with dlopen as the fallback in code, or no change (use `wayland-sys`'s own features from the app).
  3. [low] Missing CHANGELOG entry for a new `wgpu` feature.
  4. [low] `wgpu-hal/src/gles/egl.rs` The `#[cfg(all(unix, not(feature = "wayland-dlopen")))]` imports are not limited to Linux/BSD, which adds to the non-Linux breakage.
- **To land:** a maintainer decision that the feature is wanted. Keep dlopen the default for wgpu-hal and wgpu-core. Fix CI on all targets. Add a CHANGELOG entry.
- **Move-along action:** decide whether wgpu should offer compile-time libwayland linking at all. If yes, ping the author with the "dlopen stays default at the hal level" constraint. If no, close it with that reason.
- **Depth:** full

#### [#10866](https://github.com/gfx-rs/wgpu/issues/10866): Detach GLES renderbuffers when reusing draw framebuffers ([@shtse8](https://github.com/shtse8))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI red (1 failed: Test Mac aarch64); last update 2026-11-06.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviewer, no assignee)
- **Last activity:** 2026-11-06, shtse8, commented on the Mac CI failure. 7 days.
- **Approvals:** none
- **CI:** red (`Test Mac aarch64`: `wgpu_gpu::poll::wait_on_submission_with_timeout` fails with `PollError::Timeout` at `poll.rs:273` on Metal, 1180 of 1181 pass). Not caused by the PR: the PR changes only `wgpu-hal/src/gles/queue.rs` and adds a GPU test; the failing test runs on Metal in its own process. Probably a timing flake. Rerun before merge. Mergeable.
- **Summary:** When the GLES queue reuses its draw framebuffer, it now also calls `glFramebufferRenderbuffer(..., 0)` for the depth-stencil slot and every color slot, not only `glFramebufferTexture2D(..., 0)`. The Android Emulator guest GL client tracks texture and renderbuffer attachments separately, so a stale multisampled renderbuffer gives `GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE`. Adds a GPU regression test that goes from a 4x MSAA pass to single-sampled passes.
- **Findings:**
  1. [low] `wgpu-hal/src/gles/queue.rs:1143-1172` The workaround runs on every render pass on every GL driver: one extra call for depth-stencil and one for each of `max_color_attachments` (typically 8). It is spec-legal and cheap, but it is an emulator workaround that is not gated. Consider detaching only the slots that the previous pass attached, or accept the cost knowingly.
  2. [low] `tests/tests/wgpu-gpu/framebuffer_reuse.rs` The test does not reproduce the bug on CI drivers (Mesa llvmpipe, Metal and others accept texture-only detach), so it passes with and without the fix. It is still a useful smoke test. The author says this in the description.
  3. [low] No CHANGELOG entry for a user-visible GLES fix (Android Emulator).
  4. [low] `tests/tests/wgpu-gpu/framebuffer_reuse.rs:16-28` The `ShaderModuleDescriptor` literal is not indented by rustfmt (rustfmt skips it because of the long string literal). Cosmetic.
  5. [low] The description omits the template's "LLM Use?" section and the checklist, so there is no agent disclosure.
- **To land:** add a CHANGELOG entry; ask for the "LLM Use?" section; rerun the Mac job.
- **Move-along action:** review (small hal change plus one test).
- **Depth:** full

#### [#10981](https://github.com/gfx-rs/wgpu/issues/10981): Prefer an 8-bit-per-channel EGL config when choosing the GLES framebuffer configuration ([@Victorbs-01](https://github.com/Victorbs-01))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-18.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (first review; policy decision on an agent-written PR).
- **Last activity:** 2026-11-18, Victorbs-01, opened the PR.
- **Approvals:** none. GitHub says `REVIEW_REQUIRED`.
- **CI:** green
- **Summary:** In `choose_config`, each EGL tier first asks for a config with `RED_SIZE`/`GREEN_SIZE`/`BLUE_SIZE` 8. If that gives no config, it uses the original attribute list. The cause: SwiftShader on the Android 11 emulator lists an RGB565 config first when no alpha is requested, so presented colors were quantized. Changelog entry added.
- **Findings:**
  1. [medium] `wgpu-hal/src/gles/egl.rs:155-171` from code and the EGL sort rules: EGL treats a color size as a minimum and, for each color size in the request, sorts configs with more color bits first. Before this PR no color size was requested, so that rule did not apply. Now a driver that lists 10-bit configs (for example RGB10_A2 on Mesa) can return one first. The change applies to every EGL platform (Linux, Android, ANGLE), not only SwiftShader. The only test is one emulator. Ask the author to check this, or to pick the first config with exactly 8 bits from a full `choose_config` list.
  2. [low] `wgpu-hal/src/gles/egl.rs:167-170` The `_ =>` arm also catches an `Err` from the 8-bit request and retries without a log. The original request then logs its own error, so the effect is small.
  3. [low] No automated test. The description says "Lavapipe control run on the same emulator image: no behavior change", but Lavapipe is a Vulkan driver and does not use this EGL code.
  4. [low] Agent disclosure: both "automatically created by an Agent" and "description was automatically written by an Agent" are checked. "LLM Use?" says an agent wrote the code and the description, and the human reviewed and approves it. The self-review box is checked. First-time contributor. A maintainer must decide if this is within the policy on agentic PRs.
- **To land:** a policy decision on agent authorship; an answer on the 10-bit config ordering. Can land before the release only if both are settled this week.
- **Move-along action:** ask the author about configs with more than 8 bits per channel on desktop Mesa.
- **Depth:** full (two-file diff)

## Core and API

#### [#10098](https://github.com/gfx-rs/wgpu/issues/10098): Add immediates support to WebGPU and remove `Features::IMMEDIATES` ([@beicause](https://github.com/beicause))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-09-18.
- **Verdict:** Needs a lot of work
- **Waiting on:** author (no activity since [@cwfitzgerald](https://github.com/cwfitzgerald)'s 2026-09-18 comment asking to keep `Features::IMMEDIATES`. [@cwfitzgerald](https://github.com/cwfitzgerald) and [@inner-daemons](https://github.com/inner-daemons) have CHANGES_REQUESTED. The author invited [@czoli1976](https://github.com/czoli1976) to open a follow-up PR, but none exists yet.)
- **CI:** green, but stale (base from August 2026; probably conflicts with trunk now)
- **Summary:** Forwards `set_immediates` to the browser on the WebGPU backend. Removes `Features::IMMEDIATES` and `naga::valid::Capabilities::IMMEDIATES` everywhere, and renumbers the WebGPU feature bits.
- **Findings:**
  1. [high] Scope does not match the maintainer's direction. The maintainer asked to keep `Features::IMMEDIATES`, and so the naga capability and the bit table too. On the web, set the feature only when `setImmediates` exists. The PR still removes the feature in about 40 files, including naga validation, a crate that handles untrusted input.
  2. [high] `wgpu/src/backend/webgpu.rs:895, 2285-2309` The WebGPU path does not work end to end. `max_immediate_size` is hard-coded to `wgt::Limits::default()` instead of being read from the browser's `maxImmediateSize`. `create_pipeline_layout` does not forward `immediate_size` as `immediateSize`. So the browser's layout always has 0 immediate bytes, and every `setImmediates` call fails validation there.
  3. [medium] `wgpu/src/backend/webgpu.rs:3550, 3693, 3986` `set_immediates_with_u8_slice(...).unwrap()` panics on a browser without `setImmediates` (Firefox, Safari, older Chrome). Today this is a clear panic message. After this PR, it is an unwrap of a JS TypeError, and the feature check that would stop it is gone.
  4. [medium] `wgpu/src/backend/webgpu/webgpu_sys/gen_Gpu{Compute,Render}PassEncoder.rs`, `gen_GpuRenderBundleEncoder.rs` The vendored generated files are edited by hand. The upstream fix is in wasm-bindgen 0.2.128, so re-vendor with `cargo xtask vendor-web-sys` instead (unresolved thread from [@cwfitzgerald](https://github.com/cwfitzgerald)).
  5. [low] `wgpu-types/src/features.rs` If any feature bit is removed later, leave a hole instead of renumbering ([@inner-daemons](https://github.com/inner-daemons)'s unresolved thread).
- **To land:** Rewrite to the requested shape: keep `Features::IMMEDIATES`; report it on the web when `setImmediates` exists; read `maxImmediateSize` when present; forward `immediateSize` on the pipeline layout; implement the three `set_immediates`; re-vendor `webgpu_sys`; drop all removal changes. Given that the author handed off, it may be simpler to close this in favor of [@czoli1976](https://github.com/czoli1976)'s follow-up.
- **Depth:** targeted (read the WebGPU backend, core and CHANGELOG diffs, and the review discussion. Did not read the naga/test mechanical removals, because the maintainer already asked for them to be reverted.)

#### [#10155](https://github.com/gfx-rs/wgpu/issues/10155): perf(core): reuse the acceleration structure build scratch buffer ([@mstampfli](https://github.com/mstampfli))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-10-06.
- **Verdict:** Needs a bit of work
- **Waiting on:** author. On 2026-09-04 Vecvec asked for a policy that releases or shrinks the cached buffer after many builds that use only a small part of it. On 2026-10-06 Vecvec asked "are you still keen to pursue this". The author has not answered.
- **Last activity:** 2026-10-06, Vecvec, asked the author whether they still want to continue. 31 days. The author's last activity was 2026-09-03.
- **Approvals:** none
- **CI:** green (head of 2026-09-02; stale against trunk)
- **Summary:** Keeps the largest scratch buffer used for acceleration-structure builds on the device, and reuses it in later builds. This avoids one large device allocation and free per build.
- **Findings:**
  1. [medium] `wgpu-core/src/scratch.rs` (Drop) The cache keeps the largest buffer ever used until device loss or drop, with no trim. Both reviewers (inner-daemons, Vecvec) object to this, and the open design question is still the shrink/eviction policy.
  2. [low] `wgpu-core/src/device/resource.rs` `release_gpu_resources` destroys the cached buffer. A `ScratchBuffer` that retires after that call puts a buffer back into the cache, which then stays until `Device::drop`. This is harmless, but the comment "a lost device will not build again" is not the full story.
  3. [low] There is no automated test for reuse. The author verified reuse only with local instrumentation.
- **To land:** the author answers Vecvec and adds a retention/shrink policy (or the maintainers accept unbounded peak retention); rebase.
- **Move-along action:** close (stale), unless the author answers soon. The author has not replied in 31 days, and the design is not settled. Alternatively, ping the author one last time with a deadline.
- **Depth:** full

#### [#10438](https://github.com/gfx-rs/wgpu/issues/10438): Add `create_render_pipeline_async` and `create_compute_pipeline_async` to `Device` ([@abdymazhit](https://github.com/abdymazhit))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-04.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (author asked [@cwfitzgerald](https://github.com/cwfitzgerald) for review on 2026-10-24; [@sagudev](https://github.com/sagudev) approved; the two open threads are addressed but not resolved). Also needs a rebase (merge state "dirty").
- **CI:** green (66 success). CI does not run the WebGPU backend. The author tested the WebGPU path by hand in Chrome, and two downstream users run it in production.
- **Summary:** Adds async pipeline creation to the public `Device` API, which returns `impl Future<Output = Result<_, wgpu::Error>>`. WebGPU calls `create*PipelineAsync()` and maps `GPUPipelineError`/`TypeError` to `Error::Validation`/`Internal`. wgpu-core calls `create_*_pipeline_or_error` and returns a ready future. Two new required `DeviceInterface` methods (breaking for custom backends, noted in CHANGELOG). Vendors `GPUPipelineError` in `webgpu_sys`.
- **Findings:**
  1. [medium] API decision for the maintainer: on native, `*_async` blocks and returns an already-resolved future. The name and signature commit wgpu to a `Result`-returning async API, which error scopes do not see. This differs from every other creation call (those report errors through error scopes). It matches the WebGPU spec (errors reject the promise, and device loss resolves to an invalid pipeline), so this is consistent. But it is a permanent public API shape, and a later native background compile will need to fit behind it. It needs explicit sign-off.
  2. [medium] `wgpu-core/src/device/resource.rs:4496-4515, 4661-4680` `create_*_pipeline_or_error` is the entry point that Firefox (`wgpu-core-remote`) and Deno use. It now also writes trace actions, an `api_log!` line, and the "Shader translation error" log for these callers. This is probably correct: those pipelines were missing from traces before. But it changes behavior for Firefox and Deno, and the PR does not test it with the player. Also, a failed pipeline is not recorded in the trace on this path, but the synchronous path records it as invalid. Check that the trace player stays consistent when a later trace action refers to a failed async pipeline. (On the error path, no id is handed out in core, so this is probably fine.)
  3. [low] `wgpu/src/backend/wgpu_core.rs:146-290`, `wgpu/src/backend/webgpu.rs` About 250 lines of the diff move the descriptor mapping into helpers (`with_render_pipeline_descriptor`, `map_*_pipeline_descriptor`). Per the maintainer's preference, pure code motion should be its own commit. The PR is one commit.
  4. [low] `wgpu/src/backend/webgpu.rs` `pipeline_error_from_js` sets `source` to the string `"<WebGPU Error>"`, as other WebGPU errors already do. The `description` is the JS message. This is OK. A browser without `GPUPipelineError` gives `Internal` with `format!("{value:?}")`. wasm-bindgen's `instanceof` shim catches a missing global, so it does not panic.
  5. [low] `tests/tests/wgpu-gpu/pipeline.rs` The positive tests only check that a draw or dispatch with the pipeline gives no validation error. They do not read back output. The error-path and device-lost tests are good: they check that the error scope stays empty.
- **To land:** Maintainer sign-off on the API shape (finding 1). Rebase. Optional: split the descriptor-mapping refactor into its own commit. Optional: confirm with Firefox people ([@ErichDonGubler](https://github.com/ErichDonGubler), [@jimblandy](https://github.com/jimblandy)) that the new trace and log behavior in `create_*_pipeline_or_error` is fine.
- **Depth:** targeted (read the public API, dispatch, wgpu-core and wgpu_core-backend diffs in full. Skimmed the WebGPU backend; most of it is code motion. Did not read the vendored `webgpu_sys` files or the custom-backend example in detail.)

#### [#10591](https://github.com/gfx-rs/wgpu/issues/10591): [core] avoid allocation for single-region texture copies ([@pbkx](https://github.com/pbkx))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-10-14.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (decide if the change is wanted; [@inner-daemons](https://github.com/inner-daemons) commented "very low importance", did not block)
- **CI:** green
- **Summary:** In `copy_texture_to_texture`, collects the hal copy regions into a `SmallVec<[TextureCopy; 1]>`, so a single-region copy does not allocate. Adds a 2D variant of the copy readback test, a "single slice" case, and a "Texture Copy Encoding" benchmark.
- **Findings:**
  1. [medium] `wgpu-core/src/command/transfer.rs:1669` The `regions.len() == 1` branch is not necessary. `regions` is a `Map<Range>` with an exact size hint, so `regions.collect::<SmallVec<[_; 1]>>()` alone stays inline for one region and makes one exact heap allocation for more than one. The `Vec` + `.into()` arms do the same thing in a longer way. The depth-stencil and planar arms could also collect straight into the `SmallVec`. This makes the diff one line.
  2. [low] `benches/benches/wgpu-benchmark/transfer.rs` The new benchmark measures only encoding cost of 1000 copies of a 4x4 texture. A reviewer called it pointless. It adds a file that someone must maintain for a micro-optimization. Decide if it stays.
  3. [low] Correctness is fine: the region order is the same as before, and the hal call takes `&regions` as a slice in both cases. The new 2D test case is the only test that exercises the single-region path on a non-3D texture (3D always has `array_layer_count == 1`).
- **To land:** Maintainer decides if the optimization is wanted. If yes: simplify to a single `SmallVec` collect, and drop the benchmark or agree to keep it. CHANGELOG not needed (internal perf).
- **Depth:** full

#### [#10812](https://github.com/gfx-rs/wgpu/issues/10812): [core] Add child texture concept that shares underlying data and subsets TextureUsage ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-02.
- **Verdict:** Needs a lot of work
- **Waiting on:** teoxoy (assigned on 11-09; no review yet). The author must fix the high finding.
- **Last activity:** 2026-11-09, teoxoy, assigned themself. 4 days. The only commit is `050417d3` from 2026-11-02.
- **Approvals:** none
- **CI:** green (old run); has conflicts
- **Summary:** Adds `Texture::create_child_texture` (wgpu-core only), which makes a texture handle with `TextureInner::Child { parent }`. The child has a subset of the parent's usage and shares the init tracker, tracking index, clear mode, views and bind groups through five `Arc`s.
- **Findings:** unchanged. No commit since the old review, so the confirmed high finding is still present.
- **To land:** fix clear-view ownership (for example one `Arc<TextureShared>` that destroys the HAL objects once), settle the design, add tests with render-attachment, depth and mixed parent/child usage, resolve the conflicts.
- **Move-along action:** ping teoxoy to post the design feedback, starting with the double-free (old finding 1).
- **Depth:** status only

#### [#10821](https://github.com/gfx-rs/wgpu/issues/10821): Release the acquired surface texture when the device is lost ([@KanikaBreed](https://github.com/KanikaBreed))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-09.
- **Verdict:** Needs a bit of work (leaning to close)
- **Waiting on:** author (sagudev's [#10296](https://github.com/gfx-rs/wgpu/issues/10296) question from 11-03 is still unanswered).
- **Last activity:** 2026-11-09, inner-daemons, approved. 4 days. No author activity since 2026-11-03.
- **Approvals:** inner-daemons
- **CI:** green; mergeable
- **Summary:** In `Surface::present` and `Surface::discard`, if the device is lost, the PR drops `acquired_texture` before it returns the error.
- **Findings:**
  1. [medium] The approval does not answer the old concern. inner-daemons checked that the change is correct, and it is. The old concern is that the change is mostly not needed: on trunk, configure with a new device, unconfigure and `Surface` drop already release the texture. The only effect is an earlier release at a failing present or discard. sagudev's question (relation to [#10296](https://github.com/gfx-rs/wgpu/issues/10296)) and sagudev's proposal (unconditionally unconfigure on device loss) have no reply. The approval also does not resolve the third release style next to `destroy()` and `release_inner`.
  2. [low] There is no CHANGELOG entry. inner-daemons asked for one.
- **To land:** the author answers sagudev, or reworks to the unconditional-unconfigure approach. A CHANGELOG entry. If the author does not reply, close.
- **Move-along action:** decide with inner-daemons and sagudev whether the earlier release alone is worth a merge. If yes, ask for the CHANGELOG line and merge; if not, close.
- **Depth:** status only

#### [#10831](https://github.com/gfx-rs/wgpu/issues/10831): Fix truncated trace playback ([#1620](https://github.com/gfx-rs/wgpu/issues/1620)) ([@vatsanbalaji-ossagent](https://github.com/vatsanbalaji-ossagent))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-19.
- **Verdict:** Needs a bit of work. Policy question first.
- **Waiting on:** author, and a second maintainer on the trace format. andyleiserson (assigned) suggests to separate records with the record separator character (0x1E), so that the trace stays pretty-printed and can be edited by hand. He asks another maintainer to agree first.
- **Last activity:** 2026-11-19, andyleiserson, review comments (record separator, shorter changelog text). The last author activity was 2026-11-05.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** `DiskTrace` writes one compact RON action per line and ends the file with `// End of trace`. `player::parse_trace` parses each line and skips an unparsable last line. The old list format is no longer readable.
- **Findings:** none recorded in this file. andyleiserson's review of 2026-11-19 has the open items.
- **To land:** settle the agent-account policy; agree on the record format (one line per action, or a record separator with pretty-printing); rebase (conflicts); shorten the changelog text.
- **Move-along action:** agree or disagree with andyleiserson's record-separator proposal, and decide the agent-account question.
- **Depth:** status only

#### [#10838](https://github.com/gfx-rs/wgpu/issues/10838): Fix sparse tlas instance indices ([@Vecvec](https://github.com/Vecvec))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; has conflicts; CI red (1 failed: Test Linux x86_64); last update 2026-11-14.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (CI is red from a lavapipe crash that the new test triggers). inner-daemons is a requested reviewer since 11-09.
- **Last activity:** 2026-11-12, Vecvec, pushed. 1 day.
- **Approvals:** none
- **CI:** red, caused by the PR's new test hitting a driver bug. `Test Linux x86_64`: `wgpu_gpu::ray_tracing::shader::sparse_instances_index_correct` is killed with SIGSEGV on Vulkan/llvmpipe. All other jobs pass. Mergeable.
- **Summary:** In a TLAS, each `None` slot before the last `Some` now becomes a null instance, so shader instance indices match the array index. The null instance uses BLAS handle 0 where the backend allows it (Vulkan; DX12 except WARP). On other backends (Metal, WARP) it uses a small BLAS that is built at device creation. The PR adds `Alignments::null_acceleration_structure_allowed` to hal.
- **Findings:**
  1. [medium] The new test segfaults lavapipe in CI. The crash is in Mesa, but any app that builds a sparse TLAS on an affected lavapipe version now crashes through safe API. Before merge, the test needs a skip or expected failure for llvmpipe until CI has a Mesa with MR 46145. Consider also a workaround for affected lavapipe versions. A rebase on the Mesa update is not enough, because the fix is not released yet.
  2. [medium] `wgpu-core/src/null_acceleration_structure.rs:43` The empty BLAS uses `device.zero_buffer` as its vertex buffer. The zero buffer is created with `COPY_SRC | COPY_DST` (plus `TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT` only for `EXPERIMENTAL_RAY_QUERY`, `device/resource.rs:612`) and is left in the `COPY_SRC` state. It never gets `BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT` usage, and no barrier moves it to that state before the build. This breaks the hal contract (wgpu-core must declare usages and place barriers). Reasoned from code. This path runs only on Metal and DX12 WARP, where the effect is probably small, and WARP CI passes. Add the BLAS-input usage when acceleration structures are enabled, and add a barrier.
  3. [low] `wgpu-core/src/null_acceleration_structure.rs:78` `NonZeroU64::new(build_scratch_size).expect(..)` still panics at device creation if a backend returns a scratch size of 0. With one triangle this is unlikely. `.max(1)` is cheaper than an argument.
  4. [low] `wgpu-core/src/command/ray_tracing.rs:288` Null instances use `mask: 0xff`. The author tried 0 first. On lavapipe, a 0 mask, a null AS and an empty AS all become "invalid" nodes, so the mask does not avoid the bug. 0xff is acceptable, because the null or empty BLAS has no hittable geometry.
- **To land:** skip or expect-fail the new test on llvmpipe. Fix the zero-buffer usage and barrier. Then a ray tracing review.
- **Move-along action:** ping author (skip the test on llvmpipe so CI is green; zero-buffer BLAS-input usage), then inner-daemons reviews.
- **Depth:** full for the new commits; targeted for the rest (test shader math and backend spec claims not re-checked)

#### [#10839](https://github.com/gfx-rs/wgpu/issues/10839): fix: relax render bundle resource lifetimes ([@mvanhorn](https://github.com/mvanhorn))
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-06.
- **Verdict:** Needs a bit of work
- **Waiting on:** inner-daemons (review requested on 11-09)
- **Last activity:** 2026-11-09, inner-daemons, requested review from themself. 4 days. The last author activity was 2026-11-06.
- **Approvals:** none
- **CI:** green (the old `dxc.zip` failure is gone); mergeable
- **Summary:** The PR removes the `'a` bound on resource arguments of `RenderBundleEncoder` (`set_pipeline`, `set_index_buffer`, `draw_indirect`, `draw_indexed_indirect`) and of all six resource methods of `util::RenderEncoder`. This matches what `RenderPass` already allows. Fixes [#6433](https://github.com/gfx-rs/wgpu/issues/6433).
- **Findings:**
  1. [medium] `wgpu/src/util/encoder.rs:8` The `'a` parameter on the `RenderEncoder<'a>` trait is now unused. Keep it on purpose to limit the break, or remove it. This is a maintainer decision.
  2. [medium] `tests/tests/wgpu-validation/api/render_bundle.rs:~185` The new `TextureDescriptor` literal has no `texture_binding_view_dimension`, so the test does not compile on current trunk. Reasoned from code, not built.
  3. [low] The test `bundle_resource_lifetimes` is about 150 lines of mostly compile-time checks. `draw_errors_reported_at_finish` is not related to lifetimes.
  4. [low] The CHANGELOG entry is still under `#### General`. The description repeats a paragraph, the checklist is malformed, and the text reads as LLM-written. The description only says "AI was used for assistance".
- **To land:** decide the trait `'a` question. Trim the tests and drop the unrelated one. Rebase and add the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field.
- **Move-along action:** let inner-daemons review (they took it). Decide the trait-lifetime question.
- **Depth:** targeted (new commit and CI only; the code did not change since the old full review)

#### [#10862](https://github.com/gfx-rs/wgpu/issues/10862): Reuse command buffer and pass trackers ([@VitaminCpp](https://github.com/VitaminCpp))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-15.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer decision on the agent disclosure; then inner-daemons (assignee and requested reviewer since 2026-11-09). cwfitzgerald is also an assignee since 2026-11-18.
- **Last activity:** 2026-11-15, VitaminCpp, rebased and added the missing test field, and replied to rowanG077 that the `ResourceMetadata::clear` change should be a follow-up PR.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green (66 pass, 1 skipped) on the head of 2026-11-15.
- **Summary:** Adds a `TrackerPool` to `Device`. Command encoders, the pending-writes encoder, and the intermediate trackers of compute and ray tracing passes take a `Tracker` from the pool. The tracker goes back to the pool, cleared with its capacity kept, when the submission completes (`EncoderInFlight::release`) or the pass ends. This removes per-encoder allocations that scale with the device's resource count. A second commit fixes the Metal `HalCounters::query_sets` count for occlusion query sets.
- **Findings:**
  1. [low] The rebase changed only the changelog context and added `texture_binding_view_dimension: None` to the `TextureDescriptor` in `tests/tests/wgpu-gpu/tracker_pool.rs` (checked with `git range-diff`).
  2. [low] Overlap with rowanG077's change (comment of 2026-11-14, linked commit on his fork, not a PR). `wgpu-core/src/track/metadata.rs:38-41` `ResourceMetadata::clear` sets both tables to length 0, so the next `set_size` writes `None` again up to the device's highest tracker index. Each reuse of a pooled tracker costs O(highest index), not O(members). rowanG077's version clears only the occupied slots and keeps the length. His claim is correct from code. The author asks for it as a follow-up PR. This is only a performance matter; the pooled tracker is consistent either way.
  3. [low] The core change looks correct. All vectors are emptied, so a pooled tracker is consistent (size 0) and grows again through `allow_index`/`set_size`. `TextureStateSet::clear` also clears the `complex` map. `TrackerPool::release` drops the resource references before it takes the pool lock. Trackers on error paths are dropped, not returned, which costs only an allocation.
  4. [low] `wgpu-core/src/track/mod.rs:605` The pool has no cap, and pooled trackers never shrink. The pool keeps as many trackers as the peak number of encoders in flight, each sized to the highest tracker index. The author offers a cap. This is a maintainer decision.
  5. [low] The new `DEVICE_TRACKER_POOL` lock rank follows five ranks. I did not check each edge against the call paths. CI with lock validation passes.
  6. [low] Policy: both agent-disclosure boxes are checked. The LLM section says that Claude Code ported the author's prototype, wrote the tests and drafted the description, and that the author reviewed and edited it. The self-review box is checked. AGENTS.md does not allow fully agentic PRs, so the maintainers must decide if this is acceptable.
  7. [low] The Metal counter fix is a separate commit with its own changelog entry. The tracker change also has a changelog entry.
- **To land:** a maintainer decision on the agent disclosure; a review by inner-daemons. It changes tracker lifetimes in wgpu-core, so landing it after the release is safer.
- **Move-along action:** Decide the agent-disclosure question, then let inner-daemons review. Ask rowanG077 to open his `clear` change as a PR on top of this one.
- **Depth:** targeted (`git range-diff` against the 2026-11-13 head, plus `metadata.rs` for the overlap; core, hal and changelog were read in full on 2026-11-13)

#### [#10966](https://github.com/gfx-rs/wgpu/issues/10966): test: cover shared wgpu-core device wrappers ([@BlackRam-oss](https://github.com/BlackRam-oss))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-16.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (fix the title and description), then cwfitzgerald (assigned on 2026-11-18).
- **Last activity:** 2026-11-16, BlackRam-oss, pushed a docs-only commit after sagudev's comment and removed the tests.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green (65 pass, 1 skipped).
- **Summary:** Adds 5 lines of docs to `wgpu::Device::from_core`: error scopes, the uncaptured error handler and the single device-lost callback belong to the wgpu-core device, so all users of that core device share them. The PR is not a test. The first two commits added tests; the last commit removed them, so the net diff is the doc comment only.
- **Findings:**
  1. [medium] The title ("test: cover shared wgpu-core device wrappers") and the description (three regression tests, "Deno WebGPU and a native compositor can use separate `wgpu` wrappers") do not match the diff, which is a doc comment only. sagudev said on 2026-11-16 that Deno uses wgpu-core directly and that the tests were not worth it. The author agreed and narrowed the PR, but did not update the title or description. The history has two commits that add tests and one that removes them; squash on merge.
  2. [low] `wgpu/src/api/device.rs:982-985` The doc is correct. `CoreDevice` methods forward `push_error_scope`, `pop_error_scope`, `on_uncaptured_error` and `set_device_lost_callback` to the `wgc::device::Device`, whose `ErrorSink` holds the scopes and handler (`wgpu-core/src/error.rs:54-58`). `set_device_lost_closure` replaces the earlier closure (`wgpu-core/src/device/resource.rs:5733`). The doc does not say that error scopes are per thread (`ErrorSink` keys them by `ThreadId`). This is optional.
  3. [low] No `CHANGELOG` entry is necessary for a doc comment.
  4. [low] Disclosure: the PR does not use the template, so it has no checklist. The LLM section says that Codex helped with the tests and docs, reviewed the changes, and drafted the description.
- **To land:** the author changes the title and description to match the doc-only diff; one member approval; squash. It can land before the release.
- **Move-along action:** Ask the author to retitle the PR (for example "docs: note shared state in `Device::from_core`") and to update the description, then approve.
- **Depth:** full (5-line doc change; checked each claim against `wgpu/src/backend/wgpu_core.rs` and `wgpu-core/src/error.rs`)

## CI, dependencies and your own PRs

#### [#10784](https://github.com/gfx-rs/wgpu/issues/10784): test(cts): run non-`const` `modf` execution tests for `f32` ([@ErichDonGubler](https://github.com/ErichDonGubler))
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-10-30.
- **Verdict:** Land as is
- **Waiting on:** maintainer (ErichDonGubler wrote "should be an easy review once this passes CI"; CI has passed)
- **Last activity:** 2026-10-30, ErichDonGubler, self-comment. 7 days.
- **Approvals:** none
- **CI:** green (all 66 checks pass, including Test on Linux/Mac/Win/Wasm and codecov). Base is trunk 49325ca, a few days old. Note: CTS itself is not part of this CI (`cts_runner` is only Clippy), so the new CTS cases are not executed by PR CI.
- **Summary:** Adds 24 `modf` `f32` execution cases (storage_r, storage_rw, uniform input sources) to `cts_runner/test.lst`, found missing while triaging Firefox bug 2087401.
- **Findings:**
  1. [medium] The cases are not run by PR CI, so green CI says nothing about whether they pass. Maintainer should confirm with `cargo xtask cts` (or the CTS workflow) before merging. Author does not state they ran them locally and the PR has no description of results.
  2. [low] The list has no `modf` line on trunk today (grep of `cts_runner/test.lst` finds only `fract` validation), so these are new coverage, not duplicates. The `const` (constant input) variants are deliberately excluded. Reason not stated in the PR; add one line to the description.
  3. [low] No CHANGELOG needed.
- **To land:** Nothing, assuming the CTS cases pass locally or in the CTS workflow.
- **Move-along action:** review: run (or confirm) the CTS run for these 24 cases, then merge.
- **Depth:** full

#### [#10869](https://github.com/gfx-rs/wgpu/issues/10869): Update KosmicKrisp test expectations ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Status note
- **Waiting on:** author: queue it. andyleiserson approved on 2026-11-18. He does not get matching results on macOS 27, but says this does not block the PR.
- **Last activity:** 2026-11-18, andyleiserson, approved.
- **Approvals:** andyleiserson (2026-11-18)
- **CI:** green (65 pass, 1 skipped).
- **Summary:** Updates the expected failures on KosmicKrisp (Apple M5 Max, macOS 27.2.1). Adds expected failures for 17 `draw_indirect` tests, `render_pass_resource_ownership` and `vertex_indices`. Removes `draw_multiview_noncontiguous` and `subgroup_operations`. Test files only.
- **Findings:** none (status only). Disclosure: both agent boxes are checked; the self-review box is unchecked.
- **To land:** queue it. Check the self-review box. An OS-version condition is a possible follow-up.
- **Move-along action:** queue it.
- **Depth:** status only

#### [#10903](https://github.com/gfx-rs/wgpu/issues/10903): Fix binding more immediate data then pipeline supports ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-15.
- **Verdict:** Status note
- **Waiting on:** andyleiserson (assigned on 2026-11-18). No review yet.
- **Last activity:** 2026-11-18, cwfitzgerald, assigned andyleiserson. The last push was on 2026-11-15.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green (66 pass, 1 skipped).
- **Summary:** Core trims immediates to the layout size, but still calls hal `set_immediates` with an empty slice, which panics on Metal and DX12 and is a validation error on Vulkan. Render bundles did not trim and did not resend after a layout change. The fix adds a shared `ImmediateState::take_dirty` that trims and skips empty writes, and initializes the DX12 immediates root element on signature reset. Three commits, with tests in `immediates.rs` and a changelog entry.
- **Findings:** none (status only, no code review). Disclosure: both agent boxes are unchecked; the self-review box is checked.
- **To land:** a full code review (wgpu-core and wgpu-hal); one approval. It is a panic fix and can land before the release if andyleiserson reviews it this week.
- **Move-along action:** Ask andyleiserson to review it before the release.
- **Depth:** status only

#### [#10959](https://github.com/gfx-rs/wgpu/issues/10959): Fix indirect draw validation panic on noop ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; mergeable; CI green; last update 2026-11-16.
- **Verdict:** Status note
- **Waiting on:** author (queue it). andyleiserson approved on 2026-11-16.
- **Last activity:** 2026-11-18, cwfitzgerald, assigned andyleiserson. The approval is from 2026-11-16.
- **Approvals:** andyleiserson (review decision APPROVED)
- **CI:** green (66 pass, 1 skipped).
- **Summary:** Indirect draw validation reads the source buffer as `u32`, so it needs a lower bound of 4 on the dynamic offset alignment. Adapters that report a lower alignment (the noop backend reports 1) hit an assert panic. The fix adds the bound, removes the redundant assert, and borrows the limits instead of a clone per draw. Changes `wgpu-core/src/indirect_validation/{draw,mod}.rs`, adds a noop test in `tests/tests/wgpu-validation/noop.rs`, and a changelog entry.
- **Findings:** none (status only). andyleiserson asked in his approval why the test is in `noop.rs` and not with the other indirect draw tests (the existing ones have an `expect_noop` flag); he says it does not matter much. No reply. Disclosure: both agent boxes are checked; the LLM section says Opus found the bug, wrote the fix and tests, and drafted the description. The self-review box is checked.
- **To land:** optionally answer the test-location question; send to the merge queue. It can land before the release.
- **Move-along action:** Answer andyleiserson's question in one line and queue it.
- **Depth:** status only

#### [#10983](https://github.com/gfx-rs/wgpu/issues/10983): Add changelog fragment tooling ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-11-18.
- **Verdict:** Status note
- **Waiting on:** author (mark ready), then inner-daemons (review requested on 2026-11-18).
- **Last activity:** 2026-11-18, cwfitzgerald, force-push (one commit, `814696e3e`) and review request to inner-daemons.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green (66 pass, 1 skipped).
- **Summary:** Adds `xtask changelog` tooling to create, preview and assemble changelog fragments (`xtask/src/changelog/fragments.rs`, about 900 lines), and changes `.github/workflows/changelog.yml`. v31.0.0 still uses `CHANGELOG.md`. Connects to [#3683](https://github.com/gfx-rs/wgpu/issues/3683).
- **Findings:** none (status only). Disclosure: both agent boxes are checked; the LLM section says GPT-6 implemented and tested it and drafted the description, and GPT-6 Astra ran two adversarial review passes. The self-review box is unchecked.
- **To land:** check the self-review box; mark ready; one approval. It can land before the release, because v31 keeps the current workflow.
- **Move-along action:** Self-review, check the box, and mark it ready.
- **Depth:** status only

#### [#10984](https://github.com/gfx-rs/wgpu/issues/10984): Switch to changelog fragments after v31 ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** draft; mergeable; CI red (1 failed: Test Mac aarch64); last update 2026-11-18.
- **Verdict:** Status note
- **Waiting on:** v31.0.0 release, then the author (rebase and mark ready). Depends on [#10983](https://github.com/gfx-rs/wgpu/issues/10983). inner-daemons has a review request (2026-11-18).
- **Last activity:** 2026-11-18, cwfitzgerald, force-push and review request to inner-daemons.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** red (1 failed). `Test Mac aarch64` fails in `wgpu_gpu::poll::wait_on_submission_with_timeout` on Metal (Apple Paravirtual device): `PollType::Wait` with a 1 s timeout returned `Timeout` (`tests/tests/wgpu-gpu/poll.rs:273`). The PR changes only changelog tooling and docs, and [#10983](https://github.com/gfx-rs/wgpu/issues/10983) with the same first commit is green. So the failure is from the infrastructure (a slow paravirtual Mac runner), not the PR.
- **Summary:** Second commit on top of [#10983](https://github.com/gfx-rs/wgpu/issues/10983). Switches CI to fragments, adds `changelog.d/README.md`, updates `docs/release-checklist.md`, and removes the old `CHANGELOG.md` checks from `xtask/src/changelog.rs`. The plan is to rebase after v31.0.0 and cherry-pick the cutover onto the v31 release branch.
- **Findings:** none (status only). Disclosure: both agent boxes are checked; GPT-6 and GPT-6 Astra implemented it and GPT-6 drafted the description. The self-review box is checked.
- **To land:** land [#10983](https://github.com/gfx-rs/wgpu/issues/10983); release v31.0.0; rebase; rerun CI; one approval. By design it lands after the release.
- **Move-along action:** None until v31.0.0 ships.
- **Depth:** status only

#### [#11019](https://github.com/gfx-rs/wgpu/issues/11019): tests: raise wasm runner page timeout to 60 s ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-20
- **State (2026-11-20):** ready; mergeable; CI pending (8); last update 2026-11-20.
- **Verdict:** Status note
- **Waiting on:** a reviewer (none requested).
- **Last activity:** 2026-11-20, cwfitzgerald, opened the PR.
- **Approvals:** none
- **Summary:** The `Test WebAssembly` job flakes on `issue_4122::clear_buffer_range_respected` because the wasm runner gives each test page 10 s. The PR raises the page timeout to 60 s, below the 90 s limit of the nextest `wasm` profile. One line in one file.
- **To land:** CI green, one approval. It can land before the release.
- **Depth:** status only

## Drafts

#### [#6584](https://github.com/gfx-rs/wgpu/issues/6584): Test pass timestamp writes validation ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI no runs; last update 2025-10-08.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2025-10-08, ErichDonGubler, last push/update. 394 days.
- **CI:** stale
- **Summary:** Adds an empty GPU test `VALIDATE_PASS_TIMESTAMP_WRITES` whose body is `todo!()` (5 lines, 1 file). Requested by Wumpf in [#6578](https://github.com/gfx-rs/wgpu/issues/6578). No test logic exists. No comments.
- **Move-along action:** close (empty stub, 2 years old; author can reopen when it is written)

#### [#8480](https://github.com/gfx-rs/wgpu/issues/8480): Add `set_web_canvas_size` option. ([@kpreid](https://github.com/kpreid)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI no runs; last update 2025-11-05.
- **State:** stalled
- **Waiting on:** maintainer (decision: add a `SurfaceConfiguration` field defaulting to not resizing the web canvas, "option 1" of [#7938](https://github.com/gfx-rs/wgpu/issues/7938)?)
- **Last activity:** 2025-11-05, kpreid, opened. 366 days. No comments or reviews.
- **CI:** stale
- **Summary:** Adds `SurfaceConfiguration::set_web_canvas_size` (55/-16 lines, 11 files) to stop the infinite canvas resize on WebGPU/WebGL ([#7938](https://github.com/gfx-rs/wgpu/issues/7938), still open). Author says it is untested on web and "do not merge".
- **Move-along action:** decide whether option 1 is the accepted API, then ping kpreid (or close if a different fix for [#7938](https://github.com/gfx-rs/wgpu/issues/7938) is chosen)

#### [#8653](https://github.com/gfx-rs/wgpu/issues/8653): Texture format testing ([@inner-daemons](https://github.com/inner-daemons)) [draft PR]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-08-25.
- **Status:** Adds tests that check each texture format works as the device claims, plus fixes for the failures they found (the author says the fixes are LLM-written and reviewed by them). Not ready: it is a draft that you set back to draft on 2026-08-14, asking for one fix per commit plus a final commit with all tests. No author response since. The branch has 23 commits.
- **Waiting on:** author (history cleanup, then un-draft). Reviewers Erich (earlier CHANGES_REQUESTED, 2025-12-11 and later comments) and cwfitzgerald are requested, but the PR is not in a reviewable state.
- **Recommended action:** Reply with a position: keep it parked until the author restructures commits. Optionally decide whether to land only the fixes first and make the tests opt-in, since the author said that landing the tests turns CI red on everyone's machines. Close it if no update comes by an agreed date.
- **Notes:**
  - CI: green on head `972ccef` (62 checks, Aug 25). Older reports of failing tests and OOM (Erich, Jan 2026) look resolved by the later fixes.
  - Open design points from earlier reviews: Erich was unsure about exposing the API changes (CHANGES_REQUESTED, 2025-12-11) and wanted test enumeration to be less manual (one test per format). The author has said much of this may be covered by the CTS. Please state whether you want this suite in addition to the CTS.
  - Related: [#8646](https://github.com/gfx-rs/wgpu/issues/8646) (closed by this PR) and [#8670](https://github.com/gfx-rs/wgpu/issues/8670) (list of texture format issues).

**Deeper review ([#8653](https://github.com/gfx-rs/wgpu/issues/8653): Texture format testing (draft)):**

> - **Open questions from the first pass:** Are the commits now split as requested (one fix per commit plus a tests commit)? The request was on 2026-06-26, and the PR was drafted on 2026-08-14.
> - **Answers:**
>   1. **[refuted: not split]** The branch has 23 commits. The only push after the 2026-06-26 request is `972ccef1`, a trunk merge (2026-08-25). The history:
>      - **Twelve 2026-06-12 commits.** These are mostly one fix per commit. Examples: `fix(types): drop multisample claim from 16-bit norm formats`, `fix(vulkan): only report multisample caps…`, `fix(metal): set PixelFormatView usage…`, `fix(gles): …` (×4), and `fix(vulkan): …` (×2). But the test commit `d911cc57 Add texture format conformance tests` is the 4th commit, not the last. Two test commits are mixed in with the fixes (`d7a13c61`, `2e9dbea9`).
>      - **Seven 2026-06-17/18 WIP commits.** `b54c50f8`, `70e0e899 Broken further testing`, `ae97ae23 Fixed most of the tests`, `bc8af1e8 Reformatted`, `c9971a65 Fix final issues on my mac` and `91232e49` change only `tests/tests/wgpu-gpu/texture_format/mod.rs`. `25ec6e82 Claude claims to have fixed another issue` is a fix (`wgpu-types/src/texture/format.rs` plus CHANGELOG) placed after the tests.
>      - **Four merge commits from trunk** (`fa16d664`, `c2ca748f`, `0ea2839d`, `972ccef1`).
>
>      The requested shape (one fix per commit, then one tests commit, no WIP or merge commits) is not there.
> - **New findings:** none.
> - **Verdict change:** none (draft; waiting on the author).
> - **Move-along action change:** none. If the maintainer wants to move it along, the June 12 fix commits are already nearly in the requested form. The author could rebase them onto trunk, add `25ec6e82` as one more fix commit, and squash all the test commits into one final commit.

#### [#8849](https://github.com/gfx-rs/wgpu/issues/8849): make some error types constructible ([@jgraef](https://github.com/jgraef)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (1 failed: cargo-deny advisories); last update 2026-01-21.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-01-21, sagudev (collaborator) suggested a generic cloneable `AnyError` wrapper. 289 days.
- **CI:** stale
- **Summary:** Adds constructors for error types so custom backends can return them (42 lines, 3 files). Boxing the custom error breaks `Clone`/`PartialEq`/`serde` derives on `RequestAdapterError`, `RequestDeviceErrorKind`, `CreateSurfaceErrorKind`. Author says a better design is needed.
- **Move-along action:** decide whether a custom-backend error wrapper (`AnyError`-style) is wanted; if not, close

#### [#8954](https://github.com/gfx-rs/wgpu/issues/8954): fix(naga): Fix builtin metal imports ([@madsmtm](https://github.com/madsmtm)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (3 failed: Test Linux x86_64, Test Mac aarch64, Test Windows x86_64); last update 2026-01-28.
- **State:** stalled
- **Waiting on:** maintainer (decision: how to fix `using namespace metal;` removal for old macOS: 1) re-add it and reserve its names, 2) prefix all types with `metal::`, 3) import each used type explicitly)
- **Last activity:** 2026-01-28, madsmtm, asked kvark for input. 282 days. No reply.
- **CI:** stale
- **Summary:** Author tried one approach (94 files, mostly snapshot churn); description says "TODO: Solution?". Helps macOS 10.12 (11 failing tests left with [#8953](https://github.com/gfx-rs/wgpu/issues/8953) applied).
- **Move-along action:** decide which of the three options is acceptable (kvark was asked, no answer); or ping kvark

#### [#9023](https://github.com/gfx-rs/wgpu/issues/9023): Remove layer from superlayer in `Drop` if needed ([@madsmtm](https://github.com/madsmtm)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (49 failed: 01-hello-compute, 02-hello-window, API Operation Linux x86_64, API Operation Mac aarch64); last update 2026-02-08.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-02-08, madsmtm, opened. 271 days. No comments.
- **CI:** stale
- **Summary:** Fixes the macOS CALayer leak in [#9021](https://github.com/gfx-rs/wgpu/issues/9021) (still open) by removing the layer from its superlayer on `Surface` drop (52/-7 lines, 5 files). Marked "Not yet tested". Clippy and test boxes unchecked.
- **Move-along action:** ping author (is it tested? rebase, then mark ready)

#### [#9053](https://github.com/gfx-rs/wgpu/issues/9053): Host image copies ([@inner-daemons](https://github.com/inner-daemons)) [draft PR]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-08-25.
- **Status:** Adds `HOST_VISIBLE` texture usage and host-side texture copies (Vulkan `VK_EXT_host_image_copy`, Metal, DX12 UMA), mapped-texture lifecycle like buffer mapping, tests and a CHANGELOG entry. Large (130 files, +3275/-69, 63 commits). Not close to ready: cwfitzgerald's CHANGES_REQUESTED review (2026-08-14) lists 12 correctness findings and has no author reply. The last pushes are two trunk merges (Jul 28, Aug 25); the Aug 25 merge reworks the code for the arc-based wgpu-core, so the review may be partly stale.
- **Waiting on:** author (the review of 08-14 is unanswered; the PR is still draft)
- **Recommended action:** Reply with a position: ask the author to address the 08-14 review, split the `TextureDescriptor` change into its own first commit, and re-run the checks against the new arc-based code before un-drafting. CI is stale evidence for the new head.
- **Notes:**
  - CI: green on the head `a4c8c59` (65 checks, Aug 25; tests on Linux, Windows and Mac). The review says the new tests may silently skip on lavapipe/WARP if they report no host copy support. Green does not prove the feature runs.
  - Review blockers: queued map leaked when the encoder is invalidated; callback dropped on validation errors; `write_texture` not rejected while `MappingQueued`; false `TextureStillMapped` for render-then-map in one submit; Vulkan extension dependencies and 1.3 gate; zero-extent copies; DX12 `UMA` vs `CacheCoherentUMA`, `u32` overflow, leaks; WebGPU backend ignores `mapped_at_creation`; CHANGELOG example fails validation. Unverified DX12 items need a Windows box.
  - Maintainer questions: is the "always mapped as writable, textures in general layout" model acceptable? Is the new `TextureUsages::HOST_VISIBLE = 1<<18` bit placement (skips 17) intended? Do you want this to land as one PR, given its size?

**Deeper review ([#9053](https://github.com/gfx-rs/wgpu/issues/9053): Host image copies (draft)):**

> - **Open questions from the first pass:** The diff was not read. For each of cwfitzgerald's 12 findings from the 2026-08-14 review, does the current head address it?
> - **Answers:** The review (an issue comment on 2026-08-14 with the CHANGES_REQUESTED review `4939997184`) is against `54f2f4f6`. The only later commit is `a4c8c598`, a trunk merge (2026-08-25). I compared the PR-only diff at `54f2f4f6` (against its trunk parent `4fe6176a`) with the PR-only diff at head (against `f2520958`). The differences are only adaptations to trunk API changes (error-sink arguments, `.device()` accessors, `mapped_at_creation: false` in new trunk code). I then checked each finding at the head:
>   1. **[not addressed]** A queued map leaks when the encoder is invalidated. `take_pending_texture_maps` still returns `Vec::new()` for `Error(_)` (`wgpu-core/src/command/mod.rs:436-443`). `invalidate` (`mod.rs:412`) drops the `pending_texture_maps`.
>   2. **[not addressed]** The callback is dropped on validation errors. `map_texture_on_completion_inner` moves `callback` into the closure. On `require_features`, `MissingHostVisibleUsage`, `AlreadyMappedOrQueued` or an `EncoderStateError` the callback is never called (`wgpu-core/src/command/map_texture.rs:35-90`).
>   3. **[not addressed]** `write_texture` rejects only `Mapped`, not `MappingQueued` (`wgpu-core/src/device/queue.rs:1026`).
>   4. **[not addressed]** False-positive `TextureStillMapped` for render-then-map in one submit. The `is_own_map` check per command buffer is unchanged (`queue.rs:2191-2207`).
>   5. **[not addressed]** Vulkan extension dependencies. Only `ext::host_image_copy` is pushed (`wgpu-hal/src/vulkan/adapter.rs:1348-1349`). `copy_commands2` and `format_feature_flags2` are still missing. A TODO is left at `adapter.rs:3469`.
>   6. **[not addressed]** Below Vulkan 1.3 the feature is reported but does nothing. `Features::HOST_IMAGE_COPY` is reported from the feature bit with no version gate (`adapter.rs:1124-1125`), while `Tfc::HOST_COPY` is gated on `>= 1.3` (`adapter.rs:3215-3231`).
>   7. **[not addressed]** No zero-extent guard. `Texture::copy_to_memory` and `copy_from_memory` (`wgpu-core/src/resource.rs:2212-2296`, `~2300-2380`) pass the size directly to the HAL.
>   8. **[not addressed]** DX12 is gated on `UMA`, not `CacheCoherentUMA` (`wgpu-hal/src/dx12/adapter.rs:499-502`). `cache_coherent` is still only stored (`adapter.rs:445`) and never read.
>   9. **[not addressed]** DX12 `u32` overflow in `depth_pitch`. `row_pitch * rpi` is unchanged (`wgpu-hal/src/dx12/device.rs:657,700`).
>   10. **[not addressed]** Leaks on the new error paths. Vulkan `transition_image_layout` failure destroys the image but not the allocation (`wgpu-hal/src/vulkan/device.rs:1237-1245`). DX12 `resource.Map(0, None, None)?` returns early and leaks the allocation (`dx12/device.rs:602`).
>   11. **[not addressed]** The WebGPU backend ignores `mapped_at_creation` for textures (`wgpu/src/backend/webgpu.rs:2494-2499`). `unmap` is still `unimplemented!()` (`webgpu.rs:3098-3100`). `usage.bits()` is still forwarded raw.
>   12. **[not addressed]** The CHANGELOG example still calls `texture.unmap()` while `mapped` is alive, and still uses `ctx.device` (`CHANGELOG.md:67-97`).
>
>   The secondary items are also unchanged: the `register_texture_maps` silent drop (`wgpu-core/src/device/life.rs:465-468`) and `HOST_VISIBLE = 1 << 18` skipping bit 17 (`wgpu-types/src/texture.rs:222`). The request to put the `TextureDescriptor` change in a separate first commit is also not done: the head is still a 63-commit branch with trunk merges.
> - **New findings:** none (I did not do a fresh full review. The head is the reviewed code plus a merge).
> - **Verdict change:** none (draft; waiting on the author; none of the 12 findings is addressed).
> - **Move-along action change:** none. The existing "address the 08-14 review" reply is still correct. Optionally, set a deadline for an update before closing.
>
> ---

#### [#9128](https://github.com/gfx-rs/wgpu/issues/9128): [core] Implement WebGPUError for WaitIdleError. ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-06-24.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-06-24, jimblandy, moved to draft: error reporting from Firefox background polling is not resolved and they are "no longer sure we need this". 135 days.
- **CI:** stale
- **Summary:** Adds a 12-line `WebGpuError` impl for `WaitIdleError` for the Firefox poll-from-thread work. teoxoy noted device-loss can be ignored, other variants may need to lose the device.
- **Move-along action:** close (author does not need it now; reopen with the polling design)

#### [#9353](https://github.com/gfx-rs/wgpu/issues/9353): fix(wgpu-hal/vk): don't use sequential bindings if pipeline layout is set explicitly ([@dylanblokhuis](https://github.com/dylanblokhuis)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI 1 cancelled jobs; last update 2026-11-15.
- **State:** stalled
- **Waiting on:** maintainer (decision: how passthrough/SPIR-V shaders get explicit binding indices; `bgl::Origin::Derived` is rejected as the switch)
- **Last activity:** 2026-11-15, 5andr0, comment: hit [#9326](https://github.com/gfx-rs/wgpu/issues/9326) with passthrough SPIR-V, and offered an agent-written redraft with an opt-in flag and tests.
- **CI:** unknown (status API empty)
- **Summary:** In Vulkan, keeps user binding indices for explicitly set layouts instead of sequential ones (34/-16 lines, 3 files; labels vulkan, extensions). andyleiserson says explicit BGLs do not imply passthrough shaders and web content must not control this; a layout can serve both Naga and passthrough shaders. No test added. Inner-daemons links it to a future precompiled-shaders API.
- **Move-along action:** decide whether to handle this now (an opt-in flag, e.g. `PRESERVE_BINDING_INDICES`) or wait for the precompiled-shaders API; then close or ping author

#### [#9380](https://github.com/gfx-rs/wgpu/issues/9380): use threadgroup pointers instead of references in metal ([@39ali](https://github.com/39ali)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-07-15.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-12, 39ali (author update; PR updated 2026-08-12). Last human comment 2026-07-15 by inner-daemons asking if the author will return. 86 days since update.
- **CI:** green on 2026-04-21 run (58 check runs seen, codecov re-ran 2026-08-05 green); may be stale against trunk.
- **Summary:** Metal backend uses threadgroup pointers instead of references (296/-203 lines, 16 files), aims to fix [#4500](https://github.com/gfx-rs/wgpu/issues/4500) (still open, driver miscompile of workgroup memory). inner-daemons requires mesh/task shader paths to be covered before landing (thread r3128609110); teoxoy agrees at least that thread must be resolved.
- **Move-along action:** ping author (cover mesh/task shaders, rebase) or decide to split mesh/task shaders into a follow-up issue

#### [#9427](https://github.com/gfx-rs/wgpu/issues/9427): fix(core): dont panic in Queue::drop ([@atlv24](https://github.com/atlv24)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-08-07.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-12, PR updated (author push or edit); last human comment 2026-08-07, inner-daemons: "Should this be closed?". 86 days.
- **CI:** unknown (status API empty)
- **Summary:** Removes a panic in `Queue::drop` (2/-8 lines, 1 file), follow-up to [#9361](https://github.com/gfx-rs/wgpu/issues/9361) (merged/closed). On 2026-04-16 teoxoy said after the maintainer call the best option is to abort the process on OOM, not retry; PR has not been updated to do that. No test.
- **Move-along action:** ping author (implement abort per maintainer call) or close

#### [#9485](https://github.com/gfx-rs/wgpu/issues/9485): GLES external texture implementation ([@cmeissl](https://github.com/cmeissl)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-08-07.
- **State:** stalled
- **Waiting on:** author (teoxoy requested changes on 2026-06-24; inner-daemons confirmed "waiting on author" on 2026-08-07)
- **Last activity:** 2026-08-07, inner-daemons, pinged author. Author's last comment 2026-06-09. 91 days.
- **CI:** green but stale (all checks pass on a 2026-06-22 run; PR base is old, only codecov re-ran 2026-08-05)
- **Summary:** Adds GLES external texture via OES_EGL_image_external_essl3 (688 additions, 17 files, 3 commits). Open items: address teoxoy's change requests, add the `texture_from_raw_external_oes` helper (teoxoy approved the idea), docs for external sampler use, CHANGELOG.
- **Move-along action:** ping author (or close as inactive if no reply; closes [#4528](https://github.com/gfx-rs/wgpu/issues/4528) and is part of [#4386](https://github.com/gfx-rs/wgpu/issues/4386), so keep unless the author is gone)

#### [#9512](https://github.com/gfx-rs/wgpu/issues/9512): [core] Doc fixes related to mapping callback management ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-05-07.
- **State:** close to ready
- **Waiting on:** author (nothing blocks; only the draft flag)
- **Last activity:** 2026-05-07, andyleiserson, approved. PR updated 2026-07-22 (cause not visible). 183 days since approval.
- **CI:** not checked (28 additions, comment-only change)
- **Summary:** Comment-only change (3 files) on mapping callback handling. Already approved by assignee andyleiserson. Base is from May, so it may need a rebase.
- **Move-along action:** ping author to mark ready and rebase, then merge

**CI and status check ([#9512](https://github.com/gfx-rs/wgpu/issues/9512): [core] Doc fixes related to mapping callback management (draft)):**

> - **Open questions from the first pass:** CI not checked; merge state.
> - **Answers:**
>   1. [confirmed] CI: 58 check runs, all success. Head d7a3308. Old run (PR updated 2026-07-22).
>   2. [confirmed] `mergeable_state: dirty` (conflict). Needs a rebase.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none ("close to ready" stays: green but stale, conflict, still draft)

#### [#9523](https://github.com/gfx-rs/wgpu/issues/9523): Support `SUBGROUP_SIZE_CONTROL`: add `subgroup_size` in `PassthroughShaderEntryPoint` ([@ruihe774](https://github.com/ruihe774)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-08-12.
- **State:** stalled
- **Waiting on:** author (changes requested by inner-daemons 2026-06-15 and andyleiserson 2026-06-18; author said on 2026-07-22 they would revise "this week")
- **Last activity:** 2026-08-12, inner-daemons, pinged author again. Author's last comment 2026-07-22. 86 days.
- **CI:** green but stale (checks from 2026-05-26 pass; only codecov re-ran 2026-08-05)
- **Summary:** Adds `Features::SUBGROUP_SIZE_CONTROL` (Vulkan only) and a `SubgroupSize` enum (Varying/Full/Fixed) on passthrough entry points, with validation and a test. 688 additions, 20 files. Missing: trimming of long comments, review fixes, naga side. Also aligns with the gpuweb proposal [#5545](https://github.com/gfx-rs/wgpu/issues/5545).
- **Move-along action:** ping author once more; close if no reply (long-running, already pinged twice)

#### [#9659](https://github.com/gfx-rs/wgpu/issues/9659): [metal] Implement MULTI_DRAW_INDIRECT_COUNT via compute shader emulation ([@Bromles](https://github.com/Bromles)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-07-21.
- **State:** superseded (by [#9679](https://github.com/gfx-rs/wgpu/issues/9679))
- **Waiting on:** maintainer (decision: close, or keep as fallback until [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands)
- **Last activity:** 2026-06-15, cwfitzgerald, moved to draft until [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands; author agreed. 144 days.
- **CI:** not checked (stale base)
- **Summary:** Compute-shader emulation of count draws on Metal (1189 additions, 19 files). [#9679](https://github.com/gfx-rs/wgpu/issues/9679) (non-draft, still open, updated 2026-11-05, assigned to you and inner-daemons) does the same with Metal ICBs, and keeps the count on the GPU. You said you expect to take [#9679](https://github.com/gfx-rs/wgpu/issues/9679).
- **Move-along action:** close (superseded by [#9679](https://github.com/gfx-rs/wgpu/issues/9679), thank the author) once [#9679](https://github.com/gfx-rs/wgpu/issues/9679) is accepted; keep open only until then

#### [#9794](https://github.com/gfx-rs/wgpu/issues/9794): Experiment: Port VMA/D3D12MA to Rust in place of existing allocator stack ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (5 failed: Clippy WebAssembly, Clippy WebAssembly 64-bit, Test WebAssembly, publish); last update 2026-08-30.
- **State:** early / experiment
- **Waiting on:** author (needs their time to review and iterate; no reviewer assigned)
- **Last activity:** 2026-08-18, cwfitzgerald, said they still plan to merge it ("I just need to spend time on it"). 80 days. Last update to the PR 2026-08-30 (66 days).
- **CI:** none found in this read (not fetched). They said earlier it passes all tests.
- **Summary:** LLM-assisted Rust port of VMA/D3D12MA to replace gpu-allocator. 12k lines added, 41 files. Body says "Don't merge this".

#### [#9809](https://github.com/gfx-rs/wgpu/issues/9809): Make `WriteOnly` use only volatile writes ([@kpreid](https://github.com/kpreid)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (45 failed: 01-hello-compute, 02-hello-window, 03-hdr-surface, API Operation Linux x86_64); last update 2026-10-09.
- **State:** early / experiment
- **Waiting on:** maintainer (decision: do we want volatile writes in `WriteOnly` at all, given the perf cost and that const-ness depends on an unstable Rust feature, [rust-lang/rust#159094](https://github.com/rust-lang/rust/issues/159094)?)
- **Last activity:** 2026-10-09, kpreid, rebased and restored `const fn`; says "still not sure what we want to do with this PR". 28 days.
- **CI:** red, likely unrelated (nearly all jobs fail, including clippy on all targets, MSRV, doctest, examples; only Test Linux and Test Mac pass). Run is 2026-10-09. Not diagnosed; may be the unstable `const` volatile feature or CI-wide breakage.
- **Summary:** Replaces pointer writes in `WriteOnly` with `write_volatile` for soundness on write-combining memory (part of [#8897](https://github.com/gfx-rs/wgpu/issues/8897)). Small (24 additions, 50 deletions). Cannot merge until the const volatile feature is stable, and no benchmarks exist.
- **Move-along action:** decide whether to merge without `const fn` (as in the first version) or wait for the Rust feature

#### [#9843](https://github.com/gfx-rs/wgpu/issues/9843): Experimental AI slop naga cli rewrite ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-11-01.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated (probably a push). 5 days.
- **CI:** unknown (legacy status pending, check runs not read). Likely not meaningful.
- **Summary:** Full naga CLI rewrite, 105 files, +7206/-1379, 65 commits. Description says it is fully vibe coded and only open to view the diff on GitHub while the author tests the CLI locally. Claims to close [#9065](https://github.com/gfx-rs/wgpu/issues/9065) and [#8867](https://github.com/gfx-rs/wgpu/issues/8867). Checklist all unchecked.
- **Move-along action:** nothing (experiment; author is active). If it never becomes a real series, ask author to close it once the local experiment is done.

#### [#9853](https://github.com/gfx-rs/wgpu/issues/9853): Upstream Bindless Implementation ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (8 failed: Clippy WebAssembly, Clippy WebAssembly 64-bit, Test Linux x86_64, Test Mac aarch64); last update 2026-08-30.
- **State:** in progress
- **Waiting on:** author. Open: mutable bindings, WGSL bounds checking, tests ("TODO"), full human audit.
- **Last activity:** 2026-08-30 (PR updated, actor not checked). 68 days.
- **CI:** not fetched.
- **Summary:** Partial implementation of the WebGPU bindless proposal with Resource Tables. Closes [#8557](https://github.com/gfx-rs/wgpu/issues/8557). 231 files, 12k lines.

#### [#9863](https://github.com/gfx-rs/wgpu/issues/9863): WIP wgpu-hal: Use `wayland-egl` instead of `wayland-sys` ([@ids1024](https://github.com/ids1024)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (5 failed: Clippy Emscripten, Clippy IOS aarch64, Clippy VisionOS aarch64, Format & Typos); last update 2026-07-13.
- **State:** early / experiment
- **Waiting on:** other (upstream wayland-rs release, [Smithay/wayland-rs#900](https://github.com/Smithay/wayland-rs/issues/900))
- **Last activity:** 2026-07-13, ids1024, opened. No comments or reviews. 116 days.
- **CI:** not checked
- **Summary:** Swaps `wayland-sys` for the `wayland-egl` crate in wgpu-hal (62 additions, 3 files). Depends on the next wayland-rs release. Checklist is empty.
- **Move-along action:** nothing (blocked upstream; check whether wayland-rs has released and ask the author if still wanted)

#### [#9888](https://github.com/gfx-rs/wgpu/issues/9888): Add an example that provokes VUID-vkAcquireNextImageKHR-semaphore-01286 ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-07-29.
- **State:** stalled
- **Waiting on:** maintainer (decision: do we merge bug-repro examples that CI cannot run?)
- **Last activity:** 2026-07-29, kpreid, review comment. 100 days.
- **CI:** not checked
- **Summary:** Adds a Vulkan example (261 lines, 3 files) that shows a validation-layer error. ErichDonGubler requested changes (use a regression test or an issue MRE). jimblandy agreed it needs to become a test. inner-daemons argued the example form is right because surface/swapchain cases cannot run in CI.
- **Move-along action:** decide the policy on repro examples; then tell the author, or close (the underlying bug may be tracked elsewhere)

#### [#9894](https://github.com/gfx-rs/wgpu/issues/9894): Expect pass: `wgpu_gpu::clear_texture::clear_texture_uncompressed` ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (3 failed: Test Linux x86_64, Test WebAssembly, Test Windows x86_64); last update 2026-07-21.
- **State:** in progress
- **Waiting on:** other (depends on [#9903](https://github.com/gfx-rs/wgpu/issues/9903), which is merged on 2026-07-20; author must rebase and re-run CI)
- **Last activity:** 2026-07-18, jimblandy, confirmed the test fails on Mesa 25.2.7 and passes on 26.0.3. 111 days.
- **CI:** stale (earlier run failed on llvmpipe Mesa 25.2.7; [#9903](https://github.com/gfx-rs/wgpu/issues/9903) has since updated CI Mesa)
- **Summary:** Removes the GL `expect_fail` for one test (1 addition, 11 deletions). Prerequisite [#9903](https://github.com/gfx-rs/wgpu/issues/9903) (Mesa 26.1.3 in CI) is merged, so this should now pass after a rebase.
- **Move-along action:** ping author to rebase, confirm CI is green, mark ready, then review

**CI and status check ([#9894](https://github.com/gfx-rs/wgpu/issues/9894): Expect pass: `wgpu_gpu::clear_texture::clear_texture_uncompressed` (draft)):**

> - **Open questions from the first pass:** CI state; conflict state.
> - **Answers:**
>   1. [confirmed] Red on the old head: 59 runs, 56 success, 3 failure: Test Linux x86_64, Test WebAssembly, Test Windows x86_64 (2026-07-21). Each ends with "exit code 1". Logs are expired (HTTP 410).
>   2. [unknown] Cause. The first pass says the run was on llvmpipe Mesa 25.2.7, before [#9903](https://github.com/gfx-rs/wgpu/issues/9903) moved CI to Mesa 26.1.3. That matches 3 test failures, but I cannot see the logs. Test WebAssembly and Test Windows failing too is not explained by a Linux GL change; this needs a fresh run.
>   3. [confirmed] `blocked`, `mergeable: true`. No conflict. CI on the head predates [#9903](https://github.com/gfx-rs/wgpu/issues/9903), so a rebase is still needed to get a valid result.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#9897](https://github.com/gfx-rs/wgpu/issues/9897): [DO NOT MERGE] Example to probe expected alpha mode on macOS/metal ([@nicoburns](https://github.com/nicoburns)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-07-17.
- **State:** obsolete
- **Waiting on:** author (nothing needed)
- **Last activity:** 2026-07-17, nicoburns, opened. No comments. 112 days.
- **CI:** not checked
- **Summary:** Repro example for issue [#9896](https://github.com/gfx-rs/wgpu/issues/9896) (Metal advertised PostMultiplied instead of PreMultiplied alpha). [#9896](https://github.com/gfx-rs/wgpu/issues/9896) is closed (2026-08-18), fixed by merged [#9922](https://github.com/gfx-rs/wgpu/issues/9922). The example has no further use.
- **Move-along action:** close (repro for a fixed, closed issue; marked DO NOT MERGE)

#### [#9909](https://github.com/gfx-rs/wgpu/issues/9909): [hal vulkan] Clean up Vulkan Validation Layer callback ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (34 failed: 01-hello-compute, 02-hello-window, API Operation Linux x86_64, API Operation Windows x86_64); last update 2026-07-27.
- **State:** close to ready
- **Waiting on:** author (only the draft flag; CI to re-run)
- **Last activity:** 2026-07-27, inner-daemons, approved (ErichDonGubler approved 2026-07-20 with "minus CI being aang-y"). 102 days.
- **CI:** red, likely unrelated (run of 2026-07-19: most jobs fail, including clippy on all native targets, MSRV, doctest, tests; passes only format, wasm, deny, miri). Same pattern as [#9809](https://github.com/gfx-rs/wgpu/issues/9809), so probably a CI-wide problem at those times. Stale; not diagnosed.
- **Summary:** Removes suppressions for VUIDs 01912, 04627, 00184 (upstream fixed over a year ago) and fixes comment typos (2 additions, 38 deletions). Two approvals. The risk is that removing suppressions exposes validation errors in CI; it needs a fresh green run.
- **Move-along action:** ping author to rebase, mark ready, and get a green CI run; then merge

**CI and status check ([#9909](https://github.com/gfx-rs/wgpu/issues/9909): [hal vulkan] Clean up Vulkan Validation Layer callback (draft)):**

> - **Open questions from the first pass:** CI state; conflict state.
> - **Answers:**
>   1. [confirmed] Red: 59 runs, 34 failure and 25 success. Failures include Clippy on all native targets, Core and wgpu MSRV checks, Doctest, Test Linux/Mac/Windows, API Operation/Validation, Other, 01-hello-compute, 02-hello-window. Check runs date from 2026-07-19.
>   2. [unknown] Cause. The failing jobs ran about 40 s each (for example job 88232550675, Clippy Linux x86_64: 20:13:52 to 20:14:31) and ended with "Process completed with exit code 101" (a cargo error). Logs are expired (HTTP 410). A 40 s exit-101 on every build job points to a build or toolchain problem, not a test failure, but I cannot confirm. This head touches only HAL Vulkan suppressions (2 additions, 38 deletions).
>   3. [confirmed] `mergeable_state: blocked`, `mergeable: true`. No conflict.
> - **New findings:** none
> - **Verdict change:** none ("close to ready" holds only after a fresh CI run on a current base)
> - **Move-along action change:** none

#### [#9975](https://github.com/gfx-rs/wgpu/issues/9975): Add `textureSampleLevel:sampled_{1,2,3}d` coverage in CTS ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI 1 cancelled jobs; last update 2026-08-05.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-05, ErichDonGubler, push (93 days). No comments or reviews.
- **CI:** stale (2026-08-05). All green except "Other Windows x86_64", which was cancelled after the 6 hour timeout.
- **Summary:** Adds 3 lines to one file to enable these CTS cases. The description says macOS CTS "Other" takes about 30 min and the Windows run is "massively long" and times out at 6 h. Coupled to issue [#9974](https://github.com/gfx-rs/wgpu/issues/9974) and motivated by [#9951](https://github.com/gfx-rs/wgpu/issues/9951).
- **Move-along action:** decide whether the CTS time budget allows this (it blocks on CI cost); if not, close. Otherwise ping author.

#### [#10008](https://github.com/gfx-rs/wgpu/issues/10008): Add CTS coverage for `maxStorage{Buffer,Texture}sIn{Vertex,Fragment}Stage` limits ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-09-07.
- **State:** in progress
- **Waiting on:** unclear (coupled to [#10006](https://github.com/gfx-rs/wgpu/issues/10006) and [#9974](https://github.com/gfx-rs/wgpu/issues/9974), both issues; state unknown to me)
- **Last activity:** 2026-09-07, ErichDonGubler, push (60 days). No comments or reviews.
- **CI:** green (2026-09-07). Windows "Other" passed here, in 23 min.
- **Summary:** A 7-line CTS change. Resolves [#10007](https://github.com/gfx-rs/wgpu/issues/10007). Draft because of its coupling to [#10006](https://github.com/gfx-rs/wgpu/issues/10006) and [#9974](https://github.com/gfx-rs/wgpu/issues/9974).
- **Move-along action:** ping author (ask what blocks leaving draft; CI is green and the change is tiny).

**CI and status check ([#10008](https://github.com/gfx-rs/wgpu/issues/10008): Add CTS coverage for `maxStorage{Buffer,Texture}sIn{Vertex,Fragment}Stage` limits (draft)):**

> - **Open questions from the first pass:** CI state; conflict state.
> - **Answers:**
>   1. [confirmed] 63 runs, all success. Head a31d2fb (2026-09-07).
>   2. [confirmed] `blocked`, `mergeable: true`. No conflict.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10114](https://github.com/gfx-rs/wgpu/issues/10114): Can we move entry point argument I/O validation into `VaryingContext::validate_impl`? ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (4 failed: Test Linux x86_64, Test Mac aarch64, Test Windows x86_64, Validate: SPIR-V/GLSL/DOT/WGSL); last update 2026-08-24.
- **State:** early / experiment
- **Waiting on:** other reviewer (andyleiserson, per the description: "Will bring out of draft after discussing with [@andyleiserson](https://github.com/andyleiserson)")
- **Last activity:** 2026-09-01, ErichDonGubler, push (66 days). No comments or reviews.
- **CI:** red, likely related. Test Linux/Windows/Mac and "Validate: SPIR-V/GLSL/DOT/WGSL" failed on 2026-09-01; I did not read the logs. The PR is stacked on [#10026](https://github.com/gfx-rs/wgpu/issues/10026) (naga validation).
- **Summary:** One-file naga experiment (+22/-35) built on top of [#10026](https://github.com/gfx-rs/wgpu/issues/10026) in reply to a review thread there. Author is not sure it is useful.
- **Move-along action:** close (stale experiment; check first whether [#10026](https://github.com/gfx-rs/wgpu/issues/10026) has landed, then ask author to close or rebase).

#### [#10212](https://github.com/gfx-rs/wgpu/issues/10212): Record a pass's first use of a texture subresource, not its last ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-18
- **State (2026-11-20):** ready; has conflicts; CI green; last update 2026-11-15.
- **Verdict:** Status note
- **Waiting on:** andyleiserson (assigned on 2026-11-18). No review yet.
- **Last activity:** 2026-11-18, cwfitzgerald, assigned andyleiserson. The last push was on 2026-11-15.
- **Approvals:** none (review decision REVIEW_REQUIRED)
- **CI:** green (66 pass, 1 skipped). The test failure on GL is gone: `issue_10199.rs:97-104` makes both views `D2Array` ("A `D2` view of an array texture is not valid in compatibility mode").
- **Summary:** Fixes [#10199](https://github.com/gfx-rs/wgpu/issues/10199). The pass tracker merge records the first use of a subresource as the start state, not the last, which gives correct Vulkan layouts. It changes `wgpu-core/src/track/texture.rs` and adds a regression test and a changelog entry.
- **Findings:** none (status only). Disclosure: both agent boxes are checked; the text says an LLM found and bisected the bug, an agent opened the PR, and the author edited it. The self-review box is checked.
- **To land:** one member approval. It is small and can land before the release.
- **Move-along action:** Ask andyleiserson to review before the release.
- **Depth:** status only

#### [#10282](https://github.com/gfx-rs/wgpu/issues/10282): Change WebGPU backend panics to error callback insertion ([@Wumpf](https://github.com/Wumpf)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (1 failed: Format & Typos); last update 2026-09-08.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-09-08, Wumpf, opened (59 days). No comments or reviews.
- **CI:** red, probably unrelated to logic: only "Format & Typos" failed; all other jobs green (2026-09-08).
- **Summary:** Replaces panics in the WebGPU backend with synthetic errors injected through the error scope and uncaptured-error callbacks (+1076/-222, 5 files). Would fix [#8005](https://github.com/gfx-rs/wgpu/issues/8005) and [#9710](https://github.com/gfx-rs/wgpu/issues/9710). The description's TODO list is open: testing ("WIP"), commit splitting, self-review, and whether the poisoning always works.
- **Move-along action:** ping author (status after 59 days; still wanted?). Maintainer decision later: is synthetic-error injection an accepted design?

#### [#10287](https://github.com/gfx-rs/wgpu/issues/10287): [metal] correct ray query semantics and BLAS transform handling ([@acoliver](https://github.com/acoliver)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI no runs; last update 2026-09-08.
- **State:** in progress (checklist is complete and testing is described, but it is stacked and unreviewed)
- **Waiting on:** maintainer (decision: how to land the Metal RT stack; see the section above). Also needs [#10292](https://github.com/gfx-rs/wgpu/issues/10292) first.
- **Last activity:** 2026-09-09, acoliver, update (58 days).
- **CI:** none (no check runs reported).
- **Summary:** Adds a ray query state machine to naga's MSL lowering, fixes Metal intersector setup (row-major transforms, flags), fixes a SPIR-V helper cache bug, and fixes the ray tracing test context. Own commit is one on top of [#10292](https://github.com/gfx-rs/wgpu/issues/10292). Tests were run on an M4 Max only. Large diff (34 files).
- **Move-along action:** decide (the stack order and whether to split the naga/SPIR-V fixes into their own PRs; [#10292](https://github.com/gfx-rs/wgpu/issues/10292) first).

**Deeper review ([#10287](https://github.com/gfx-rs/wgpu/issues/10287): [metal] correct ray query semantics and BLAS transform handling [draft]):**

> - **Open questions from the first pass:** stack order; whether to split the naga/SPIR-V fixes.
> - **Answers:**
>   1. [confirmed] Its own commit `7067e4ee` depends on [#10292](https://github.com/gfx-rs/wgpu/issues/10292) only through `wgpu-hal/src/metal/command/tests.rs`. It adds a 70-line hal test to that file, which [#10292](https://github.com/gfx-rs/wgpu/issues/10292) creates. The naga MSL state machine, the `conv.rs` fixes, the tvOS/visionOS gate and the test-context identity matrix do not depend on the AS sync work.
>   2. [confirmed] The SPIR-V helper-cache fix (`naga/src/back/spv/ray/query.rs`, `Proceed` -> `Terminate`) is already on trunk (`naga/src/back/spv/ray/query.rs:1491` at HEAD `ea947da4` inserts `Terminate`). That part of the PR, with its test and CHANGELOG line, is obsolete.
> - **New findings:**
>   1. [medium] A real trunk bug that can land alone: `wgpu-hal/src/metal/conv.rs:447` and `:478` (trunk) call `descriptor.allowDuplicateIntersectionFunctionInvocation()`, which is the getter, so the flag is never set. `7067e4ee` replaces it with the setter. The `setTransformationMatrixLayout(RowMajor)` change and the identity-matrix test fix (`tests/tests/wgpu-gpu/ray_tracing/mod.rs:86` on trunk is `[1,0,0,0, 1,0,0,0, 1,0,0,0]`, a rank-1 matrix) are also small and independent.
> - **Verdict change:** none (in progress). Note: the SPIR-V part is stale.
> - **Move-along action change:** ask the author to take out the hal `conv.rs` fixes and the test identity matrix as a small PR on trunk. Drop the SPIR-V part. Keep the MSL ray-query state machine as its own naga PR.

**CI and status check ([#10287](https://github.com/gfx-rs/wgpu/issues/10287), [#10288](https://github.com/gfx-rs/wgpu/issues/10288), [#10289](https://github.com/gfx-rs/wgpu/issues/10289), [#10292](https://github.com/gfx-rs/wgpu/issues/10292): acoliver Metal ray-tracing stack (drafts)):**

> - **Open questions from the first pass:** CI "none"; conflict state.
> - **Answers:**
>   1. [confirmed] CI: none for all four. 0 check runs and 0 workflow runs (`actions/runs?head_sha=...` total_count 0) for 5be6365 ([#10287](https://github.com/gfx-rs/wgpu/issues/10287)), d4d91fe ([#10288](https://github.com/gfx-rs/wgpu/issues/10288)), f8dafeb ([#10289](https://github.com/gfx-rs/wgpu/issues/10289)), 91aa55e ([#10292](https://github.com/gfx-rs/wgpu/issues/10292)).
>   2. [unknown] Whether it is awaiting approval. There are no `action_required` runs for these SHAs, so no run was even created. Drafts may not trigger workflows in this repo, or the fork workflows were never started. I cannot tell from the API.
>   3. [confirmed] `mergeable_state: dirty` (conflict) for all four.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10288](https://github.com/gfx-rs/wgpu/issues/10288): [naga] require the acceleration structure binding array capability for non-uniform indexing of acceleration structure arrays ([@acoliver](https://github.com/acoliver)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI no runs; last update 2026-09-08.
- **State:** in progress
- **Waiting on:** author (depends on [#10287](https://github.com/gfx-rs/wgpu/issues/10287), which depends on [#10292](https://github.com/gfx-rs/wgpu/issues/10292))
- **Last activity:** 2026-09-09, acoliver, update (58 days).
- **CI:** none.
- **Summary:** Fixes a naga panic (`unreachable!()`) on non-uniform indexing of `binding_array<acceleration_structure>` by returning a validation error, and gates MSL emission on the capability. Adds one snapshot test. It is naga-only and backend-independent, so it could land alone if rebased on trunk.
- **Move-along action:** ping author (ask to rebase this naga fix onto trunk as a standalone PR, since it does not need the Metal changes), then review.

**Deeper review ([#10288](https://github.com/gfx-rs/wgpu/issues/10288): [naga] require the acceleration structure binding array capability for non-uniform indexing of acceleration structure arrays [draft]):**

> - **Open questions from the first pass:** Is the naga change independent of the stack?
> - **Answers:**
>   1. [confirmed for the analyzer fix; refuted for the PR as written]
>      - The analyzer arm (`naga/src/valid/analyzer.rs:582` at the PR head) can be applied to trunk unchanged. The trunk code at `analyzer.rs:571-585` (HEAD) sends `binding_array<acceleration_structure>` to the buffer branch. There, `global.space` is `Handle`, which hits `_ => unreachable!()`. Any runtime index into such an array panics in validation. This applies to Vulkan and DX12 users with `ACCELERATION_STRUCTURE_BINDING_ARRAY`.
>      - The rest is not independent. The MSL parts (`naga/src/back/msl/mod.rs:905` adds the capability to MSL `supported_capabilities`; `writer.rs:7511` lets AS binding arrays through for MSL 2.4 and later) are Metal enablement work that belongs with [#10289](https://github.com/gfx-rs/wgpu/issues/10289). The new snapshot `ray-query-binding-arrays` targets `METAL | SPIRV`, and its MSL output depends on the [#10287](https://github.com/gfx-rs/wgpu/issues/10287) ray-query emission.
> - **New findings:**
>   1. [low] The description says "The MSL writer additionally gates emission ... on the capability". The code gates emission on `lang_version >= 2.4` and adds the capability to MSL. The wording is inaccurate.
>   2. [low] Because the global declaration already requires the same capability (`naga/src/valid/interface.rs:997`), the new check can never fail for a module that validates. It is a panic fix and not new validation. This is correct, but the CHANGELOG line ("Reject ... unless") overstates it.
> - **Verdict change:** none.
> - **Move-along action change:** ask the author for a trunk-based PR that has only the analyzer arm and a SPIR-V-only (or validation-only) snapshot. Move the MSL lines to [#10289](https://github.com/gfx-rs/wgpu/issues/10289).

**CI and status check:** see the block under [#10287](https://github.com/gfx-rs/wgpu/issues/10287).

#### [#10289](https://github.com/gfx-rs/wgpu/issues/10289): [metal] advertise extended vertex formats and acceleration structure binding arrays ([@acoliver](https://github.com/acoliver)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI no runs; last update 2026-11-05.
- **State:** in progress
- **Waiting on:** author (depends on [#10288](https://github.com/gfx-rs/wgpu/issues/10288); also a conflict with trunk) and the maintainer for the stack decision
- **Last activity:** 2026-11-05, matthargett (contributor), comment asking if the stack is ready (1 day). Last author activity 2026-09-08 (59 days).
- **CI:** none.
- **Summary:** Enables `EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS` and `ACCELERATION_STRUCTURE_BINDING_ARRAY` on Metal and raises the per-stage AS limit from 1 to 8. Adds a stride check in wgpu-core on Metal. Large (48 files, +6.2k), 9 commits, "dirty" merge state. Description says Rebase, two commits.
- **Move-along action:** ping author (rebase; reply to matthargett's question) after the stack decision above.

**Deeper review ([#10289](https://github.com/gfx-rs/wgpu/issues/10289): [metal] advertise extended vertex formats and acceleration structure binding arrays [draft]):**

> - **Open questions from the first pass:** none specific (stack order).
> - **Answers:** n/a.
> - **New findings:**
>   1. [high, reasoned from code] `wgpu-hal/src/metal/adapter.rs:65` raises `MAX_ACCELERATION_STRUCTURES_PER_SHADER_STAGE` from 1 to 8. The budget comment at `:52-61` splits 31 buffer slots: 8 storage + 12 uniform + 8 vertex + 2 internal + 1 AS = 31. `VERTEX_BUFFER_SLOT_START = 23` (`:67`). With 8 AS, a vertex stage that uses the maximum of storage, uniform and AS bindings reaches buffer index 28 before the internal buffers. This overlaps vertex buffer slots 23-30. I found no hal check against `VERTEX_BUFFER_SLOT_START`. The PR text counts only storage and uniform ("31 minus 8 minus 12"). The stale comment "We use the last buffer for an acceleration structure" is still at `:61`.
>   2. [medium] `b5c64227` adds a backend-specific stride rule in wgpu-core (`wgpu-core/src/command/ray_tracing.rs`, `if state.device.backend() == wgt::Backend::Metal`). This is a validation divergence in a crate that processes untrusted input. It is acceptable for a native-only feature, but the maintainer should approve it explicitly.
> - **Verdict change:** none (in progress). Add the slot-budget issue to the author's list.
> - **Move-along action change:** none beyond the first pass. Also ask the author to recompute the per-stage AS limit with vertex buffers included.

**CI and status check:** see the block under [#10287](https://github.com/gfx-rs/wgpu/issues/10287).

#### [#10292](https://github.com/gfx-rs/wgpu/issues/10292): [metal] synchronize acceleration structure builds with a seeded fence ([@acoliver](https://github.com/acoliver)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI no runs; last update 2026-09-09.
- **State:** in progress (the base of the stack)
- **Waiting on:** maintainer (decision: land this instead of [#9645](https://github.com/gfx-rs/wgpu/issues/9645)) and Vecvec (assignee on [#9645](https://github.com/gfx-rs/wgpu/issues/9645), tests Metal RT)
- **Last activity:** 2026-09-09, acoliver, opened (58 days).
- **CI:** none. Merge state "dirty".
- **Summary:** Supersedes [#9645](https://github.com/gfx-rs/wgpu/issues/9645). Fixes [#9215](https://github.com/gfx-rs/wgpu/issues/9215) (TLAS build reading a BLAS that is still building) with a seeded encoder fence, per-pass AS usage tracking, and reserved command buffer slots; adds regression tests. It is a much bigger change than [#9645](https://github.com/gfx-rs/wgpu/issues/9645) (14 files, +2.6k). Checklist complete. Tested on M4 Max only.
- **Move-along action:** decide (choose [#10292](https://github.com/gfx-rs/wgpu/issues/10292) or the smaller [#9645](https://github.com/gfx-rs/wgpu/issues/9645) for [#9215](https://github.com/gfx-rs/wgpu/issues/9215); ask Vecvec to review. Needs a rebase either way).

**Deeper review ([#10292](https://github.com/gfx-rs/wgpu/issues/10292): [metal] synchronize acceleration structure builds with a seeded fence [draft]):**

> - **Open questions from the first pass:**
>   1. Does it replace [#9645](https://github.com/gfx-rs/wgpu/issues/9645) (is it "superseding" and landable)?
>   2. What does the rework contain?
> - **Answers:**
>   1. [refuted as a landable replacement; confirmed as a superset] It contains [#9645](https://github.com/gfx-rs/wgpu/issues/9645) (see [#9645](https://github.com/gfx-rs/wgpu/issues/9645), answer 1) and fixes the cross-command-buffer case. The net diff against `6e8d0bd1` is 14 files, +2588/-102, and it mixes at least five independent changes:
>      - (a) queue-wide seeded `MTLFence` for cross-command-buffer AS ordering (`mod.rs:493-531`, `:582-640`);
>      - (b) fence waits and updates in ray-query compute and render passes. These come from AS bindings or the new `BindGroup::uses_acceleration_structures` (`command.rs:274`, `end_compute_pass`, `end_render_pass`);
>      - (c) `useResource:usage:` for build inputs, scratch and compact-size buffers;
>      - (d) a new `command_buffer_slot.rs` and `MAX_UNSUBMITTED_COMMAND_BUFFERS = MAX_COMMAND_BUFFERS - 1`;
>      - (e) a new `completion.rs`: `Device::wait` uses completed handlers in place of 1 ms polling. This changes every Metal user, not only ray tracing.
>
>      The history is not reviewable commit by commit: `9e7d8b19` adds an MTLEvent chain, and `14f3ea9a` removes it.
>   2. [confirmed] The pieces needed for [#9215](https://github.com/gfx-rs/wgpu/issues/9215) are (a), optionally (b), and the issue_9215 tests. (c) is defensive. (d) is needed only because (a) allocates internal seed command buffers. (e) is unrelated to [#9215](https://github.com/gfx-rs/wgpu/issues/9215).
> - **New findings:**
>   1. [medium] `wgpu-hal/src/metal/mod.rs:529` uses `.expect(...)` when the queue fence is missing. The fence is created only when `EXPERIMENTAL_RAY_QUERY` is enabled (`device.rs:747-753`). wgpu-core gates all AS commands and AS bindings on that feature, so this cannot panic from wgpu today. It is a hidden invariant for direct hal users.
>   2. [medium] Every ray-query pass now waits for and updates the single queue fence, so all ray-query passes and AS builds on a queue are serialized on the GPU. The CHANGELOG claims "without serializing unrelated later submissions", which is true only for passes without AS. This needs a performance check before landing.
>   3. [low] `wait_for_acceleration_structure_build` can encode `waitForFence_beforeStages` in the middle of a render pass, at `set_bind_group`, after earlier draws. I did not confirm whether Metal applies a mid-pass render fence wait to the whole pass or only to later draws. Still unknown.
>   4. [low] The new hal unit tests in `wgpu-hal/src/metal/command/tests.rs` (773 lines) need a real Metal GPU with ray tracing. They print `eprintln!` traces from `#[cfg(test)]` hooks in the hot encode paths. These hooks add test-only code in production files.
>   5. [info] No CI has run on this PR (draft, merge state "dirty").
> - **Verdict change:** in progress -> needs to be split. It is not a replacement for [#9645](https://github.com/gfx-rs/wgpu/issues/9645).
> - **Move-along action change:** after [#9645](https://github.com/gfx-rs/wgpu/issues/9645) lands, ask acoliver to split the rest:
>   - (1) cross-command-buffer queue fence (a+d) with the separate-command-buffer and pending-writes issue_9215 tests;
>   - (2) ray-query pass waits (b+c);
>   - (3) `completion.rs` poll wakeups as its own Metal PR.
>
>   The tests in `ac6935b1` can go with [#9645](https://github.com/gfx-rs/wgpu/issues/9645) now.

**CI and status check:** see the block under [#10287](https://github.com/gfx-rs/wgpu/issues/10287).

#### [#10304](https://github.com/gfx-rs/wgpu/issues/10304): Extension API proposal ([@andyleiserson](https://github.com/andyleiserson)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (50 failed: 01-hello-compute, 02-hello-window, 03-hdr-surface, API Operation Linux x86_64); last update 2026-10-27.
- **State:** close to ready (as a proposal; the author will split out the actual change)
- **Waiting on:** author (split the infrastructure into a separate PR, per jimblandy's comment on 2026-10-22)
- **Last activity:** 2026-10-22, jimblandy, "I think we should go with this" and said andyleiserson will split it out (15 days). Author's last push 2026-10-28 (9 days).
- **CI:** red (2026-10-28), nearly all jobs failed within 1-3 minutes, which looks like a build/config failure from a rebase; I did not read logs. Format & Typos also failed.
- **Summary:** Proposes an extension API for passing backend (hal) data through wgpu-core, to be used first for a DX12 adapter LUID filter wanted by Firefox ([#9932](https://github.com/gfx-rs/wgpu/issues/9932), [#10011](https://github.com/gfx-rs/wgpu/issues/10011)). Later commits show other uses (swapchain hooks, present timing, device-creation callbacks, `create_texture_from_raw`). Maintainers agreed on the direction in the maintainers meeting and in comments. Open design questions: `Any` needing `'static`, naming, serde compatibility.
- **Move-along action:** nothing (author is active and has agreed to split it). Maintainer may state final approval on the direction if not already done.

#### [#10354](https://github.com/gfx-rs/wgpu/issues/10354): Add fern log capture and test shader debugPrintf output ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-13
- **State (2026-11-20):** draft; mergeable; CI red (1 failed: Test Mac aarch64); last update 2026-11-13.
- **Verdict:** Status note
- **Waiting on:** author. The PR still depends on [#9389](https://github.com/gfx-rs/wgpu/issues/9389) (Depends-On). Mergify said the dependency is satisfied on 09-15.
- **Last activity:** 2026-11-13, cwfitzgerald, push. 0 days.
- **Approvals:** none
- **CI:** red (1 fail, 64 pass). `Test Mac aarch64` fails in `wgpu_gpu::debug_printf::debug_printf_output` on Metal (Apple Paravirtual device). The shader messages "invocation=0 value=7" and three more did not arrive: "timed out waiting on channel". This is a failure of the new test on the CI runner and not an infrastructure error. Mergeable.
- **Summary:** Adds fern log capture to the GPU test harness and a test for shader debugPrintf output. It is draft work.
- **Findings:** none (status only)
- **To land:** decide if Metal on the CI runner supports debugPrintf; expect the failure on Metal or skip it there; undraft; get a review.
- **Move-along action:** Decide how to handle debugPrintf on Metal CI (expected failure or skip), then undraft.
- **Depth:** status only

#### [#10355](https://github.com/gfx-rs/wgpu/issues/10355): Naga inline pass ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (19 failed: Clippy Android ARMv7, Clippy Android aarch64, Clippy Android x86_64, Clippy Emscripten); last update 2026-11-01.
- **State:** in progress (marked experimental by the author)
- **Waiting on:** author (and [#10409](https://github.com/gfx-rs/wgpu/issues/10409) must land first)
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** red (partial view: Clippy Linux x86_64 and custom-backend failed; cause not checked, may be caused by changes in this PR).
- **Summary:** Adds a naga inlining pass with Never/Always/SPIR-V strategies; the SPIR-V one inlines only to enable unrestricted pointer parameters. Closes [#5158](https://github.com/gfx-rs/wgpu/issues/5158). Stacked on [#10409](https://github.com/gfx-rs/wgpu/issues/10409). 44 files, +4852. Open design questions in the description: whether to gate behind an instance flag, and whether `Always` helps driver compile time. Testing section empty, self-review unchecked.
- **Move-along action:** decide whether an inlining pass with a strategy option (and possibly an instance flag) is a direction you want, so the author can finish it. Review after [#10409](https://github.com/gfx-rs/wgpu/issues/10409).

#### [#10396](https://github.com/gfx-rs/wgpu/issues/10396): Update repository MSRV values together with Renovate ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI green; last update 2026-09-18.
- **State:** early / experiment
- **Waiting on:** author. Open: the description is still the empty template and checklist is unchecked. Merge state is "blocked".
- **Last activity:** 2026-09-18, opened (3 files, 244 lines). 49 days.
- **CI:** not fetched.
- **Summary:** Makes Renovate update the repo MSRV values together.

#### [#10403](https://github.com/gfx-rs/wgpu/issues/10403): gles: Support `DownlevelFlags::DEPTH_BIAS_CLAMP` ([@evilpie](https://github.com/evilpie)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (43 failed: 01-hello-compute, 02-hello-window, 03-hdr-surface, API Operation Linux x86_64); last update 2026-10-24.
- **State:** stalled
- **Waiting on:** other (upstream: new `glow` release with [grovesNL/glow#371](https://github.com/grovesNL/glow/issues/371), and [wasm-bindgen#5336](https://github.com/wasm-bindgen/wasm-bindgen/issues/5336)). Not a maintainer decision.
- **Last activity:** 2026-10-24, evilpie: "Just waiting for a new release of `glow` now." 13 days.
- **CI:** red and stale. Last run 2026-09-19. Most jobs (clippy, MSRV, tests) failed within about one minute, most likely because of the unreleased glow/wasm-bindgen dependencies. Not verified. Base is old.
- **Summary:** Small change (3 files, +19/-1) that exposes depth bias clamp on GLES. Description is the empty template. No review so far.
- **Move-along action:** nothing (blocked upstream). Optionally ask whether glow 371 is on a release path.

#### [#10409](https://github.com/gfx-rs/wgpu/issues/10409): Rework naga's compaction pass to also reorder expressions ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI 10 cancelled jobs; last update 2026-11-01.
- **State:** close to ready
- **Waiting on:** author (to undraft); then maintainer review
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** green (all pass except macOS jobs cancelled).
- **Summary:** Compaction pass now renumbers expressions into emission order, so later passes can append expressions. 7 files, +1326/-68. Description is complete: tests, benchmarks (compaction about 3% slower normally, about 70% when reordering runs), LLM disclosure, self-review and changelog checked. Needed by [#10355](https://github.com/gfx-rs/wgpu/issues/10355). Unchecked: validation/feature gates.
- **Move-along action:** ping author to undraft if it is meant to land now; it is the only PR here that looks reviewable. Maintainer should decide whether the compaction cost is acceptable.

**CI and status check ([#10409](https://github.com/gfx-rs/wgpu/issues/10409): Rework naga's compaction pass to also reorder expressions (draft)):**

> - **Open questions from the first pass:** CI state; conflict state.
> - **Answers:**
>   1. [confirmed] 67 runs: 55 success, 2 neutral, 10 `cancelled`. All 10 cancelled jobs are macOS ones (API Operation/Validation Mac, Clippy MacOS x2/IOS/VisionOS, Core and wgpu MSRV MacOS, Other Mac, Test Mac). No failure. The Mac jobs ran about 1 s (job 124892922920, 2026-11-01), so they were cancelled at start, not by a test. The check-run annotation on other PRs says macOS arm64 runners have queue delays. So the cancellation looks like infra or a concurrency cancel, not a code problem.
>   2. [confirmed] `blocked`, `mergeable: true`. No conflict.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none (a rerun of the Mac jobs is needed before landing)

#### [#10774](https://github.com/gfx-rs/wgpu/issues/10774): [naga] Implement vec2 <-> u32 bitcasts ([@DouglasDwyer](https://github.com/DouglasDwyer)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-10-28.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-10-28, DouglasDwyer, PR created. 9 days.
- **CI:** not checked. PR has merge conflicts. Depends on [#10613](https://github.com/gfx-rs/wgpu/issues/10613) (not checked).
- **Summary:** Adds i16/u16 vector to 32-bit bitcasts in frontends and backends (GLSL returns an error). 78 files, +2336/-653. Related to [#8896](https://github.com/gfx-rs/wgpu/issues/8896). Description has template text left in, "self-reviewed" and "description has enough context" unchecked. No comments or reviews.
- **Move-along action:** ping author (rebase after [#10613](https://github.com/gfx-rs/wgpu/issues/10613), finish description and self-review); close if no answer.

#### [#10783](https://github.com/gfx-rs/wgpu/issues/10783): Cover limit checks for workgroup storage size per-variable rounding in CTS ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (3 failed: API Validation Linux x86_64, API Validation Mac aarch64, API Validation Windows x86_64); last update 2026-11-18.
- **State:** stalled (blocked on upstream)
- **Waiting on:** other (gpuweb/cts PR 4723 merge and a CTS revendor), then author
- **Last activity:** 2026-10-30, ErichDonGubler, comment that CI failures are expected. 7 days. (PR updated 2026-11-05, probably a CI re-run or push.)
- **CI:** red, expected per author. Partial view (8 of 69 runs) showed passes. PR has merge conflicts.
- **Summary:** Adds CTS coverage for [#9732](https://github.com/gfx-rs/wgpu/issues/9732). 5 files, +48/-10. The new CTS path does not exist in the vendored CTS yet.
- **Move-along action:** nothing until the CTS PR lands; then ping author to revendor and rebase.

#### [#10790](https://github.com/gfx-rs/wgpu/issues/10790): [d3d12] Use the driver-reported size for committed texture allocations ([@navkagleb](https://github.com/navkagleb)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI 10 cancelled jobs; last update 2026-10-30.
- **State:** close to ready
- **Waiting on:** maintainer (decision: is the breaking change to `dx12::Device::texture_from_raw`, which now takes `&self`, acceptable? The author asks this in the body). The author may also just need to undraft.
- **Last activity:** 2026-10-30, navkagleb, opened. 7 days. No comments or reviews.
- **CI:** green (the cancelled macOS jobs and a few others were cancelled, not failed).
- **Summary:** Fixes [#10113](https://github.com/gfx-rs/wgpu/issues/10113). D3D12 committed textures used a one-mip size estimate, so the `texture_memory` counter was too low. It now uses the driver-reported size. 4 files, +35/-9, changelog added. Tested only with the halmark example.
- **Move-along action:** decide whether the `texture_from_raw` signature change is acceptable, then tell the author to undraft.

**CI and status check ([#10790](https://github.com/gfx-rs/wgpu/issues/10790): [d3d12] Use the driver-reported size for committed texture allocations (draft)):**

> - **Open questions from the first pass:** CI state; conflict state.
> - **Answers:**
>   1. [confirmed] 67 runs: 55 success, 2 neutral, 10 cancelled. The cancelled set is the same 10 macOS jobs as [#10409](https://github.com/gfx-rs/wgpu/issues/10409). No failure.
>   2. [confirmed] `blocked`, `mergeable: true`. No conflict.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none (rerun the Mac jobs before landing)

#### [#10803](https://github.com/gfx-rs/wgpu/issues/10803): Document more features with their own documentation modules in wgpu. ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI 10 cancelled jobs; last update 2026-11-01.
- **State:** in progress
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** not checked.
- **Summary:** Adds doc modules for features such as volatile/coherent decorations. Docs only, 7 files, +107. Works toward [#9647](https://github.com/gfx-rs/wgpu/issues/9647). Author plans to write the docs by hand and check with Claude before undrafting. Multiview fixes were split to [#10804](https://github.com/gfx-rs/wgpu/issues/10804).
- **Move-along action:** nothing (author is active).

#### [#10804](https://github.com/gfx-rs/wgpu/issues/10804): Multiview fixes ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (9 failed: API Operation Linux x86_64, API Operation Windows x86_64, API Validation Linux x86_64, API Validation Windows x86_64); last update 2026-11-01.
- **State:** in progress
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** not checked. PR has merge conflicts (mergeable_state dirty).
- **Summary:** Claude-written fixes for multiview on GLES and Metal and descriptor field name differences. 39 files, +646/-123, 8 commits, "tests existing". Opened 2026-11-01 as a split from [#10803](https://github.com/gfx-rs/wgpu/issues/10803). Needs a rebase. Touches wgpu-core/hal, so it needs careful review and a changelog check, and it is large for one PR.
- **Move-along action:** ping author to rebase and say whether to undraft; consider asking to split per backend.

#### [#10806](https://github.com/gfx-rs/wgpu/issues/10806): Passthrough shaders rework master ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; mergeable; CI red (1 failed: Format & Typos); last update 2026-11-02.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-11-02, inner-daemons, PR updated. 4 days.
- **CI:** not checked.
- **Summary:** Author says it is a "don't merge" master branch for passthrough shader reworks, with parts to be split off; it is all LLM generated with lower quality standards. 5 files, +563/-2. Description has a broken "Closes" line (closes [#9052](https://github.com/gfx-rs/wgpu/issues/9052)).
- **Move-along action:** nothing. Review only the split-off PRs when they appear.

#### [#10811](https://github.com/gfx-rs/wgpu/issues/10811): [naga msl-out] Fix clip distances in mesh shader vertex outputs ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI red (1 failed: Validate: MSL); last update 2026-11-02.
- **State:** in progress (looks near complete, but still draft)
- **Waiting on:** author
- **Last activity:** 2026-11-02, inner-daemons, PR created. 4 days.
- **CI:** not checked. PR has merge conflicts (mergeable_state dirty).
- **Summary:** Fixes clip distance output in MSL mesh shader vertex outputs. 8 files, +452/-1, new tests, LLM found and fixed it, author says they reviewed it before undrafting. Needs a rebase.
- **Move-along action:** ping author to rebase and undraft, then review.

**CI and status check ([#10811](https://github.com/gfx-rs/wgpu/issues/10811): [naga msl-out] Fix clip distances in mesh shader vertex outputs (draft)):**

> - **Open questions from the first pass:** CI not read.
> - **Answers:**
>   1. [confirmed] 67 runs: 64 success, 2 neutral, 1 failure: `Validate: MSL` (job 125081591729, 2026-11-02). Related to the PR. The log shows `xcrun -sdk macosx metal -mmacosx-version-min=10.11 -std=metal3.0 ... tests/out/msl/wgsl-mesh-shader-clip-distances.metal` fails with `34:32: error: 'clip_distance' attribute cannot be applied to types` and `61:93: error: invalid type 'ms_mainVertexOutput' for mesh vertex type`. So the Metal compiler rejects the generated MSL for the new snapshot. This is a real failure of the change (naga MSL output for mesh clip distances), not infra.
>   2. [confirmed] `mergeable_state: dirty` (conflict).
> - **New findings:** [high] The PR's own new snapshot `naga/tests/out/msl/wgsl-mesh-shader-clip-distances.metal` does not compile with the Metal compiler, so the fix is not yet correct or the test needs a version gate.
> - **Verdict change:** none (it was not marked close to ready); it is red for a real reason plus has a conflict.
> - **Move-along action change:** none, but tell the author about the Metal validation failure.

#### [#10852](https://github.com/gfx-rs/wgpu/issues/10852): ci: tolerate transient download failures in the merge queue ([@matthargett](https://github.com/matthargett)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-20):** draft; has conflicts; CI green; last update 2026-11-05.
- **State:** in progress
- **Waiting on:** maintainer (decision: is the scope wanted?). The body is one sentence about retries, but the diff adds a new checksum-pinned `download-release` action with a Python script, a Python test file run in CI, curl timeouts, and `max_checks_retries: 1` in the Mergify queue rules. The author says it may not fix the flakiness.
- **Last activity:** 2026-11-05, matthargett, pushed (5 commits). 1 day. No comments or reviews.
- **CI:** green.
- **Summary:** Makes CI downloads (DXC, Mesa, Vulkan SDK, WARP, Agility SDK, Dota2 shaders) fail hard and retry, and retries merge queue checks once. 11 files, +385/-18.
- **Move-along action:** decide whether to accept the new action and the Mergify retry change, or ask for a smaller PR. Review if yes.

Problems: no rate-limit errors. I did not fetch CI for [#9794](https://github.com/gfx-rs/wgpu/issues/9794), [#9853](https://github.com/gfx-rs/wgpu/issues/9853), [#10354](https://github.com/gfx-rs/wgpu/issues/10354) and [#10396](https://github.com/gfx-rs/wgpu/issues/10396), and I did not read the comment or review timelines for [#9853](https://github.com/gfx-rs/wgpu/issues/9853) and [#10396](https://github.com/gfx-rs/wgpu/issues/10396) (the first call showed no comments count; [#9853](https://github.com/gfx-rs/wgpu/issues/9853) has no comment count and [#10396](https://github.com/gfx-rs/wgpu/issues/10396) none). "Last activity" for [#9853](https://github.com/gfx-rs/wgpu/issues/9853) and [#10354](https://github.com/gfx-rs/wgpu/issues/10354) is the PR updated_at time, not a confirmed human action.

---

## Issues

#### [#8354](https://github.com/gfx-rs/wgpu/issues/8354): Implement DXGI swapchain for Vulkan ([@cwfitzgerald](https://github.com/cwfitzgerald)) [issue]
- **Status:** Tracking issue. Plan: land the swapchain interface refactor ([#8356](https://github.com/gfx-rs/wgpu/issues/8356)) and the DXGI backend ([#8388](https://github.com/gfx-rs/wgpu/issues/8388)), get testers, make it default early in a release cycle. [#8356](https://github.com/gfx-rs/wgpu/issues/8356) is closed (2025-10-27; merge status unverified). [#8388](https://github.com/gfx-rs/wgpu/issues/8388) is still open, last updated 2026-10-07, assigned to inner-daemons and andyleiserson. The later checklist items are not started. Comments: your maintainer-meeting summary (2025-10-22), a question about the "shader story" (answered by a non-maintainer, linking [#7443](https://github.com/gfx-rs/wgpu/issues/7443) and [#4155](https://github.com/gfx-rs/wgpu/issues/4155)), no activity from the team since.
- **Waiting on:** other reviewers (inner-daemons, andyleiserson on [#8388](https://github.com/gfx-rs/wgpu/issues/8388))
- **Recommended action:** Act: keep open. Tick the [#8356](https://github.com/gfx-rs/wgpu/issues/8356) checkbox, since that PR is closed. Then push [#8388](https://github.com/gfx-rs/wgpu/issues/8388) through review; the issue is the tracker for the rollout plan. Do not close.
- **Notes:**
  - The checklist boxes are still empty even though [#8356](https://github.com/gfx-rs/wgpu/issues/8356) is closed. Confirm it was merged and not just closed (the closing data does not show a merging PR, `closed_by_pull_requests` is empty).
  - The issue is 13 months old. If the "turn on by default early in the cycle" plan still holds, [#8388](https://github.com/gfx-rs/wgpu/issues/8388) must land soon to fit the next release.
  - The explanation of the "shader story" asked by Imberflur is still answered only by a community member; consider confirming it or editing the body.

---

#### [#9125](https://github.com/gfx-rs/wgpu/issues/9125): Increase GitHub Actions concurrent job limits 20->60 ([@inner-daemons](https://github.com/inner-daemons)) [issue]
- **Status:** You contacted GitHub support. On 2026-09-17 you reported the limit went from 20 to 40; the next step GitHub offers is a team plan, which you said is too expensive for how the repo is managed, and you will discuss alternatives. Andy suggested also asking for more cache storage, because PR caches evict trunk caches. No later update, and no linked PR.
- **Waiting on:** maintainer (you), then GitHub support
- **Recommended action:** Reply with a position: post the outcome of the talk with GitHub, retitle to "20->40 done, 60 pending", and either keep it open for the 60 limit and the cache request or close it as partly resolved. Suggest closing if no further increase is expected.
- **Notes:**
  - The cache storage request from Andy was not mentioned in your 09-17 reply. Ask GitHub about it in the same conversation.
  - CI on [#9053](https://github.com/gfx-rs/wgpu/issues/9053) and [#8653](https://github.com/gfx-rs/wgpu/issues/8653) ran roughly 20-25 minutes end to end on Aug 25, before the 09-17 increase, so there is no data yet on the effect.
  - Alternatives to a paid plan (fewer matrix jobs, merge queue only for heavy jobs) are not yet discussed in the issue.

#### [#10622](https://github.com/gfx-rs/wgpu/issues/10622): Encourage giving rationales for PRs ([@kpreid](https://github.com/kpreid)) [issue]
- **Status:** kpreid proposes renaming the "Connections" section of the PR template to "Context" and telling authors to give a use case or other justification when there is no issue to link. You and kpreid are assigned. One comment (Vipitis) supports the change, adds an example CONTRIBUTING, and asks for a link to the "designing new features" section of `CONTRIBUTING.md` in the template. No PR yet. The template in the local trunk checkout still says "Connections".
- **Waiting on:** maintainer (you) to confirm the direction, or kpreid to open the PR
- **Recommended action:** Act: reply with a position "yes, send the PR" and ask kpreid to include the `CONTRIBUTING.md` link. It is a small template-only change and one-line agreement unblocks it.
- **Notes:**
  - Fits your own rule that PRs need a description with motivation (you asked for one on [#9053](https://github.com/gfx-rs/wgpu/issues/9053)).
  - Label `firefox: triaged` is already set, so no triage step remains.
  - Keep the checkbox section in sync if the template is reworded.
