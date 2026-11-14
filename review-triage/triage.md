# wgpu PR triage

Last update: 2026-11-14.

This is a pre-review pass over the open PRs in gfx-rs/wgpu: 141 open PRs (92 ready for review, 49 drafts), and the 3 issues assigned to [@cwfitzgerald](https://github.com/cwfitzgerald). Merge queue PRs from Mergify are not counted. It is read only. No code was built or tested, and nothing was posted to GitHub. CI status comes from GitHub check runs. Findings that are reasoned from code, not run, are marked as such in the details.

This file shows the current state only. It is updated in place; `UPDATING.md` gives the procedure. In the tables, "Last update" is the date of the last GitHub activity on the PR.

## Attention first

- **Five PRs need the `TextureDescriptor::texture_binding_view_dimension` field** that [#8014](https://github.com/gfx-rs/wgpu/issues/8014) added. GitHub shows no conflict for them. [#10776](https://github.com/gfx-rs/wgpu/issues/10776) is confirmed: its Clippy jobs fail with E0063 at `tests/tests/wgpu-validation/api/texture.rs:824`. [#10910](https://github.com/gfx-rs/wgpu/issues/10910), [#10862](https://github.com/gfx-rs/wgpu/issues/10862) and [#10839](https://github.com/gfx-rs/wgpu/issues/10839) are reasoned from code, not built.
- **[#10910](https://github.com/gfx-rs/wgpu/issues/10910) has your approval and red CI from the PR.** It needed [#10710](https://github.com/gfx-rs/wgpu/issues/10710), which is merged; the author must rebase.
- **[#10890](https://github.com/gfx-rs/wgpu/issues/10890) exposes unvalidated count draws to Deno JS** on Vulkan and DX12. Decide the validation question for [#9679](https://github.com/gfx-rs/wgpu/issues/9679) and [#10890](https://github.com/gfx-rs/wgpu/issues/10890) together.
- **[#10795](https://github.com/gfx-rs/wgpu/issues/10795) fell out of the merge queue** with `CHECKS_FAILED` after you queued it.
- **Approvals from inner-daemons do not satisfy branch protection.** [#10821](https://github.com/gfx-rs/wgpu/issues/10821), [#10820](https://github.com/gfx-rs/wgpu/issues/10820), [#10796](https://github.com/gfx-rs/wgpu/issues/10796), [#10897](https://github.com/gfx-rs/wgpu/issues/10897), [#10891](https://github.com/gfx-rs/wgpu/issues/10891), [#10901](https://github.com/gfx-rs/wgpu/issues/10901), [#10343](https://github.com/gfx-rs/wgpu/issues/10343) and [#10546](https://github.com/gfx-rs/wgpu/issues/10546) have such an approval and show `REVIEW_REQUIRED`. Each needs an approval from a member with write access.
- **5andr0 has 5 agent-written PRs open** ([#10908](https://github.com/gfx-rs/wgpu/issues/10908), [#10921](https://github.com/gfx-rs/wgpu/issues/10921), [#10922](https://github.com/gfx-rs/wgpu/issues/10922), [#10926](https://github.com/gfx-rs/wgpu/issues/10926) and draft [#10928](https://github.com/gfx-rs/wgpu/issues/10928)) and wrote on [#9295](https://github.com/gfx-rs/wgpu/issues/9295) that maintainers should treat them as "issues with suggested fixes". Needs one policy answer.
- **[#8014](https://github.com/gfx-rs/wgpu/issues/8014) is merged with an unchecked finding:** the review of 2026-11-06 said that core does not check the view-dimension field against the texture dimension, and that the GLES HAL allocates from it. Check it on trunk.

---

# Part A: your review queue

These are the 19 open PRs that are assigned to you or that request your review (16 ready, 3 drafts), and 3 issues.

### In the merge queue (1)

| PR | Title | Note |
|---|---|---|
| [#10457](https://github.com/gfx-rs/wgpu/issues/10457) | naga/glsl vector select without mix | You approved and queued it on 11-14. |

### Needs a bit of work (13)

| PR | Title | Waiting on | Main item |
|---|---|---|---|
| [#10871](https://github.com/gfx-rs/wgpu/issues/10871) | Ray tracing invocation reordering feature and adapter property | you (review requested) | Exposes naga hit objects as a wgpu feature with the Vulkan EXT and a reorder-mode adapter query. Vecvec approved on 11-14. The FFI structs match the Vulkan headers. The Vulkan path is not tested on any CI adapter or hardware. |
| [#10506](https://github.com/gfx-rs/wgpu/issues/10506) | Metal: one command buffer for many passes **[Mac CI cancelled]** | you (design) | Has conflicts. **High, confirmed:** with indirect validation off, ICB setup for multi-draws is encoded after the pass that uses it, which is UB even with valid draw arguments. Medium: a timestamp written between passes attaches to the next pass. Needs your design sign-off. |
| [#9679](https://github.com/gfx-rs/wgpu/issues/9679) | Metal ICBs for multi-draw indirect count | author | Conflicts in `wgpu-hal/src/metal/mod.rs`. Core does not validate count-draw arguments and Metal clamps only the count. Fallback cost. The [#10506](https://github.com/gfx-rs/wgpu/issues/10506) ordering bug also affects its count-draw setup. No error-path tests. The deno part is in [#10890](https://github.com/gfx-rs/wgpu/issues/10890). |
| [#10609](https://github.com/gfx-rs/wgpu/issues/10609) | naga GLSL matrix interfaces | inner-daemons | CI green. The SPIR-V-frontend sampling change has no mention or test. A false overflow at `u32::MAX`. In/out blocks default to Std430 without explanation. No CHANGELOG attribution. |
| [#10505](https://github.com/gfx-rs/wgpu/issues/10505) | gles/wgl: pump messages while awaiting shutdown | author | The findings are not posted to the PR: the loop does not dispatch the messages it removes, and the thread and window leak if `PostMessageA` fails. No CHANGELOG attribution. |
| [#10596](https://github.com/gfx-rs/wgpu/issues/10596) | BF16 + cooperative matrix | author (kvark) | inner-daemons requested changes on 11-13 and asks whether the planned `subgroup_matrix` extension replaces cooperative matrices. The const evaluator folds bf16 comparisons that the validator rejects. No snapshot inputs. Medium: the validator accepts `As` from a matrix to bf16. [#10901](https://github.com/gfx-rs/wgpu/issues/10901) fixes the same capability bug in another way. Has conflicts. |
| [#10414](https://github.com/gfx-rs/wgpu/issues/10414) | naga non-square cooperative matrix tiles | author | Shape validation ignores the scalar type and mixes configurations. Has conflicts; a rebased branch exists from infosave2007. If [#10901](https://github.com/gfx-rs/wgpu/issues/10901) lands first, regenerate the snapshot. |
| [#10511](https://github.com/gfx-rs/wgpu/issues/10511) | dx12: 1 s hang for dropped SurfaceTextures | you | Has conflicts. Keep `pending_present_count` on resize and reset it only for a new swapchain. The details have a probe that the author can run. Land before [#10548](https://github.com/gfx-rs/wgpu/issues/10548). |
| [#10548](https://github.com/gfx-rs/wgpu/issues/10548) | dx12: DXGI_PRESENT_RESTART | you | Has conflicts. Hard-codes `Present(1, RESTART)`, which forces vsync and drops tearing. Combined fix: one `Present(interval, flags \| RESTART)`, then `pending_present_count += 1`. Depends on [#10511](https://github.com/gfx-rs/wgpu/issues/10511). |
| [#10152](https://github.com/gfx-rs/wgpu/issues/10152) | Queue::present_with_damage | you | The author nudged twice. Open decisions: merge the hal/wgt `Rect` types, and add a public `wgpu::Rect` or not. EGL y-flip uses an unchecked u32 subtraction. Has conflicts. |
| [#10438](https://github.com/gfx-rs/wgpu/issues/10438) | create_*_pipeline_async | you | Has conflicts. The public API shape needs your sign-off. `create_*_pipeline_or_error` traces and logs, which also affects Firefox and Deno. |
| [#10591](https://github.com/gfx-rs/wgpu/issues/10591) | core: SmallVec for single-region texture copies | you (wanted or not?) | The `len()==1` branch is not needed. The benchmark's value is disputed. |
| [#9757](https://github.com/gfx-rs/wgpu/issues/9757) | Xcode ray tracing detection | inner-daemons | The CHANGELOG leaves out the user-visible fix. A `// others...?` placeholder is in the code. |

### Needs a lot of work (2)

| PR | Title | Waiting on | Main item |
|---|---|---|---|
| [#9295](https://github.com/gfx-rs/wgpu/issues/9295) | naga/spv layoutless workgroup types | you, then kvark | Has conflicts. Two confirmed bugs: pointer types are emitted before the workgroup copies, and the mesh output wrapper uses the default type IDs. Draft [#10928](https://github.com/gfx-rs/wgpu/issues/10928) (agent-written, no human owner) is a rework that fixes the pointer-order bug, handles SPIR-V below 1.4 and adds `spirv-val` tests, but keeps the mesh-wrapper bug. Recommended: ask kvark to take the [#10928](https://github.com/gfx-rs/wgpu/issues/10928) commit into [#9295](https://github.com/gfx-rs/wgpu/issues/9295), then fix the mesh wrapper and add a pointer-parameter test. |
| [#10098](https://github.com/gfx-rs/wgpu/issues/10098) | Immediates on WebGPU, remove Features::IMMEDIATES | author (inactive since 09-18) | Removes the feature that you asked to keep. The WebGPU path cannot work. Has conflicts. Consider closing it. |

### Drafts (3)

| PR | State | Waiting on | Recommended action |
|---|---|---|---|
| [#10016](https://github.com/gfx-rs/wgpu/issues/10016) | Stalled since 09-27 | you, then kvark | Decide whether the Vulkan memory model is split out (inner-daemons wants it split, kvark does not). |
| [#9053](https://github.com/gfx-rs/wgpu/issues/9053) | Not close | inner-daemons | None of your 12 findings from 08-14 is addressed. |
| [#8653](https://github.com/gfx-rs/wgpu/issues/8653) | Not close | inner-daemons | Not split as asked. Park it, and decide whether you want this suite in addition to the CTS. |

### Issues (3)

| Issue | Recommended action |
|---|---|
| [#10622](https://github.com/gfx-rs/wgpu/issues/10622) Encourage PR rationales | Act: ask kpreid to send the PR, and link CONTRIBUTING "designing new features". |
| [#9125](https://github.com/gfx-rs/wgpu/issues/9125) Actions job limits | GitHub raised the limit to 40 on 09-17. Reply with the result, then keep the issue open for 60 plus cache storage, or close it as partly resolved. |
| [#8354](https://github.com/gfx-rs/wgpu/issues/8354) DXGI swapchain for Vulkan | Keep it open. Tick [#8356](https://github.com/gfx-rs/wgpu/issues/8356) (merged). [#8388](https://github.com/gfx-rs/wgpu/issues/8388) waits on andyleiserson; see section 7. |

---

# Part B: all other open PRs

Each PR appears once below, in the section for its main action.

## 1. Merge now, or after one mechanical step (17)

"Member approval" means an approval from a member with write access.

| PR | Title | Step before merge | Last update | Note |
|---|---|---|---|---|
| [#10698](https://github.com/gfx-rs/wgpu/issues/10698) | spv-out: UB in rayQueryGenerateIntersection | none: in the merge queue | 11-14 | Approved by Vecvec. A user confirmed the fix on an RTX 3080 Ti. |
| [#10896](https://github.com/gfx-rs/wgpu/issues/10896) | `Device::is_surface_supported` | none: you approved and queued it | 11-14 | Trivial forwarder. One Linux test job is red (a `player` `test_api` timeout). |
| [#10797](https://github.com/gfx-rs/wgpu/issues/10797) | msl: subgroupBallot lanes 32-63 | none: you approved and queued it | 11-14 | Check that the queue took it; it had a CHANGELOG conflict. |
| [#10795](https://github.com/gfx-rs/wgpu/issues/10795) | core: `Texture::mark_uninitialized` | read the Mergify check, queue again | 11-14 | You approved and queued it; Mergify removed it with `CHECKS_FAILED`. |
| [#10930](https://github.com/gfx-rs/wgpu/issues/10930) (yours) | v30.0.2 changelog and release checklist | merge | 11-14 | Approved by ErichDonGubler. |
| [#10897](https://github.com/gfx-rs/wgpu/issues/10897) | wgsl-in: reject composite `const_assert` conditions | CHANGELOG rebase, member approval | 11-09 | Fixes a panic that the fuzzer found. Approved by inner-daemons. |
| [#10884](https://github.com/gfx-rs/wgpu/issues/10884) | Renovate: zizmor-action 0.8.1 | none | 11-11 | CI green. |
| [#10883](https://github.com/gfx-rs/wgpu/issues/10883) | Renovate: install-action 2.90.3 | none | 11-11 | No failures. |
| [#10837](https://github.com/gfx-rs/wgpu/issues/10837) | vk: pass-begin timestamp at TOP_OF_PIPE | re-approval | 11-13 | inner-daemons' review is done. The last push only added the doc comment that the reviewer suggested. |
| [#10675](https://github.com/gfx-rs/wgpu/issues/10675) | core: hold `command_indices` around surface (un)configure | member approval; agentic-PR policy check | 11-12 | Approved by sagudev. Two users report the crash. |
| [#10546](https://github.com/gfx-rs/wgpu/issues/10546) | vk: VK_COLOR_SPACE_PASS_THROUGH_EXT | rebase (has conflicts) | 11-03 | Approved by inner-daemons. The author has pinged jimblandy twice. |
| [#10343](https://github.com/gfx-rs/wgpu/issues/10343) | hlsl: unbounded `[]` binding arrays | member approval | 10-17 | Approved by inner-daemons. |
| [#10638](https://github.com/gfx-rs/wgpu/issues/10638) | hal: link API docs from dyn methods | review | 10-21 | Docs only. jimblandy self-assigned and has not reviewed. |
| [#10739](https://github.com/gfx-rs/wgpu/issues/10739) | renovate `rebaseWhen = conflicted` | review | 11-02 | The author is not sure it is needed. Low risk. |
| [#10694](https://github.com/gfx-rs/wgpu/issues/10694) | wgsl-in: rayQueryGenerateIntersection scope fix | rerun SPIR-V validate job (curl failure) | 10-28 | jimblandy assigned, no review. First in the ray query order. |
| [#10278](https://github.com/gfx-rs/wgpu/issues/10278) | msl: typed 64-bit literals in div/rem guard | author rebases, then approve the new CI runs | 09-09 | No check runs on the head. Has a conflict. Code is small and correct. |
| [#10784](https://github.com/gfx-rs/wgpu/issues/10784) | CTS: non-const `modf` f32 tests | confirm the 24 CTS cases pass | 10-30 | PR CI does not run the CTS. |

## 2. Waiting on a decision from you (or another maintainer)

| PR | Question | Last update |
|---|---|---|
| [#10890](https://github.com/gfx-rs/wgpu/issues/10890) | Expose `chromium-experimental-multi-draw-indirect` in Deno? **High:** wgpu-core does not validate count-draw arguments, and this PR makes them reachable from JS on Vulkan and DX12. It keeps the feature under `STRICT_WEBGPU_COMPLIANCE`. Medium: when the feature is missing, the encoder is not invalidated, so `finish()` succeeds. Duplicates the [#9679](https://github.com/gfx-rs/wgpu/issues/9679) alignment change; land [#9679](https://github.com/gfx-rs/wgpu/issues/9679) first. | 11-09 |
| [#10891](https://github.com/gfx-rs/wgpu/issues/10891) | Enforce 2047 elements in array constructors? Approved by inner-daemons. The CHANGELOG says the spec requires the limit; the WGSL spec gives 2047 only as a minimum that implementations must support. The change rejects native shaders with const arrays of 2048 or more elements. | 11-13 |
| [#10886](https://github.com/gfx-rs/wgpu/issues/10886), [#10888](https://github.com/gfx-rs/wgpu/issues/10888) | Renovate: libloading 0.9 and bincode 3. Both need code changes. libloading: renamed error variants, an `AsFilename` bound, and a cargo-deny duplicate. bincode: v3 has no `serde` feature, which `wgpu-benchmark` uses. Port them or tell Renovate to ignore them. | 11-13 |
| [#10597](https://github.com/gfx-rs/wgpu/issues/10597) | Make `naga::Handle::from_usize` public? jimblandy against, inner-daemons for. Marked fully agent-created. | 10-31 |
| [#10518](https://github.com/gfx-rs/wgpu/issues/10518) | Do we want physical storage buffer pointers in naga IR at all? 12k lines in one commit, no reviewer. Not fully behind its capability. Has conflicts. A 7-part split is proposed in the details. | 10-07 |
| [#10583](https://github.com/gfx-rs/wgpu/issues/10583) | Is per-backend dedup of array aliases the accepted direction? kpreid suggests a shared `naga::proc` helper. Nested aliases are missed. | 11-10 |
| [#10697](https://github.com/gfx-rs/wgpu/issues/10697) | Accept the non-standard `rayQueryGetCurrentTMax` builtin? Has conflicts and an unanswered init-tracking comment. | 10-28 |
| [#10677](https://github.com/gfx-rs/wgpu/issues/10677) | Timeline semaphore padding: IMG only, IMG plus gfxstream, or all drivers? Has conflicts. | 11-04 |
| [#10840](https://github.com/gfx-rs/wgpu/issues/10840) | Turn off timeline semaphores on every Arm Mali driver version (one version tested)? Also adds a 297-line repro crate. | 11-05 |
| [#10775](https://github.com/gfx-rs/wgpu/issues/10775) | Is the `Extensions<T>` unsafe token plus separate `request_adapters` argument the extension pattern? andyleiserson asked on 11-03. Has conflicts. | 11-04 |
| [#9876](https://github.com/gfx-rs/wgpu/issues/9876) | Want `ComputePipeline::get_subgroup_size` as non-WebGPU public API? jimblandy is assigned and has not engaged. | 10-30 |
| [#9966](https://github.com/gfx-rs/wgpu/issues/9966) | spv BallotFindLSB/MSB. inner-daemons handed it to you and will raise it at the next maintainers meeting; the author has pinged twice. Needs a rebase and a CHANGELOG fix. | 11-13 |
| [#10725](https://github.com/gfx-rs/wgpu/issues/10725) | MSL version probing: opt-in only, and keep the version settings? (andyleiserson to decide.) A tester reports that the fallback works on OCLP Kepler with a 1.6 ms probe. Conflicts in four files. | 11-11 |
| [#10731](https://github.com/gfx-rs/wgpu/issues/10731), [#10743](https://github.com/gfx-rs/wgpu/issues/10743) | Support workarounds for OCLP-only (patched macOS) Intel compiler bugs at all? If no, close both. If yes, fold [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731). Both have conflicts. | 10-28 |
| [#10778](https://github.com/gfx-rs/wgpu/issues/10778) | Offer compile-time libwayland linking at all? As written it breaks macOS, iOS and Android builds. CI red. | 10-29 |
| [#10801](https://github.com/gfx-rs/wgpu/issues/10801) | Vulkan SDK 1.4.363: merge and file an issue for the MoltenVK `struct_layout` failures? | 11-02 |
| [#9645](https://github.com/gfx-rs/wgpu/issues/9645) | Metal AS build sync. Recommended: land [#9645](https://github.com/gfx-rs/wgpu/issues/9645) first (rebase, add the same-command-buffer tests, say it covers only that case), then ask acoliver to split [#10292](https://github.com/gfx-rs/wgpu/issues/10292). | 08-07 |
| [#10812](https://github.com/gfx-rs/wgpu/issues/10812) | Child textures in `Arc` ([#5138](https://github.com/gfx-rs/wgpu/issues/5138)). teoxoy is assigned. The confirmed double free (section 5) is unfixed. Has conflicts. Suggested instead: one `Arc<TextureShared>` that destroys the HAL objects once. | 11-09 |

## 3. Waiting on a reviewer: ping or reassign

| PR | Reviewer | Last update | Note |
|---|---|---|---|
| [#8388](https://github.com/gfx-rs/wgpu/issues/8388) (yours) | andyleiserson | 11-13 | DXGI swapchain on Vulkan. You answered the review threads on 11-13; the fixes are in fixup commits to squash after approval. CI green. |
| [#10903](https://github.com/gfx-rs/wgpu/issues/10903) (yours) | none | 11-13 | Fix for binding more immediate data than the pipeline supports ([#10902](https://github.com/gfx-rs/wgpu/issues/10902)). CI green. Touches wgpu-core and wgpu-hal. Request a reviewer. |
| [#10873](https://github.com/gfx-rs/wgpu/issues/10873) | none | 11-07 | External texture plumbing in spv-out (andyleiserson). Clean, CI green. Low findings only. Needs a small rebase on `writer.rs`. |
| [#10866](https://github.com/gfx-rs/wgpu/issues/10866) | none | 11-06 | GLES: detach renderbuffers when the draw framebuffer is reused (Android Emulator). Small and correct, with a regression test. Needs a CHANGELOG entry. The red Mac job is a Metal poll timeout. |
| [#10862](https://github.com/gfx-rs/wgpu/issues/10862) | inner-daemons | 11-09 | Pools command-encoder and pass trackers. The logic looks correct. Both agent-disclosure boxes are checked. The test needs the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field. |
| [#10860](https://github.com/gfx-rs/wgpu/issues/10860) | inner-daemons | 11-09 | Metal watchOS MSL profiles. It enables indirect draw and dispatch and base vertex/first instance on Apple5 watchOS 11+, with evidence for indirect draws only. |
| [#10839](https://github.com/gfx-rs/wgpu/issues/10839) | inner-daemons | 11-09 | Render bundle resource lifetimes. CI green. Open: unused `'a` on `RenderEncoder`, bloated tests, the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field. |
| [#10561](https://github.com/gfx-rs/wgpu/issues/10561) | jimblandy | 10-28 | spv-in DebugPrintf, about 100 lines, looks correct. Has a conflict. |
| [#10229](https://github.com/gfx-rs/wgpu/issues/10229) | jimblandy | 10-20 | glsl: no `invariant gl_FragCoord` on ES. Only a CHANGELOG line is missing; take over and merge. |
| [#10554](https://github.com/gfx-rs/wgpu/issues/10554) | Wumpf | 10-25 | WGL explicit GL version. CI fails on Claude co-author trailers; has conflicts. Take over with one clean commit and credit. |
| [#10802](https://github.com/gfx-rs/wgpu/issues/10802) | none | 11-01 | deno: external memory size. Small, CI green. |

## 4. Waiting on the author: ping, take over, or close

| PR | Last update | Ask | If no answer |
|---|---|---|---|
| [#10910](https://github.com/gfx-rs/wgpu/issues/10910) | 11-11 | Multi-planar usage validation (andyleiserson). You approved it. **CI is red from the PR.** Rebase on trunk ([#10710](https://github.com/gfx-rs/wgpu/issues/10710) is merged), add the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field, fix the `#TODO` CHANGELOG links. | n/a |
| [#10776](https://github.com/gfx-rs/wgpu/issues/10776) | 11-13 | Multi-planar copy validation (andyleiserson). **CI is red from the PR:** the test literal at `tests/tests/wgpu-validation/api/texture.rs:824` lacks the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field. No reviewer. | n/a |
| [#10600](https://github.com/gfx-rs/wgpu/issues/10600) | 11-07 | **CI is red from the PR:** convert the 5 `write_function_signature` calls in `spv/ray/hit_object.rs`, and fix the snapshot conflicts. The author wants review first; jimblandy has not reviewed. | n/a (andyleiserson) |
| [#10838](https://github.com/gfx-rs/wgpu/issues/10838) | 11-14 | Sparse TLAS instance indices. CI is red because the test segfaults lavapipe (a Mesa bug); skip or expect it. Medium: the empty BLAS reads the zero buffer, which has no BLAS-input usage and gets no barrier. | n/a (Vecvec, active) |
| [#10875](https://github.com/gfx-rs/wgpu/issues/10875) | 11-13 | Metal: lose the device when a command buffer fails. andyleiserson asked how it interacts with `release_gpu_resources`, [#9579](https://github.com/gfx-rs/wgpu/issues/9579) and [#9278](https://github.com/gfx-rs/wgpu/issues/9278), and suggested a split. Medium: every Metal error code loses the device, including the iOS error for work submitted in the background. Has conflicts. | ping |
| [#10522](https://github.com/gfx-rs/wgpu/issues/10522) | 11-13 | GLES DMA-BUF import. andyleiserson requested changes on 11-13 with minor notes (GL error reporting, require `GL_OES_EGL_image`, CHANGELOG attribution). | ping |
| [#10901](https://github.com/gfx-rs/wgpu/issues/10901) | 11-13 | Cooperative matrix store emit order and missing SPIR-V extensions. Both fixes are correct; approved by inner-daemons, who asked for a validator check on the store stride. Overlaps [#10596](https://github.com/gfx-rs/wgpu/issues/10596). CHANGELOG conflict. | n/a |
| [#10880](https://github.com/gfx-rs/wgpu/issues/10880) | 11-10 | Vulkan: trim command pools to what the last submission used. Correct, but the keep-last-count rule can churn when large and small submissions alternate. inner-daemons asked for benchmarks. The self-review box is unchecked. | ping |
| [#10820](https://github.com/gfx-rs/wgpu/issues/10820) | 11-11 | Remove `F16_IN_F32`. Approved by inner-daemons. Add a breaking-change CHANGELOG entry and fix the stale comment at `naga/src/valid/mod.rs:247`. | add them on merge |
| [#10629](https://github.com/gfx-rs/wgpu/issues/10629) | 11-07 | Vulkan multi-planar DMA-buf import (andyleiserson). No tests; CHANGELOG `#TODO`; one error names the wrong extension. | n/a |
| [#10800](https://github.com/gfx-rs/wgpu/issues/10800) **[Mac CI cancelled]** | 11-06 | Answer matthargett's A10X crash. Rebase so Mac CI runs. Also carries the [#10801](https://github.com/gfx-rs/wgpu/issues/10801) SDK bump commit. | n/a (inner-daemons) |
| [#10821](https://github.com/gfx-rs/wgpu/issues/10821) | 11-09 | inner-daemons approved, but trunk already releases the texture, and sagudev's [#10296](https://github.com/gfx-rs/wgpu/issues/10296) question has no answer. Ask the author to reproduce on trunk. | close |
| [#10926](https://github.com/gfx-rs/wgpu/issues/10926) | 11-12 | Vulkan: filter buffer memory types by available budget. The logic looks correct. The "fifteen CPU cases" in the description are not in the PR. Agent-written description, self-review box unchecked. | close under the agentic-PR policy |
| [#10922](https://github.com/gfx-rs/wgpu/issues/10922) | 11-11 | DX12-only public option for DXC `-denorm`. No use case; agent-written draft description; an explicit mode below shader model 6.2 fails only at pipeline creation. | close |
| [#10908](https://github.com/gfx-rs/wgpu/issues/10908), [#10921](https://github.com/gfx-rs/wgpu/issues/10921) | 11-12 | Agent-written IR field for SPIR-V `NoContraction`, and a stacked one-op `Exp2` control. [#10908](https://github.com/gfx-rs/wgpu/issues/10908) makes SPIR-V that loads on trunk fail in the frontend, in validation, and on MSL, WGSL, WebGL2 and SM5 output (from code). The author says that no human owns them. | close both |
| [#10155](https://github.com/gfx-rs/wgpu/issues/10155) | 10-06 | Policy for releasing the cached AS scratch buffer (Vecvec). The author is silent since 09-03. Has conflicts. | close |
| [#10515](https://github.com/gfx-rs/wgpu/issues/10515) | 10-27 | Is grovesNL's option 1 (Emscripten built like wasm32-unknown-unknown) the plan? CI red from the PR. | close; DouglasDwyer has a prototype |
| [#9940](https://github.com/gfx-rs/wgpu/issues/9940) | 10-21 | jimblandy's CHANGES_REQUESTED: exponential SPIR-V constant writer and a panic in matrix-column const eval. | n/a (andyleiserson) |
| [#10661](https://github.com/gfx-rs/wgpu/issues/10661) | 10-23 | Snapshot test requested by inner-daemons. Has a conflict. | take over (about 50 lines) |
| [#10676](https://github.com/gfx-rs/wgpu/issues/10676) | 10-28 | Answer inner-daemons' LLM-policy question. Medium: with an override in the module, pipeline-constant processing const-folds `let i = 9; arr[i]` and fails pipeline creation for valid WGSL. | close |
| [#10585](https://github.com/gfx-rs/wgpu/issues/10585) | 10-17 | Rebase (CHANGELOG). Approved by inner-daemons. | resolve the conflict yourself and merge |
| [#10673](https://github.com/gfx-rs/wgpu/issues/10673) | 11-01 | Mark draft or close: superseded by [#10800](https://github.com/gfx-rs/wgpu/issues/10800). | close after [#10800](https://github.com/gfx-rs/wgpu/issues/10800) merges |
| [#10750](https://github.com/gfx-rs/wgpu/issues/10750) | 11-09 | VK_EXT_hdr_metadata. CI green. Fully agent-written, the API returns a bare bool, and it has a conflict in `wgpu-core/src/instance.rs`. | close under the agentic-PR policy |
| [#10796](https://github.com/gfx-rs/wgpu/issues/10796) | 11-09 | Approved by inner-daemons. The fix is incomplete (summed constant offsets wrap, for example `4294967292+8`), and the PR is agent-created by its own disclosure. | close under the agentic-PR policy |
| [#10831](https://github.com/gfx-rs/wgpu/issues/10831) | 11-09 | Opened by an agent account. teoxoy asked ErichDonGubler who reviews it; no answer. Ask the named human to own it. | close under the agentic-PR policy |

## 5. Highest-severity findings

- **[#10890](https://github.com/gfx-rs/wgpu/issues/10890) (high, untrusted input):** count draws reach wgpu-core from Deno JS without argument validation on Vulkan and DX12.
- **[#10812](https://github.com/gfx-rs/wgpu/issues/10812) (high, safe-API UB):** child and parent textures share clear views, but every `Texture::drop` destroys them.
- **[#10506](https://github.com/gfx-rs/wgpu/issues/10506) (high, UB; your queue):** ICB setup for multi-draws is encoded after the pass that uses it when indirect validation is off.
- **[#9940](https://github.com/gfx-rs/wgpu/issues/9940) (high, DoS):** the SPIR-V constant writer is exponential in nesting depth. Already found by jimblandy.
- **[#10908](https://github.com/gfx-rs/wgpu/issues/10908) (high, regression):** SPIR-V with `NoContraction` that translates on trunk fails with this PR (from code).
- **[#10289](https://github.com/gfx-rs/wgpu/issues/10289) (high, from code; draft):** raising the per-stage acceleration structure limit from 1 to 8 overflows Metal's 31-slot buffer budget in the vertex stage.
- **[#10838](https://github.com/gfx-rs/wgpu/issues/10838) (medium):** the empty BLAS reads the zero buffer without BLAS-input usage or a barrier.
- **[#10875](https://github.com/gfx-rs/wgpu/issues/10875) (medium):** every Metal command buffer error loses the device, including the iOS background error.
- **[#10796](https://github.com/gfx-rs/wgpu/issues/10796) (medium):** HLSL storage offsets can wrap.
- **[#10518](https://github.com/gfx-rs/wgpu/issues/10518) (medium, untrusted input):** ordinary SPIR-V member decorations reach the access analysis, which can reject large valid shaders.
- **[#10812](https://github.com/gfx-rs/wgpu/issues/10812) (medium):** parent and children share one tracker index.
- **[#10676](https://github.com/gfx-rs/wgpu/issues/10676) (medium):** valid WGSL with an override can fail pipeline creation.
- **[#10731](https://github.com/gfx-rs/wgpu/issues/10731) (medium, performance):** a compile-and-pipeline probe on every Intel Mac at enumeration.

## 6. Policy: agent-written PRs

AGENTS.md does not allow fully agentic PRs. These need a policy call, ideally one consistent answer (draft [#10751](https://github.com/gfx-rs/wgpu/issues/10751) proposes a stricter policy and also waits on you):

- **5andr0:** [#10908](https://github.com/gfx-rs/wgpu/issues/10908), [#10921](https://github.com/gfx-rs/wgpu/issues/10921), [#10922](https://github.com/gfx-rs/wgpu/issues/10922), [#10926](https://github.com/gfx-rs/wgpu/issues/10926) and draft [#10928](https://github.com/gfx-rs/wgpu/issues/10928). The author wrote on [#9295](https://github.com/gfx-rs/wgpu/issues/9295) that maintainers should treat these PRs as issues with suggested fixes, and declined to write the [#10928](https://github.com/gfx-rs/wgpu/issues/10928) description by hand when inner-daemons asked.
- Marked or apparently fully agent-created: [#10796](https://github.com/gfx-rs/wgpu/issues/10796), [#10597](https://github.com/gfx-rs/wgpu/issues/10597), [#10750](https://github.com/gfx-rs/wgpu/issues/10750), [#10831](https://github.com/gfx-rs/wgpu/issues/10831), [#10675](https://github.com/gfx-rs/wgpu/issues/10675) (the fix itself is good), [#10862](https://github.com/gfx-rs/wgpu/issues/10862).
- Mostly LLM-written with unticked self-review or an open policy question: [#10518](https://github.com/gfx-rs/wgpu/issues/10518), [#10676](https://github.com/gfx-rs/wgpu/issues/10676), [#10725](https://github.com/gfx-rs/wgpu/issues/10725), [#10839](https://github.com/gfx-rs/wgpu/issues/10839), [#10880](https://github.com/gfx-rs/wgpu/issues/10880).
- [#10554](https://github.com/gfx-rs/wgpu/issues/10554) fails the typos check because of Claude co-author trailers. The fix itself is reviewed and wanted.
- Your own [#10869](https://github.com/gfx-rs/wgpu/issues/10869) has the self-review box unchecked.

## 7. Your own PRs

| PR | State | Next step |
|---|---|---|
| [#10930](https://github.com/gfx-rs/wgpu/issues/10930) | v30.0.2 changelog and release checklist. Approved by ErichDonGubler | Merge. |
| [#8388](https://github.com/gfx-rs/wgpu/issues/8388) | DXGI swapchain on Vulkan. CI green. You answered the review threads on 11-13 | Wait for andyleiserson, then squash the fixup commits. |
| [#10903](https://github.com/gfx-rs/wgpu/issues/10903) | Immediates fix ([#10902](https://github.com/gfx-rs/wgpu/issues/10902)). CI green, no reviewer | Request a reviewer. |
| [#10869](https://github.com/gfx-rs/wgpu/issues/10869) | KosmicKrisp test expectations. CI green, no review | Tick the self-review box; request a reviewer. |
| [#10212](https://github.com/gfx-rs/wgpu/issues/10212) | Ready for review. **CI red from the PR:** the regression test makes a D2 view of a D2Array texture, which fails on GL/llvmpipe | Fix the test. |
| [#10354](https://github.com/gfx-rs/wgpu/issues/10354) (draft) | `debug_printf_output` times out on Mac aarch64 Metal | Skip or expect the test on Metal. Waits on [#9389](https://github.com/gfx-rs/wgpu/issues/9389). |
| [#9794](https://github.com/gfx-rs/wgpu/issues/9794), [#9853](https://github.com/gfx-rs/wgpu/issues/9853), [#10396](https://github.com/gfx-rs/wgpu/issues/10396) (drafts) | Stalled | [#10396](https://github.com/gfx-rs/wgpu/issues/10396) has the empty template description. |

## 8. Drafts (42 not written by you and not in your queue)

**Close candidates (7):** [#6584](https://github.com/gfx-rs/wgpu/issues/6584) (empty `todo!()` stub), [#9897](https://github.com/gfx-rs/wgpu/issues/9897) (DO NOT MERGE repro for a fixed issue), [#9128](https://github.com/gfx-rs/wgpu/issues/9128) (jimblandy does not need it), [#9659](https://github.com/gfx-rs/wgpu/issues/9659) (superseded by [#9679](https://github.com/gfx-rs/wgpu/issues/9679), close once [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands), [#10114](https://github.com/gfx-rs/wgpu/issues/10114) (stale experiment), [#9427](https://github.com/gfx-rs/wgpu/issues/9427) (ping or close), [#9523](https://github.com/gfx-rs/wgpu/issues/9523) (pinged twice, ping once more then close).

**Close to ready; ask the author to undraft (5):** [#9512](https://github.com/gfx-rs/wgpu/issues/9512) (approved by andyleiserson in May; has a conflict), [#9909](https://github.com/gfx-rs/wgpu/issues/9909) (two approvals; CI red from July; needs a rebase), [#10409](https://github.com/gfx-rs/wgpu/issues/10409) (naga compaction reorder; decide if the about 70% slowdown when it reorders is acceptable), [#10790](https://github.com/gfx-rs/wgpu/issues/10790) (dx12 driver-reported size; decide if the `texture_from_raw` signature change is acceptable), [#10811](https://github.com/gfx-rs/wgpu/issues/10811) (msl mesh clip distances; **CI shows a real failure**; also a conflict).

**Need your decision (10):**
- [#8480](https://github.com/gfx-rs/wgpu/issues/8480): is `set_web_canvas_size` the accepted API?
- [#8849](https://github.com/gfx-rs/wgpu/issues/8849): do you want an `AnyError`-style wrapper?
- [#8954](https://github.com/gfx-rs/wgpu/issues/8954): which Metal import fix? kvark was asked and never answered.
- [#9353](https://github.com/gfx-rs/wgpu/issues/9353): opt-in flag for explicit passthrough binding indices, or wait for precompiled shaders?
- [#9809](https://github.com/gfx-rs/wgpu/issues/9809): `WriteOnly` volatile writes without `const fn`?
- [#9888](https://github.com/gfx-rs/wgpu/issues/9888): policy on repro examples that CI cannot run
- [#9975](https://github.com/gfx-rs/wgpu/issues/9975): does the CTS time budget allow it?
- [#10355](https://github.com/gfx-rs/wgpu/issues/10355): an inline pass in naga (stacked on [#10409](https://github.com/gfx-rs/wgpu/issues/10409))
- [#10852](https://github.com/gfx-rs/wgpu/issues/10852): accept a new download action and Mergify retry change, or ask for a smaller PR?
- [#10928](https://github.com/gfx-rs/wgpu/issues/10928): agent-written rework of [#9295](https://github.com/gfx-rs/wgpu/issues/9295) with no human owner; see [#9295](https://github.com/gfx-rs/wgpu/issues/9295) in Part A.

**Ping the author (10):** [#9023](https://github.com/gfx-rs/wgpu/issues/9023), [#9380](https://github.com/gfx-rs/wgpu/issues/9380), [#9485](https://github.com/gfx-rs/wgpu/issues/9485), [#9894](https://github.com/gfx-rs/wgpu/issues/9894), [#10008](https://github.com/gfx-rs/wgpu/issues/10008), [#10282](https://github.com/gfx-rs/wgpu/issues/10282), [#10804](https://github.com/gfx-rs/wgpu/issues/10804), [#10288](https://github.com/gfx-rs/wgpu/issues/10288), [#10289](https://github.com/gfx-rs/wgpu/issues/10289), [#10774](https://github.com/gfx-rs/wgpu/issues/10774).

**Metal ray tracing stack:** [#10292](https://github.com/gfx-rs/wgpu/issues/10292) is the base, then [#10287](https://github.com/gfx-rs/wgpu/issues/10287), [#10288](https://github.com/gfx-rs/wgpu/issues/10288) and [#10289](https://github.com/gfx-rs/wgpu/issues/10289). None of the four has workflow runs, and all four have conflicts. The author last pushed on 09-09. [#10292](https://github.com/gfx-rs/wgpu/issues/10292) should be split (see [#9645](https://github.com/gfx-rs/wgpu/issues/9645) in section 2).

**Nothing to do (7):** [#9843](https://github.com/gfx-rs/wgpu/issues/9843), [#9863](https://github.com/gfx-rs/wgpu/issues/9863) (blocked upstream), [#10304](https://github.com/gfx-rs/wgpu/issues/10304) (author splitting it), [#10403](https://github.com/gfx-rs/wgpu/issues/10403) (waits on a glow release), [#10783](https://github.com/gfx-rs/wgpu/issues/10783) (waits on [gpuweb/cts#4723](https://github.com/gpuweb/cts/issues/4723)), [#10803](https://github.com/gfx-rs/wgpu/issues/10803), [#10806](https://github.com/gfx-rs/wgpu/issues/10806).

**Policy (1):** [#10751](https://github.com/gfx-rs/wgpu/issues/10751) (contributor policy update).

## 9. Dependencies and landing order

- **[#8014](https://github.com/gfx-rs/wgpu/issues/8014) field:** [#10776](https://github.com/gfx-rs/wgpu/issues/10776), [#10910](https://github.com/gfx-rs/wgpu/issues/10910), [#10862](https://github.com/gfx-rs/wgpu/issues/10862) and [#10839](https://github.com/gfx-rs/wgpu/issues/10839) each need `texture_binding_view_dimension` in their test literals.
- **Multi-planar chain:** [#10910](https://github.com/gfx-rs/wgpu/issues/10910) and [#10776](https://github.com/gfx-rs/wgpu/issues/10776) are unblocked; both need a rebase.
- **naga ray query:** [#10694](https://github.com/gfx-rs/wgpu/issues/10694), [#10698](https://github.com/gfx-rs/wgpu/issues/10698), then [#10600](https://github.com/gfx-rs/wgpu/issues/10600) (convert the `hit_object.rs` calls), then [#10697](https://github.com/gfx-rs/wgpu/issues/10697) (adapt to the [#10600](https://github.com/gfx-rs/wgpu/issues/10600) helper signature; its GPU test needs the [#10698](https://github.com/gfx-rs/wgpu/issues/10698) fix on Vulkan).
- **naga cooperative matrix:** [#10901](https://github.com/gfx-rs/wgpu/issues/10901) and [#10596](https://github.com/gfx-rs/wgpu/issues/10596) fix the same capability bug in two ways; land one fix. The [#10414](https://github.com/gfx-rs/wgpu/issues/10414) snapshot may change after [#10901](https://github.com/gfx-rs/wgpu/issues/10901).
- **naga workgroup layouts:** [#9295](https://github.com/gfx-rs/wgpu/issues/9295) and draft [#10928](https://github.com/gfx-rs/wgpu/issues/10928) compete; see Part A.
- **naga NoContraction:** [#10921](https://github.com/gfx-rs/wgpu/issues/10921) is stacked on [#10908](https://github.com/gfx-rs/wgpu/issues/10908).
- **Metal multi-draw:** land [#9679](https://github.com/gfx-rs/wgpu/issues/9679) before [#10890](https://github.com/gfx-rs/wgpu/issues/10890), then drop the duplicate alignment change from [#10890](https://github.com/gfx-rs/wgpu/issues/10890). The [#10506](https://github.com/gfx-rs/wgpu/issues/10506) ordering bug also affects [#9679](https://github.com/gfx-rs/wgpu/issues/9679).
- **Metal adapter:** [#10860](https://github.com/gfx-rs/wgpu/issues/10860) edits the ICB expression in `wgpu-hal/src/metal/adapter.rs` that [#10927](https://github.com/gfx-rs/wgpu/issues/10927) (merged) changed; it needs a rebase. [#10875](https://github.com/gfx-rs/wgpu/issues/10875) and [#9679](https://github.com/gfx-rs/wgpu/issues/9679) both change `wgpu-hal/src/metal/mod.rs` (not checked for a conflict).
- **Metal MSL / Intel:** [#10725](https://github.com/gfx-rs/wgpu/issues/10725), then [#10731](https://github.com/gfx-rs/wgpu/issues/10731), then [#10743](https://github.com/gfx-rs/wgpu/issues/10743) (or squash [#10743](https://github.com/gfx-rs/wgpu/issues/10743) into [#10731](https://github.com/gfx-rs/wgpu/issues/10731)).
- **Metal workgroup memory:** [#10800](https://github.com/gfx-rs/wgpu/issues/10800) supersedes [#10673](https://github.com/gfx-rs/wgpu/issues/10673). [#10800](https://github.com/gfx-rs/wgpu/issues/10800) also contains the [#10801](https://github.com/gfx-rs/wgpu/issues/10801) SDK commit. [#10800](https://github.com/gfx-rs/wgpu/issues/10800) does not cover passthrough MSL with `[[threadgroup(n)]]` arguments, which [#10673](https://github.com/gfx-rs/wgpu/issues/10673) handles.
- **Metal AS sync:** [#10292](https://github.com/gfx-rs/wgpu/issues/10292) is a superset of [#9645](https://github.com/gfx-rs/wgpu/issues/9645). Land [#9645](https://github.com/gfx-rs/wgpu/issues/9645) first, then a split [#10292](https://github.com/gfx-rs/wgpu/issues/10292).
- **Vulkan workarounds:** [#10840](https://github.com/gfx-rs/wgpu/issues/10840) and [#10677](https://github.com/gfx-rs/wgpu/issues/10677) both add a workaround flag at bit `1 << 3`, which [#10692](https://github.com/gfx-rs/wgpu/issues/10692) (merged) also took; both need a new bit.
- **Child textures:** [#10795](https://github.com/gfx-rs/wgpu/issues/10795) is approved. [#10812](https://github.com/gfx-rs/wgpu/issues/10812) has the unfixed high finding and conflicts.
- **dx12 present (your queue):** [#10511](https://github.com/gfx-rs/wgpu/issues/10511) before [#10548](https://github.com/gfx-rs/wgpu/issues/10548).
- **CHANGELOG only conflicts:** [#10891](https://github.com/gfx-rs/wgpu/issues/10891), [#10897](https://github.com/gfx-rs/wgpu/issues/10897), [#10901](https://github.com/gfx-rs/wgpu/issues/10901), [#10796](https://github.com/gfx-rs/wgpu/issues/10796).

## 10. Infrastructure problems seen in CI

- The "Post head-approved check" job fails because a GitHub App installation is suspended ([#10698](https://github.com/gfx-rs/wgpu/issues/10698)).
- Cancelled macOS jobs are on the heads of [#10506](https://github.com/gfx-rs/wgpu/issues/10506) and [#10800](https://github.com/gfx-rs/wgpu/issues/10800), and on drafts [#10409](https://github.com/gfx-rs/wgpu/issues/10409), [#10790](https://github.com/gfx-rs/wgpu/issues/10790), [#9353](https://github.com/gfx-rs/wgpu/issues/9353), [#9975](https://github.com/gfx-rs/wgpu/issues/9975) and [#10803](https://github.com/gfx-rs/wgpu/issues/10803).
- Possible flakes, not confirmed: `player` `test_api` timeout ([#10896](https://github.com/gfx-rs/wgpu/issues/10896)), Metal `poll::wait_on_submission_with_timeout` ([#10866](https://github.com/gfx-rs/wgpu/issues/10866)).
- [#10278](https://github.com/gfx-rs/wgpu/issues/10278) has no check runs. The Metal ray tracing stack ([#10287](https://github.com/gfx-rs/wgpu/issues/10287), [#10288](https://github.com/gfx-rs/wgpu/issues/10288), [#10289](https://github.com/gfx-rs/wgpu/issues/10289), [#10292](https://github.com/gfx-rs/wgpu/issues/10292)) has no workflow runs at all.

## Limits of this pass

- No builds or tests were run. High and medium findings that say "from code" were not reproduced.
- [#10910](https://github.com/gfx-rs/wgpu/issues/10910), [#10862](https://github.com/gfx-rs/wgpu/issues/10862) and [#10839](https://github.com/gfx-rs/wgpu/issues/10839): the missing [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field is reasoned from the struct definition and the diffs.
- [#8388](https://github.com/gfx-rs/wgpu/issues/8388): the commits after the 11-06 review were not checked against the review threads.
- [#10875](https://github.com/gfx-rs/wgpu/issues/10875) iOS background behavior and [#10860](https://github.com/gfx-rs/wgpu/issues/10860) watchOS indirect dispatch were not checked on hardware.
- [#10891](https://github.com/gfx-rs/wgpu/issues/10891): the 2047 limit was read in the WGSL specification; Tint's behavior was not checked.
- [#10922](https://github.com/gfx-rs/wgpu/issues/10922): DXC `-denorm` values were not checked against DXC sources.
- The cause of the `CHECKS_FAILED` on [#10795](https://github.com/gfx-rs/wgpu/issues/10795) in the merge queue was not read.

---

# Details

Entries are grouped by area and sorted by number. **Reviewed** is the date of the last code review of the entry; day counts in an entry are as of that date. **State** is read again at each update.

## naga

#### [#9295](https://github.com/gfx-rs/wgpu/issues/9295): fix(naga/spv): emit layoutless types for Workgroup storage class variables ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-21.
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

#### [#9966](https://github.com/gfx-rs/wgpu/issues/9966): naga: support SPIR-V OpGroupNonUniformBallotFindLSB/MSB and GLSL subgroupBallotFindLSB/MSB ([@nazar-pc](https://github.com/nazar-pc))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI pending; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainers. inner-daemons wrote on 11-13 that he will raise it at the next maintainers meeting.
- **Last activity:** 2026-10-23, inner-daemons, comment handing off to cwfitzgerald (after an author ping the same day). 14 days ago. The author has pinged twice (09-14, 10-23).
- **Approvals:** none. inner-daemons' two CHANGES_REQUESTED reviews (2026-08-21) are still on record. The author addressed them on 2026-09-02 (whitespace split out to [#10131](https://github.com/gfx-rs/wgpu/issues/10131), GPU execution test added, fewer `write_expr` calls).
- **CI:** green, but stale (head 3981248 from 2026-09-02, codecov/patch stuck "in_progress"). The base is from August, so a rebase and re-run are needed.
- **Summary:** Adds `Statement::SubgroupBallotFindBit` (as jimblandy asked on 2026-08-05). It adds SPIR-V and GLSL front-end parsing, native SPIR-V and GLSL output, ternary-chain polyfills for HLSL, MSL and WGSL, validation gated on `SUBGROUP` + `BALLOT`, snapshots, and a wgpu GPU test.
- **Findings:**
  1. [medium] `CHANGELOG.md` adds a new `#### naga` heading (lowercase) instead of using the existing Naga or New Features subsection. It will conflict after a rebase. The entry also has no trailing period.
  2. [medium] `tests/Cargo.toml` turns on the `glsl` feature for the whole `wgpu-test` crate, only so that the new test can use a GLSL compute shader. Check that this is acceptable, or use SPIR-V passthrough or a pre-made SPIR-V.
  3. [low] There are 3 unresolved inner-daemons threads. "Argument evaluated many times" is outdated and was addressed by evaluating words into locals (`naga/src/back/hlsl/writer.rs:3074`). Two are not actionable ("weird function", "too tired to verify polyfill"), but the polyfill still has no second reviewer.
  4. [low] Behavior on an all-zero ballot is not documented in the IR doc or the polyfills. Confirm that the polyfills match the SPIR-V/GLSL semantics, which give an undefined or -1 result.
- **To land:** rebase onto trunk and get green CI. Fix the CHANGELOG placement. A maintainer must review the polyfills, and inner-daemons must dismiss or convert their CHANGES_REQUESTED review.
- **Move-along action:** review (cwfitzgerald). Ask inner-daemons to approve or dismiss their stale CHANGES_REQUESTED review.
- **Depth:** targeted (I read the timeline in full, the HLSL polyfill and the validation. I skimmed the MSL, WGSL and GLSL writers and the GPU test.)

**CI and status check ([#9966](https://github.com/gfx-rs/wgpu/issues/9966): naga: SPIR-V OpGroupNonUniformBallotFindLSB/MSB and GLSL subgroupBallotFindLSB/MSB):**

> - **Open questions from the first pass:** full CI on the current head.
> - **Answers:**
>   1. [confirmed] 64 check runs: 63 success, 1 `codecov/patch` still `in_progress`. No failure. The codecov entry is stuck and not a test result. Head 3981248, `blocked`, `mergeable: true` (so no conflict, despite the first pass expecting a CHANGELOG conflict after a rebase).
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10016](https://github.com/gfx-rs/wgpu/issues/10016): Emit spec-compliant SPIR-V under the Vulkan memory model, volatile support in MSL ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-09-27.
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

#### [#10229](https://github.com/gfx-rs/wgpu/issues/10229): glsl-out: never emit `invariant gl_FragCoord` on OpenGL ES ([@jonatino](https://github.com/jonatino))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-20.
- **Verdict:** Needs a bit of work (CHANGELOG only)
- **Waiting on:** maintainer (assignee jimblandy; nobody with write access has reviewed)
- **Last activity:** 2026-10-20, KalybosPro (outside user), comment with a real-world failure on the Android emulator (glyphon text shader). 17 days. Before that: 2026-09-09, grovesNL said "This seems correct to me".
- **Approvals:** none formal. grovesNL agrees in a comment.
- **CI:** green (63/63, from 2026-09-01)
- **Summary:** Widens the GLSL arm that skips `invariant gl_FragCoord` from WebGL2 only to all GLSL ES versions. `gl_FragCoord` is read-only in the ES fragment stage, so `invariant` on it is illegal.
- **Findings:**
  1. [low] `CHANGELOG.md` has no entry. This is a user-visible fix: GL/GLES on Android fails to compile shaders that use `@invariant @builtin(position)`.
  2. [low] `naga/tests/in/wgsl/invariant.toml` tests only `is_webgl = true`, so the new non-WebGL ES path has no snapshot. Set `is_webgl = false` (or add a second target). The snapshot should not change, and that proves the fix.
- **To land:** add a CHANGELOG line. The test tweak is optional.
- **Move-along action:** take over: add the CHANGELOG line (and optionally the toml tweak), then merge. The diff is 5 lines and the spec reasoning is confirmed.
- **Depth:** full

#### [#10278](https://github.com/gfx-rs/wgpu/issues/10278): fix(naga msl-out): typed 64-bit literals in integer div/rem zero guards ([@Statesman-forge](https://github.com/Statesman-forge))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI no runs; last update 2026-09-09.
- **Verdict:** Land as is (after CI runs)
- **Waiting on:** maintainer (CI workflow approval for a first-time contributor, then review. Assignee is jimblandy.)
- **Last activity:** 2026-09-07, Statesman-forge, opened the PR and pushed the CHANGELOG fixup. No human comments since. 60 days.
- **Approvals:** none
- **CI:** none. The head `ee279c43` has 0 check runs, so the workflows were probably never approved.
- **Summary:** The MSL `naga_div`/`naga_mod` helpers for `long`/`ulong` used 32-bit literals in `metal::select`, so the call was ambiguous and the MSL failed to compile. The PR emits `1L`/`1uL`/`0uL` etc. for 64-bit operands.
- **Findings:**
  1. [low] `naga/src/back/msl/writer.rs` (Divide/Modulo arms): the `Modulo` arm repeats the comment "See the corresponding comment ..." and duplicates the suffix logic from `Divide`. This is acceptable as is, but a small shared helper would be cleaner. It is not required.
  2. [low] The bug is still present on trunk (`writer.rs:6190`, `6216`, `6304`). The PR is based on 9920c99 (Sep 7), so a rebase may be needed. GitHub mergeable state is "unknown".
- **To land:** approve the CI workflows. If CI is green, merge. A rebase may be needed for the CHANGELOG.
- **Move-along action:** review (approve the workflow run, then merge if green). The code is small and correct, and the snapshot shows the typed guards.
- **Depth:** full

**CI and status check ([#10278](https://github.com/gfx-rs/wgpu/issues/10278): fix(naga msl-out): typed 64-bit literals in integer div/rem zero guards):**

> - **Open questions from the first pass:** is there CI on the head; is it because fork workflows await approval.
> - **Answers:**
>   1. [confirmed] No CI on the head `ee279c43`: 0 check runs on all pages.
>   2. [refuted] It is not `action_required` now. `actions/runs?head_sha=ee279c43...` returns 8 `pull_request` runs (CI, CTS, Shaders, Docs, cargo-generate, Lazy, changelog, Publish), all created 2026-09-07T17:50:29Z, all `completed` / `failure`, and each has 0 jobs (`runs/ID/jobs` total_count 0). The CI run was last updated 2026-10-07, 30 days later, and its logs return 404. The only `action_required` runs in the repo are 4 old Copilot runs from 2025-12-15 on another SHA.
>   3. [unknown] Why they failed with 0 jobs. The most likely reason is that a first-time-contributor run waited for approval and was auto-failed after 30 days, but the API does not state this. The author is `FIRST_TIME_CONTRIBUTOR`. The PR has 0 comments. `mergeable_state: dirty`.
>   4. [confirmed] Consequence: approving workflows is not possible now (the old runs are over). Someone must re-trigger CI: the author pushes a rebase (needed anyway, `dirty`), then a maintainer approves the new runs.
> - **New findings:** none
> - **Verdict change:** none ("Land as is (after CI runs)" still holds, but it needs a rebase first because of the conflict)
> - **Move-along action change:** replace "approve the CI workflows" with: ask the author to rebase (conflict, probably CHANGELOG), then approve the new workflow runs.

#### [#10343](https://github.com/gfx-rs/wgpu/issues/10343): naga/hlsl: emit unbounded `[]` for unsized binding arrays without a size override ([@arcusbuilds](https://github.com/arcusbuilds))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; merge state not computed; CI green; last update 2026-10-17.
- **Verdict:** Land as is
- **Waiting on:** maintainer (needs approval from someone with write access. The author pinged ErichDonGubler.)
- **Last activity:** 2026-10-17, arcusbuilds, pinged ErichDonGubler for a write-access review. 20 days.
- **Approvals:** inner-daemons (2026-09-24, on `54fe5e89`, before the two trunk merges. inner-daemons has no write access.)
- **CI:** green (65 success, 2 neutral)
- **Summary:** The HLSL writer hit `unreachable!()` for an unsized `binding_array<T>` with no `binding_array_size`, which affects only standalone naga. The PR emits `[]` for `ArraySize::Dynamic`.
- **Findings:**
  1. [low] `CHANGELOG.md`: the entry links to issue [#10267](https://github.com/gfx-rs/wgpu/issues/10267) and not to PR [#10343](https://github.com/gfx-rs/wgpu/issues/10343). This is cosmetic.
  2. [low] At SM 5.0 the emitted `[]` is invalid, but binding arrays need SM 5.1+ anyway, and the toml comment explains why the test uses SM 6.0. There is no action.
- **To land:** nothing (fixing the CHANGELOG link is optional).
- **Move-along action:** merge
- **Depth:** full

#### [#10414](https://github.com/gfx-rs/wgpu/issues/10414): feat(naga): support non-square cooperative matrix tiles ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-28.
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

#### [#10457](https://github.com/gfx-rs/wgpu/issues/10457): naga/glsl: avoid unsupported mix for vector select ([@arcusbuilds](https://github.com/arcusbuilds))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-14.
- **Verdict:** Land as is
- **Waiting on:** the merge queue. cwfitzgerald approved on 11-14, with a policy note to the author about the self-review box, and sent `@mergifyio queue`.
- **Last activity:** 2026-11-11, arcusbuilds, "Trimmed to a single WebGL2 snapshot and dropped the `supports_integer_mix` unit test." 2 days.
- **Approvals:** cwfitzgerald
- **CI:** green (66 pass) on the 11-11 head; has conflicts (CHANGELOG.md only, from `git merge-tree` against trunk `4696814d4`). Trunk moved after the author's 11-11 merge.
- **Summary:** The GLSL backend emits a componentwise `?:` for a vector `select` with integer or boolean values when the target is older than GLSL 4.50 or ES 3.10. Those versions have no integer `mix` overload. The three operands are baked, so each one is evaluated only once. It adds `Version::supports_integer_mix` to `naga-types`. The PR now has one snapshot test (`10093-select-vector-condition-webgl2`) and a CHANGELOG entry.
- **Findings:**
  1. [high, process] The PR body still has "This PR was automatically created by an Agent" and "The description was automatically written by an Agent" checked, and "I self-reviewed and fully understand this PR" unchecked. AGENTS.md does not allow fully agentic PRs. The code is small and looks correct, so this is a policy decision.
  2. [low] `naga/src/back/glsl/writer.rs:951-968` The bake rule bakes all three operands, also when they are cheap. The snapshot shows `ivec4 _e6 = (b + ivec4(7));` and `bvec4 _e7 = lessThan(a, b);`, which is correct but adds temporaries.
  3. [low] The PR does not use `GL_EXT_shader_integer_mix` on desktop GL when it is available, and it removes the TODO for it. The `?:` fallback is correct, so this is acceptable.
- **To land:** A maintainer decides on the agent policy (or the author checks the self-review box). cwfitzgerald dismisses or re-reviews the change request. The author rebases for the CHANGELOG conflict.
- **Move-along action:** none.
- **Depth:** full (the whole diff is small; I read all of it)

#### [#10518](https://github.com/gfx-rs/wgpu/issues/10518): naga: add experimental physical storage buffer pointers ([@nickdarnell](https://github.com/nickdarnell))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-07.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer (jimblandy assigned). Nobody has reviewed it. The only review is from the zizmor bot.
- **Last activity:** 2026-10-03, nickdarnell, opened the PR. 34 days ago. There is no human activity after that.
- **Approvals:** none
- **CI:** green on the visible jobs (head 96ca387 from 2026-10-03). There is a merge conflict (`dirty`). zizmor flagged `.github/workflows/shaders.yml:152,155` (use `$/...` self-repo syntax).
- **Summary:** About 12.3k added lines in one squashed commit. It adds `AddressSpace::PhysicalStorage`, pointer casts and offsets, i8/u8 scalars, `MatrixLoad`/`MatrixStore`, coherent and atomic pointer ops, `StructMember::access`, and `FunctionArgument::immutable_pointee`. It also adds a 798-line interprocedural alias/access analysis in the validator, SPIR-V in/out support, a new `naga/physical-storage-gpu` workspace crate, and a new CI job. All of it is behind new capabilities that no wgpu feature maps.
- **Findings:**
  1. [high] Policy: the author did not check "I self-reviewed and fully understand this PR". The author also states "Heavy Opus 5.5 + Astra 6" and "relied on ... the LLMs to work through all the work". AGENTS.md requires that the contributor can vouch for every change. At this size, in naga, that has to be established first.
  2. [high] Scope: the author admits that it is not minimal and that the commits are not logically scoped. It changes core IR types (`StructMember`, `FunctionArgument`: breaking public API, which touches every front end and snapshot). It adds 8-bit integer scalars, a new validator analysis pass, and four new `Capabilities` bits (`naga/src/valid/mod.rs`). Each of these needs its own design discussion and its own PR.
  3. [medium] The description and the `naga/src/valid/mod.rs` docs refer to `naga/PHYSICAL_STORAGE.md`, but that file is not in the PR. Both CHANGELOG entries use a `#NNNNN` placeholder. The description asks reviewers to "check whether default-members was actually updated" (it is correctly excluded), so the author has not verified the PR.
  4. [medium] i8/u8 scalars are added to override types (`naga/src/valid/mod.rs:741`) and to the constant evaluator (267 lines). These are reachable without the physical-pointer capability only if `SHADER_INT8` gating is complete in every path. The safe-boundary claim ("safe shader-module creation rejects the capability") depends on wgpu-core never setting these bits. That is true today, because no feature maps them and the PR does not touch wgpu-core or wgpu-naga-bridge.
  5. [low] A new CI job runs on Linux/lavapipe with a GPU harness crate that has 3.3k lines of test helpers. This adds maintenance cost.
- **To land:** a maintainer decision on whether naga IR should carry physical pointers. If yes, split the work into reviewable PRs (IR + validation, SPIR-V out, SPIR-V in, int8, harness). The author must confirm understanding, add the missing doc, fix the placeholders and rebase.
- **Move-along action:** decide whether physical-storage pointers in naga IR are wanted, and if they are, ask the author to split. Without that answer, nobody can start a review.
- **Depth:** targeted (I read the description, the file list, CHANGELOG, Cargo.toml, the capability definitions and the WGSL-lower changes. I did not read the validator analysis, the SPIR-V writer/reader or the 4.4k lines of tests.)

**Deeper review ([#10518](https://github.com/gfx-rs/wgpu/issues/10518): naga: add experimental physical storage buffer pointers):**

> - **Open questions from the first pass:**
>   1. Is everything behind an experimental capability that only an unsafe opt-in can enable?
>   2. Can untrusted WGSL (or other safe input) reach the new code without that capability?
>   3. Does the new IR have validation holes?
>   4. How can the PR be split?
> - **Answers:**
>   1. [refuted, partly] The pointer machinery is gated, but one new IR feature is not.
>      - **Gated:** `AddressSpace::PhysicalStorage` pointer and value-pointer types require `PHYSICAL_STORAGE_BUFFER_ADDRESSES` (`naga/src/valid/type.rs:546,628`). Globals cannot use the address space (`naga/src/valid/interface.rs:1021`). `PointerCast` needs a physical pointer on one side and `u64` on the other. `PointerOffset`, `PointerAlignment` and `AtomicPointer` need a physical pointer. `CoherentPointer` also needs `COHERENT_PHYSICAL_MEMORY`. `MatrixLoad` and `MatrixStore` need a physical pointer and `SHADER_INT64` (`naga/src/valid/expression.rs` hunks at ~1347-1450 and 1556; `function.rs` `S::MatrixStore`). `immutable_pointee` needs a physical pointer argument (`naga/src/valid/function.rs` ~1969). Width-1 integers need `SHADER_INT8` (`naga/src/valid/type.rs:325,358`). The wgpu crates do not set any of these bits (I searched wgpu-core, the bridge, hal, wgpu and wgpu-types).
>      - **Not gated:** `StructMember::access`. The SPIR-V front end fills it from `NonWritable`/`NonReadable` member decorations on any struct (`naga/src/front/spv/mod.rs:2643`). The type validator only rejects unknown flags (`naga/src/valid/type.rs:756`). The snapshot diff shows the effect: the existing glslang test `spv-shadow.ron` now has `access: Some(("LOAD"))`. Any module with such a member runs `valid::access::validate` for every function (`naga/src/valid/access.rs:737`, called from `naga/src/valid/function.rs:2103`). The SPIR-V back end then emits `NonReadable`/`NonWritable` member decorations (`naga/src/back/spv/writer.rs:3853`), which trunk never emitted for input that is not physical.
>   2. [refuted for SPIR-V input, confirmed for WGSL] WGSL has no new syntax. The lowering always sets `access: None` and `immutable_pointee: false`, so the access analysis returns early. The overload sets now include I8/U8 (`naga/src/proc/overloads/scalar_set.rs:61-128`), and the WGSL lowering filters them in two places. If one site is missed, the result is a `SHADER_INT8` validation error, not unsafety. The WGSL snapshots do not change. Safe SPIR-V input (`ShaderSource::SpirV`) does reach new code without the capability: the member-access analysis (answer 1). The front end also now accepts `PhysicalStorageBufferAddresses` in the strict capability list (`naga/src/front/spv/mod.rs:57`), so the type validator rejects the input later instead of the parser. The native.rs parse paths that I read have no reachable panics (the `shift.unwrap()` calls are guarded).
>   3. [confirmed: holes exist, all in the ungated path]
>      - The member-access analysis is now the only thing that keeps an untrusted write away from a member that the output SPIR-V declares `NonWritable`. A write to `NonWritable` memory is invalid SPIR-V. On trunk the decoration was dropped, so no such risk existed. The 798-line flow analysis on the untrusted path needs its own review. I did not find a missing write path: `Store`, `Atomic`, `MatrixStore`/`CooperativeStore` and `Call` are all checked (`access.rs:518-618`).
>      - Possible false rejection and DoS: the analysis inlines every callee at every call site (`access.rs:582-618`). It iterates loops to a fixed point and stops after 100,000 steps for each validated function (`access.rs:777`), with `PointerAccessAnalysisLimit`. Each step can clone the full `State`. A large glslang shader with a `readonly buffer` and a deep call graph can now fail validation where it passed on trunk. I did not measure this.
>      - The restrictions are tracked by type, not by address space (`pointee_restrictions`, `access.rs:251`). A `Function` variable whose struct type is the same decorated type cannot have that member written. This is an edge case.
>      - `wgpu/src/backend/webgpu.rs:2154`, pre-existing pattern: the WebGPU (wasm) back end validates SPIR-V/GLSL input with `Capabilities::all()` and then calls `back::wgsl::write_string(...).unwrap()`. With this PR, i8 or physical-pointer SPIR-V input passes validation, and then the WGSL writer's new `Int8Unsupported`/`PhysicalStorageUnsupported` errors cause a panic in that path. This is not memory unsafety, but the PR adds new panic inputs for a safe API.
>      - Unrelated tightening: `CooperativeLoad` now requires `COOPERATIVE_MATRIX` and an f16 or f32 pointee (`naga/src/valid/expression.rs:1582`). This is probably harmless, but it changes an existing feature and has no test or changelog entry.
>   4. Proposed split, in dependency order. Each part should be reviewable by itself:
>      1. `i8`/`u8` scalars: `SHADER_INT8`, the layouter, the constant evaluator (267 lines), SPIR-V in/out of width-1 constants, and the overload-set changes. The SPIR-V front end's new width-2 constant parsing also belongs here or in a separate fix.
>      2. The IR field changes as a pure API PR: `StructMember::access` and `FunctionArgument::immutable_pointee`, with defaults, without the SPIR-V front end setting `access`. Discuss separately whether naga should honor member `NonWritable` from SPIR-V at all. If yes, it needs its own analysis PR and its own budget review. If no, `access` should be accepted only on structs that are reachable through physical pointers, or behind a capability.
>      3. `AddressSpace::PhysicalStorage` with type validation, `PointerCast`/`PointerOffset`/`PointerAlignment`, and the SPIR-V writer for them.
>      4. The SPIR-V front end for physical pointers (`native.rs`, `next_block.rs`).
>      5. Ordered and coherent atomics (`AtomicPointer`, `CoherentPointer`, `COHERENT_PHYSICAL_MEMORY`, failure order).
>      6. `MatrixLoad`/`MatrixStore` and the cooperative-matrix changes, including the `CooperativeLoad` tightening.
>      7. The `naga/physical-storage-gpu` harness crate and the CI job, last and optional.
> - **New findings:**
>   1. [high] Ungated untrusted-input path: `StructMember::access` from SPIR-V decorations plus the access analysis plus re-emitted decorations (see answers 1 and 3). This contradicts the CHANGELOG claim that safe wgpu shader-module creation rejects the new features.
>   2. [medium] Risk of false rejection and performance cost from the inlining analysis with a 100,000-step budget, on every SPIR-V module with readonly or writeonly block members. Not measured.
>   3. [low] New panic inputs in `wgpu/src/backend/webgpu.rs:2154` (`unwrap` after validating with `Capabilities::all()`).
>   4. [low] Capability bit `1 << 46` collides with [#10311](https://github.com/gfx-rs/wgpu/issues/10311) (`RAY_TRACING_INVOCATION_REORDER`). The PR that lands second must renumber its bits.
> - **Verdict change:** none ("Needs a lot of work"). Answer 1 is a stronger reason for the same verdict.
> - **Move-along action change:** add to the maintainer decision: "member access restrictions from SPIR-V must not be parsed outside the physical-pointer capability (or need their own design)". Give the author the split list above.
>
> ---

**CI and status check ([#10518](https://github.com/gfx-rs/wgpu/issues/10518): naga: add experimental physical storage buffer pointers):**

> - **Open questions from the first pass:** full CI.
> - **Answers:**
>   1. [confirmed] 68 check runs: 66 success, 2 neutral (Mergify). No failure. Head 96ca387. `dirty` (conflict).
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10561](https://github.com/gfx-rs/wgpu/issues/10561): [spv-in] Support NonSemantic.DebugPrintf ([@evilpie](https://github.com/evilpie))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI red (1 failed); last update 2026-10-28.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (jimblandy assigned; no review in 30 days)
- **Last activity:** 2026-10-24, evilpie, pushed the CHANGELOG commit. 13 days ago. There are no human comments or reviews at all.
- **Approvals:** none
- **CI:** red. "Test Windows x86_64" failed. The log tail shows only wgpu-info output, so I could not find the cause. Test Linux and Test Mac passed, and the change is naga spv-in only, so the failure is probably unrelated, but this is not confirmed. The PR also has a merge conflict (`mergeable_state: dirty`).
- **Summary:** The SPIR-V front end now accepts the `NonSemantic.DebugPrintf` ext-inst set. It records `OpString` values and lowers `DebugPrintf` to `Statement::DebugPrintf`. Validation still requires `DEBUG_PRINTF` and checks the format specifiers. Adds one spvasm snapshot.
- **Findings:**
  1. [medium] Merge conflict, and the red Windows test job needs a re-run after the rebase.
  2. [low] `naga/src/front/spv/next_block.rs` ~1697: the GLSL.std.450 arm is wrapped in an `else if` and re-indented, so about 130 lines of churn hide a 30-line change. If `if Some(set_id) == self.ext_debug_printf_id { ...; continue; }` is placed before the existing check, the old structure stays.
  3. [low] The only test is the positive case. There are no error-path tests: unknown string id, `DebugPrintf` without the `DEBUG_PRINTF` capability, an unsupported ext-inst number, or a Vulkan-only specifier such as `%v4f`, which naga's validator rejects.
  4. [low] Operand-count math is correct: `expect_at_least(6)` returns `wc - 6`, which is the argument count.
- **To land:** rebase, get green CI, and optionally reduce the re-indent and add one negative test.
- **Move-along action:** review (small: about 100 lines of real code). Reassign from jimblandy if they have no time.
- **Depth:** full

**CI and status check ([#10561](https://github.com/gfx-rs/wgpu/issues/10561): [spv-in] Support NonSemantic.DebugPrintf):**

> - **Open questions from the first pass:** cause of the "Test Windows x86_64" failure; related to the PR or not.
> - **Answers:**
>   1. [confirmed] It is one job, 1 of 67. Job 120851992387, run 2026-10-24. The log shows `nextest`: 1563 tests run, 1562 passed, 1 timed out, 12 skipped. The failing test is `player::player tests::test_api`, `TIMEOUT [113.797s] (513/1563)` (log line 2479 and the summary at line 10450). The job then exits with code 1 (line 10459).
>   2. [confirmed] It is unrelated. The PR changes only `naga/src/front/spv/*`, `naga/tests/{in,out}` and `CHANGELOG.md`. `player` is a wgpu-core replay test. The log shows it replaying `bind-group.ron`, `buffer-copy.ron` and others, and being cut off by the timeout. It runs on a Windows runner with a software or real GPU adapter; I cannot say which. All `naga` tests passed in the same job. Test Linux and Test Mac passed. I did not run anything; this is from the log.
>   3. [unknown] Whether this is a known flaky test. I did not find other timeouts in the PRs I checked, since the other Windows jobs on these PRs passed.
>   4. [confirmed] `mergeable_state: dirty` (merge conflict), as the first pass said. A rebase gives a new run anyway.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none (rebase, and the Windows job will rerun)

#### [#10583](https://github.com/gfx-rs/wgpu/issues/10583): [naga msl-out] preserve equivalent array aliases ([@pbkx](https://github.com/pbkx))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-10.
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

#### [#10585](https://github.com/gfx-rs/wgpu/issues/10585): [naga hlsl-out] emit 64-bit integer literal suffixes ([@pbkx](https://github.com/pbkx))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI red (1 failed); last update 2026-10-17.
- **Verdict:** Needs a bit of work (rebase only)
- **Waiting on:** author (merge conflict) or maintainer (could resolve the CHANGELOG conflict and merge)
- **Last activity:** 2026-10-17, inner-daemons, approved after the author addressed both nits. 20 days.
- **Approvals:** inner-daemons (2026-10-17, on the current head `443c6345`. inner-daemons has no write access.)
- **CI:** red: `Test Windows x86_64`. This looks unrelated: 3 timeouts (`timestamp_normalization::utils::u64_mul_u32`, `shift_right_u96` on DX12 WARP, and `bind_groups::bind_group_with_max_binding_index` on GL llvmpipe), and the timed-out shaders contain no 64-bit literals. The new `int64_literals` test passed on DX12. A rerun is needed.
- **Summary:** The HLSL backend now writes `LL`/`ULL` and not `L`/`uL`. Under DXC with `-HV 2018` (which wgpu-hal uses), `L` does not give a 64-bit literal, so large values were truncated. The PR adds a GPU regression test.
- **Findings:**
  1. [low] `naga/src/back/hlsl/help.rs:936-937`: the storage-texture scalar wrapper still emits `0uL`/`1uL`. The values are 0 and 1, so they are not truncated, but this is inconsistent. A one-line follow-up fixes it.
  2. [low] GitHub reports `dirty` (merge conflict, probably CHANGELOG).
- **To land:** rebase, then rerun Windows CI.
- **Move-along action:** ping author to rebase (or take over the CHANGELOG conflict), then merge.
- **Depth:** full

#### [#10596](https://github.com/gfx-rs/wgpu/issues/10596): Support BF16 type storage and cooperative matrix operations ([@kvark](https://github.com/kvark))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
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

#### [#10597](https://github.com/gfx-rs/wgpu/issues/10597): naga: make Handle::from_usize public ([@kuntal-devrat](https://github.com/kuntal-devrat))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-31.
- **Verdict:** Needs a bit of work (on a decision, not on the code)
- **Waiting on:** maintainer (decision). jimblandy objected on 2026-10-14. inner-daemons argued for it on 2026-10-22 (in [#10088](https://github.com/gfx-rs/wgpu/issues/10088)). cwfitzgerald wrote on 2026-10-14 in [#10088](https://github.com/gfx-rs/wgpu/issues/10088) that they would "be happy to accept [#10597](https://github.com/gfx-rs/wgpu/issues/10597)" if it is required. On 2026-10-31 the author restated the use case and offered alternatives (`TryFrom<usize>`, a feature gate, `HandleMap`).
- **Last activity:** 2026-10-31, kuntal-devrat, comment restating the use case and offering alternatives. 6 days ago.
- **Approvals:** inner-daemons (2026-10-14, on the old commit 0585e13, not the current head 9b793f9)
- **CI:** green (head 9b793f9, 2026-10-31). `mergeable_state: blocked` because no approval is on the current head.
- **Summary:** Makes `Handle::from_usize` `pub`, with a `# Panics` doc and unit tests. The use case is from [#10088](https://github.com/gfx-rs/wgpu/issues/10088) (Random-Scientist): an `Index` impl that packs a handle into slice metadata.
- **Findings:**
  1. [medium] Process: the PR says "This PR was automatically created by an Agent" and "The description was automatically written by an Agent". AGENTS.md does not allow fully agentic PRs. The PR author is also not the person with the use case.
  2. [medium] Design: jimblandy's objection still stands. A public `usize -> Handle` conversion weakens arena type safety. The use case is unusual, and inner-daemons agrees that it does not *need* this API.
  3. [low] The code is trivial and correct. The panic message changed from "Failed to insert into arena. Handle overflows" to "Handle index overflows", which is still accurate for the arena path.
- **To land:** a decision (public, `unsafe`/`TryFrom`, or no). If the answer is yes, a fresh approval on the current head.
- **Move-along action:** decide whether `Handle::from_usize` goes public (jimblandy and inner-daemons disagree). If the answer is no, close it as fully agent-created with a use case that does not need it, and point Random-Scientist to a `HandleMap` follow-up.
- **Depth:** full

**CI and status check ([#10597](https://github.com/gfx-rs/wgpu/issues/10597): naga: make Handle::from_usize public):**

> - **Open questions from the first pass:** full CI.
> - **Answers:**
>   1. [confirmed] 67 check runs: 65 success, 2 neutral. No failure. Head 9b793f9. `blocked` (no approval on the head), `mergeable: true`.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10600](https://github.com/gfx-rs/wgpu/issues/10600): refactor(naga spv-out): Add wrapped function preamble helper ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI red (47 failed); last update 2026-11-07.
- **Verdict:** Needs a bit of work (does not compile on current trunk)
- **Waiting on:** author (adapt to [#10311](https://github.com/gfx-rs/wgpu/issues/10311) and rebase). The author asks that review go ahead before that. jimblandy is assigned and has not reviewed.
- **Last activity:** 2026-11-07, andyleiserson, comment about the [#10311](https://github.com/gfx-rs/wgpu/issues/10311) conflict. 6 days.
- **Approvals:** none
- **CI:** red (47 fail). The PR causes it: CI builds the merge with trunk, and `naga/src/back/spv/ray/hit_object.rs` (added by [#10311](https://github.com/gfx-rs/wgpu/issues/10311)) calls the old `write_function_signature(&[ty], ret)` signature. Errors are E0061 and E0599 (`consume` on `u32`) at `hit_object.rs:51`, `:94` and later. Also a Windows `install-action` bash startup failure, which is infra. Has conflicts: `naga/tests/out/spv/wgsl-image.spvasm` and `wgsl-skybox.spvasm` (from `git merge-tree`; these files also changed in [#10905](https://github.com/gfx-rs/wgpu/issues/10905)).
- **Summary:** Adds `write_function_signature` and `write_wrapped_function_signature` in `spv/helpers.rs`. They emit `OpFunction`, `OpFunctionParameter` and the optional `OpName`s. The div/mod, std140-conversion and ray-query/ray-pipeline helpers move to them, and the ray helpers get debug names. Most changed files are SPIR-V snapshot churn.
- **Findings:**
  1. [medium] `naga/src/back/spv/ray/hit_object.rs:51,183,281,507,588` (trunk) [#10311](https://github.com/gfx-rs/wgpu/issues/10311) added 5 calls to the old helper signature. [#10600](https://github.com/gfx-rs/wgpu/issues/10600) removes that signature, so the merged tree does not compile. Each call needs the same conversion as the ray-query calls: a name, `(type, name)` parameter pairs, and the `(function, id, args)` return order. Then the hit-object snapshots must be regenerated. (Confirmed from CI logs.)
  2. [medium] Semantic conflict with [#10697](https://github.com/gfx-rs/wgpu/issues/10697) (one call in `write_ray_query_get_current_t_max`). This is unchanged; see Cross-PR notes.
  3. [low] `naga/src/back/spv/writer.rs` (std140 conversion) The `format!("{:?}_from_std140", ...)` debug name is now built when `WriterFlags::DEBUG` is off too. The cost is small.
  4. [low] The function ID is now allocated after the type lookups. This gives ID-order snapshot churn with no behavior change.
- **To land:** Rebase onto trunk. Convert the 5 `hit_object.rs` calls. Resolve the 2 snapshot conflicts and regenerate SPIR-V snapshots. Get a review (jimblandy).
- **Move-along action:** ask jimblandy to review the helper now, as the author asks, and ask andyleiserson to rebase onto [#10311](https://github.com/gfx-rs/wgpu/issues/10311) after that.
- **Depth:** targeted (I checked the CI failure, the conflict and the call sites; I did not re-read the full refactor, because I cannot tell what the 11-07 force push changed)

#### [#10609](https://github.com/gfx-rs/wgpu/issues/10609): naga: Support GLSL matrix interfaces and column component access ([@rleux](https://github.com/rleux))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-09.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-23.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (SAY-5). inner-daemons requested a snapshot test on 2026-10-22, and the author has not replied.
- **Last activity:** 2026-10-23, inner-daemons, comment explaining to BenjaminBrienen why a snapshot test is wanted beyond the validation test. 14 days ago.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-28.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (ilyas-mallah). On 2026-10-27, inner-daemons asked whether the LLM-written description breaks the contributor policy, and the author has not answered. The author pushed a trunk merge about 11 h after the question but did not reply. A third party (HashimTheArab, not a contributor) answered with a hostile comment on 2026-10-28.
- **Last activity:** 2026-10-28, HashimTheArab, hostile comment. The last maintainer-side activity is 2026-10-27 (inner-daemons). 9 days ago.
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
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-10-28.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-28.
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
  5. [low] The GPU test `single_hit` generates a second hit after a commit. On Vulkan this depends on the [#10698](https://github.com/gfx-rs/wgpu/issues/10698) fix.
- **To land:** a maintainer decision on the API; a rebase; adaptation to [#10600](https://github.com/gfx-rs/wgpu/issues/10600) if that lands first; baking of the expression (or a reasoned answer on why it is not needed); an answer to Vecvec's init-tracking note.
- **Move-along action:** decide whether `rayQueryGetCurrentTMax` is an API we want in the `wgpu_ray_query` extension (and ping jimblandy, who is assigned)
- **Depth:** targeted (I read the IR, wgsl-in, validator, the spv/hlsl/msl lowering, the tests and the docs. I did not read the generated snapshot files line by line.)

**Deeper review:** see the block under [#10600](https://github.com/gfx-rs/wgpu/issues/10600).

#### [#10698](https://github.com/gfx-rs/wgpu/issues/10698): naga(spv-out): Fix UB in `rayQueryGenerateIntersection` ([@DouglasDwyer](https://github.com/DouglasDwyer))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-14.
- **Verdict:** Land as is
- **Waiting on:** the merge queue. The author merged trunk, and cwfitzgerald sent `@mergifyio queue` on 11-14.
- **Last activity:** 2026-11-12, ssoj13, confirmation comment. 1 day.
- **Approvals:** Vecvec
- **CI:** green for code (runs from October). "Post head-approved check" fails because the GitHub App installation is suspended. This is infra. Has conflicts: CHANGELOG.md only. `naga/src/back/spv/ray/query.rs` merges cleanly with trunk (from `git merge-tree`).
- **Summary:** A one-line fix. When a hit is committed, the SPIR-V GenerateIntersection helper reads `OpRayQueryGetIntersectionTKHR` with the committed intersection instead of the candidate intersection. Reading the candidate `t` of an AABB is UB. It updates 6 snapshots and adds a CHANGELOG entry.
- **Findings:**
  1. [low] There is no runtime test. The snapshots show the change, and the GPU test in [#10697](https://github.com/gfx-rs/wgpu/issues/10697) exercises this path. A user now confirms the fix on hardware.
- **To land:** rebase for the CHANGELOG conflict. The CI runs are old, so let CI run again after the rebase.
- **Move-along action:** none.
- **Depth:** status only

#### [#10796](https://github.com/gfx-rs/wgpu/issues/10796): fix(hlsl-out): handle overflowing storage buffer offsets ([@kuntal-devrat](https://github.com/kuntal-devrat))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (agent-policy decision), then author (summed-offset fix, inline comment, rebase)
- **Last activity:** 2026-11-09, inner-daemons, approval. 4 days.
- **Approvals:** inner-daemons (GitHub still reports REVIEW_REQUIRED)
- **CI:** green (65 pass); has conflicts (CHANGELOG.md only)
- **Summary:** In `fill_access_chain`, `stride * index` for a constant index into a storage array is now `checked_mul`. On overflow, the whole address becomes `4294967295`. This removes a debug panic and a release wrap to an in-bounds address. It fixes [#10753](https://github.com/gfx-rs/wgpu/issues/10753).
- **Findings:**
  1. [high, policy] The checklist has "This PR was automatically created by an Agent" and "The description was automatically written by an Agent" checked. "I self-reviewed and fully understand this PR" is also checked, and the body says "I reviewed the diff and understand the fix". So the old finding still holds: by its own disclosure, an agent created the PR and wrote the description. The approval does not change this. A maintainer must decide if it is acceptable.
  2. [medium] `naga/src/back/hlsl/storage.rs:141-157` The fix is still incomplete. The chain terms are emitted as separate HLSL `a+b` additions, so a large sum still wraps at runtime. Example: `array<S>` with `struct S {a,b,c: u32}` (stride 12), `buf[357913941].c` gives `4294967292+8`, which wraps to 4, an in-bounds address. Only the `stride * index` product is checked (`storage.rs:648`). (Reasoned from code, not run.)
  3. [low] `naga/tests/in/wgsl/storage-buffer-offset-overflow.wgsl:6` inner-daemons' request for a comment or an issue-number file name is not addressed.
  4. [low] The address `0xFFFFFFFF` is not 4-byte aligned. `ByteAddressBuffer` ignores the low bits, so it is still out of bounds. This is acceptable.
- **To land:** a policy decision. If accepted: check the summed constant offsets for overflow, with a test for the struct-member case. Address the inline comment. Rebase.
- **Move-along action:** decide whether to accept this agent-created PR. If yes, tell inner-daemons and the author about the summed-offset wrap before merge.
- **Depth:** status only for code (no new commits); I re-checked finding 2 in the current code.

#### [#10797](https://github.com/gfx-rs/wgpu/issues/10797): [naga msl-out] Keep lanes 32-63 in subgroupBallot ([@ErkoRisthein](https://github.com/ErkoRisthein))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-14.
- **Verdict:** Land as is
- **Waiting on:** the merge queue. cwfitzgerald approved on 11-14 and sent `@mergifyio queue`. If the State line shows conflicts, the queue cannot take it until a CHANGELOG rebase.
- **Last activity:** 2026-11-09, ErkoRisthein, reply and force push. 4 days.
- **Approvals:** inner-daemons, cwfitzgerald
- **CI:** green (66 pass, Mac jobs included); has conflicts (CHANGELOG.md only)
- **Summary:** The MSL `subgroupBallot` truncated the 64-bit `simd_ballot` mask to `uint`, so lanes 32-63 were lost on 64-wide AMD GPUs. The output is now `uint4(as_type<uint2>((uint64_t)simd_ballot(p)), 0, 0)`. The PR also refactors `subgroup_operations/mod.rs` into a shared `run` function and adds `upper_lanes.wgsl`, a test with no divergent control flow.
- **Findings:**
  1. [low] `naga/src/back/msl/writer.rs:4510`: correct. `as_type<uint2>` on a 64-bit value puts the low word in `.x`, which matches the WGSL lane layout. The author tested on AMD (R9 M370X) and Apple M3.
  2. [low] The new test's out-of-range lanes (`subgroupShuffleUp/Down`, `subgroupBroadcast(id, 63u)` on narrower subgroups) are guarded, so indeterminate values are not checked. The test has no failure expectations, and CI is green on all backends.
  3. [low] The original `SUBGROUP_OPERATIONS` failure expectations still know only the Apple pattern, so that test can still fail on AMD Macs for divergent-flow cases. The author leaves this out of scope. This is acceptable.
- **To land:** rebase for the CHANGELOG conflict. Optionally, inner-daemons looks at the new test, because it was added after the approval.
- **Move-along action:** check that the queue took it.
- **Depth:** full (fix and new test; the `mod.rs` change is a refactor into `run`)

#### [#10800](https://github.com/gfx-rs/wgpu/issues/10800): Fix [#10736](https://github.com/gfx-rs/wgpu/issues/10736) by removing the wg_memory_sizes logic ([@inner-daemons](https://github.com/inner-daemons))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI 10 cancelled jobs; last update 2026-11-06.
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

#### [#10820](https://github.com/gfx-rs/wgpu/issues/10820): Remove shader `F16_IN_F32` downlevel flag ([@szymros](https://github.com/szymros))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI pending; last update 2026-11-11.
- **Verdict:** Needs a bit of work (small: CHANGELOG and one stale comment)
- **Waiting on:** author (CHANGELOG entry, stale comment), then maintainer to merge (inner-daemons assigned)
- **Last activity:** 2026-11-11, szymros, force push. 2 days.
- **Approvals:** inner-daemons (on `909e3c92a`, before the 11-10 and 11-11 force pushes; GitHub reports REVIEW_REQUIRED)
- **CI:** green (64 pass; codecov/project pending, which does not block); mergeable
- **Summary:** Removes `wgt::DownlevelFlags::SHADER_F16_IN_F32`, `naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32`, the validator check on `quantizeToF16`/`pack2x16float`/`unpack2x16float`, the llvmpipe special case in the Vulkan adapter, the bridge mapping and the related tests. The llvmpipe bug is fixed upstream (Mesa 25.3.0).
- **Findings:**
  1. [medium] `CHANGELOG.md` The PR removes public API from `wgpu-types` and `naga`, but has no CHANGELOG entry. The checklist box is unchecked.
  2. [low] `naga/src/valid/mod.rs:247` A stale comment still names `SHADER_FLOAT16_IN_FLOAT32`.
  3. [low] The PR description does not note that Firefox must block Mesa before 25.3.0 on its side, as inner-daemons suggested.
- **To land:** a CHANGELOG entry under breaking changes. Remove the stale comment. Optionally, add the Firefox note to the description. Merge.
- **Move-along action:** ping szymros for the CHANGELOG entry and stale comment, then merge.
- **Depth:** targeted (I read the full diff again; it is all deletions plus 2 comments)

#### [#10873](https://github.com/gfx-rs/wgpu/issues/10873): Plumbing for external textures in SPIR-V backend ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI no runs; last update 2026-11-07.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
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

#### [#10897](https://github.com/gfx-rs/wgpu/issues/10897): [naga wgsl-in] Reject composite `const_assert` conditions instead of panicking ([@MilosLord](https://github.com/MilosLord))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-09.
- **Verdict:** Land as is
- **Waiting on:** author or maintainer (resolve the `CHANGELOG.md` conflict, then merge).
- **Last activity:** 2026-11-09, inner-daemons, approval and self-assignment. 4 days.
- **Approvals:** inner-daemons
- **CI:** green; has conflicts (`CHANGELOG.md` only)
- **Summary:** `const_assert vec2i()` and similar composite conditions panic in the WGSL front end (`unreachable!()` at `lower/mod.rs:1508` and `:2374`), because `get_const_val_from` returns `NonConst` for anything other than a scalar `Literal` or `ZeroValue`. The PR maps `NonConst` to the existing `Error::NotBool`, which is the error `const_assert(5)` already gives. This removes a panic on untrusted shader input found by fuzzing.
- **Findings:**
  1. [low] I checked `GlobalCtx::get_const_val_from` (`naga/src/proc/mod.rs:571`). It handles `Literal`, scalar `ZeroValue` and `Constant` (through the constant initializer). A valid `bool` const expression always evaluates to one of these, so `NonConst` here means "not a scalar bool". The fix is correct.
  2. [low] `naga/tests/naga/wgsl_errors.rs:4325` The test covers module scope and function scope. It does not cover the named-constant case from the description (`const v = vec2(true); const_assert v;`). Optional.
  3. [low] PR template: both agent-disclosure boxes are present and unchecked. The "LLM Use?" section says Claude Code and Codex wrote the fix, test and first description draft, and the author reviewed them.
- **To land:** rebase to resolve the CHANGELOG conflict; merge.
- **Move-along action:** merge after a CHANGELOG rebase (inner-daemons is assigned and approved).
- **Depth:** full

#### [#10901](https://github.com/gfx-rs/wgpu/issues/10901): Naga coop store emit ([@Sandvoxel](https://github.com/Sandvoxel))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (inner-daemons asked for a naga validator check on 11-13; CHANGELOG conflict).
- **Last activity:** 2026-11-13, inner-daemons, approval plus a request to add validation. 0 days.
- **Approvals:** inner-daemons (with a follow-up request)
- **CI:** green; has conflicts (`CHANGELOG.md`)
- **Summary:** Fixes two cooperative-matrix bugs in the SPIR-V path. (1) The WGSL front end pushed `Statement::CooperativeStore` without first flushing the emitter, so a non-constant stride was emitted after the store, and spv-out panicked with "Expression [N] is not cached". The fix flushes the emitter like `textureStore` does. (2) When a cooperative matrix type appears only as an expression result (for example `let m = coopLoad(...)`), compaction removes the type, so spv-out did not request `SPV_KHR_cooperative_matrix` or the Vulkan memory model. The new `Writer::require_cooperative_matrix()` is now also called from the `CooperativeLoad` and `CooperativeMultiplyAdd` paths. Adds a snapshot case and a GPU test.
- **Findings:**
  1. [medium] `naga/src/valid/function.rs:1692` The validator arm for `Statement::CooperativeStore` checks the target type and the pointer, but it never checks `data.stride`: not that it is in scope (emitted before the statement), and not its type. This is why bug (1) reached spv-out as a panic. inner-daemons asked for this check on 11-13. Other front ends (SPIR-V in) or hand-built IR can still trigger the spv-out panic. The feature is experimental and gated, so this is not a safe-API hole. (Reasoned from code.)
  2. [low] `naga/src/back/spv/block.rs:2227`, `:2269` The capability request is now added at each cooperative expression. A more general fix is to request capabilities when a local cooperative-matrix type id is created. [#10596](https://github.com/gfx-rs/wgpu/issues/10596) fixes the same bug in a different way (see Cross-PR notes).
  3. [low] PR template: the agent-disclosure lines are not present. The "LLM Use?" section says Claude found the bugs and wrote the fixes and tests, and the author reviewed them.
  4. [low] The description notes that Vulkan needs the stride to be a multiple of 16 bytes, and nothing validates this. The GPU test uses `size + 4` to stay legal. This is a known follow-up, not a defect of this PR.
- **To land:** add the validator check that inner-daemons asked for (stride in scope and `u32`, with a test); rebase for the CHANGELOG conflict; coordinate with [#10596](https://github.com/gfx-rs/wgpu/issues/10596).
- **Move-along action:** wait for Sandvoxel to add the validator check; no maintainer action now.
- **Depth:** full (I read the GPU test; I did not check the MSL output beyond the snapshot diff)

#### [#10908](https://github.com/gfx-rs/wgpu/issues/10908): naga: Preserve SPIR-V NoContraction through translation ([@5andr0](https://github.com/5andr0))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-12.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer. A naga maintainer must decide if naga wants a new per-expression IR field (`Function::expression_controls`) at all. On [#9295](https://github.com/gfx-rs/wgpu/issues/9295) (2026-11-13), the author says that he has no time for "the human part" and asks maintainers to treat his PRs as issues with suggested fixes.
- **Last activity:** 2026-11-12, 5andr0, PR body edit. 1 day.
- **Approvals:** none
- **CI:** green (66 pass, 3 skipped); mergeable
- **Summary:** Adds `ArithmeticControl::NoContraction` and a `Function::expression_controls` map to the naga IR. The SPIR-V frontend reads `NoContraction` on add/sub/mul/dot, and the SPIR-V backend writes it again. GLSL 4.00+/ESSL 3.20+ and HLSL SM 6+ put each controlled operation into a `precise` helper function (HLSL: `[noinline]` with an `inout` operand array, to avoid an NVIDIA D3D12 driver bug). MSL, WGSL and older GLSL/HLSL return a new error. The HLSL writer also hoists loop-body values into variables before the loop so that the `continuing` block can read them. 1465 lines added, about 1000 of them in `naga/src`.
- **Findings:**
  1. [high, reasoned from code, not run] `naga/src/front/spv/mod.rs:1035`, `naga/src/valid/function.rs:2184` Regression for SPIR-V input that trunk accepts. On trunk, `NoContraction` goes to the "Unknown decoration" arm and is ignored. With this PR:
     - `NoContraction` on any op other than add/sub/mul/dot (for example `OpFDiv`, `OpFNegate`, or GLSL.std.450 ext-insts) gives `UnsupportedNoContraction`, either in `register_arithmetic_control` or from the left-over check at the end of `parse`.
     - `OpVectorTimesScalar`/`OpMatrixTimes*` lower to `Binary::Multiply` with operand types that do not match. The frontend accepts them, but the validator then fails with `InvalidArithmeticControl`, because it requires `left` type == `right` type and scalar/vector only.
     - glslang and DXC emit `NoContraction` for `precise`. So shaders that load today fail after this PR, and there is no option to opt out.
  2. [high, reasoned from code, not run] `naga/src/back/msl/writer.rs:4763`, `naga/src/back/wgsl/writer.rs:145`, `naga/src/back/glsl/writer.rs:157`, `naga/src/back/hlsl/writer.rs:463` Any reachable controlled expression is now a hard error on MSL, WGSL, GLSL below 400 / ESSL below 320 (this includes WebGL2's ESSL 300), and HLSL below SM 6.0. Today these targets translate the shader and drop the decoration. So a wgpu user who passes `ShaderSource::SpirV` with `precise` math gets a shader-creation error on Metal and GL. `naga` CLI spv-to-wgsl translation also breaks. The CHANGELOG entry calls this "an explicit error", but it does not say that working inputs stop working.
  3. [medium] No test covers the paths that the title names. There is no SPIR-V input that has `NoContraction`, no SPIR-V output check for the decoration, and no GLSL output test. All tests (`hlsl_inout.rs`, `function_control_serialization.rs`) insert controls by hand into WGSL-parsed modules. The 40 `.ron` snapshot changes only add `expression_controls: {}`.
  4. [medium] `naga/src/back/hlsl/writer.rs:3045-3140` The HLSL loop hoisting rewrites how `Call`, `Atomic`, `WorkGroupUniformLoad`, subgroup and `RayQuery::Proceed` results are declared in every loop with a `continuing` gate, in functions that have controls. It is a large change to the HLSL writer for a narrow feature. The PR body says that it also fixes atomic-load regressions, but these regressions exist only with controls present.
  5. [low] `naga/tests/naga/hlsl_inout.rs:28` The test helper `save` writes files when the `OUTPUT326`/`CASE326` environment variables are set. This is debug code from the agent harness.
  6. [low] `CHANGELOG.md` The entry has no "By [@5andr0](https://github.com/5andr0) in [#10908](https://github.com/gfx-rs/wgpu/issues/10908)" attribution.
  7. [low] Process: the PR body says "100% Astra" (an agent). The boxes "changes authored by an Agent" and "description written by an Agent" are checked. The box "self-reviewed and fully understand" is also checked. But the author's 2026-11-13 comment on [#9295](https://github.com/gfx-rs/wgpu/issues/9295) says that he will not do the human part.
- **To land:** A maintainer decides if the IR design is acceptable. Make unsupported controls degrade the way trunk does (drop the control, maybe with a warning or an option) instead of failing. Accept `NoContraction` on all ops that the frontend lowers, or drop it on them. Add SPIR-V in/out and GLSL snapshot tests. Remove the debug env code. Find a human owner.
- **Move-along action:** decide if naga wants `expression_controls` in the IR. If not, close the PR and open an issue: "NoContraction is dropped by translation".
- **Depth:** full for `naga/src` and the CHANGELOG. I skimmed the tests. I did not trace every HLSL hoisting case.

#### [#10921](https://github.com/gfx-rs/wgpu/issues/10921): naga: preserve scalar Exp2 producer controls ([@5andr0](https://github.com/5andr0))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-12.
- **Verdict:** Close
- **Waiting on:** maintainer. The PR stacks on [#10908](https://github.com/gfx-rs/wgpu/issues/10908), and its body says that no human claims self-review, understanding or ownership.
- **Last activity:** 2026-11-12, 5andr0, PR body edit. 1 day.
- **Approvals:** none
- **CI:** green (66 pass, 3 skipped); mergeable
- **Summary:** On top of [#10908](https://github.com/gfx-rs/wgpu/issues/10908), adds `ArithmeticControl::Exp2F32NoRefactoringV1`. The SPIR-V frontend maps `FPFastMathMode 0` on a scalar f32 GLSL.std.450 `Exp2` (SPIR-V 1.2+, `FloatControls2`) to this control, and the SPIR-V backend writes it again. HLSL emits a `[noinline]` `precise exp2` helper. GLSL, MSL and WGSL reject the control. The PR also adds `SPV_KHR_float_controls2` and `FloatControls2` to the frontend's supported lists. 909 lines added over [#10908](https://github.com/gfx-rs/wgpu/issues/10908), 633 of them tests.
- **Findings:**
  1. [high] Process: the "self-reviewed and fully understand" box is unchecked. The body says "human self-review, understanding, and ownership are not asserted". The repository does not allow fully agentic PRs.
  2. [medium] `naga/src/ir/mod.rs:2523` The IR variant is very narrow (one op, one type, a "V1" suffix). Its doc comment talks about "transport permissions", a "qualified DXC compiler" and "D3D source provenance". This is the author's porting contract, not a general IR concept.
  3. [low] `naga/src/front/spv/mod.rs` `FPFastMathMode` with nonzero flags gives `UnsupportedExp2Control(0)`, with a placeholder id 0. Any `FloatControls2` module that uses fast-math flags fails. Trunk rejected the extension anyway, so this is not a regression.
  4. [low] It carries all [#10908](https://github.com/gfx-rs/wgpu/issues/10908) findings, including the `OUTPUT326` test debug code.
- **To land:** Not recommended. It needs [#10908](https://github.com/gfx-rs/wgpu/issues/10908) to land first, a human owner, and a maintainer decision that this IR variant is wanted.
- **Move-along action:** close. Point to [#10908](https://github.com/gfx-rs/wgpu/issues/10908) for the shared design decision.
- **Depth:** full for the 447-line `naga/src` delta over [#10908](https://github.com/gfx-rs/wgpu/issues/10908). I did not read `exp2_control.rs` (633 lines of tests).

## Metal

#### [#9645](https://github.com/gfx-rs/wgpu/issues/9645): [metal] synchronize acceleration structure builds using fences ([@jtbirdsell](https://github.com/jtbirdsell))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-08-07.
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
- **State (2026-11-14):** ready; has conflicts; CI red (3 failed); last update 2026-11-09.
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

#### [#9757](https://github.com/gfx-rs/wgpu/issues/9757): Workaround xcode misreporting ray tracing support ([@mate-h](https://github.com/mate-h))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-26.
- **Verdict:** Needs a bit of work
- **Waiting on:** other reviewer (inner-daemons). andyleiserson rewrote most of the PR and asked for someone else to review it. inner-daemons asked why the change is debug-only, and Andy answered on 2026-10-26. There is no response after that.
- **CI:** green
- **Summary:** Ray tracing support is now detected from `MTLGPUFamily::Apple6` (Apple's feature table), with `supportsRaytracing` and `supportsRaytracingFromRender` as a runtime fallback. The objc2 `responds_to` workaround is now active only under `cfg(debug_assertions)`, because the panic comes from objc2's debug-only method check, not from Metal. A one-time warning is logged when the device is `CaptureMTLDevice`.
- **Findings:**
  1. [medium] `CHANGELOG.md:203` The entry describes only the debug warning and the workaround removal. It does not describe the user-visible fix (ray tracing is now detected under Xcode capture and on Apple6+). The phrase "remove the feature detection workaround in question entirely" means little to users. Rewrite it.
  2. [low] `wgpu-hal/src/metal/adapter.rs:30-34` The check is keyed on wgpu-hal's `debug_assertions`, but the panic comes from objc2's `debug_assertions`. A per-package profile override (wgpu-hal release, objc2 debug) would bring back the panic under capture. This is rare. Mention it in the doc comment, or consider objc2's feature that disables the check ([#10028](https://github.com/gfx-rs/wgpu/issues/10028)).
  3. [low] `wgpu-hal/src/metal/adapter.rs:118` The `// others...?` placeholder is left in `MTL_DEBUG_DEVICES`. Andy noted that `MTLDebugDevice` and `MTLToolsDevice` exist. Add them or remove the placeholder.
  4. [low] The PR description and the unchecked checklist still describe mate-h's original approach. Update them before merge so the squash message is accurate. The outdated review thread from Andy is still open.
- **To land:** fix the CHANGELOG wording, resolve the `others...?` placeholder, refresh the description, and get a review from inner-daemons or the maintainer.
- **Depth:** full

#### [#9876](https://github.com/gfx-rs/wgpu/issues/9876): Query Subgroup Size on Metal ([@Vollkornaffe](https://github.com/Vollkornaffe))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-30.
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
- **State (2026-11-14):** ready; has conflicts; CI 7 cancelled jobs; last update 2026-11-05.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-01.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-11.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer (andyleiserson). Andy asked on 2026-11-04 whether `msl_version`/`max_msl_version` are needed and whether probing can be opt-in. The author answered the same day. Andy has not replied.
- **Last activity:** 2026-11-11, osimon9, test report. 2 days.
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
- **Move-along action:** ping andyleiserson to decide on opt-in probing and the version knobs. Point to osimon9's timing data.
- **Depth:** status only (conversation changed; code not re-reviewed)

#### [#10731](https://github.com/gfx-rs/wgpu/issues/10731): fix(metal): Work around legacy Intel compute compiler failures ([@ITJamie](https://github.com/ITJamie))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-28.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-28.
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
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI red (1 failed); last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (inner-daemons is assigned and requested; no review yet).
- **Last activity:** 2026-11-09, matthargett, pushed `b0d0dcbb3`. 4 days.
- **Approvals:** none
- **CI:** red, infra. "Test Mac aarch64" failed in 8 s because `actions/checkout` could not load its `action.yml`. All other jobs pass. Mergeable.
- **Summary:** Adds `watchos` keys to the Metal adapter `available!` checks. watchOS 26/11/10 select MSL 4.0/3.2/3.1, and older releases keep MSL 1.0. The GPU family, argument buffer, read-write texture, counter sampling and fragment read-write checks now run on watchOS 10+. New: indirect draw/dispatch (`INDIRECT_EXECUTION`) and base vertex/first instance are enabled on Apple5+ watch GPUs on watchOS 11+. The render ICB probe floor goes from watchOS 6 to 11, and the probe is skipped on 32-bit watchOS. CHANGELOG entry added.
- **Findings:**
  1. [medium] `wgpu-hal/src/metal/adapter.rs:898`, `:931`, `:939` The new `watchos_indirect_drawing` sets `indirect_draw_dispatch`, which enables both indirect draws and indirect dispatch. The evidence in the description covers indirect draws on two watches (S8 on watchOS 11.6, Series 10 on watchOS 27). It does not mention indirect dispatch or base vertex/first instance. Ask for that evidence, or confirm it from Apple's family tables.
  2. [low] `wgpu-hal/src/metal/adapter.rs:863` The 32-bit watchOS skip is now described ("the probe aborts instead of failing"), but there is still no log or evidence. It covers all `arm64_32` watches (S4 to S8), including the S8 used as the indirect-drawing floor.
  3. [low] No CI builds or runs watchOS. The PR adds no test. The old host-only table test is gone because the table is gone.
  4. [low] The description has no "LLM Use?" section and no checklist.
  5. [low, interaction] [#10927](https://github.com/gfx-rs/wgpu/issues/10927) edits the same `indirect_command_buffers_rendering` expression (`!is_virtual && ...`). Whichever lands second has a small textual conflict.
- **To land:** get evidence for indirect dispatch and base vertex/first instance on Apple5 watchOS 11, or limit the change to draws. Rebase after [#10927](https://github.com/gfx-rs/wgpu/issues/10927) if it lands first. Rerun the Mac test job.
- **Move-along action:** review (about 70 lines). Ask about the evidence for indirect dispatch.
- **Depth:** full

#### [#10875](https://github.com/gfx-rs/wgpu/issues/10875): metal: Lose the device when a command buffer fails ([@lexoliu](https://github.com/lexoliu))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
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
- **State (2026-11-14):** ready; mergeable; CI red (4 failed); last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer decision on count-draw validation (finding 1), then author. inner-daemons is assigned and requested. There is no review yet.
- **Last activity:** 2026-11-09, matthargett, opened the PR (split from [#9679](https://github.com/gfx-rs/wgpu/issues/9679)). 4 days.
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
- **Move-along action:** decide whether Deno may expose count draws before wgpu-core validates their arguments.
- **Depth:** full

## Vulkan

#### [#10546](https://github.com/gfx-rs/wgpu/issues/10546): Enable `VK_COLOR_SPACE_PASS_THROUGH_EXT` such that application can declare native surface's color ([@yikerman](https://github.com/yikerman))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-03.
- **Verdict:** Land as is
- **Waiting on:** maintainer (jimblandy is assigned; the author pinged them twice)
- **Last activity:** 2026-10-31, yikerman, "Can this be merged? [@jimblandy](https://github.com/jimblandy)" (also merged trunk). 6 days.
- **Approvals:** inner-daemons (twice; the last approval is on 59cabd9. After it came one doc-only commit and a trunk merge)
- **CI:** green
- **Summary:** Adds `SurfaceColorSpace::PassThrough` / `SurfaceColorSpaces::PASS_THROUGH`. On Vulkan it maps to `VK_COLOR_SPACE_PASS_THROUGH_EXT`. On Metal it sets `CAMetalLayer.colorspace = nil`. `Auto` never resolves to it. Other backends never report it.
- **Findings:**
  1. [low] `wgpu-hal/src/metal/adapter.rs:455` Metal reports `PASS_THROUGH` for every format. The author has no Apple device, so the Metal path is untested. The reviewer requested this path. The risk is small: it only sets a nil colorspace and turns EDR off.
  2. [low] `wgpu/src/documentation/color/hdr_surfaces.rs:115` has a grammar typo: "The named color spaces here uses".
  3. [low] The only open thread is a praise comment ("Like this test"). Nothing in it blocks the merge.
- **To land:** nothing.
- **Move-along action:** merge.
- **Depth:** full

#### [#10629](https://github.com/gfx-rs/wgpu/issues/10629): Vulkan multi-planar DMA-buf import support ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-07.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (description still says "Testing TBD"; CHANGELOG has `#TODO`), then reviewer inner-daemons (review requested; jimblandy assigned)
- **Last activity:** 2026-11-07, andyleiserson, force push (rebase). 6 days.
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

#### [#10677](https://github.com/gfx-rs/wgpu/issues/10677): vulkan: Give every semaphore a value when chaining timeline submit info ([@laurenz-canva](https://github.com/laurenz-canva))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-04.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (inner-daemons/jimblandy, to decide the scope), then the author, to rebase
- **Last activity:** 2026-11-04, inner-daemons, replied that the change stays a documented workaround and can be turned on for more drivers. 2 days.
- **Approvals:** none (inner-daemons left a COMMENTED review asking for a workaround flag; the author added it on 2026-10-23)
- **CI:** green (on a stale base)
- **Summary:** Adds `Workarounds::PAD_TIMELINE_SEMAPHORE_VALUES` for `IMAGINATION_PROPRIETARY`. When it is set and a submit has timeline values, every wait and signal semaphore gets a value. This works around the PowerVR driver, which reads `pWaitSemaphoreValues[i]` without checking the count. It fixes a crash on Pixel 10 and Teclast BXM, and probably [#7669](https://github.com/gfx-rs/wgpu/issues/7669).
- **Findings:**
  1. [medium] The PR has merge conflicts with trunk (`mergeable_state: dirty`). It must be rebased. The workaround flag value `0x8` also collides with [#10692](https://github.com/gfx-rs/wgpu/issues/10692) and [#10840](https://github.com/gfx-rs/wgpu/issues/10840), which both use `1 << 3`.
  2. [medium] Scope is open. Two users (MarcelStruckWO, jonatino) report that Android 16 emulators (gfxstream/"ranchu") need the same padding. jonatino wants the padding on for all drivers. A maintainer must decide: keep it gated to the IMG driver, add gfxstream, or turn it on for all drivers as a documented workaround. The padding is spec-legal and costs very little.
  3. [low] There are no tests. leekichko offered 4 unit tests for `SemaphoreList::add_to_submit` (commit bce0159 on leekichko/wgpu). The author can cherry-pick them.
  4. [low] The CHANGELOG entry names only the Pixel 10. It could also name [#7669](https://github.com/gfx-rs/wgpu/issues/7669) and the BXM family.
- **To land:** rebase onto trunk, decide the driver scope, and optionally cherry-pick leekichko's tests.
- **Move-along action:** decide whether the padding is gated to IMG only, IMG plus gfxstream, or all drivers, then tell the author to rebase.
- **Depth:** full

#### [#10750](https://github.com/gfx-rs/wgpu/issues/10750): Add opt-in static HDR metadata submission for Vulkan surfaces ([@ecke101](https://github.com/ecke101))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-09.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-13.
- **Verdict:** Land as is
- **Waiting on:** inner-daemons (assigned) to re-approve after the last push. The author re-requested review on 11-13.
- **Last activity:** 2026-11-13, kikijiki, force push with the new doc comment and re-requested review. 0 days.
- **Approvals:** inner-daemons approved on 11-12, before the last push. GitHub now shows "review required", so the approval does not count for the current head.
- **CI:** green; mergeable
- **Summary:** Render and compute passes write `beginning_of_pass_write_index` at `TOP_OF_PIPE` through a new private `unsafe fn write_pass_begin_timestamp`, instead of at `BOTTOM_OF_PIPE`. The end-of-pass stamp and `write_timestamp` do not change. This fixes near-zero pass durations on RADV when the GPU is busy.
- **Findings:**
  1. [low] `wgpu-hal/src/vulkan/command.rs:42-54` The begin stamp can now latch before earlier passes finish, so a pass duration can include queue wait. This is the usual profiler trade-off and is better than zero durations. The new doc comment explains why the helper exists.
  2. [low] No test (the bug depends on timing). Acceptable.
- **To land:** inner-daemons re-approves the last push.
- **Move-along action:** merge after inner-daemons re-approves (the last push only changed a doc comment).
- **Depth:** full (2-file diff)

#### [#10840](https://github.com/gfx-rs/wgpu/issues/10840): Vulkan: Don't use timeline semaphores on Arm's proprietary Mali driver ([@tadeohepperle](https://github.com/tadeohepperle))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-05.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviewer or assignee yet)
- **Last activity:** 2026-11-05, tadeohepperle, opened the PR. 1 day.
- **Approvals:** none
- **CI:** green
- **Summary:** Adds `Workarounds::DISABLE_TIMELINE_SEMAPHORES` for `ARM_PROPRIETARY`, which makes wgpu fall back to the binary `FencePool`. Without it, the Mali driver uses 16 B of kernel memory per timeline signal and loses the device after about 4.2M submissions. The PR also adds a 297-line `examples/bug-repro/03_arm_timeline_semaphore_leak` crate.
- **Findings:**
  1. [medium] `CHANGELOG.md` has a placeholder link: `[#XXXXX](https://github.com/gfx-rs/wgpu/pull/XXXXX)`.
  2. [medium] `wgpu-hal/src/vulkan/adapter.rs:2372` The flag is on for every Arm proprietary driver version, but only `g18p0` was tested. The Arm proprietary driver runs most Mali Android devices, so the change affects a large population. The author's own data shows an alternative: destroy and recreate the timeline semaphore periodically. That keeps timeline semantics but adds complexity. A maintainer must choose.
  3. [low] The repro crate is a workspace and default member, so CI builds it and it adds to `Cargo.lock`. This follows the existing `bug-repro/01`, `02` pattern. It is the maintainer's choice whether to keep it.
  4. [low] Bit `1 << 3` collides with [#10692](https://github.com/gfx-rs/wgpu/issues/10692).
- **To land:** fix the CHANGELOG PR number, decide between disabling timeline semaphores on all Arm proprietary versions and the recreate approach, and keep or drop the repro crate.
- **Move-along action:** review, and decide whether disabling timeline semaphores on every Arm proprietary driver is acceptable.
- **Depth:** full for the hal change; skimmed the repro crate.

#### [#10880](https://github.com/gfx-rs/wgpu/issues/10880): [hal vulkan] Trim command pools to what the last submission used ([@purian23](https://github.com/purian23))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-10.
- **Verdict:** Needs a bit of work
- **Waiting on:** reviewer inner-daemons (assigned) and matthargett (asked to benchmark). The design question from inner-daemons has no answer yet.
- **Last activity:** 2026-11-10, purian23, force push. 3 days.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** In `reset_all`, the Vulkan command encoder now frees the command buffers that the last encoding did not use, and then calls `vkTrimCommandPool` (core 1.1, or `VK_KHR_maintenance1` on 1.0). `ALLOCATION_GRANULARITY` goes from 16 to 1. The author saw 10 to 13 MB of retained command buffers on Mesa anv, because wgpu-core pools hal encoders for the life of the device.
- **Findings:**
  1. [medium] `wgpu-hal/src/vulkan/command.rs:201-210` The heuristic keeps only the count that the last encoding used. A pooled hal encoder that alternates between a large and a small wgpu `CommandEncoder` now frees, trims and reallocates command buffers on each cycle. With granularity 1, each extra buffer is one `vkAllocateCommandBuffers` call (`command.rs:122-132`). inner-daemons raised the same risk on 11-09 and asked for benchmarks. The PR has no performance numbers beyond "CPU use stayed the same" on one Intel machine. Reasoned from code, not measured.
  2. [low] `wgpu-hal/src/vulkan/command.rs:205` `split_off(len - unused)` frees the buffers at the end of `free`, which are the buffers this encoding used, not the unused ones at the front. All buffers were reset by `reset_command_pool` just before, so the count is correct and the behavior is correct. The comment "Keep only what this encoding used" can mislead a reader.
  3. [low] `wgpu-hal/src/vulkan/adapter.rs:2713` The `maintenance1` function table loads when `device_api_version < 1.1`. This matches the condition that adds `VK_KHR_maintenance1` to the required extensions (`adapter.rs:1240`) and the adapter filter (`adapter.rs:2402`). Correct.
  4. [low] No test. A test that counts retained command buffers is possible through hal, but not required for a memory heuristic.
  5. [low] PR template: "I self-reviewed and fully understand this PR" is not checked. The "LLM Use?" section says Claude helped to find the cause and write the patch, and the author reviewed and tested it. There are no agent-disclosure boxes.
- **To land:** answer the alternating-submission concern with numbers (or a hysteresis, for example trim only when the free count stays high for several resets); author checks the self-review box; fix the comment wording.
- **Move-along action:** ping matthargett for the benchmark that inner-daemons asked for.
- **Depth:** full

#### [#10926](https://github.com/gfx-rs/wgpu/issues/10926): vulkan: filter buffer memory types by available budget ([@5andr0](https://github.com/5andr0))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI pending; last update 2026-11-12.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviewer, no assignee). Policy decision on an agent-written description with an unchecked self-review box.
- **Last activity:** 2026-11-12, 5andr0, marked ready for review. 1 day.
- **Approvals:** none
- **CI:** green (codecov/patch pending); mergeable
- **Summary:** For buffers only, the Vulkan hal now removes from `memory_type_bits` every memory type whose heap would exceed `MemoryBudgetThresholds::for_resource_creation`, and passes that mask to gpu-allocator. Before, the check looked only at the heap that gpu-allocator would pick and failed if that heap was over budget. The goal: a small host-visible device-local heap that is full no longer blocks staging buffers, because gpu-allocator can fall back to system memory.
- **Findings:**
  1. [medium] Description and code do not match. The Testing section says "Fifteen CPU cases execute the exact helper against value-only stubs". The PR contains no tests (only `CHANGELOG.md` and `wgpu-hal/src/vulkan/device.rs` change). The fallback path has no test at all.
  2. [medium] Policy: the "description was automatically written by an Agent" box is checked, "I self-reviewed and fully understand this PR" is not checked, and the description says "the submitter's review and ownership attestation remain unchecked". The repository does not allow PRs that the contributor cannot vouch for. Same author as [#10922](https://github.com/gfx-rs/wgpu/issues/10922).
  3. [low] `wgpu-hal/src/vulkan/device.rs:1046-1052` The new helper copies gpu-allocator's required-flags table, as the old `error_if_would_oom_on_resource_allocation` copies its preferred-flags table. Both break silently if gpu-allocator changes. The old function has a doc comment that states this risk. The new function has no doc comment.
  4. [low] Behavior change for `GpuOnly`: if the main VRAM heap is over budget, the mask can leave only the small host-visible device-local (BAR) type, and the buffer goes there. This is probably acceptable, but the description does not mention it. Textures and acceleration structures still use the old reject-only check (`device.rs:1275`, `:2712`, `:3058`), which the description states.
  5. [low] `CHANGELOG.md:338` The entry has no "By [@5andr0](https://github.com/5andr0) in [#10926](https://github.com/gfx-rs/wgpu/issues/10926)" attribution and has an extra blank line after it.
- **To land:** author confirms review and ownership (checks the box, or a maintainer accepts the PR as is); add the tests that the description describes, or remove the claim; add a doc comment; fix the CHANGELOG entry.
- **Move-along action:** decide whether to review a PR with an agent-written description and no self-review attestation; if yes, assign a Vulkan reviewer.
- **Depth:** full

## GLES, DX12, present and platform

#### [#8388](https://github.com/gfx-rs/wgpu/issues/8388): Implement DXGI Swapchain on Vulkan ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-13.
- **Verdict:** Status note
- **Waiting on:** andyleiserson (re-review). cwfitzgerald answered the review threads on 11-13; the fixes are in fixup commits that will be squashed after approval.
- **Last activity:** 2026-11-13, cwfitzgerald, push of the head. 0 days.
- **Approvals:** none
- **CI:** green (66 pass, 3 skipped) on the new head; mergeable (the old conflict is gone).
- **Summary:** Adds a DXGI swapchain path to the Vulkan backend on Windows. The PR is ready for review and has review state CHANGES_REQUESTED.
- **Findings:** none (status only). Open review threads: 27 of 46 are unresolved (same count as the old report). The author has not replied to them in the threads. Some of them may be fixed by the 30 new commits. I did not check this.
- **To land:** check the 27 threads against the new commits; resolve or answer each; re-request review from andyleiserson; get an approval.
- **Move-along action:** Reply to andyleiserson's threads and re-request his review, since the 30 new commits probably answer many of them.
- **Depth:** status only

#### [#10152](https://github.com/gfx-rs/wgpu/issues/10152): Add damage-aware presentation via Queue::present_with_damage ([@sauliusvl](https://github.com/sauliusvl))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-19.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer. The author addressed all of cwfitzgerald's CHANGES_REQUESTED threads (all resolved) on 2026-09-28 and asked for re-review twice. The author also asked an open question: should the hal `Rect<T>` become a re-export of `wgt::Rect`? The PR now has merge conflicts (`dirty`).
- **CI:** green (on the old base)
- **Summary:** Adds `wgpu::Queue::present_with_damage(SurfaceTexture, &[Rect<u32>])`. It moves `Rect` from `wgpu-core` to `wgpu-types`, and `wgpu-core` clips the rects to the surface. It adds a `damage_rects` parameter to `hal::Queue::present` (a breaking hal change). Vulkan uses `VK_KHR_incremental_present` and EGL uses `eglSwapBuffersWithDamage{KHR,EXT}`. DX12, Metal, WGL and WebGPU ignore the rects.
- **API / cross-backend notes:** The design is acceptable as a hint-only API: empty means full present, out-of-range rects are clipped, and unsupported backends do a full present. The docs correctly require the texture to hold the whole frame. The safe boundary is good: `wgpu-core` clips and drops empty rects (`wgpu-core/src/present.rs:374-384`), so hal gets only in-bounds, non-empty rects, which the hal safety docs require. `Rect::intersect` uses saturating math and has overflow unit tests. The `dispatch` default method keeps custom backends compiling.
- **Findings:**
  1. [medium] `wgpu-hal/src/lib.rs:2829` and `wgpu-types/src/origin_extent.rs:88` There are now two identical `Rect<T>` types, in hal and in wgt. The author asked whether to merge them, and you suggested merging in an earlier review. Decide, and probably make hal re-export `wgt::Rect`. Also, `wgpu::Rect` is a new, generic name in the `wgpu` root namespace, and `intersect` exists only on `Rect<u32>`. Confirm that you want this public name.
  2. [medium] `wgpu-hal/src/gles/egl.rs:1228-1263` The EGL path is untested on hardware (the description says so). The y-flip uses `sc.extent.height - rect.y - rect.h` in `u32`. This is safe only if `sc.extent` equals the core config extent that the rects were clipped to. That is true today, but the subtraction depends on it. Use `saturating_sub` or add a debug assert. Ask for one Mesa/EGL test run (for example, Wayland GLES) before landing.
  3. [low] `wgpu-hal/src/vulkan/swapchain/native.rs:655-673` `VkPresentRegionsKHR` is pushed before the user chain from `set_next_present_chain` ([#9847](https://github.com/gfx-rs/wgpu/issues/9847)) is spliced in. If that chain also contains `VkPresentRegionsKHR`, the pNext chain has a duplicate struct, which violates the Vulkan spec. That API is unsafe, so this is the caller's fault, but add a line to its safety docs. `VK_KHR_incremental_present` is now enabled on every device that supports it, even if no damage is passed. The risk is low.
  4. [low] Tests cover only `Rect::intersect`. There is no test that calls `present_with_damage` through `wgpu`. Presentation is hard to test in CI, but a test with the noop backend or an offscreen surface (if the test infra has one) would at least cover the core clipping path, including rects that are fully off-surface or empty. The trace `Action::Present` does not record the rects, so replays use a full present. This is acceptable for a hint, but say so.
  5. [low] `CHANGELOG.md` adds a new `#### Hal` subsection under New Features. Check that this follows the CHANGELOG conventions. The project normally does not log hal-only changes, but this is a breaking trait change, so an entry is reasonable.
- **To land:** rebase; decide on merging the hal `Rect` and on the public `wgpu::Rect` name; harden the EGL y-flip subtraction; get one real EGL test run (or mark EGL as best-effort); document the incremental-present conflict on `set_next_present_chain`.
- **Depth:** targeted (I read core, the public API, and the Vulkan and EGL hal paths in full. I skimmed the mechanical signature changes in dx12, metal, noop, wgl, web and the examples. I did not check the EGL proc signature against Khronos headers.)

#### [#10505](https://github.com/gfx-rs/wgpu/issues/10505): gles/wgl: pump hidden-window messages while awaiting shutdown ([@acarl005](https://github.com/acarl005))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-07.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (cwfitzgerald, assigned). No review is posted, so the author does not know about the findings.
- **Last activity:** 2026-11-07, acarl005, merged trunk into the branch. 6 days.
- **Approvals:** none
- **CI:** green on the merge head; mergeable
- **Summary:** The WGL hidden-window thread runs a `GetMessageA` loop instead of blocking on a channel. Shutdown is a posted `WM_APP` message from `Drop for InstanceDevice`. This fixes a hang when the input language changes ([#10504](https://github.com/gfx-rs/wgpu/issues/10504)).
- **Findings:**
  1. [medium] `wgpu-hal/src/gles/wgl.rs:444-460` The loop removes messages but does not call `TranslateMessage`/`DispatchMessageA`. Messages posted to the hidden window are discarded and never reach the wndproc. Use the standard loop and dispatch everything except the shutdown message.
  2. [medium] `wgpu-hal/src/gles/wgl.rs:349-361` If `PostMessageA` fails, the error is only logged, and the thread and window leak for the life of the process.
  3. [low] `wgpu-hal/src/gles/wgl.rs:351` `WindowFromDC(self.dc)` finds the window. Storing the `HWND` is simpler.
  4. [low] `CHANGELOG.md:325` No "By [@acarl005](https://github.com/acarl005) in [#10505](https://github.com/gfx-rs/wgpu/issues/10505)" attribution.
  5. [low] No test; not practical. Acceptable.
- **To land:** dispatch messages in the loop; handle or document a `PostMessageA` failure; fix the CHANGELOG attribution.
- **Move-along action:** post a review with findings 1, 2 and 4.
- **Depth:** status only (verified that the PR diff is unchanged and the old findings still apply)

#### [#10511](https://github.com/gfx-rs/wgpu/issues/10511): fix(dx12): fix 1 second hang for dropped `SurfaceTexture`s ([@DouglasDwyer](https://github.com/DouglasDwyer))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-07.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviews yet)
- **CI:** green
- **Summary:** Adds `pending_present_count` to the DX12 swapchain, so that `acquire_texture` waits on the frame-latency waitable object only when a matching present exists. This stops the 1 s timeout when acquired textures are dropped without a present (common during resize).
- **Findings:**
  1. [medium] `wgpu-hal/src/dx12/mod.rs:1408` Each `configure` resets the counter to `maximum_frame_latency`, also on the `ResizeBuffers` path. That path keeps the old swapchain, so it probably keeps the same waitable semaphore. Example: acquire-and-drop until the counter is 0, so the semaphore is also drained. Then resize. The counter is back at max but the semaphore is still 0, so the next acquire can stall for 1 s again. This is the resize case the PR targets. Ask the author to confirm that `ResizeBuffers`/`SetMaximumFrameLatency` resets the semaphore count. If it does not, keep the counter across a resize (clamped to the new max) and reset it only on the new-swapchain path. The author's repro passed, so the reset may be real. It is not documented.
  2. [low] `wgpu-hal/src/dx12/mod.rs:1645` If no present is pending, the code skips the wait with no throttling. That is the intent, but a short comment on the invariant (waits <= presents + max latency) would help future readers. The field doc comment partly covers this.
  3. [low] No test. A GPU test is hard here (it needs a real window). The manual repro is linked. Acceptable.
  4. [info] Overlap with [#10548](https://github.com/gfx-rs/wgpu/issues/10548): both PRs edit the `SwapChain` struct and the tail of `Queue::present` in `wgpu-hal/src/dx12/mod.rs`. The textual conflict is trivial. See [#10548](https://github.com/gfx-rs/wgpu/issues/10548) for the semantic interaction.
- **To land:** answer or fix finding 1.
- **Depth:** full

**Deeper review ([#10511](https://github.com/gfx-rs/wgpu/issues/10511): fix(dx12): fix 1 second hang for dropped `SurfaceTexture`s):**

> - **Open questions from the first pass:**
>   1. Does the frame latency waitable object keep its semaphore count across `IDXGISwapChain::ResizeBuffers`? The PR resets its counter to `maximum_frame_latency` on every `configure`.
> - **Answers:**
>   1. [unknown: the docs do not say; the evidence points to "count is kept"] Microsoft docs for `GetFrameLatencyWaitableObject`, `IDXGISwapChain2::SetMaximumFrameLatency`, `ResizeBuffers` and "Reduce latency with DXGI 1.3 swap chains" do not describe the object's count, and do not say that `ResizeBuffers` resets it. They do not even call it a semaphore. Secondary evidence:
>      - vkd3d-proton (`libs/vkd3d/swapchain.c`) models the object as a semaphore that starts at count 1. `SetFrameLatency` releases only the increase (`MaxLatency - old`), with the comment "Only increasing the latency is handled here; apparently it is the application's responsibility to reduce the semaphore value". Its `ChangeProperties` (the `ResizeBuffers` path) drains the present queue and does not touch the semaphore. Every completed present releases 1.
>      - In wgpu, the `ResizeBuffers` path keeps the same `IDXGISwapChain3` (`wgpu-hal/src/dx12/mod.rs:1411-1427` at the PR head). It closes our handle in `release_resources` (`:1322`) and gets a new handle with `GetFrameLatencyWaitableObject` (`:1591`). The new handle most likely points at the same kernel object. `SetMaximumFrameLatency` is called with the same value (`:1584`), so under the vkd3d model nothing is released.
>
>      If the count is kept, the stall comes back. Example with `maximum_frame_latency = 2`: acquire+drop (counter 1, semaphore 1), resize (counter reset to 2, semaphore 1), acquire+drop (counter 1, semaphore 0), resize (counter 2, semaphore 0), acquire: the counter allows a wait, the semaphore is 0, and no present is in flight, so the wait times out after 1 s. This is the resize-with-dropped-frames case the PR targets.
>
>      The direction of the error matters. If the counter is too high, it stalls for 1 s. If it is too low, one wait is skipped and one frame is not throttled. So keeping the counter across `ResizeBuffers` is safe whichever way Windows behaves. Resetting it is safe only if Windows resets the semaphore.
>
>      How the author can test this on Windows: set `Dx12UseFrameLatencyWaitableObject::Wait` and `maximum_frame_latency = 2`. Acquire and drop two textures, so the semaphore is drained. Call `configure` with a new size, which takes the `ResizeBuffers` path. Then get `Surface::waitable_handle()` (`wgpu-hal/src/dx12/mod.rs:547`, through `as_hal`) and call `WaitForSingleObject(h, 0)`. `WAIT_TIMEOUT` means the count is kept. `WAIT_OBJECT_0` means it was reset. As an end-to-end check, run the acquire+drop / resize / acquire+drop / resize / acquire sequence and time the last acquire.
> - **New findings:**
>   1. [low] `wgpu-hal/src/dx12/mod.rs:1645-1648` `SwapChain::wait` returns `Ok(false)` on timeout. The new code still decrements the counter, but the semaphore was not consumed, so the counter is now 1 too low. This is the safe direction (one wait is skipped later, with no stall). The bool result was ignored before this PR too. Decrement only on `Ok(true)`, or leave it and say so in a comment.
> - **Verdict change:** none (Needs a bit of work). Finding 1 from the first pass now has a concrete fix that does not depend on the unknown.
> - **Move-along action change:** Ask the author to keep `pending_present_count` across the `ResizeBuffers` path, and to reset it only when a new swapchain is created (see the combined change under [#10548](https://github.com/gfx-rs/wgpu/issues/10548)). Optional: ask them to run the probe above and record the result in a comment.

#### [#10515](https://github.com/gfx-rs/wgpu/issues/10515): Fix support for Emscripten (v6.0.10+) by making it use `web-sys` ([@almarklein](https://github.com/almarklein))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI red (4 failed); last update 2026-10-27.
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

#### [#10522](https://github.com/gfx-rs/wgpu/issues/10522): hal: Add GLES DMA-BUF import ([@terjdev](https://github.com/terjdev))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-13.
- **Verdict:** Needs a bit of work
- **Waiting on:** author. andyleiserson requested changes on 11-13 with minor notes: report GL errors in the import path, also require `GL_OES_EGL_image`, move `dmabuf.rs` into `gles/egl.rs`, add the CHANGELOG attribution.
- **Last activity:** 2026-11-04, andyleiserson, said "started reviewing ... should be able to finish soon". 2 days.
- **Approvals:** none (inner-daemons and HigherOrderLogic left COMMENTED reviews; both say they are not maintainers)
- **CI:** green (2026-10-24)
- **Summary:** Adds `gles::Device::texture_from_dmabuf_fd` (single-plane DMA-BUF to EGLImage to `GL_TEXTURE_2D`, zero copy) and `gles::ImportTextureError`. The EGL extension caps are cached in `PrivateCapabilities`. Scope was reduced to only this after maintainer discussion (2026-10-21 meeting).
- **Findings:**
  1. [medium] No tests. The description still says it has tests and mentions a `TextureBacking` field, but both were removed when the scope was reduced. Andy asked (2026-10-15) for tests that export a DMA-BUF from Vulkan, not from `libgbm`. Update the description, and either add the test or agree to defer it.
  2. [low] `wgpu-hal/src/gles/device.rs` `glEGLImageTargetTexture2DOES` is resolved only through `eglGetProcAddress`, which can return non-null for unsupported functions. `GL_OES_EGL_image` is not checked in the GL extension string. Add it to the adapter-time cap check.
  3. [low] `CHANGELOG.md` The entry has no "By [@terjdev](https://github.com/terjdev) in [#10522](...)" attribution, which is the repo convention.
  4. [low] Only `GL_TEXTURE_2D` is used. Some drivers import YUV and tiled formats only as `GL_TEXTURE_EXTERNAL_OES`. This is acceptable for the single-plane RGB scope, but the doc comment could say so.
- **To land:** Andy's review. Update the description. Add the attribution. Optionally add the OES_EGL_image check and the test (or an issue for the test).
- **Move-along action:** ping andyleiserson (they said they will finish soon; nothing else blocks).
- **Depth:** full

#### [#10548](https://github.com/gfx-rs/wgpu/issues/10548): fix(dx12): use DXGI_PRESENT_RESTART for new swapchains ([@DouglasDwyer](https://github.com/DouglasDwyer))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-14.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviews yet). The description says "Depends on [#10511](https://github.com/gfx-rs/wgpu/issues/10511)", but the diff does not contain [#10511](https://github.com/gfx-rs/wgpu/issues/10511)'s changes. They are two independent branches against trunk.
- **CI:** green
- **Summary:** The first `Present` after every `configure` uses `Present(1, DXGI_PRESENT_RESTART)`. This drops stale old-size frames, so DWM does not stretch them during a live resize. It is part of a smooth-resize approach that also needs a winit change.
- **Findings:**
  1. [medium] `wgpu-hal/src/dx12/mod.rs:1745-1765` The restart branch hard-codes sync interval 1 and ignores the computed `interval`/`flags`. For `Immediate`/`Mailbox`, the first frame after each configure now waits for vsync, and `DXGI_PRESENT_ALLOW_TEARING` is dropped. Since resize calls configure every frame, this changes Immediate-mode behavior for the whole resize. Use `Present(interval, flags | DXGI_PRESENT_RESTART)`. The author says one present at interval 1 worked for them. They did not say why the configured interval must be overridden.
  2. [medium] Interaction with [#10511](https://github.com/gfx-rs/wgpu/issues/10511): after both land, the restart branch must also do `pending_present_count += 1`. Otherwise every resize loses one count, and acquire skips one wait. When you merge the second PR, put the increment after the `if/else`. It is also not verified whether presents that RESTART discards still signal the frame-latency waitable object. If they do not, the [#10511](https://github.com/gfx-rs/wgpu/issues/10511) counter is off by the number of discarded frames. Ask the author to check this, because they depend on both PRs together.
  3. [low] `wgpu-hal/src/dx12/mod.rs:1745` The `if/else` duplicates the `Present` call and `Ok(())`. It can be one call with computed `(interval, flags)`. This also fixes finding 1.
  4. [low] RESTART is applied to every swapchain kind and present mode on every configure, and also on the first present of a new swapchain (where nothing is queued). That is harmless but not necessary. The behavior is not gated, and the author tested only on their own Windows 10/11 machines.
- **To land:** pass the configured interval and flags with `DXGI_PRESENT_RESTART`; land in order after [#10511](https://github.com/gfx-rs/wgpu/issues/10511) and fix the counter increment in the restart path; get an answer on discarded presents and the waitable object.
- **Depth:** full

**Deeper review ([#10548](https://github.com/gfx-rs/wgpu/issues/10548): fix(dx12): use DXGI_PRESENT_RESTART for new swapchains):**

> - **Open questions from the first pass:**
>   1. Do presents that `DXGI_PRESENT_RESTART` discards still signal the frame latency waitable object?
>   2. What exact combined change do [#10511](https://github.com/gfx-rs/wgpu/issues/10511) and [#10548](https://github.com/gfx-rs/wgpu/issues/10548) need?
> - **Answers:**
>   1. [unknown] The `DXGI_PRESENT` doc says only "the runtime will discard outstanding queued presents". It is "valid only for flip-model swap chains and full screen", and it is meant for "the previously queued presentation might have assumed that the window is an old size". Nothing says whether a discarded present releases the latency object. wgpu has no code comment on this. vkd3d-proton ignores `RESTART` (every present releases 1), so it gives no evidence. My reasoning, not verified: if discarded presents did not signal, any waitable-object app that uses `RESTART` would lose latency slots permanently and then hang, so DXGI probably retires them through the same path. Treat this as unproven.
>
>      Also unknown: whether anything is still queued when the `RESTART` present runs. `configure` calls `wait_for_present_queue_idle` and `ResizeBuffers` first, and `ResizeBuffers` may already flush the flip queue. The author's video shows a visible improvement, so something is probably discarded. The visible change could also come from the forced sync interval 1.
>
>      How the author can test this: use `maximum_frame_latency = 3` and Fifo. Do not wait on the object for this probe. Present 3 frames back to back, then present once with `RESTART`, then sleep about 200 ms so that everything retires. Count how many `WaitForSingleObject(h, 0)` calls succeed before the first `WAIT_TIMEOUT`. Use `IDXGISwapChain::GetFrameStatistics` / `GetLastPresentCount` (or PresentMon) to find how many presents were discarded. If the success count is lower by the number discarded, discarded presents do not signal.
>
>      If they do not signal, the [#10511](https://github.com/gfx-rs/wgpu/issues/10511) counter is too high by the number discarded after each resize, which is the stall direction. Then the counter cannot stay correct without knowing how many presents were discarded, and the design needs another answer (for example, do not use `RESTART` when `Wait` is selected).
>   2. Combined change (on top of both PRs, `wgpu-hal/src/dx12/mod.rs`):
>
>      In `Surface::configure`:
>      ```rust
>      // A new waitable object can be waited on `maximum_frame_latency` times before it stalls.
>      let mut pending_present_count = config.maximum_frame_latency as usize;
>      let swap_chain = match self.swap_chain.write().take() {
>          Some(sc) => {
>              // `ResizeBuffers` keeps the waitable object and its count.
>              pending_present_count = sc.pending_present_count;
>              let raw = unsafe { sc.release_resources() };
>              // ... ResizeBuffers as before ...
>      ```
>      The struct literal keeps `pending_present_count` and `restart_pending: true`. If the app raises `maximum_frame_latency` on a resize, the counter is too low (safe). Matching the increase exactly would need the old latency stored in `SwapChain`. That is extra complexity and is not needed.
>
>      In `Queue::present`, replace the `if/else` from [#10548](https://github.com/gfx-rs/wgpu/issues/10548) with one call:
>      ```rust
>      let (interval, mut flags) = match sc.present_mode { /* as before */ };
>      if mem::take(&mut sc.restart_pending) {
>          // Drop frames queued at the old size, so DWM does not stretch them.
>          flags |= Dxgi::DXGI_PRESENT_RESTART;
>      }
>
>      profiling::scope!("IDXGISwapchain3::Present");
>      unsafe { sc.raw.Present(interval, flags) }
>          .ok()
>          .into_device_result("Present")?;
>      sc.pending_present_count += 1;
>
>      Ok(())
>      ```
>      This keeps the configured interval and `ALLOW_TEARING` (interval 0 is required with `ALLOW_TEARING`, so `Present(1, RESTART)` must not be used for Immediate). It counts the `RESTART` present. For Fifo (the default) it is the same call the author tested (`Present(1, RESTART)`). `mem` is already imported in this file (`use core::{ffi, fmt, mem, ops::Deref}`).
> - **New findings:** none.
> - **Verdict change:** none (Needs a bit of work).
> - **Move-along action change:** Ask the author to apply the combined change above when the two PRs are stacked, and to run the discard probe. Land [#10511](https://github.com/gfx-rs/wgpu/issues/10511) first.

#### [#10554](https://github.com/gfx-rs/wgpu/issues/10554): gles/wgl: request an explicit core version from wglCreateContextAttribsARB ([@jimm98y](https://github.com/jimm98y))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI red (1 failed); last update 2026-10-25.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (Wumpf). On 2026-10-25 the author approved Wumpf's suggestion and asked "is there anything else I should do?". Nobody replied. The concrete fixes are mechanical, but the author is not a Rust developer and pushes from their fork's `trunk`, so maintainers cannot push.
- **Last activity:** 2026-10-25, jimm98y, asked what else to do after Wumpf's CHANGES_REQUESTED review. 12 days.
- **Approvals:** none (Wumpf: CHANGES_REQUESTED 2026-10-25)
- **CI:** red. "Format & Typos" fails. This is related: commits `d3493063` and `4a12198e` carry `Co-Authored-By: Claude` trailers, which Wumpf linked as the failure. mergeable_state is dirty (conflicts, probably CHANGELOG).
- **Summary:** The WGL path asks for core contexts 4.6 down to 3.3, and falls back to the version-less request only at the end. It also retries the first `wglCreateContext` once. This fixes the Parallels-on-M4 driver, which returns GL 2.1 and so made wgpu fall back to WARP.
- **Findings:**
  1. [medium] Process: the PR was agent-created and agent-described. The "self-reviewed" box is unchecked, and the author says they cannot judge Rust. ErichDonGubler flagged the LLM policy. Wumpf then checked the approach independently and ended at the same fix, so a maintainer now vouches for the code.
  2. [low] `wgpu-hal/src/gles/wgl.rs` The retry of `wglCreateContext` is a blind second attempt that only Parallels needs. It is cheap and harmless, but there is no evidence beyond one VM.
  3. [low] The explicit 4.6 to 3.3 loop changes context creation on every Windows GL driver. The risk is low, because the drivers that matter return the highest version for any explicit request. There is no hardware test beyond the reporter's VM.
- **To land:** remove the AI co-author trailers (or squash with a clean message), rebase onto trunk, make CI green.
- **Move-along action:** take over. Wumpf already said they would clean it up but cannot push. Re-create the PR from a maintainer branch with one clean commit and credit jimm98y. The other option is to reply to the author with the exact steps (squash, drop the trailers, rebase), but they may not manage them.
- **Depth:** full

#### [#10775](https://github.com/gfx-rs/wgpu/issues/10775): feat(hal/dx12): Extension to restrict enumeration to a specific adapter LUID ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-04.
- **Verdict:** Needs a bit of work (a design decision is open; the code is in good shape)
- **Waiting on:** maintainer. Andy asked for input on 2026-11-03 about the unsafe-to-construct `Extensions<T>` token plus a separate `request_adapters` argument. No reviewer is assigned and no review exists.
- **Last activity:** 2026-11-04, andyleiserson, pushed "fix wasm" after their 2026-11-03 request for design input. 2 days.
- **Approvals:** none
- **CI:** green
- **Summary:** Lands part of the [#10304](https://github.com/gfx-rs/wgpu/issues/10304) extension API. It adds `wgt::BackendMap`, `BackendAdapterOptionsMap`, an `Extensions<T>` token that is unsafe to construct, `hal::Api::AdapterOptions`, an `options` argument on `hal::Instance::enumerate_adapters`, `dx12::Dx12AdapterOptions { luid }`, and a new public `wgpu::Instance::request_adapters(options, extensions)`. It has a DX12 GPU test for select-by-LUID and for unknown-LUID giving `NotFound`. Fixes [#9932](https://github.com/gfx-rs/wgpu/issues/9932).
- **Findings:**
  1. [medium] Design question for the maintainer: one token type `Extensions<T>` gates all extensions. Some are safe (`Dx12AdapterOptions` docs say "These options are safe"), and all still need `unsafe`. Andy says the `request_adapter` / `request_adapters` split bothers them more than the token. This needs a decision before merge, because it sets the pattern for every future extension ([#10304](https://github.com/gfx-rs/wgpu/issues/10304), [#4067](https://github.com/gfx-rs/wgpu/issues/4067)).
  2. [low] `CHANGELOG.md` covers only the LUID filter. It does not mention the new public `wgpu::Instance::request_adapters`, `wgpu::Extensions`, `BackendAdapterOptionsMap`, or the breaking `hal::Instance::enumerate_adapters` signature (that matters for out-of-tree hal backends).
  3. [low] `wgpu-hal/src/lib.rs` The doc on `Api::AdapterOptions` says "used with `request_adapter_ext`", but that function does not exist.
  4. [low] `wgpu-core/src/instance.rs` `request_adapters` now has its own `profiling::scope!` and `api_log!` with the name "Instance::request_adapter". `request_adapter` calls it, so each call logs twice under the same name.
- **To land:** a design decision on the token and API shape. CHANGELOG entries for the new public API. A fix for the stale doc reference.
- **Move-along action:** decide whether the `Extensions<T>` token plus separate-argument shape is the extension pattern, then review (or assign a reviewer).
- **Depth:** targeted (read the core, hal, wgpu API surface, DX12 filter, test and the first half of `wgpu-types/src/backend.rs`; skipped the `backend_map_type!` / `adapter_options!` macro bodies and the per-backend boilerplate)

**Deeper review ([#10775](https://github.com/gfx-rs/wgpu/issues/10775): feat(hal/dx12): Extension to restrict enumeration to a specific adapter LUID):**

> - **Open questions from the first pass:**
>   1. Review the skipped `backend_map_type!` / `backend_type!` macro bodies and the per-backend boilerplate.
>   2. Can safe code construct a non-empty `Extensions<T>`, or smuggle an extension to the wrong backend?
> - **Answers:**
>   1. [reviewed] `wgpu-types/src/backend.rs:390-447` `backend_map_type!`: a `const` assert ties it to the crate root. It generates a `CheckTrait`/`CheckStruct<const B: usize>` pair, so two value types for one backend in the same crate fail with a conflicting-impl error. The inner macro implements `Sealed<M>`, `BackendMapValue<M>` (with `BACKEND`) and `DynBackendMapValue<M>`. The matcher takes the struct body as one `tt`, so braced and unit structs work. Tuple and generic structs do not, which is acceptable. `Backend` discriminants are 0..=5 and `Backend::COUNT` is 6 (`backend.rs:80`), so `T::BACKEND as usize` stays in bounds. The per-backend boilerplate (Vulkan, Metal, GL, Noop types, an `_options` parameter on each `enumerate_adapters`, two example call sites passing `None`) is mechanical and correct. The DX12 filter compares `GetDesc().AdapterLuid` and drops an adapter if `GetDesc` fails, with an error log. That is reasonable.
>   2. [refuted: no safe construction and no wrong-backend delivery; a panic is possible] `wgpu-types/src/tokens.rs:13`: `Extensions<T>(Option<T>)` has a private field. The only constructors are `Default` (empty) and `unsafe fn new` (`:27`). It derives `Clone`/`Copy`/`Eq`/`PartialEq`/`Debug`, which cannot make a new non-empty value. It implements `Deref` to `Option<T>` (`:32`) and not `DerefMut`, so safe code cannot change the contents. `BackendMap` has no interior mutability. So safe code cannot construct a non-empty token or change one after construction.
>
>      Wrong backend: `BackendMap::set` indexes by `T::BACKEND`, and the hal reads with `map.get::<A::AdapterOptions>()` (`wgpu-hal/src/dynamic/instance.rs:64`). `get` downcasts through `Any` and checks the `TypeId`, so a backend can never receive another type as its own options. The seal is only partial: `__backend_map_private` is `pub` (`backend.rs:235`, "quasi-seal"). A downstream crate can implement `Sealed`/`BackendMapValue` for its own type with `BACKEND = Dx12` and put it in the map. Then `get` reaches `downcast_ref::<T>().unwrap()` (`backend.rs:298`, also `remove` at `:306`) and panics during `request_adapters`. This needs deliberate misuse, and an `unsafe { Extensions::new(..) }` call to get there, and the result is a panic, not UB. Using `.and_then(|v| v.downcast_ref())` (ignore foreign values) or documenting the panic would close it.
> - **New findings:**
>   1. [low-medium] `wgpu/src/backend/wgpu_core.rs:854` changes `request_adapter` from `apply_limit_buckets: false` to `options.apply_limit_buckets`. On trunk, `wgpu::Instance::request_adapter` ignores the user's `apply_limit_buckets` (`wgpu-types/src/adapter.rs:58`). This PR quietly fixes that, so adapters change for users who set it. It is user-visible, it is not in the CHANGELOG, and it is not related to the LUID filter. It should be its own commit with a CHANGELOG entry.
>   2. [low] `wgpu-types/src/backend.rs:480` `backend_type!` is `#[macro_export]`ed (public API) but nothing uses it. Its doc example names `wgt::BackendDeviceOptions`, which does not exist. Drop it until a user exists.
>   3. [low] `backend.rs:298,306` the `unwrap` panic described in answer 2.
>   4. [nit] `BackendAdapterOptions` (`wgpu-types/src/adapter.rs`) is only a type-level tag for `BackendMap<dyn BackendAdapterOptions>`. No value type implements it. The `BackendMap` doc says that `SomeTrait` "is implemented by the value type for each backend", which is not true here. Fix the doc or implement the trait in the inner macro.
> - **Verdict change:** none (Needs a bit of work; the design decision on the token is still open). The token itself is sound as an "unsafe to construct" gate.
> - **Move-along action change:** none. Add new finding 1 to the review notes.

#### [#10778](https://github.com/gfx-rs/wgpu/issues/10778): Make `wayland-sys`'s `dlopen` feature togglable ([@HigherOrderLogic](https://github.com/HigherOrderLogic))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI red (15 failed); last update 2026-10-29.
- **Verdict:** Needs a lot of work
- **Waiting on:** author (CI red). No reviewer is assigned and nobody has commented. A maintainer decision on whether to accept the feature is also needed.
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
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-06.
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

#### [#10922](https://github.com/gfx-rs/wgpu/issues/10922): dx12: add opt-in DXC f32 denormal modes ([@5andr0](https://github.com/5andr0))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-11.
- **Verdict:** Needs a lot of work
- **Waiting on:** maintainer, for a design decision (does wgpu want a DX12-only denormal option?) and a policy decision. No reviewer, no assignee.
- **Last activity:** 2026-11-11, 5andr0, marked ready for review. 2 days.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** Adds `Dx12BackendOptions::dxc_denormal_mode: Option<DxcDenormalMode>` (`Any`, `Preserve`, `FlushToZero`), re-exported from `wgpu`. When set, DXC gets `-denorm <mode>`, the mode goes into the shader cache key, and instance creation fails if FXC is in use. Shader compilation fails below shader model 6.2. The PR also moves the DXC argument list into a helper `dxc_arguments` and adds about 230 lines of tests.
- **Findings:**
  1. [high] Policy: the description says "This is an agent-written draft" and "The submitting human remains responsible for ... writing the final significant description". That was not done. "I self-reviewed and fully understand this PR" is not checked. Same author as [#10926](https://github.com/gfx-rs/wgpu/issues/10926).
  2. [medium] Design: there is no linked issue and no concrete use case. Denormal behavior is a cross-backend topic (Vulkan has `shaderDenormPreserveFloat32` and `shaderDenormFlushToZeroFloat32` through `SPV_KHR_float_controls`). A DX12-only public field in `wgpu-types` fixes the API shape without that discussion. A maintainer should decide whether wgpu wants this.
  3. [medium] `wgpu-hal/src/dx12/shader_compilation.rs:145-146` On a device whose shader model is below 6.2, an explicit mode makes every pipeline creation fail late with `PipelineError::Linkage`. The device is still created and reports no problem. A check at adapter or device creation would give an earlier, clearer error. Reasoned from code.
  4. [low] `wgpu-hal/src/dx12/instance.rs:51-61` The `StaticDxc` without `static_dxc` pre-check is redundant: `new_static_dxc` already fails in that build. It only changes the error message.
  5. [low] Complexity: about 370 changed lines in `shader_compilation.rs` for one compiler flag. The `#[ignore]` test `denormal_actual_dxc` prints full disassembly and needs a DLL in the working directory. `typos.toml` gains an exception (`Ges`) only for a test string.
  6. [low] `CHANGELOG.md:195-197` The entry wraps over three lines, unlike the other entries, and has no "By [@5andr0](https://github.com/5andr0) in [#10922](https://github.com/gfx-rs/wgpu/issues/10922)" attribution.
  7. [low] Adding a public field to `Dx12BackendOptions` breaks exhaustive struct literals. The description says this. The CHANGELOG does not.
- **To land:** maintainer decides on the API (and whether it should be cross-backend); author writes the description and attests review; move the shader-model check earlier; reduce the tests; fix the CHANGELOG.
- **Move-along action:** decide whether wgpu wants a DX12-only denormal option. If not, close and point to a design issue.
- **Depth:** full (did not check the DXC `-denorm` values against DXC sources; they match the documented values `any`, `preserve`, `ftz`)

## Core and API

#### [#10098](https://github.com/gfx-rs/wgpu/issues/10098): Add immediates support to WebGPU and remove `Features::IMMEDIATES` ([@beicause](https://github.com/beicause))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-09-18.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-10-06.
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
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-04.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-21.
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

#### [#10638](https://github.com/gfx-rs/wgpu/issues/10638): [hal] Link API hal docs from dyn methods ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-21.
- **Verdict:** Land as is
- **Waiting on:** other reviewer (jimblandy, assigned 2026-10-21, no review since)
- **Last activity:** 2026-10-21, jimblandy, assigned themself. The author's last push was 2026-10-16. 16 days.
- **Approvals:** none
- **CI:** green (RUSTDOCFLAGS `-D warnings` is set in CI, so the intra-doc links resolve)
- **Summary:** The PR adds a doc line "Dynamic version of [Trait::method]" to each method of the `Dyn*` traits in `wgpu-hal/src/dynamic/`. Docs only.
- **Findings:**
  1. [low] Trunk added one `DynCommandEncoder` method after the PR was opened. That method will have no link after the merge. This is not a blocker.
- **To land:** nothing
- **Move-along action:** ping jimblandy, or merge it yourself (docs only, trivial)
- **Depth:** full

#### [#10675](https://github.com/gfx-rs/wgpu/issues/10675): fix(core): hold `command_indices` around HAL surface (un)configure and queue drop ([@AdrianEddy](https://github.com/AdrianEddy))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-12.
- **Verdict:** Land as is
- **Waiting on:** maintainer. jimblandy is the assignee. sagudev's approval does not satisfy the review requirement, so a member must approve.
- **Last activity:** 2026-11-12, kixelated, comment. 1 day. The last review was 2026-11-02 (sagudev).
- **Approvals:** sagudev
- **CI:** green; mergeable
- **Summary:** Holds `Device::command_indices.write()` during HAL surface configure/unconfigure and during `Queue::drop`'s `wait_for_idle`. This serializes those calls with submit and present, and stops the multi-threaded `vkDeviceWaitIdle` race and the related crashes.
- **Findings:** unchanged. No issue found in the lock ordering. A second user (kixelated) now reports the same issue, after JMS55's Bevy DLSS report.
- **To land:** a member approval.
- **Move-along action:** approve and merge, or ping jimblandy. Decide the agent-disclosure question.
- **Depth:** status only

#### [#10776](https://github.com/gfx-rs/wgpu/issues/10776): Multi-planar texture copy validation ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (16 failed); last update 2026-11-13.
- **Verdict:** Needs a bit of work (the code itself looks ready for review)
- **Waiting on:** author (andyleiserson). Rebased on trunk on 11-13 after [#10710](https://github.com/gfx-rs/wgpu/issues/10710) merged.
- **Last activity:** 2026-11-10, andyleiserson, force-pushed. 3 days.
- **Approvals:** none
- **CI:** red from the PR. `tests/tests/wgpu-validation/api/texture.rs:824` builds a `TextureDescriptor` without `texture_binding_view_dimension` (error E0063 in the Clippy log).
- **Summary:** Folds `extract_texture_selector` into `validate_texture_copy_range`, validates copy extents in plane texels for single-plane copies (plane-0 texels plus chroma-subsampling alignment for all-plane copies), and allows texture-buffer copies and copies into one plane of a multi-planar texture.
- **Findings:**
  1. [medium] It is stacked on [#10710](https://github.com/gfx-rs/wgpu/issues/10710) and contains the old [#10710](https://github.com/gfx-rs/wgpu/issues/10710) commit. After [#10710](https://github.com/gfx-rs/wgpu/issues/10710) merges (with its feedback commit), the branch needs another rebase.
  2. [medium] `tests/tests/wgpu-gpu/planar_texture/mod.rs`, `tests/tests/wgpu-gpu/zero_init.rs` and `tests/tests/wgpu-validation/api/texture.rs` (one new literal each) The `TextureDescriptor` literals have no `texture_binding_view_dimension`, so the tests do not compile on current trunk. Reasoned from code, not built.
  3. [low] Earlier review notes still apply: the T2T aspect check now applies only to depth-stencil formats, which is correct; zero-init per plane is safe because the init tracker is per (mip, layer) and any plane write is treated as partial; `plane_size` adds a new `assert_eq!` in a `#[doc(hidden)]` public function.
  4. [low] With [#9624](https://github.com/gfx-rs/wgpu/issues/9624) merged, this PR now also enables copies into one plane on Metal. The Metal `copy_texture_to_texture` asserts apply only when the source and destination pixel formats differ. For plane copies that `copy_format` allows, the plane format equals the other side's format, so the asserts are not reached. Reasoned from code.
- **To land:** merge [#10710](https://github.com/gfx-rs/wgpu/issues/10710). Rebase. Add the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field. Review.
- **Move-along action:** assign a reviewer now (ErichDonGubler has the [#10710](https://github.com/gfx-rs/wgpu/issues/10710) context), so that the review runs in parallel with the [#10710](https://github.com/gfx-rs/wgpu/issues/10710) merge.
- **Depth:** targeted (verified that the force-push did not change the patches; did not re-review the code)

#### [#10795](https://github.com/gfx-rs/wgpu/issues/10795): [core] `Texture::mark_uninitialized` for lazy zeroing ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-14.
- **Verdict:** Land as is
- **Waiting on:** maintainer. cwfitzgerald approved and queued it on 11-14, but Mergify removed it from the queue with `CHECKS_FAILED`.
- **Last activity:** 2026-11-09, inner-daemons, approved. 4 days.
- **Approvals:** inner-daemons, cwfitzgerald
- **CI:** green; mergeable
- **Summary:** Adds the public wgpu-core `Texture::mark_uninitialized`, which resets the init tracker so that wgpu lazily zeroes the texture on the next read. This is for the Firefox/Servo presentation path ([#5138](https://github.com/gfx-rs/wgpu/issues/5138)).
- **Findings:** unchanged from the old entry (all low): reads fail on `TextureClearMode::None` textures instead of seeing zeros; the doc should state queue-order semantics and has grammar slips; the test covers only the buffer-copy clear path.
- **To land:** nothing. A member approval or merge if inner-daemons' approval does not count.
- **Move-along action:** read the Mergify check, then queue it again.
- **Depth:** status only

#### [#10802](https://github.com/gfx-rs/wgpu/issues/10802): [deno] Report external memory size for GPUBuffer and GPUTexture to encourage GC of them ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-01.
- **Verdict:** Needs a bit of work
- **Waiting on:** maintainer (no reviewer assigned, no review)
- **Last activity:** 2026-11-01, sagudev, force-pushed and renamed the PR. 5 days.
- **Approvals:** none
- **CI:** green
- **Summary:** In `deno_webgpu`, `createBuffer` and `createTexture` now report their size to V8 as external memory. A weak finalizer subtracts that size again. This applies pressure on the GC so that unused GPU objects are collected. Fixes [#10523](https://github.com/gfx-rs/wgpu/issues/10523). The PR also makes `wgpu_core::resource::Buffer::check_is_valid` and `Texture::check_valid` public.
- **Findings:**
  1. [low] `deno_webgpu/device.rs` The accounting is fixed at creation time. `destroy()`, unmap (which frees the `data` Vec), and later `mapAsync` allocations do not change the reported size. So the PR over-reports until GC. That is probably acceptable for a GC hint, but a short comment should say so.
  2. [low] `wgpu-core/src/resource.rs:680,1888` This makes two `pub(crate)` validity checks public on wgpu-core types. The names are not consistent (`check_is_valid` and `check_valid`). The change is small, but it widens the wgpu-core API only for deno. Rename both to the same name, or accept as is.
  3. [low] The finalizer setup code is the same for buffer and texture. It could be one helper.
  4. [low] There is no test. That is normal for deno_webgpu GC behavior. No CHANGELOG entry is needed (the change is deno-only).
- **To land:** a reviewer must look at it. Possibly unify the check method names.
- **Move-along action:** review (or assign a deno-familiar reviewer). The PR is small and CI is green.
- **Depth:** full

#### [#10812](https://github.com/gfx-rs/wgpu/issues/10812): [core] Add child texture concept that shares underlying data and subsets TextureUsage ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; has conflicts; CI green; last update 2026-11-09.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-09.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-09.
- **Verdict:** Needs a bit of work. Policy question first.
- **Waiting on:** ErichDonGubler or teoxoy. On 11-09 teoxoy asked Erich: "do you want to take this PR or should I?" No answer.
- **Last activity:** 2026-11-09, teoxoy, comment. 4 days. The last author activity was 2026-11-05.
- **Approvals:** none
- **CI:** green; mergeable
- **Summary:** `DiskTrace` writes one compact RON action per line and ends the file with `// End of trace`. `player::parse_trace` parses each line and skips an unparsable last line. The old list format is no longer readable.
- **Findings:** unchanged from the old entry.
- **To land:** settle the agent-account policy. Decide the RON format. A full review by Erich or teoxoy.
- **Move-along action:** decide the agent-account question, and tell teoxoy and Erich who owns the review.
- **Depth:** status only

#### [#10838](https://github.com/gfx-rs/wgpu/issues/10838): Fix sparse tlas instance indices ([@Vecvec](https://github.com/Vecvec))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-14.
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
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-09.
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
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-09.
- **Verdict:** Needs a bit of work
- **Waiting on:** inner-daemons (assignee and requested reviewer since 11-09). Maintainer decision on the agent-disclosure checklist.
- **Last activity:** 2026-11-09, inner-daemons, commented "Makes sense as an optimization, we should probably prioritize this", and assigned themself. 4 days.
- **Approvals:** none
- **CI:** green (run from 11-06, before [#8014](https://github.com/gfx-rs/wgpu/issues/8014) merged); mergeable
- **Summary:** Adds a `TrackerPool` to `Device`. Command encoders, the pending-writes encoder, and the intermediate trackers of compute and ray tracing passes take a `Tracker` from the pool. The tracker goes back to the pool, cleared with its capacity kept, when the submission completes (`EncoderInFlight::release`) or the pass ends. This removes per-encoder allocations that scale with the device's resource count. A second commit fixes the Metal `HalCounters::query_sets` count for occlusion query sets.
- **Findings:**
  1. [medium] `tests/tests/wgpu-gpu/tracker_pool.rs:~93` The new `TextureDescriptor` literal has no `texture_binding_view_dimension` field, so the test does not compile on current trunk (after [#8014](https://github.com/gfx-rs/wgpu/issues/8014)). Reasoned from code, not built. Rebase and add the field.
  2. [low] The core change looks correct. `ResourceMetadata::clear` now truncates `owned` to length 0. All other vectors are also emptied, so a pooled tracker is consistent (size 0) and grows again through `allow_index`/`set_size`. The only other caller, the `UsageScope` drop path, calls `set_size` again in `new_pooled`. `TextureStateSet::clear` also clears the `complex` map. `TrackerPool::release` drops the resource references before it takes the pool lock. The ordered-usage masks are not reset, but they are the same for every tracker of one device. Trackers on error paths are dropped and not returned, which only costs an allocation.
  3. [low] `wgpu-core/src/track/mod.rs:605` The pool has no cap and pooled trackers never shrink. Each pooled tracker keeps vectors sized to the highest tracker index the device ever had, and the pool keeps as many trackers as the peak number of in-flight encoders. The author offers a cap. This is a maintainer decision.
  4. [low] The new `DEVICE_TRACKER_POOL` lock rank is added as a follower of five ranks. I did not check each edge against the call paths. CI with debug builds passes.
  5. [low] Policy: both agent-disclosure boxes are checked ("automatically created by an Agent" and "description automatically written by an Agent"). The LLM section says that Claude Code ported the author's prototype, wrote the tests and drafted the description, and that the author reviewed and edited it. AGENTS.md does not allow fully agentic PRs. The maintainers must decide whether this is acceptable.
  6. [low] The Metal counter fix is a separate commit and is described. CHANGELOG entries are present for both changes.
- **To land:** rebase and fix the test literal. A maintainer decision on the agent disclosure. A review by inner-daemons.
- **Move-along action:** decide the agent-disclosure question, then let inner-daemons review (they want to prioritize it).
- **Depth:** full (core, hal and CHANGELOG diff read; tests read for coverage only)

#### [#10871](https://github.com/gfx-rs/wgpu/issues/10871): Add a ray tracing invocation reordering feature/adapter property to wgpu ([@JMS55](https://github.com/JMS55))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI pending; last update 2026-11-14.
- **Verdict:** Needs a bit of work
- **Waiting on:** you (review requested). Vecvec approved on 11-14.
- **Last activity:** 2026-11-11, JMS55, merged trunk. 2 days.
- **Approvals:** Vecvec
- **CI:** red, but not caused by the PR. On Linux and Mac every test passed (1667 and 1211 tests). The jobs failed in "Upload comparison images" and "Upload test timings" with `Failed to CreateArtifact` (GitHub artifact service outage). The Windows job was cancelled. Rerun. Mergeable.
- **Summary:** [#10311](https://github.com/gfx-rs/wgpu/issues/10311) (merged 2026-11-06) added hit objects and `reorderThread` to naga: the IR, WGSL `enable wgpu_ray_tracing_invocation_reorder`, SPIR-V output, and `Capabilities::RAY_TRACING_INVOCATION_REORDER`. No wgpu feature mapped to that capability, so wgpu users could not use it. This PR adds that part: the feature `EXPERIMENTAL_RAY_TRACING_INVOCATION_REORDER` (bit 25, in the experimental mask), the mapping in `wgpu-naga-bridge`, and Vulkan support through `VK_EXT_ray_tracing_invocation_reorder` (extension enable, feature struct, `ShaderInvocationReorderEXT` SPIR-V capability). It also adds `Adapter::ray_tracing_invocation_reorder_mode`, which says whether `reorderThread` actually reorders (hardware hint) or is a no-op.
- **Findings:**
  1. [medium] The Vulkan path has no test coverage. The author says it is "not tested" on hardware. In CI, `invocation_reorder_output` reports "Unsupported: Features" on llvmpipe, Metal and GL, so the GPU test runs nowhere. Only the noop validation tests run. Ask for a run on Ada or Blackwell (reorder mode) and on Turing or Ampere (no-op mode), or ask a maintainer with such hardware to run it.
  2. [low] `wgpu-hal/src/vulkan/adapter.rs:33` `ash` has no bindings for the EXT, so the PR defines its own `FeaturesEXT` and `PropertiesEXT` structs with `unsafe impl` for `p_next` chaining. I checked them against `vulkan_core.h` from Vulkan SDK 1.4.363: the structure types (1000581000 and 1000581001), the field layout and `VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT = 1` match.
  3. [low] `wgpu-types/src/features.rs:713` The docs say the feature "Requires `EXPERIMENTAL_RAY_TRACING_PIPELINES`", but wgpu-core does not enforce this at device creation. The Vulkan adapter only reports the feature together with RT pipelines, and a request for it alone still enables `VK_KHR_ray_tracing_pipeline`. Hit objects are only valid in ray tracing stages, so a device with only this feature cannot use it. No unsafety follows. Either enforce the dependency or soften the doc.
  4. [low] `CHANGELOG.md:181` The new text is added to the [#10311](https://github.com/gfx-rs/wgpu/issues/10311) entry under `#### Naga`. The new wgpu feature and the new `Adapter` method are wgpu API additions and may belong in the general new-features section.
  5. [low] `examples/standalone/custom_backend` and `wgpu/src/dispatch.rs` add a required `AdapterInterface` method. This is a break for custom backends, the same as other recent additions. The disclosure checklist is unticked. The LLM section says that AI wrote the code from a prompt and the author reviewed it and cleaned up the docs. This is not fully agentic.
- **To land:** rerun CI. A hardware test of the Vulkan path. Decide the CHANGELOG placement and the feature-dependency check.
- **Move-along action:** review (you are a requested reviewer), and ask JMS55 or Vecvec for a hardware run.
- **Depth:** full (all non-test code read; tests read for coverage)

#### [#10896](https://github.com/gfx-rs/wgpu/issues/10896): Expose `is_surface_supported` on device ([@sagudev](https://github.com/sagudev))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-14.
- **Verdict:** Land as is
- **Waiting on:** the merge queue. cwfitzgerald approved on 11-14 and sent `@mergifyio queue`.
- **Last activity:** 2026-11-13, sagudev, rebased. 0 days.
- **Approvals:** inner-daemons, cwfitzgerald
- **CI:** red, not caused by the PR. `Test Linux x86_64` failed in `player::tests::test_api` with `Result::unwrap()` on `Err(Timeout)` (`player/tests/player/main.rs:114`). The PR does not touch the player or any code that the player test uses. It is probably a flake. Rerun. Mergeable.
- **Summary:** Adds `wgpu::Device::is_surface_supported`, which forwards to the device's adapter. Users often keep only `Device` and `Queue`, and WebGPU style is to not use the adapter after device creation. This follows [#10895](https://github.com/gfx-rs/wgpu/issues/10895) (queue getter on device). It adds a public `wgpu_core::device::Device::adapter()` getter. Part of [#10892](https://github.com/gfx-rs/wgpu/issues/10892).
- **Findings:**
  1. [low] `wgpu/src/api/device.rs:135` The doc says "Returns whether this adapter may present". On `Device` it should say "the adapter this device was created from".
  2. [low] There is no test. The method only forwards to the existing adapter path (`CoreAdapter::is_surface_supported` makes the same call). The WebGPU backend returns `true`, the same as `WebAdapter`. The custom backend example adds a new required `DeviceInterface` method.
  3. [low] The new `pub fn adapter()` on the wgpu-core `Device` widens the wgpu-core API for Firefox and Servo. This is harmless.
- **To land:** rerun CI. Optionally fix the doc wording.
- **Move-along action:** none.
- **Depth:** full

#### [#10910](https://github.com/gfx-rs/wgpu/issues/10910): fix: Correct supported usages for multi-planar textures ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-11.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (andyleiserson): rebase on trunk. [#10710](https://github.com/gfx-rs/wgpu/issues/10710), which it needed, merged on 11-13.
- **Last activity:** 2026-11-11, cwfitzgerald, reply in the review thread. 2 days.
- **Approvals:** cwfitzgerald
- **CI:** red, caused by the PR. `Test Linux x86_64` fails in `wgpu_gpu::zero_init::zero_init_multiplanar_texture` on Vulkan/llvmpipe (`zero_init.rs:2167`, plane 1 is not zero). Mergeable.
- **Summary:** `create_texture` now validates the usage of a multi-planar texture against the format features of the multi-planar format itself, not against the intersection of its plane formats. It also requires the features of the plane formats (for example `TEXTURE_FORMAT_16BIT_NORM` for P010). The Vulkan and Metal HAL report multi-planar capabilities from the planes, limited to sampling and color attachment. wgpu-core removes storage capabilities for multi-planar formats, so `STORAGE_BINDING` is no longer allowed. The guaranteed features change to NV12 `attachment` and P010 `basic`.
- **Findings:**
  1. [high] `wgpu-types/src/texture/format.rs:~1003` Cause of the CI failure (reasoned from code, not run). NV12 now reports `RENDER_ATTACHMENT`. `map_texture_usage_for_texture` (`wgpu-core/src/conv.rs:155`) then adds `COLOR_TARGET` to an NV12 texture that has only `COPY_SRC` usage, so its clear mode becomes `RenderPass`. On trunk, `Texture::get_clear_view` (`wgpu-core/src/resource.rs:1928`) does not index by plane, and `clear_texture_via_render_passes` clears only the plane-0 view. Plane 1 is never zeroed. The test reads plane 1 and fails. This exposes uninitialized texture memory through a safe API, which is the bug that [#10710](https://github.com/gfx-rs/wgpu/issues/10710) fixes. [#10910](https://github.com/gfx-rs/wgpu/issues/10910) must land after [#10710](https://github.com/gfx-rs/wgpu/issues/10710), or it must be rebased on it.
  2. [medium] `tests/tests/wgpu-gpu/planar_texture/mod.rs:~262` and `tests/tests/wgpu-validation/api/texture.rs` (two literals) The new `TextureDescriptor` literals have no `texture_binding_view_dimension`, so they do not compile on current trunk. Reasoned from code, not built.
  3. [low] The CHANGELOG entries link `#TODO`. The description has "TODO: rewrite this description". All checklist boxes are unticked. The agent-disclosure lines are left commented out, and the LLM section says "Yes, especially the tests".
  4. [low] `STORAGE_BINDING` on NV12/P010 is now rejected. This is a user-visible break, and the CHANGELOG states it. NV12 guaranteed features now also include `TRANSIENT_ATTACHMENT` (part of `attachment`). The old plane-intersection path allowed it too, but the new test loop does not cover it.
  5. [low] The usage validation itself looks correct. The plane-feature check runs before the usage check, and both the guaranteed path and the adapter path are tested on GPU. The Vulkan early return for multi-planar formats skips the MSAA flags, which is correct because multi-planar textures cannot be multisampled.
- **To land:** land [#10710](https://github.com/gfx-rs/wgpu/issues/10710) first and rebase on it, which should make CI green. Add the [#8014](https://github.com/gfx-rs/wgpu/issues/8014) field to the test literals. Replace the `#TODO` links and finish the description.
- **Move-along action:** ping author: rebase on [#10710](https://github.com/gfx-rs/wgpu/issues/10710) after it merges (the CI failure is the plane-1 clear bug that [#10710](https://github.com/gfx-rs/wgpu/issues/10710) fixes).
- **Depth:** full

## CI, dependencies and your own PRs

#### [#10739](https://github.com/gfx-rs/wgpu/issues/10739): chore(ci renovate): Configure `rebaseWhen = conflicted` ([@andyleiserson](https://github.com/andyleiserson))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-02.
- **Verdict:** Land as is (small config change; author is unsure it is needed)
- **Waiting on:** maintainer (jimblandy is assignee; no review given)
- **Last activity:** 2026-10-27, andyleiserson, comment ("not sure top-level `rebaseWhen` is necessary"). 10 days.
- **Approvals:** none
- **CI:** green on its own checks. `renovate/config-validation` status is success (2026-10-26). Check runs for Clippy etc. pass (I saw the first 3 only; codecov pass). Base is stale (trunk 0c7c121, ~11 days old).
- **Summary:** Sets `rebaseWhen: "conflicted"` globally in `renovate.json` and removes the `lockFileMaintenance`-only `behind-base-branch` override. Goal: stop Renovate rebasing PRs on every trunk push, which evicts them from the Mergify queue and burns CI.
- **Findings:**
  1. [medium] Description says "Untested" and the author doubts the top-level setting is needed (Renovate may already rebase for other reasons). It is a low-risk, easily reverted config change, but the benefit is unproven. Note [#10721](https://github.com/gfx-rs/wgpu/issues/10721)/#10722 were force-pushed on 2026-11-05 despite no conflict, which may show the effect (or lack of it) once this lands.
  2. [low] Removing the lockfile-specific `behind-base-branch` is a behavior change: with `recreateWhen: always` the lock PR is recreated weekly anyway, so likely fine.
  3. [low] Description mentions [#10706](https://github.com/gfx-rs/wgpu/issues/10706) as a previous attempt; fine.
- **To land:** Nothing. Optionally squash, which is already the stated plan.
- **Move-along action:** merge (low risk, reversible). Reconsider only if you want evidence first.
- **Depth:** full

**CI and status check ([#10739](https://github.com/gfx-rs/wgpu/issues/10739): chore(ci renovate): Configure `rebaseWhen = conflicted`):**

> - **Open questions from the first pass:** full CI.
> - **Answers:**
>   1. [confirmed] 66 check runs: 64 success, 2 neutral. No failure. Head a809737 (branch in `gfx-rs/wgpu`). `blocked` (review), `mergeable: true`.
> - **New findings:** none
> - **Verdict change:** none
> - **Move-along action change:** none

#### [#10784](https://github.com/gfx-rs/wgpu/issues/10784): test(cts): run non-`const` `modf` execution tests for `f32` ([@ErichDonGubler](https://github.com/ErichDonGubler))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-10-30.
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

#### [#10801](https://github.com/gfx-rs/wgpu/issues/10801): ci: Update Vulkan SDK to 1.4.363 ([@inner-daemons](https://github.com/inner-daemons))
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-02.
- **Verdict:** Needs a bit of work
- **Waiting on:** author (inner-daemons) for the MoltenVK follow-up; or maintainer decision to accept as is
- **Last activity:** 2026-11-02, andyleiserson, tested the SDK on MoltenVK/KosmicKrisp (comment). 4 days.
- **Approvals:** none
- **CI:** green (all 66 checks pass, including Linux/Mac/Win tests and shader validation).
- **Summary:** Bumps the CI Vulkan SDK from 1.4.357 to 1.4.363 and updates the `wgsl-early-depth-test-conservative` SPIR-V to GLSL snapshot (new `#extension GL_ARB_conservative_depth : require`). Author says Claude generated it.
- **Findings:**
  1. [medium] Andy's comment: with the new SDK `wgpu_gpu::shader::struct_layout::{storage,uniform}_input` fail on MoltenVK (a 24-byte struct with `vec3` reads wrong offsets). Possible upstream regression, maybe intermittent. `draw_index` on MoltenVK now passes. They say it does not need to block landing, but suggests marking the struct_layout tests flaky/expected-fail on MoltenVK. CI does not cover MoltenVK, so this is invisible in CI.
  2. [low] Description has a blank "Connections" and "Testing: None". The CHANGELOG checkbox is ticked but there is no CHANGELOG change (none needed; fix the checkbox).
  3. [low] PR changes only the SDK default and one snapshot, so snapshot churn is minimal. Good.
- **To land:** Decide whether to take as is. If not, add a MoltenVK expected-fail/flaky marker for the two struct_layout tests (and ideally file an upstream/KosmicKrisp issue).
- **Move-along action:** decide: merge now (Andy said non-blocking) and track the MoltenVK failures in a separate issue; or ping inner-daemons to add the test marker. I suggest merge plus an issue.
- **Depth:** full

#### [#10869](https://github.com/gfx-rs/wgpu/issues/10869): Update KosmicKrisp test expectations ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-06.
- **Verdict:** Status note
- **Waiting on:** reviewer. No review requested.
- **Last activity:** 2026-11-06, cwfitzgerald, one commit. 7 days.
- **Approvals:** none
- **CI:** green (65 pass); mergeable.
- **Summary:** Updates the expected failures of tests on KosmicKrisp for Vulkan SDK 1.4.363.1 on Apple M5 Max. Adds expected failures for 17 `draw_indirect` tests, `render_pass_resource_ownership` and `vertex_indices`. Removes `draw_multiview_noncontiguous` and `subgroup_operations`. Touches 5 test files only. Connects to [#9187](https://github.com/gfx-rs/wgpu/issues/9187).
- **Findings:** none (status only). The disclosure says Claude Code made the changes and wrote the description, and the author edited the description. The self-review box is unchecked.
- **To land:** the author checks the self-review box; one approval.
- **Move-along action:** Check the self-review box and ask a maintainer to approve.
- **Depth:** status only

#### [#10883](https://github.com/gfx-rs/wgpu/issues/10883): Update taiki-e/install-action action to v2.90.3 ([@renovate](https://github.com/renovate))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI pending; last update 2026-11-11.
- **Verdict:** Status note
- **Waiting on:** reviewer; CI is still running.
- **Last activity:** 2026-11-11, renovate. 2 days.
- **Approvals:** none
- **CI:** pending (64 pass, 1 pending, 3 skipped); mergeable. No failures.
- **Summary:** Patch bump of a GitHub Action. No code change needed.
- **Findings:** none
- **To land:** wait for the pending job; merge.
- **Move-along action:** Merge when green.
- **Depth:** status only

#### [#10884](https://github.com/gfx-rs/wgpu/issues/10884): Update zizmorcore/zizmor-action action to v0.8.1 ([@renovate](https://github.com/renovate))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-11.
- **Verdict:** Status note
- **Waiting on:** reviewer.
- **Last activity:** 2026-11-11, renovate. 2 days.
- **Approvals:** none
- **CI:** green (65 pass); mergeable.
- **Summary:** Minor bump of the zizmor GitHub Action. No code change needed.
- **Findings:** none
- **To land:** merge.
- **Move-along action:** Merge.
- **Depth:** status only

#### [#10886](https://github.com/gfx-rs/wgpu/issues/10886): Update Rust crate libloading to 0.9 ([@renovate](https://github.com/renovate))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (18 failed); last update 2026-11-13.
- **Verdict:** Status note
- **Waiting on:** author or a maintainer. Code changes are needed.
- **Last activity:** 2026-11-13, renovate. 0 days.
- **Approvals:** none
- **CI:** red (18 fail, 47 pass). Cause: compile errors with libloading 0.9: E0277 "the trait bound `P: AsFilename` is not satisfied", and E0599 "no variant named `CreateCString` / `CreateCStringWithTrailing` found for enum `libloading::Error`" (the error enum changed). The `cargo-deny` job also fails: "found 2 duplicate entries for crate 'libloading'" (other crates still use the old version). No MSRV-specific message was seen; the MSRV jobs fail from the same compile errors. Mergeable.
- **Summary:** Major bump (0.8 to 0.9) of libloading. The code that matches on `libloading::Error` variants must change, and the generic `AsFilename` bound must be fixed. The duplicate crate entry may need a skip entry in `deny.toml` or a wait for dependents to update.
- **Findings:** none
- **To land:** update the code for the new libloading API; fix or skip the duplicate in cargo-deny; green CI.
- **Move-along action:** Decide: port to 0.9 now, or close and ignore the major until dependents move.
- **Depth:** status only

#### [#10888](https://github.com/gfx-rs/wgpu/issues/10888): Update Rust crate bincode to v3 ([@renovate](https://github.com/renovate))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (56 failed); last update 2026-11-09.
- **Verdict:** Status note
- **Waiting on:** author or a maintainer. Code changes are needed.
- **Last activity:** 2026-11-09, renovate. 4 days.
- **Approvals:** none
- **CI:** red (56 fail, 8 pass). Cause: cargo cannot resolve the dependency. `wgpu-benchmark` depends on `bincode` with feature `serde`, and bincode 3.0.0 has no `serde` feature. The jobs stop at dependency resolution (exit code 101). Mergeable.
- **Summary:** Major bump to bincode 3. The API changed: the `serde` feature is gone, so `benches/Cargo.toml` and the code that uses bincode must be rewritten. Renovate also failed to update the lock file.
- **Findings:** none
- **To land:** change the benchmark code and `Cargo.toml` to the bincode 3 API, or close the PR.
- **Move-along action:** Decide: port the benchmarks to bincode 3, or close and ignore the major.
- **Depth:** status only

#### [#10903](https://github.com/gfx-rs/wgpu/issues/10903): Fix binding more immediate data than pipeline supports ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI green; last update 2026-11-13.
- **Verdict:** Status note
- **Waiting on:** reviewer. No review requested.
- **Last activity:** 2026-11-13, cwfitzgerald, push. 0 days.
- **Approvals:** none
- **CI:** green (66 pass); mergeable.
- **Summary:** Core trims immediates to the layout size, but it still calls hal `set_immediates` with an empty slice. That panics on Metal and DX12 and is a validation error on Vulkan. Render bundles did not trim and did not resend after a layout change. The fix adds a shared `ImmediateState::take_dirty` that trims and skips empty writes. It also initializes the DX12 immediates root element on signature reset. Files: `wgpu-core/src/command/bundle.rs`, `pass.rs`, `wgpu-hal/src/dx12/command.rs`, `wgpu-hal/src/lib.rs`, a changelog entry and tests in `immediates.rs`.
- **Findings:** none (status only, no code review done).
- **To land:** a full code review (this touches wgpu-core and wgpu-hal); one approval.
- **Move-along action:** Get a reviewer for the core and hal changes.
- **Depth:** status only

#### [#10930](https://github.com/gfx-rs/wgpu/issues/10930): Add v30.0.2 changelog and correct the release checklist ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-14.
- **Verdict:** Status note
- **Waiting on:** merge. ErichDonGubler approved on 11-13.
- **Last activity:** 2026-11-13, ErichDonGubler, approval. 0 days.
- **Approvals:** ErichDonGubler
- **CI:** green (66 pass); mergeable. It targets trunk.
- **Summary:** Moves the v30.0.2 changelog and release-checklist corrections to trunk. Files: `CHANGELOG.md`, `docs/release-checklist.md`. It has the label "changelog: released entry changed". It is the roll-up of the v30.0.2 release work, and [#10929](https://github.com/gfx-rs/wgpu/issues/10929) is the release PR that rolled up the closed backports.
- **Findings:** none (status only).
- **To land:** merge.
- **Move-along action:** Merge.
- **Depth:** status only

## Drafts

#### [#6584](https://github.com/gfx-rs/wgpu/issues/6584): Test pass timestamp writes validation ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI no runs; last update 2025-10-08.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2025-10-08, ErichDonGubler, last push/update. 394 days.
- **CI:** stale
- **Summary:** Adds an empty GPU test `VALIDATE_PASS_TIMESTAMP_WRITES` whose body is `todo!()` (5 lines, 1 file). Requested by Wumpf in [#6578](https://github.com/gfx-rs/wgpu/issues/6578). No test logic exists. No comments.
- **Move-along action:** close (empty stub, 2 years old; author can reopen when it is written)

#### [#8480](https://github.com/gfx-rs/wgpu/issues/8480): Add `set_web_canvas_size` option. ([@kpreid](https://github.com/kpreid)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI no runs; last update 2025-11-05.
- **State:** stalled
- **Waiting on:** maintainer (decision: add a `SurfaceConfiguration` field defaulting to not resizing the web canvas, "option 1" of [#7938](https://github.com/gfx-rs/wgpu/issues/7938)?)
- **Last activity:** 2025-11-05, kpreid, opened. 366 days. No comments or reviews.
- **CI:** stale
- **Summary:** Adds `SurfaceConfiguration::set_web_canvas_size` (55/-16 lines, 11 files) to stop the infinite canvas resize on WebGPU/WebGL ([#7938](https://github.com/gfx-rs/wgpu/issues/7938), still open). Author says it is untested on web and "do not merge".
- **Move-along action:** decide whether option 1 is the accepted API, then ping kpreid (or close if a different fix for [#7938](https://github.com/gfx-rs/wgpu/issues/7938) is chosen)

#### [#8653](https://github.com/gfx-rs/wgpu/issues/8653): Texture format testing ([@inner-daemons](https://github.com/inner-daemons)) [draft PR]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-08-25.
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
- **State (2026-11-14):** draft; has conflicts; CI red (1 failed); last update 2026-01-21.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-01-21, sagudev (collaborator) suggested a generic cloneable `AnyError` wrapper. 289 days.
- **CI:** stale
- **Summary:** Adds constructors for error types so custom backends can return them (42 lines, 3 files). Boxing the custom error breaks `Clone`/`PartialEq`/`serde` derives on `RequestAdapterError`, `RequestDeviceErrorKind`, `CreateSurfaceErrorKind`. Author says a better design is needed.
- **Move-along action:** decide whether a custom-backend error wrapper (`AnyError`-style) is wanted; if not, close

#### [#8954](https://github.com/gfx-rs/wgpu/issues/8954): fix(naga): Fix builtin metal imports ([@madsmtm](https://github.com/madsmtm)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (3 failed); last update 2026-01-28.
- **State:** stalled
- **Waiting on:** maintainer (decision: how to fix `using namespace metal;` removal for old macOS: 1) re-add it and reserve its names, 2) prefix all types with `metal::`, 3) import each used type explicitly)
- **Last activity:** 2026-01-28, madsmtm, asked kvark for input. 282 days. No reply.
- **CI:** stale
- **Summary:** Author tried one approach (94 files, mostly snapshot churn); description says "TODO: Solution?". Helps macOS 10.12 (11 failing tests left with [#8953](https://github.com/gfx-rs/wgpu/issues/8953) applied).
- **Move-along action:** decide which of the three options is acceptable (kvark was asked, no answer); or ping kvark

#### [#9023](https://github.com/gfx-rs/wgpu/issues/9023): Remove layer from superlayer in `Drop` if needed ([@madsmtm](https://github.com/madsmtm)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (49 failed); last update 2026-02-08.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-02-08, madsmtm, opened. 271 days. No comments.
- **CI:** stale
- **Summary:** Fixes the macOS CALayer leak in [#9021](https://github.com/gfx-rs/wgpu/issues/9021) (still open) by removing the layer from its superlayer on `Surface` drop (52/-7 lines, 5 files). Marked "Not yet tested". Clippy and test boxes unchecked.
- **Move-along action:** ping author (is it tested? rebase, then mark ready)

#### [#9053](https://github.com/gfx-rs/wgpu/issues/9053): Host image copies ([@inner-daemons](https://github.com/inner-daemons)) [draft PR]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-09-23.
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
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-06-24.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-06-24, jimblandy, moved to draft: error reporting from Firefox background polling is not resolved and they are "no longer sure we need this". 135 days.
- **CI:** stale
- **Summary:** Adds a 12-line `WebGpuError` impl for `WaitIdleError` for the Firefox poll-from-thread work. teoxoy noted device-loss can be ignored, other variants may need to lose the device.
- **Move-along action:** close (author does not need it now; reopen with the polling design)

#### [#9353](https://github.com/gfx-rs/wgpu/issues/9353): fix(wgpu-hal/vk): don't use sequential bindings if pipeline layout is set explicitly ([@dylanblokhuis](https://github.com/dylanblokhuis)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI 1 cancelled jobs; last update 2026-09-02.
- **State:** stalled
- **Waiting on:** maintainer (decision: how passthrough/SPIR-V shaders get explicit binding indices; `bgl::Origin::Derived` is rejected as the switch)
- **Last activity:** 2026-08-07, inner-daemons, asked the author if they will return. 91 days. Author last commented 2026-04-14.
- **CI:** unknown (status API empty)
- **Summary:** In Vulkan, keeps user binding indices for explicitly set layouts instead of sequential ones (34/-16 lines, 3 files; labels vulkan, extensions). andyleiserson says explicit BGLs do not imply passthrough shaders and web content must not control this; a layout can serve both Naga and passthrough shaders. No test added. Inner-daemons links it to a future precompiled-shaders API.
- **Move-along action:** decide whether to handle this now (an opt-in flag, e.g. `PRESERVE_BINDING_INDICES`) or wait for the precompiled-shaders API; then close or ping author

#### [#9380](https://github.com/gfx-rs/wgpu/issues/9380): use threadgroup pointers instead of references in metal ([@39ali](https://github.com/39ali)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-08-12.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-12, 39ali (author update; PR updated 2026-08-12). Last human comment 2026-07-15 by inner-daemons asking if the author will return. 86 days since update.
- **CI:** green on 2026-04-21 run (58 check runs seen, codecov re-ran 2026-08-05 green); may be stale against trunk.
- **Summary:** Metal backend uses threadgroup pointers instead of references (296/-203 lines, 16 files), aims to fix [#4500](https://github.com/gfx-rs/wgpu/issues/4500) (still open, driver miscompile of workgroup memory). inner-daemons requires mesh/task shader paths to be covered before landing (thread r3128609110); teoxoy agrees at least that thread must be resolved.
- **Move-along action:** ping author (cover mesh/task shaders, rebase) or decide to split mesh/task shaders into a follow-up issue

#### [#9427](https://github.com/gfx-rs/wgpu/issues/9427): fix(core): dont panic in Queue::drop ([@atlv24](https://github.com/atlv24)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-08-12.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-12, PR updated (author push or edit); last human comment 2026-08-07, inner-daemons: "Should this be closed?". 86 days.
- **CI:** unknown (status API empty)
- **Summary:** Removes a panic in `Queue::drop` (2/-8 lines, 1 file), follow-up to [#9361](https://github.com/gfx-rs/wgpu/issues/9361) (merged/closed). On 2026-04-16 teoxoy said after the maintainer call the best option is to abort the process on OOM, not retry; PR has not been updated to do that. No test.
- **Move-along action:** ping author (implement abort per maintainer call) or close

#### [#9485](https://github.com/gfx-rs/wgpu/issues/9485): GLES external texture implementation ([@cmeissl](https://github.com/cmeissl)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-08-12.
- **State:** stalled
- **Waiting on:** author (teoxoy requested changes on 2026-06-24; inner-daemons confirmed "waiting on author" on 2026-08-07)
- **Last activity:** 2026-08-07, inner-daemons, pinged author. Author's last comment 2026-06-09. 91 days.
- **CI:** green but stale (all checks pass on a 2026-06-22 run; PR base is old, only codecov re-ran 2026-08-05)
- **Summary:** Adds GLES external texture via OES_EGL_image_external_essl3 (688 additions, 17 files, 3 commits). Open items: address teoxoy's change requests, add the `texture_from_raw_external_oes` helper (teoxoy approved the idea), docs for external sampler use, CHANGELOG.
- **Move-along action:** ping author (or close as inactive if no reply; closes [#4528](https://github.com/gfx-rs/wgpu/issues/4528) and is part of [#4386](https://github.com/gfx-rs/wgpu/issues/4386), so keep unless the author is gone)

#### [#9512](https://github.com/gfx-rs/wgpu/issues/9512): [core] Doc fixes related to mapping callback management ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-07-22.
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
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-09-02.
- **State:** stalled
- **Waiting on:** author (changes requested by inner-daemons 2026-06-15 and andyleiserson 2026-06-18; author said on 2026-07-22 they would revise "this week")
- **Last activity:** 2026-08-12, inner-daemons, pinged author again. Author's last comment 2026-07-22. 86 days.
- **CI:** green but stale (checks from 2026-05-26 pass; only codecov re-ran 2026-08-05)
- **Summary:** Adds `Features::SUBGROUP_SIZE_CONTROL` (Vulkan only) and a `SubgroupSize` enum (Varying/Full/Fixed) on passthrough entry points, with validation and a test. 688 additions, 20 files. Missing: trimming of long comments, review fixes, naga side. Also aligns with the gpuweb proposal [#5545](https://github.com/gfx-rs/wgpu/issues/5545).
- **Move-along action:** ping author once more; close if no reply (long-running, already pinged twice)

#### [#9659](https://github.com/gfx-rs/wgpu/issues/9659): [metal] Implement MULTI_DRAW_INDIRECT_COUNT via compute shader emulation ([@Bromles](https://github.com/Bromles)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-07-21.
- **State:** superseded (by [#9679](https://github.com/gfx-rs/wgpu/issues/9679))
- **Waiting on:** maintainer (decision: close, or keep as fallback until [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands)
- **Last activity:** 2026-06-15, cwfitzgerald, moved to draft until [#9679](https://github.com/gfx-rs/wgpu/issues/9679) lands; author agreed. 144 days.
- **CI:** not checked (stale base)
- **Summary:** Compute-shader emulation of count draws on Metal (1189 additions, 19 files). [#9679](https://github.com/gfx-rs/wgpu/issues/9679) (non-draft, still open, updated 2026-11-05, assigned to you and inner-daemons) does the same with Metal ICBs, and keeps the count on the GPU. You said you expect to take [#9679](https://github.com/gfx-rs/wgpu/issues/9679).
- **Move-along action:** close (superseded by [#9679](https://github.com/gfx-rs/wgpu/issues/9679), thank the author) once [#9679](https://github.com/gfx-rs/wgpu/issues/9679) is accepted; keep open only until then

#### [#9794](https://github.com/gfx-rs/wgpu/issues/9794): Experiment: Port VMA/D3D12MA to Rust in place of existing allocator stack ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (5 failed); last update 2026-08-30.
- **State:** early / experiment
- **Waiting on:** author (needs their time to review and iterate; no reviewer assigned)
- **Last activity:** 2026-08-18, cwfitzgerald, said they still plan to merge it ("I just need to spend time on it"). 80 days. Last update to the PR 2026-08-30 (66 days).
- **CI:** none found in this read (not fetched). They said earlier it passes all tests.
- **Summary:** LLM-assisted Rust port of VMA/D3D12MA to replace gpu-allocator. 12k lines added, 41 files. Body says "Don't merge this".

#### [#9809](https://github.com/gfx-rs/wgpu/issues/9809): Make `WriteOnly` use only volatile writes ([@kpreid](https://github.com/kpreid)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI red (45 failed); last update 2026-10-09.
- **State:** early / experiment
- **Waiting on:** maintainer (decision: do we want volatile writes in `WriteOnly` at all, given the perf cost and that const-ness depends on an unstable Rust feature, [rust-lang/rust#159094](https://github.com/rust-lang/rust/issues/159094)?)
- **Last activity:** 2026-10-09, kpreid, rebased and restored `const fn`; says "still not sure what we want to do with this PR". 28 days.
- **CI:** red, likely unrelated (nearly all jobs fail, including clippy on all targets, MSRV, doctest, examples; only Test Linux and Test Mac pass). Run is 2026-10-09. Not diagnosed; may be the unstable `const` volatile feature or CI-wide breakage.
- **Summary:** Replaces pointer writes in `WriteOnly` with `write_volatile` for soundness on write-combining memory (part of [#8897](https://github.com/gfx-rs/wgpu/issues/8897)). Small (24 additions, 50 deletions). Cannot merge until the const volatile feature is stable, and no benchmarks exist.
- **Move-along action:** decide whether to merge without `const fn` (as in the first version) or wait for the Rust feature

#### [#9843](https://github.com/gfx-rs/wgpu/issues/9843): Experimental AI slop naga cli rewrite ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-11-01.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated (probably a push). 5 days.
- **CI:** unknown (legacy status pending, check runs not read). Likely not meaningful.
- **Summary:** Full naga CLI rewrite, 105 files, +7206/-1379, 65 commits. Description says it is fully vibe coded and only open to view the diff on GitHub while the author tests the CLI locally. Claims to close [#9065](https://github.com/gfx-rs/wgpu/issues/9065) and [#8867](https://github.com/gfx-rs/wgpu/issues/8867). Checklist all unchecked.
- **Move-along action:** nothing (experiment; author is active). If it never becomes a real series, ask author to close it once the local experiment is done.

#### [#9853](https://github.com/gfx-rs/wgpu/issues/9853): Upstream Bindless Implementation ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (8 failed); last update 2026-08-30.
- **State:** in progress
- **Waiting on:** author. Open: mutable bindings, WGSL bounds checking, tests ("TODO"), full human audit.
- **Last activity:** 2026-08-30 (PR updated, actor not checked). 68 days.
- **CI:** not fetched.
- **Summary:** Partial implementation of the WebGPU bindless proposal with Resource Tables. Closes [#8557](https://github.com/gfx-rs/wgpu/issues/8557). 231 files, 12k lines.

#### [#9863](https://github.com/gfx-rs/wgpu/issues/9863): WIP wgpu-hal: Use `wayland-egl` instead of `wayland-sys` ([@ids1024](https://github.com/ids1024)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (5 failed); last update 2026-07-13.
- **State:** early / experiment
- **Waiting on:** other (upstream wayland-rs release, [Smithay/wayland-rs#900](https://github.com/Smithay/wayland-rs/issues/900))
- **Last activity:** 2026-07-13, ids1024, opened. No comments or reviews. 116 days.
- **CI:** not checked
- **Summary:** Swaps `wayland-sys` for the `wayland-egl` crate in wgpu-hal (62 additions, 3 files). Depends on the next wayland-rs release. Checklist is empty.
- **Move-along action:** nothing (blocked upstream; check whether wayland-rs has released and ask the author if still wanted)

#### [#9888](https://github.com/gfx-rs/wgpu/issues/9888): Add an example that provokes VUID-vkAcquireNextImageKHR-semaphore-01286 ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-07-29.
- **State:** stalled
- **Waiting on:** maintainer (decision: do we merge bug-repro examples that CI cannot run?)
- **Last activity:** 2026-07-29, kpreid, review comment. 100 days.
- **CI:** not checked
- **Summary:** Adds a Vulkan example (261 lines, 3 files) that shows a validation-layer error. ErichDonGubler requested changes (use a regression test or an issue MRE). jimblandy agreed it needs to become a test. inner-daemons argued the example form is right because surface/swapchain cases cannot run in CI.
- **Move-along action:** decide the policy on repro examples; then tell the author, or close (the underlying bug may be tracked elsewhere)

#### [#9894](https://github.com/gfx-rs/wgpu/issues/9894): Expect pass: `wgpu_gpu::clear_texture::clear_texture_uncompressed` ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI red (3 failed); last update 2026-07-22.
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
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-07-17.
- **State:** obsolete
- **Waiting on:** author (nothing needed)
- **Last activity:** 2026-07-17, nicoburns, opened. No comments. 112 days.
- **CI:** not checked
- **Summary:** Repro example for issue [#9896](https://github.com/gfx-rs/wgpu/issues/9896) (Metal advertised PostMultiplied instead of PreMultiplied alpha). [#9896](https://github.com/gfx-rs/wgpu/issues/9896) is closed (2026-08-18), fixed by merged [#9922](https://github.com/gfx-rs/wgpu/issues/9922). The example has no further use.
- **Move-along action:** close (repro for a fixed, closed issue; marked DO NOT MERGE)

#### [#9909](https://github.com/gfx-rs/wgpu/issues/9909): [hal vulkan] Clean up Vulkan Validation Layer callback ([@jimblandy](https://github.com/jimblandy)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI red (34 failed); last update 2026-07-29.
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
- **State (2026-11-14):** draft; mergeable; CI 1 cancelled jobs; last update 2026-08-05.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-08-05, ErichDonGubler, push (93 days). No comments or reviews.
- **CI:** stale (2026-08-05). All green except "Other Windows x86_64", which was cancelled after the 6 hour timeout.
- **Summary:** Adds 3 lines to one file to enable these CTS cases. The description says macOS CTS "Other" takes about 30 min and the Windows run is "massively long" and times out at 6 h. Coupled to issue [#9974](https://github.com/gfx-rs/wgpu/issues/9974) and motivated by [#9951](https://github.com/gfx-rs/wgpu/issues/9951).
- **Move-along action:** decide whether the CTS time budget allows this (it blocks on CI cost); if not, close. Otherwise ping author.

#### [#10008](https://github.com/gfx-rs/wgpu/issues/10008): Add CTS coverage for `maxStorage{Buffer,Texture}sIn{Vertex,Fragment}Stage` limits ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-09-07.
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
- **State (2026-11-14):** draft; mergeable; CI red (4 failed); last update 2026-09-01.
- **State:** early / experiment
- **Waiting on:** other reviewer (andyleiserson, per the description: "Will bring out of draft after discussing with [@andyleiserson](https://github.com/andyleiserson)")
- **Last activity:** 2026-09-01, ErichDonGubler, push (66 days). No comments or reviews.
- **CI:** red, likely related. Test Linux/Windows/Mac and "Validate: SPIR-V/GLSL/DOT/WGSL" failed on 2026-09-01; I did not read the logs. The PR is stacked on [#10026](https://github.com/gfx-rs/wgpu/issues/10026) (naga validation).
- **Summary:** One-file naga experiment (+22/-35) built on top of [#10026](https://github.com/gfx-rs/wgpu/issues/10026) in reply to a review thread there. Author is not sure it is useful.
- **Move-along action:** close (stale experiment; check first whether [#10026](https://github.com/gfx-rs/wgpu/issues/10026) has landed, then ask author to close or rebase).

#### [#10212](https://github.com/gfx-rs/wgpu/issues/10212): Record a pass's first use of a texture subresource, not its last ([@cwfitzgerald](https://github.com/cwfitzgerald))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** ready; mergeable; CI red (1 failed); last update 2026-11-13.
- **Verdict:** Status note
- **Waiting on:** author. The PR is ready for review but CI is red because of its own new test.
- **Last activity:** 2026-11-13, cwfitzgerald, push of the rebased commit. 0 days.
- **Approvals:** none (review state REVIEW_REQUIRED)
- **CI:** red (1 fail, 65 pass). `Test Linux x86_64` fails in `wgpu_gpu::regression::issue_10199::whole_texture_first_use` on GL/llvmpipe. Cause: `create_bind_group` for the `write_layer` bind group uses a D2 view, but the texture has `texture_binding_view_dimension = D2Array`. This is a bug in the new test and not infrastructure. The old failure "Check changelog for errors" is gone (not in the failed list). Mergeable.
- **Summary:** Fixes [#10199](https://github.com/gfx-rs/wgpu/issues/10199). The pass tracker merge now records the first use of a subresource as the start state and not the last. This gives correct Vulkan layouts. 4 files: `wgpu-core/src/track/texture.rs`, the changelog and a new regression test.
- **Findings:** none (status only)
- **To land:** fix the new test so it uses a D2Array view on GL, or skip the test on downlevel GL; wait for green CI; get a review.
- **Move-along action:** Fix the bind group view dimension in `issue_10199.rs` for GL, then request review.
- **Depth:** status only

#### [#10282](https://github.com/gfx-rs/wgpu/issues/10282): Change WebGPU backend panics to error callback insertion ([@Wumpf](https://github.com/Wumpf)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (1 failed); last update 2026-09-08.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-09-08, Wumpf, opened (59 days). No comments or reviews.
- **CI:** red, probably unrelated to logic: only "Format & Typos" failed; all other jobs green (2026-09-08).
- **Summary:** Replaces panics in the WebGPU backend with synthetic errors injected through the error scope and uncaptured-error callbacks (+1076/-222, 5 files). Would fix [#8005](https://github.com/gfx-rs/wgpu/issues/8005) and [#9710](https://github.com/gfx-rs/wgpu/issues/9710). The description's TODO list is open: testing ("WIP"), commit splitting, self-review, and whether the poisoning always works.
- **Move-along action:** ping author (status after 59 days; still wanted?). Maintainer decision later: is synthetic-error injection an accepted design?

#### [#10287](https://github.com/gfx-rs/wgpu/issues/10287): [metal] correct ray query semantics and BLAS transform handling ([@acoliver](https://github.com/acoliver)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI no runs; last update 2026-09-09.
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
- **State (2026-11-14):** draft; merge state not computed; CI no runs; last update 2026-09-09.
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
- **State (2026-11-14):** draft; merge state not computed; CI no runs; last update 2026-11-05.
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
- **State (2026-11-14):** draft; merge state not computed; CI no runs; last update 2026-09-09.
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
- **State (2026-11-14):** draft; merge state not computed; CI red (50 failed); last update 2026-10-28.
- **State:** close to ready (as a proposal; the author will split out the actual change)
- **Waiting on:** author (split the infrastructure into a separate PR, per jimblandy's comment on 2026-10-22)
- **Last activity:** 2026-10-22, jimblandy, "I think we should go with this" and said andyleiserson will split it out (15 days). Author's last push 2026-10-28 (9 days).
- **CI:** red (2026-10-28), nearly all jobs failed within 1-3 minutes, which looks like a build/config failure from a rebase; I did not read logs. Format & Typos also failed.
- **Summary:** Proposes an extension API for passing backend (hal) data through wgpu-core, to be used first for a DX12 adapter LUID filter wanted by Firefox ([#9932](https://github.com/gfx-rs/wgpu/issues/9932), [#10011](https://github.com/gfx-rs/wgpu/issues/10011)). Later commits show other uses (swapchain hooks, present timing, device-creation callbacks, `create_texture_from_raw`). Maintainers agreed on the direction in the maintainers meeting and in comments. Open design questions: `Any` needing `'static`, naming, serde compatibility.
- **Move-along action:** nothing (author is active and has agreed to split it). Maintainer may state final approval on the direction if not already done.

#### [#10354](https://github.com/gfx-rs/wgpu/issues/10354): Add fern log capture and test shader debugPrintf output ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** draft; merge state not computed; CI red (1 failed); last update 2026-11-13.
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
- **State (2026-11-14):** draft; merge state not computed; CI red (19 failed); last update 2026-11-01.
- **State:** in progress (marked experimental by the author)
- **Waiting on:** author (and [#10409](https://github.com/gfx-rs/wgpu/issues/10409) must land first)
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** red (partial view: Clippy Linux x86_64 and custom-backend failed; cause not checked, may be caused by changes in this PR).
- **Summary:** Adds a naga inlining pass with Never/Always/SPIR-V strategies; the SPIR-V one inlines only to enable unrestricted pointer parameters. Closes [#5158](https://github.com/gfx-rs/wgpu/issues/5158). Stacked on [#10409](https://github.com/gfx-rs/wgpu/issues/10409). 44 files, +4852. Open design questions in the description: whether to gate behind an instance flag, and whether `Always` helps driver compile time. Testing section empty, self-review unchecked.
- **Move-along action:** decide whether an inlining pass with a strategy option (and possibly an instance flag) is a direction you want, so the author can finish it. Review after [#10409](https://github.com/gfx-rs/wgpu/issues/10409).

#### [#10396](https://github.com/gfx-rs/wgpu/issues/10396): Update repository MSRV values together with Renovate ([@cwfitzgerald](https://github.com/cwfitzgerald)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-09-18.
- **State:** early / experiment
- **Waiting on:** author. Open: the description is still the empty template and checklist is unchecked. Merge state is "blocked".
- **Last activity:** 2026-09-18, opened (3 files, 244 lines). 49 days.
- **CI:** not fetched.
- **Summary:** Makes Renovate update the repo MSRV values together.

#### [#10403](https://github.com/gfx-rs/wgpu/issues/10403): gles: Support `DownlevelFlags::DEPTH_BIAS_CLAMP` ([@evilpie](https://github.com/evilpie)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI red (43 failed); last update 2026-10-24.
- **State:** stalled
- **Waiting on:** other (upstream: new `glow` release with [grovesNL/glow#371](https://github.com/grovesNL/glow/issues/371), and [wasm-bindgen#5336](https://github.com/wasm-bindgen/wasm-bindgen/issues/5336)). Not a maintainer decision.
- **Last activity:** 2026-10-24, evilpie: "Just waiting for a new release of `glow` now." 13 days.
- **CI:** red and stale. Last run 2026-09-19. Most jobs (clippy, MSRV, tests) failed within about one minute, most likely because of the unreleased glow/wasm-bindgen dependencies. Not verified. Base is old.
- **Summary:** Small change (3 files, +19/-1) that exposes depth bias clamp on GLES. Description is the empty template. No review so far.
- **Move-along action:** nothing (blocked upstream). Optionally ask whether glow 371 is on a release path.

#### [#10409](https://github.com/gfx-rs/wgpu/issues/10409): Rework naga's compaction pass to also reorder expressions ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI 10 cancelled jobs; last update 2026-11-01.
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

#### [#10751](https://github.com/gfx-rs/wgpu/issues/10751): Update contributor policy ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-10-27.
- **State:** in progress
- **Waiting on:** author
- **Last activity:** 2026-10-27, inner-daemons, replied to ErichDonGubler's review comments. 10 days.
- **CI:** not checked (docs-only change, 4 files, +10/-9).
- **Summary:** Proposes stricter LLM wording in CONTRIBUTING.md, AGENTS.md and the PR template; adds a note that agents must record that this guidance was followed. Author says they do not need it to land, only wants to put out the idea. Three open threads from ErichDonGubler (wording "agents", backticks on AGENTS.md, template checkbox for translation); author agreed on the first, offered an alternative on the third. No push since.
- **Move-along action:** decide whether you want this policy change (tougher LLM rules are a maintainer-level decision). Then ping author to finish or close.

#### [#10774](https://github.com/gfx-rs/wgpu/issues/10774): [naga] Implement vec2 <-> u32 bitcasts ([@DouglasDwyer](https://github.com/DouglasDwyer)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI no runs; last update 2026-10-28.
- **State:** stalled
- **Waiting on:** author
- **Last activity:** 2026-10-28, DouglasDwyer, PR created. 9 days.
- **CI:** not checked. PR has merge conflicts. Depends on [#10613](https://github.com/gfx-rs/wgpu/issues/10613) (not checked).
- **Summary:** Adds i16/u16 vector to 32-bit bitcasts in frontends and backends (GLSL returns an error). 78 files, +2336/-653. Related to [#8896](https://github.com/gfx-rs/wgpu/issues/8896). Description has template text left in, "self-reviewed" and "description has enough context" unchecked. No comments or reviews.
- **Move-along action:** ping author (rebase after [#10613](https://github.com/gfx-rs/wgpu/issues/10613), finish description and self-review); close if no answer.

#### [#10783](https://github.com/gfx-rs/wgpu/issues/10783): Cover limit checks for workgroup storage size per-variable rounding in CTS ([@ErichDonGubler](https://github.com/ErichDonGubler)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (3 failed); last update 2026-11-05.
- **State:** stalled (blocked on upstream)
- **Waiting on:** other (gpuweb/cts PR 4723 merge and a CTS revendor), then author
- **Last activity:** 2026-10-30, ErichDonGubler, comment that CI failures are expected. 7 days. (PR updated 2026-11-05, probably a CI re-run or push.)
- **CI:** red, expected per author. Partial view (8 of 69 runs) showed passes. PR has merge conflicts.
- **Summary:** Adds CTS coverage for [#9732](https://github.com/gfx-rs/wgpu/issues/9732). 5 files, +48/-10. The new CTS path does not exist in the vendored CTS yet.
- **Move-along action:** nothing until the CTS PR lands; then ping author to revendor and rebase.

#### [#10790](https://github.com/gfx-rs/wgpu/issues/10790): [d3d12] Use the driver-reported size for committed texture allocations ([@navkagleb](https://github.com/navkagleb)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI 10 cancelled jobs; last update 2026-10-30.
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
- **State (2026-11-14):** draft; mergeable; CI 10 cancelled jobs; last update 2026-11-01.
- **State:** in progress
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** not checked.
- **Summary:** Adds doc modules for features such as volatile/coherent decorations. Docs only, 7 files, +107. Works toward [#9647](https://github.com/gfx-rs/wgpu/issues/9647). Author plans to write the docs by hand and check with Claude before undrafting. Multiview fixes were split to [#10804](https://github.com/gfx-rs/wgpu/issues/10804).
- **Move-along action:** nothing (author is active).

#### [#10804](https://github.com/gfx-rs/wgpu/issues/10804): Multiview fixes ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (9 failed); last update 2026-11-01.
- **State:** in progress
- **Waiting on:** author
- **Last activity:** 2026-11-01, inner-daemons, PR updated. 5 days.
- **CI:** not checked. PR has merge conflicts (mergeable_state dirty).
- **Summary:** Claude-written fixes for multiview on GLES and Metal and descriptor field name differences. 39 files, +646/-123, 8 commits, "tests existing". Opened 2026-11-01 as a split from [#10803](https://github.com/gfx-rs/wgpu/issues/10803). Needs a rebase. Touches wgpu-core/hal, so it needs careful review and a changelog check, and it is large for one PR.
- **Move-along action:** ping author to rebase and say whether to undraft; consider asking to split per backend.

#### [#10806](https://github.com/gfx-rs/wgpu/issues/10806): Passthrough shaders rework master ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; mergeable; CI red (1 failed); last update 2026-11-02.
- **State:** early / experiment
- **Waiting on:** author
- **Last activity:** 2026-11-02, inner-daemons, PR updated. 4 days.
- **CI:** not checked.
- **Summary:** Author says it is a "don't merge" master branch for passthrough shader reworks, with parts to be split off; it is all LLM generated with lower quality standards. 5 files, +563/-2. Description has a broken "Closes" line (closes [#9052](https://github.com/gfx-rs/wgpu/issues/9052)).
- **Move-along action:** nothing. Review only the split-off PRs when they appear.

#### [#10811](https://github.com/gfx-rs/wgpu/issues/10811): [naga msl-out] Fix clip distances in mesh shader vertex outputs ([@inner-daemons](https://github.com/inner-daemons)) [draft]
- **Reviewed:** 2026-11-06
- **State (2026-11-14):** draft; has conflicts; CI red (1 failed); last update 2026-11-02.
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
- **State (2026-11-14):** draft; has conflicts; CI green; last update 2026-11-05.
- **State:** in progress
- **Waiting on:** maintainer (decision: is the scope wanted?). The body is one sentence about retries, but the diff adds a new checksum-pinned `download-release` action with a Python script, a Python test file run in CI, curl timeouts, and `max_checks_retries: 1` in the Mergify queue rules. The author says it may not fix the flakiness.
- **Last activity:** 2026-11-05, matthargett, pushed (5 commits). 1 day. No comments or reviews.
- **CI:** green.
- **Summary:** Makes CI downloads (DXC, Mesa, Vulkan SDK, WARP, Agility SDK, Dota2 shaders) fail hard and retry, and retries merge queue checks once. 11 files, +385/-18.
- **Move-along action:** decide whether to accept the new action and the Mergify retry change, or ask for a smaller PR. Review if yes.

Problems: no rate-limit errors. I did not fetch CI for [#9794](https://github.com/gfx-rs/wgpu/issues/9794), [#9853](https://github.com/gfx-rs/wgpu/issues/9853), [#10354](https://github.com/gfx-rs/wgpu/issues/10354) and [#10396](https://github.com/gfx-rs/wgpu/issues/10396), and I did not read the comment or review timelines for [#9853](https://github.com/gfx-rs/wgpu/issues/9853) and [#10396](https://github.com/gfx-rs/wgpu/issues/10396) (the first call showed no comments count; [#9853](https://github.com/gfx-rs/wgpu/issues/9853) has no comment count and [#10396](https://github.com/gfx-rs/wgpu/issues/10396) none). "Last activity" for [#9853](https://github.com/gfx-rs/wgpu/issues/9853) and [#10354](https://github.com/gfx-rs/wgpu/issues/10354) is the PR updated_at time, not a confirmed human action.

---

#### [#10928](https://github.com/gfx-rs/wgpu/issues/10928): fix(naga/spv): separate workgroup and buffer aggregate layouts ([@5andr0](https://github.com/5andr0))
- **Reviewed:** 2026-11-13
- **State (2026-11-14):** draft; mergeable; CI green; last update 2026-11-12.
- **Verdict:** Needs a bit of work
- **Waiting on:** a human owner. The PR is a draft. inner-daemons (2026-11-12) asked 5andr0 to review it himself and write the description by hand before he undrafts it. 5andr0 (2026-11-13) declined.
- **Last activity:** 2026-11-12, 5andr0, opened the PR. 1 day.
- **Approvals:** none
- **CI:** green (66 pass, 3 skipped); mergeable
- **Summary:** This is a rework of [#9295](https://github.com/gfx-rs/wgpu/issues/9295). Types that only Workgroup/TaskPayload variables use are declared without `Offset`/`ArrayStride`/`MatrixStride`/`ColMajor` on all SPIR-V versions (VUID-StandaloneSpirv-None-10684). Types that a buffer also uses get an undecorated copy. Loads and stores convert between the two: `OpCopyLogical` on 1.4+, and below 1.4 struct extract/construct plus a cached loop function for arrays. The copies are declared inside the type-arena loop, so later `ptr<workgroup, S>` arena types point to the copy.
- **Findings:**
  1. [medium, reasoned from code, not run] `naga/src/back/spv/mesh_shader.rs:91,114,127` The mesh output wrapper still uses `get_handle_type_id` for the vertex/primitive array types. If one of those types is also used by a buffer, `mesh_output` has the copy type, and the `OpAccessChain` result types are wrong. This is the same bug as [#9295](https://github.com/gfx-rs/wgpu/issues/9295) deeper-review answer 2, and the file is unchanged. The trigger is rare (it needs a buffer that shares the mesh output array type).
  2. [medium] No test covers a `ptr<workgroup, S>` function parameter with a shared `S`. The fix for the [#9295](https://github.com/gfx-rs/wgpu/issues/9295) pointer-order bug (`writer.rs:4390`, and `type_id_for_space` in `localtype_from_inner`) looks correct from the code. But the fixture `workgroup-shared-layout.wgsl` passes `Buffer` only by value. Add the minimal WGSL from the [#9295](https://github.com/gfx-rs/wgpu/issues/9295) deeper review as a test.
  3. [low] `naga/src/back/spv/writer.rs:2627-2850` The path below SPIR-V 1.4 (struct rebuild plus a hand-built SPIR-V loop function, about 200 lines) is needed only when a type is shared between a buffer and workgroup memory on a target below 1.4. wgpu-hal selects 1.5+ on Vulkan 1.2+. The maintainer should decide if this path is worth its complexity.
  4. [low] `naga/tests/out/spv/wgsl-workgroup-var-init.spvasm` When only a member of a workgroup struct is shared, the default (undecorated) struct declaration is still emitted and is not used. This is harmless, but it adds dead types.
  5. [low] Process: the description is agent-written, and the box "self-reviewed and fully understand" is unchecked. The body says "This draft makes no human attestation". The author says that he will not do the human part. CHANGELOG entry: present.
- **To land:** A human owner (kvark as co-author, or a maintainer) adopts the commit. Fix the mesh wrapper to use `type_id_for_space`/`workgroup_type_ids`. Add the `ptr<workgroup, S>` parameter test. Decide whether to keep the pre-1.4 copy path. Rewrite the description by hand and undraft.
- **Move-along action:** ask kvark if he will move [#10928](https://github.com/gfx-rs/wgpu/issues/10928)'s commit into [#9295](https://github.com/gfx-rs/wgpu/issues/9295) (or take over [#10928](https://github.com/gfx-rs/wgpu/issues/10928)) and close the other PR.
- **Depth:** full for `naga/src/back/spv` and the new tests. I spot-checked the snapshot churn (`workgroup-var-init`, `atomicOps`). I did not read every changed `.spvasm`.

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
