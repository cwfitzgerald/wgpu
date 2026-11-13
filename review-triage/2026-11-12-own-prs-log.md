# Own PR readiness pass, 2026-11-12: execution log

Scope: the open non-draft PRs by cwfitzgerald in gfx-rs/wgpu, plus draft #10354 (added by request). Nothing is pushed and nothing is posted on GitHub.

## Setup

- `trunk` is at `c92258fc` (Update Cargo.lock, #10889).
- Problem: the first full `jj git fetch` abandoned local commit `b49d1227` (Update Cargo.lock), which the resource table stack uses as its base. 26 commits became conflicted. Fix: `jj op restore e4ac6503da38`. All later jj commands use `--config git.abandon-unreachable-commits=false`.
- Problem: a full fetch of `cwfitzgerald` moved local `cw/bindless-impl` to the remote position. Fix: restored again; fetched only the 7 PR branches with `--branch`.
- Problem: SSH agent refuses to sign most of the time (`communication with agent failed`). `upstream` trunk was fetched over HTTPS with `GIT_CONFIG_GLOBAL=/dev/null` into `refs/remotes/upstream/trunk`.
- Started to track `claude/wgpu-issue-10902-kj0xus` and `codex/debug-printf-tests` (necessary to make the commits mutable).
- Workspaces: `.worktrees/pr-10903`, `.worktrees/pr-8388`, `.worktrees/pr-10354`.

## Work items

| PR | State at start | Agent | Status |
|---|---|---|---|
| #10903 immediate data fix | conflicting | opus, workspace pr-10903 | running |
| #8388 DXGI swapchain on Vulkan | conflicting, changes requested | opus, workspace pr-8388 | running |
| #10354 fern log capture (draft) | conflicting | opus, workspace pr-10354 | running |
| #10930, #10923, #10869, #10212 | mergeable | opus, read only | running |

All four agents are finished. #10212 moved from "mergeable" to "needs rebase" and was done by the coordinator in workspace `pr-10212`.

## Results

Problem for all workspaces: no test binary links on this machine (`rust-lld: error: could not open 'atls.lib'`; the MSVC ATL component is not installed). Validation is `cargo fmt --check` and `cargo clippy --tests` only. No GPU test ran.

| PR | Local head (bookmark) | Result |
|---|---|---|
| #10930 | not changed | Ready. No comments. Optional: add the 3 "backported in" suffixes that the v30 branch has (#10175, #10286, #10507). |
| #10869 | not changed | Ready. No comments. |
| #10923 | not changed | No defect. sagudev's last comment has no answer; it asks for a choice between this PR and #10919. |
| #10212 | `5d0821af` | Did not compile on trunk (#8014 added `TextureDescriptor::texture_binding_view_dimension`). Rebased; 2 field lines squashed into the commit. |
| #10903 | `8e869c93` | Rebased. One conflict in `tests/tests/wgpu-gpu/immediates.rs`; the same new field was added in `create_target()`. No comments. |
| #10354 | `ab41ed79` | Rebased. Conflicts in `tests/src/init.rs` and `tests/src/lib.rs`; `Queue::submit` return type fix squashed. Extra commit `xqtlqqpl` (drop redundant log filters) above the bookmark. Not ready to leave draft: see findings. |
| #8388 | `b144ea3d` | Rebased. Conflicts in `deno_webgpu/lib.rs` and `CHANGELOG.md`; new `TextureDescriptor` field added in `vulkan/swapchain/dxgi.rs`. Five extra commits above the bookmark for review threads T24, T28-T32, T44, T45. |

### #10354 findings

1. Vulkan without validation layers: the debugPrintf test fails after about 10 s instead of a skip, because the adapter exposes `DEBUG_PRINTF` without the layers.
2. `capture_preserves_output_filter` panics under single-thread `cargo test`, because it installs a second global logger. It passes under nextest.
3. The PR description is stale (#9389 is merged; the testing section lists only Metal).

### #8388 open review threads (andyleiserson, changes requested)

- Decisions for the owner: T19 (RenderDoc docs), T20 (document the downgrade to Native), T22 (typed borrow in place of `transmute_copy`), T26 (`dxgi_sync.hlsl` refers to a test that does not exist), T35 (remove the unreachable swapchain reuse path), T36 (drop order of DLL holders and COM objects, a real bug), T37 (back buffer leak on partial failure), T38 (close NT handles after import).
- Lower priority: T21, T23, T27, T33, T34, T39, T40, T41, submit-failure deadlock.
- Reply only: T25, T42, T43.
- Old problem, not from the rebase: 4 intermediate commits give dead-code warnings with `wgpu-hal --no-default-features --features vulkan`.

## Extra commits above the bookmarks (not part of the pure rebase)

- #8388: `oqukxmln`, `yllokpyq`, `ouynuvlw`, `pouporwt`, `xrquovml`.
- #10354: `xqtlqqpl`.

## 2026-11-13: #8388 review fixes

Workspace `pr-8388`, stack `cw/dxgi-on-vulkan..@-`, tip `ploxzlxr`. The full commit list with the review entry of each commit is in `2026-11-12-pr-8388-review.md`.

### Implementation

| Batch | Commits | Scope |
|---|---|---|
| Batch 1 | 13 (`klmlpptx` .. `lszxlllw`) | E1-E20, R20, R23, R24 |
| Batch 2 | 12 (`nxxtwysq` .. `nlvmmort`) | R1-R27 items |
| Fix round | 14 (`trsqlrzl` .. `ploxzlxr`) | adversarial review findings, R4, R6, R15, R26, E20 |

- Batch 2 was merged onto batch 1. Two conflicts in `dxgi.rs`; the coordinator resolved them. One resolution lost the `\` line continuations of the `EnumAdapterByLuid` error log, so the message had two runs of spaces. Fixed in `qqqlxonk`.

### Binary size (B3)

Vulkan-only Windows app:

- Default release: 5,548,032 -> 5,650,944 bytes (+102,912, +1.85%).
- Size-optimized: 3,721,728 -> 3,778,560 bytes (+56,832, +1.53%).

About 43 KB `vulkan::swapchain::dxgi`, 22 KB `auxil::dxgi`, 5 KB `windows` bindings. Fixed cost; the linker cannot remove it. Owner decision: no cargo feature.

### Runtime

- Test binaries link now (the Visual Studio ATL component is installed).
- The real-window `wgpu_surface` test passes with Vulkan native, Vulkan `dxgi-hwnd`, Vulkan `dxgi-visual` and DX12, on an NVIDIA RTX 4070.
- `cube` and `hello_triangle` ran with both DXGI kinds and logged the DXGI interop swapchain.
- Not tested: Vulkan validation layers (not installed), AMD or Intel hardware, resize under load, the `.cso` blobs.

### Adversarial reviews

- Code review: no reachable synchronization or lifetime bug. 16 findings: 12 fixed as obvious in the fix round, 12 open decisions D1-D12 (in the review document, section D).
- Document review: 51 corrections to the review document; all applied, with line numbers derived again at `ploxzlxr`.

### Problems

- `atls.lib` link error: fixed by installing the Visual Studio ATL component.
- `rustup component list` from the repository root auto-installed toolchain 1.93.
- Agent calls need a second attempt, because a hook rewrites the prompt.

### Status

Nothing on #8388 is pushed. No replies are posted.
