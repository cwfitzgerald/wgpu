# Add `texture-formats-tier1` / `texture-formats-tier2` (wgpu#8122)

## EXECUTION STATUS (updated 2026-06-09 — KEEP THIS SECTION UP TO DATE)

**Instructions for the executing agent:** update this section after every commit you land (and whenever you deviate from the plan), so any future agent can resume from the plan alone. Record jj change IDs, deviations, and verification gaps.

**This plan lives in-tree** as `plan.md` at the repo root, in a `private:` commit at the tip of the stack together with `formats.md` (the authoritative capability table). That commit must never be part of the PR. Workflow:
- Land new work commits *before* the private commit so it stays at the tip: `jj new --insert-before <private-change-id>`, do the work, `jj commit -m "..."` (descendants, i.e. the private commit, auto-rebase).
- Update this plan by amending the private commit: `jj new <private-change-id>`, edit `plan.md`, `jj squash`.

### Landed commits (jj change IDs, stack order oldest→newest, on top of trunk `6fbbb0fb`)

1. `sktxqvuw` — **wgpu-types: add TEXTURE_FORMATS_TIER1/TIER2 features and Features::with_implied** (plan commit 1). Bits 1<<18/19, flags + full doc comments, `with_implied()` + `with_implied` unit test, limits.rs test-union uncomment, CHANGELOG entry under Added/New Features→General.
2. `xupryxvs` — **wgpu-core: auto-enable implied features at device creation** (plan commit 2). `create_device_and_queue` rebuilds the descriptor with `required_features.with_implied()` (one clone, flows to support check / experimental check / `adapter.open()` / Device); `debug_assert`s in `Adapter::new` (TIER2⇒TIER1, TIER1⇒RG11B10); new noop tests in `tests/tests/wgpu-validation/api/implied_features.rs` (registered in `api/mod.rs`).
3. `wzwwxqol` — **noop hal: report all texture format capabilities instead of none** (NOT in original plan; split out on user request). `noop::texture_format_capabilities` returns `Tfc::all()` (was `empty()`). Required because commit 4's adapter-caps-driven plane validation consults hal caps, and noop's empty caps broke every noop planar test. User explicitly chose `Tfc::all()` over guaranteed-table-derived caps.
4. `tvtlyxoz` — **wgpu-core: validate NV12/P010 plane formats against adapter caps, not feature gates** (plan commit 3). Both plane sites use the private feature-check-free `Device::get_texture_format_features`; create_texture_view gates on `texture.desc.format.is_multi_planar_format() && desc.range.aspect.to_plane().is_some()`. Dropped `TEXTURE_FORMAT_16BIT_NORM` from planar/zero-init/validation tests. Added negative test `planar_texture_plane_format_still_gated_standalone`. **Deviation:** `planar_texture_render_attachment_unsupported` was converted to `planar_texture_usages_follow_adapter_plane_caps` (asserts creation now *succeeds* on noop) — with all-caps noop, P010+RENDER_ATTACHMENT is valid; planar usage validation is adapter-driven by design.
5. `ytyvroxy` — **naga: replace STORAGE_TEXTURE_16BIT_NORM_FORMATS with TEXTURE_FORMATS_TIER1/TIER2 capabilities** (plan commit 4). Caps 1<<44/45 (bit 9 left reserved with comment); interface.rs `AddressSpace::Handle` arm restructured to destructure `ImageClass::Storage { format, access }` — tier1 gates the 23 formats, tier2 gates `LOAD|STORE` outside {r32uint,r32sint,r32float,r64uint}; all four backend `supported_capabilities()` lists updated; bridge sets TIER1 from `TIER1 | ADAPTER_SPECIFIC | 16BIT_NORM (transitional)` and TIER2 from `TIER2 | ADAPTER_SPECIFIC`; new test `texture_formats_tier_capabilities` in `naga/tests/naga/validation.rs`; CHANGELOG entry under Changes→naga.

### Deviations / discoveries the plan didn't anticipate

- **naga snapshot tests do NOT all run with `Capabilities::all()`** (memo §"Verified current state" was wrong): validation caps default to `Capabilities::default()` = `MULTISAMPLED_SHADING | CUBE_ARRAY_TEXTURES` unless the input's `.toml` sets `capabilities` (which *replaces*, not unions). Had to add `capabilities = "TEXTURE_FORMATS_TIER2"` to `naga/tests/in/wgsl/abstract-types-texture.toml` and new `naga/tests/in/glsl/images.toml` (rw storage textures). Future naga-affecting commits: grep `naga/tests/in` for newly-gated constructs.
- **Pre-existing test failure on this machine** (Windows, debug): `naga wgsl_errors::recursion_depth_template` aborts with stack overflow — verified failing on clean trunk; exclude with `-E 'not test(recursion_depth_template)'`.
- **CHANGELOG entries use the `#99999` placeholder** (2 so far: features entry + naga entry) — user must fill real PR number at PR-open time.
- `formats.md` (authoritative capability table) is tracked in the `private:` commit at the stack tip, alongside this plan. It must not leak into any PR commit.
- `.gpuconfig` regeneration: `cargo xtask test` regenerates it (needed once after adding feature bits; already done). `cargo xtask test -- -E '...'` does NOT pass filters to nextest — run `cargo nextest run -p wgpu-test --test wgpu-gpu -E '...'` directly after regen. 6 GPUs found locally (Vulkan/DX12: NVIDIA RTX 4060 Laptop, Intel RaptorLake-S, WARP, noop).
- The `InvalidSampleCount` error path and anything else consulting noop adapter caps now sees `Tfc::all()`.

### Next steps

Commit 5 (Vulkan hal detection) is next; commits 5–13 unstarted. Per-commit verification: `cargo clippy --workspace --all-targets` + `cargo nextest run` for touched crates + wgpu-validation suite; GPU tests via `cargo nextest run -p wgpu-test --test wgpu-gpu` on this machine's adapters for commits 5, 6, 9, 10. Commit between each step; terse commit messages; no co-authored-by.

---

## Context

WebGPU added two device features (gpuweb PRs #5160/#5213 tier1, #5226 tier2) that expand guaranteed texture-format capabilities. wgpu doesn't implement them; Firefox needs them (Bugzilla 1982218/1982451) and web authors are blocked (issue #8122 comments). The issue's intent is to **migrate away from the native-only `Features::TEXTURE_FORMAT_16BIT_NORM`** to these spec features. The authoritative capability table is `C:\Users\conno\Configura\wgpu\formats.md`. An abandoned WIP by ErichDonGubler (GitHub commit `296c548b`, predates the features.rs refactor) is reference material for doc comments only — its bit choices and deno enum edits are stale.

Delivery: **one PR, curated stack of small standalone commits** (jj). Each commit must compile and pass tests on its own. Be conservative and spec-exact throughout.

### Spec semantics (what each feature grants)

- **tier1** (implies `rg11b10ufloat-renderable` at requestDevice):
  - STORAGE ro/wo for: R8Unorm, R8Snorm, R8Uint, R8Sint, Rg8Unorm, Rg8Snorm, Rg8Uint, Rg8Sint, R16Uint, R16Sint, R16Float, Rg16Uint, Rg16Sint, Rg16Float, Rgb10a2Uint, Rgb10a2Unorm, Rg11b10Ufloat (17 formats)
  - RENDER_ATTACHMENT + blend + MSAA(x4) + resolve for: R8Snorm, Rg8Snorm, Rgba8Snorm
  - Gates availability of R16Unorm/R16Snorm/Rg16Unorm/Rg16Snorm/Rgba16Unorm/Rgba16Snorm with: sample type **unfilterable-float**, RENDER_ATTACHMENT + blend + MSAA(x4) (**no resolve**), storage ro/wo
- **tier2** (implies tier1): STORAGE_READ_WRITE for: R8Unorm, R8Uint, R8Sint, Rgba8Unorm, Rgba8Uint, Rgba8Sint, R16Uint, R16Sint, R16Float, Rgba16Uint, Rgba16Sint, Rgba16Float, Rgba32Uint, Rgba32Sint, Rgba32Float (15 formats; never rg32*)

### User-confirmed decisions

1. `TEXTURE_FORMAT_16BIT_NORM` is **fully removed** in this series (breaking). Norm16 guaranteed sample type becomes `unfilterable-float` per spec (breaking; filterable norm16 stays reachable via `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` — bind-group sample-type matching already has the adapter-specific FILTERABLE escape hatch at `wgpu-core/src/device/resource.rs:3605-3607`).
2. Implied features **auto-enable in wgpu-core** at device creation (spec-conformant for all frontends in one place).
3. Full scope: all 4 hal backends, limit-bucket enablement, CTS list updates, external-texture P010 relaxation.

---

## Commit series

Dependencies: 1 → {2,3,4,5–8,9,11}; 4 before 9; 5–8 before 10 and 12; 9 before 10; 10 before 13.

### Commit 1 — ✅ DONE (`sktxqvuw`) — wgpu-types: define `TEXTURE_FORMATS_TIER1/TIER2` + `with_implied()`

`wgpu-types/src/features.rs`:
- `webgpu_impl` module: `WEBGPU_FEATURE_TEXTURE_FORMATS_TIER1 = 1 << 18`, `..._TIER2 = 1 << 19` (FeaturesWebGPU currently tops out at `PRIMITIVE_INDEX = 1 << 17`; verify free bits before committing).
- New flags in the `FeaturesWebGPU` block after `PRIMITIVE_INDEX` (~line 1826) with `#[name("texture-formats-tier1")]` / `#[name("texture-formats-tier2")]` (no `wgpu-` alias — web features don't get one; see the `features_names` test ~line 1977). The `#[name]` string drives `as_str()`/`FromStr`, which is what deno_webgpu parses, so deno needs no enum work.
- Doc comments: adapt WIP `296c548b` lists, **corrected**: norm16 = unfilterable-float, MSAA x4, no resolve; fix variant casing (`Rgb10a2Uint`, `Rg11b10Ufloat`); document implied-features behavior and the 16BIT_NORM migration (incl. filterability caveat).
- Add `Features::with_implied()` (tier2 ⇒ tier1 ⇒ RG11B10UFLOAT_RENDERABLE) + unit test.
- Same commit: uncomment the prepared `.union(Features::TEXTURE_FORMATS_TIER{1,2})` lines in the `enumerate_webgpu_features` test (`wgpu-core/src/limits.rs:561-562`) — it fails as soon as the bits enter `all_webgpu_mask()`. CHANGELOG entry.

### Commit 2 — ✅ DONE (`xupryxvs`) — wgpu-core: implied features at device creation + adapter invariant

- `Adapter::create_device_and_queue` (`wgpu-core/src/instance.rs:~850`): expand `desc.required_features.with_implied()` before the support check (~856), experimental check (~864), `adapter.open()` (~901), and the features stored on the Device.
- `debug_assert!` in `Adapter::new` (`instance.rs:~699`): TIER1 ⇒ RG11B10UFLOAT_RENDERABLE, TIER2 ⇒ TIER1 (documents the hal contract).
- Tests via noop backend (noop reports `Features::all()`); document the native-visible effect (requesting TIER1 silently enables RG11B10UFLOAT_RENDERABLE) in docs + CHANGELOG.
- Browser backend needs nothing here (browser does its own expansion; `map_wgt_features(self.inner.features())` reads back).

### Commit 3 — ✅ DONE (as two commits: `wzwwxqol` noop hal Tfc::all() + `tvtlyxoz` plane validation; see status section) — wgpu-core: NV12/P010 plane format special-case (standalone #9119 follow-up)

Resolves teoxoy's concern: P010 planes are R16Unorm/Rg16Unorm, which after migration require TIER1 the app may not have enabled.
- **No new method needed**: the existing `Device::get_texture_format_features` (`wgpu-core/src/device/resource.rs:4935`, private wrapper over `wgpu_core::Adapter::get_texture_format_features` at `instance.rs:740`) is already feature-check-free. At the two plane sites — the create_texture per-plane loop (`resource.rs:1591`; the parent format was already feature-checked) and create_texture_view (`resource.rs:1799`, when `desc.range.aspect` is a plane of a multi-planar texture format; the `aspect_specific_format` equality at 1827-1836 pins the view format to the plane format) — call `self.get_texture_format_features(plane_format)` instead of `describe_format_features`.
- Adapter-caps-driven plane validation is the correct semantics here: NV12/P010 are native-extension formats with no spec-guaranteed table, the guaranteed table for norm16-without-tier1 would wrongly report `(none, basic)`, and teoxoy's contract is that backends only advertise NV12/P010 where the plane formats actually work.
- Prove standalone-ness by dropping `TEXTURE_FORMAT_16BIT_NORM` from planar tests now: `tests/tests/wgpu-validation/api/texture.rs:66/158/214/344`, `tests/tests/wgpu-gpu/planar_texture/mod.rs:269/591`, `tests/tests/wgpu-gpu/zero_init.rs:493/513`.

### Commit 4 — ✅ DONE (`ytyvroxy`) — naga: tier capabilities replacing `STORAGE_TEXTURE_16BIT_NORM_FORMATS`

WGSL-level gating is required, not just BGL checks — `cts_runner/fail.lst:130` documents `webgpu:shader,validation,extension,readonly_and_readwrite_storage_textures:*` failing on exactly this interaction.
- `naga/src/valid/mod.rs`: add `Capabilities::TEXTURE_FORMATS_TIER1 = 1 << 44`, `TEXTURE_FORMATS_TIER2 = 1 << 45` (highest existing bit is 43); **remove** `STORAGE_TEXTURE_16BIT_NORM_FORMATS = 1 << 9` (breaking naga change, changelogged).
- `naga/src/valid/interface.rs` (~1042-1060, follow the existing norm16 precedent): TIER1 gates the 23 storage texel formats (17 tier1-new + 6 norm16) on storage-texture globals; TIER2 gates `LOAD|STORE` (read_write) access on any format outside {R32Uint, R32Sint, R32Float, R64Uint} — deliberately looser than WGSL's exact tier2 list so naga stays a superset for `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` users (e.g. rw on rg32float); exact-list enforcement remains wgpu-core's BGL job. Document the residual conformance gap.
- No WGSL frontend changes (device features, not `enable` extensions).
- Backends: swap the removed cap for both new caps in all four `supported_capabilities()` lists (`msl/mod.rs:~900`, `hlsl/mod.rs:~796`, `spv/mod.rs:~1191`, `glsl/mod.rs:~655`).
- Bridge `wgpu-naga-bridge/src/lib.rs:75-78`: set each cap from `TEXTURE_FORMATS_TIERn | TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (the OR avoids regressing adapter-specific users whose shaders naga accepted unconditionally before); transitionally also OR `TEXTURE_FORMAT_16BIT_NORM` into TIER1 until commit 10.
- New validator unit tests in `naga/tests/naga/validation.rs` (no tests exist for the norm16 cap today). Snapshots run `Capabilities::all()` so no snapshot churn.

### Commits 5–8 — wgpu-hal: per-backend detection (one commit each)

Contract: a backend advertising a tier must report at least the tier-mandated `Tfc` bits in `texture_format_capabilities` for every listed format. All four backends verified to need **no Tfc changes** — only feature-bit detection. Compute `tier2 = tier1 && rw-condition` so the invariant holds by construction; fold RG11B10UFLOAT_RENDERABLE into the tier1 condition. Noop (reports `Features::all()`) and `dynamic` (pass-through) need nothing.

**Commit 5 — Vulkan** (`wgpu-hal/src/vulkan/adapter.rs`): new helpers beside `is_format_16bit_norm_supported` (~3127), wired into `PhysicalDeviceFeatures::to_wgpu` (~985), using the existing `supports_format()` (~3187) per-format-query style (deliberately not the `shaderStorageImageExtendedFormats` shortcut):
- tier1 storage: `FormatFeatureFlags::STORAGE_IMAGE` (optimal) on all 17 formats (`R8_UNORM…B10G11R11_UFLOAT_PACK32`). One bit covers ro+wo.
- snorm8 renderable: `COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND` on R8_SNORM/R8G8_SNORM/R8G8B8A8_SNORM + 4x sample check via `vkGetPhysicalDeviceImageFormatProperties` (pattern: `supports_astc_3d` ~3202). Resolve needs no extra check.
- norm16 full: existing six-format check **plus** `COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND` + the 4x check (keep `is_format_16bit_norm_supported` untouched until commit 10).
- tier2: `STORAGE_IMAGE` on the 15 rw formats (in practice tier2 == tier1 on Vulkan; check explicitly anyway).
- No SPIR-V plumbing: naga emits formatted storage reads only (`naga/src/back/spv/writer.rs:~2501` never requires `StorageImageReadWithoutFormat`), and `StorageImageExtendedFormats` is already declared unconditionally (`vulkan/adapter.rs:~2567`). Add a comment saying so.

**Commit 6 — DX12** (`wgpu-hal/src/dx12/adapter.rs`): add `get_format_support(device, DXGI_FORMAT) -> Option<D3D12_FEATURE_DATA_FORMAT_SUPPORT>` + `supports_msaa4` (MULTISAMPLE_QUALITY_LEVELS, count 4, quality != 0) helpers; refactor the bgra8unorm block (515-535) onto them; per-format loops in `expose()`:
- tier1 storage (17 DXGI formats): `Support1 ⊇ TYPED_UNORDERED_ACCESS_VIEW` and `Support2 ⊇ UAV_TYPED_LOAD | UAV_TYPED_STORE` — the critical runtime check; most of these are in D3D12's optional typed-UAV-load list.
- snorm8: `Support1 ⊇ RENDER_TARGET | BLENDABLE | MULTISAMPLE_RENDERTARGET | MULTISAMPLE_RESOLVE | MULTISAMPLE_LOAD` + msaa4. SNORM RT is optional per FL11 tables — runtime query is authoritative.
- norm16: `SHADER_LOAD | RENDER_TARGET | BLENDABLE | MULTISAMPLE_RENDERTARGET | MULTISAMPLE_LOAD` + UAV load/store + msaa4 (resolve not required). Note: the existing unconditional `TEXTURE_FORMAT_16BIT_NORM` at line 476 over-promises storage on paper; tier1 must not repeat that.
- tier2: UAV_TYPED_LOAD|STORE on the 15 rw formats. Keep the per-format loop as source of truth (optionally `debug_assert` agreement with `options.TypedUAVLoadAdditionalFormats`).

**Commit 7 — Metal** (`wgpu-hal/src/metal/adapter.rs`): in `CapabilitiesQuery::features()` near the RG11B10 line (~1251), entirely from existing fields:
- `tier1 = format_any8_snorm_all && format_rg11b10_all && format_rgb10a2_unorm_all && format_rgb10a2_uint_write` (≈ Apple3+/Mac; everything else in the tier1 table is unconditional per the Metal Feature Set Tables, and tfc already asserts it).
- `tier2 = tier1 && read_write_texture_tier ∉ {TierNone, Tier1}` (MTL rw Tier2's 18-format list = WebGPU tier2 list + r32*; tfc already maps it exactly).
- Cite the Feature Set Tables in comments. Before landing, double-check norm16 blend+MSAA universality on Apple1/Apple2 (exposure risk limited since tier1 needs `format_any8_snorm_all` anyway).

**Commit 8 — GLES** (`wgpu-hal/src/gles/adapter.rs`): comment-only — document near ~483 why the tiers are never advertised. ES 3.2 changes nothing vs 3.1: the image-load-store format list (Table 8.27) is identical (rgba32f/16f, r32f, rgba32ui/16ui/8ui, r32ui, rgba32i/16i/8i, r32i, rgba8, rgba8_snorm — none of tier1's added storage formats), and GLSL ES 3.20 §4.4.7 restricts images without readonly/writeonly to r32f/r32i/r32ui, so tier2 rw is impossible. GL_NV_image_formats adds the tier1 storage formats but doesn't lift the rw restriction; snorm8 isn't renderable without EXT_render_snorm modeling (nor guaranteed in core desktop GL).

### Commit 9 — wgpu-types: tier grants in the format table + GPU tests

`wgpu-types/src/texture/format.rs`:
- `guaranteed_format_features()` (914-1066): prelude lets next to the existing `rg11b10f_*`/`bgra8unorm_*` pattern — tier1 storage flag/usage additions, `(snorm8_f, snorm8_u) = tier1 ? (msaa_resolve, attachment) : (none, basic)`, norm16 tuple (transitionally keyed on `tier1 || 16bit_norm` until commit 10), tier2 `STORAGE_READ_WRITE` additions per the 15-format list. Apply per-arm deltas exactly per `formats.md`. Rgba8Snorm keeps its unconditional storage and gains attachment/blend/msaa/resolve under tier1.
- Blendable handling (~1045-1056): `is_blendable` derives from `sample_type(None,None) == float{filterable:true}`, which norm16 will stop satisfying in commit 10 — add an explicit clause mirroring the FLOAT32_BLENDABLE one now (`tier1 && matches!(self, norm16…)`).
- New tests `tests/tests/wgpu-gpu/texture_formats_tier1.rs` / `_tier2.rs` (patterns: `bgra8unorm_storage.rs`, `float32_filterable.rs`; register in `main.rs`): storage BGL accepted with tier1 / rejected without; snorm8 render+blend+resolve; norm16 attachment/blend/MSAA and resolve-rejection; shader-module gating (naga caps); rw storage compute readback for tier2; implied-features assertion on a real adapter.

### Commit 10 — Migration: remove `TEXTURE_FORMAT_16BIT_NORM`

Complete usage inventory (grep-verified):
- `wgpu-types/src/features.rs:633,649-650` definition (leave "bit 1 reserved" comment — feature bits are serialized) + `:1987` test list.
- `format.rs:875` `required_features()` norm16 arm → `TEXTURE_FORMATS_TIER1`; six variant doc comments (lines 146/150/176/180/226/230); `sample_type()` norm16 arm (1144-1149) → `unfilterable_float`; drop transitional keying from commit 9.
- External-texture P010 relaxation (`wgpu-core/src/device/resource.rs:2100-2110`): plane sample-type check currently requires `Float{filterable:true}`; accept any `Float{..}` when the plane belongs to a multi-planar texture (justified by hal contract: P010 only advertised where planes sample linearly). Flag for explicit reviewer sign-off.
- `wgpu-naga-bridge:76-78` drop the transitional OR; hal advertisements: vulkan:845-848 (+3127 helper), dx12:476, metal:1162, gles:489-492 (+448-481 detection) — replace with tier1-derived behavior or delete.
- `limits.rs:247-250` EXEMPT_FEATURES: drop the 16BIT_NORM union, rewrite the 242-246 comment (plane special-case from commit 3 is the resolution; do NOT exempt the tier features). Update the copied constant in `tests/tests/wgpu-validation/limit_buckets.rs:76-79`.
- CHANGELOG "Breaking" entry: removal + norm16 filterability downgrade + migration guidance (`TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`).

### Commit 11 — wgpu browser backend mapping

`wgpu/src/backend/webgpu.rs` `FEATURES_MAPPING` (~736): add `GpuFeatureName::TextureFormatsTier1/Tier2` (already in `gen_GpuFeatureName.rs:53-54`). deno_webgpu needs zero changes (FromStr-based; adapter masks with `all_webgpu_mask()`; implication inherited from core).

### Commit 12 — limits.rs buckets

Add TIER1+TIER2 to `UPLEVEL.features` (replacing the line-351 "not implemented" comment); hardware buckets inherit. **Gate on verification**: a device missing a bucket feature falls to the empty default bucket — verify with `cargo run -p wgpu-info` across hardware classes (your machine's Vulkan/DX12 adapters + WARP; coordinate Metal/M-series + llvmpipe via CI or maintainers). If a class fails (tier2 on llvmpipe/WARP most likely), demote to per-bucket placement like SHADER_F16.

### Commit 13 — CTS lists

`cts_runner/test.lst` / `fail.lst`: add tier capability-check suites (confirm exact selectors against the pinned CTS revision via `cargo xtask cts -- --list`, e.g. `webgpu:api,validation,capability_checks,features,texture_formats_tier1:*`); move `webgpu:shader,validation,extension,readonly_and_readwrite_storage_textures:*` out of `fail.lst:130` once it passes.

---

## Key risks / review-sensitive points

1. **Norm16 filterability downgrade** — headline breaking change (user-approved); changelog must spell out the adapter-specific-features migration path.
2. **External-texture P010 relaxation** encodes a hal contract — needs reviewer sign-off (flag in PR description).
3. **Implied features** are new machinery in wgpu-core; native-visible (extra features silently on). Spec-mandated; document prominently.
4. **naga public API break** (cap removal + renumber); residual WGSL rw-exactness gap (naga gates rw loosely; wgpu-core BGL enforces the exact list) is deliberate and documented.
5. **DX12/Vulkan detection** relies on optional per-format support — never assume feature-level guarantees; runtime queries are authoritative everywhere.
6. **Bucket additions** can demote devices to the empty fallback bucket — measure before landing commit 12.

## Verification

- Per commit: `cargo clippy --all-targets` + `cargo nextest run` (wgpu-types/wgpu-core/naga unit + validation tests run without GPU; noop backend covers feature plumbing).
- GPU tests: `cargo nextest run -p wgpu-test` (or `cargo xtask test`) on this machine's Vulkan + DX12 + WARP adapters — exercises commits 5, 6, 9, 10. Metal coverage via CI.
- CTS: `cargo xtask cts` with the tier suites after commit 13; confirm `readonly_and_readwrite_storage_textures` now passes.
- `cargo run -p wgpu-info` before/after commit 12 to confirm bucket assignment doesn't regress.
- cbindgen sanity for Firefox consumers: the new `WEBGPU_FEATURE_*` consts follow the existing plain-`pub const u64` pattern; no extra work expected, but diff `wgpu-types` cbindgen output if practical.

A more detailed frontend design memo (per-format table deltas, full rationale) from planning is at `C:\Users\conno\.claude\plans\following-the-lead-of-wise-naur-agent-a973ffba68103a917.md`.
