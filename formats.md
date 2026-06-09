Legend: ✓ = always; **t1** = requires `texture-formats-tier1`; **t2** = requires `texture-formats-tier2`; **core** = requires `core-features-and-limits`; **f32b** = requires `float32-blendable`; blank = unsupported. SampleType `ufloat` = `unfilterable-float`. ST cols = `STORAGE_BINDING` access modes (wo/ro/rw). Copy = texel block copy footprint (bytes); RT = render target pixel byte cost (bytes).

| Format | Req. | SampleType | RENDER_ATT | blend | MSAA | resolve | ST wo | ST ro | ST rw | Copy | RT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **8 bits/component (1-byte RT align)** | | | | | | | | | | | |
| r8unorm | | float, ufloat | ✓ | ✓ | ✓ | ✓ | t1 | t1 | t2 | 1 | 1 |
| r8snorm | | float, ufloat | t1 | t1 | t1 | t1 | t1 | t1 | | 1 | 1 |
| r8uint | | uint | ✓ | | core | | t1 | t1 | t2 | 1 | 1 |
| r8sint | | sint | ✓ | | core | | t1 | t1 | t2 | 1 | 1 |
| rg8unorm | | float, ufloat | ✓ | ✓ | ✓ | ✓ | t1 | t1 | | 2 | 2 |
| rg8snorm | | float, ufloat | t1 | t1 | t1 | t1 | t1 | t1 | | 2 | 2 |
| rg8uint | | uint | ✓ | | core | | t1 | t1 | | 2 | 2 |
| rg8sint | | sint | ✓ | | core | | t1 | t1 | | 2 | 2 |
| rgba8unorm | | float, ufloat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | t2 | 4 | 8 |
| rgba8unorm-srgb | | float, ufloat | ✓ | ✓ | ✓ | ✓ | | | | 4 | 8 |
| rgba8snorm | | float, ufloat | t1 | t1 | t1 | t1 | ✓ | ✓ | | 4 | 8 |
| rgba8uint | | uint | ✓ | | core | | ✓ | ✓ | t2 | 4 | 4 |
| rgba8sint | | sint | ✓ | | core | | ✓ | ✓ | t2 | 4 | 4 |
| bgra8unorm | | float, ufloat | ✓ | ✓ | ✓ | ✓ | `bgra8unorm-storage` | | | 4 | 8 |
| bgra8unorm-srgb | core | float, ufloat | ✓ | ✓ | ✓ | ✓ | | | | 4 | 8 |
| **16 bits/component (2-byte RT align)** | | | | | | | | | | | |
| r16unorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 2 | 2 |
| r16snorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 2 | 2 |
| r16uint | | uint | ✓ | | core | | t1 | t1 | t2 | 2 | 2 |
| r16sint | | sint | ✓ | | core | | t1 | t1 | t2 | 2 | 2 |
| r16float | | float, ufloat | ✓ | ✓ | ✓ | ✓ | t1 | t1 | t2 | 2 | 2 |
| rg16unorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 4 | 4 |
| rg16snorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 4 | 4 |
| rg16uint | | uint | ✓ | | core | | t1 | t1 | | 4 | 4 |
| rg16sint | | sint | ✓ | | core | | t1 | t1 | | 4 | 4 |
| rg16float | | float, ufloat | ✓ | ✓ | ✓ | ✓ | t1 | t1 | | 4 | 4 |
| rgba16unorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 8 | 8 |
| rgba16snorm | t1 | ufloat | ✓ | ✓ | ✓ | | ✓ | ✓ | | 8 | 8 |
| rgba16uint | | uint | ✓ | | core | | ✓ | ✓ | t2 | 8 | 8 |
| rgba16sint | | sint | ✓ | | core | | ✓ | ✓ | t2 | 8 | 8 |
| rgba16float | | float, ufloat | ✓ | ✓ | core | core | ✓ | ✓ | t2 | 8 | 8 |
| **32 bits/component (4-byte RT align)** | | | | | | | | | | | |
| r32uint | | uint | ✓ | | | | ✓ | ✓ | ✓ | 4 | 4 |
| r32sint | | sint | ✓ | | | | ✓ | ✓ | ✓ | 4 | 4 |
| r32float | | float (if `float32-filterable`), ufloat | ✓ | f32b | core | | ✓ | ✓ | ✓ | 4 | 4 |
| rg32uint | | uint | ✓ | | | | core | core | | 8 | 8 |
| rg32sint | | sint | ✓ | | | | core | core | | 8 | 8 |
| rg32float | | float (if `float32-filterable`), ufloat | ✓ | f32b | | | core | core | | 8 | 8 |
| rgba32uint | | uint | ✓ | | | | ✓ | ✓ | t2 | 16 | 16 |
| rgba32sint | | sint | ✓ | | | | ✓ | ✓ | t2 | 16 | 16 |
| rgba32float | | float (if `float32-filterable`), ufloat | ✓ | f32b | | | ✓ | ✓ | t2 | 16 | 16 |
| **mixed width, 32 bits/texel (4-byte RT align)** | | | | | | | | | | | |
| rgb10a2uint | | uint | ✓ | | core | | t1 | t1 | | 4 | 8 |
| rgb10a2unorm | | float, ufloat | ✓ | ✓ | ✓ | ✓ | t1 | t1 | | 4 | 8 |
| rg11b10ufloat | | float, ufloat | `rg11b10ufloat-renderable` | `rg11b10ufloat-renderable` | `rg11b10ufloat-renderable` | `rg11b10ufloat-renderable` | t1 | t1 | | 4 | 8 |

Notes: several empty cells in the source carry backend-implementation comments (`<!-- Metal -->` / `<!-- Vulkan -->`) marking where a capability happens to exist on a specific backend but isn't spec-guaranteed; those are rendered as blank (unsupported) here since they're HTML comments, not spec values.
