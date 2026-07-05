//! Emulates the CPU-side, single-threaded encoding work of one frame of an advanced
//! game renderer: a depth prepass, shadow cascade passes, a g-buffer pass, an SSAO
//! compute chain, a lighting pass, a transparency pass, a bloom downsample/upsample
//! chain, and a few post-processing passes.
//!
//! The frame is encoded in three configurations per run (see [`FrameConfig`]), so a
//! single invocation produces directly comparable numbers for bindful rendering with
//! per-object vertex buffers, bindful rendering with one shared vertex/index buffer,
//! and (when the device supports binding arrays) bindless materials with the shared
//! buffers.
//!
//! All resources are 1x1 and the shaders are trivial — the benchmark measures command
//! encoding, bind group / pipeline state tracking, pass setup/teardown, and submission,
//! not GPU work. By default it runs on the noop backend so that all of wgpu-core's
//! validation and tracking runs with zero GPU-driver variance; set `WGPU_BACKEND` (or
//! `WGPU_ADAPTER_NAME`) to run against a real backend like the other benchmarks.

use std::{
    num::{NonZeroU32, NonZeroU64},
    time::{Duration, Instant},
};

use nanorand::{Rng, WyRand};
use wgpu_benchmark::{iter_many, BenchmarkContext};

use crate::DeviceState;

/// Stride between per-object entries in the object uniform buffer; matches the
/// default `min_uniform_buffer_offset_alignment`.
const OBJECT_STRIDE: u32 = 256;
const TEXTURES_PER_MATERIAL: u32 = 4;
const VERTEX_BUFFERS_PER_DRAW: u32 = 2;
/// AO computation plus a horizontal and a vertical blur.
const SSAO_DISPATCHES: u32 = 3;
/// Opaque and alpha-tested variants for the depth prepass and shadow passes.
const DEPTH_ONLY_PIPELINES: u32 = 2;
/// Transparent draws are depth-sorted rather than state-sorted, so they switch
/// state much more often than opaque draws.
const TRANSPARENT_DRAWS_PER_PIPELINE: u32 = 4;

/// Scaling knobs for the frame benchmark.
///
/// Each knob can be overridden on the command line as
/// `--param frame.<knob>=<value>` with the knob name in kebab-case, e.g.
/// `cargo bench -p wgpu-benchmark -- "Frame" --param frame.opaque-draws=5000`.
#[derive(Clone, Copy)]
struct FrameParams {
    /// Opaque objects, each drawn once in the depth prepass and once in the g-buffer pass.
    opaque_draws: u32,
    /// Number of shadow cascade render passes.
    shadow_cascades: u32,
    /// Draws per shadow cascade, pulled from the opaque object pool.
    shadow_draws_per_cascade: u32,
    /// Transparent objects drawn in the forward transparency pass. These rebind their
    /// material every draw as if depth-sorted.
    transparent_draws: u32,
    /// Distinct material bind groups shared by opaque and transparent draws.
    materials: u32,
    /// Consecutive g-buffer draws sharing a material bind group before rebinding.
    draws_per_material: u32,
    /// Distinct g-buffer pipeline variants.
    gbuffer_pipelines: u32,
    /// Consecutive prepass/shadow/g-buffer draws sharing a pipeline before switching.
    draws_per_pipeline: u32,
    /// Distinct transparent pipeline variants.
    transparent_pipelines: u32,
    /// Mip levels in the bloom chain, producing `2 * mips - 1` render passes.
    bloom_mips: u32,
    /// Fullscreen post-processing passes (the first composites HDR + bloom), each with
    /// its own pipeline, ping-ponging between two targets.
    post_passes: u32,
}

impl FrameParams {
    const DEFAULT: Self = Self {
        opaque_draws: 2500,
        shadow_cascades: 4,
        shadow_draws_per_cascade: 1000,
        transparent_draws: 500,
        materials: 1000,
        draws_per_material: 3,
        gbuffer_pipelines: 32,
        draws_per_pipeline: 64,
        transparent_pipelines: 8,
        bloom_mips: 6,
        post_passes: 4,
    };

    /// A very lightweight configuration so test mode just checks that the
    /// benchmark does not break.
    const TEST: Self = Self {
        opaque_draws: 8,
        shadow_cascades: 2,
        shadow_draws_per_cascade: 4,
        transparent_draws: 4,
        materials: 4,
        draws_per_material: 2,
        gbuffer_pipelines: 2,
        draws_per_pipeline: 4,
        transparent_pipelines: 2,
        bloom_mips: 2,
        post_passes: 2,
    };

    fn resolve(ctx: &BenchmarkContext) -> Self {
        let d = if ctx.is_test() {
            Self::TEST
        } else {
            Self::DEFAULT
        };
        Self {
            opaque_draws: ctx.param("frame.opaque-draws", d.opaque_draws),
            shadow_cascades: ctx.param("frame.shadow-cascades", d.shadow_cascades),
            shadow_draws_per_cascade: ctx
                .param("frame.shadow-draws-per-cascade", d.shadow_draws_per_cascade),
            transparent_draws: ctx.param("frame.transparent-draws", d.transparent_draws),
            materials: ctx.param("frame.materials", d.materials),
            draws_per_material: ctx.param("frame.draws-per-material", d.draws_per_material),
            gbuffer_pipelines: ctx.param("frame.gbuffer-pipelines", d.gbuffer_pipelines),
            draws_per_pipeline: ctx.param("frame.draws-per-pipeline", d.draws_per_pipeline),
            transparent_pipelines: ctx
                .param("frame.transparent-pipelines", d.transparent_pipelines),
            bloom_mips: ctx.param("frame.bloom-mips", d.bloom_mips),
            post_passes: ctx.param("frame.post-passes", d.post_passes),
        }
    }

    /// Total draws + dispatches encoded per frame, used as the throughput count.
    fn total_commands(&self) -> u32 {
        self.opaque_draws * 2
            + self.shadow_cascades * self.shadow_draws_per_cascade
            + self.transparent_draws
            + SSAO_DISPATCHES
            + 1 // lighting
            + (2 * self.bloom_mips - 1)
            + self.post_passes
    }

    /// Total render + compute passes encoded per frame.
    fn total_passes(&self) -> u32 {
        // prepass + shadows + gbuffer + ssao + lighting + transparency + bloom + post
        1 + self.shadow_cascades + 4 + (2 * self.bloom_mips - 1) + self.post_passes
    }
}

/// One of the three configurations the frame is encoded in each run.
///
/// All configurations encode the same draw streams through the same passes; they
/// differ only in which vertex/index buffers each draw binds and in how material
/// textures reach the material-using passes (g-buffer and transparency).
#[derive(Clone, Copy, PartialEq)]
enum FrameConfig {
    /// Bindful materials, a distinct vertex/index buffer per object — the baseline,
    /// where the tracker sees new buffers on every draw.
    BindfulPerObject,
    /// Bindful materials, with one shared vertex/index buffer bound once per pass,
    /// eliminating all per-draw vertex/index buffer traffic.
    BindfulShared,
    /// Shared vertex/index buffer, and all material textures bound at once in a
    /// single `binding_array` bind group per pass instead of per-draw material
    /// rebinds. Only runs when the device supports binding arrays.
    BindlessShared,
}

impl FrameConfig {
    /// Label prefix grouping this configuration's sub-results in the output.
    fn tag(self) -> &'static str {
        match self {
            Self::BindfulPerObject => "[Bindful, per-object VB]",
            Self::BindfulShared => "[Bindful, shared VB]",
            Self::BindlessShared => "[Bindless, shared VB]",
        }
    }

    /// Whether the shared vertex/index buffer is bound once per pass instead of
    /// per-object buffers on every draw.
    fn shared_buffers(self) -> bool {
        !matches!(self, Self::BindfulPerObject)
    }
}

/// A single draw within a pass, precomputed at setup so the timed encoding loop
/// performs no random-number generation or other data-dependent work.
struct DrawCommand {
    /// Pipeline to switch to before this draw, when it differs from the previous draw.
    pipeline: Option<u32>,
    /// Material bind group to bind before this draw, when it differs from the previous draw.
    material: Option<u32>,
    /// Object index, selecting the per-object dynamic offset and vertex/index buffers.
    object: u32,
}

/// Generate the draw stream for one pass.
///
/// Pipeline and material switches happen at fixed strides, but which pipeline or
/// material gets selected is randomized (deterministically, from `seed`), and objects
/// are drawn in shuffled order, so state changes and resource accesses don't follow
/// an unrealistically cache-friendly order.
fn generate_stream(
    seed: u64,
    mut objects: Vec<u32>,
    pipeline_count: u32,
    draws_per_pipeline: u32,
    materials: Option<(u32, u32)>,
) -> Vec<DrawCommand> {
    let mut random = WyRand::new_seed(seed);
    random.shuffle(&mut objects);

    let mut current_pipeline = None;
    let mut current_material = None;
    objects
        .iter()
        .enumerate()
        .map(|(draw_idx, &object)| DrawCommand {
            pipeline: (draw_idx as u32)
                .is_multiple_of(draws_per_pipeline)
                .then(|| switch_state(&mut random, &mut current_pipeline, pipeline_count))
                .flatten(),
            material: materials.and_then(|(material_count, draws_per_material)| {
                (draw_idx as u32)
                    .is_multiple_of(draws_per_material)
                    .then(|| switch_state(&mut random, &mut current_material, material_count))
                    .flatten()
            }),
            object,
        })
        .collect()
}

/// Pick a random state index different from the current one, returning it only
/// when it actually changes.
fn switch_state(random: &mut WyRand, current: &mut Option<u32>, count: u32) -> Option<u32> {
    let next = match *current {
        None => random.generate::<u32>() % count,
        Some(index) if count > 1 => (index + 1 + random.generate::<u32>() % (count - 1)) % count,
        Some(index) => index,
    };
    if *current == Some(next) {
        None
    } else {
        *current = Some(next);
        Some(next)
    }
}

struct RenderPipelineArgs<'a> {
    label: &'a str,
    layout: &'a wgpu::PipelineLayout,
    vertex_entry: &'a str,
    fragment_entry: Option<&'a str>,
    vertex_buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    targets: &'a [Option<wgpu::ColorTargetState>],
    depth_stencil: Option<wgpu::DepthStencilState>,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    args: RenderPipelineArgs<'_>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(args.label),
        layout: Some(args.layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(args.vertex_entry),
            buffers: args.vertex_buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: args.depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        fragment: args.fragment_entry.map(|entry_point| wgpu::FragmentState {
            module,
            entry_point: Some(entry_point),
            targets: args.targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn depth_state(
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(depth_write_enabled),
        depth_compare: Some(depth_compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Resources for the bindless configuration, present only when the device supports
/// the required binding-array features. Only the material-using passes (g-buffer and
/// transparency) differ from the bindful configurations.
struct BindlessState {
    /// One bind group holding every material texture in a `binding_array`, bound
    /// once per pass instead of rebinding materials per draw.
    material_bind_group: wgpu::BindGroup,
    gbuffer_pipelines: Vec<wgpu::RenderPipeline>,
    transparent_pipelines: Vec<wgpu::RenderPipeline>,
}

struct FrameState {
    device_state: DeviceState,

    frame_bind_group: wgpu::BindGroup,
    object_bind_group: wgpu::BindGroup,
    material_bind_groups: Vec<wgpu::BindGroup>,

    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    shared_vertex_buffer: wgpu::Buffer,
    shared_index_buffer: wgpu::Buffer,

    bindless: Option<BindlessState>,

    prepass_pipelines: Vec<wgpu::RenderPipeline>,
    shadow_pipelines: Vec<wgpu::RenderPipeline>,
    gbuffer_pipelines: Vec<wgpu::RenderPipeline>,
    transparent_pipelines: Vec<wgpu::RenderPipeline>,
    ssao_pipeline: wgpu::ComputePipeline,
    ssao_blur_pipeline: wgpu::ComputePipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    bloom_down_pipeline: wgpu::RenderPipeline,
    bloom_up_pipeline: wgpu::RenderPipeline,
    post_pipelines: Vec<wgpu::RenderPipeline>,

    depth_view: wgpu::TextureView,
    shadow_views: Vec<wgpu::TextureView>,
    gbuffer_views: Vec<wgpu::TextureView>,
    hdr_view: wgpu::TextureView,
    bloom_views: Vec<wgpu::TextureView>,
    post_ping_views: [wgpu::TextureView; 2],
    backbuffer_view: wgpu::TextureView,

    ssao_bind_groups: Vec<wgpu::BindGroup>,
    lighting_bind_group: wgpu::BindGroup,
    bloom_down_bind_groups: Vec<wgpu::BindGroup>,
    bloom_up_bind_groups: Vec<wgpu::BindGroup>,
    post_bind_groups: Vec<wgpu::BindGroup>,

    prepass_stream: Vec<DrawCommand>,
    shadow_streams: Vec<Vec<DrawCommand>>,
    gbuffer_stream: Vec<DrawCommand>,
    transparent_stream: Vec<DrawCommand>,
}

impl FrameState {
    /// Create and prepare all the resources needed for the frame benchmark.
    fn new(params: FrameParams) -> Self {
        // The noop device would otherwise default to no features at all, so request
        // the binding-array features the bindless configuration needs (the noop
        // adapter exposes every feature); a real backend selected through the env
        // vars requests all of its adapter's features instead. Either way, whether
        // the bindless configuration runs is detected from `device.features()` below.
        let device_state = DeviceState::new_noop_or_env(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: params.materials
                    * TEXTURES_PER_MATERIAL,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        });
        let device = &device_state.device;

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let total_objects = params.opaque_draws + params.transparent_draws;

        let tiny_texture =
            |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };

        let texture_entry =
            |binding: u32, visibility: wgpu::ShaderStages, sample_type: wgpu::TextureSampleType| {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }
            };
        let sampler_entry =
            |binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::SamplerBindingType| {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility,
                    ty: wgpu::BindingType::Sampler(ty),
                    count: None,
                }
            };
        let filterable = wgpu::TextureSampleType::Float { filterable: true };
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        // Bind group layouts. Slot 0 holds per-frame data (or the pass input for
        // fullscreen passes), slot 1 per-material data, and the last slot per-object
        // data accessed through dynamic offsets — the classic bindful engine layout.
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frame BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(OBJECT_STRIDE as u64),
                },
                count: None,
            }],
        });
        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Object BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(OBJECT_STRIDE as u64),
                },
                count: None,
            }],
        });

        let mut material_bgl_entries = Vec::new();
        for i in 0..TEXTURES_PER_MATERIAL {
            material_bgl_entries.push(texture_entry(i, wgpu::ShaderStages::FRAGMENT, filterable));
        }
        material_bgl_entries.push(sampler_entry(
            TEXTURES_PER_MATERIAL,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::SamplerBindingType::Filtering,
        ));
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material BGL"),
            entries: &material_bgl_entries,
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit BGL"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::FRAGMENT, filterable),
                sampler_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
            ],
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite BGL"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT, filterable),
                sampler_entry(
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
            ],
        });
        let lighting_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting BGL"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(2, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(3, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(4, wgpu::ShaderStages::FRAGMENT, filterable),
                texture_entry(
                    5,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::TextureSampleType::Depth,
                ),
                sampler_entry(
                    6,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
                sampler_entry(
                    7,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Comparison,
                ),
            ],
        });
        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO BGL"),
            entries: &[
                texture_entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::TextureSampleType::Depth,
                ),
                storage_entry(1),
            ],
        });
        let ssao_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Blur BGL"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::COMPUTE, filterable),
                storage_entry(1),
            ],
        });

        // All buffers are created `mapped_at_creation` (then immediately unmapped) so their
        // init trackers start fully drained and the encode loop takes the initialized fast
        // path from iteration 1. Usage-driven init at first submit can't do this for VERTEX
        // buffers: those get an extra `+1` size bump, and bounds checking uses the real
        // (larger) allocated size, so the tail must stay in the init domain but usage never
        // reaches it. TODO: review whether mapping at setup is the right long-term answer.
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: OBJECT_STRIDE as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        camera_buffer.unmap();
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frame BG"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Object Buffer"),
            size: total_objects as u64 * OBJECT_STRIDE as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        object_buffer.unmap();
        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Object BG"),
            layout: &object_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_buffer,
                    offset: 0,
                    size: NonZeroU64::new(OBJECT_STRIDE as u64),
                }),
            }],
        });

        // Materials: a few shared samplers and a pile of tiny textures.
        let material_samplers: Vec<_> = (0..4)
            .map(|i| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some(&format!("Material Sampler {i}")),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                })
            })
            .collect();

        let mut material_texture_views = Vec::new();
        for i in 0..params.materials * TEXTURES_PER_MATERIAL {
            material_texture_views.push(tiny_texture(
                &format!("Material Texture {i}"),
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ));
        }
        random.shuffle(&mut material_texture_views);

        let mut material_bind_groups = Vec::with_capacity(params.materials as usize);
        for material_idx in 0..params.materials {
            let mut entries = Vec::with_capacity(TEXTURES_PER_MATERIAL as usize + 1);
            for texture_idx in 0..TEXTURES_PER_MATERIAL {
                entries.push(wgpu::BindGroupEntry {
                    binding: texture_idx,
                    resource: wgpu::BindingResource::TextureView(
                        &material_texture_views
                            [(material_idx * TEXTURES_PER_MATERIAL + texture_idx) as usize],
                    ),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: TEXTURES_PER_MATERIAL,
                resource: wgpu::BindingResource::Sampler(
                    &material_samplers[material_idx as usize % material_samplers.len()],
                ),
            });
            material_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &material_bgl,
                entries: &entries,
            }));
        }

        // Per-object geometry pools.
        let mut vertex_buffers =
            Vec::with_capacity((total_objects * VERTEX_BUFFERS_PER_DRAW) as usize);
        for _ in 0..total_objects * VERTEX_BUFFERS_PER_DRAW {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 3 * 16,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: true,
            });
            buffer.unmap();
            vertex_buffers.push(buffer);
        }
        random.shuffle(&mut vertex_buffers);

        let mut index_buffers = Vec::with_capacity(total_objects as usize);
        for _ in 0..total_objects {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 3 * 4,
                usage: wgpu::BufferUsages::INDEX,
                mapped_at_creation: true,
            });
            buffer.unmap();
            index_buffers.push(buffer);
        }
        random.shuffle(&mut index_buffers);

        // Shared geometry for the shared-VB configurations, bound once per pass the
        // way an engine with suballocated meshes would. Contents are irrelevant; the
        // measured difference is the per-draw buffer binds and their tracker churn
        // disappearing entirely.
        let shared_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shared Vertex Buffer"),
            size: 3 * 16,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        shared_vertex_buffer.unmap();
        let shared_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shared Index Buffer"),
            size: 3 * 4,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        shared_index_buffer.unmap();

        // Attachments and pass intermediates.
        let attachment =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let depth_view = tiny_texture("Depth", wgpu::TextureFormat::Depth32Float, attachment);
        let shadow_views: Vec<_> = (0..params.shadow_cascades)
            .map(|i| {
                tiny_texture(
                    &format!("Shadow Cascade {i}"),
                    wgpu::TextureFormat::Depth32Float,
                    attachment,
                )
            })
            .collect();
        let gbuffer_formats = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba16Float,
        ];
        let gbuffer_views: Vec<_> = gbuffer_formats
            .iter()
            .enumerate()
            .map(|(i, &format)| tiny_texture(&format!("G-Buffer {i}"), format, attachment))
            .collect();
        let hdr_view = tiny_texture("HDR Color", wgpu::TextureFormat::Rgba16Float, attachment);
        let ssao_usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let ssao_raw_view = tiny_texture("SSAO Raw", wgpu::TextureFormat::Rgba8Unorm, ssao_usage);
        let ssao_blur_view = tiny_texture("SSAO Blur", wgpu::TextureFormat::Rgba8Unorm, ssao_usage);
        let ssao_final_view =
            tiny_texture("SSAO Final", wgpu::TextureFormat::Rgba8Unorm, ssao_usage);
        let bloom_views: Vec<_> = (0..params.bloom_mips)
            .map(|i| {
                tiny_texture(
                    &format!("Bloom Mip {i}"),
                    wgpu::TextureFormat::Rgba16Float,
                    attachment,
                )
            })
            .collect();
        let post_ping_views = [
            tiny_texture("Post Ping 0", wgpu::TextureFormat::Rgba8Unorm, attachment),
            tiny_texture("Post Ping 1", wgpu::TextureFormat::Rgba8Unorm, attachment),
        ];
        let backbuffer_view = tiny_texture(
            "Backbuffer",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        // Pass input bind groups.
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let bind_group =
            |label: &str, layout: &wgpu::BindGroupLayout, resources: &[wgpu::BindingResource]| {
                let entries: Vec<_> = resources
                    .iter()
                    .enumerate()
                    .map(|(i, resource)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: resource.clone(),
                    })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &entries,
                })
            };

        // The SSAO chain reads the prepass depth and blurs the result in two steps.
        let ssao_bind_groups = vec![
            bind_group(
                "SSAO BG",
                &ssao_bgl,
                &[
                    wgpu::BindingResource::TextureView(&depth_view),
                    wgpu::BindingResource::TextureView(&ssao_raw_view),
                ],
            ),
            bind_group(
                "SSAO Blur H BG",
                &ssao_blur_bgl,
                &[
                    wgpu::BindingResource::TextureView(&ssao_raw_view),
                    wgpu::BindingResource::TextureView(&ssao_blur_view),
                ],
            ),
            bind_group(
                "SSAO Blur V BG",
                &ssao_blur_bgl,
                &[
                    wgpu::BindingResource::TextureView(&ssao_blur_view),
                    wgpu::BindingResource::TextureView(&ssao_final_view),
                ],
            ),
        ];

        let lighting_bind_group = bind_group(
            "Lighting BG",
            &lighting_bgl,
            &[
                wgpu::BindingResource::TextureView(&gbuffer_views[0]),
                wgpu::BindingResource::TextureView(&gbuffer_views[1]),
                wgpu::BindingResource::TextureView(&gbuffer_views[2]),
                wgpu::BindingResource::TextureView(&gbuffer_views[3]),
                wgpu::BindingResource::TextureView(&ssao_final_view),
                wgpu::BindingResource::TextureView(&shadow_views[0]),
                wgpu::BindingResource::Sampler(&linear_sampler),
                wgpu::BindingResource::Sampler(&comparison_sampler),
            ],
        );

        // Bloom downsample chain: HDR -> mip 0 -> mip 1 -> ...; each pass samples the
        // previous level. The upsample chain then blends each mip back into the one
        // above it.
        let bloom_down_bind_groups: Vec<_> = (0..params.bloom_mips as usize)
            .map(|i| {
                let input = if i == 0 {
                    &hdr_view
                } else {
                    &bloom_views[i - 1]
                };
                bind_group(
                    &format!("Bloom Down BG {i}"),
                    &blit_bgl,
                    &[
                        wgpu::BindingResource::TextureView(input),
                        wgpu::BindingResource::Sampler(&linear_sampler),
                    ],
                )
            })
            .collect();
        let bloom_up_bind_groups: Vec<_> = (0..params.bloom_mips as usize - 1)
            .rev()
            .map(|i| {
                bind_group(
                    &format!("Bloom Up BG {i}"),
                    &blit_bgl,
                    &[
                        wgpu::BindingResource::TextureView(&bloom_views[i + 1]),
                        wgpu::BindingResource::Sampler(&linear_sampler),
                    ],
                )
            })
            .collect();

        // Post pass 0 composites HDR + bloom; each following pass samples the previous
        // ping-pong target.
        let mut post_bind_groups = vec![bind_group(
            "Composite BG",
            &composite_bgl,
            &[
                wgpu::BindingResource::TextureView(&hdr_view),
                wgpu::BindingResource::TextureView(&bloom_views[0]),
                wgpu::BindingResource::Sampler(&linear_sampler),
            ],
        )];
        for i in 1..params.post_passes as usize {
            post_bind_groups.push(bind_group(
                &format!("Post BG {i}"),
                &blit_bgl,
                &[
                    wgpu::BindingResource::TextureView(&post_ping_views[(i - 1) % 2]),
                    wgpu::BindingResource::Sampler(&linear_sampler),
                ],
            ));
        }

        // Pipelines.
        let module = device.create_shader_module(wgpu::include_wgsl!("frame.wgsl"));

        let pipeline_layout = |label: &str, bgls: &[&wgpu::BindGroupLayout]| {
            let bgls: Vec<_> = bgls.iter().map(|&bgl| Some(bgl)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            })
        };
        let depth_only_layout = pipeline_layout("Depth Only", &[&frame_bgl, &object_bgl]);
        let geometry_layout =
            pipeline_layout("Geometry", &[&frame_bgl, &material_bgl, &object_bgl]);
        let lighting_layout = pipeline_layout("Lighting", &[&frame_bgl, &lighting_bgl]);
        let blit_layout = pipeline_layout("Blit", &[&blit_bgl]);
        let composite_layout = pipeline_layout("Composite", &[&composite_bgl]);
        let ssao_layout = pipeline_layout("SSAO", &[&frame_bgl, &ssao_bgl]);
        let ssao_blur_layout = pipeline_layout("SSAO Blur", &[&frame_bgl, &ssao_blur_bgl]);

        let geometry_vertex_attributes_0 = wgpu::vertex_attr_array![0 => Float32x4];
        let geometry_vertex_attributes_1 = wgpu::vertex_attr_array![1 => Float32x4];
        let geometry_vertex_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &geometry_vertex_attributes_0,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &geometry_vertex_attributes_1,
            }),
        ];

        let depth_only_pipelines = |label: &str| -> Vec<wgpu::RenderPipeline> {
            (0..DEPTH_ONLY_PIPELINES)
                .map(|i| {
                    create_render_pipeline(
                        device,
                        &module,
                        RenderPipelineArgs {
                            label: &format!("{label} Pipeline {i}"),
                            layout: &depth_only_layout,
                            vertex_entry: "vs_geometry",
                            fragment_entry: None,
                            vertex_buffers: &geometry_vertex_buffers,
                            targets: &[],
                            depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
                        },
                    )
                })
                .collect()
        };
        let prepass_pipelines = depth_only_pipelines("Prepass");
        let shadow_pipelines = depth_only_pipelines("Shadow");

        let gbuffer_targets: Vec<_> = gbuffer_formats
            .iter()
            .map(|&format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();
        let gbuffer_pipelines: Vec<_> = (0..params.gbuffer_pipelines)
            .map(|i| {
                create_render_pipeline(
                    device,
                    &module,
                    RenderPipelineArgs {
                        label: &format!("G-Buffer Pipeline {i}"),
                        layout: &geometry_layout,
                        vertex_entry: "vs_geometry",
                        fragment_entry: Some("fs_gbuffer"),
                        vertex_buffers: &geometry_vertex_buffers,
                        targets: &gbuffer_targets,
                        depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Equal)),
                    },
                )
            })
            .collect();

        let hdr_target = |blend| {
            [Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })]
        };
        let transparent_pipelines: Vec<_> = (0..params.transparent_pipelines)
            .map(|i| {
                create_render_pipeline(
                    device,
                    &module,
                    RenderPipelineArgs {
                        label: &format!("Transparent Pipeline {i}"),
                        layout: &geometry_layout,
                        vertex_entry: "vs_geometry",
                        fragment_entry: Some("fs_color"),
                        vertex_buffers: &geometry_vertex_buffers,
                        targets: &hdr_target(Some(wgpu::BlendState::ALPHA_BLENDING)),
                        depth_stencil: Some(depth_state(false, wgpu::CompareFunction::LessEqual)),
                    },
                )
            })
            .collect();

        // Bindless variants of the material-using passes. Prepass, shadows, SSAO,
        // lighting, bloom, and post are material-free and shared with the bindful
        // configurations.
        let supports_bindless = device.features().contains(
            wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
        // TODO: as of writing llvmpipe segfaults the bindless benchmark on ci
        && device_state.adapter_info.driver != "llvmpipe";

        let bindless = supports_bindless.then(|| {
            let texture_count = params.materials * TEXTURES_PER_MATERIAL;

            let mut bindless_bgl_entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: filterable,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(NonZeroU32::new(texture_count).unwrap()),
            }];
            for i in 0..material_samplers.len() as u32 {
                bindless_bgl_entries.push(sampler_entry(
                    1 + i,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ));
            }
            let bindless_material_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bindless Material BGL"),
                    entries: &bindless_bgl_entries,
                });

            let material_texture_view_refs: Vec<_> = material_texture_views.iter().collect();
            let mut bindless_bg_entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&material_texture_view_refs),
            }];
            for (i, sampler) in material_samplers.iter().enumerate() {
                bindless_bg_entries.push(wgpu::BindGroupEntry {
                    binding: 1 + i as u32,
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            }
            let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bindless Material BG"),
                layout: &bindless_material_bgl,
                entries: &bindless_bg_entries,
            });

            // The bindless module can only be created when the features are present,
            // so it lives in its own file rather than as extra entry points in
            // frame.wgsl.
            let bindless_module =
                device.create_shader_module(wgpu::include_wgsl!("frame-bindless.wgsl"));
            let bindless_geometry_layout = pipeline_layout(
                "Bindless Geometry",
                &[&frame_bgl, &bindless_material_bgl, &object_bgl],
            );

            let gbuffer_pipelines: Vec<_> = (0..params.gbuffer_pipelines)
                .map(|i| {
                    create_render_pipeline(
                        device,
                        &bindless_module,
                        RenderPipelineArgs {
                            label: &format!("Bindless G-Buffer Pipeline {i}"),
                            layout: &bindless_geometry_layout,
                            vertex_entry: "vs_geometry",
                            fragment_entry: Some("fs_gbuffer"),
                            vertex_buffers: &geometry_vertex_buffers,
                            targets: &gbuffer_targets,
                            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Equal)),
                        },
                    )
                })
                .collect();
            let transparent_pipelines: Vec<_> = (0..params.transparent_pipelines)
                .map(|i| {
                    create_render_pipeline(
                        device,
                        &bindless_module,
                        RenderPipelineArgs {
                            label: &format!("Bindless Transparent Pipeline {i}"),
                            layout: &bindless_geometry_layout,
                            vertex_entry: "vs_geometry",
                            fragment_entry: Some("fs_color"),
                            vertex_buffers: &geometry_vertex_buffers,
                            targets: &hdr_target(Some(wgpu::BlendState::ALPHA_BLENDING)),
                            depth_stencil: Some(depth_state(
                                false,
                                wgpu::CompareFunction::LessEqual,
                            )),
                        },
                    )
                })
                .collect();

            BindlessState {
                material_bind_group,
                gbuffer_pipelines,
                transparent_pipelines,
            }
        });

        let lighting_pipeline = create_render_pipeline(
            device,
            &module,
            RenderPipelineArgs {
                label: "Lighting Pipeline",
                layout: &lighting_layout,
                vertex_entry: "vs_fullscreen",
                fragment_entry: Some("fs_color"),
                vertex_buffers: &[],
                targets: &hdr_target(None),
                depth_stencil: None,
            },
        );

        let bloom_down_pipeline = create_render_pipeline(
            device,
            &module,
            RenderPipelineArgs {
                label: "Bloom Down Pipeline",
                layout: &blit_layout,
                vertex_entry: "vs_fullscreen",
                fragment_entry: Some("fs_color"),
                vertex_buffers: &[],
                targets: &hdr_target(None),
                depth_stencil: None,
            },
        );
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let bloom_up_pipeline = create_render_pipeline(
            device,
            &module,
            RenderPipelineArgs {
                label: "Bloom Up Pipeline",
                layout: &blit_layout,
                vertex_entry: "vs_fullscreen",
                fragment_entry: Some("fs_color"),
                vertex_buffers: &[],
                targets: &hdr_target(Some(additive)),
                depth_stencil: None,
            },
        );

        let ldr_target = [Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let post_pipelines: Vec<_> = (0..params.post_passes)
            .map(|i| {
                create_render_pipeline(
                    device,
                    &module,
                    RenderPipelineArgs {
                        label: &format!("Post Pipeline {i}"),
                        layout: if i == 0 {
                            &composite_layout
                        } else {
                            &blit_layout
                        },
                        vertex_entry: "vs_fullscreen",
                        fragment_entry: Some("fs_color"),
                        vertex_buffers: &[],
                        targets: &ldr_target,
                        depth_stencil: None,
                    },
                )
            })
            .collect();

        let compute_pipeline = |label: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &module,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let ssao_pipeline = compute_pipeline("SSAO Pipeline", &ssao_layout);
        let ssao_blur_pipeline = compute_pipeline("SSAO Blur Pipeline", &ssao_blur_layout);

        // Precomputed draw streams. Prepass, shadow, and g-buffer passes draw the same
        // objects in different orders, as if sorted by different criteria.
        let prepass_stream = generate_stream(
            0x0BAD_5EED_0001,
            (0..params.opaque_draws).collect(),
            DEPTH_ONLY_PIPELINES,
            params.draws_per_pipeline,
            None,
        );
        let shadow_streams: Vec<_> = (0..params.shadow_cascades)
            .map(|cascade| {
                let objects = (0..params.shadow_draws_per_cascade)
                    .map(|i| (cascade * params.shadow_draws_per_cascade + i) % params.opaque_draws)
                    .collect();
                generate_stream(
                    0x0BAD_5EED_0100 + cascade as u64,
                    objects,
                    DEPTH_ONLY_PIPELINES,
                    params.draws_per_pipeline,
                    None,
                )
            })
            .collect();
        let gbuffer_stream = generate_stream(
            0x0BAD_5EED_0002,
            (0..params.opaque_draws).collect(),
            params.gbuffer_pipelines,
            params.draws_per_pipeline,
            Some((params.materials, params.draws_per_material)),
        );
        let transparent_stream = generate_stream(
            0x0BAD_5EED_0003,
            (params.opaque_draws..total_objects).collect(),
            params.transparent_pipelines,
            TRANSPARENT_DRAWS_PER_PIPELINE,
            Some((params.materials, 1)),
        );

        Self {
            device_state,

            frame_bind_group,
            object_bind_group,
            material_bind_groups,

            vertex_buffers,
            index_buffers,
            shared_vertex_buffer,
            shared_index_buffer,

            bindless,

            prepass_pipelines,
            shadow_pipelines,
            gbuffer_pipelines,
            transparent_pipelines,
            ssao_pipeline,
            ssao_blur_pipeline,
            lighting_pipeline,
            bloom_down_pipeline,
            bloom_up_pipeline,
            post_pipelines,

            depth_view,
            shadow_views,
            gbuffer_views,
            hdr_view,
            bloom_views,
            post_ping_views,
            backbuffer_view,

            ssao_bind_groups,
            lighting_bind_group,
            bloom_down_bind_groups,
            bloom_up_bind_groups,
            post_bind_groups,

            prepass_stream,
            shadow_streams,
            gbuffer_stream,
            transparent_stream,
        }
    }

    /// Encode one pass's draw stream, rebinding only the state each draw actually
    /// changes, the way a state-sorted engine would.
    ///
    /// Config decisions are hoisted out of the per-draw loop as far as possible so
    /// the timed loop measures the configuration, not extra per-draw branching.
    fn encode_draw_stream(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        stream: &[DrawCommand],
        pipelines: &[wgpu::RenderPipeline],
        has_materials: bool,
        config: FrameConfig,
    ) {
        render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
        let object_slot = if has_materials { 2 } else { 1 };
        let materials_per_draw = has_materials && config != FrameConfig::BindlessShared;
        if has_materials && config == FrameConfig::BindlessShared {
            // Every material texture lives in the one binding_array bind group,
            // bound once here instead of rebinding materials per draw.
            render_pass.set_bind_group(
                1,
                &self.bindless.as_ref().unwrap().material_bind_group,
                &[],
            );
        }
        let per_draw_buffers = !config.shared_buffers();
        if !per_draw_buffers {
            // The shared geometry is bound once and stays bound for the whole pass;
            // draws index into it, so no further buffer binds are needed.
            for i in 0..VERTEX_BUFFERS_PER_DRAW {
                render_pass.set_vertex_buffer(i, self.shared_vertex_buffer.slice(..));
            }
            render_pass.set_index_buffer(
                self.shared_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
        }
        for command in stream {
            if let Some(pipeline) = command.pipeline {
                render_pass.set_pipeline(&pipelines[pipeline as usize]);
            }
            if materials_per_draw {
                if let Some(material) = command.material {
                    render_pass.set_bind_group(
                        1,
                        &self.material_bind_groups[material as usize],
                        &[],
                    );
                }
            }
            render_pass.set_bind_group(
                object_slot,
                &self.object_bind_group,
                &[command.object * OBJECT_STRIDE],
            );
            if per_draw_buffers {
                for i in 0..VERTEX_BUFFERS_PER_DRAW {
                    render_pass.set_vertex_buffer(
                        i,
                        self.vertex_buffers
                            [(command.object * VERTEX_BUFFERS_PER_DRAW + i) as usize]
                            .slice(..),
                    );
                }
                render_pass.set_index_buffer(
                    self.index_buffers[command.object as usize].slice(..),
                    wgpu::IndexFormat::Uint32,
                );
            }
            render_pass.draw_indexed(0..3, 0, 0..1);
        }
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    fn depth_only_pass_descriptor<'a>(
        label: &'a str,
        view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPassDescriptor<'a> {
        wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }
    }

    fn color_attachment(
        view: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })
    }

    fn encode_prepass(&self, config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Depth Prepass");

        let mut encoder = self.encoder("Depth Prepass");
        {
            let mut render_pass = encoder.begin_render_pass(&Self::depth_only_pass_descriptor(
                "Depth Prepass",
                &self.depth_view,
            ));
            self.encode_draw_stream(
                &mut render_pass,
                &self.prepass_stream,
                &self.prepass_pipelines,
                false,
                config,
            );
        }
        encoder.finish()
    }

    fn encode_shadows(&self, config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Shadows");

        let mut encoder = self.encoder("Shadows");
        for (cascade, view) in self.shadow_views.iter().enumerate() {
            let label = format!("Shadow Cascade {cascade}");
            let mut render_pass =
                encoder.begin_render_pass(&Self::depth_only_pass_descriptor(&label, view));
            self.encode_draw_stream(
                &mut render_pass,
                &self.shadow_streams[cascade],
                &self.shadow_pipelines,
                false,
                config,
            );
        }
        encoder.finish()
    }

    fn encode_gbuffer(&self, config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("G-Buffer");

        let mut encoder = self.encoder("G-Buffer");
        {
            let color_attachments: Vec<_> = self
                .gbuffer_views
                .iter()
                .map(|view| Self::color_attachment(view, wgpu::LoadOp::Clear(wgpu::Color::BLACK)))
                .collect();
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("G-Buffer"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    // Read-only: depth testing against the prepass results.
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let pipelines = if config == FrameConfig::BindlessShared {
                &self.bindless.as_ref().unwrap().gbuffer_pipelines
            } else {
                &self.gbuffer_pipelines
            };
            self.encode_draw_stream(
                &mut render_pass,
                &self.gbuffer_stream,
                pipelines,
                true,
                config,
            );
        }
        encoder.finish()
    }

    fn encode_ssao(&self, _config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("SSAO");

        let mut encoder = self.encoder("SSAO");
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SSAO"),
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, &self.frame_bind_group, &[]);
            compute_pass.set_pipeline(&self.ssao_pipeline);
            compute_pass.set_bind_group(1, &self.ssao_bind_groups[0], &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
            compute_pass.set_pipeline(&self.ssao_blur_pipeline);
            compute_pass.set_bind_group(1, &self.ssao_bind_groups[1], &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
            compute_pass.set_bind_group(1, &self.ssao_bind_groups[2], &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.finish()
    }

    fn encode_lighting(&self, _config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Lighting");

        let mut encoder = self.encoder("Lighting");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lighting"),
                color_attachments: &[Self::color_attachment(
                    &self.hdr_view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                )],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.lighting_pipeline);
            render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
            render_pass.set_bind_group(1, &self.lighting_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        encoder.finish()
    }

    fn encode_transparency(&self, config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Transparency");

        let mut encoder = self.encoder("Transparency");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparency"),
                color_attachments: &[Self::color_attachment(&self.hdr_view, wgpu::LoadOp::Load)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let pipelines = if config == FrameConfig::BindlessShared {
                &self.bindless.as_ref().unwrap().transparent_pipelines
            } else {
                &self.transparent_pipelines
            };
            self.encode_draw_stream(
                &mut render_pass,
                &self.transparent_stream,
                pipelines,
                true,
                config,
            );
        }
        encoder.finish()
    }

    /// One fullscreen draw sampling `bind_group`, rendered to `target`.
    fn encode_blit_pass(
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Self::color_attachment(target, load)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn encode_bloom(&self, _config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Bloom");

        let mut encoder = self.encoder("Bloom");
        for (i, bind_group) in self.bloom_down_bind_groups.iter().enumerate() {
            Self::encode_blit_pass(
                &mut encoder,
                &format!("Bloom Down {i}"),
                &self.bloom_views[i],
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                &self.bloom_down_pipeline,
                bind_group,
            );
        }
        let up_targets = (0..self.bloom_views.len() - 1).rev();
        for (bind_group, i) in self.bloom_up_bind_groups.iter().zip(up_targets) {
            Self::encode_blit_pass(
                &mut encoder,
                &format!("Bloom Up {i}"),
                &self.bloom_views[i],
                wgpu::LoadOp::Load,
                &self.bloom_up_pipeline,
                bind_group,
            );
        }
        encoder.finish()
    }

    fn encode_post(&self, _config: FrameConfig) -> wgpu::CommandBuffer {
        profiling::scope!("Post Process");

        let mut encoder = self.encoder("Post Process");
        let pass_count = self.post_pipelines.len();
        for i in 0..pass_count {
            let target = if i == pass_count - 1 {
                &self.backbuffer_view
            } else {
                &self.post_ping_views[i % 2]
            };
            Self::encode_blit_pass(
                &mut encoder,
                &format!("Post Pass {i}"),
                target,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                &self.post_pipelines[i],
                &self.post_bind_groups[i],
            );
        }
        encoder.finish()
    }
}

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<wgpu_benchmark::SubBenchResult>> {
    let params = FrameParams::resolve(&ctx);
    anyhow::ensure!(
        [
            params.opaque_draws,
            params.shadow_cascades,
            params.shadow_draws_per_cascade,
            params.transparent_draws,
            params.materials,
            params.draws_per_material,
            params.gbuffer_pipelines,
            params.draws_per_pipeline,
            params.transparent_pipelines,
            params.bloom_mips,
            params.post_passes,
        ]
        .iter()
        .all(|&v| v >= 1),
        "all frame benchmark parameters must be at least 1"
    );

    let state = FrameState::new(params);

    // This benchmark hangs on Apple Paravirtualized GPUs. No idea why.
    if state.device_state.adapter_info.name.contains("Paravirtual") {
        anyhow::bail!("Benchmark unsupported on Paravirtualized GPUs");
    }

    println!(
        "  {} draws + {} dispatches across {} passes per frame",
        params.total_commands() - SSAO_DISPATCHES,
        SSAO_DISPATCHES,
        params.total_passes(),
    );
    println!(
        "  knobs: opaque-draws={} shadow-cascades={} shadow-draws-per-cascade={} \
         transparent-draws={} materials={} draws-per-material={} gbuffer-pipelines={} \
         draws-per-pipeline={} transparent-pipelines={} bloom-mips={} post-passes={}",
        params.opaque_draws,
        params.shadow_cascades,
        params.shadow_draws_per_cascade,
        params.transparent_draws,
        params.materials,
        params.draws_per_material,
        params.gbuffer_pipelines,
        params.draws_per_pipeline,
        params.transparent_pipelines,
        params.bloom_mips,
        params.post_passes,
    );

    type PhaseEncoder = fn(&FrameState, FrameConfig) -> wgpu::CommandBuffer;
    let phases: [(&str, PhaseEncoder); 8] = [
        ("Encode: Depth Prepass", FrameState::encode_prepass),
        ("Encode: Shadows", FrameState::encode_shadows),
        ("Encode: G-Buffer", FrameState::encode_gbuffer),
        ("Encode: SSAO", FrameState::encode_ssao),
        ("Encode: Lighting", FrameState::encode_lighting),
        ("Encode: Transparency", FrameState::encode_transparency),
        ("Encode: Bloom", FrameState::encode_bloom),
        ("Encode: Post Process", FrameState::encode_post),
    ];

    let mut results = Vec::new();
    for config in [
        FrameConfig::BindfulPerObject,
        FrameConfig::BindfulShared,
        FrameConfig::BindlessShared,
    ] {
        // Without binding-array support only the bindful configurations report.
        if config == FrameConfig::BindlessShared && state.bindless.is_none() {
            continue;
        }

        let tag = config.tag();
        let mut labels = vec![format!("{tag} Total Encoding")];
        labels.extend(phases.iter().map(|&(name, _)| format!("{tag} {name}")));
        labels.push(format!("{tag} Submit"));

        results.extend(iter_many(
            &ctx,
            labels,
            "draws",
            params.total_commands(),
            || {
                let mut durations = Vec::with_capacity(phases.len() + 2);
                durations.push(Duration::ZERO); // Placeholder for total encoding time.
                let mut buffers = Vec::with_capacity(phases.len());

                let encoding_start = Instant::now();
                for &(_, encode) in &phases {
                    let phase_start = Instant::now();
                    buffers.push(encode(&state, config));
                    durations.push(phase_start.elapsed());
                }
                durations[0] = encoding_start.elapsed();

                let submit_start = Instant::now();
                state.device_state.queue.submit(buffers);
                durations.push(submit_start.elapsed());

                state
                    .device_state
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();

                durations
            },
        ));
    }

    Ok(results)
}
