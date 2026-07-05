//! Isolates the fixed per-pass CPU overhead of encoding and submitting many
//! tiny render/compute passes.
//!
//! Each iteration encodes `passes` passes into a single command encoder, where
//! every pass contains at most one command. This deliberately maximizes the
//! ratio of per-pass bookkeeping (begin/end pass, attachment setup, bind-group
//! and pipeline processing) to actual command work, so any fixed cost added per
//! pass by wgpu-core shows up cleanly. Four modes cover the render/compute and
//! empty/one-command combinations. By default it runs on the noop backend so
//! that all of wgpu-core's validation and tracking runs with zero GPU-driver
//! variance; set `WGPU_BACKEND` (or `WGPU_ADAPTER_NAME`) to run against a real
//! backend like the other benchmarks.

use std::{num::NonZeroU64, time::Instant};

use wgpu_benchmark::{iter_many, BenchmarkContext};

use crate::DeviceState;

/// The single bound buffer is this small: per-pass overhead never looks at
/// buffer contents, so a minimal binding suffices to exercise bind-group
/// processing.
const BUFFER_SIZE: u64 = 16;

/// Scaling knobs for the pass-overhead benchmark.
///
/// Each knob can be overridden on the command line as
/// `--param pass-overhead.<knob>=<value>` with the knob name in kebab-case, e.g.
/// `cargo bench -p wgpu-benchmark -- "Pass Overhead" --param pass-overhead.passes=500`.
#[derive(Clone, Copy)]
struct PassOverheadParams {
    /// Number of passes encoded per iteration.
    passes: u32,
    /// Number of color attachments on each render pass (compute modes ignore this).
    color_attachments: u32,
}

impl PassOverheadParams {
    const DEFAULT: Self = Self {
        passes: 2000,
        color_attachments: 1,
    };

    /// A very lightweight configuration so test mode just checks that the
    /// benchmark does not break.
    const TEST: Self = Self {
        passes: 8,
        color_attachments: 1,
    };

    fn resolve(ctx: &BenchmarkContext) -> Self {
        let d = if ctx.is_test() {
            Self::TEST
        } else {
            Self::DEFAULT
        };
        Self {
            passes: ctx.param("pass-overhead.passes", d.passes),
            color_attachments: ctx.param("pass-overhead.color-attachments", d.color_attachments),
        }
    }
}

struct PassOverheadState {
    device_state: DeviceState,
    params: PassOverheadParams,

    /// One distinct 1x1 render-attachment view per color slot. They must be
    /// distinct textures/views: reusing one view in two color slots of the same
    /// pass errors with a subresource overlap.
    color_views: Vec<wgpu::TextureView>,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,

    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
}

impl PassOverheadState {
    /// Create and prepare all the resources needed for the pass-overhead benchmark.
    fn new(params: PassOverheadParams) -> Self {
        let device_state = DeviceState::new_noop_or_env(&wgpu::DeviceDescriptor::default());
        let device = &device_state.device;

        let color_count = params.color_attachments as usize;

        // Distinct textures/views, one per color slot.
        let color_views: Vec<_> = (0..color_count)
            .map(|i| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Color Target {i}")),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Color Target View {i}")),
                    ..Default::default()
                })
            })
            .collect();

        // A single small uniform buffer, bound by both pipelines, so the
        // one-command modes exercise per-pass bind-group processing.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform"),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        // A single small storage buffer for the compute kernel to write to, so
        // its body cannot be optimized into invalidity.
        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage"),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(BUFFER_SIZE),
            },
            count: None,
        };

        // Render bind group: one uniform buffer visible to the fragment stage.
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render BGL"),
            entries: &[uniform_entry(wgpu::ShaderStages::FRAGMENT)],
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render BG"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Compute bind group: a uniform buffer to read and a storage buffer to write.
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute BGL"),
            entries: &[
                uniform_entry(wgpu::ShaderStages::COMPUTE),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(BUFFER_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer.as_entire_binding(),
                },
            ],
        });

        // A fullscreen-triangle vertex shader plus a fragment shader writing to
        // every color target, and a trivial no-op-ish compute kernel that writes
        // one value so it cannot be optimized away.
        let mut shader_src = String::from(
            "\
struct Uniforms { value: vec4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index) - 1);
    let y = f32(i32(vertex_index & 1u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct FragmentOutput {
",
        );
        for i in 0..params.color_attachments {
            shader_src.push_str(&format!("    @location({i}) color{i}: vec4<f32>,\n"));
        }
        shader_src.push_str(
            "}

@fragment
fn fs_main() -> FragmentOutput {
    var out: FragmentOutput;
",
        );
        for i in 0..params.color_attachments {
            shader_src.push_str(&format!("    out.color{i} = u.value;\n"));
        }
        shader_src.push_str(
            "    return out;
}

@group(0) @binding(1) var<storage, read_write> data: array<f32, 4>;

@compute @workgroup_size(1)
fn cs_main() {
    data[0] = u.value.x;
}
",
        );

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pass Overhead Shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let targets: Vec<_> = (0..params.color_attachments)
            .map(|_| {
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Layout"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device_state,
            params,
            color_views,
            render_pipeline,
            render_bind_group,
            compute_pipeline,
            compute_bind_group,
        }
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    fn color_attachments(&self) -> Vec<Option<wgpu::RenderPassColorAttachment<'_>>> {
        self.color_views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect()
    }

    /// Encode `passes` render passes, each with at most `with_draw` a single draw.
    /// Empty passes still need the color attachment (a pass with no attachment
    /// errors out).
    fn encode_render(&self, with_draw: bool) -> wgpu::CommandBuffer {
        let color_attachments = self.color_attachments();
        let mut encoder = self.encoder("Render");
        for _ in 0..self.params.passes {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if with_draw {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.render_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
        encoder.finish()
    }

    /// Encode `passes` compute passes, each with at most a single dispatch.
    fn encode_compute(&self, with_dispatch: bool) -> wgpu::CommandBuffer {
        let mut encoder = self.encoder("Compute");
        for _ in 0..self.params.passes {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            if with_dispatch {
                compute_pass.set_pipeline(&self.compute_pipeline);
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.finish()
    }
}

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<wgpu_benchmark::SubBenchResult>> {
    let params = PassOverheadParams::resolve(&ctx);
    anyhow::ensure!(
        params.passes >= 1,
        "pass-overhead.passes must be at least 1"
    );
    // Rgba8Unorm counts as 8 bytes/sample against the default 32-byte
    // max_color_attachment_bytes_per_sample limit, so 4 is the max here.
    anyhow::ensure!(
        (1..=4).contains(&params.color_attachments),
        "pass-overhead.color-attachments must be in 1..=4"
    );

    let state = PassOverheadState::new(params);

    println!(
        "  knobs: passes={} color-attachments={}",
        params.passes, params.color_attachments,
    );

    type ModeEncoder = fn(&PassOverheadState) -> wgpu::CommandBuffer;
    let modes: [(&str, ModeEncoder); 4] = [
        ("Render pass (empty)", |s| s.encode_render(false)),
        ("Render pass (1 draw)", |s| s.encode_render(true)),
        ("Compute pass (empty)", |s| s.encode_compute(false)),
        ("Compute pass (1 dispatch)", |s| s.encode_compute(true)),
    ];

    let mut results = Vec::new();
    for (name, encode) in modes {
        let labels = vec![format!("{name} Encoding"), format!("{name} Submit")];

        // `iter_many` defaults to only 1s of looping, so real runs should pass
        // `--time` or `--iters` for stable numbers.
        results.extend(iter_many(&ctx, labels, "passes", params.passes, || {
            let encoding_start = Instant::now();
            let buffer = encode(&state);
            let encoding_duration = encoding_start.elapsed();

            let submit_start = Instant::now();
            state.device_state.queue.submit([buffer]);
            let submit_duration = submit_start.elapsed();

            state
                .device_state
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();

            vec![encoding_duration, submit_duration]
        }));
    }

    Ok(results)
}
