//! Regression tests for the per-encoder `EncoderVecPool` that recycles the
//! `commands` and `dynamic_offsets` heap vectors across render/compute passes.
//!
//! These drive the *real* record -> finish -> submit path on the noop backend
//! (no GPU/driver needed, exactly like the `Frame` benchmark). That matters
//! because the pool's acquire/release only run during actual pass encoding: the
//! rest of the wgpu-core unit-test suite is pure and never touches it.
//!
//! # Why these tests exist
//!
//! When the pool first shipped, `dynamic_offsets` was released to the pool
//! while still *non-empty*: the submit-time drain loop empties `commands`, but
//! dynamic offsets are consumed *by reference* (each `SetBindGroup` slices into
//! the shared vector) and so are never popped. The next pass then acquired a
//! vec with stale leading offsets, and since encoding reads offsets starting at
//! index 0, it silently applied the *wrong* offset — corruption in release
//! builds, a `debug_assert!` panic in debug. The fix was an explicit
//! `base.dynamic_offsets.clear();` at each release site
//! (`render::encode_render_pass` / `compute::encode_compute_pass`).
//!
//! # How these tests catch a regression
//!
//! Rust's test profile compiles with `debug_assertions` **on**, so the pool's
//! `debug_assert!(vec.is_empty())` guard in `EncoderVecPool::push_bounded` is
//! live here. That guard is the primary tripwire: if the `clear()` is ever
//! removed, a pass that recorded a non-empty dynamic-offset slice will panic at
//! `encoder.finish()` (encoding happens eagerly in `finish()`, not at submit).
//!
//! DO NOT "fix" a failure in these tests by disabling debug assertions — the
//! assertion firing is the whole point.
//!
//! Each test iterates several times so that the *second and later* passes are
//! served from the pool (a pool hit re-using a released vec). The first
//! iteration alone only exercises the pool *miss* (fresh allocation) path and
//! would not observe stale offsets, so the iteration count is load-bearing.

/// Standard `min_uniform_buffer_offset_alignment`; dynamic offsets must be a
/// multiple of it.
const OBJECT_STRIDE: u32 = 256;

/// Iterations of record->finish->submit per test. Must be >= 3: iteration 1 is
/// a pool miss (fresh vecs), iterations 2+ recycle vecs released by the prior
/// iteration, which is the path under test.
const ITERATIONS: usize = 4;

/// A trivial vertex-only shader that reads a uniform through a dynamic offset,
/// so the render pipeline genuinely uses the dynamic-offset bind group.
const RENDER_SHADER: &str = "
    struct Object { data: vec4<f32> }
    @group(0) @binding(0)
    var<uniform> object: Object;

    @vertex
    fn vs_main() -> @builtin(position) vec4<f32> {
        return object.data;
    }
";

/// A trivial compute shader that reads a uniform through a dynamic offset.
const COMPUTE_SHADER: &str = "
    struct Object { data: vec4<f32> }
    @group(0) @binding(0)
    var<uniform> object: Object;

    @group(1) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn cs_main() {
        output = object.data;
    }
";

/// A bind group layout with a single uniform buffer binding that uses a
/// dynamic offset — the ingredient that makes `dynamic_offsets` non-empty.
fn dynamic_offset_bgl(
    device: &wgpu::Device,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dynamic offset BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(OBJECT_STRIDE as u64),
            },
            count: None,
        }],
    })
}

/// Exercise the render-pass release site: record a render pass that binds a
/// dynamic-offset bind group with a NON-EMPTY offset slice, draw, end the pass,
/// finish, and submit — repeatedly, so released command/offset vecs are
/// re-acquired from the pool.
///
/// A regression that stops clearing `dynamic_offsets` before release trips the
/// pool's `debug_assert!(is_empty)` at `encoder.finish()` on the second
/// iteration (the first pool hit).
#[test]
fn render_pass_recycle_clears_dynamic_offsets() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render shader"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    let bgl = dynamic_offset_bgl(&device, wgpu::ShaderStages::VERTEX);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pipeline layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    // Depth-only pipeline: no fragment stage means no color targets to match.
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    });

    // Two objects, so a dynamic offset of `OBJECT_STRIDE` is legal and non-zero.
    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object buffer"),
        size: 2 * OBJECT_STRIDE as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object BG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &object_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(OBJECT_STRIDE as u64),
            }),
        }],
    });

    let depth_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    for iteration in 0..ITERATIONS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            // NON-EMPTY dynamic offset slice: this is what populates
            // `base.dynamic_offsets` and must be cleared before release.
            // Alternate the offset per iteration so a stale leftover offset
            // would be detectable in principle, not just by the assertion.
            let offset = (iteration as u32 % 2) * OBJECT_STRIDE;
            pass.set_bind_group(0, &bind_group, &[offset]);
            pass.draw(0..3, 0..1);
        }
        // Encoding (and therefore the pool release) happens eagerly here.
        let cmd = wgpu_test::valid(&device, || encoder.finish());
        queue.submit([cmd]);
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Exercise the compute-pass release site, which had the same
/// non-empty-`dynamic_offsets` bug as the render path. Same shape as the render
/// test: bind a dynamic-offset group with a non-empty offset slice, dispatch,
/// and iterate so released vecs are recycled.
#[test]
fn compute_pass_recycle_clears_dynamic_offsets() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute shader"),
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    // Group 0 holds the dynamic-offset uniform; group 1 holds the output.
    let object_bgl = dynamic_offset_bgl(&device, wgpu::ShaderStages::COMPUTE);
    let output_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("output BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[Some(&object_bgl), Some(&output_bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object buffer"),
        size: 2 * OBJECT_STRIDE as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object BG"),
        layout: &object_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &object_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(OBJECT_STRIDE as u64),
            }),
        }],
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output buffer"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("output BG"),
        layout: &output_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    for iteration in 0..ITERATIONS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            // NON-EMPTY dynamic offset slice on group 0.
            let offset = (iteration as u32 % 2) * OBJECT_STRIDE;
            pass.set_bind_group(0, &object_bind_group, &[offset]);
            pass.set_bind_group(1, &output_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let cmd = wgpu_test::valid(&device, || encoder.finish());
        queue.submit([cmd]);
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Exercise the drop/early-teardown path: begin a render pass, record commands
/// (including a non-empty dynamic offset), then drop the pass and encoder
/// WITHOUT finishing. The vectors should be freed on drop and never reach the
/// pool dirty. A subsequent clean iteration must still recycle safely — if a
/// dirty vec had leaked into the pool via the drop path, that later pass's
/// `debug_assert!(is_empty)` would fire.
#[test]
fn render_pass_early_drop_does_not_poison_pool() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render shader"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    let bgl = dynamic_offset_bgl(&device, wgpu::ShaderStages::VERTEX);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pipeline layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    });

    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object buffer"),
        size: 2 * OBJECT_STRIDE as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object BG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &object_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(OBJECT_STRIDE as u64),
            }),
        }],
    });

    let depth_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    let pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("render pass"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    // 1. Begin a pass, record commands with a non-empty dynamic offset, then
    //    drop both the pass and the encoder without ever calling finish().
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dropped encoder"),
        });
        let mut pass = encoder.begin_render_pass(&pass_descriptor);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[OBJECT_STRIDE]);
        pass.draw(0..3, 0..1);
        // `pass` and `encoder` drop here, without finish()/submit().
        drop(pass);
        drop(encoder);
    }

    // 2. Now run several normal iterations. These must acquire from the pool
    //    without hitting a poisoned (non-empty) vec.
    for iteration in 0..ITERATIONS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&pass_descriptor);
            pass.set_pipeline(&pipeline);
            let offset = (iteration as u32 % 2) * OBJECT_STRIDE;
            pass.set_bind_group(0, &bind_group, &[offset]);
            pass.draw(0..3, 0..1);
        }
        let cmd = wgpu_test::valid(&device, || encoder.finish());
        queue.submit([cmd]);
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

// -- Encoder-owned, lazily-acquired pool tests ------------------------------
//
// The pool moved from the device onto each command encoder, and a pass now
// acquires its pooled backing storage lazily, on its first recorded command,
// rather than at pass begin. These tests exercise the resulting behavior:
//
//  * an empty pass (zero recorded commands) never acquires and so never pushes
//    anything into the pool (verified indirectly: the pool is not poisoned and
//    a subsequent pass still recycles cleanly);
//  * a run of passes on a *single* encoder recycles within that encoder;
//  * separate encoders have independent pools that cannot cross-contaminate.
//
// The pool depth is not observable from this (separate) crate, so laziness is
// verified via the `debug_assert!(is_empty)` tripwire in `push_bounded` (live
// under the test profile) plus the requirement that empty passes submit
// cleanly on the noop backend (they must still emit their attachment
// load/clear/store ops).

/// Build the depth-only render pipeline + dynamic-offset bind group + depth
/// view used by the render tests, so the several tests below don't each repeat
/// it. Returns `(pipeline, bind_group, depth_view)`.
fn render_fixtures(
    device: &wgpu::Device,
) -> (wgpu::RenderPipeline, wgpu::BindGroup, wgpu::TextureView) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render shader"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });
    let bgl = dynamic_offset_bgl(device, wgpu::ShaderStages::VERTEX);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pipeline layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    });

    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object buffer"),
        size: 2 * OBJECT_STRIDE as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object BG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &object_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(OBJECT_STRIDE as u64),
            }),
        }],
    });

    let depth_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    (pipeline, bind_group, depth_view)
}

/// A color render target view, used to check that an *empty* render pass with
/// an attachment still runs (its clear/load/store ops must execute).
fn color_view(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("color"),
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
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Record + finish + submit one non-empty render pass into a fresh encoder,
/// recording a draw with the given dynamic offset so it acquires pooled
/// storage.
fn run_render_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    depth_view: &wgpu::TextureView,
    offset: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[offset]);
        pass.draw(0..3, 0..1);
    }
    let cmd = wgpu_test::valid(device, || encoder.finish());
    queue.submit([cmd]);
}

/// An empty render pass (with a color attachment) and an empty compute pass
/// must record, finish, and submit cleanly without ever acquiring — hence
/// touching — the encoder's pool. Attachment load/clear/store ops must still
/// run (laziness gates pool acquisition only, never HAL pass execution).
#[test]
fn empty_passes_do_not_touch_pool() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    // Empty render pass with a color attachment: no commands recorded.
    let color = color_view(&device);
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("empty render encoder"),
        });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("empty render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Record nothing.
        }
        let cmd = wgpu_test::valid(&device, || encoder.finish());
        queue.submit([cmd]);
    }

    // Empty compute pass: no commands recorded.
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("empty compute encoder"),
        });
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("empty compute pass"),
                timestamp_writes: None,
            });
            // Record nothing.
        }
        let cmd = wgpu_test::valid(&device, || encoder.finish());
        queue.submit([cmd]);
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// An empty pass followed by a non-empty pass in the *same* encoder: the
/// non-empty pass must record and recycle correctly. The empty pass must not
/// have poisoned the (per-encoder) pool.
#[test]
fn empty_pass_then_nonempty_pass_same_encoder() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (pipeline, bind_group, depth_view) = render_fixtures(&device);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mixed encoder"),
    });
    // Empty pass first (records nothing, never acquires).
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("empty pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    // Then a non-empty pass on the same encoder.
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nonempty pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[0]);
        pass.draw(0..3, 0..1);
    }
    let cmd = wgpu_test::valid(&device, || encoder.finish());
    queue.submit([cmd]);

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Three non-empty render passes recorded on a *single* encoder must recycle
/// their command/offset/arena vectors within that encoder: the 2nd and 3rd
/// passes are served from vectors the previous passes released at encode time.
/// A stale (non-empty) recycled vec would trip `debug_assert!(is_empty)`.
///
/// Passes are encoded in `finish()`, and each pass's release feeds the next
/// pass's acquire, so all three passes recycle within the one `finish()`.
#[test]
fn three_passes_one_encoder_recycle() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (pipeline, bind_group, depth_view) = render_fixtures(&device);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi-pass encoder"),
    });
    for i in 0..3u32 {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        // Alternate the offset so a stale leftover would be detectable.
        pass.set_bind_group(0, &bind_group, &[(i % 2) * OBJECT_STRIDE]);
        pass.draw(0..3, 0..1);
    }
    let cmd = wgpu_test::valid(&device, || encoder.finish());
    queue.submit([cmd]);

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Two separate encoders, each recording a non-empty pass, must work
/// independently. This is the cross-encoder analogue of
/// `render_pass_early_drop_does_not_poison_pool`: dropping one encoder must not
/// corrupt a second, freshly created encoder.
///
/// Note that the encoder pool is *not* strictly per-encoder any more: at
/// teardown a warm pool is returned to a bounded device-level pool, and a fresh
/// encoder seeds its pool from there (see `Device::acquire_vec_pool` /
/// `recycle_vec_pool`). So the second encoder here may well reuse vectors the
/// first encoder's dropped pool released — but those vectors always arrive
/// *empty* (drained on release, and the pool only ever holds empty vecs), so no
/// stale command/offset/arena state can cross between encoders. A regression
/// that let a non-empty vec into the pool would trip `debug_assert!(is_empty)`
/// on the second encoder's first pool hit.
#[test]
fn two_encoders_independent_pools() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (pipeline, bind_group, depth_view) = render_fixtures(&device);

    // First encoder: record a pass with a non-empty dynamic offset, then drop
    // the encoder WITHOUT finishing — its per-encoder pool is dropped with it.
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dropped encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dropped pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[OBJECT_STRIDE]);
            pass.draw(0..3, 0..1);
        }
        drop(encoder);
    }

    // Second, independent encoder: several iterations that recycle within it.
    for iteration in 0..ITERATIONS {
        run_render_pass(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &depth_view,
            (iteration as u32 % 2) * OBJECT_STRIDE,
        );
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// A second pass on the same encoder must see no stale command/offset/arena
/// state left over from the first pass's release. The first pass records a
/// pipeline, a dynamic-offset bind group, and a draw (populating commands, the
/// offset vec, and the bind-group / pipeline arenas); the second pass then
/// recycles those released vectors and must record correctly with a *different*
/// offset. A stale offset or non-empty recycled vec would corrupt the second
/// pass or trip `debug_assert!(is_empty)`.
#[test]
fn second_pass_same_encoder_sees_no_stale_state() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (pipeline, bind_group, depth_view) = render_fixtures(&device);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("two-pass encoder"),
    });
    for (i, offset) in [0u32, OBJECT_STRIDE].into_iter().enumerate() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if i == 0 { "first pass" } else { "second pass" }),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[offset]);
        pass.draw(0..3, 0..1);
    }
    let cmd = wgpu_test::valid(&device, || encoder.finish());
    queue.submit([cmd]);

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Cross-encoder recycling: a warm pool from a *submitted* encoder is returned
/// to the device pool at teardown and seeds the *next* encoder's pool, so the
/// second encoder records against vectors the first encoder grew — with no
/// stale state leaking across the encoder boundary.
///
/// Encoder A records a render pass with a NON-EMPTY dynamic offset and a draw
/// (growing its command/offset/arena vectors), then finishes and submits, which
/// drains those vectors and returns the warm pool to the device. Encoder B is
/// created afterwards (seeding its pool from the device pool, i.e. from A's
/// released vectors) and records a pass with a *different* offset. B must record
/// correctly: the recycled vectors must arrive empty, or a stale offset would
/// corrupt B's pass and a non-empty recycled vec would trip
/// `debug_assert!(is_empty)` at B's `finish()`. Several B iterations run so the
/// device-pool recycle/seed round-trip is exercised repeatedly.
#[test]
fn cross_encoder_recycles_submitted_pool() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (pipeline, bind_group, depth_view) = render_fixtures(&device);

    // Encoder A: record, finish, and SUBMIT — its warm pool retires into the
    // device pool (unlike `two_encoders_independent_pools`, which drops A
    // without finishing).
    run_render_pass(&device, &queue, &pipeline, &bind_group, &depth_view, 0);

    // Encoders B..: each is created after A retired, so it may seed its pool
    // from A's (or a prior B's) released vectors. Start with a different offset
    // than A (`OBJECT_STRIDE` vs A's `0`) so a stale leftover offset would be
    // observable in principle, not only via the emptiness assertion; then
    // alternate within the valid 0/`OBJECT_STRIDE` range of the 2-object buffer.
    for iteration in 0..ITERATIONS {
        run_render_pass(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &depth_view,
            (1 - iteration as u32 % 2) * OBJECT_STRIDE,
        );
    }

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
