//! Regression tests for the per-device [`CommandVecPool`] that recycles the
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
//! `debug_assert!(vec.is_empty())` guard in `CommandVecPool::push_bounded` is
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
