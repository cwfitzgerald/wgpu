use std::num::{NonZeroU32, NonZeroU64};

use wgpu::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static BINDING_ARRAY_UNIFORM_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::BUFFER_BINDING_ARRAY
                    | Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            )
            .limits(Limits {
                max_uniform_buffers_per_shader_stage: 17,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_buffers(ctx, BufferType::Uniform).await });

#[gpu_test]
static BINDING_ARRAY_STORAGE_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::BUFFER_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .limits(Limits {
                max_storage_buffers_per_shader_stage: 17,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_buffers(ctx, BufferType::Storage).await });

enum BufferType {
    Storage,
    Uniform,
}

/// Test to see how texture bindings array work and additionally making sure
/// that non-uniform indexing is working correctly.
///
/// If non-uniform indexing is not working correctly, AMD will produce the wrong
/// output due to non-native support for non-uniform indexing within a WARP.
async fn binding_array_buffers(ctx: TestingContext, buffer_type: BufferType) {
    let storage_mode = match buffer_type {
        BufferType::Storage => "storage",
        BufferType::Uniform => "uniform",
    };

    let shader = r#"
        struct ImAU32 {
            value: u32,
            _padding: u32,
            _padding2: u32,
            _padding3: u32,
        };

        @group(0) @binding(0)
        var<{storage_mode}> buffers: binding_array<ImAU32>;

        @group(0) @binding(1)
        var<storage, read_write> output_buffer: array<u32>;

        @compute
        @workgroup_size(16, 1, 1)
        fn compMain(@builtin(global_invocation_id) id: vec3u) {
            output_buffer[id.x] = buffers[id.x].value;
        }
    "#;
    let shader = shader.replace("{storage_mode}", storage_mode);

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binding Array Buffer"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let image = image::load_from_memory(include_bytes!("../3x3_colors.png")).unwrap();
    // Resize image to 4x4
    let image = image
        .resize_exact(4, 4, image::imageops::FilterType::Gaussian)
        .into_rgba8();

    // Create one buffer for each pixel
    let mut buffers = Vec::with_capacity(64);
    for data in image.pixels() {
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            usage: match buffer_type {
                BufferType::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST,
                BufferType::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
            // 16 to allow padding for uniform buffers
            size: 16,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut()[0..4].copy_from_slice(&data.0);
        buffer.unmap();
        buffers.push(buffer);
    }

    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 4 * 4 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: match buffer_type {
                            BufferType::Storage => BufferBindingType::Storage { read_only: true },
                            BufferType::Uniform => BufferBindingType::Uniform,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                    },
                    count: Some(NonZeroU32::new(16).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

    let buffer_references: Vec<_> = buffers
        .iter()
        .map(|b| b.as_entire_buffer_binding())
        .collect();

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::BufferArray(&buffer_references),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("compMain"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut render_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.dispatch_workgroups(1, 1, 1);
    }

    let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 4 * 4 * 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4 * 4 * 4);

    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});

    ctx.device.poll(Maintain::Wait);

    let data = slice.get_mapped_range();

    assert_eq!(&data[..], &*image);
}
