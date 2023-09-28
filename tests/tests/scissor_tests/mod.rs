use wgpu_test::{gpu_test, image, infra::GpuTestConfiguration, TestingContext};

struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

const TEXTURE_HEIGHT: u32 = 2;
const TEXTURE_WIDTH: u32 = 2;
const BUFFER_SIZE: usize = (TEXTURE_WIDTH * TEXTURE_HEIGHT * 4) as usize;

fn scissor_test_impl(ctx: &TestingContext, scissor_rect: Rect, expected_data: [u8; BUFFER_SIZE]) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Offscreen texture"),
        size: wgpu::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("solid_white.wgsl"));

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                entry_point: "vs_main",
                module: &shader,
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                entry_point: "fs_main",
                module: &shader,
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    let readback_buffer = image::ReadbackBuffers::new(&ctx.device, &texture);
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&pipeline);
            render_pass.set_scissor_rect(
                scissor_rect.x,
                scissor_rect.y,
                scissor_rect.width,
                scissor_rect.height,
            );
            render_pass.draw(0..3, 0..1);
        }
        readback_buffer.copy_from(&ctx.device, &mut encoder, &texture);
        ctx.queue.submit(Some(encoder.finish()));
    }
    assert!(readback_buffer.check_buffer_contents(&ctx.device, &expected_data));
}

#[gpu_test]
static SCISSOR_TEST_FULL_RECT: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    scissor_test_impl(
        &ctx,
        Rect {
            x: 0,
            y: 0,
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
        },
        [255; BUFFER_SIZE],
    );
});

#[gpu_test]
static SCISSOR_TEST_EMPTY_RECT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(|ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            [0; BUFFER_SIZE],
        );
    });

#[gpu_test]
static SCISSOR_TEST_EMPTY_RECT_WITH_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_sync(|ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: TEXTURE_WIDTH / 2,
                y: TEXTURE_HEIGHT / 2,
                width: 0,
                height: 0,
            },
            [0; BUFFER_SIZE],
        );
    });

#[gpu_test]
static SCISSOR_TEST_CUSTOM_RECT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(|ctx| {
        let mut expected_result = [0; BUFFER_SIZE];
        expected_result[((3 * BUFFER_SIZE) / 4)..][..BUFFER_SIZE / 4]
            .copy_from_slice(&[255; BUFFER_SIZE / 4]);

        scissor_test_impl(
            &ctx,
            Rect {
                x: TEXTURE_WIDTH / 2,
                y: TEXTURE_HEIGHT / 2,
                width: TEXTURE_WIDTH / 2,
                height: TEXTURE_HEIGHT / 2,
            },
            expected_result,
        );
    });
