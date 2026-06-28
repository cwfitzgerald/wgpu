//! A native-only example that hammers the GPU with a configurable fullscreen
//! fragment-shader workload, while letting you tweak the surface present mode and
//! desired maximum frame latency live from the keyboard. It is handy for eyeballing
//! swapchain behaviour (present pacing, tearing, frame latency) under load.
//!
//! It runs on the shared [`crate::framework`]: the workload and pause controls live in
//! [`Example::update`], and the present-mode / frame-latency controls go through the
//! framework's [`Example::reconfigure_surface`] hook (the framework owns the surface).
//!
//! Controls (logged on startup):
//!
//! * `P`             - cycle to the next supported present mode
//! * `Up` / `Down`   - increase / decrease the GPU workload (shader iterations)
//! * `Left` / `Right`- decrease / increase `desired_maximum_frame_latency`
//! * `Space`         - pause / resume animation
//! * `Esc`           - quit (handled by the framework)

use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
};

use crate::framework::Example;

/// Matches the `Params` uniform block in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    resolution: [f32; 2],
    time: f32,
    iterations: u32,
}

/// Workload tuning.
const INITIAL_ITERATIONS: u32 = 2_000;
const ITERATION_STEP: u32 = 500;
const MIN_ITERATIONS: u32 = 1;

/// Frame latency tuning. wgpu clamps the effective value, but we keep the
/// requested value in a sane range so cycling stays intuitive.
const MIN_FRAME_LATENCY: u32 = 1;
const MAX_FRAME_LATENCY: u32 = 16;

struct GpuStress {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    /// Requested by `P`, applied by [`Example::reconfigure_surface`] (which has the supported
    /// present-mode list).
    cycle_present_mode: bool,
    /// Desired `desired_maximum_frame_latency`; diffed against the live config each frame.
    desired_frame_latency: u32,

    iterations: u32,
    paused: bool,
    elapsed: f32,
    last_frame: web_time::Instant,

    /// Surface resolution, fed to the shader; kept current via [`Example::resize`].
    resolution: [f32; 2],
}

impl GpuStress {
    fn adjust_iterations(&mut self, delta: i32) {
        let next = (self.iterations as i64 + delta as i64).max(MIN_ITERATIONS as i64);
        self.iterations = next as u32;
        log::info!("workload -> {} iterations", self.iterations);
    }

    fn adjust_frame_latency(&mut self, delta: i32) {
        let next = (self.desired_frame_latency as i32 + delta)
            .clamp(MIN_FRAME_LATENCY as i32, MAX_FRAME_LATENCY as i32) as u32;
        self.desired_frame_latency = next;
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        log::info!(
            "animation {}",
            if self.paused { "paused" } else { "running" }
        );
    }
}

impl Example for GpuStress {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        log::info!(
            "gpu_stress controls:\n  \
             P            cycle present mode\n  \
             Up / Down    increase / decrease GPU workload (shader iterations)\n  \
             Right / Left increase / decrease desired_maximum_frame_latency\n  \
             Space        pause / resume animation\n  \
             Esc          quit"
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_stress shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_stress params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_stress bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_stress bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_stress pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gpu_stress pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            cycle_present_mode: false,
            desired_frame_latency: config.desired_maximum_frame_latency,
            iterations: INITIAL_ITERATIONS,
            paused: false,
            elapsed: 0.0,
            last_frame: web_time::Instant::now(),
            resolution: [config.width as f32, config.height as f32],
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.resolution = [config.width as f32, config.height as f32];
    }

    fn update(&mut self, event: WindowEvent) {
        let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    logical_key,
                    state: ElementState::Pressed,
                    repeat: false,
                    ..
                },
            ..
        } = event
        else {
            return;
        };

        match logical_key {
            Key::Named(NamedKey::ArrowUp) => self.adjust_iterations(ITERATION_STEP as i32),
            Key::Named(NamedKey::ArrowDown) => self.adjust_iterations(-(ITERATION_STEP as i32)),
            Key::Named(NamedKey::ArrowRight) => self.adjust_frame_latency(1),
            Key::Named(NamedKey::ArrowLeft) => self.adjust_frame_latency(-1),
            Key::Named(NamedKey::Space) => self.toggle_pause(),
            Key::Character(s) if s.eq_ignore_ascii_case("p") => self.cycle_present_mode = true,
            _ => {}
        }
    }

    fn reconfigure_surface(
        &mut self,
        config: &mut wgpu::SurfaceConfiguration,
        present_modes: &[wgpu::PresentMode],
    ) -> bool {
        let mut changed = false;

        if core::mem::take(&mut self.cycle_present_mode) && !present_modes.is_empty() {
            let current = present_modes
                .iter()
                .position(|&mode| mode == config.present_mode)
                .unwrap_or(0);
            let next = present_modes[(current + 1) % present_modes.len()];
            if next != config.present_mode {
                config.present_mode = next;
                log::info!("present mode -> {next:?}");
                changed = true;
            }
        }

        if config.desired_maximum_frame_latency != self.desired_frame_latency {
            config.desired_maximum_frame_latency = self.desired_frame_latency;
            log::info!(
                "desired_maximum_frame_latency -> {}",
                self.desired_frame_latency
            );
            changed = true;
        }

        changed
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let now = web_time::Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if !self.paused {
            self.elapsed += dt;
        }

        let params = Params {
            resolution: self.resolution,
            time: self.elapsed,
            iterations: self.iterations,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_stress encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu_stress pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            // Single fullscreen triangle; the vertex shader generates positions.
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<GpuStress>("gpu_stress");
}
