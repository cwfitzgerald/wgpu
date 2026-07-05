//! Emulates the CPU-side encoding cost of dependent-dispatch GPGPU patterns: a
//! parallel-reduction chain, a ping-pong iteration, and indirect dispatches whose
//! arguments are written by a prior dispatch.
//!
//! Unlike the `Computepass Encoding` benchmark, whose dispatches are all independent,
//! every dispatch here reads what the previous one wrote, forcing wgpu-core to detect
//! the hazard and insert a memory barrier between consecutive dispatches. That
//! barrier / hazard-tracking encoding cost is what's under test: the kernels are
//! trivial and the buffers minimal, since the tracker only sees the encoded
//! command-stream shape, never the buffer contents. By default it runs on the noop
//! backend so that all of wgpu-core's validation and tracking runs with zero
//! GPU-driver variance; set `WGPU_BACKEND` (or `WGPU_ADAPTER_NAME`) to run against a
//! real backend like the other benchmarks.

use std::{num::NonZeroU64, time::Instant};

use nanorand::{Rng, WyRand};
use wgpu_benchmark::{iter_many, BenchmarkContext};

use crate::DeviceState;

/// All storage buffers are this small: hazard tracking never looks at buffer
/// contents, so the conceptual element counts scale the *number* of dispatches
/// rather than the buffer sizes.
const BUFFER_SIZE: u64 = 16;

/// Scaling knobs for the GPGPU benchmark.
///
/// Each knob can be overridden on the command line as
/// `--param gpgpu.<knob>=<value>` with the knob name in kebab-case, e.g.
/// `cargo bench -p wgpu-benchmark -- "GPGPU" --param gpgpu.pingpong-iterations=100`.
#[derive(Clone, Copy)]
struct GpgpuParams {
    /// Conceptual input element count for the reduction chain. Each dispatch halves
    /// it, so its ceil-log2 sets the number of dependent dispatches in the chain.
    reduction_elements: u32,
    /// Dependent dispatches alternating between the two ping-pong buffers.
    pingpong_iterations: u32,
    /// Write-args-then-dispatch-indirect pairs, all reusing one args buffer.
    indirect_chains: u32,
}

impl GpgpuParams {
    const DEFAULT: Self = Self {
        reduction_elements: 1 << 20,
        pingpong_iterations: 1000,
        indirect_chains: 64,
    };

    /// A very lightweight configuration so test mode just checks that the
    /// benchmark does not break.
    const TEST: Self = Self {
        reduction_elements: 8,
        pingpong_iterations: 4,
        indirect_chains: 2,
    };

    fn resolve(ctx: &BenchmarkContext) -> Self {
        let d = if ctx.is_test() {
            Self::TEST
        } else {
            Self::DEFAULT
        };
        Self {
            reduction_elements: ctx.param("gpgpu.reduction-elements", d.reduction_elements),
            pingpong_iterations: ctx.param("gpgpu.pingpong-iterations", d.pingpong_iterations),
            indirect_chains: ctx.param("gpgpu.indirect-chains", d.indirect_chains),
        }
    }

    /// Dependent dispatches in the reduction chain: how many times the input must
    /// be halved to reach a single element (at least one).
    fn reduction_levels(&self) -> u32 {
        self.reduction_elements
            .next_power_of_two()
            .trailing_zeros()
            .max(1)
    }
}

struct GpgpuState {
    device_state: DeviceState,
    params: GpgpuParams,

    reduction_pipeline: wgpu::ComputePipeline,
    /// Bind group `k` reads the level-`k` buffer and writes the level-`k+1` buffer,
    /// so consecutive dispatches always hazard on the buffer between them.
    reduction_bind_groups: Vec<wgpu::BindGroup>,

    pingpong_pipeline: wgpu::ComputePipeline,
    /// A→B and B→A over the same two buffers, so every dispatch hazards on both.
    pingpong_bind_groups: [wgpu::BindGroup; 2],

    args_write_pipeline: wgpu::ComputePipeline,
    args_write_bind_group: wgpu::BindGroup,
    indirect_pipeline: wgpu::ComputePipeline,
    indirect_args_buffer: wgpu::Buffer,
}

impl GpgpuState {
    /// Create and prepare all the resources needed for the GPGPU benchmark.
    fn new(params: GpgpuParams) -> Self {
        let device_state = DeviceState::new_noop_or_env(&wgpu::DeviceDescriptor::default());
        let device = &device_state.device;

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let storage_buffer = |label: &str, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: BUFFER_SIZE,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(BUFFER_SIZE),
            },
            count: None,
        };

        // Reduction and ping-pong dispatches read through binding 0 what the previous
        // dispatch wrote through binding 1, producing a read-after-write hazard on
        // every consecutive pair.
        let chain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Chain BGL"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let args_write_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Args Write BGL"),
            entries: &[storage_entry(0, false)],
        });

        let chain_bind_group = |label: &str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &chain_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            })
        };

        // One buffer per reduction level, plus the input.
        let levels = params.reduction_levels() as usize;
        let mut reduction_buffers = Vec::with_capacity(levels + 1);
        for i in 0..=levels {
            reduction_buffers.push(storage_buffer(
                &format!("Reduction Level {i}"),
                wgpu::BufferUsages::STORAGE,
            ));
        }
        random.shuffle(&mut reduction_buffers);
        let reduction_bind_groups: Vec<_> = (0..levels)
            .map(|i| {
                chain_bind_group(
                    &format!("Reduction BG {i}"),
                    &reduction_buffers[i],
                    &reduction_buffers[i + 1],
                )
            })
            .collect();

        let ping_buffer = storage_buffer("Ping", wgpu::BufferUsages::STORAGE);
        let pong_buffer = storage_buffer("Pong", wgpu::BufferUsages::STORAGE);
        let pingpong_bind_groups = [
            chain_bind_group("Ping->Pong BG", &ping_buffer, &pong_buffer),
            chain_bind_group("Pong->Ping BG", &pong_buffer, &ping_buffer),
        ];

        let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Args"),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let args_write_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Args Write BG"),
            layout: &args_write_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: indirect_args_buffer.as_entire_binding(),
            }],
        });

        let module = device.create_shader_module(wgpu::include_wgsl!("compute_gpgpu.wgsl"));

        let pipeline = |label: &str, bgls: &[&wgpu::BindGroupLayout]| {
            let bgls: Vec<_> = bgls.iter().map(|&bgl| Some(bgl)).collect();
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let reduction_pipeline = pipeline("Reduction Pipeline", &[&chain_bgl]);
        let pingpong_pipeline = pipeline("Ping-Pong Pipeline", &[&chain_bgl]);
        let args_write_pipeline = pipeline("Args Write Pipeline", &[&args_write_bgl]);
        // The indirect dispatch binds nothing; the only tracked resource is the
        // args buffer itself, read with INDIRECT usage.
        let indirect_pipeline = pipeline("Indirect Pipeline", &[]);

        Self {
            device_state,
            params,

            reduction_pipeline,
            reduction_bind_groups,

            pingpong_pipeline,
            pingpong_bind_groups,

            args_write_pipeline,
            args_write_bind_group,
            indirect_pipeline,
            indirect_args_buffer,
        }
    }

    fn begin_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        label: &str,
    ) -> wgpu::ComputePass<'a> {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        })
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// A log-depth chain of dependent dispatches with a distinct bind group per
    /// level: level `k` reads what level `k - 1` wrote, so a barrier is inserted
    /// between every pair of consecutive dispatches.
    fn encode_reduction(&self) -> wgpu::CommandBuffer {
        profiling::scope!("Reduction");

        let mut encoder = self.encoder("Reduction");
        {
            let mut compute_pass = self.begin_pass(&mut encoder, "Reduction");
            compute_pass.set_pipeline(&self.reduction_pipeline);
            for bind_group in &self.reduction_bind_groups {
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.finish()
    }

    /// Many dependent dispatches alternating two bind groups over the same two
    /// buffers (A→B, B→A, ...), stressing repeated barrier insertion on the same
    /// small resource set at a high dispatch count.
    fn encode_pingpong(&self) -> wgpu::CommandBuffer {
        profiling::scope!("Ping-Pong");

        let mut encoder = self.encoder("Ping-Pong");
        {
            let mut compute_pass = self.begin_pass(&mut encoder, "Ping-Pong");
            compute_pass.set_pipeline(&self.pingpong_pipeline);
            for i in 0..self.params.pingpong_iterations {
                compute_pass.set_bind_group(0, &self.pingpong_bind_groups[(i % 2) as usize], &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.finish()
    }

    /// Pairs of dispatches where the first writes the indirect-args buffer as
    /// storage and the second consumes it via `dispatch_workgroups_indirect`, so
    /// the args buffer's usage bounces between STORAGE write and INDIRECT read on
    /// every dispatch.
    fn encode_indirect(&self) -> wgpu::CommandBuffer {
        profiling::scope!("Indirect");

        let mut encoder = self.encoder("Indirect");
        {
            let mut compute_pass = self.begin_pass(&mut encoder, "Indirect");
            // The args-write bind group survives the pipeline switches (the indirect
            // pipeline binds nothing), so it is bound once for the whole pass.
            compute_pass.set_bind_group(0, &self.args_write_bind_group, &[]);
            for _ in 0..self.params.indirect_chains {
                compute_pass.set_pipeline(&self.args_write_pipeline);
                compute_pass.dispatch_workgroups(1, 1, 1);
                compute_pass.set_pipeline(&self.indirect_pipeline);
                compute_pass.dispatch_workgroups_indirect(&self.indirect_args_buffer, 0);
            }
        }
        encoder.finish()
    }
}

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<wgpu_benchmark::SubBenchResult>> {
    let params = GpgpuParams::resolve(&ctx);
    anyhow::ensure!(
        [
            params.reduction_elements,
            params.pingpong_iterations,
            params.indirect_chains,
        ]
        .iter()
        .all(|&v| v >= 1),
        "all gpgpu benchmark parameters must be at least 1"
    );

    let state = GpgpuState::new(params);

    // This benchmark hangs on Apple Paravirtualized GPUs. No idea why.
    if state.device_state.adapter_info.name.contains("Paravirtual") {
        anyhow::bail!("Benchmark unsupported on Paravirtualized GPUs");
    }

    println!(
        "  knobs: reduction-elements={} ({} levels) pingpong-iterations={} indirect-chains={}",
        params.reduction_elements,
        params.reduction_levels(),
        params.pingpong_iterations,
        params.indirect_chains,
    );

    type PatternEncoder = fn(&GpgpuState) -> wgpu::CommandBuffer;
    let patterns: [(&str, u32, PatternEncoder); 3] = [
        (
            "Reduction",
            params.reduction_levels(),
            GpgpuState::encode_reduction,
        ),
        (
            "Ping-pong",
            params.pingpong_iterations,
            GpgpuState::encode_pingpong,
        ),
        (
            "Indirect",
            // A write-args dispatch plus an indirect dispatch per chain.
            params.indirect_chains * 2,
            GpgpuState::encode_indirect,
        ),
    ];

    let mut results = Vec::new();
    for (name, dispatches, encode) in patterns {
        let labels = vec![format!("{name} (encode)"), format!("{name} (submit)")];

        results.extend(iter_many(&ctx, labels, "dispatches", dispatches, || {
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
