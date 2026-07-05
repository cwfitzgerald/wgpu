#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use wgpu_benchmark::Benchmark;

mod bind_groups;
mod compute_gpgpu;
mod computepass;
mod frame;
mod pass_overhead;
mod renderpass;
mod resource_creation;
mod shader;

struct DeviceState {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl DeviceState {
    fn new() -> Self {
        #[cfg(feature = "tracy")]
        tracy_client::Client::start();

        let base_backend = if cfg!(target_os = "macos") {
            // We don't want to use Molten-VK on Mac.
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(base_backend),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });

        let adapter = block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .unwrap();

        let adapter_info = adapter.get_info();

        println!(
            "  Using adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            label: None,
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        Self {
            adapter_info,
            device,
            queue,
        }
    }

    /// Create a device on the noop backend, which runs all of wgpu-core's validation
    /// and tracking but issues no work to a GPU driver, isolating wgpu's CPU overhead
    /// from driver variance.
    ///
    /// The noop adapter exposes every feature, so `desc` may request features that
    /// real adapters would have to be probed for.
    fn new_noop(desc: &wgpu::DeviceDescriptor) -> Self {
        #[cfg(feature = "tracy")]
        tracy_client::Client::start();

        let (device, queue) = wgpu::Device::noop(desc);
        let adapter_info = device.adapter_info();

        println!(
            "  Using adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        Self {
            adapter_info,
            device,
            queue,
        }
    }

    /// Create a device on the noop backend unless `WGPU_BACKEND` or
    /// `WGPU_ADAPTER_NAME` selects a real backend.
    ///
    /// The noop backend runs all of wgpu-core's validation and tracking without any
    /// driver work, so it is the default, most consistent way to run the benchmarks
    /// that use this. `desc` only applies to the noop device; a real backend requests
    /// all of its adapter's features and limits like [`DeviceState::new`] always does.
    fn new_noop_or_env(desc: &wgpu::DeviceDescriptor) -> Self {
        let backend = std::env::var("WGPU_BACKEND").unwrap_or_default();
        let adapter_name = std::env::var("WGPU_ADAPTER_NAME").unwrap_or_default();
        if (backend.is_empty() || backend.eq_ignore_ascii_case("noop")) && adapter_name.is_empty() {
            Self::new_noop(desc)
        } else {
            Self::new()
        }
    }
}

fn main() {
    let benchmarks = vec![
        Benchmark {
            name: "Device::create_bind_group",
            func: bind_groups::run_bench,
        },
        Benchmark {
            name: "Device::create_buffer",
            func: resource_creation::run_bench,
        },
        Benchmark {
            name: "naga::front",
            func: shader::frontends,
        },
        Benchmark {
            name: "naga::valid",
            func: shader::validation,
        },
        Benchmark {
            name: "naga::compact",
            func: shader::compact,
        },
        Benchmark {
            name: "naga::back",
            func: shader::backends,
        },
        Benchmark {
            name: "Renderpass Encoding",
            func: renderpass::run_bench,
        },
        Benchmark {
            name: "Computepass Encoding",
            func: computepass::run_bench,
        },
        Benchmark {
            name: "GPGPU Compute Encoding",
            func: compute_gpgpu::run_bench,
        },
        Benchmark {
            name: "Frame Encoding",
            func: frame::run_bench,
        },
        Benchmark {
            name: "Pass Overhead Encoding",
            func: pass_overhead::run_bench,
        },
    ];

    wgpu_benchmark::main(benchmarks);
}
