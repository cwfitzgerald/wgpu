use std::sync::Arc;

use anyhow::Context;

pub use params::{cpu_test, GpuTest};

use crate::infra::params::CpuTest;

mod params;
mod report;
mod single;

pub type MainResult = anyhow::Result<()>;

pub fn main<const GPU_TEST_COUNT: usize, const CPU_TEST_COUNT: usize>(
    gpu_test_list: [Arc<dyn params::GpuTest + Send + Sync>; GPU_TEST_COUNT],
    cpu_test_list: [CpuTest; CPU_TEST_COUNT],
) -> MainResult {
    let args = libtest_mimic::Arguments::from_args();

    let config_text =
        &std::fs::read_to_string(format!("{}/../.gpuconfig", env!("CARGO_MANIFEST_DIR")))
            .context("Failed to read .gpuconfig, did you run the tests via `cargo xtask test`?")?;
    let report =
        report::GpuReport::from_json(config_text).context("Could not pare .gpuconfig JSON")?;

    // Gpu tests
    let mut tests: Vec<_> = gpu_test_list
        .into_iter()
        .flat_map(|test| {
            report
                .devices
                .iter()
                .enumerate()
                .map(move |(adapter_index, adapter)| {
                    single::run_test(test.clone(), adapter, adapter_index)
                })
        })
        .collect();
    // Cpu tests
    tests.extend(cpu_test_list.into_iter().map(|test| {
        libtest_mimic::Trial::test(test.name(), move || {
            test.call();
            Ok(())
        })
    }));

    libtest_mimic::run(&args, tests).exit_if_failed();

    Ok(())
}
