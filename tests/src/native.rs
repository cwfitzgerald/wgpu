#![cfg(not(target_arch = "wasm32"))]

use std::{future::Future, pin::Pin};

use parking_lot::Mutex;

use crate::{
    config::GpuTestConfiguration, params::TestInfo, report::AdapterReport, run::execute_test,
};

type NativeTestFuture = Pin<Box<dyn Future<Output = ()> + Send + Sync>>;

struct NativeTest {
    name: String,
    future: NativeTestFuture,
}

impl NativeTest {
    fn from_configuration(
        config: GpuTestConfiguration,
        adapter: &AdapterReport,
        adapter_index: usize,
    ) -> Self {
        let backend = &adapter.info.backend;
        let device_name = &adapter.info.name;

        let test_info = TestInfo::from_configuration(&config, adapter);

        let full_name = format!(
            "[{running_msg}] [{backend:?}/{device_name}/{adapter_index}] {base_name}",
            running_msg = test_info.running_msg,
            base_name = config.name,
        );
        Self {
            name: full_name,
            future: Box::pin(async move {
                execute_test(config, Some(test_info), adapter_index).await;
            }),
        }
    }

    pub fn into_trial(self) -> libtest_mimic::Trial {
        libtest_mimic::Trial::test(self.name, || {
            pollster::block_on(self.future);
            Ok(())
        })
    }
}

pub static TEST_LIST: Mutex<Vec<crate::GpuTestConfiguration>> = Mutex::new(Vec::new());

pub type MainResult = anyhow::Result<()>;

pub fn main() -> MainResult {
    use anyhow::Context;

    use crate::report::GpuReport;

    let config_text =
        &std::fs::read_to_string(format!("{}/../.gpuconfig", env!("CARGO_MANIFEST_DIR")))
            .context("Failed to read .gpuconfig, did you run the tests via `cargo xtask test`?")?;
    let report = GpuReport::from_json(config_text).context("Could not pare .gpuconfig JSON")?;

    let mut test_guard = TEST_LIST.lock();
    execute_native(test_guard.drain(..).flat_map(|test| {
        report
            .devices
            .iter()
            .enumerate()
            .map(move |(adapter_index, adapter)| {
                NativeTest::from_configuration(test.clone(), adapter, adapter_index)
            })
    }));

    Ok(())
}

fn execute_native(tests: impl IntoIterator<Item = NativeTest>) {
    let args = libtest_mimic::Arguments::from_args();
    let trials = tests.into_iter().map(NativeTest::into_trial).collect();

    libtest_mimic::run(&args, trials).exit_if_failed();
}
