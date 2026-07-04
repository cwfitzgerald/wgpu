use std::{collections::HashMap, str::FromStr, time::Duration};

#[derive(Clone, Copy)]
pub enum LoopControl {
    Iterations(u32),
    Time(Duration),
}

impl Default for LoopControl {
    fn default() -> Self {
        LoopControl::Time(Duration::from_secs(2))
    }
}

impl LoopControl {
    pub(crate) fn finished(&self, iterations: u32, elapsed: Duration) -> bool {
        match self {
            LoopControl::Iterations(target) => iterations >= *target,
            LoopControl::Time(target) => elapsed >= *target,
        }
    }
}

pub struct BenchmarkContext {
    pub(crate) override_iters: Option<LoopControl>,
    pub default_iterations: LoopControl,
    pub(crate) is_test: bool,
    pub(crate) params: HashMap<String, String>,
}

impl BenchmarkContext {
    pub fn is_test(&self) -> bool {
        self.is_test
    }

    /// Look up a benchmark-specific parameter passed on the command line as
    /// `--param NAME=VALUE`, falling back to `default` when not provided.
    pub fn param<T>(&self, name: &str, default: T) -> T
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        match self.params.get(name) {
            Some(value) => value
                .parse()
                .unwrap_or_else(|e| panic!("invalid value {value:?} for --param {name}: {e}")),
            None => default,
        }
    }
}
