Collection of CPU benchmarks for `wgpu`.

These benchmarks are designed as a first line of defence against performance regressions and generally approximate the performance for users.

## Usage

```sh
# Run all benchmarks
cargo bench -p wgpu-benchmark
# Run a specific benchmarks that contains "filter" in its name
cargo bench -p wgpu-benchmark -- "filter"
```

Use `WGPU_BACKEND` and `WGPU_ADAPTER_NAME` to adjust which device the benchmarks use. [More info on env vars](../README.md#environment-variables).

## Comparing Against a Baseline

To compare the current benchmarks against a baseline, you can use the `--save-baseline` and `--baseline` flags.

For example, to compare v28 against trunk, you could run the following:

```sh
git checkout v28
# Run the baseline benchmarks
cargo bench -p wgpu-benchmark -- --save-baseline "v28"

git checkout trunk
# Run the current benchmarks
cargo bench -p wgpu-benchmark -- --baseline "v28"
```

The current benchmarking framework was added before v28, so comparisons only work after it was added. Before that the same commands will work, but comparison will be done using `criterion`.

## Integration with Profilers

The benchmarks can be run with a profiler to get more detailed information about where time is being spent.
Integrations are available for `tracy` and `superluminal`.

#### Tracy

Tracy is available prebuilt for Windows on [github](https://github.com/wolfpld/tracy/releases/latest/).

```sh
# Once this is running, you can connect to it with the Tracy Profiler
cargo bench -p wgpu-benchmark --features tracy,profiling/profile-with-tracy
```

#### Superluminal

Superluminal is a paid product for windows available [here](https://superluminal.eu/).

```sh
# This command will build the benchmarks, and display the path to the executable
cargo bench -p wgpu-benchmark --features profiling/profile-with-superluminal -- -h

# Have Superluminal run the following command (replacing with the path to the executable)
<path_to_exe> --bench "filter"
```

#### `perf` and others

You can follow the same pattern as above to run the benchmarks with other profilers.
For example, the command line tool `perf` can be used to profile the benchmarks.

```sh
# This command will build the benchmarks, and display the path to the executable
cargo bench -p wgpu-benchmark -- -h

# Run the benchmarks with perf
perf record <path_to_exe> --bench "filter"
```

## Benchmarks

#### `Renderpass Encoding`

This benchmark measures the performance of recording and submitting a render pass with a large
number of draw calls and resources, emulating an intense, more traditional graphics application.
By default it measures 10k draw calls, with 90k total resources.

Within this benchmark, both single threaded and multi-threaded recording are tested, as well as splitting
the render pass into multiple passes over multiple command buffers.
If available, it also tests a bindless approach, binding all textures at once instead of switching
the bind group for every draw call.

#### `Computepass Encoding`

This benchmark measures the performance of recording and submitting a compute pass with a large
number of dispatches and resources.
By default it measures 10k dispatch calls, with 60k total resources, emulating an unusually complex and sequential compute workload.

Within this benchmark, both single threaded and multi-threaded recording are tested, as well as splitting
the compute pass into multiple passes over multiple command buffers.
If available, it also tests a bindless approach, binding all resources at once instead of switching
the bind group for every draw call.
TODO(https://github.com/gfx-rs/wgpu/issues/5766): The bindless version uses only 1k dispatches with 6k resources since it would be too slow for a reasonable benchmarking time otherwise.

#### `GPGPU Compute Encoding`

This benchmark emulates the CPU-side encoding cost of dependent-dispatch GPGPU patterns. Unlike
`Computepass Encoding`, whose dispatches are all independent, every dispatch here reads what the
previous one wrote, forcing wgpu-core to detect the hazard and insert a memory barrier between
consecutive dispatches — that barrier / hazard-tracking encoding cost is what's under test. Three
patterns are measured, each with its own encode and submit timings:

- **Reduction** — a log-depth chain of dependent dispatches with a distinct bind group per level,
  where level k reads the buffer level k-1 wrote.
- **Ping-pong** — many dependent dispatches alternating two storage buffers (A→B, B→A, ...),
  stressing repeated barrier insertion on the same small resource set.
- **Indirect** — `dispatch_workgroups_indirect` where the args buffer is written by a prior
  dispatch, exercising indirect-args tracking and INDIRECT-usage hazards.

The kernels are trivial and the buffers minimal: only the encoded command-stream shape and the
resulting hazards matter. Like the frame benchmark it defaults to the noop backend; set
`WGPU_BACKEND` (or `WGPU_ADAPTER_NAME`) to run it against a real backend instead.

The workload can be scaled with repeatable `--param` flags:

```sh
cargo bench -p wgpu-benchmark -- "GPGPU" --param gpgpu.pingpong-iterations=10000
```

| Knob                        | Default | Pressures                                              |
| --------------------------- | ------- | ------------------------------------------------------ |
| `gpgpu.reduction-elements`  | 1048576 | Reduction chain depth (ceil-log2 dependent dispatches) |
| `gpgpu.pingpong-iterations` | 1000    | Repeated barriers on the same two buffers              |
| `gpgpu.indirect-chains`     | 64      | Indirect-args hazards (2 dispatches per chain)         |

#### `Frame Encoding`

This benchmark emulates the CPU-side, single-threaded encoding work of one frame of an advanced
game renderer: a depth prepass, shadow cascade passes, a g-buffer pass, an SSAO compute chain, a
lighting pass, a transparency pass, a bloom downsample/upsample chain, and a chain of
post-processing passes.

All resources are 1x1 and the shaders are trivial: the benchmark measures command encoding, bind
group / pipeline state tracking, pass setup/teardown, and submission — not GPU work. Unlike the
other benchmarks it defaults to the noop backend, which runs all of wgpu-core's validation and
tracking with no driver calls, giving the most consistent numbers. Set `WGPU_BACKEND` (or
`WGPU_ADAPTER_NAME`) to run it against a real backend instead.

By default the frame encodes ~9.5k draws (plus 3 compute dispatches) across 24 passes: per-object
data is bound with dynamic uniform offsets on every draw, materials come from a pool of 1000
distinct bind groups rebound every ~3 draws (every draw for transparents, as if depth-sorted),
and the g-buffer pass switches between 32 pipeline variants every ~64 draws. Timings are reported
per frame phase, plus the total encoding time and the submit.

Each run encodes the same frame in three configurations, so their results can be compared
directly:

- **`[Bindful, per-object VB]`** — the baseline: every draw binds its own vertex/index buffers,
  so the tracker sees new buffers on every draw.
- **`[Bindful, shared VB]`** — identical material and pipeline churn, but one shared
  vertex/index buffer is bound once per pass, as an engine with suballocated meshes would,
  eliminating per-draw buffer binds and their tracker churn.
- **`[Bindless, shared VB]`** — additionally binds every material texture at once in a single
  `binding_array` bind group per pass, dropping the per-draw material rebinds in the g-buffer and
  transparency passes. Like the renderpass/computepass benchmarks, this configuration only runs
  if the device supports the required binding-array features (the default noop backend does).

The workload can be scaled with repeatable `--param` flags to pressure specific subsystems:

```sh
cargo bench -p wgpu-benchmark -- "Frame" --param frame.draws-per-material=1 --param frame.materials=4000
```

| Knob                             | Default | Pressures                                      |
| -------------------------------- | ------- | ---------------------------------------------- |
| `frame.opaque-draws`             | 2500    | Per-draw encoding (prepass + g-buffer)         |
| `frame.shadow-cascades`          | 4       | Pass count, depth-only draw volume             |
| `frame.shadow-draws-per-cascade` | 1000    | Per-draw encoding                              |
| `frame.transparent-draws`        | 500     | High-churn draws (material rebound every draw) |
| `frame.materials`                | 1000    | Number of distinct bind groups                 |
| `frame.draws-per-material`       | 3       | Bind group rebind frequency                    |
| `frame.gbuffer-pipelines`        | 32      | Number of distinct pipelines                   |
| `frame.draws-per-pipeline`       | 64      | Pipeline switch frequency                      |
| `frame.transparent-pipelines`    | 8       | Number of distinct pipelines                   |
| `frame.bloom-mips`               | 6       | Pass count (`2 * mips - 1` bloom passes)       |
| `frame.post-passes`              | 4       | Pass count, pipeline count                     |

#### `Device::create_buffer`

This benchmark measures the performance of creating large buffers.

#### `Device::create_bind_group`

This benchmark measures the performance of creating large bind groups of 5 to 50,000 resources.

#### `naga::back`, `naga::compact`, `naga::front`, and `naga::valid`

These benchmark measures the performance of naga parsing, validating, and generating shaders.
