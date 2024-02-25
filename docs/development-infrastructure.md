Our test suite is a combination of:
- Standard Rust unit tests declared inline in all crates
- Shader Input/Output snapshot tests, naga frontend, and validation tests in [`naga/tests`](/naga/tests/)
- End-to-End GPU-Enabled tests written against `wgpu` in [`tests/tests`](/tests/tests/)
- GPU-Enabled screenshot tests of our examples declared inline in [`examples/`](/examples/)

All of these tests are hooked up to the standard rust testing infrastructure, though we have some special requirements that come from our use of the GPU in tests.

Our testing harness is designed to automatically run on all available gpus and apis available on the system. To do this while still being fast, we have some custom infrastructure to manage the list of tests.

## Required Software

To run the tests, you will need to have the following software installed:
- [cargo-nextest](https://nexte.st)
  - `cargo install cargo-nextest`
- (Windows) DXC
  - Download binary from `https://github.com/microsoft/DirectXShaderCompiler/releases/latest`
  - Extract the zip file and make sure dxil.dll and dxcompiler.dll are in your PATH.

## Running Tests with `cargo xtask`

`cargo xtask` is a small rust program that helps automate common development tasks.

To run the tests on all available gpus and apis, use the following command:

```sh
cargo xtask test
```

To run the a specific test, you can add an argument just like you would with `cargo nextest`:

```sh
cargo xtask test my_test
```

To run the gpu-enabled tests on a specific gpu on all apis, you can filter for the name of the gpu.

```sh
# Find the adapter name that wgpu uses
cargo list-gpus
# Run using the name of the adapter
cargo xtask test "Intel(R) UHD Graphics 630"
```

To run the tests on a specific api (`Vulkan`, `Metal`, `Dx12`, `Gl`), you can filter for the name of the api:

```sh
cargo xtask test "Vulkan"
```

To filter in multiple ways, you can use cargo nextest's [filtering syntax](https://nexte.st/book/filter-expressions.html):

```sh
# You must use ' for the outer quotes and " for the inner quotes.
cargo xtask test -E 'tests("Vulkan") & test("Intel\(R\) UHD Graphics 630")'
```

## 