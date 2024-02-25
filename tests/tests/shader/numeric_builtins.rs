use crate::shader::{shader_input_output_test, ComparisonValue, InputStorageType, ShaderTest};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

fn numeric_bulitin_test(create_test: fn() -> Vec<ShaderTest>) -> GpuTestConfiguration {
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_async(move |ctx| {
            shader_input_output_test(ctx, InputStorageType::Storage, create_test())
        })
}

fn abs() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[rustfmt::skip]
    let float_abs: &[(f32, f32)] = &[
        // value, abs(value)
        (  20.0,  20.0),
        ( -10.0,  10.0),
        (  -0.0,  0.0),
        ( -f32::MIN_POSITIVE, f32::MIN_POSITIVE),
        (  f32::NEG_INFINITY, f32::INFINITY),
    ];

    for &(input, output) in float_abs {
        let test = ShaderTest::new(
            format!("abs<f32>({input}) == {output})"),
            String::from("value: f32"),
            String::from("output[0] = bitcast<u32>(abs(input.value));"),
            &[input],
            vec![ComparisonValue::F32(output)],
        );

        tests.push(test);
    }

    #[rustfmt::skip]
    let int_abs: &[(i32, i32)] = &[
        // value, abs(value)
        (  20,  20),
        ( -10,  10),
        (  i32::MIN, i32::MIN),
    ];

    for &(input, output) in int_abs {
        let test = ShaderTest::new(
            format!("abs<i32>({input}) == {output})"),
            String::from("value: i32"),
            String::from("output[0] = bitcast<u32>(abs(input.value));"),
            &[input],
            vec![ComparisonValue::I32(output)],
        );

        tests.push(test);
    }

    #[rustfmt::skip]
    let uint_abs: &[(u32, u32)] = &[
        // value, abs(value)
        (  20,  20),
        (  10,  10),
    ];

    for &(input, output) in uint_abs {
        let test = ShaderTest::new(
            format!("abs<u32>({input}) == {output})"),
            String::from("value: u32"),
            String::from("output[0] = abs(input.value);"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    tests
}

fn clamp() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[rustfmt::skip]
    let float_clamp_values: &[(f32, f32, f32, &[f32])] = &[
        // value - low - high - valid outputs

        // normal clamps
        (   20.0,  0.0,  10.0,  &[10.0]),
        (  -10.0,  0.0,  10.0,  &[0.0]),
        (    5.0,  0.0,  10.0,  &[5.0]),

        // med-of-three or min/max
        (    3.0,  2.0,  1.0,   &[1.0, 2.0]),
    ];

    for &(input, low, high, output) in float_clamp_values {
        let test = ShaderTest::new(
            format!("clamp<f32>({input}, {low}, {high}) == {output:?})"),
            String::from("value: f32, low: f32, high: f32"),
            String::from("output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));"),
            &[input, low, high],
            output.iter().map(|&f| ComparisonValue::F32(f)).collect(),
        );

        tests.push(test);
    }

    #[rustfmt::skip]
    let int_clamp_values: &[(i32, i32, i32, i32)] = &[
        (   20,  0,  10,  10),
        (  -10,  0,  10,  0),
        (    5,  0,  10,  5),

        // integer clamps must use min/max
        (    3,  2,  1,   1),
    ];

    for &(input, low, high, output) in int_clamp_values {
        let test = ShaderTest::new(
            format!("clamp<i32>({input}, {low}, {high}) == {output})"),
            String::from("value: i32, low: i32, high: i32"),
            String::from("output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));"),
            &[input, low, high],
            vec![ComparisonValue::I32(output)],
        );

        tests.push(test);
    }

    tests
}

#[gpu_test]
static ABS: GpuTestConfiguration = numeric_bulitin_test(abs);
#[gpu_test]
static CLAMP: GpuTestConfiguration = numeric_bulitin_test(clamp);
