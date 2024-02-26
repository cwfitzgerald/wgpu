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

        // integer clamps must use min/max, not med-of-three
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

    #[rustfmt::skip]
    let uint_clamp_values: &[(u32, u32, u32, u32)] = &[
        (   20,  0,  10,  10),
        (   10,  0,  10,  10),
        (    5,  0,  10,  5),

        // integer clamps must use min/max, not med-of-three
        (    3,  2,  1,   1),
    ];

    for &(input, low, high, output) in uint_clamp_values {
        let test = ShaderTest::new(
            format!("clamp<u32>({input}, {low}, {high}) == {output})"),
            String::from("value: u32, low: u32, high: u32"),
            String::from("output[0] = clamp(input.value, input.low, input.high);"),
            &[input, low, high],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    tests
}

fn count_leading_zeros() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let int_clz_values: &[i32] = &[i32::MAX, 20, 5, 0, -10, -1, i32::MIN];

    for &input in int_clz_values {
        let output = input.leading_zeros();

        let test = ShaderTest::new(
            format!("countLeadingZeros<i32>({input}) == {output})"),
            String::from("value: i32"),
            String::from("output[0] = bitcast<u32>(countLeadingZeros(input.value));"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    let uint_clz_values: &[u32] = &[u32::MAX, 20, 10, 5, 0];

    for &input in uint_clz_values {
        let output = input.leading_zeros();

        let test = ShaderTest::new(
            format!("countLeadingZeros<u32>({input}) == {output})"),
            String::from("value: u32"),
            String::from("output[0] = countLeadingZeros(input.value);"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    tests
}

fn count_one_bits() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let int_cob_values: &[i32] = &[i32::MAX, 20, 5, 0, -10, -1, i32::MIN];

    for &input in int_cob_values {
        let output = input.count_ones();

        let test = ShaderTest::new(
            format!("countOneBits<i32>({input}) == {output})"),
            String::from("value: i32"),
            String::from("output[0] = bitcast<u32>(countOneBits(input.value));"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    let uint_cob_values: &[u32] = &[u32::MAX, 20, 10, 5, 0];

    for &input in uint_cob_values {
        let output = input.count_ones();

        let test = ShaderTest::new(
            format!("countOneBits<u32>({input}) == {output})"),
            String::from("value: u32"),
            String::from("output[0] = countOneBits(input.value);"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    tests
}

fn count_trailing_zeros() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let int_ctz_values: &[i32] = &[i32::MAX, 1 << 30, 20, 5, 0, -1 << 30, -10, -1, i32::MIN];

    for &input in int_ctz_values {
        let output = input.trailing_zeros();

        let test = ShaderTest::new(
            format!("countTrailingZeros<i32>({input}) == {output})"),
            String::from("value: i32"),
            String::from("output[0] = bitcast<u32>(countTrailingZeros(input.value));"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    let uint_ctz_values: &[u32] = &[u32::MAX, 1 << 31, 20, 10, 5, 0];

    for &input in uint_ctz_values {
        let output = input.trailing_zeros();

        let test = ShaderTest::new(
            format!("countTrailingZeros<u32>({input}) == {output})"),
            String::from("value: u32"),
            String::from("output[0] = countTrailingZeros(input.value);"),
            &[input],
            vec![ComparisonValue::U32(output)],
        );

        tests.push(test);
    }

    tests
}

fn extract_bits_unsigned() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let uint_extract_bits_values: &[(u32, u32, u32, u32)] = &[
        // value, offset, bits, expected

        // standard cases
        (0xDE_AD_BE_EF, 0, 8, 0xEF),
        (0xDE_AD_BE_EF, 8, 8, 0xBE),
        (0xDE_AD_BE_EF, 16, 8, 0xAD),
        (0xDE_AD_BE_EF, 24, 8, 0xDE),
        // 0 bits
        (0xDE_AD_BE_EF, 0, 0, 0),
        (0xDE_AD_BE_EF, 8, 0, 0),
        (0xDE_AD_BE_EF, 16, 0, 0),
        (0xDE_AD_BE_EF, 24, 0, 0),
        // offset out of bounds
        (0xDE_AD_BE_EF, 32, 8, 0),
        (0xDE_AD_BE_EF, 48, 8, 0),
        (0xDE_AD_BE_EF, 64, 8, 0),
        // size out of bounds
        (0xDE_AD_BE_EF, 0, 32, 0xDE_AD_BE_EF),
        (0xDE_AD_BE_EF, 8, 32, 0xDE_AD_BE),
        (0xDE_AD_BE_EF, 16, 32, 0xDE_AD),
        (0xDE_AD_BE_EF, 24, 32, 0xDE),

        // Try to catch overflows
        (0xDE_AD_BE_EF, u32::MAX, u32::MAX, 0),
    ];

    for &(value, offset, bits, expected) in uint_extract_bits_values {
        let test = ShaderTest::new(
            format!("extractBits<u32>({value}, {offset}, {bits}) == {expected})"),
            String::from("value: u32, offset: u32, bits: u32"),
            String::from("output[0] = extractBits(input.value, input.offset, input.bits);"),
            &[value, offset, bits],
            vec![ComparisonValue::U32(expected)],
        );

        tests.push(test);
    }

    tests
}

#[gpu_test]
static ABS: GpuTestConfiguration = numeric_bulitin_test(abs);
#[gpu_test]
static CLAMP: GpuTestConfiguration = numeric_bulitin_test(clamp);
#[gpu_test]
static COUNT_LEADING_ZEROS: GpuTestConfiguration = numeric_bulitin_test(count_leading_zeros);
#[gpu_test]
static COUNT_ONE_BITS: GpuTestConfiguration = numeric_bulitin_test(count_one_bits);
#[gpu_test]
static COUNT_TRAILING_ZEROS: GpuTestConfiguration = numeric_bulitin_test(count_trailing_zeros);
#[gpu_test]
static EXTRACT_BITS_UNSIGNED: GpuTestConfiguration = numeric_bulitin_test(extract_bits_unsigned);
