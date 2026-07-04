// Trivial shaders for the frame benchmark.
//
// The benchmark measures CPU-side encoding and tracking overhead, so these
// shaders deliberately do no meaningful work and reference no bindings; the
// bind group interfaces exist only in the pipeline layouts.

struct GeometryInput {
    @location(0) position: vec4<f32>,
    @location(1) data: vec4<f32>,
}

@vertex
fn vs_geometry(input: GeometryInput) -> @builtin(position) vec4<f32> {
    return input.position + input.data;
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) % 2) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @location(3) emissive: vec4<f32>,
}

@fragment
fn fs_gbuffer() -> GBufferOutput {
    return GBufferOutput(vec4<f32>(1.0), vec4<f32>(0.5), vec4<f32>(0.25), vec4<f32>(0.0));
}

@fragment
fn fs_color() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@compute @workgroup_size(1)
fn cs_main() {}
