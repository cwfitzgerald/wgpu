// Trivial bindless shaders for the frame benchmark.
//
// Same entry points as frame.wgsl, but every material texture is exposed
// through one binding_array so a single bind group can hold all materials at
// once. The textureLoads exist only to make the binding_array part of the
// pipeline interface; the benchmark still measures CPU-side encoding and
// tracking overhead, not GPU work.

enable wgpu_binding_array;

@group(1) @binding(0)
var material_textures: binding_array<texture_2d<f32>>;

struct GeometryInput {
    @location(0) position: vec4<f32>,
    @location(1) data: vec4<f32>,
}

@vertex
fn vs_geometry(input: GeometryInput) -> @builtin(position) vec4<f32> {
    return input.position + input.data;
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @location(3) emissive: vec4<f32>,
}

@fragment
fn fs_gbuffer() -> GBufferOutput {
    let albedo = textureLoad(material_textures[0], vec2u(0), 0);
    return GBufferOutput(albedo, vec4<f32>(0.5), vec4<f32>(0.25), vec4<f32>(0.0));
}

@fragment
fn fs_color() -> @location(0) vec4<f32> {
    return textureLoad(material_textures[0], vec2u(0), 0);
}
