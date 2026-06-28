struct Params {
    resolution: vec2<f32>,
    time: f32,
    iterations: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Generates a single triangle that covers the whole screen, no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index) - 1);
    let y = f32(i32(vertex_index & 1u) * 2 - 1);
    out.clip_position = vec4<f32>(x * 2.0, y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalized, aspect-corrected coordinates centered at the origin.
    let uv = (in.clip_position.xy / params.resolution - 0.5)
        * vec2<f32>(params.resolution.x / params.resolution.y, 1.0);

    // Slow time so motion is gentle.
    let t = params.time * 0.15;

    // Average many bounded low-frequency waves. `params.iterations` is the
    // GPU-load knob; every term is low-frequency in space and slow in time,
    // so the image stays smooth at any iteration count instead of strobing.
    // The accumulator feeds the output, so the compiler can't fold the loop
    // away.
    var acc = 0.0;
    for (var i = 0u; i < params.iterations; i = i + 1u) {
        let fi = f32(i);
        let freq = 1.0 + fract(fi * 0.013) * 3.0;
        let phase = fi * 0.7;
        acc = acc + sin(uv.x * freq + t + phase) * cos(uv.y * freq - t + phase);
    }
    // Normalize by the trip count so brightness stays stable as the workload
    // changes (no flash when you press Up/Down).
    let v = 0.5 + 0.5 * acc / f32(params.iterations);

    let color = vec3<f32>(
        0.5 + 0.5 * sin(v * 6.2831 + 0.0),
        0.5 + 0.5 * sin(v * 6.2831 + 2.094),
        0.5 + 0.5 * sin(v * 6.2831 + 4.188),
    );
    return vec4<f32>(color, 1.0);
}
