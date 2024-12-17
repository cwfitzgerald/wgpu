struct VertexOutput {
    float2 uv : LOC0;
    float4 position : SV_Position;
};

static const float c_scale = 1.2;

Texture2D<float4> u_texture : register(t0);
static const uint u_sampler = 1;
StructuredBuffer<uint> sampler_array : register(t0, space255);
SamplerState nagaSamplerArray[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerArray[2048]: register(s0, space1);

struct VertexOutput_vert_main {
    float2 uv_2 : LOC0;
    float4 position : SV_Position;
};

struct FragmentInput_frag_main {
    float2 uv_3 : LOC0;
};

VertexOutput ConstructVertexOutput(float2 arg0, float4 arg1) {
    VertexOutput ret = (VertexOutput)0;
    ret.uv = arg0;
    ret.position = arg1;
    return ret;
}

VertexOutput_vert_main vert_main(float2 pos : LOC0, float2 uv : LOC1)
{
    const VertexOutput vertexoutput = ConstructVertexOutput(uv, float4((c_scale * pos), 0.0, 1.0));
    const VertexOutput_vert_main vertexoutput_1 = { vertexoutput.uv, vertexoutput.position };
    return vertexoutput_1;
}

float4 frag_main(FragmentInput_frag_main fragmentinput_frag_main) : SV_Target0
{
    float2 uv_1 = fragmentinput_frag_main.uv_3;
    float4 color = u_texture.Sample(nagaSamplerArray[u_sampler], uv_1);
    if ((color.w == 0.0)) {
        discard;
    }
    float4 premultiplied = (color.w * color);
    return premultiplied;
}

float4 fs_extra() : SV_Target0
{
    return float4(0.0, 0.5, 0.0, 0.5);
}
