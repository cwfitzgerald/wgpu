Texture2D<float4> Texture : register(t0);
static const uint Sampler = 1;
StructuredBuffer<uint> sampler_array : register(t0, space255);
SamplerState nagaSamplerArray[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerArray[2048]: register(s0, space1);

float4 test(Texture2D<float4> Passed_Texture, SamplerState Passed_Sampler)
{
    float4 _e5 = Passed_Texture.Sample(Passed_Sampler, float2(0.0, 0.0));
    return _e5;
}

float4 main() : SV_Target0
{
    const float4 _e2 = test(Texture, nagaSamplerArray[Sampler]);
    return _e2;
}
