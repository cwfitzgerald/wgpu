//! Tests that requesting a device automatically enables the features implied
//! (per the WebGPU spec) by the requested features.

fn device_features(required_features: wgpu::Features) -> wgpu::Features {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    });
    device.features()
}

#[test]
fn no_implied_features() {
    let features = device_features(wgpu::Features::empty());
    assert_eq!(features, wgpu::Features::empty());
}

#[test]
fn tier1_implies_rg11b10ufloat_renderable() {
    let features = device_features(wgpu::Features::TEXTURE_FORMATS_TIER1);
    assert_eq!(
        features,
        wgpu::Features::TEXTURE_FORMATS_TIER1 | wgpu::Features::RG11B10UFLOAT_RENDERABLE
    );
}

#[test]
fn tier2_implies_tier1() {
    let features = device_features(wgpu::Features::TEXTURE_FORMATS_TIER2);
    assert_eq!(
        features,
        wgpu::Features::TEXTURE_FORMATS_TIER2
            | wgpu::Features::TEXTURE_FORMATS_TIER1
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
    );
}
