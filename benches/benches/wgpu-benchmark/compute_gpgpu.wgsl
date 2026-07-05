// Trivial kernel for the GPGPU compute encoding benchmark.
//
// The benchmark measures CPU-side encoding and hazard tracking, which only
// see the bind group layouts and usages, so the kernel deliberately does no
// work and references no bindings; the storage buffer interfaces exist only
// in the pipeline layouts.

@compute @workgroup_size(1)
fn cs_main() {}
