# RalphGPU Non-Goals

## Explicit Exclusions (from prompt.md)

### Graphics Pipeline
- **Rasterization**: Not implementing raster units, pixel shaders, or fragment processing.
- **Ray Tracing (RT)**: No RT cores or BVH traversal hardware.
- **Reason**: Focus is on compute (GPGPU), not graphics rendering.

### Display Engine
- **Display Controllers**: No HDMI/DP encoders, display timing generators.
- **Framebuffer Management**: No scanout, CRTC, or plane management.
- **Reason**: Out of scope for compute-focused design.

### Video Codecs
- **Hardware Encoders/Decoders**: No NVENC/NVDEC equivalents.
- **Exception**: Video instructions (DP4A, etc.) ARE in scope for compute purposes.
- **Reason**: Codec engines are separate IP blocks, not compute core.

### Proprietary Microarchitecture
- **NVIDIA Internal Details**: Not referencing proprietary scheduler algorithms, cache policies beyond PTX-visible behavior.
- **Reason**: ISA authority is public PTX documentation only.

## Performance vs. Correctness
- **Cycle-Accurate Optimization**: Secondary to functional correctness.
- **Reason**: Verification is the primary goal; performance tuning comes after correctness.

## Out-of-Scope PTX Features
- **Driver-Level Operations**: cuLaunchKernel, memory allocation APIs.
- **Reason**: These are software stack, not hardware implementation.

## Physical Implementation
- **Synthesis Optimization**: Not optimizing for specific FPGA/ASIC targets.
- **Power Management**: No clock gating, power domains.
- **Reason**: Focus on architectural correctness, not physical implementation.
