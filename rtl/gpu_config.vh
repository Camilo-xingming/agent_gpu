//============================================================================
// RalphGPU - Profile Configuration
// Select a profile at compile time:
//   -DGPU_PROFILE_LITE
//   -DGPU_PROFILE_BALANCED
//   -DGPU_PROFILE_HPC
//
// If no profile is defined, gpu_defines/memory_config fall back to defaults.
//============================================================================

`ifndef GPU_CONFIG_VH
`define GPU_CONFIG_VH

// Profile selector (set via compiler defines or Makefile GPU_PROFILE)
// `define GPU_PROFILE_LITE
// `define GPU_PROFILE_BALANCED
// `define GPU_PROFILE_HPC

`endif
