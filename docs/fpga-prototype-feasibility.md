# Issue #158 FPGA Prototype Feasibility (Xilinx + Intel)

## Scope
- Target: vendor synthesis closure + a minimal on-board demo.
- This document covers feasibility only (board/toolchain choice + implementation path).

## Current RalphGPU Baseline (Repo Facts)
- Build system has a generic synthesis gate only: `make synth` runs Yosys with top `ralph_gpu_top`.
- No existing Vivado/Quartus project scripts, constraints, or board wrappers.
- Config supports profile-based scaling (`GPU_PROFILE_LITE/BALANCED/HPC`) from `rtl/gpu_defines.vh` and `rtl/memory_config.vh`.
- In attempted front-end synthesis, Yosys reports many memories being expanded into register arrays (example: `register_file`, `shared_memory`, `tensor_core`, `async_copy_engine`, `mbarrier_unit`). This is a major FPGA area/timing risk.

## Toolchain Feasibility

### Xilinx (AMD)
- Recommended tool: Vivado Design Suite Standard Edition (free tier).
- Device support includes Artix-7 family (sufficient for low-cost prototype boards).
- Supported host OS for Vivado 2026.1 is Windows/Linux (RHEL/Ubuntu variants listed); macOS is not listed.

Conclusion:
- Xilinx flow is feasible, but must run on Linux/Windows build host (not native macOS).

### Intel
- Recommended tool: Quartus Prime Lite Edition (license-free).
- Cyclone V support is available in Quartus Prime Standard/Lite editions.
- Quartus 25.3 support matrix is Windows/Linux oriented (no macOS support listed).

Conclusion:
- Intel flow is feasible, with the same host OS constraint (Linux/Windows).

## Board Recommendation

### Xilinx candidate (recommended)
- Board: Digilent Arty A7-100T.
- Why:
  - Uses Artix-7 (Vivado Standard-supported family).
  - Has 256MB DDR3L and sufficient peripherals for a bring-up demo (UART/JTAG, GPIO/Pmod).
  - Entry-level cost and broad community examples.

### Intel candidate (recommended)
- Board: Terasic DE10-Nano.
- Why:
  - Cyclone V SoC target with mature Quartus Lite flow.
  - Includes 1GB DDR3 and enough I/O for a simple demo pipeline.
  - Affordable board class for prototype validation.

## Capacity/Risk Assessment
- Full `ralph_gpu_top` is unlikely to close quickly on entry/mid-range FPGA without pruning because:
  - Memory-heavy blocks are currently inferred in a style that expands to flip-flops in Yosys front-end checks.
  - Current architecture includes many high-end units (HBM path, tensor path, wide memory subsystems) not required for first demo.
- Therefore, a phased "FPGA_DEMO" configuration is required for practical first silicon on both vendors.

## Proposed Demo Definition (Phase-1 Success Criteria)
- Single-SM reduced build (start from LITE profile, then further trim if needed).
- Basic kernel execution proof (vector add or simple ALU/PTX program).
- Observable outputs via UART/JTAG memory readback.
- Timing target: modest Fmax (25-50 MHz) first, then iterate.

## Implementation Plan After Feasibility Sign-off
1. Add vendor build trees:
   - `fpga/xilinx/` (Vivado Tcl project + XDC for Arty A7)
   - `fpga/intel/` (Quartus project Tcl/QSF + SDC for DE10-Nano)
2. Add `FPGA_DEMO` compile profile:
   - reduce SM count and memory footprint
   - gate non-essential units for first demo
   - convert major SRAM structures to vendor BRAM-friendly coding/templates
3. Add board top wrappers:
   - clock/reset + UART status
   - DDR stub/simple memory model for first pass
4. Run synthesis/P&R on both targets and publish utilization/timing reports.

## External References
- AMD Vivado installation/system support (2026.1): https://www.amd.com/en/support/downloads/installer-info/archive-installation.html
- AMD Vivado editions/device support (includes Artix-7 in Standard): https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html
- Digilent Arty A7 board page: https://digilent.com/shop/arty-a7-artix-7-fpga-development-board/
- Intel Quartus Prime Lite FAQ (license-free): https://www.intel.com/content/www/us/en/support/programmable/support-resources/design-software/sof-qts-free.html
- Intel device support summary (Cyclone V with Standard/Lite): https://www.intel.com/content/www/us/en/support/programmable/articles/000089620.html
- Intel Quartus support matrix (current Windows/Linux support): https://www.intel.com/content/www/us/en/support/programmable/support-resources/design-software/os-support.html
- Terasic DE10-Nano board page: https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046
