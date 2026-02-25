# FPGA Demo Flow (Issue #158)

This directory adds a vendor synthesis baseline for a minimal on-board demo.

## Demo RTL

- Shared demo logic: `fpga/common/ralph_gpu_fpga_demo_top.v`
- UART TX helper: `fpga/common/uart_tx.v`
- Behavior:
  - Rotating LED heartbeat
  - Periodic UART banner (`RALPHGPU\r\n`)

## Xilinx (Arty A7-100T)

- Top: `fpga/xilinx/arty_a7_demo_top.v`
- Script: `fpga/xilinx/run_vivado.tcl`
- Constraint: `fpga/xilinx/arty_a7_demo.xdc`

Run synthesis:

```bash
make fpga_xilinx_synth
```

Run implementation + bitstream:

```bash
vivado -mode batch -source fpga/xilinx/run_vivado.tcl -tclargs build/fpga/xilinx 1
```

## Intel (DE10-Nano)

- Top: `fpga/intel/de10_nano_demo_top.v`
- Script: `fpga/intel/run_quartus.tcl`
- Constraint: `fpga/intel/de10_nano_demo.sdc`

Run synthesis:

```bash
make fpga_intel_synth
```

## Local Sanity Simulation

```bash
make test_fpga_demo
```

This verifies the shared demo top (heartbeat + UART activity) before vendor tool runs.
