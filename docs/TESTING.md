# Testing Guide

This project uses Icarus Verilog for simulation-based validation.

## Prerequisites

- Icarus Verilog
- vvp runtime
- Optional: GTKWave

On some hosts, `iverilog`/`vvp` are not in non-interactive PATH.
Use absolute paths if needed:

```bash
/opt/homebrew/bin/iverilog
/opt/homebrew/bin/vvp
```

## Fast sanity

```bash
cd ~/RalphGPU
make test IVERILOG=/opt/homebrew/bin/iverilog VVP=/opt/homebrew/bin/vvp
```

This runs core unit tests (ALU, MUL, decoder, regfile, shared memory, warp scheduler).

## Tensor multiwarp perf test (current hotspot)

```bash
cd ~/RalphGPU
/opt/homebrew/bin/iverilog -g2012 -Irtl -DSM_V2 -DSIMULATION \
  -o build/tb_multiwarp.vvp tb/tb_sm_v2_perf_tensor_multiwarp.v rtl/*.v
cd build
/opt/homebrew/bin/vvp tb_multiwarp.vvp
```

Track these metrics from test output:

- `Writebacks`
- `IPC`
- `PASS/FAIL`

## Current expected behavior (in-flight)

- Unit tests: PASS
- Tensor multiwarp branch under active optimization; see linked issue/PR for latest acceptance criteria.

## Troubleshooting

- `command not found: iverilog` / `vvp`: use absolute tool paths.
- `gh command not found` while git push works: git credential and GitHub CLI are separate.
  Use `/opt/homebrew/bin/gh` explicitly on hosts where PATH is minimal.
