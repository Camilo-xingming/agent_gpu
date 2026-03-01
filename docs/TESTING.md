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

## Full regression (audited test_/bench_ suite)

```bash
cd ~/RalphGPU
make regression IVERILOG=/opt/homebrew/bin/iverilog VVP=/opt/homebrew/bin/vvp
```

`make regression` executes the audited must-run targets in dependency-safe order and always prints a summary:

```text
Regression Summary: X/Y passed, Z failed
```

If any target fails, it also prints `Failed targets: ...` and exits non-zero (CI-safe).

### Must-run targets in `make regression` (stable gating set)

`test`
`test_regfile_banked`
`test_bw_scheduler_scoreboard`
`test_sfu`
`test_tensor_fp4_fp8`
`test_tensor_fp4_fp8_frm`
`test_phase2`
`test_sm_v2_core`
`test_tensor_core_fp4`
`test_sm_v2_sched_raw_hazard`
`test_sm_v2_perf_gemm16_ptx`
`test_cron_optimization`
`bench_atomic_minimal`
`bench_app_compile`

### Extended non-gating targets

Run separately when needed:

```bash
make regression REGRESSION_TARGETS="$(REGRESSION_EXTENDED_TARGETS)"
```

Extended set:
`test_vector_add`
`test_multi_sm`
`test_sm_v2_full`
`test_sm_v2_perf_gemm16_wmma_ptx`
`test_sm_v2_perf_gemm64_wgmma_ptx`
`test_sm_v2_perf_tensor`
`test_sm_v2_perf_tensor_multiwarp`
`test_warp_valid_d1`
`test_dual_fetch`
`test_ptx`
`bench_atomics`
`bench_divergence`

### Audit notes

- On March 1, 2026 `test_tensor_core_fp4`, `test_sm_v2_sched_raw_hazard`, and `test_sm_v2_perf_gemm16_ptx` were stabilized and promoted into the must-run gating suite.
- Remaining known extended failures/noise include `test_vector_add`, `test_sm_v2_perf_gemm16_wmma_ptx`, and very long/noisy execution in `test_multi_sm` and `test_ptx`.
- Aggregate wrappers are intentionally excluded from gating to avoid duplicate work: `test_all`, `bench_all`, `test_sm_v2`.
- Alias target `test_sm_v2_perf` remains excluded from gating to avoid duplicate work because it maps to must-run target `test_sm_v2_perf_gemm16_ptx`.
- Non-regression operational targets (`dashboard*`, `perf_report`, `wave`, etc.) are not part of pass/fail gating.

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
