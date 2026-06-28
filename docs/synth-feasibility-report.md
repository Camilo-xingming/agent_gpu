# Yosys Synthesis Feasibility Report

This report identifies non-synthesizable constructs and Yosys-specific errors across the RTL modules.

## Summary
- Total Modules: 67
- Passed: 52
- Failed: 15

## Issue Categories

### `Timeout Error`
Affected modules:
- `rtl/dpx_unit.v`
- `rtl/l1_data_cache_optimized.v`
- `rtl/tensor_memory.v`
- `rtl/l1_data_cache.v`
- `rtl/memory_controller.v`
- `rtl/l2_cache.v`
- `rtl/branch_predictor.v`
- `rtl/register_file.v`
- `rtl/memory_coalescing_unit.v`
- `rtl/wgmma.v`

### `rtl/memory_controller_hbm.v:227: ERROR: 2nd expression of procedural for-loop is not constant!`
Affected modules:
- `rtl/memory_controller_hbm.v`

### `rtl/register_file_banked.v:378: ERROR: 2nd expression of procedural for-loop is not constant!`
Affected modules:
- `rtl/register_file_banked.v`

### `rtl/fp6_mac_pipeline.v:72: ERROR: Left hand side of 1st expression of procedural for-loop is not a register!`
Affected modules:
- `rtl/fp6_mac_pipeline.v`

### `rtl/streaming_multiprocessor_v2.v:1315: ERROR: syntax error, unexpected TOK_ID`
Affected modules:
- `rtl/streaming_multiprocessor_v2.v`

### `rtl/fp4_mac_pipeline.v:81: ERROR: Left hand side of 1st expression of procedural for-loop is not a register!`
Affected modules:
- `rtl/fp4_mac_pipeline.v`

## Module-by-Module Pass/Fail Matrix

| Module | Status | Error Details |
|--------|--------|---------------|
| `rtl/reconvergence_stack.v` | PASS | `` |
| `rtl/memory_interface_wide.v` | PASS | `` |
| `rtl/command_processor.v` | PASS | `` |
| `rtl/ralph_gpu_top.v` | PASS | `` |
| `rtl/fpu.v` | PASS | `` |
| `rtl/fp16_unit.v` | PASS | `` |
| `rtl/dpx_unit.v` | FAIL | `Timeout Error` |
| `rtl/control_flow_unit.v` | PASS | `` |
| `rtl/cache_policy_unit.v` | PASS | `` |
| `rtl/chi_controller.v` | PASS | `` |
| `rtl/l1_data_cache_optimized.v` | FAIL | `Timeout Error` |
| `rtl/sm_writeback_arbiter.v` | PASS | `` |
| `rtl/tensor_memory.v` | FAIL | `Timeout Error` |
| `rtl/mbarrier_unit.v` | PASS | `` |
| `rtl/multimem_unit.v` | PASS | `` |
| `rtl/atomic_unit.v` | PASS | `` |
| `rtl/sm_fetch_pipeline.v` | PASS | `` |
| `rtl/alu.v` | PASS | `` |
| `rtl/l1_data_cache.v` | FAIL | `Timeout Error` |
| `rtl/stack_debug_unit.v` | PASS | `` |
| `rtl/performance_counters.v` | PASS | `` |
| `rtl/mul_unit.v` | PASS | `` |
| `rtl/sfu.v` | PASS | `` |
| `rtl/cluster_barrier_unit.v` | PASS | `` |
| `rtl/tensor_core.v` | PASS | `` |
| `rtl/tma_unit.v` | PASS | `` |
| `rtl/video_unit.v` | PASS | `` |
| `rtl/command_queue.v` | PASS | `` |
| `rtl/memory_controller_hbm.v` | FAIL | `rtl/memory_controller_hbm.v:227: ERROR: 2nd expression of procedural for-loop is not constant!` |
| `rtl/memory_controller.v` | FAIL | `Timeout Error` |
| `rtl/sm_special_reg.v` | PASS | `` |
| `rtl/fpu64.v` | PASS | `` |
| `rtl/l2_cache.v` | FAIL | `Timeout Error` |
| `rtl/icache.v` | PASS | `` |
| `rtl/lz4_decompressor.v` | PASS | `` |
| `rtl/register_file_banked.v` | FAIL | `rtl/register_file_banked.v:378: ERROR: 2nd expression of procedural for-loop is not constant!` |
| `rtl/warp_scheduler.v` | PASS | `` |
| `rtl/shared_memory.v` | PASS | `` |
| `rtl/forwarding_unit.v` | PASS | `` |
| `rtl/fma_int32.v` | PASS | `` |
| `rtl/advanced_scheduler.v` | PASS | `` |
| `rtl/wgmma_tile_engine.v` | PASS | `` |
| `rtl/sm_wbq_bank.v` | PASS | `` |
| `rtl/st_bulk_unit.v` | PASS | `` |
| `rtl/blackwell_scheduler.v` | PASS | `` |
| `rtl/branch_predictor.v` | FAIL | `Timeout Error` |
| `rtl/griddep_unit.v` | PASS | `` |
| `rtl/fp6_mac_pipeline.v` | FAIL | `rtl/fp6_mac_pipeline.v:72: ERROR: Left hand side of 1st expression of procedural for-loop is not a register!` |
| `rtl/wb_fifo.v` | PASS | `` |
| `rtl/tlb_enhanced.v` | PASS | `` |
| `rtl/streaming_multiprocessor_v2.v` | FAIL | `rtl/streaming_multiprocessor_v2.v:1315: ERROR: syntax error, unexpected TOK_ID` |
| `rtl/warp_shuffle.v` | PASS | `` |
| `rtl/sm_gmem_arbiter.v` | PASS | `` |
| `rtl/decoder.v` | PASS | `` |
| `rtl/texture_unit.v` | PASS | `` |
| `rtl/register_file.v` | FAIL | `Timeout Error` |
| `rtl/memory_interface.v` | PASS | `` |
| `rtl/l2_interconnect.v` | PASS | `` |
| `rtl/tlb.v` | PASS | `` |
| `rtl/warp_collective_unit.v` | PASS | `` |
| `rtl/dual_issue_scheduler.v` | PASS | `` |
| `rtl/cvt_unit.v` | PASS | `` |
| `rtl/memory_coalescing_unit.v` | FAIL | `Timeout Error` |
| `rtl/memory_qos.v` | PASS | `` |
| `rtl/async_copy_engine.v` | PASS | `` |
| `rtl/fp4_mac_pipeline.v` | FAIL | `rtl/fp4_mac_pipeline.v:81: ERROR: Left hand side of 1st expression of procedural for-loop is not a register!` |
| `rtl/wgmma.v` | FAIL | `Timeout Error` |
