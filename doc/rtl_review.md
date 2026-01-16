# RTL Review Results

Scope: Core RTL modules (scheduler, memory, caches, execution units) and top-level integration.

## Findings

### Critical
- `rtl/forwarding_unit.v:162` `ex_is_load` is declared inside the module body instead of the ANSI port list, which is illegal and will not compile.
- `rtl/memory_controller.v:149` Request queue never dequeues (`req_head` never advances) and no read responses are ever generated (`rtl/memory_controller.v:333`), so reads can never complete.
- `rtl/ralph_gpu_top.v:172`, `rtl/ralph_gpu_top.v:226`, `rtl/ralph_gpu_top.v:244` Instruction/AXI arbitration muxes requests but broadcasts responses to all SMs while selection is combinational, so one SM can consume another SM's response.

### High
- `rtl/ralph_gpu_top.v:167`, `rtl/ralph_gpu_top.v:295`, `rtl/ralph_gpu_top.v:312` Block scheduling and ID tracking are not per-SM; nonblocking increments inside loops only add once per cycle, leading to duplicate/skip block IDs.
- `rtl/memory_interface.v:124` Next-state uses `is_write_buf` before it is captured from the request; `rtl/memory_interface.v:176` uses `mask_buf` instead of `req_mask`; `rtl/memory_interface.v:137` compares lane index to lane count (not last active index), causing early termination on sparse masks.
- `rtl/warp_scheduler.v:144`, `rtl/warp_scheduler.v:185` `warp_ready` is driven by both sequential and combinational logic (multiple drivers), which is undefined.
- `rtl/memory_controller.v:202` Crosses `clk` -> `mem_clk` without CDC protection by reading the request queue in the other clock domain.
- `rtl/memory_coalescing_unit.v:60`, `rtl/memory_coalescing_unit.v:94` Coalescing uses live `req_*` signals instead of latched request data, so coalescing can change mid-transaction.
- `rtl/l2_cache.v:247`, `rtl/l2_cache.v:476` L2 never drives `mem_req_pending` and fills misses with zeros, so miss handling is non-functional.
- `rtl/l2_cache.v:224` Response routing clears all ports pending on a bank in one cycle, so multiple ports targeting the same bank can be dropped or mis-served.

### Medium
- `rtl/memory_coalescing_unit.v:76` `num_unique_lines` is 2 bits with `MAX_COALESCED=4`, so a fourth unique line overflows and is dropped.
- `rtl/l1_data_cache.v:106` `first_active` uses `0` as a sentinel, so lane 0 can be overwritten by later lanes; empty masks still use `saved_addr[0]` for tag lookup.
- `rtl/async_copy_engine.v:31`, `rtl/async_copy_engine.v:62`, `rtl/async_copy_engine.v:256` `pending_count`/`req_count` can overflow at default depth; wait-group logic checks `func` live instead of a latched operation.
- `rtl/warp_scheduler.v:110` `pc_is_branch` is unused, so branch PC updates are ignored.
- `rtl/l2_cache.v:168` Byte write mask `l1_req_wmask` is never applied; partial stores overwrite entire lines.
- `rtl/l1_data_cache_optimized.v:497` WCB allocation loops allocate into every invalid entry (no early exit), duplicating the same write.
- `rtl/l1_data_cache_optimized.v:468` Prefetcher state updates from `req_addr[0]` even when lane 0 is inactive.
- `rtl/fpu.v:436`, `rtl/fpu64.v:866`, `rtl/sfu.v:211` `valid_out` treats masked-off lanes as valid, so `valid_out` can assert without active lanes.
- `rtl/warp_shuffle.v:46` `shfl.up` does not enforce `width`, so lanes above `width` may read invalid sources.
- `rtl/texture_unit.v:278` Bilinear path is TODO but still behaves as point sampling.

## Questions
- Are the memory controller and L2 cache intended to be fully functional now, or still placeholders? (Several items above are structural blockers for correctness.)
- Should backpressure be fully supported on all memory request paths (imem, global, shared), or only on global memory?

## Suggested Fix Order
1. Memory controller queue/response + CDC; L2 miss path and memory interface.
2. Top-level arbiter response routing and SM backpressure integration.
3. Warp scheduling/branch PC handling and multi-warp allocation.
4. Remaining correctness fixes (mask handling, coalescing, valid_out masking).

## Addendum: SM V2 RTL Performance Microbench
- Testbench: `tb/tb_sm_v2_perf_gemm16.v` (4096 FP32 FMA ops, single warp, no memory traffic).
- Observed on current RTL: 4096 writebacks in 4110 cycles (IPC ~0.997).
- IFQ + WBQ decoupling now sustains near-1 IPC on FP32 FMA streams.

## Addendum: SM V2 Tensor Core Microbench
- Testbench: `tb/tb_sm_v2_perf_tensor.v` (2048 WMMA MMA ops, FP16, single warp).
- Observed on current RTL: 2048 writebacks in 2064 cycles (IPC ~0.992) with TC_NUM_CORES=8, TC_LATENCY=4.
- Added tensor issue queue + tensor writeback queue to prevent dropped WMMA results and honor backpressure.

## Addendum: SM V2 Writeback Queue Fix
- `rtl/streaming_multiprocessor_v2.v`: Added per-FU writeback queues + inflight backpressure to prevent dropped results when multiple FUs complete in the same cycle.
- `tb/tb_sm_v2_integration.v`: Tests 1-4 now complete without TIMEOUT (WB arbitration + multi-cycle FPU).
