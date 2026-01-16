# NVIDIA-Style Architecture Review

Scope: `rtl/streaming_multiprocessor_v2.v`, `rtl/ralph_gpu_top.v`, memory hierarchy RTL, tensor core RTL, and system integration.

Reviewer stance: GPU architect (NVIDIA-style). Evaluate correctness readiness, scalability, and performance risks vs production-class SMs.

## Executive Summary
This RTL is a strong research/bring-up GPU with a credible SM pipeline and tensor path, but it is not production-ready. The largest gaps are in memory hierarchy correctness, front-end robustness, scheduler/issue width, and system-level bandwidth. Microbench IPC is good in isolation, but real workloads will be dominated by memory and divergence penalties.

## Strengths
- SM v2 has functional scoreboard + per-FU WBQ/backpressure to prevent dropped results.
- Tensor path supports configurable cores/latency and FP16/FP4 datatypes.
- Multi-SM dispatch works for basic block scheduling; regressions are green.

## Critical Gaps vs NVIDIA-Class Architecture

### 1) Memory Hierarchy (Highest Risk)
- L1D is stubbed at top-level; no real cache or coalescing in active path.
- L2 cache and memory controller still contain placeholder behavior (miss path, responses, CDC handling).
- Coalescing logic uses live signals instead of latched requests; can mis-serve.
Impact: correctness risk + catastrophic perf on memory-bound kernels.

### 2) Front-End / Control Flow
- No instruction cache, limited fetch buffering, limited control-flow reconvergence.
- Warp divergence handling is minimal; no reconvergence stack.
Impact: branch-heavy code will stall or mis-utilize the SM.

### 3) Scheduler / Issue / Register File
- Single-issue scheduler, low warp count, no dual-issue or specialized schedulers.
- No modeling of register file banking or operand reuse.
Impact: limits IPC and throughput vs NVIDIA schedulers.

### 4) System Integration & Bandwidth
- Single AXI port with simplified arbitration; no interconnect or L2 slicing.
- Memory responses are not scaled for multi-SM traffic patterns.
Impact: bandwidth collapse and underutilization at scale.

### 5) Tensor Core Dataflow
- Tensor core is present but lacks advanced dataflow scheduling (WGMMA tile orchestration).
- No shared-memory dataflow optimization.
Impact: tensor throughput is capped by scheduling and data movement.

## Performance Implications
- Compute microbench IPC is strong (~1 IPC for FP32 and WMMA).
- Real kernels will be memory-limited due to cache/coalescing/memory subsystem gaps.
- Divergence-heavy kernels will suffer due to minimal front-end and reconvergence.

## Prioritized Roadmap (NVIDIA-Style)

### P0: Correctness + Memory Subsystem
1. Implement L1D cache (or explicit bypass) with per-SM request/response queues.
2. Fix coalescing to use latched request metadata.
3. Fix L2 miss handling and memory controller response generation.
4. Add CDC-safe queues between clock domains if `mem_clk` is used.
Files: `rtl/l1_data_cache*.v`, `rtl/memory_coalescing_unit.v`, `rtl/l2_cache.v`,
`rtl/memory_controller.v`, `rtl/memory_interface.v`.

### P1: Front-End Robustness
1. Add I-cache and simple prefetch or fetch buffer.
2. Add reconvergence stack or basic divergence tracking with per-warp PC stacks.
Files: `rtl/streaming_multiprocessor_v2.v`, new `rtl/icache.v`.

### P2: Scheduler and Issue Width
1. Increase `WARPS_PER_SM`; update RF capacity.
2. Add dual-issue or split schedulers (e.g., ALU/Tensor).
3. Add register file bank conflict modeling.
Files: `rtl/warp_scheduler.v`, `rtl/register_file.v`, `rtl/streaming_multiprocessor_v2.v`.

### P3: Tensor Core Dataflow
1. Implement WGMMA-like tiling and SMEM staging.
2. Improve tensor pipeline scheduling and reuse paths.
Files: `rtl/tensor_core.v`, new `rtl/wgmma.v` integration.

### P4: System Integration
1. Add L2 slice topology + interconnect arbitration.
2. Add perf counters for per-SM throughput and memory stalls.
Files: `rtl/ralph_gpu_top.v`, `rtl/l2_cache.v`, new interconnect module.

## Estimated Effort (Rough, Engineering Weeks)
- P0: 6–10 weeks
- P1: 4–6 weeks
- P2: 4–8 weeks
- P3: 3–6 weeks
- P4: 4–8 weeks

## Notes for Verification
- Add randomized kernel tests and memory stress regressions.
- Add per-SM instruction trace capture for debug.
- Expand PTX coverage with self-checking tests.

---
Reviewer conclusion: The compute core is promising and functionally coherent. To reach NVIDIA-class behavior, the memory subsystem and front-end need a full implementation, and the scheduler must scale in width and warp capacity. Once P0/P1 are addressed, IPC on real workloads should improve dramatically.
