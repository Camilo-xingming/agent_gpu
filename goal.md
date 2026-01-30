# RalphGPU Development Goals

基于 task_plan.md 的测试结果，以下是需要修复的关键问题：

## [arch] Analyze shared memory timeout
priority: critical
estimate: 2h
tags: rtl, memory, debug

Review LD.SHARED/ST.SHARED execution path in streaming_multiprocessor_v2.v.
test_08_memory_shared times out at 200000 cycles.
Identify why shared memory operations never complete.

## [impl] Fix shared memory wiring
priority: critical
estimate: 4h
tags: rtl, memory
depends on: task-001

Implement the fix for shared memory operations.
Ensure LD.SHARED and ST.SHARED complete correctly.
File: rtl/streaming_multiprocessor_v2.v, rtl/shared_memory.v

## [test] Verify shared memory fix
priority: high
estimate: 1h
tags: verification
depends on: task-002

Run test_08_memory_shared and confirm PASS.
Run: make sim TEST=test_08_memory_shared

## [arch] Analyze FP32 arithmetic failures
priority: high
estimate: 2h
tags: rtl, fpu, debug

Debug test_04_fp32_arith returning result=0.
Trace ADD.F32, SUB.F32, MUL.F32, DIV.F32, FMA execution.
File: rtl/fpu.v, rtl/streaming_multiprocessor_v2.v

## [impl] Fix FP32 arithmetic pipeline
priority: high
estimate: 4h
tags: rtl, fpu
depends on: task-004

Fix the FP32 arithmetic issues.
Ensure proper pipeline latency and result writeback.

## [test] Verify FP32 fix
priority: high
estimate: 1h
tags: verification
depends on: task-005

Run test_04_fp32_arith and confirm PASS.

## [arch] Analyze atomic operation timeout
priority: high
estimate: 2h
tags: rtl, atomic, debug

Debug test_09_atomic timeout at 200000 cycles.
Review ATOM.ADD, ATOM.CAS execution paths.
File: rtl/atomic_unit.v, rtl/memory_interface.v

## [impl] Fix atomic operation pipeline
priority: high
estimate: 6h
tags: rtl, atomic
depends on: task-007

Fix the atomic operation execution path.
Ensure proper memory ordering and completion signaling.

## [test] Verify atomic operations
priority: high
estimate: 1h
tags: verification
depends on: task-008

Run test_09_atomic and confirm PASS.

## [arch] Analyze CVT failures
priority: medium
estimate: 2h
tags: rtl, cvt, debug

Debug test_10_cvt returning 0xDEAD (error marker).
Review CVT instruction handling.
File: rtl/decoder.v, rtl/fpu.v

## [impl] Fix CVT operations
priority: medium
estimate: 4h
tags: rtl, cvt
depends on: task-010

Fix type conversion operations (int<->float, size conversions).

## [test] Verify CVT fix
priority: medium
estimate: 1h
tags: verification
depends on: task-011

Run test_10_cvt and confirm PASS.

## [arch] Fix scheduler issue in system tests
priority: medium
estimate: 4h
tags: rtl, scheduler, debug

tb_mbarrier_system and tb_wgmma_system timeout despite unit tests passing.
Analyze instruction fetch/issue pipeline stall.
File: rtl/streaming_multiprocessor_v2.v, rtl/warp_scheduler.v

## [impl] Fix instruction fetch pipeline
priority: medium
estimate: 6h
tags: rtl, scheduler
depends on: task-013

Fix the instruction fetch/issue that causes system tests to stall.

## [test] Run full verification suite
priority: medium
estimate: 2h
tags: verification
depends on: task-014

Run all unit tests and system tests.
Generate verification report.

## [arch] Analyze FP16 arithmetic failures
priority: low
estimate: 2h
tags: rtl, fpu, fp16

Debug test_06_fp16_arith returning 0xDEAD.
Review FP16 support in FPU.

## [impl] Fix FP16 operations
priority: low
estimate: 4h
tags: rtl, fpu, fp16
depends on: task-016

Implement proper FP16 arithmetic support.

## [test] Verify FP16 fix
priority: low
estimate: 1h
tags: verification
depends on: task-017

Run test_06_fp16_arith and confirm PASS.

## [docs] Update documentation
priority: low
estimate: 2h
tags: docs
depends on: task-015

Update ARCHITECTURE.md, progress.md with all fixes.
Document verification results and performance metrics.
