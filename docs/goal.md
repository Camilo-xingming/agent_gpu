# RalphGPU 优化目标

## 优化 RalphGPU 以实现更高的性能和功能覆盖率

基于当前测试结果和 progress.md 的分析，目标是修复关键问题并提升 IPC。

---

## [arch] ~~分析 Shared Memory 超时问题~~ ✅ FIXED
priority: critical
estimate: 2h
tags: rtl, memory, debug
**STATUS: PASS** - test_08_memory_shared 返回 0xCAFE (成功)

## [impl] ~~修复 Shared Memory 路径~~ ✅ FIXED
priority: critical  
estimate: 4h
tags: rtl, memory
**STATUS: PASS** - 共享内存 LD/ST 正常工作

## [test] ~~验证 Shared Memory 修复~~ ✅ PASS
priority: high
estimate: 1h
tags: verification
**STATUS: PASS** - C[0] = 0xCAFE

## [arch] ~~分析 FP16 运算失败~~ ✅ FIXED
priority: high
estimate: 2h
tags: rtl, fpu, fp16
**STATUS: PASS** - test_06_fp16_arith 返回 0xCAFE (成功)

## [impl] ~~修复 FP16 运算单元~~ ✅ FIXED
priority: high
estimate: 4h
tags: rtl, fpu, fp16
**STATUS: PASS** - FP16 算术正常工作

## [test] ~~验证 FP16 修复~~ ✅ PASS
priority: high
estimate: 1h
tags: verification
**STATUS: PASS** - C[0] = 0xCAFE

## [arch] ~~分析 Atomic 操作超时~~ ✅ FIXED
priority: high
estimate: 2h
tags: rtl, atomic, debug
**STATUS: FIXED** - 发现 atomic_unit 内存接口未连接，已修复

## [impl] ~~修复 Atomic 操作流水线~~ ✅ FIXED
priority: high
estimate: 6h
tags: rtl, atomic
**STATUS: FIXED** - 已连接 atomic 到 gmem 仲裁器，添加了 atomic_mem_pending 状态跟踪

## [test] ~~验证 Atomic 操作~~ ✅ FIXED
priority: high
estimate: 1h
tags: verification
**STATUS: FIXED**
- ✅ simple_atomic_test PASS (0xCAFE) - single atom.add works
- ✅ All 8 atomic operations execute correctly (visible in AXI trace)
- ~~❌ Complex tests with branch divergence fail~~ ✅ FIXED (2026-02-04)
- **Fix:** Changed atomic writeback mask from `atomic_mask_pending` to `atomic_result_mask`
- **File:** rtl/streaming_multiprocessor_v2.v line 5001
- **Test:** Created atomic_divergent_test.ptx for validation

## [arch] 分析 CVT 操作失败
priority: medium
estimate: 2h
tags: rtl, cvt, debug

调试 test_10_cvt 返回 0xDEAD。
检查 CVT 指令处理。
关键文件: rtl/decoder.v, rtl/cvt_unit.v

## [impl] 修复 CVT 操作
priority: medium
estimate: 4h
tags: rtl, cvt
depends on: task-010

修复类型转换操作 (int<->float, 大小转换)。

## [test] 验证 CVT 修复
priority: medium
estimate: 1h
tags: verification
depends on: task-011

运行 test_10_cvt 确认 PASS。

## [perf] 优化 IPC - 减少 stall
priority: medium
estimate: 4h
tags: rtl, performance

分析当前 stall 来源 (RAW, FU, MEM, WBQ)。
优化流水线以减少 stall cycle。
目标: 将 multi-warp IPC 从 0.68 提升到 0.80+。

## [docs] 更新文档
priority: low
estimate: 2h
tags: docs

更新 progress.md, ARCHITECTURE.md。
记录所有修复和性能指标。
