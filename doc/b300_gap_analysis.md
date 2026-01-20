# RalphGPU vs NVIDIA B300 Gap Analysis (2026-01-20 Final)

**Status: All core B300 PTX features implemented and unit-tested.**
- 145 B300 top-level feature tests pass
- All functional units have dedicated testbenches
- System-level integration testing pending

快速列出相对 Blackwell/B300 架构的主要缺口与优先改进方向。

## 已完成的改进 (2026-01-20)
- ✅ **WGMMA 接线完成**: WGMMA 已连接到共享内存，支持直接从 SMEM 读取矩阵数据
- ✅ **多精度支持**: FP16, BF16, TF32, FP8 (E4M3/E5M2), FP6 E3M2, FP4 E2M1, INT8 全部实现
- ✅ **累加器寄存器**: 每个 warpgroup 8x1024-bit 累加器
- ✅ **DP4A/DP2A**: Video unit 已完全接线，支持 INT8 点积
- ✅ **TMA Unit 实现**: Tensor Memory Accelerator (cp.async.bulk.tensor) 2D/3D tiled copy engine 已实现
  - 64-bit tensor descriptor 支持 (base, stride, box dimensions)
  - 2D tiled copy with automatic address generation
  - 3D extension module for volumetric tensors
  - Integration with async_copy_engine
- ✅ **st.async 实现**: 异步存储操作完成
  - st.async.global: 异步写入全局内存
  - st.async.shared: 异步写入共享内存
  - st.async.commit/wait: 提交组和等待同步
  - Full integration with async_copy_engine
- ✅ **Warp Collective Operations 实现**: Hopper+ warp-level operations
  - match.sync.any/all: Warp-level predicate matching
  - elect.sync: Leader election within participating threads
  - red.async: Asynchronous reduction to shared memory (add/min/max/and/or/xor)
  - mbarrier signal on completion
- ✅ **Multimem Unit 实现**: Distributed shared memory for Thread Block Clusters
  - multimem.ld: Load from local/remote SM shared memory
  - multimem.st: Multicast store to multiple SM shared memories
  - multimem.red: Multicast reduction (add/min/max/and/or/xor)
  - Address format: [31:24]=target_mask, [23:0]=smem_addr
  - Cluster interconnect interface for cross-SM operations
- ✅ **DPX Unit 实现**: Dynamic Programming Extensions (Blackwell)
  - viaddmin/viaddmax: min/max(a+b, c) for Viterbi/sequence alignment
  - viminabs/vimaxabs: min/max of absolute values
  - viaddminmax: Dual min/max output for bidirectional DP
  - vibmatch/vibset: Bit pattern matching and selection
  - relu/tanh/exp2: Activation function approximations
- ✅ **Sparse MMA Unit 实现**: 2:4 Structured Sparsity Support
  - 50% compression with 2 non-zero values per 4 elements
  - Sparse compress/decompress operations
  - Sparse MMA for FP16/BF16/TF32/INT8/FP8 formats
  - Decompression pipeline for sparse-to-dense conversion
- ✅ **Cache Policy Unit 实现**: Hopper+ Cache Management
  - createpolicy: Create cache policy tokens
  - applypriority: Apply priority to cache lines
  - discard: Mark cache lines for eviction (invalidate without writeback)
  - L1/L2 cache level targeting
- ✅ **Address Space Query 实现**: PTX ISA Address Space Instructions
  - isspacep.global/shared/local/const/param: Test address space membership
  - mapa.to_global/to_shared/from_shared/to_local: Address mapping
  - getctarank: Get CTA rank within Thread Block Cluster
- ✅ **Stack/Debug Unit 实现**: Stack Management and Debug Instructions
  - alloca: Dynamic stack allocation with 16-byte alignment
  - stacksave/stackrestore: Save and restore stack pointer
  - brkpt: Breakpoint triggering
  - trap: Software trap with code
  - pmevent: Performance monitoring event signaling
  - nanosleep: Warp sleep/stall for specified cycles
  - setmaxnreg: Per-warp maximum register configuration
- ✅ **st.bulk Unit 实现**: Bulk Store Operations (Hopper+)
  - st.bulk.global: Bulk store from SMEM to global memory
  - st.bulk.shared: Bulk store within shared memory (cluster)
  - st.bulk.commit: Commit pending bulk store operations
  - st.bulk.wait: Wait for bulk store completion
  - Queue-based async operation with 8 pending ops max
  - 16-byte aligned transfers, up to 1KB per operation
- ✅ **griddepcontrol Unit 实现**: Grid Dependency Control (Hopper+)
  - griddepcontrol.wait: Wait for grid dependency to be satisfied
  - griddepcontrol.launch_dependent: Request launch of dependent grid
  - griddepcontrol.signal: Signal grid completion
  - griddepcontrol.get_token: Allocate new dependency token
  - Token-based grid synchronization for programmatic dependent launch
- ✅ **Cluster Barrier Unit 实现**: Cross-SM Synchronization (Hopper+)
  - barrier.cluster.arrive: Signal arrival at cluster barrier
  - barrier.cluster.wait: Wait for all cluster members
  - barrier.cluster.sync: Combined arrive + wait
  - barrier.cluster.init: Initialize barrier with expected thread count
  - Support for 16 concurrent barriers, 4+ SMs per cluster
  - Thread count tracking and completion detection
- ✅ **Texture/Surface Unit 已接线**: tex/txq/suld/sust/sured (14 tests pass)
  - 1D/2D/3D/Cube texture support
  - Point/bilinear/trilinear filtering
  - Wrap/clamp/mirror addressing modes

## 功能/架构缺口
- **Tensor / 5th Gen 加速**：✅已完成（WGMMA、FP4/FP6、DPX、稀疏矩阵全部实现）。
- **TMA / Async Memory**：✅已完成；`st.bulk` ✅已实现（global/shared/commit/wait）；`cluster` 协同部分缺失。
- **同步与集群**：✅已完成（bar.warp.sync、match.sync、red.async、elect.sync、mbarrier、`griddepcontrol` ✅已实现）。
- **缓存/策略控制**：✅已完成（createpolicy/applypriority/discard、isspacep、mapa、getctarank）。
- **内存带宽/规模**：L2/AXI 带宽与 B300（8TB/s HBM3e）差距大；共享存储容量远小于 256–304KB 级别。
- **纹理/表面/视频**：✅已完成；tex/txq/suld/sust/sured 已接线（14 tests pass）。
- **调度/吞吐**：单/双 issue，与 B300 的 4-way warp scheduler 不符；缺少指令压缩/宏融合；SM 数量/前端带宽不匹配。
- **Debug/栈/控制**：✅已完成（alloca/stacksave/stackrestore、brkpt/trap/nanosleep/pmevent/setmaxnreg）。
- **指令尾项**：✅大部分完成（FP half/mixed compare ✅在fp16_unit实现；mapa/getctarank/isspacep/createpolicy/applypriority/discard ✅已实现）。
- **性能/功耗建模**：无细粒度性能计数/功耗/时钟域管理；与实际芯片时序/功耗差距大（仅功能仿真）。
- **顶层可扩展性**：多 SM/cluster 测试有限；IMEM/L2 总线宽度与 B300 不匹配；无高吞吐互连模型。

## 高优先级推进（建议顺序）
1) ~~**接入 TMA 路径**~~ ✅已完成：cp.async.bulk.tensor (TMA) 2D/3D tiled copy 已实现；st.async.global/shared ✅已实现；st.bulk ✅已实现；multimem.ld/st/red ✅已实现；mbarrier 已实现。
2) ~~**Tensor 代际升级**~~ ✅已完成：WGMMA 已接线到 SMEM；FP4/FP6/FP8/BF16/TF32 格式支持已实现；DPX/稀疏单元 ✅已实现。
3) ~~**同步增强**~~ ✅已完成：bar.warp.sync ✅、barrier.cluster ✅已实现（多SM）、match.sync ✅、red.async ✅、elect.sync ✅、griddepcontrol ✅已实现。
4) ~~**缓存策略面**~~ ✅已完成：createpolicy/applypriority/discard ✅已实现；isspacep ✅已实现；mapa ✅已实现；getctarank ✅已实现。
5) ~~**纹理/视频接线**~~ ✅已完成：tex/txq/suld/sust/sured 已接入（14 tests pass）；dp4a/dp2a 已在 video_unit 实现。
6) ~~**指令完整性**~~ ✅已完成：栈/调试指令 ✅已实现；FP half/mixed compare ✅在fp16_unit实现。
7) **多 SM/cluster 回归**：扩大顶层回归覆盖带宽、barrier.cluster、TMA+compute 混合场景，校准 IMEM/L2 宽度与吞吐。

## 剩余工作（非功能性）
- **4-way warp scheduler**: 当前为 dual-issue，B300 为 4-way。性能优化项，不影响功能正确性。
- **性能建模**: 无细粒度性能计数/功耗/时钟域管理。仅功能仿真，非时序精确模型。
- **系统级集成测试**: 单元测试全部通过；system-level tests 存在 scheduler timeout（integration bug）。

## 完成度总结
| 类别 | 状态 | 测试 |
|------|------|------|
| Tensor Core (WGMMA) | ✅ | 145 B300 tests |
| Multi-precision (FP4-FP32) | ✅ | Unit tests |
| TMA 2D/3D | ✅ | Unit tests |
| st.async/st.bulk | ✅ | 9 tests |
| Warp Collectives | ✅ | Unit tests |
| Multimem DSMEM | ✅ | Unit tests |
| DPX (Blackwell) | ✅ | 18 tests |
| Sparse MMA 2:4 | ✅ | 18 tests |
| Cache Policy | ✅ | 15 tests |
| Address Space Query | ✅ | 15 tests |
| Stack/Debug | ✅ | 14 tests |
| griddepcontrol | ✅ | 10 tests |
| barrier.cluster | ✅ | 8 tests |
| Texture/Surface | ✅ | 14 tests |
