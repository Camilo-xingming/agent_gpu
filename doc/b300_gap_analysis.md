# RalphGPU vs NVIDIA B300 Gap Analysis (2026-01-18)

快速列出相对 Blackwell/B300 架构的主要缺口与优先改进方向。

## 功能/架构缺口
- **Tensor / 5th Gen 加速**：WGMMA 未接线（仅解码标记，无发射/写回）；5th-gen Tensor 指令、FP4/FP6 混合支持缺失；DPX/稀疏矩阵等 Blackwell 新特性未覆盖。
- **TMA / Async Memory**：`OP_CPASYNC` 为固定延迟计数 stub（无真实 LSU/TMA 搬运）；`st.async/st.bulk/multimem` 未实现；`mbarrier`、`cluster` 协同缺失，无法覆盖 Blackwell 的 TMA+cluster 模式。
- **同步与集群**：仅有 `bar.sync`，缺 `bar.warp.sync`、`barrier.cluster`、`match.sync`、`red.async`、`griddepcontrol`、`elect.sync`、`mbarrier`；无 cluster 级 barrier/调度。
- **缓存/策略控制**：cache policy 指令（createpolicy/applypriority/discard/prefetch hints）未落地；无动态缓存优先级/策略管理。
- **内存带宽/规模**：L2/AXI 带宽与 B300（8TB/s HBM3e）差距大；共享存储容量远小于 256–304KB 级别。
- **纹理/表面/视频**：模块存在但未与 SM 接线；tex/txq/suld/sust/sured 及 SIMD 视频饱和/舍入规则未实现。
- **调度/吞吐**：单/双 issue，与 B300 的 4-way warp scheduler 不符；缺少指令压缩/宏融合；SM 数量/前端带宽不匹配。
- **Debug/栈/控制**：alloca/stacksave/stackrestore、brkpt/trap/nanosleep/pmevent/setmaxnreg 未实现，影响工具链兼容。
- **指令尾项**：FP half/mixed compare 未完成；dp4a/dp2a 未接入视频/ALU；mapa/getctarank/isspacep/createpolicy/applypriority/discard 缺失。
- **性能/功耗建模**：无细粒度性能计数/功耗/时钟域管理；与实际芯片时序/功耗差距大（仅功能仿真）。
- **顶层可扩展性**：多 SM/cluster 测试有限；IMEM/L2 总线宽度与 B300 不匹配；无高吞吐互连模型。

## 高优先级推进（建议顺序）
1) **接入 TMA 路径**：用 `rtl/async_copy_engine.v` 或 LSU 扩展实现 cp.async 真实搬运、commit/wait 组语义；补全 st.async/multimem；接线 mbarrier/wait_all。
2) **Tensor 代际升级**：接线 WGMMA + 长延迟 scoreboarding；定义 FP4/FP6/FP8 新格式映射，探索 DPX/稀疏单元（若追 B300 级）。
3) **同步增强**：bar.warp.sync、barrier.cluster、match.sync/red.async/griddepcontrol/elect.sync/mbarrier/tensormap，含 cluster token/ID 管理。
4) **缓存策略面**：实现 createpolicy/applypriority/discard/isspacep/mapa/getctarank，支持 cache hint 到 L1/L2/TMA。
5) **纹理/视频接线**：接入 tex/txq/suld/sust/sured，补 SIMD 视频饱和/舍入；连通 dp4a/dp2a。
6) **指令完整性**：补 FP half/mixed compare、栈/调试指令（alloca/stacksave/stackrestore、brkpt/trap/nanosleep/pmevent/setmaxnreg）。
7) **多 SM/cluster 回归**：扩大顶层回归覆盖带宽、barrier.cluster、TMA+compute 混合场景，校准 IMEM/L2 宽度与吞吐。
