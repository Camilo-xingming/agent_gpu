# RalphGPU Top-Level Verification Plan (ralph_gpu_top)

目标：用 PTX 作为测试入口，自动转换为二进制/hex，运行在 Verilog testbench 上，覆盖 `ralph_gpu_top` 当前和后续实现的全部功能，且自检 (无人工比对)。

## 流程概述
1) **PTX→HEX**：`python tools/ptx_assembler.py --input foo.ptx --output foo.hex`（指令存储用 hex）；数据初值放 `dram_init.hex`。保留 `.ptx` 供追溯。
2) **Testbench 模板**：参考 `tb/tb_ralph_gpu.v` 风格：加载程序/数据，驱动 `kernel_start`，轮询 `kernel_done`（带超时），结束后对比输出区域或寄存器，失败打印首个不匹配地址/数据。
3) **自检契约**：
   - `kernel_done` 必须在超时前拉高；
   - 指定内存区与 golden 匹配（或小核对寄存器）；
   - AXI/协议无 X/Z，非法 opcode 断言。
4) **生成 Golden**：
   - 小型核：在 bench 中内联期望数组；
   - 复杂核：Python 参考模型生成 `golden.hex`（同 ptx_assembler 脚本或专用模型脚本）。
5) **运行/报告**：`iverilog -g2012 ...` + `vvp`，stdout 只要 PASS/FAIL，若 FAIL 打印首个 mismatch。

## 覆盖分类与示例内核
1) **控制/分支/汇聚**：顺/逆跳、divergence+reconverge、CALL/RET/EXIT、部分 warp mask。
2) **ALU/逻辑/比较**：add/sub/and/or/xor/not，移位，bmsk/szext/fns/lop3/shf/cnot，mad.cc/madc 进位链，setp/selp/slct。
3) **乘除**：mul.lo/hi/wide，mul24/mad24，div/rem (s/u)，mad.hi；dp4a/dp2a（待接线后启用）。
4) **浮点**：FP32/FP16/BF16/FP64 add/sub/mul/fma/div，copysign/testp，rcp/sqrt/rsqrt/sin/cos/lg2/ex2/tanh；FP 比较矩阵（half/mixed 实现后补）。
5) **CVT**：int↔fp、fp16/bf16↔fp32，cvt.pack 低 16-bit 打包。
6) **内存**：ld/st global/shared/local/param/const，v2/v4，跨行对齐/错位、合并访问，shared bank conflict 微基准，local spill/填充。
7) **原子/归约**：ATOM/RED 全子集，含全局/共享、竞争场景。
8) **Warp 级**：SHFL (up/down/bfly/idx)、VOTE/REDUX、activemask 特殊寄存器。
9) **同步**：
   - bar.sync + membar 顺序性，双发射下 sync stall/释放；
   - cp.async stub：验证 wait_group/all 阻塞/解阻、EXIT 等待 pending（暂不检查数据搬运）。
10) **Tensor**：WMMA/MMA 正确性；WGMMA 上线后补充 fence/commit/wait。
11) **特殊寄存器**：laneid/warpid/smid/activemask 返回值正确。
12) **预取**：prefetch 仅为 hint，验证无架构副作用（数据不被破坏）。
13) **多 SM（若 top 配置支持）**：bar.sync/membar 作用域、独立 warp 调度、全局内存一致性。

## 回归矩阵
- **Smoke（最小集）**：每类 1 个用例（≈13 个），覆盖 happy path。
- **Extended**：增加 stride/合并模式、半精度/混合 FP 比较、原子高竞争、深度分支、部分 warp mask、长延迟流水线交错。
- **XFAIL/占位**：cp.async 真正搬运、st.async/multimem、WGMMA、texture/surface/video、cluster barrier/mbarrier，在功能接线前标记预期失败。

## 自动化建议
- 脚本 pipeline：
  1) assemble PTX → program hex；
  2) 生成 `dram_init.hex` & `golden.hex`（Python 模型或内联）；
  3) 生成 bench（或参数化模板）并调用 `iverilog -g2012 -I rtl -s <tb>`，`vvp` 执行；
  4) 解析日志，汇总 PASS/FAIL/时间。
- 触发策略：改动 `rtl/` 或 `tools/ptx_assembler.py` 时跑 smoke；夜跑 full 矩阵。

## 约束与已知限制
- cp.async 目前为固定延迟计数，无真实 LSU 搬运；测试仅检 wait 语义。
- st.async/multimem、WGMMA、texture/surface/video、mbarrier/cluster barrier 尚未接线；相关测试先标记 XFAIL。
- 顶层 IMEM 宽度需与 SM V2 对齐（8B 线）；多 SM 配置需确认顶层参数。
