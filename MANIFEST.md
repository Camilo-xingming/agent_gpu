# RalphGPU — CUDA-compatible GPU RTL (Verilog)

> 开源 GPU 设计项目。RTL 实现 + PTX 兼容 + 仿真验证。

## 根目录文件

### 项目文档

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| README.md | 项目概述 + 快速开始 | Jerry/Coders | 所有人 | 初次了解 | overview, readme |
| design_document.md | 架构设计文档 | Jerry/Coders | 所有人 | 设计决策时 | architecture, design |
| goal.md | 项目目标 | Jerry | 所有人 | Planning 时 | goal, vision |
| PERFORMANCE_REPORT.md | 性能报告 | Coders | Jerry | Review 时 | performance, benchmark |
| findings.md | 调查发现记录 | Coders | Coders | 调试时 | debug, findings |
| progress.md | 进度记录 | Coders | Jerry | Review 时 | progress, status |
| prompt.md | Agent 初始 prompt | Jerry | Agents | 启动时 | prompt |
| task_plan.md | 任务计划 | Jerry | Coders | Sprint Planning | plan, tasks |
| plan-149.md | Issue #149 实现计划（Pipeline Replay / Scheduler Fairness） | Coders | Coders | 开发 #149 时 | plan, scheduler, replay |
| research-149.md | Issue #149 调研（WB Precision 根因分析） | Coders | Coders | 开发 #149 时 | research, scheduler, WB |

### 构建 & 配置

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| Makefile | 编译、测试、lint、dashboard 目标 | Coders | Coders | 每次构建/测试 | build, test, lint |
| reapply_fixes.sh | 批量重新应用 RTL 修复（branch stall + R0 scoreboard） | Coders | Coders | RTL 修复被覆盖时 | fix, batch, shell |

### Debug 注入脚本（Python）

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| add_atomic_debug.py | 向 atomic_unit.v 注入状态机调试 $display | Coders | Coders | 调试 atomic 时 | debug, atomic |
| add_branch_stall_debug.py | 向调度器注入 branch stall 调试输出 | Coders | Coders | 调试 branch stall 时 | debug, branch, stall |
| add_hazard_detail.py | 向调度器注入 hazard 细节调试（scoreboard） | Coders | Coders | 调试 hazard 时 | debug, hazard, scoreboard |
| add_imem_debug.py | 向 testbench 注入 IMEM 请求/响应调试 | Coders | Coders | 调试指令取回时 | debug, imem |
| add_ready_debug.py | 向调度器注入 warp_ready 调试输出 | Coders | Coders | 调试 warp 就绪逻辑时 | debug, scheduler, ready |
| add_sched_inputs_debug.py | 向调度器注入输入信号调试（valid/ready/eligible） | Coders | Coders | 调试调度输入时 | debug, scheduler, inputs |
| add_scheduler_debug.py | 向调度器注入 eligibility 调试输出 | Coders | Coders | 调试调度逻辑时 | debug, scheduler, eligible |
| add_stall_debug.py | 向调度器注入 warp stall 条件调试 | Coders | Coders | 调试 stall 时 | debug, stall |
| add_vcd_dump.py | 向 testbench 注入 VCD 波形 dump | Coders | Coders | 需要波形分析时 | debug, vcd, waveform |
| check_imem_runtime.py | 向 testbench 注入 IMEM 运行时内容检查 | Coders | Coders | 调试 IMEM 数据时 | debug, imem, runtime |
| debug_atomic_serialization.py | 向 atomic_unit.v 注入序列化调试（pending_mask） | Coders | Coders | 调试 atomic 序列化时 | debug, atomic, serialize |
| safe_atomic_debug.py | 向 atomic_unit.v 注入安全版调试输出 | Coders | Coders | 调试 atomic 时 | debug, atomic, safe |

### RTL/TB 自动修复脚本（Python）

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| apply_d1_patch.py | 为 SM v2 添加 warp_inst_valid_d1 寄存器 | Coders | Coders | 修复 d1 stale 问题时 | fix, d1, pipeline |
| cleanup_tb.py | 清理 testbench 中重复的 VCD dump 块 | Coders | Coders | TB 清理时 | fix, testbench, cleanup |
| fix_axi_model.py | 修复 testbench 中的 AXI 模型逻辑 | Coders | Coders | AXI 问题时 | fix, axi, testbench |
| fix_branch_stall.py | 移除调度时过早的 branch stall 设置 | Coders | Coders | 修复 branch stall 时 | fix, branch, stall |
| fix_debug_syntax.py | 修复损坏的 $display 调试语法 | Coders | Coders | 调试语法错误时 | fix, syntax, display |
| fix_dup_display.py | 移除调度器中重复的 $display 行 | Coders | Coders | 清理重复调试时 | fix, duplicate, display |
| fix_r0_scoreboard.py | 修复 R0 不应进入 scoreboard 的 bug | Coders | Coders | 修复 R0 hazard 时 | fix, r0, scoreboard |
| fix_tb_structure.py | 修复 testbench 文件结构（endmodule 后多余代码） | Coders | Coders | TB 结构错误时 | fix, testbench, structure |
| fix_testbench.py | 修复 testbench hex 路径 + 添加加载调试 | Coders | Coders | TB hex 路径问题时 | fix, testbench, hex |

### Python 测试运行器

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| test_gemm.py | 测试 2x2 GEMM 矩阵乘法 | Coders | Coders | 验证 GEMM 时 | test, gemm, matrix |
| test_llm_suite.py | LLM 算子测试套件（attention, GELU, softmax 等） | Coders | Coders | 验证 LLM 算子时 | test, llm, suite |
| test_nano_llm.py | 测试 Nano-LLM（Mini Transformer Layer） | Coders | Coders | 验证 transformer 时 | test, nano, llm |
| test_transformer.py | 测试完整 Transformer Block 前向传播 | Coders | Coders | 验证 transformer 时 | test, transformer, forward |

### Hex 测试程序（GPU 指令内存镜像）

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| batched_matmul_4x4x4_fp16.hex | 批量 4x4x4 FP16 矩阵乘法测试 | Coders | 仿真器 | 运行 matmul 测试时 | hex, matmul, fp16, batch |
| batched_matmul_4x4x4_fp16_looped.hex | 带循环的批量 4x4x4 FP16 矩阵乘法 | Coders | 仿真器 | 运行 matmul 测试时 | hex, matmul, fp16, loop |
| divergence_test.hex | 线程分歧测试 | Coders | 仿真器 | 验证分歧处理时 | hex, divergence |
| dp4a_simple.hex | DP4A 整数点积指令简单测试 | Coders | 仿真器 | 验证 DP4A 时 | hex, dp4a, int8 |
| dp4a_top_test.hex | DP4A 顶层集成测试 | Coders | 仿真器 | 验证 DP4A 集成时 | hex, dp4a, top |
| gemm16_fma.hex | 16x16 GEMM FMA 测试 | Coders | 仿真器 | 验证 GEMM 时 | hex, gemm, fma |
| llm_attention_score.hex | LLM 注意力分数计算（Q*K^T + DP4A） | Coders | 仿真器 | 验证 attention 时 | hex, llm, attention |
| llm_dot_product.hex | LLM 向量点积 | Coders | 仿真器 | 验证点积时 | hex, llm, dot |
| llm_gelu.hex | LLM GELU 激活函数 | Coders | 仿真器 | 验证 GELU 时 | hex, llm, gelu |
| llm_gemm_2x2.hex | LLM 2x2 矩阵乘法 | Coders | 仿真器 | 验证 GEMM 时 | hex, llm, gemm |
| llm_layernorm.hex | LLM LayerNorm 归一化 | Coders | 仿真器 | 验证 LayerNorm 时 | hex, llm, layernorm |
| llm_relu.hex | LLM ReLU 激活函数 | Coders | 仿真器 | 验证 ReLU 时 | hex, llm, relu |
| llm_residual_add.hex | LLM 残差连接加法 | Coders | 仿真器 | 验证残差加时 | hex, llm, residual |
| llm_rmsnorm.hex | LLM RMSNorm 归一化 | Coders | 仿真器 | 验证 RMSNorm 时 | hex, llm, rmsnorm |
| llm_silu.hex | LLM SiLU 激活函数 | Coders | 仿真器 | 验证 SiLU 时 | hex, llm, silu |
| llm_softmax.hex | LLM Softmax 函数 | Coders | 仿真器 | 验证 Softmax 时 | hex, llm, softmax |
| loop_test.hex | 循环控制流测试 | Coders | 仿真器 | 验证循环时 | hex, loop, control |
| matmul_4x4_fp16.hex | 4x4 FP16 矩阵乘法 | Coders | 仿真器 | 验证 matmul 时 | hex, matmul, fp16 |
| multi_op_test.hex | 多操作综合测试 | Coders | 仿真器 | 综合验证时 | hex, multi, ops |
| trig_cos.hex | 余弦三角函数测试 | Coders | 仿真器 | 验证 SFU cos 时 | hex, trig, cos |
| trig_sin.hex | 正弦三角函数测试 | Coders | 仿真器 | 验证 SFU sin 时 | hex, trig, sin |
| trig_tan.hex | 正切三角函数测试 | Coders | 仿真器 | 验证 SFU tan 时 | hex, trig, tan |

## 子目录

| 目录 | 用途 | 文件数 | 写者 | 读者 | 详见 |
|------|------|--------|------|------|------|
| rtl/ | RTL 源码（Verilog modules） | ~63 | Coders | Coders/Jerry | rtl/MANIFEST.md（待建） |
| tb/ | Testbench 验证文件 | ~121 | Coders | Coders | tb/MANIFEST.md |
| docs/ | 设计文档、回顾、分析 | ~9 | Coders/SM Coach | 所有人 | docs/MANIFEST.md |
| tools/ | Python 工具脚本 | ~8 | Coders | Coders | tools/MANIFEST.md |
| asm/ | PTX 汇编源码 + 编译后 hex | ~89 | Coders | 仿真器 | — |
| doc/ | 架构分析、评审、商业化文档 | ~26 | Coders | 所有人 | — |
| examples/ | 示例程序（PTX + Triton） | ~4 | Coders | 所有人 | — |
| hex/ | 预编译 hex 测试程序 | ~3 | Coders | 仿真器 | — |
| scripts/ | 回归测试 + 性能分析脚本 | ~4 | Coders | Coders | — |
| shared/ | Agent 共享数据（DB、日志、任务） | ~5 | Agents | Agents | — |
| test_results/ | 仿真测试结果输出 | ~2 | 仿真器 | Coders | — |
| tests/ | 综合测试套件（Python + PTX） | ~16 | Coders | Coders | — |
| verification_output/ | 验证输出（hex + ptx + 结果 JSON） | ~117 | 仿真器 | Coders | — |
| .github/workflows/ | CI 配置 | — | Jerry | GitHub Actions | — |
| .ctx/ | 上下文文件（自动生成） | — | Agent 自动 | Agent 自动 | — |

## 开发规范

- **PR Merge 前检查**：`make test` 全过 + 针对性测试 + `make lint` 零 warnings + Synthesis 零 errors
- **分支命名**：`issue-NNN/agent-name`（如 `issue-149/opus`）
- **研究/计划文件**：`research-NNN.md` / `plan-NNN.md` 放在 coder 的 workspace，不提交到 repo
