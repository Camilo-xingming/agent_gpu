# RalphGPU — CUDA-compatible GPU RTL (Verilog)

> 开源 GPU 设计项目。RTL 实现 + PTX 兼容 + 仿真验证。

## 根目录文件

### 项目核心

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| README.md | 项目概述 + 快速开始 | Jerry/Coders | 所有人 | 初次了解 | overview, readme |
| Makefile | 编译、测试、lint、dashboard 目标 | Coders | Coders | 每次构建/测试 | build, test, lint |
| .gitignore | Git 忽略规则 | Coders | Git | — | git, ignore |
| MANIFEST.md | 文件索引（本文件） | All | All | 查找文件时 | index, manifest |
| plan-147-coalescing.md | Issue #147 内存合并单元实现计划 | CoderOpus | Coders | 实现 #147 时 | plan, coalescing, memory |
| AGENTS.md | Heartbeat 规则：长任务（>3min）需每 60s 发 ⏳ 状态 | Coders | Coders | 长任务执行时 | heartbeat, agents, rules |

## 子目录

| 目录 | 用途 | 文件数 | 写者 | 读者 | 详见 |
|------|------|--------|------|------|------|
| rtl/ | RTL 源码（Verilog modules） | ~63 | Coders | Coders/Jerry | rtl/MANIFEST.md（待建） |
| tb/ | Testbench 验证文件 | ~121 | Coders | Coders | tb/MANIFEST.md |
| programs/ | Hex 测试程序（GPU 指令内存镜像） | ~22 | Coders | 仿真器 | 见下方 |
| docs/ | 设计文档、回顾、分析、进度 | ~18 | Coders/Jerry | 所有人 | docs/MANIFEST.md |
| tools/ | Python 工具 + debug 注入 + 自动修复脚本 | ~20 | Coders | Coders | tools/MANIFEST.md |
| tests/ | 综合测试套件（Python + PTX） | ~20 | Coders | Coders | — |
| asm/ | PTX 汇编源码 + 编译后 hex | ~89 | Coders | 仿真器 | — |
| doc/ | 架构分析、评审、商业化文档 | ~26 | Coders | 所有人 | — |
| examples/ | 示例程序（PTX + Triton） | ~4 | Coders | 所有人 | — |
| hex/ | 预编译 hex 测试程序 | ~3 | Coders | 仿真器 | — |
| scripts/ | 回归测试 + 性能分析脚本 | ~4 | Coders | Coders | — |
| shared/ | Agent 共享数据（DB、日志、任务） | ~5 | Agents | Agents | — |
| test_results/ | 仿真测试结果输出 | ~2 | 仿真器 | Coders | — |
| verification_output/ | 验证输出（hex + ptx + 结果 JSON） | ~117 | 仿真器 | Coders | — |
| .github/workflows/ | CI 配置 | — | Jerry | GitHub Actions | — |
| .ctx/ | 上下文文件（自动生成） | — | Agent 自动 | Agent 自动 | — |

### programs/ — Hex 测试程序

GPU 指令内存镜像文件，供 testbench `$readmemh` 加载。

| 文件 | 用途 | 关键词 |
|------|------|--------|
| batched_matmul_4x4x4_fp16.hex | 批量 4x4x4 FP16 矩阵乘法测试 | matmul, fp16, batch |
| batched_matmul_4x4x4_fp16_looped.hex | 带循环的批量 4x4x4 FP16 矩阵乘法 | matmul, fp16, loop |
| divergence_test.hex | 线程分歧测试 | divergence |
| dp4a_simple.hex | DP4A 整数点积指令简单测试 | dp4a, int8 |
| dp4a_top_test.hex | DP4A 顶层集成测试 | dp4a, top |
| gemm16_fma.hex | 16x16 GEMM FMA 测试 | gemm, fma |
| llm_attention_score.hex | LLM 注意力分数计算（Q*K^T + DP4A） | llm, attention |
| llm_dot_product.hex | LLM 向量点积 | llm, dot |
| llm_gelu.hex | LLM GELU 激活函数 | llm, gelu |
| llm_gemm_2x2.hex | LLM 2x2 矩阵乘法 | llm, gemm |
| llm_layernorm.hex | LLM LayerNorm 归一化 | llm, layernorm |
| llm_relu.hex | LLM ReLU 激活函数 | llm, relu |
| llm_residual_add.hex | LLM 残差连接加法 | llm, residual |
| llm_rmsnorm.hex | LLM RMSNorm 归一化 | llm, rmsnorm |
| llm_silu.hex | LLM SiLU 激活函数 | llm, silu |
| llm_softmax.hex | LLM Softmax 函数 | llm, softmax |
| loop_test.hex | 循环控制流测试 | loop, control |
| matmul_4x4_fp16.hex | 4x4 FP16 矩阵乘法 | matmul, fp16 |
| multi_op_test.hex | 多操作综合测试 | multi, ops |
| trig_cos.hex | 余弦三角函数测试 | trig, cos |
| trig_sin.hex | 正弦三角函数测试 | trig, sin |
| trig_tan.hex | 正切三角函数测试 | trig, tan |

### tools/debug/ — Debug 注入脚本

向 RTL/TB 注入调试 `$display` 输出的 Python 脚本。

| 文件 | 用途 |
|------|------|
| add_atomic_debug.py | 向 atomic_unit.v 注入状态机调试 |
| add_branch_stall_debug.py | 向调度器注入 branch stall 调试 |
| add_hazard_detail.py | 向调度器注入 hazard 细节调试 |
| add_imem_debug.py | 向 testbench 注入 IMEM 请求/响应调试 |
| add_ready_debug.py | 向调度器注入 warp_ready 调试 |
| add_sched_inputs_debug.py | 向调度器注入输入信号调试 |
| add_scheduler_debug.py | 向调度器注入 eligibility 调试 |
| add_stall_debug.py | 向调度器注入 warp stall 条件调试 |
| add_vcd_dump.py | 向 testbench 注入 VCD 波形 dump |
| check_imem_runtime.py | 向 testbench 注入 IMEM 运行时检查 |
| debug_atomic_serialization.py | 向 atomic_unit.v 注入序列化调试 |
| safe_atomic_debug.py | 向 atomic_unit.v 注入安全版调试 |

### tools/fixes/ — RTL/TB 自动修复脚本

| 文件 | 用途 |
|------|------|
| apply_d1_patch.py | 为 SM v2 添加 warp_inst_valid_d1 寄存器 |
| cleanup_tb.py | 清理 testbench 中重复的 VCD dump 块 |
| fix_axi_model.py | 修复 testbench 中的 AXI 模型逻辑 |
| fix_branch_stall.py | 移除调度时过早的 branch stall 设置 |
| fix_debug_syntax.py | 修复损坏的 $display 调试语法 |
| fix_dup_display.py | 移除调度器中重复的 $display 行 |
| fix_r0_scoreboard.py | 修复 R0 不应进入 scoreboard 的 bug |
| fix_tb_structure.py | 修复 testbench 文件结构 |
| fix_testbench.py | 修复 testbench hex 路径 + 加载调试 |
| reapply_fixes.sh | 批量重新应用 RTL 修复 |

### docs/ — 新增文档（从根目录移入）

| 文件 | 用途 |
|------|------|
| design_document.md | 架构设计文档 |
| goal.md | 项目目标 |
| PERFORMANCE_REPORT.md | 性能报告 |
| findings.md | 调查发现记录 |
| progress.md | 进度记录 |
| task_plan.md | 任务计划 |
| plan-149.md | Issue #149 实现计划 |
| research-149.md | Issue #149 调研 |
| DESIGN_REPLAY.md | Pipeline Replay è®¾è®¡å®žçŽ° |
| fpga-prototype-feasibility.md | FPGA åŽŸåž‹éªŒè¯å¯é¡æ€§åˆ†æž |
| issue-151-command-processor-plan.md | Command Processor å®žçŽ°è®¡åˆ’ |
| memory_coalescing_unit_design.md | å†…å­˜åå¹¶å•å…ƒè®¾è®¡ |

### tests/ — 新增测试（从根目录移入）

| 文件 | 用途 |
|------|------|
| test_gemm.py | 测试 2x2 GEMM 矩阵乘法 |
| test_llm_suite.py | LLM 算子测试套件 |
| test_nano_llm.py | 测试 Nano-LLM |
| test_transformer.py | 测试完整 Transformer Block |

## 开发规范

- **PR Merge 前检查**：`make test` 全过 + 针对性测试 + `make lint` 零 warnings + Synthesis 零 errors
- **分支命名**：`issue-NNN/agent-name`（如 `issue-149/opus`）
- **研究/计划文件**：`research-NNN.md` / `plan-NNN.md` 放在 coder 的 workspace，不提交到 repo
