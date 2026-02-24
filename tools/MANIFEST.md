# tools/ — Python 工具脚本

> 编译、测试、性能分析辅助工具。通过 Makefile 目标调用。

## 文件清单

| 文件 | 用途 | 写者 | 读者/调用者 | 何时用 | 关键词 |
|------|------|------|------------|--------|--------|
| ptx_assembler.py | PTX 汇编器（文本 → 二进制） | Coders | Makefile / Coders | 编译 PTX 测试时 | ptx, assembler |
| gpu_simulator.py | GPU 仿真驱动 | Coders | Makefile | 跑仿真时 | simulator, sim |
| cuda_kernel_compiler.py | CUDA-like C 前端编译器（C-like kernel -> PTX） | Coders | Makefile / Coders | 验证 CUDA-like 编译链路时 | cuda, compiler, ptx |
| perf_dashboard.py | 性能 Dashboard（IPC/stall/utilization） | Coders | `make dashboard` | 性能评估时 | performance, ipc, stall |
| perf_report.py | 性能报告生成 | Coders | Makefile | PR review 时 | performance, report |
| rtl_frm_compare.py | RTL vs FRM 对比验证 | Coders | `make frm-compare` (CI) | 每次 CI | frm, compare, verify |
| test_generator.py | 测试用例生成器 | Coders | Makefile | 新增测试时 | test, generator |
| run_single_test.py | 单个测试运行器 | Coders | 手动 | 调试单个测试时 | test, runner |
| verification_framework.py | 验证框架 | Coders | Makefile | 集成验证时 | verify, framework |
| check-manifest.sh | CI 检查：MANIFEST.md 覆盖率（检测未列入的文件） | Jerry | CI (`manifest-check` job) | 每次 PR | ci, manifest, coverage |

## 子目录

### debug/ — Debug 注入脚本

向 RTL/TB 源码注入 `$display` 调试输出的 Python 脚本（12 个文件）。

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

### fixes/ — RTL/TB 自动修复脚本

自动修复 RTL/TB 常见问题的脚本（10 个文件）。

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
| reapply_fixes.sh | 批量重新应用 RTL 修复（branch stall + R0 scoreboard） |
