# tools/ — Python 工具脚本

> 编译、测试、性能分析辅助工具。通过 Makefile 目标调用。

## 文件清单

| 文件 | 用途 | 写者 | 读者/调用者 | 何时用 | 关键词 |
|------|------|------|------------|--------|--------|
| ptx_assembler.py | PTX 汇编器（文本 → 二进制） | Coders | Makefile / Coders | 编译 PTX 测试时 | ptx, assembler |
| gpu_simulator.py | GPU 仿真驱动 | Coders | Makefile | 跑仿真时 | simulator, sim |
| perf_dashboard.py | 性能 Dashboard（IPC/stall/utilization） | Coders | `make dashboard` | 性能评估时 | performance, ipc, stall |
| perf_report.py | 性能报告生成 | Coders | Makefile | PR review 时 | performance, report |
| rtl_frm_compare.py | RTL vs FRM 对比验证 | Coders | `make frm-compare` (CI) | 每次 CI | frm, compare, verify |
| test_generator.py | 测试用例生成器 | Coders | Makefile | 新增测试时 | test, generator |
| run_single_test.py | 单个测试运行器 | Coders | 手动 | 调试单个测试时 | test, runner |
| verification_framework.py | 验证框架 | Coders | Makefile | 集成验证时 | verify, framework |
| check-manifest.sh | CI 检查：MANIFEST.md 覆盖率（检测未列入的文件） | Jerry | CI (`manifest-check` job) | 每次 PR | ci, manifest, coverage |
