# RalphGPU — CUDA-compatible GPU RTL (Verilog)

> 开源 GPU 设计项目。RTL 实现 + PTX 兼容 + 仿真验证。

## 根目录文件

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| README.md | 项目概述 + 快速开始 | Jerry/Coders | 所有人 | 初次了解 | overview, readme |
| Makefile | 编译、测试、lint、dashboard 目标 | Coders | Coders | 每次构建/测试 | build, test, lint |
| design_document.md | 架构设计文档 | Jerry/Coders | 所有人 | 设计决策时 | architecture, design |
| goal.md | 项目目标 | Jerry | 所有人 | Planning 时 | goal, vision |
| PERFORMANCE_REPORT.md | 性能报告 | Coders | Jerry | Review 时 | performance, benchmark |
| findings.md | 调查发现记录 | Coders | Coders | 调试时 | debug, findings |
| progress.md | 进度记录 | Coders | Jerry | Review 时 | progress, status |
| prompt.md | Agent 初始 prompt | Jerry | Agents | 启动时 | prompt |
| task_plan.md | 任务计划 | Jerry | Coders | Sprint Planning | plan, tasks |

## 子目录

| 目录 | 用途 | 文件数 | 写者 | 读者 | 详见 |
|------|------|--------|------|------|------|
| rtl/ | RTL 源码（Verilog modules） | ~63 | Coders | Coders/Jerry | rtl/MANIFEST.md（待建） |
| tb/ | Testbench 验证文件 | ~121 | Coders | Coders | tb/MANIFEST.md |
| docs/ | 设计文档、回顾、分析 | ~9 | Coders/SM Coach | 所有人 | docs/MANIFEST.md |
| tools/ | Python 工具脚本 | ~8 | Coders | Coders | tools/MANIFEST.md |
| .github/workflows/ | CI 配置 | — | Jerry | GitHub Actions | — |
| .ctx/ | 上下文文件（自动生成） | — | Agent 自动 | Agent 自动 | — |

## 开发规范

- **PR Merge 前检查**：`make test` 全过 + 针对性测试 + `make lint` 零 warnings + Synthesis 零 errors
- **分支命名**：`issue-NNN/agent-name`（如 `issue-149/opus`）
- **研究/计划文件**：`research-NNN.md` / `plan-NNN.md` 放在 coder 的 workspace，不提交到 repo
