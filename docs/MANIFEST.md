# docs/ — 设计文档、分析报告、Sprint 回顾

> 持久化的技术分析和流程记录。Sprint 回顾由 Lily cron 自动追加。

## 文件清单

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| RETRO.md | Sprint 回顾记录 | Lily (retro cron) / SM Coach | Lily (Sprint Planning) | 每日 Planning 09:00 | retro, sprint, lessons |
| TESTING.md | 测试策略和方法 | Coders | Coders | 写新测试前 | test, strategy |
| PTX_COMPATIBILITY.md | PTX 指令兼容性矩阵 | Coders | Coders | 新增指令时 | ptx, compatibility |
| WARP_INST_VALID_D1_STATUS.md | Warp 指令有效性分析 | Coders | Coders | 调试 pipeline 时 | warp, pipeline, debug |
| ARCHITECTURE_GUIDE.md | RalphGPU æž¶æž„æŒ‡å — (Hopper/Blackwell) | Coders | Coders | è®¾è®¡åˆ†æž æ—¶ | architecture, guide |
| ISA_REFERENCE.md | ISA æŒ‡ä»¤é›†å­è€ƒæ‰‹å†Œ | Coders | Coders | æ±‡ç¼–è°ƒè¯•æ—¶ | isa, reference |
| PERF_DASHBOARD.md | æ€§èƒ½çœ‹æ ¿ (MatMul Benchmark) | Coders | Lily/Jerry | æ€§èƒ½åˆ†æž æ—¶ | performance, dashboard |
| atomic-contention-analysis.md | Atomic 竞争分析 | Coders | Coders | atomic 相关 issue | atomic, contention |
| memory-optimization-reference.md | 内存优化参考 | Coders | Coders | 内存子系统优化时 | memory, optimization |
| performance-verification-plan.md | 性能验证计划 | Coders | Coders/Jerry | 性能相关 sprint | performance, verification |
| plan-150-multi-sm.md | Multi-SM 实现计划 (#150) | Coders | Coders | 多 SM 架构时 | multi-sm, axi, plan |
| fpga-prototype-feasibility.md | FPGA 原型可行性评估 | Coders | Jerry/Lily | 做 FPGA 相关决策前 | fpga, prototype, feasibility |

## pm/ 子目录 — 项目管理文档

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| pm/README.md | PM 文档索引 | Jerry/SM Coach | 所有人 | 查找 PM 文档时 | pm, index |
| pm/RACI.md | 职责分配矩阵 (R/A/C/I) | Jerry/SM Coach | 所有人 | 职责不清时 | raci, roles, responsibility |
| pm/DoD.md | Definition of Done（PR 级 + Sprint 级） | Jerry/SM Coach | Lily + Coders | PR review / Sprint Review | dod, done, criteria |
| pm/communication-plan.md | 沟通计划：频道矩阵、升级路径、标注规则 | Jerry/SM Coach | 所有人 | 新 agent 上线 / 沟通问题时 | communication, discord, escalation |
| pm/decision-log.md | 决策日志：重要技术和流程决策追溯 | Jerry/Lily | 所有人 | 追溯决策原因时 | decision, log, why |
| pm/velocity.md | Velocity 追踪：每 Sprint 计划/完成统计 | Lily (review cron) | Jerry/Lily | Sprint Review / Planning | velocity, sprint, metrics |

## pm/checklists/ 子目录 — 操作清单（来自 RETRO lesson learned）

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| pm/checklists/pr-submission.md | PR 提交前自检 | SM Coach / RETRO | Coders | 提 PR 前 | pr, checklist, submission |
| pm/checklists/sprint-planning.md | Sprint Planning 检查项 | SM Coach / RETRO | Lily | Planning 前/中/后 | planning, checklist |
| pm/checklists/post-incident.md | 故障恢复后清理确认 | SM Coach / RETRO | Lily / Jerry | 故障恢复后 | incident, recovery, cleanup |
| pm/checklists/agent-onboarding.md | 新 agent 上线配置检查 | SM Coach / RETRO | Jerry | 新增/切换 agent 时 | agent, onboarding, config |

## architecture/ 子目录

| 文件 | 用途 | 写者 | 读者 | 关键词 |
|------|------|------|------|--------|
| dual-issue.md | 双发射架构设计 | Coders | Coders | dual-issue, scheduler |
| waw-analysis.md | WAW 冒险分析 | Coders | Coders | waw, hazard, scoreboard |

## 历史工作文件（Coder 产出，按需查阅）

| 文件 | 用途 | 写者 | 读者 | 关键词 |
|------|------|------|------|--------|
| PERFORMANCE_REPORT.md | 性能测试报告 | Coders | Coders/Jerry | performance, report |
| design_document.md | 设计文档 | Coders | Coders | design |
| findings.md | 问题调查结论 | Coders | Coders | findings, debug |
| goal.md | 目标定义文档 | Coders | Coders | goal |
| plan-149.md | Issue #149 实现计划 | Coders | Coders | plan, 149, scheduler |
| progress.md | 进度跟踪文档 | Coders | Coders | progress |
| research-149.md | Issue #149 调研报告 | Coders | Coders | research, 149 |
| task_plan.md | PTX 测试套件任务计划 | Coders | Coders | task, ptx, test |
