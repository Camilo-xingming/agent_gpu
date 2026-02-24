# RACI Matrix — RalphGPU Project

> 职责分配矩阵。R=执行 A=负责(最终拍板) C=咨询 I=通知

## 角色

| 缩写 | 角色 | 说明 |
|---|---|---|
| Jerry | Stakeholder | 项目所有者，最终决策权 |
| Lily | PO (Product Owner) | Sprint 管理、任务分配、PR 合并 |
| Coders | Dev Team | CoderOpus / CoderGemini / CoderCodex |
| SM | SM Coach (Jerry's Claude Code) | 流程监控、辅导，不执行任务 |

## 开发流程

| 活动 | R | A | C | I |
|---|---|---|---|---|
| 代码实现 | Coders | Lily | — | Jerry |
| PR 提交 | Coders | Coders | — | Lily |
| PR Review / Merge | Lily (standup 自动触发) | Lily | Coders | Jerry |
| Bug 修复 | Coders | Lily | — | Jerry |
| 技术方案设计 | Coders | Lily | Jerry | — |

## Sprint 管理

| 活动 | R | A | C | I |
|---|---|---|---|---|
| Backlog Triage | Lily | Jerry | — | Coders |
| Sprint Planning | Lily | Lily | Jerry | Coders |
| 任务分配 | Lily | Lily | — | Coders |
| Daily Standup | Coders+Lily | Lily | — | Jerry |
| Sprint Review | Lily | Lily | Jerry | Coders |
| Sprint Retrospective | Lily | Lily | Jerry | Coders |

## 基础设施

| 活动 | R | A | C | I |
|---|---|---|---|---|
| OpenClaw 配置变更 | SM/Jerry | Jerry | — | Lily |
| Agent 模型切换 | SM/Jerry | Jerry | — | Lily |
| Session 清理 (自动) | launchd | Jerry | — | — |
| Patch 脚本维护 | SM/Jerry | Jerry | — | — |
| CI/CD 修改 | Coders/Lily | Jerry | Lily | Coders |

## 流程监控

| 活动 | R | A | C | I |
|---|---|---|---|---|
| Scrum 流程合规检查 | SM | Jerry | — | Lily |
| SM 技能辅导 | SM | Jerry | — | Lily |
| Agent 异常处理 | Lily | Lily | SM | Jerry |
| Blocker 升级 | Lily | Lily | Jerry | Coders |

---

*更新时间: 2026-02-24*
