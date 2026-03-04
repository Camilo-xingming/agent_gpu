# 沟通计划 (Communication Plan)

> 谁在哪里发什么内容、什么频率、怎么升级。

## 频道矩阵

| 频道 | 用途 | 谁发 | 频率 | 内容规范 |
|------|------|------|------|----------|
| #ralphgpu | Jerry 专属频道 | Lily (Sprint Review/Planning 汇报) | 每 Sprint | Sprint 成果 demo、重大决策通知 |
| #ralphgpu-dev | 开发主频道 | Lily + Coders | 持续 | 任务分配、完成通知、Blocker 报告 |
| #ralphgpu-logs | 详细日志 | Coders | 按需 | 调试过程、长日志、代码 diff |
| #ralphgpu-memory | 公告 | Lily | 低频 | Sprint 开始/结束、里程碑公告 |
| GitHub Issues | 任务跟踪 | 所有人 | 持续 | Issue comment 记录进展和决策 |
| GitHub PRs | 代码审查 | Coders (提交) + Lily (review) | 持续 | PR description + review comment |

## Discord 消息规范

### 硬性规则

1. **#ralphgpu-dev 消息 1-3 行**，只发：结论、行动、结果、具体请求
2. **禁止在 Discord 进行技术讨论** — 技术讨论用文件标注（见下方）
3. 分析过程、代码 diff、长日志 → 发 #ralphgpu-logs 或不发
4. 禁止"想出声"：不发推理过程、不发"我先看..."、不发中间分析
5. 同一问题不连发多条，合并成 1 条总结 + 1 条任务指派

### 消息模板

```
任务完成：[描述]。PR: [链接]
Blocker：[Issue 编号] 被 [原因] 阻塞，需要 [谁] [做什么]
分配任务：<@BOT_ID> [具体任务描述]，参考 [Issue/文件链接]
```

## 文件标注（主要协作方式）

文件标注优先于 Discord，是技术讨论和 review 的主要方式。

### 标注格式

```
> @AgentName YYYY-MM-DD: [标注内容]
```

### 文件位置约定

| 内容类型 | 文件位置 |
|----------|----------|
| Issue 相关讨论 | `gh issue comment #NNN` |
| 设计方案 review | `docs/design-issue-NNN.md` |
| 跨 agent 协调 | `~/.openclaw/shared-memory/ralphgpu/` |
| 研究/计划 | `research-NNN.md` / `plan-NNN.md`（workspace 内） |

### 工作流

1. Agent A 在文件中写内容
2. Agent A 在 #ralphgpu-dev 发 1 行通知 + 文件路径
3. Agent B 读文件、在文件中添加标注
4. Agent B 在 #ralphgpu-dev 发 1 行通知"已标注"

## 升级路径

```
Coder 遇到问题
  |
  v
在 #ralphgpu-dev 报告 Blocker（1 行）
  |
  v
Lily 尝试解决（重新分配、调整 scope、提供指导）
  |
  v
连续 3 次 Standup 无进展？
  |-- 是 --> Lily 升级给 Jerry（#ralphgpu 或 Issue comment）
  |-- 否 --> 继续 Lily 跟进
```

## Standup 格式（每小时自动触发）

Coder 需回答 3 个问题（Standup cron 自动收集 GitHub activity）：

1. **上次以来完成了什么？** — 已合并的 PR、已关闭的 Issue
2. **接下来做什么？** — 当前 assigned Issue
3. **有什么阻碍？** — Blocker（如有）

> 注：Standup 由 cron 自动执行，从 GitHub 数据采集，不需要 Coder 主动发消息。异常（无进展、Blocker）时 Lily 在 #ralphgpu-dev 跟进。

## Sprint 会议（Thread 形式，2026-02-28 起执行）

所有 Sprint 会议在 **#ralphgpu** 以 thread 形式进行，Lily 不得单独决定 Sprint 内容。

| 会议 | Thread 标题 | 流程 |
|------|------------|------|
| Planning | "Sprint N Planning" | Lily 提候选 → Coders 讨论 → 共识 → 创建 Issues |
| Review | "Sprint N Review" | 逐 Issue 验收 → 记录到 GitHub |
| Retro | "Sprint N Retro" | 收集 lessons → 更新 KNOWLEDGE.md |

Thread 即会议记录，讨论完产出最终文档到 GitHub。

## 信息持久化原则

Discord 只是通知副本，所有有价值的信息必须沉淀到文档：

| 信息 | 持久化位置 |
|------|------------|
| Sprint Review 结果 | GitHub Milestone description + Issue comments |
| Sprint Planning 结果 | GitHub Milestone description |
| Sprint 会议讨论过程 | #ralphgpu thread（自动保留） |
| Blocker / 无进展 | GitHub Issue comment |
| 技术决策 | `docs/pm/decision-log.md` |
| 经验教训 | `docs/RETRO.md` |

---

*更新时间: 2026-02-28*
