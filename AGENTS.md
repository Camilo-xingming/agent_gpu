# AGENTS.md — RalphGPU Team Heartbeat Rules

## 长任务心跳机制

所有 coder（CoderCodex / CoderGemini / CoderClaude）执行长任务（>3 分钟）时，必须遵守以下心跳规则。

### 规则

1. **触发条件**：任务预计耗时 > 3 分钟
2. **心跳频率**：每 60 秒发一条状态消息到 **#ralphgpu-dev**（`1475083010968649778`）
3. **心跳格式**：`⏳ [任务名] 进行中：[当前步骤/结果]`
4. **停止条件**：任务完成后停止心跳，发最终结果消息

### 示例

```
⏳ #218 AGENTS.md心跳规则 进行中：已更新 3 个 workspace AGENTS.md
⏳ #150 multi-SM 进行中：lint 通过，开始跑 tb
⏳ #147 coalescing 进行中：发现 arbiter 死锁，正在修
```

### 违规后果

- 长任务无心跳 → Lily 会 timeout 并 reassign
- 心跳内容必须是结果/进展，不是"我在看…"（禁止想出声）

## Sprint Watchdog Cron（每 2 小时）

新增 `scripts/sprint-watchdog.sh` 用于检查当前 Sprint milestone 的 issue 分配与分支启动 SLA（2h）。

建议在 codex 机器安装 cron（每 2 小时执行一次）：

```cron
0 */2 * * * cd ~/RalphGPU-codex && env HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 ./scripts/sprint-watchdog.sh --repo ssql2014/RalphGPU >> ~/.openclaw/shared-memory/ralphgpu/sprint-watchdog.log 2>&1
```

脚本返回码：
- `0`：全部 OK
- `1`：存在 WARN（已分配但超过 2h 无 branch）
- `2`：存在 FAIL（issue 无 assignee）

## Retro Action Items → GitHub Issues（1h SLA）

Retro 产出的 action items 不能只留在文档/聊天，必须进入 GitHub 跟踪。

### 规则

1. **时限**：Retro 结束后 1 小时内，所有未完成 action item 必须创建（或关联）GitHub Issue。
2. **标签**：Retro action item 对应 issue 必须带 `process` label。
3. **去重**：若已有 open issue 覆盖该项，不重复建单；在原 issue comment 增量更新并回链 Retro。
4. **可验收**：每个 action item issue 必须写清可验证的 acceptance criteria。
5. **可追溯**：在 Retro 记录中写明 `Action Item -> Issue #` 映射。

### 模板

```md
Title: process: [Sprint N Retro] <action item short name>

## Background
- Retro source: Sprint N (RETRO.md section or ceremony issue link)
- Problem summary:

## Proposed Change
- 

## Acceptance Criteria
- [ ]

## Owner / Target Sprint
- Owner:
- Target Sprint:

## References
- Retro entry link
- Related issues/PRs
```

## Sprint 会议流程（2026-02-28 起执行）

所有 Sprint 会议必须在 **#ralphgpu** 频道以 **thread** 形式公开讨论。Lily 不得单独决定 Sprint 内容。

### 三个会议

| 会议 | Thread 标题 | 参与者 | 产出 |
|------|------------|--------|------|
| Planning | "Sprint N Planning" | Lily + 全部 Coders | Lily 提候选 items → Coders 讨论可行性/工作量 → 共识 → 创建 Issues + Milestone |
| Review | "Sprint N Review" | Lily + 全部 Coders | 逐 Issue 验收结果，记录到 Issue comment + Milestone description |
| Retro | "Sprint N Retro" | Lily + 全部 Coders | 收集 lessons learned → 更新 KNOWLEDGE.md + docs/RETRO.md |

### 流程

1. Lily 在 #ralphgpu 开 thread（标题格式固定如上）
2. Lily 提议议题/候选 items，引导讨论
3. Coders 在 thread 内参与讨论（评估工作量、提出风险、建议优先级）
4. 达成共识后，Lily 将结论写入 GitHub（Issue/Milestone/RETRO.md）；Retro action items 必须在 1h 内映射到 GitHub Issues
5. Thread 本身即会议记录，可追溯
