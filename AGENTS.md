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
