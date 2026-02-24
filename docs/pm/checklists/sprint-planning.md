# Sprint Planning Checklist

> Lily 在 Sprint Planning 前/中/后必须确认的事项。来源：RETRO action items + SM 辅导记录。

## Planning 前（前一天晚上确认）

- [ ] Backlog triage 完成：候选 Issue 都有 priority label（P1/P2）和 size label（S/M/L）
- [ ] 上一个 Sprint 的 carry-over items 已明确处理（移入/scope down/关闭）
- [ ] 上一个 Sprint 的 RETRO action items 已 review，需要带入新 Sprint 的标记
- [ ] Jerry 有无新需求或优先级调整（检查 #ralphgpu 最近消息）

## Planning 中

- [ ] Sprint Goal 写入 Milestone description
- [ ] 每个 Issue 有 assignee（GitHub 上设置，不只是口头说）
- [ ] 每个 Issue 有明确的验收标准（DoD 可检查）
- [ ] WIP 限制 = 1/coder，不超配
- [ ] 总容量合理：Sprint 时长 x coder 数 >= Issue 总量
- [ ] 通知所有 coder 各自任务（#ralphgpu-dev @mention）

## Planning 后

- [ ] Milestone 创建且状态 = open
- [ ] 所有 Sprint items 关联到 Milestone
- [ ] 结果同步给 Jerry（#ralphgpu 简报）

## 常见问题（从 RETRO 总结）

| 问题 | 根因 | 预防 |
|------|------|------|
| Issue 无 assignee | Planning 时只口头分配 | 当场在 GitHub 设 assignee |
| Backlog 输入不足 | Triage 未提前做 | 前一天晚上确认 |
| Carry-over 无限期 open | Review 时没处理 | Planning 前强制清理 |
| Agent 未确认任务 | 只发了一条消息没 @mention | 用 `<@BOT_ID>` 格式 |

---

*更新时间: 2026-02-24*
