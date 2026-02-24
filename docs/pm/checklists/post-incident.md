# Post-Incident Checklist

> 故障恢复后确认所有清理步骤已执行。来源：session 膨胀、"All models failed" 经验。

## "All models failed" 恢复

- [ ] 检查 `/tmp/openclaw/session-cleanup.log` 最近记录
- [ ] 确认触发的 agent 的 session 文件大小：`ls -lh ~/.claude/projects/-Users-jerry--openclaw-workspace*/*.jsonl | sort -k5 -h`
- [ ] 如有 >150KB 的 session：手动删除或等 30min 自动清理
- [ ] 运行 `python3 ~/.openclaw/scripts/cleanup-stale-sessions.py` 清除 stale 引用
- [ ] 确认 agent 已恢复工作（检查 Discord 最新消息或 agent-context.json）
- [ ] 如频繁发生（>3 次/小时）：检查 `openclaw logs` 是否有 CLI daemon 崩溃

## Agent 超时/Crash 恢复

- [ ] 检查 agent-context.json 的 phase（知道断在哪）
- [ ] 检查 git status：有无未 push 的代码
- [ ] 有未 push 代码 → 先 commit + push 保存
- [ ] 在 #ralphgpu-dev 通知 Lily agent 状态
- [ ] 如需 reassign：Lily 更新 GitHub Issue assignee + 通知新 agent

## OpenClaw 配置变更后

- [ ] `openclaw gateway restart` 已执行
- [ ] 所有 agent 测试连通（发一条消息看回复）
- [ ] `openclaw logs` 无异常 error
- [ ] 如改了 patch 脚本：确认 patch 已重新应用

## Auth-Profile Cooldown 恢复

- [ ] 找到 `~/.openclaw/agents/*/agent/auth-profiles.json`
- [ ] `errorCount` 重置为 0
- [ ] 删除 `cooldownUntil` 和 `failureCounts` 字段
- [ ] `openclaw gateway restart`

---

*更新时间: 2026-02-24*
