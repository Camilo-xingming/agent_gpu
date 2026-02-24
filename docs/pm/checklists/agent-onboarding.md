# Agent 上线 Checklist

> 新增 agent 或切换 backend 时的配置检查。来源：MEMORY.md 配置踩坑记录。

## 新 Agent 上线

- [ ] `openclaw.json` agents.list 中添加 agent 配置
- [ ] model ID 无 `[1m]` 等后缀（历史遗留问题）
- [ ] 设置 Discord bot token 和 requireMention
- [ ] 创建 agent 目录：`~/.openclaw/agents/{agent-id}/`
- [ ] 创建 agent-context.json（初始 current=null）
- [ ] 创建 sessions/sessions.json（初始 `{}`）
- [ ] 创建 SOUL.md（包含：身份、频道规则、Session Start Review、Context Update、Background Task 指南）
- [ ] 如使用自定义 CLI backend：创建 wrapper script（必须加 `--dangerously-skip-permissions`）
- [ ] 测试消息收发：发 Discord 消息确认 agent 响应
- [ ] 更新 RACI.md 和 communication-plan.md

## Backend 切换

- [ ] 更新 `openclaw.json` 中 model 配置
- [ ] **清除旧 session store**（旧 session ID 指向旧 config dir，新 backend 找不到）
- [ ] 如跨 provider：确认 tool_use.id 格式兼容（跨 provider 会冲突）
- [ ] 如需 proxy bypass：配置 `cliBackends.env`（如 CoderCodex 的 NO_PROXY）
- [ ] `openclaw gateway restart`
- [ ] 测试完整流程：消息 → CLI 调用 → 响应

## 常见踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| CLI 启动后立即退出 | wrapper 没加 `--dangerously-skip-permissions` | 检查 wrapper script |
| "All models failed" | 旧 session 膨胀 | 清 session store |
| 消息不响应 | requireMention 但没用 `<@ID>` | 用 Discord ID 格式 |
| tool_use.id 报错 | 跨 provider fallback | 同 provider 内 fallback |

---

*更新时间: 2026-02-24*
