# 决策日志 (Decision Log)

> 重要技术和流程决策的追溯记录。每条记录包含背景、备选方案和决策理由。

## 决策记录

| 日期 | 决策 | 背景 | 备选方案 | 决策人 |
|------|------|------|----------|--------|
| 2026-02-24 | 事件驱动 Sprint 周期 | Sprint 完成后需等到 cron 定时才触发 Review，浪费时间 | (A) 固定间隔 cron (B) 手动触发 (C) **事件驱动：完成即触发 Review->Retro->Planning 链** | Jerry |
| 2026-02-24 | 文件标注为主要协作方式 | Discord 消息容易淹没，技术讨论碎片化，不可检索 | (A) Discord 讨论 (B) GitHub Issue comment only (C) **文件标注优先 + Discord 仅通知** | Jerry |
| 2026-02-24 | MANIFEST.md 文件索引体系 | Agent 查找文件效率低，靠 grep 盲搜，浪费 context window | (A) 不做索引 (B) 数据库索引 (C) **每目录 MANIFEST.md 表格** (灵感来自 OpenViking L0/L1/L2) | Jerry |
| 2026-02-24 | Issue-based session rotation | Session 膨胀导致 "All models failed"，需要控制 session 生命周期 | (A) never-resume：每次 fresh session，丢失上下文 (B) always-resume：session 无限增长 (C) **Issue 粒度：一个 Issue 一个 session，完成即清理** | Jerry |
| 2026-02-24 | STATUS.md 废弃，GitHub Milestone 为唯一 Sprint 状态源 | 维护独立状态文件容易过时，与 GitHub 数据不同步 | (A) 继续维护 STATUS.md (B) **GitHub Milestone description + Issues 为唯一真实来源** | Jerry |
| 2026-02-24 | Cron 优化：bash gather + AI 判断 | Cron 每次调用 AI 采集数据，消耗大量 token | (A) 全 AI (B) **bash gather 零 token 采集 → AI 仅处理异常** | Jerry |
| 2026-02-24 | Lily 角色定义为 PO | Lily 之前同时承担 PO + SM 职责，职责模糊 | (A) Lily=SM (B) **Lily=PO，SM 职能由脚本自动执行** | Jerry |

## 记录规范

新增决策时填写以下字段：

- **日期**：决策日期 (YYYY-MM-DD)
- **决策**：一句话描述最终选择
- **背景**：为什么需要做这个决策（痛点是什么）
- **备选方案**：考虑过哪些方案，用 **(X)** 标记最终选择
- **决策人**：谁拍板的（通常是 Jerry，技术细节可能是 Lily）

---

*更新时间: 2026-02-24*
