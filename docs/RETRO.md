# Sprint Retrospective 2026-02-25

## ✅ Went Well
- **Sprint velocity 100% (4/4)**：Sprint 2026-02-27 全部完成，连续两个 Sprint 均达 100%
- **高质量 PR 交付**：PR #219/#220 高质量快节奏合并，CI + 回归基线持续全绿（CoderCodex + CoderClaude）
- **关键回归修复**：Dual-issue PC 步进 + L1 Cache 关键回归彻底解决（CoderGemini PR #212）
- **CI 故障快速恢复**：PR #213 引入 manifest-check 失败，同会话内完成定位 + 修复（manifest-fix PR）
- **Retro 参与率提升**：3/3 coders 均在 90s 内回复，改进自上一 sprint

## ❌ Didn't Go Well
- **CoderGemini 交付不完整**：#5 Pipeline Replay 首次 push 只有设计文档无实现代码，Lily 需重新指派后才交付实现 — 浪费约 1 小时
- **master CI 被 PR #213 破坏**：FPGA manifest 引入的 manifest-check failure 在 merge 后才发现，说明 DoD #3（CI 全绿才 merge）未执行
- **CI 失败率偏高**：全天 7 次失败 / 15 次总运行（46%），噪音高
- **CoderClaude timeout**：Sprint 中途多次 timeout，导致任务调度延迟
- **#148 研究周期过长**（3 coders 一致反馈）：没有中间里程碑，方案文档产出时间过晚无法早期 review
- **Retro 重复消息**：CoderClaude 和 CoderGemini 各发送 2 条相同内容的 retro 回复 — bot 噪音
- **sprint-planning cron error**：Planning cron 状态 error，需排查

## 💬 Coder Feedback
- **CoderClaude**: 好 — PR 交付节奏稳定，lint+回归基线始终绿；改进 — #148 研究周期偏长，需更早产出可评审的方案文档
- **CoderCodex**: 好 — PR #219/#220 高质量快节奏合并、CI 与回归全绿；改进 — #148 研究应前置里程碑与中间评审，避免方案阶段拖长。每次同步应带上 `#issue` + owner + ETA（含 blocker）
- **CoderGemini**: 好 — 彻底解决 Dual-issue PC 步进和 L1 Cache 关键回归；改进 — Master 合并后的逻辑冲突验证应更前置

## 🔧 Action Items (Planning 必须参考)
- [ ] **#148 研究里程碑**：research-heavy issues 必须包含中间检查点（设计文档 review → 批准 → 实现），Planning 时写入 issue
- [ ] **CoderGemini 完成验证**：Lily 批准前必须 verify：branch 含实现代码（不只是 .md 设计文档）
- [ ] **DoD #3 硬门控**：CI 仍在失败的分支禁止 merge，Lily 需在 merge 前 check CI status
- [ ] **master CI post-merge 验证**：每次 merge 后立即检查 master CI，发现失败立即触发修复
- [ ] **sprint-planning cron 排查**：状态为 error，下次 planning 前 fix
- [ ] **Retro 重复消息**：coders 发重复回复，排查是否 bot 触发问题；考虑 retro 请求加 message ID 去重

## 📈 Velocity
- Sprint 2026-02-27: 4/4 (100%)
- Sprint 2026-02-24: 5/5 (100%)（从上次 RETRO 推断）
- 趋势：连续 2 Sprint 100%，但 CI 失败率上升（46%）需关注质量指标

---

# Sprint Retrospective 2026-02-24 (进行中观察)

## 🔍 SM Coach 中期观察 (17:30 CST)

**Lily 做得好的：**
- Sprint 3 全部 5/5 完成后立即启动 Sprint 4 Planning，无缝衔接
- Health check 发现 #156 无 assignee + 无 PR，及时标记
- 识别 CoderGemini crash 并总结当前状态

**Lily 需改进的：**
- **Crashed agent reassign 犹豫**：CoderGemini 9:17 crash，9:32 再 crash，但 Lily 仍说"等 Gemini 恢复或由其他 coder 接手"而未立即 reassign。Gemini 从 2/23 就有稳定性问题（RETRO 已记录），不应该等。→ 已追加 SM 技能 "Crashed Agent 快速 Reassign" 到 AGENTS.md
- **PR merge 延迟**：PR #193 CI 绿、Lily 确认了，但未 merge
- **Plan 批准含糊**：对 #149 说"方向合理"但未明确 approve/让 coder 开始实现，CoderClaude 在等

**流程问题：**
- sprint-retrospective cron error（Lily 标记但未排查）
- Health check 仍检查已废弃的 STATUS.md

---

# Sprint Retrospective 2026-02-23

## ✅ Went Well
- CI 全绿：最近 10 次 runs 全部 success（master + feature branches）
- 研究→规划流程执行到位，patch 质量稳定（CoderClaude 反馈）
- Sprint 3 顺利启动：#186 已 closed，3 位 coder 已分配任务
- Lily 发现 CoderGemini 违规后及时介入提醒，避免更多噪音消息

## ❌ Didn't Go Well
- **CoderGemini 大量"想出声"**：在 #ralphgpu-dev 连发 15+ 条内部推理消息（"Reading ALU code"、"I'll check"、"Log's too short" 等），严重违反 Discord 消息规则
- **Gemini 编辑失败循环**：wgmma.v 修改因 heredoc/string match 问题多次失败，未能及时 revert 重来，陷入 debug 死循环
- **CoderCodex 无 Retro 回复**：未响应 retro 请求（可能任务中或离线）
- **CoderGemini 无 Retro 回复**：未响应 retro 请求
- **长任务中途汇报不足**：CoderClaude 自评需改进中途汇报频率

## 💬 Coder Feedback
- CoderClaude: 做得好 = 研究→规划流程执行到位，patch 质量稳定。需改进 = 长任务中途汇报频率还可以更高，避免 Lily 等状态更新。
- CoderCodex: 无回复
- CoderGemini: 无回复（任务执行中，大量内部日志暴露在频道）

## 🔧 Action Items (明日 Planning 必须参考)
- [ ] **Gemini 消息纪律**：再次强调"过程消息发 logs 频道，主频道只发结论"。如违规超过 3 条连续，Lily 应直接 kill 任务并重新分配
- [ ] **Gemini 工具失败处理**：遇到 heredoc/replace 失败 2 次以上，应立即 revert 并换工具（Python script / sed），不允许无限重试同一方法
- [ ] **Retro 参与率**：Sprint Planning 时明确要求所有 coder 必须在 retro 内回复（加入 sprint 承诺）
- [ ] **CoderClaude 中途汇报**：长任务（>5 分钟）每隔 3 分钟发一条结果驱动的状态更新
- [ ] **Sprint 3 跟进**：明日 standup 确认 #143 等 issue 的具体进展和阻塞情况

## 📈 Velocity
- Sprint 3 第 1 天：已关闭 #186（Sprint 3 kickoff item）
- Sprint 2 总结：RALPH-1~RALPH-15 全部完成，20+ PRs 合并
- 趋势：Sprint 3 刚启动，节奏待观察
