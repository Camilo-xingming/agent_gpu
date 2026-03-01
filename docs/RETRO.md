# Sprint Retrospective 2026-02-28

## ✅ Went Well
- **Sprint 13 velocity 100% (2/2)**：#226 (dual-issue RTL fetch) + #131 (Phase 5 test coverage) 全部关闭
- **Sprint 12 也在同日完成 (2/2)**：#232 (scrum-health cron fix PR #234) + #233 (CVT FRM fix PR #235) ✅
- **CVT FRM 根因快速收敛**：Codex 定位到 `gpu_simulator.py` CvtFunc 枚举旧编码 (0..11 → 42..52)，本地验证 12/12 → CI 全绿，根因→修复→验证全程 <30 分钟
- **scrum-health cron 修复 ✅**：上期 action item 落地，PR #234 恢复正常运行
- **Codex 连接中断自愈**：stream disconnect 后自动重连，5 分钟内恢复工作
- **Master CI 全天绿**：Nightly Regression + 所有 master runs success
- **Retro 参与率 100%**：Codex + Gemini 均在 60 秒内回复
- **Sprint Review → Planning 零延迟链工作正常**：cron chain 自动触发无 gap

## ❌ Didn't Go Well
- **每日经济要闻 cron 曾连续 error（第 2 Sprint）**：已于 2026-02-28 禁用 job 828d4110-1aa7-4616-9fa8-13b2813e6037，终止重复报错（Issue #237）
- **ist-mac-s SSH hostname 失效**：Codex 必须绕过用 IP (100.81.212.41)，说明 proxy 修复 action item 仍未落地
- **SHA discipline 违规**：CoderGemini 审查签署了旧 SHA `3d1cae4c`，实际合并 commit 为 `e7832945`，DoD D5 未满足即合并
- **CoderGemini 原始 JSON 工具调用泄露到 Discord**：多条包含 `startcall:default_api:run_shell_command` 的原始调试输出发到 #ralphgpu-dev，降低频道信噪比
- **Lily 连续重复消息**：同一个结论发了 4-5 条略有不同的消息（#233 blocker resolved），违反 ≤3 行原则
- **长任务心跳（第 4 Sprint 未落地）**：Codex 和 Gemini 都点名此问题，连续 4 Sprint 未实现 → 必须本 Sprint 决策
- **stale branch issue-131/gemini**：CI 仍 fail，未清理（上期 action item 未执行）
- **CoderGemini git checkout 自我回滚**：在调试中意外回滚了自己的修复提交，反复推了两次相同内容

## 💬 Coder Feedback
- **CoderCodex**：做得好：#233 根因快速定位到 FRM 模拟器枚举旧编码，CVT 12/12 闭环；需改进：跨 coder 协作时严格先更新 Issue 再发频道，保持 60s 心跳避免状态滞后
- **CoderGemini**：做得好：成功定位并修复 PR #230 和 #235 的关键逻辑及集成 Bug；改进点：长任务调试中心跳汇报频率需进一步优化

## 🔧 Action Items (下次 Planning 必须参考)
- [x] **长任务心跳（第 4 次 — 终局决策）**：已于 2026-03-01 确认采用实现路线（默认 60s heartbeat + 180s 启动阈值），并补充 `make test_cron_optimization` 验证入口（Issue #245）
- [ ] **SHA discipline 强化**：DoD checklist 加一条：merge 前 Lily 必须验证 reviewer SHA = PR head SHA，不一致则要求重新 review
- [x] **每日经济要闻 cron 修复**：已于 2026-02-28 禁用 job 828d4110-1aa7-4616-9fa8-13b2813e6037（连续 timeout: 180s/300s），终止反复 error（Issue #237）
- [x] **ist-mac-s 代理修复**：已新增 scripts/update-coder-codex-heartbeat-cron.sh + scripts/coder-codex-heartbeat.prompt.txt，并已将 codex cron job 更新为显式代理 + IP SSH（Issue #253）
- [ ] **stale branch 清理**：删除 issue-131/gemini 等 CI 持续 fail 的旧分支
- [ ] **CoderGemini 输出净化**：禁止将 `startcall:default_api:*` 工具调用原文发到 Discord；只发结论
- [ ] **Lily 消息去重**：同一件事不允许发超过 2 条消息；确认/通知合并为一条
- [ ] **Issue-first 协议**：任何状态变更必须先 `gh issue comment`，再发 Discord（Codex 的建议 — 再次强调）

## 📈 Velocity
- 今日: 2/2 (Sprint 13: #226 + #131)，另 Sprint 12 收尾 2/2 (#232 + #233)
- 趋势：连续 5 个 Sprint velocity 100%，但流程质量项（心跳、SHA、频道规范）持续拖尾

---

# Sprint Retrospective 2026-02-27

## ✅ Went Well
- **Sprint 5 velocity 100% (6/6)**：#148, #221, #222, #223, #227, #228 全部关闭，完美收尾
- **Gemini 独扛 PR #231**：rebase + mojibake 清理 + CVT 单测 12/12 全绿，单 coder 完整交付
- **PR #230 cross-review 完成**：Gemini 完成 `sm_fetch_pipeline.v` 双发射审查，PASS ✅ + SHA 绑定
- **Master CI 全天绿**：Nightly Regression + 2x CI runs on master 全部 success
- **Network anomaly 主动上报**：ist-mac-01/02 unreachable from ist-mac-s 被检测并升级给 Jerry
- **Codex 回来后给出实质性 retro 建议**：超时升级机制方向明确

## ❌ Didn't Go Well
- **Codex 10h+ 失联**：PR #230 和 #231 review 队列全天被阻塞，Lily 未在 30min 内升级（直到 Jerry 主动问才处理）
- **Sprint Planning 断档**：Sprint 4 结束到 Sprint 5 milestone 创建之间出现 gap，多条 "No active Sprint milestone" 告警，cron 报错循环
- **Sprint 2026-02-27 milestone（M#12）空创建**：0 item，Planning 未正确填充 backlog
- **scrum-health cron 错误**：上次运行 16h 前，error 状态，未修复
- **每日经济政治要闻 cron 错误**：持续 error 状态，未排查
- **ist-mac-s 代理异常**：curl/gh 从 ist-mac-s 出站走 Clash 7897 端口被拒，影响 GitHub CLI 操作
- **前两次 Retro action items 执行率低**：长任务心跳、同步消息模板、Post-merge SOP 连续 2 Sprint 未落地
- **PR #231 feature branch CI 仍 fail**：issue-131/gemini branch 未清理或修复

## 💬 Coder Feedback
- **CoderCodex**：好 — Lily 及时识别并公开同步了 Codex 失联风险；需改进 — 加"等待依赖超 30 分钟自动升级并切换 owner"机制，避免任务再卡 10h+
- **CoderGemini**：无回复
- **CoderClaude**：已移除（2026-02-27），不再可用

## 🔧 Action Items (下次 Planning 必须参考)
- [ ] **Codex 超时升级 SOP（30min 规则）**：coder 无响应 30min → Lily 自动重新分配，不等 10h（基于 Codex 建议）
- [ ] **Sprint Review → Planning 零延迟**：Sprint Review 触发后立即运行 Planning cron，不允许 milestone gap
- [ ] **scrum-health cron 修复**：检查错误原因，恢复正常运行
- [ ] **每日经济要闻 cron 修复**：检查 error，修复或禁用
- [ ] **ist-mac-s 代理修复**：为 GitHub CLI 配置显式 proxy（`HTTPS_PROXY=http://127.0.0.1:7897`）或排查 TUN 出站路由
- [ ] **长任务心跳（第 3 次提出 — 强制执行或移除）**：本次 Sprint 必须上线或永久关闭此 item
- [ ] **Post-merge SOP（Checklist 条目）**：追加到 pr-submission.md — 合并后必须跑 `make regression` + `make lint` 并 comment 结果
- [ ] **stale branch 清理**：关闭 issue-131/gemini CI fail 的 stale branch，避免持续报警
- [ ] **Jerry escalation 阈值**：Lily 必须在 agent 下线 30min（不是 10h+）时主动告警
- [ ] **#148 研究时间盒**：研究类 issue 最多 2h，超时必须产出草稿文档提交 review，不允许无限期研究

## 📈 Velocity
- Sprint 5：6/6 (100%) 🎉 — #148 #221 #222 #223 #227 #228
- Sprint 2026-02-27（M#12）：0/0（空 milestone，Planning 未执行）
- 趋势：连续 3 Sprint velocity 100%，但 Sprint Planning 质量下降（空 milestone、无 item 分配）

---

# Sprint Retrospective 2026-02-25

## ✅ Went Well
- **Velocity 100%（3/3）**：#149（WB precision replay）、#152（regfile banking）、#156（perf dashboard）全部关闭，Sprint 2026-02-24 满分完成
- **Retro 参与率 100%（连续第 2 次）**：3 位 coder 全部在 30 秒内回复（2/23: 33% → 2/24: 100% → 2/25: 100%）
- **PR 交付质量稳定**：CoderClaude + CoderCodex 一致反映 PR #219/#220 高质量、lint + 回归基线全绿
- **Dual-issue PC 步进 + L1 Cache 关键回归**：CoderGemini 彻底解决，不再反复
- **Retro-gather.sh 自动化运转**：cron 脚本无需手动触发，数据采集正常

## ❌ Didn't Go Well
- **CI 7 failures（out of 15 runs）**：Sprint Review 时 CI 全绿，但后续出现 7 次 failure——master 有不稳定提交或 runner 环境问题未被及时捕获
- **#148 研究周期偏长**：CoderClaude + CoderCodex 独立反映研究阶段拖延，未前置里程碑或中间评审节点
- **Post-merge 逻辑冲突验证滞后**：CoderGemini 指出 master 合并后冲突验证应更前置，合并后才发现问题
- **Feb 24 Action Items 落地率不足**：「长任务心跳机制」仍未实际落地；「All models failed 排查」无进展记录
- **同步消息不规范**：CoderCodex 指出 standup 消息未统一带 issue# + owner + ETA

## 💬 Coder Feedback
- **CoderClaude**：好 — PR 交付节奏稳定，lint + 回归始终绿；改进 — #148 研究周期偏长，需更早产出可评审方案文档
- **CoderCodex**：好 — PR #219/#220 高质量，CI 全绿；改进 — #148 研究应前置里程碑；同步消息统一带 `#issue` + owner + ETA
- **CoderGemini**：好 — 彻底解决 Dual-issue PC 步进和 L1 Cache 关键回归；改进 — Master 合并后逻辑冲突验证应更前置

## 🔧 Action Items (下次 Planning 必须参考)
- [ ] **#148 研究时间盒**：研究类 issue 最多 2h，超时必须产出草稿文档提交 review，不允许无限期研究
- [ ] **Post-merge 冲突验证 SOP**：每次 PR merge 后，coder 必须跑 `make regression` + `make lint` 并在 issue comment 贴结果
- [ ] **CI 失败根因排查**：@CoderClaude 分析 7 次 failure 来源（master 不稳定 or runner 环境）
- [ ] **长任务心跳 — 强制执行**：长任务（>3min）每 60s 发状态到 #ralphgpu-dev，违规 3 次 → kill + 重新分配（升级为强制规则）
- [ ] **同步消息模板**：格式统一为 `#<issue> [owner] 状态 | ETA: XX | Blocker: XX`

## 📈 Velocity
- Sprint 2026-02-24：3/3 (100%) — #149 #152 #156
- 趋势：连续 2 个 Sprint 100% velocity，Retro 参与率 33% → 100% → 100%，节奏稳定
- CI 健康警告：15 runs 中 7 failures，较 Feb 24 白天有退步，需排查

---

# Sprint Retrospective 2026-02-24

## ✅ Went Well
- **Velocity 100%（3/3）**：#149（WB precision replay）、#152（regfile banking）、#156（perf dashboard）全部关闭，满分完成
- **CI 全绿**：今日 15 次 CI runs（master + feature branches）全部 success，0 failures，Nightly Regression 也通过
- **Retro 参与率 100%**：3 位 coder 全部在 ~15 秒内回复（对比 2/23 只有 CoderClaude 回复，显著改进）
- **Cron-optimization review 质量高**：CoderClaude 给出 4 项详细标注（状态文件、/tmp 竞态、错误分级、scrum-health 覆盖率 91%）
- **SM 自审快速执行**：发现 #149/#156 assignee 缺失 + Sprint 3 milestone 未关闭，3 项在 1 分钟内全部修正

## ❌ Didn't Go Well
- **"All models failed" 出现多次**：discord_dev 中 4 次 Lily agent 回复前 CLI 崩溃（claude-opus/sonnet/haiku 全部 fail），说明底层 Claude CLI 有不稳定问题，需排查
- **长任务超时透明度不足**：所有 3 位 coder 一致反映长任务执行状态不可见，没有中间心跳，timeout 时外部无法感知
- **CoderClaude 中途汇报**：自评长任务仍需拆更小步骤，2/23 action item 未完全执行
- **Sprint 3 milestone 未及时关闭**：5/5 items 完成后 milestone 仍显示 open，需 SM 提醒或自动化

## 💬 Coder Feedback
- **CoderClaude**：做得好 — cron-optimization 全部交付；需改进 — 长任务超时问题，下次拆更小步骤避免 timeout
- **CoderCodex**：做得好：任务指派到验证闭环很快；需要改进：agent 超时时缺少中间进度可见性，后续统一加 60s 心跳与超时自动接管
- **CoderGemini**：做得好：高效完成 Sprint 3 并快速收敛 cron-optimization；改进：需加强长任务超时监控，确保 Agent 执行状态实时可见

## 🔧 Action Items
- [x] **Retro 参与率**：已连续 100%（✅ 已改进）
- [ ] **长任务心跳机制**：→ 升级到 Feb 25 action items
- [ ] **"All models failed" 排查**：记录时间戳，检查 CLI daemon 稳定性
- [x] **Sprint milestone 自动关闭**：Velocity 100% 时立即执行关闭（✅ 已执行）
- [x] **Cron-optimization 实现**：retro-gather.sh 已运行（✅ 完成）

## 📈 Velocity
- 3/3 (100%) — #149 #152 #156
- Sprint 4 Day 1 满分，CI 全绿，节奏最佳
- Retro 参与率从 33% → 100%（显著改善）
