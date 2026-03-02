# Sprint Retrospective — Sprint 22

## Sprint Goal
IPC performance optimization — L1 cache enable + stall analysis + memory coalescing

## Velocity: 3/3 (100%)
- #289 L1D Cache enable — CLOSED
- #290 Stall-driven IPC optimization — CLOSED
- #291 Memory Coalescing Unit implementation — CLOSED (PR #297)

## Went Well
- IPC boost: tensor_multiwarp fetch stall 90.5% to 4.2%, IPC 0.026 to 0.250 (#290 CoderCodex)
- L1D + MCU integration smooth: 3 issues all merged cleanly, CI 14/14 stable
- Cross-review quality: CoderGemini caught L1D response floating + Store deadlock in PR #297, fixed by CoderCodex before merge
- L2 Cache metadata bug fix: CoderGemini identified way_reg sampling issue in full-line write bypass

## Did Not Go Well
- #291 interface spec insufficient: MCU-SM-L1D handshake issues found late in review, not in design doc (CoderCodex)
- vector_add baseline failure not isolated early: pre-existing failure discovered at Sprint end (CoderCodex)
- L2 fix initial way_reg sampling miss: FSM state transition missed correct sampling, required retry (CoderGemini)

## Action Items (Sprint 24)
- Interface closure check: verify all new module request/response paths before PR submission
- Baseline smoke test upfront: run vector_add + cvt_unit at Sprint start, post results to Issue
- FSM trace observability: establish complete trace points for complex FSM transitions
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
- [x] **Codex 超时升级 SOP（30min 规则）**：已文档化到 `docs/pm/pr-submission.md`（Issue #268），coder 无响应 30min → Lily 自动重新分配
- [ ] **Sprint Review → Planning 零延迟**：Sprint Review 触发后立即运行 Planning cron，不允许 milestone gap
- [ ] **scrum-health cron 修复**：检查错误原因，恢复正常运行
- [ ] **每日经济要闻 cron 修复**：检查 error，修复或禁用
- [ ] **ist-mac-s 代理修复**：为 GitHub CLI 配置显式 proxy（`HTTPS_PROXY=http://127.0.0.1:7897`）或排查 TUN 出站路由
- [ ] **长任务心跳（第 3 次提出 — 强制执行或移除）**：本次 Sprint 必须上线或永久关闭此 item
- [x] **Post-merge SOP（Checklist 条目）**：已落地 `docs/pm/pr-submission.md`（Issue #268），合并后必须跑 `make regression` + `make lint` 并 comment 结果
- [ ] **stale branch 清理**：关闭 issue-131/gemini CI fail 的 stale branch，避免持续报警
- [x] **Jerry escalation 阈值**：已纳入 `docs/pm/pr-submission.md`（Issue #268），agent 下线 30min 时主动升级并重新分配
- [x] **#148 研究时间盒**：已文档化到 `docs/pm/pr-submission.md`（Issue #268），研究类 issue 最多 2h，超时必须产出草稿文档提交 review

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
- [x] **#148 研究时间盒**：已文档化到 `docs/pm/pr-submission.md`（Issue #268），研究类 issue 最多 2h，超时必须产出草稿文档提交 review
- [x] **Post-merge 冲突验证 SOP**：已文档化到 `docs/pm/pr-submission.md`（Issue #268），每次 PR merge 后必须跑 `make regression` + `make lint` 并在 issue comment 贴结果
- [ ] **CI 失败根因排查**：@CoderClaude 分析 7 次 failure 来源（master 不稳定 or runner 环境）
- [ ] **长任务心跳 — 强制执行**：长任务（>3min）每 60s 发状态到 #ralphgpu-dev，违规 3 次 → kill + 重新分配（升级为强制规则）
- [x] **同步消息模板**：格式统一为 `#<issue> [owner] 状态 | Blocker: XX`（Issue #268）

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

---

# Sprint Retrospective Sprint 22 (2026-03-01)

## ✅ Went Well
- **Velocity 100% (3/3)**：#289 L1D Cache enable + #290 Stall-driven IPC + #291 Memory Coalescing 全部關閉
- **IPC 顯著提升**：tensor_multiwarp fetch stall 90.5% → 4.2%，IPC 0.026 → 0.49（CoderCodex 數據）
- **CI 全绿 (14/14)**：Sprint 期間 master + 所有 feature branch 全部 success
- **Memory Coalescing 核心 Bug 修復**：L2 Cache 全行寫繞過邏輯元數據更新 Bug 被識別並修復（CoderGemini）
- **Cross-agent review 有效**：PR #297 經 CoderGemini review，SHA 驗證正確
- **Sprint 24 無縫銜接**：Planning 自動觸發，3 issues 分配，無 milestone gap

## ❌ Didn't Go Well
- **Sprint 21/22 milestone 未及時關閉**：兩個已完成 Sprint milestone 仍留 open 狀態（0 open issues），ceremony 需手動清理
- **Memory Coalescing (#291) 複雜度超預期**：出現多輪調試（branch issue-291/gemini 獨立追蹤），主線 PR 時間延長
- **CoderGemini 過程性消息仍有洩漏**：retro 請求觸發後重複發送類似消息（"completed cron job" 系統消息出現在頻道）
- **長任務心跳（第 5 Sprint 仍未落地）**：再次被 CoderCodex 標記為 action item

## 💬 Coder Feedback
- **CoderCodex**：好 — #289/#290 收斂快一次合入，IPC 大幅提升；需改進 — Memory Coalescing 多輪調試耗時，應設 2h time-box 後出草稿 PR
- **CoderGemini**：好 — L2 Cache Bug 定位準確，cross-review 覆蓋率高；需改進 — 長任務心跳機制必須本 Sprint 落地

## 🔧 Action Items
- [ ] **長任務心跳（第 5 次 — 終局）**：Sprint 24 必須上線或永久關閉。CoderCodex 實現，Lily 監督
- [ ] **milestone 自動清理**：ceremony 腳本加邏輯：open milestone + 0 open issues → 自動 close，不留殘留
- [ ] **#291 類複雜 issue time-box**：估時超 2h 的 issue 必須先出草稿 PR（即使不完整），避免無限期調試
- [ ] **CoderGemini 系統消息淨化**：cron job 完成消息不應發到 Discord 主頻道，改為靜默或 -logs

## 📈 Velocity
- Sprint 22：3/3 (100%) — #289 #290 #291
- CI：14/14 success 🟢
- 趨勢：連續 Sprint velocity 100%，但流程質量項（心跳、milestone 清理、消息規範）持續拖尾

---

# Sprint Retrospective Sprint 25 (2026-03-02)

## ✅ Went Well
- **Velocity 100% (3/3)**：#309 CoderGemini raw output 净化 + #313 长任务心跳终局落地 + #314 milestone 自动清理 全部关闭
- **长任务心跳终于落地**：连续 5 Sprint 的 action item 本次真正完成，`make test_cron_optimization` 防回退测试通过
- **Milestone 自动清理上线**：0 open issues → auto close，消除历史遗留 milestone 积压
- **CoderCodex 交付质量高**：#313/#314 一次收敛并合并，带防回退测试
- **CI 全绿**：master 连续多次 CI success，Nightly Regression 通过
- **Cross-review 有效**：CoderGemini 完成所有 Sprint 25 交叉审查任务

## ❌ Didn't Go Well
- **`Stats:` 泄漏仍未完全解决**：CoderGemini 输出中仍有系统状态字符串泄漏到 Discord 消息，需脚本拦截
- **PR #317 跨 Sprint 遗留**：#309 issue 已关闭但 PR #317 仍 open，需 cross-review 后合并清理
- **多个 ceremony cron 并发**：本 Sprint 结束时出现两个 ceremony 实例同时运行，重复发消息

## 💬 Coder Feedback
- **CoderCodex**：做得好 — #313/#314 一次收敛并合并，长任务心跳与 milestone auto-clean 都补了防回退测试；需改进 — 需确保 stale branch 及时清理
- **CoderGemini**：做得好 — 高效完成了 Sprint 25 的所有交叉审查任务；需改进 — 必须立即通过脚本拦截彻底解决消息中的 `Stats:` 泄漏问题

## 🔧 Action Items (Sprint 26 Planning 必须参考)
- [ ] **CoderGemini `Stats:` 泄漏根治**：在 gateway 层或 cron prompt 层增加输出过滤，彻底阻止系统状态字符串进入 Discord 消息
- [ ] **PR #317 cross-review + merge**：CoderCodex review，通过后 Lily merge
- [ ] **stale branch 清理**：删除 issue-131/gemini 等 CI 持续 fail 的旧分支
- [ ] **SHA discipline 检查**：Lily merge 前必须验证 reviewer SHA = PR head SHA

## 📈 Velocity
- Sprint 25：3/3 (100%) — #309 #313 #314
- CI：全绿 🟢
- 趋势：连续多 Sprint velocity 100%，流程自动化质量持续提升

# Sprint Retrospective Sprint 27 (2026-03-02)

## ✅ Went Well
- **#323 bench_atomics RAW stall 修复快速收敛**：在高噪声调试条件下定位 load-use hazard 根因并完成修复闭环（CoderCodex）
- **CI 全绿**：Sprint 期间 master 连续 CI success，包含 issue-314/codex 等 feature branch
- **CoderCodex 执行节奏稳定**：#319/#320 分支/issue 状态同步及时（CoderCodex 自评）

## ❌ Didn't Go Well
- **Velocity 33% (1/3)** — Sprint 27 最低完成率之一。#324/#325 整个 Sprint 无 branch/PR 启动
- **#324/#325 未及时启动**：#323 收尾时 #324/#325 仍无最小 WIP（无 branch、无 issue comment），导致任务完全积压到 sprint 结束
- **Sprint overdue 4h**：milestone 到期 4h 后仍无关闭，watchdog 触发告警
- **CoderGemini 分配混乱**：#325 在 sprint 内有 CoderGemini 工作记录但未正式 assign，导致 watchdog 报 no_assignee

## 💬 Coder Feedback
- **CoderCodex**：好 — #323 根因快速收敛，PR 闭环干净；改进 — #324/#325 应在 #323 收尾前完成最小启动（branch + WIP comment）；PR review 响应需加快（#317 有延迟）
- **CoderGemini**：已在 Sprint 内开始 #325 分析（12:22 汇报），但无正式 branch/commit 记录

## 🔧 Action Items (Sprint 28 Planning 必须参考)
- [ ] **WIP 启动纪律**：Sprint 开始 2h 内每个 issue 必须有 branch + WIP comment，否则 watchdog 告警
- [ ] **并行启动**：持有多个 issues 的 coder 必须并行建 branch，不能等第一个完成再动第二个
- [ ] **carry-over P0**：#324 #325 作为 carry-over 在 Sprint 28 优先分配，标记 P0
- [ ] **CoderGemini 任务归属明确化**：assign 前必须在 GitHub Issue 上 comment 认领，不允许"隐形工作"
- [ ] **PR review 响应时限**：被 review 的 PR 必须在 4h 内给出 PASS/FAIL，否则 Lily 重新分配 reviewer

## 📈 Velocity
- Sprint 27：1/3 (33%) — #323 ✅，#324 #325 carry-over
- CI：全绿 🟢
- 趋势：Sprint 27 velocity 骤降，主因是 WIP 启动延迟，非技术困难

---

# Sprint Retrospective Sprint 27 (2026-03-02)

## ✅ Went Well
- **#323 bench_atomics RAW stall 修复 完成**：CoderCodex 在高噪声调试条件下快速收敛到可验证根因，完成修复闭环
- **CI 全绿**：Sprint 期间 master 连续多次 CI success
- **CoderGemini 服务协议审查**：完成法律与商业风险点补充审查

## ❌ Didn't Go Well
- **Velocity 1/3 (33%)**：#324/#325 Sprint 全程零进展，没有创建任何 branch
- **Sprint 无团队讨论直接创建**：Jerry 指出 Planning 应该先 @coders 讨论确认 scope，再创建 milestone。不应依赖静态 AGENTS.md，应靠每次 cron job 注入 prompt 来驱动流程
- **#324/#325 最小启动缺失**：CoderCodex 指出应在 #323 收尾前完成最小启动（branch + WIP comment），实际未执行
- **CoderGemini 本地环境网络问题**：影响 #325 实际进展

## 💬 Coder Feedback
- **CoderCodex**：好 — #323 快速收敛；需改进 — #324/#325 应在 #323 收尾前完成最小启动
- **CoderGemini**：好 — 服务协议审查；需改进 — 本地网络问题导致 #325 卡住，需提前上报

## 🔧 Action Items (Sprint 28 Planning 必须参考)
- [ ] **Planning 先讨论再创建**：cron ceremony prompt 中加入：创建 milestone 前必须先 @coders 在 dev_channel 确认 scope + 能否交付，无异议才建
- [ ] **WIP 最小启动**：Sprint 启动当天每个 assigned issue 必须建好 branch + WIP comment，即使未开始编码
- [ ] **#324 carry-over**：Warp scheduler active warp occupancy tracking + per-cycle IPC metric
- [ ] **#325 carry-over**：Cache hit rate profiling — L1D/L2 miss rate baseline + tuning（CoderGemini 环境问题需先确认）

## 📈 Velocity
- Sprint 27：1/3 (33%) — #323 ✅，#324 #325 carry-over → Sprint 28
- CI：全绿 🟢
- 趋势：velocity 下滑，Sprint Planning 质量需提升

