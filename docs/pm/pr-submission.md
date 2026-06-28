# PR 提交流程与 Post-merge SOP

> 目标：把 PR 提交、合并后验证、超时升级、研究时间盒和 standup 模板固化为可执行规则。

## 1. PR 提交前 Checklist（作者）

- [ ] Issue 已更新 WIP：`WIP: branch issue-<num>/...`
- [ ] 分支已 push 到远端，且 PR 目标分支正确（`master`）
- [ ] PR body 含 `Closes #<num>`
- [ ] `make test` 与 `make lint` 已执行并记录结果

## 2. Post-merge Checklist（必须）

PR 合并后，作者必须在 30 分钟内完成以下动作：

- [ ] 在主分支执行 `make regression`
- [ ] 在主分支执行 `make lint`
- [ ] 在对应 issue comment 回填结果（成功或失败都要回填）

Issue comment 模板：

```text
进展: post-merge check for #<issue>
- make regression: PASS/FAIL
- make lint: PASS/FAIL
- 说明: 如失败，附失败目标与下一步修复计划
```

## 3. Codex 30min 超时升级 SOP

触发条件：被分配 coder 在 30 分钟内没有有效进展更新。

- 0-15 分钟：保持执行，正常跟进
- 15-30 分钟：在 `#ralphgpu-dev` 点名催进展
- >=30 分钟：Lily 立即执行重新分配，并在 issue comment 记录 `Blocked/Reassigned`
- 若阻塞源自账号、网络、billing 等 owner 依赖：同步 @Jerry，避免继续空转

## 4. 研究类 Issue 2h 时间盒

研究类 issue 最长 2 小时，不允许无限期研究。

- 0-90 分钟：完成信息收敛与方案比较
- 90-120 分钟：输出可评审草稿（`docs/issue-<num>-*.md`）
- 到 120 分钟仍未闭环：必须提交草稿并请求 review，再决定继续研究或拆分子任务

## 5. Standup 消息模板（统一）

统一格式：

```text
#<issue> [owner] 状态 | Blocker: <none/具体阻塞>
```

示例：

```text
#268 [Tony-zf1] 文档已提交，等待 review | Blocker: none
#266 [CoderGemini] dispatch 拆分中，test 失败待修 | Blocker: ci runner unstable
```

## 6. Cross-review SLA（30min，Issue #561）

目标：把 cross-review 响应时间变成可机检指标，避免 review 请求长时间无人响应。

### Canonical comment 格式（必须）

在 PR comment 使用固定关键词：

- 请求方（开始计时）：
  - `**[Codex]** CROSS_REVIEW_REQUEST @reviewer 请在 30 分钟内给出 CROSS_REVIEW_PASS / CROSS_REVIEW_FAIL`
- 审查方（停止计时）：
  - `**[Gemini]** CROSS_REVIEW_PASS`
  - 或 `**[Gemini]** CROSS_REVIEW_FAIL`

### SLA 判定

- 起点：`CROSS_REVIEW_REQUEST` comment 的 `createdAt`
- 终点：首个后续 `CROSS_REVIEW_PASS/FAIL` comment 的 `createdAt`
- 通过标准：`终点 - 起点 <= 30 分钟`

### 自动检查与升级

- 机检脚本：`scripts/sprint-watchdog.sh --review-sla-minutes 30 --json`
- 违规证据：JSON 中 `cross_review_sla.violations[]`
- 超时处理：watchdog 在 `#ralphgpu-dev` 发送 1 行告警并提示 Lily 立即跟进/按需重新分配 reviewer

---

*更新时间: 2026-03-01*
