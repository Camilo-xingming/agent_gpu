# PR 提交 Checklist

> Coder 提交 PR 前必须逐项确认。来源：DoD + 历次 PR 踩坑。

## 提交前自检

- [ ] `make test` 本地通过
- [ ] `make lint` 零 warnings
- [ ] 本次变更有对应的测试用例（新功能 = 新测试，bug fix = 回归测试）
- [ ] RTL 变更：无 synthesis errors
- [ ] PR description 包含：背景、变更内容、测试方法、已知限制
- [ ] PR title 关联 Issue（如 `fix #150: xxx` 或 `feat #160: xxx`）
- [ ] branch 已 push 到 remote（不要本地改完不 push 就说"完成"）
- [ ] 不包含 debug 临时代码（`$display` 调试输出、hardcoded 测试值）
- [ ] MANIFEST.md 已更新（如有新增/删除/移动文件）

## 提交后

- [ ] 在 #ralphgpu-dev 通知 Lily：`@Lily #ISSUE 完成，PR #XX 已提交`
- [ ] 更新 agent-context.json：phase → "pr_submitted"

## 常见遗漏（从 RETRO 总结）

| 遗漏 | 后果 | 预防 |
|------|------|------|
| branch 未 push | Lily 无法 review | 提 PR 前 `git push` |
| 缺 PR description | Review 效率低 | 用模板填写 |
| 忘记关联 Issue | Sprint tracking 断裂 | title 带 `#NUM` |
| MANIFEST 未更新 | CI manifest-check 失败 | 新增文件必更新 |

---

*更新时间: 2026-02-25*

## merge 后（Lily 执行）

- [ ] merge 后立即检查 master CI 状态：`gh run list --branch master --limit 3`
- [ ] 若 master CI 失败，立即触发修复（不等下次 standup）

## post-merge 验证（Coder 执行）

- [ ] PR merge 后，coder 在 issue comment 贴 `make regression` + `make lint` 结果
- [ ] 结果格式：`Post-merge verify: regression ✅ / lint ✅ | SHA: <head_sha>`
- [ ] 若任一失败，立即开新 PR 修复，不关闭 issue
