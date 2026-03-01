# Definition of Done (DoD)

> PR 和 Sprint 的完成标准。不满足 DoD 的工作不算"完成"。

## PR 级 DoD

一个 PR 满足以下所有条件才可合并：

| # | 条件 | 验证方式 |
|---|------|----------|
| 1 | `make test` 全部通过 | CI 自动检查 |
| 2 | `make lint` 零 warnings | CI 自动检查 |
| 3 | 针对本次变更有对应的测试用例 | Reviewer 确认 |
| 4 | 无 Synthesis errors（RTL 变更时） | CI 自动检查 |
| 5 | 不引入新的 timing violations（如适用） | CI 或手动验证 |
| 6 | PR title 关联 Issue（如 `fix #150: xxx`） | watchdog 检查 |
| 7 | PR description 包含背景、变更内容、测试方法 | Reviewer 确认 |
| 8 | 不包含 debug 临时代码（`$display` 调试输出、hardcoded 测试值） | Reviewer 确认 |
| 9 | MANIFEST.md 已更新（如有新增/删除/移动文件） | CI 或 Reviewer 确认 |
| 10 | branch 已 push 到 remote | watchdog 检查（branch 存在） |
| 11 | 至少 1 个不同 coder 的 review（cross-review） | watchdog 检查（reviews API） |

### PR Description 必须包含

1. **背景**：这个 PR 解决什么问题？对应哪个 Issue？
2. **变更内容**：改了什么，为什么这样改
3. **测试方法**：跑了哪些测试，关键结果截取
4. **已知限制**（如有）：本次未覆盖的场景

### Merge 后验证

- merge 后检查 master CI 状态，失败则立即修复
- coder 在 issue comment 贴 `make regression` + `make lint` 结果

## Sprint 级 DoD

一个 Sprint 满足以下条件视为完成：

| # | 条件 | 验证方式 |
|---|------|----------|
| 1 | Milestone 内所有 Issue 已关闭 | GitHub Milestone |
| 2 | 每个 Issue 对应的 PR 已合并到 master | GitHub PR 状态 |
| 3 | master 分支 CI 全绿 | GitHub Actions |
| 4 | Sprint Review 已执行 | watchdog 检查（milestone closed） |
| 5 | Sprint Retrospective 已执行 | watchdog 检查（RETRO.md 含 Sprint title） |
| 6 | Carry-over items 已明确处理 | 未完成 Issue 移入新 Milestone 或关闭 |

## 常见遗漏（从 RETRO 总结）

| 遗漏 | 后果 | 预防 |
|------|------|------|
| branch 未 push | 无法 review | 提 PR 前 `git push` |
| 缺 PR description | Review 效率低 | 用模板填写 |
| 忘记关联 Issue | Sprint tracking 断裂 | title 带 `#NUM` |
| MANIFEST 未更新 | CI manifest-check 失败 | 新增文件必更新 |

## 例外处理

- **Hotfix**：紧急修复可跳过"针对性测试"要求，但必须在下个 Sprint 补上测试
- **Spike/Research**：研究类 Issue 不要求 PR，但必须产出 `research-NNN.md` 文档
- **Carry-over**：Sprint 未完成的 Issue 在 Review 时必须明确状态 — 移入下一 Sprint / scope down / 关闭

---

*更新时间: 2026-03-01*
