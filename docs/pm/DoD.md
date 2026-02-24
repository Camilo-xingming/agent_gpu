# Definition of Done (DoD)

> PR 和 Sprint 的完成标准。不满足 DoD 的工作不算"完成"。

## PR 级 DoD

一个 PR 满足以下所有条件才可合并：

| 条件 | 验证方式 |
|------|----------|
| `make test` 全部通过 | CI 自动检查 |
| `make lint` 零 warnings | CI 自动检查 |
| 针对本次变更有对应的测试用例 | Reviewer (Lily) 人工确认 |
| PR description 包含背景、变更内容、测试方法 | Reviewer (Lily) 人工确认 |
| Coder 在 #ralphgpu-dev 确认任务完成 | Lily 确认收到 |
| 无 Synthesis errors（RTL 变更时） | CI 自动检查 |
| 不引入新的 timing violations（如适用） | CI 或手动验证 |

### PR Description 必须包含

1. **背景**：这个 PR 解决什么问题？对应哪个 Issue？
2. **变更内容**：改了什么，为什么这样改
3. **测试方法**：跑了哪些测试，关键结果截取
4. **已知限制**（如有）：本次未覆盖的场景

## Sprint 级 DoD

一个 Sprint 满足以下条件视为完成：

| 条件 | 验证方式 |
|------|----------|
| Milestone 内所有 Issue 已关闭 | GitHub Milestone 页面 |
| 每个 Issue 对应的 PR 已合并到 master | GitHub PR 状态 |
| master 分支 CI 全绿 | GitHub Actions |
| Sprint Review 已执行（Jerry 已收到 demo） | Discord #ralphgpu 记录 |
| Sprint Retrospective 已执行 | `docs/RETRO.md` 有当日记录 |
| Carry-over items 已明确处理 | 未完成 Issue 标注新 Milestone 或关闭 |

## 例外处理

- **Hotfix**：紧急修复可跳过"针对性测试"要求，但必须在下个 Sprint 补上测试
- **Spike/Research**：研究类 Issue 不要求 PR，但必须产出 `research-NNN.md` 文档
- **Carry-over**：Sprint 未完成的 Issue 在 Review 时必须明确状态 — 移入下一 Sprint / scope down / 关闭

---

*更新时间: 2026-02-24*
