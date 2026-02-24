# Velocity 追踪

> 每 Sprint 的计划/完成/遗留统计。Sprint Review cron 自动追加新行。

## Velocity 表

| Sprint | 日期 | Goal | 计划 | 完成 | 遗留 | Velocity | 备注 |
|--------|------|------|------|------|------|----------|------|
| Sprint 3 | 2026-02-24 | WB precision + regfile banking + perf dashboard | 3 | 3 | 0 | 100% | #149 #152 #156 全部关闭，CI 全绿 |
| Sprint 4 | — | — | — | — | — | — | （待 Planning 填入） |

## 统计方式

- **计划**：Sprint Planning 时 Milestone 内的 Issue 数量
- **完成**：Sprint 结束时已关闭的 Issue 数量（PR 已合并 + CI 通过）
- **遗留**：未完成的 Issue 数量（carry-over 到下一 Sprint 或关闭）
- **Velocity**：完成 / 计划 * 100%

## 更新规则

1. **Sprint Planning 时**：Lily 填入 Sprint 编号、日期、Goal、计划数
2. **Sprint Review 时**：Sprint Review cron 填入完成数、遗留数、Velocity、备注
3. 遗留 Issue 在备注中标注处理方式：`carry->Sprint N+1` 或 `closed (descoped)`

## 趋势分析

> 当积累 5+ Sprint 数据后，在此处添加趋势总结（平均 velocity、稳定性、瓶颈模式）。

---

*更新时间: 2026-02-24*
