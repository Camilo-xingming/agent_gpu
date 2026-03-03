# Velocity 追踪

> 每 Sprint 的计划/完成/遗留统计。Sprint Review cron 自动追加新行。

## Velocity 表

| Sprint | 日期 | Goal | 计划 | 完成 | 遗留 | Velocity | 备注 |
|--------|------|------|------|------|------|----------|------|
| Sprint 3 | 2026-02-24 | WB precision + regfile banking + perf dashboard | 3 | 3 | 0 | 100% | #149 #152 #156 全部关闭，CI 全绿 |
| Sprint 4 | — | — | — | — | — | — | （待 Planning 填入） |
| Sprint 22 | 2026-03-01 | IPC 性能优化 — L1 cache 启用 + stall 分析 + memory coalescing | 3 | 3 | 0 | 100% | #289 #290 #291 全部关闭，CI 全绿，PR #292 #295 #297 merged |

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
| Sprint 2026-02-25 | 2026-02-24 | 扩展支持 + 自动化驱动 + 知识积累 (Multi-SM p... | 3 | 3 | 0 | 100% | auto-generated |
| Sprint 2026-02-26 | 2026-02-24 | 工具链稳定性 — 完成 cron-optimization 脚本实... | 3 | 3 | 0 | 100% | auto-generated |
| Sprint 2026-02-27 | 2026-02-24 | GPU software stack foundation — pipeline replay + FPGA ... | 4 | 4 | 0 | 100% | auto-generated |
| Sprint 5 | 2026-02-27 | GPU core feature expansion — dual-issue RTL + L1D activ... | 6 | 6 | 0 | 100% | auto-generated |
| Sprint 2026-02-28 | 2026-02-27 | Merge in-flight PRs (#230 dual-port icache, #231 CVT fix)... | 2 | 2 | 0 | 100% | auto-generated |
| Sprint 14 | 2026-02-28 | Tensor Core e2e + Texture/Surface verification | 2 | 2 | 0 | 100% | auto-generated |
| Sprint 15 | 2026-02-28 | 终结 Retro 积压：心跳机制终局 + 经济要闻 c... | 2 | 0 | 2 | 0% | auto-generated |
| Sprint 16 | 2026-02-28 | 恢复 master CI（P0）+ 清理 stale branches。Items: ... | 2 | 2 | 0 | 100% | auto-generated |
| Sprint 17 | 2026-02-28 | 基础设施可靠性 — ist-mac-s proxy 修复 + CI bil... | 2 | 2 | 0 | 100% | auto-generated |
| Sprint 24 | 2026-03-02 | Memory subsystem e2e validation — MCU/L1D perf + vector_add fix + memory benchmark | 3 | 3 | 0 | 100% | auto-generated |
| Sprint 25 | 2026-03-02 | 流程质量收尾 + 系统稳定性 — CoderGemini raw output 净化 + milestone 自动清理 + 长任务心跳终局落地 | 3 | 3 | 0 | 100% | auto-generated |
| Sprint 26 | 2026-03-02 | 净化 + 清理 — CoderGemini 输出过滤根治 + stale branch 清理 + PR #317 收尾 | 4 | 4 | 0 | 100% | auto-generated |
| Sprint 27 | 2026-03-02 | GPU 可观测性 + 稳定性 — bench_atomics RAW stall 修复 + warp occupancy + L1D/L2 cache baseline | 1 | 3 | 2 | 33% | carry-over: #324 #325 |
| Sprint 28 | 2026-03-02 | 可观测性完成 — Warp 占用率 tracking (#324) + L1D/L2 cache 命中率基线 (#325) | 2 | 2 | 0 | 100% | auto-generated |
| Sprint 29 | 2026-03-02 | Core Efficiency & Stall Debugging — FU utilization fix + Tensor WB spike + bench_divergence guard | 3 | 3 | 0 | 100% | bonus: #333 memory_heavy fix |
| Sprint 30 | 2026-03-02 | GPU 性能修复 — Tensor WB 4096/4096 完整 fix + bench_divergence scoreboard deadlock + lint 清理 | 0 | 5 | 5 | 0% | carry-over: #337 #338 #339 #340 #341 |
| Sprint 31 | 2026-03-02 | Tensor WB 4096/4096 完整修復 + bench_divergence deadlock fix + lint 收尾 | 6 | 6 | 0 | 100% | #337 #338 #339 #340 #341 #343 全部关闭，6/6 完成 |
