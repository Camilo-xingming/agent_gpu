# docs/ — 设计文档、分析报告、Sprint 回顾

> 持久化的技术分析和流程记录。Sprint 回顾由 Lily cron 自动追加。

## 文件清单

| 文件 | 用途 | 写者 | 读者 | 何时读 | 关键词 |
|------|------|------|------|--------|--------|
| RETRO.md | Sprint 回顾记录 | Lily (retro cron) / SM Coach | Lily (Sprint Planning) | 每日 Planning 09:00 | retro, sprint, lessons |
| TESTING.md | 测试策略和方法 | Coders | Coders | 写新测试前 | test, strategy |
| PTX_COMPATIBILITY.md | PTX 指令兼容性矩阵 | Coders | Coders | 新增指令时 | ptx, compatibility |
| WARP_INST_VALID_D1_STATUS.md | Warp 指令有效性分析 | Coders | Coders | 调试 pipeline 时 | warp, pipeline, debug |
| atomic-contention-analysis.md | Atomic 竞争分析 | Coders | Coders | atomic 相关 issue | atomic, contention |
| memory-optimization-reference.md | 内存优化参考 | Coders | Coders | 内存子系统优化时 | memory, optimization |
| performance-verification-plan.md | 性能验证计划 | Coders | Coders/Jerry | 性能相关 sprint | performance, verification |

## architecture/ 子目录

| 文件 | 用途 | 写者 | 读者 | 关键词 |
|------|------|------|------|--------|
| dual-issue.md | 双发射架构设计 | Coders | Coders | dual-issue, scheduler |
| waw-analysis.md | WAW 冒险分析 | Coders | Coders | waw, hazard, scoreboard |
