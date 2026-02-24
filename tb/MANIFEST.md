# tb/ — Testbench 验证文件

> ~121 个 Verilog testbench。命名规则：`tb_{module_name}.v` 或 `tb_{feature}_{scenario}.v`。

## 命名规约

| 前缀模式 | 测试范围 | 示例 |
|----------|---------|------|
| `tb_{module}.v` | 单模块单元测试 | `tb_alu.v`, `tb_atomic_unit.v` |
| `tb_{module}_{scenario}.v` | 特定场景测试 | `tb_alu_dual_issue_conflict.v` |
| `tb_bench_{workload}.v` | 性能 benchmark | `tb_bench_atomics.v`, `tb_bench_divergence.v` |
| `tb_sm_v2_*.v` | SM v2 集成测试 | `tb_sm_v2_perf_tensor_multiwarp.v` |
| `tb_b300_*.v` | Blackwell 架构测试 | `tb_b300_features.v` |

## 写者与维护

- **写者**：Coders（CoderClaude / CoderCodex / CoderGemini）
- **读者**：Coders（开发时参考）、CI（自动运行）
- **何时用**：`make test` 全量运行；`make test_xxx` 单独运行
- **新增规则**：每个 PR 必须包含针对性 testbench，CI 自动验证

## 注意事项

- 文件数量大（121+），不逐一列出。用 `ls tb/tb_*{关键词}*.v` 按关键词查找
- testbench 内部注释应说明测试目标和预期行为
