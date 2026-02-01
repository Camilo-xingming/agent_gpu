# RalphGPU Performance Verification Plan

**目标：** 验证 Codex 完成的 Track 1-2 优化的性能提升

**优化内容回顾：**
- Track 1: 原子操作（per-lane, 队列, 共享内存快速路径）
- Track 2: 分支发散（循环重聚, 栈溢出保护）

---

## 任务分工

### @gemini - 基准测试设计
负责设计微基准测试（microbenchmarks），验证各项优化的具体性能提升

### @claude - 综合性能分析
负责设计应用级测试和性能分析框架，评估整体影响

---

## Gemini 任务清单

**✅ 状态：已完成 (2026-01-31)**
- `tb/bench_atomics.ptx` - 113 instructions, 6 种测试模式
- `tb/bench_divergence.ptx` - 78 instructions, 3 种测试场景
- `tb/tb_bench_atomics.v` - 520 lines
- `tb/tb_bench_divergence.v` - 488 lines
- 编译验证：两个基准测试均成功编译

### 1. 原子操作基准测试
创建 `tb/bench_atomics.ptx` 和对应的测试台：

**测试场景：**
- **单 warp 原子性能**
  - 测试项：连续原子操作吞吐量
  - 对比：优化前（串行）vs 优化后（队列）
  - 指标：cycles per atomic operation

- **多 warp 原子竞争**
  - 测试项：多个 warp 同时执行原子操作
  - 对比：优化前（阻塞）vs 优化后（队列化）
  - 指标：总完成时间，warp 停滞周期

- **Per-lane 原子验证**
  - 测试项：所有 lane 各自执行原子操作
  - 验证：结果正确性 + 并行度
  - 指标：是否真的并行执行

- **共享内存原子快速路径**
  - 测试项：shared memory atomics vs global memory atomics
  - 对比：延迟差异
  - 指标：cycles per operation

### 2. 分支发散基准测试
创建 `tb/bench_divergence.ptx`：

**测试场景：**
- **简单循环（向后分支）**
  - 测试项：for (i=0; i<N; i++) {...}
  - 验证：重聚点是否正确（fall-through）
  - 指标：执行周期数

- **嵌套分支**
  - 测试项：if + for 嵌套
  - 验证：栈是否正确管理
  - 指标：栈深度使用，是否溢出

- **栈溢出边界测试**
  - 测试项：深度嵌套控制流
  - 验证：溢出保护是否生效
  - 指标：overflow/underflow 标志

### 3. 输出格式
为每个测试生成：
```
测试名称
├── 优化前: XXX cycles
├── 优化后: XXX cycles
├── 提升: XX%
└── 结论: [PASS/FAIL]
```

---

## Claude Code 任务清单

### 1. 性能分析框架
创建 `scripts/perf_analysis.py`：

**功能：**
- 解析仿真日志，提取性能计数器
- 绘制性能对比图表
- 生成 Markdown 报告

**需要提取的指标：**
- 总执行周期数
- Warp 停滞周期（atomic, branch, memory）
- IPC (Instructions Per Cycle)
- 原子操作吞吐量
- 分支预测准确率
- 内存访问延迟

### 2. 应用级测试
选择或创建真实应用场景：

**推荐测试：**
- **向量加法（带原子累加）**
  - 场景：reduction 操作
  - 验证：原子队列效果

- **矩阵乘法（带同步）**
  - 场景：barrier + shared memory atomics
  - 验证：共享内存原子快速路径

- **并行归约（parallel reduction）**
  - 场景：大量原子操作 + 分支
  - 验证：综合性能提升

### 3. 回归测试
确保优化没有破坏现有功能：

**测试集：**
- `tb/tb_ralph_gpu.v`（已有的测试）
- 所有 `test_*.ptx` 程序
- 验证输出正确性

### 4. 性能报告模板
创建 `RalphGPU/docs/PERFORMANCE_REPORT.md`：

```markdown
# RalphGPU Performance Report

## Executive Summary
- 原子操作性能提升: XX%
- 分支发散处理改进: XX%
- 综合性能提升: XX%

## Detailed Results
[表格 + 图表]

## Regression Analysis
[确认无功能退化]

## Recommendations
[进一步优化建议]
```

---

## 协作流程

1. **Gemini** 先完成微基准测试设计和实现
2. **Claude** 同时开发性能分析框架
3. Gemini 运行基准测试，生成原始数据
4. Claude 分析数据，生成报告
5. 双方 review，确认结论

---

## 成功标准

✅ **原子操作：**
- Per-lane 原子吞吐量 >10x 提升
- 多 warp 原子队列减少停滞 >50%
- 共享内存原子延迟 <5 cycles

✅ **分支发散：**
- 循环重聚逻辑正确（无错误执行）
- 栈溢出保护有效（深度嵌套不崩溃）
- 分支开销降低 >20%

✅ **整体：**
- 所有现有测试通过
- 无性能退化
- 综合性能提升 >30%

---

## 时间估算

- Gemini 微基准：1-2 小时
- Claude 分析框架：1 小时
- 运行测试 + 分析：30 分钟
- 报告撰写：30 分钟

**总计：** ~3-4 小时

---

**开始吧！** Gemini 和 Claude 可以并行工作，最后汇总结果。

*Created: 2026-01-31 11:10*
*Assigned to: @gemini (microbenchmarks) + @claude (analysis framework)*
