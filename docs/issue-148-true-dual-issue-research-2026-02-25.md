# Issue #148 True Dual-Issue 研究草稿（2026-02-25，2h timebox）

## 1. 本次 timebox 目标

- 在不无限扩散的前提下，给出可评审的技术路径。
- 明确当前瓶颈是否仍是 fetch 供给，而不是 scheduler 宽度。
- 给出可执行的中间里程碑（Milestone）和可量化 DoD/CI 门槛。

本次输出是研究草稿，不是完整 RTL 交付。

## 2. 现状复核（基于当前 `origin/master`）

### 2.1 Scheduler 具备多 lane 发射能力，但依赖 `warp_inst_valid`

- `blackwell_scheduler` 以 `warp_inst_valid` 作为 eligibility 的必要条件：`rtl/blackwell_scheduler.v:265-281`。
- 同时 `perf_sched_stall_ifetch` 已明确暴露“调度器可发射但没拿到指令”的 stall：`rtl/blackwell_scheduler.v:159,279,281`。
- slot1 在同 FU 冲突下会被抑制（`issue_valid_r[1]=0`）：`rtl/blackwell_scheduler.v:420-431`。

结论：调度器不是“完全不能双发”，但要先保证同周期有足够 warp 指令供给。

### 2.2 Fetch 主路径仍是单请求仲裁

- `sm_fetch_pipeline` 当前只有一个 `fetch_valid_arb` + 一个 `fetch_warp_id`：`rtl/sm_fetch_pipeline.v:155-169`。
- 对外仅导出单路 `fetch_req/fetch_fire`：`rtl/sm_fetch_pipeline.v:234-235`。
- 连接的是单路 icache fetch 接口：`rtl/sm_fetch_pipeline.v:205` 对应 `icache` 单路端口定义 `rtl/icache.v:25-30`。

结论：当前 fetch 供给路径天然每拍至多主导一个 warp 请求，是 true dual-issue 的主要上限。

### 2.3 SM 顶层仍对 slot1 做冲突闸门

- `issue1_fire = sched_issue_valid_mask[1] && !decode_stalled_slot1 && !lane_unit_conflict`：`rtl/streaming_multiprocessor_v2.v:2089`。
- `lane_unit_conflict` 覆盖 ALU/MUL/FP/MEM/control 等单实例资源：`rtl/streaming_multiprocessor_v2.v:979-990`。

结论：即使 fetch 扩展后，仍需用 workload 规避同 FU 冲突，才能稳定观察 dual-issue 提升。

## 3. 当前代码状态快照（本次复核）

### 3.1 `issue-148/codex`（本分支）

- `make build/tb_sm_v2_integration.vvp`：可通过（集成可编译）。
- `make test_bw_scheduler_scoreboard`：通过（6/0）。
- 旧 dual-issue 相关 testbench（如 `tb/tb_multiwarp_mixed_fu.v`、`tb/tb_alu_dual_issue_conflict.v`）在当前接口下编译失败，主要是 `l1d_resp_rdata` 等端口形态已演进。

结论：主线可编译，但 dual-issue 专项回归基线缺失。

### 3.2 `issue-148/opus`（已有原型）

- 已有 `dual-port icache + dual-port fetch pipeline` 原型提交（`36de4ef`）。
- 分支状态：`ahead 2, behind 6`（相对 `origin/master`）。
- `make build/tb_sm_v2_integration.vvp`：可编译通过。

结论：原型具备继续推进价值，但需要先 rebase + 补验证闭环。

## 4. 方案比较

| 方案 | 核心思路 | 优点 | 主要风险 | 结论 |
|---|---|---|---|---|
| A. Dual-port Fetch + Banked ICache | 两路 fetch（按 warp parity）+ icache 双端口/分 bank | 最符合 true dual-issue 目标，可扩到 HPC 4-lane | icache miss/冲突状态机复杂度上升 | 推荐主线 |
| B. 单端口 + 更深预取/NIB | 不改 icache 端口，只增强提前填充 | 侵入小、快 | 上限受单口约束，难稳定同拍双 warp 就绪 | 仅可做过渡 |
| C. 仅 scheduler 策略优化 | 更激进调度，不改 fetch | 开发快 | 供给瓶颈不变，收益有限 | 不建议单独采用 |

## 5. 建议执行路线（带中间里程碑）

> 目标是“每个里程碑都可独立评审 + 可回退”，避免一次性大改。

### M0：基线与门槛冻结（0.5d）

交付：
- 修复/新增可运行 dual-issue 专项 testbench（替换已失配的旧 TB）。
- 固定三类指标采集：`issue0_fire`、`issue1_fire`、同拍 dual-issue 次数、`perf_sched_stall_ifetch`。

Gate：
- CI 中至少有 1 个 dual-issue 专项 test 可稳定运行。

### M1：Fetch 端口抽象先行（0.5~1d）

交付：
- 在 `sm_fetch_pipeline` 内部抽象 Port A/B 仲裁与 pending 跟踪。
- 对外先保持单口 icache 适配层（功能等价，不引入性能承诺）。

Gate：
- 现有回归不退化，行为与主线一致。

### M2：ICache 双端口/分 bank（1~2d）

交付：
- `icache` 增加 Port B 接口。
- 明确 bank conflict 规则与 miss 排队策略（至少 1-deep queue）。

Gate：
- 定向测试覆盖：A/B 同拍命中、同 bank 冲突、A miss + B hit、A/B 双 miss。

### M3：端到端 dual-issue 行为验证（1d）

交付：
- 在 mixed-FU workload 观测到稳定 `issue0_fire && issue1_fire`。
- 统计 `perf_sched_stall_ifetch` 相对 M0 明显下降。

Gate：
- 同拍 dual-issue 次数 > 0 且可重复。

### M4：性能与合并门控（0.5d）

交付：
- IPC 与 stall 分解报告（前后对比）。
- merge-CI 必须包含 dual-issue 专项 test（DoD 门控）。

Gate（建议）：
- mixed-FU 场景 IPC 提升 >= 30%（与 issue 目标一致）。

## 6. 风险与回退策略

- 风险 1：双端口 miss 状态机复杂，容易引入 corner bug。  
  回退：在参数开关下保留 single-port fallback（`ENABLE_DUAL_FETCH=0`）。

- 风险 2：旧 TB 与新接口脱节，导致“看似改完但无可用证据”。  
  回退：先完成 M0，把专项验证恢复为 CI 可运行项再推进 M2。

- 风险 3：slot1 受 FU 冲突抑制，性能收益被 workload 结构掩盖。  
  回退：测试集拆分为 cross-FU 与 same-FU 两类，分别统计。

## 7. 本次建议结论

- 继续沿方案 A（dual-port fetch + banked icache）推进。
- 但必须先补 M0（可运行 dual-issue 专项基线）再进入大改。
- `issue-148/opus` 可作为实现参考，但不能跳过 rebase 和 CI gate。
