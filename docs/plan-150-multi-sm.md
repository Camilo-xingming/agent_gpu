# Plan: #150 Multi-SM Support (2+ Streaming Multiprocessors)

## 研究结论

现有代码已有 multi-SM 骨架（`generate` loop、NUM_SM 参数化、L2/interconnect 模块已写），但存在 3 个阻塞性问题：

### 问题 1: AXI 仲裁器响应路由损坏（Critical）
- **现状**: `ralph_gpu_top.v:982-989` 的 AXI 仲裁器用 last-write-wins priority select
- **Bug**: AXI 响应信号（`rdata`, `bid`, `bresp`, `bvalid`, `rlast`, `rid`）广播给所有 SM（见 L536-552）
- **后果**: 2 SM 并发运行时，SM0 的 read data 会被 SM1 接收，导致数据损坏
- **根因**: 缺少 outstanding transaction tracking，不知道哪个 response 属于哪个 SM

### 问题 2: L2 Cache 未接入数据路径
- **现状**: `L2_ENABLE` 默认=0，即使=1，`l2_req_valid` 也 tie 到 0
- **后果**: L1D miss 直接走 32-bit AXI bus，multi-SM 时带宽崩溃
- **模块状态**: `l2_cache` 和 `l2_interconnect` 已写完，接口匹配，只需接线

### 问题 3: Block 分配只处理 X 维度
- **现状**: `sm_block_id_y`/`sm_block_id_z` 硬编码为 0（L498-499）
- **后果**: 多维 grid 的 kernel 拿到错误的 `%ctaid.y`/`%ctaid.z`

---

## 实现计划

### Task 1: 修复 AXI 仲裁器 — 正确的 per-SM 响应路由

**方案**: 用 AXI ID 的高位编码 SM ID，实现 response demux

具体改动 (`ralph_gpu_top.v`):

1. 扩展 AXI_ID_WIDTH: 从模块内部扩展为 4 + SM_ID_W
   - 高 SM_ID_W 位 = SM index
   - 低 4 位 = SM 本地 transaction ID

2. 替换 priority-select 仲裁器为 round-robin:
   - Round-robin pointer 跟踪上次选中的 SM
   - 每次 grant 更新 pointer
   - On AW/AR grant: prepend SM ID to AXI ID

3. Response demux: 用 response ID 高位路由到正确 SM
   - `resp_sm_r = m_axi_rid[EXT_ID_W-1 -: SM_ID_W]` 提取 read response 的目标 SM
   - `resp_sm_b = m_axi_bid[EXT_ID_W-1 -: SM_ID_W]` 提取 write response 的目标 SM
   - Gate rvalid/bvalid per SM

4. 修改 SM 实例化中的 AXI response 连接：从广播改为按 SM gated

**文件**: `ralph_gpu_top.v` L170-215（信号声明）, L524-552（SM AXI连接）, L975-1015（仲裁器）

### Task 2: 接入 L2 Cache

**方案**: L1D miss path -> L2 cache -> HBM controller

具体改动:

1. 改 `L2_ENABLE` 默认值为 1

2. 在 `l1d_full` block 中，将 L1D refill/writeback bridge 改为连接 L2 而非直接 AXI:
   - L1D read miss -> 发 L2 request（`l2_req_valid[sm]`）
   - L1D writeback -> 发 L2 write request
   - L2 response -> 填充 L1D cache line

3. 连接 L2 cache 的 `l1_req_*` 端口到各 SM 的 L1D miss 信号

4. 保留 L1D bypass mode 不变（仍然直接访问 bypass_mem）

**文件**: `ralph_gpu_top.v` L382-484（l1d_full block）, L805-874（L2 integration）

### Task 3: 修复 3D Block 分配

**方案**: 将线性 block index 反算出 (x, y, z)

具体改动:

1. 添加 `sm_block_id_y[]` 和 `sm_block_id_z[]` 数组

2. 在 SCHED_DISPATCH 中从线性 index 计算 3D block ID:
   - `sm_block_id_z[i] <= next_block / (grid_dim_x * grid_dim_y)`
   - `sm_block_id_y[i] <= (next_block / grid_dim_x) % grid_dim_y`
   - `sm_block_id_x[i] <= next_block % grid_dim_x`

3. 修改 SM 实例化：`.block_id_y(sm_block_id_y[sm])`, `.block_id_z(sm_block_id_z[sm])`

**文件**: `ralph_gpu_top.v` L155-156, L496-499, L1053-1066

### Task 4: 多 SM 并行执行测试

**新文件**: `tb/tb_multi_sm.v`

测试内容:
1. 2-SM, 4-block kernel -> 验证两个 SM 各处理 2 blocks
2. 验证不同 block_id -> 结果写入不同地址（无数据损坏）
3. 测量 2-SM vs 1-SM 执行 cycle 数 -> 验证接近 2x speedup
4. 3D grid (2x2x1) -> 验证 block_id_y 正确

---

## 验收标准对照

| 标准 | 覆盖方式 |
|------|----------|
| 2-SM 配置正确执行多 CTA kernel | Task 1 + Task 4 test 1-2 |
| 性能接近线性 scaling | Task 2 (L2) + Task 4 test 3 |

## 风险

- L2 cache 模块虽然已写完但可能有 bug -> 先用 L1D bypass mode 验证 AXI 修复，再开 L2
- 除法器合成面积大 -> block ID 计算可改用迭代减法（可优化但不在本 sprint 范围）
