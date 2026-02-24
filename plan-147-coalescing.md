# Plan: #147 Memory Coalescing Unit (SM Level)

> SM 级内存合并单元，将 32 线程 LD/ST 合并为最少的 128B cache-line 事务。

## 现状分析

### 已有代码
- `memory_coalescing_unit.v` (385 行): 独立模块，有完整的合并逻辑 + 状态机，但未被 SM v2 实例化。
- `warp_memory_unit` (同文件): 简化版，只处理 perfect coalesce (同一 cache line)。也未被实例化。
- `sm_gmem_arbiter.v`: 4 源仲裁器 (Normal > Atomic > ACE > Texture)，直连 memory_interface。
- `memory_interface.v`: AXI4 桥，有简单的 burst 合并 (req_is_fullwarp_contig_read)，但不做真正的 coalescing。

### 当前数据路径 (SM v2, line 1283-1287)
```
store: rf_rd_data_a (32 addr) -> gmem_normal_req -> arbiter -> memory_interface -> AXI
load:  L1 miss -> l1_miss_req_addr_vec (强制 contiguous base+4i) -> gmem_normal_req -> arbiter -> memory_interface
```

**关键发现**: Load 路径已经在 L1 miss 时做了"伪合并"— 把所有 lane 地址替换成 base+4i（line 1234），本质上是假设 L1 miss 时所有 lane 在同一 cache line。**Store 路径完全没有合并**，32 个独立地址直接传给 memory_interface。

## 方案设计

### 目标
在 sm_gmem_arbiter 之前插入 coalescing unit，对 **store 路径** 做真正的合并（load 路径已被 L1 处理，但也可以受益于更好的合并）。

### 架构位置
```
pipeline issue -> [NEW: coalescing_unit] -> sm_gmem_arbiter -> memory_interface -> AXI
                      ^                        ^
                  合并 store 请求          原有仲裁逻辑不变
```

### 子任务

#### 1. 重写 memory_coalescing_unit.v
现有模块有正确的设计思路但接口不匹配 SM v2 的 gmem_normal_req 信号。需要：

- **输入**: 32-lane addr/wdata/mask + write flag (和现在 gmem_normal_req 格式一致)
- **输出**: 与 sm_gmem_arbiter 的 normal_req 接口一致
- **合并逻辑**:
  - 分析 32 个地址，找出 unique cache lines (128B 对齐)
  - 每个 unique line 生成一个 memory transaction
  - MAX_COALESCED = 4 (最多 4 个不同 cache line，超出的 thread 需要多轮)
- **多轮处理**: 如果 >4 unique lines，用状态机分批发送
- **完美合并快速路径**: 如果所有 thread 在同一 cache line，1 cycle 直通

#### 2. 对齐检测和分段逻辑
- 128B cache line 边界检测 (addr bits 31:7)
- 字节偏移计算 (addr bits 6:0)
- 写掩码生成（per-byte 掩码 within 128B line）
- 跨 cache line 访问检测（单个 thread 的 4B 访问可能跨线，但 32-bit aligned 不会）

#### 3. SM v2 集成
在 streaming_multiprocessor_v2.v 中:

**改动点 (line ~1283-1287)**:
```verilog
// BEFORE: gmem_normal_req 直连 store/L1miss
assign gmem_normal_req_valid = gmem_store_req_valid || l1_miss_req_valid;
...

// AFTER: store 经过 coalescing, L1 miss 仍直通
wire coal_req_valid, coal_req_write;
wire [NUM_LANES*32-1:0] coal_req_addr;
wire [SIMD_WIDTH-1:0] coal_req_wdata;
wire [NUM_LANES-1:0] coal_req_mask;
wire coal_ready;

memory_coalescing_unit u_coalescing (
    .clk(clk), .rst_n(rst_n),
    .req_valid(gmem_store_req_valid),
    .req_write(1'b1),
    .req_addr(rf_rd_data_a),
    .req_wdata(rf_rd_data_b),
    .req_mask(issue_mask),
    // output -> merged with L1 miss path
    ...
);

// Mux: coalesced store OR L1 miss (L1 miss has priority, store waits)
assign gmem_normal_req_valid = coal_out_valid || l1_miss_req_valid;
```

**Stall 信号改动**: 当 coalescing 正在处理多轮 store 时，需要 stall pipeline。在 stall_decode 逻辑中加入 coal_ready。

#### 4. 性能基准
- 修改 tb_memory_coalescing_unit.v 测试：
  - 连续地址 (完美合并): 32 thread -> 1 transaction
  - 分散地址 (无合并): 32 thread -> 4+ transactions
  - 混合模式: 部分连续 + 部分分散
- 添加 perf counter 到 performance_counters.v: coalesce_ratio, coalesced_transactions

#### 5. Python Test Generator
- tests/test_coalescing.py: 生成 PTX 程序测试不同访问模式
  - Stride-1 (完美合并)
  - Stride-32 (跨 cache line)
  - Random scatter

## 风险和注意事项

1. **Store stall 影响**: 多轮 coalescing 会增加 store 延迟，但减少总带宽需求。需确保 stall 信号正确接入 scheduler。
2. **L1 miss 路径不改**: 现有 L1 miss 已做伪合并 (line 1234)，改它风险高且收益低。
3. **memory_interface.v 不改**: Issue #147 也建议在 SM 层面做 coalescing，不动 memory_interface。
4. **Atomic 路径不改**: Atomic 走独立通道，不经过 coalescing。

## 实现顺序

1. [ ] 重写 memory_coalescing_unit.v — 适配 SM v2 接口
2. [ ] 修改 streaming_multiprocessor_v2.v — 插入 coalescing unit
3. [ ] 修改 stall 逻辑 — coal_ready 接入 stall_decode
4. [ ] 更新 tb_memory_coalescing_unit.v — 端到端测试
5. [ ] 添加 perf counters — stat_coalesce_ratio
6. [ ] make lint 通过
7. [ ] Python test generator (可选，scope 大可后续)
