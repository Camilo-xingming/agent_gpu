# Command Processor Design — Issue #151

> Kernel launch interface: command queue + CTA dispatch + host MMIO
> Author: CoderOpus | Date: 2026-02-24 | Branch: issue-151/opus

## 1. 现状分析

### 当前 kernel launch 流程
`ralph_gpu_top.v` 中 host 通过 raw CSR 接口直接写寄存器：
```
CSR 0x004 (GPU_CONTROL) → kernel_start_reg
CSR 0x008 (KERNEL_PC)   → kernel_pc_reg
CSR 0x00C-0x020          → grid_dim_{x,y,z}, block_dim_{x,y,z}
```
然后 scheduler FSM (SCHED_IDLE → DISPATCH → WAIT → DONE) 分配 CTA 到各 SM。

### 问题
1. **单 kernel 限制**：一次只能运行一个 kernel，无法排队
2. **无参数传递**：kernel params 无法通过 command queue 传入
3. **无 shared memory 大小配置**：每个 kernel 不能声明自己需要的 shared memory
4. **无标准 host 接口**：CSR 不是 AXI-Lite，testbench 直连寄存器
5. **无 completion fence**：只有简单的 irq_kernel_done

---

## 2. 架构设计

### 2.1 整体架构

```
  Host (AXI-Lite)                        Command Processor
  ┌───────────┐         MMIO            ┌──────────────────────┐
  │ Driver /  │ ──────────────────────> │ Host Interface       │
  │ Testbench │ <────────────────────── │   (AXI-Lite Slave)   │
  └───────────┘                         ├──────────────────────┤
                                        │ Command Queue        │
                Global Memory           │   (Ring Buffer)      │
  ┌───────────┐   AXI Read             │   - head/tail ptrs   │
  │   DRAM    │ <────────────────────── │   - descriptor fetch │
  └───────────┘                         ├──────────────────────┤
                                        │ CTA Dispatcher       │
                                        │   - block_id gen     │
                                        │   - SM allocation    │
                SM[0..N-1]              │   - occupancy track  │
  ┌───────────┐  kernel_start          ├──────────────────────┤
  │    SM0    │ <────────────────────── │ Completion Engine    │
  │    SM1    │ <────────────────────── │   - fence tracking   │
  │    ...    │                         │   - interrupt gen     │
  └───────────┘                         └──────────────────────┘
```

### 2.2 Command Processor 子模块

| 模块 | 职责 |
|------|------|
| `cp_host_interface` | AXI-Lite slave，接收 doorbell/CSR 写入 |
| `cp_command_queue` | 管理 ring buffer head/tail，从 global memory 取 descriptor |
| `cp_cta_dispatcher` | 解析 kernel descriptor，生成 CTA block_id，分配到 SM |
| `cp_completion_engine` | 跟踪 kernel 完成状态，生成中断/fence |

---

## 3. 接口规格

### 3.1 Host Interface — AXI-Lite Slave (MMIO)

保留现有 CSR 地址空间兼容性，新增 Command Processor 寄存器：

| Offset | Name | RW | 描述 |
|--------|------|----|------|
| 0x000 | GPU_STATUS | RO | `{error, busy, ready}` (保持兼容) |
| 0x004 | GPU_CONTROL | RW | bit[0]=legacy start, bit[1]=CP enable |
| 0x008 | KERNEL_PC | RW | Legacy kernel PC (兼容) |
| 0x00C-0x020 | GRID/BLOCK_DIM | RW | Legacy dims (兼容) |
| **0x030** | **CMD_QUEUE_BASE_LO** | RW | Command queue base addr [31:0] |
| **0x034** | **CMD_QUEUE_BASE_HI** | RW | Command queue base addr [63:32] (reserved) |
| **0x038** | **CMD_QUEUE_SIZE** | RW | Queue entries (power of 2, max 256) |
| **0x03C** | **CMD_QUEUE_HEAD** | RO | Head pointer (CP consumes) |
| **0x040** | **CMD_QUEUE_TAIL** | RW | Tail pointer (host produces, doorbell) |
| **0x044** | **CMD_FENCE_VALUE** | RO | Last completed kernel fence ID |
| **0x048** | **CMD_FENCE_SIGNAL** | RW | Fence value to signal interrupt on |
| **0x04C** | **CP_STATUS** | RO | CP state + active kernel count |

**Doorbell 机制**：Host 写 CMD_QUEUE_TAIL 时，CP 自动开始消费 queue entries。

### 3.2 Kernel Descriptor 格式 (在 Global Memory 中)

每个 descriptor 64 bytes (16 x 32-bit words)：

| Word | Field | 描述 |
|------|-------|------|
| 0 | `kernel_pc` | Kernel 入口 PC |
| 1 | `grid_dim_x` | Grid X 维度 |
| 2 | `grid_dim_y` | Grid Y 维度 |
| 3 | `grid_dim_z` | Grid Z 维度 |
| 4 | `block_dim_x` | Block X 维度 |
| 5 | `block_dim_y` | Block Y 维度 |
| 6 | `block_dim_z` | Block Z 维度 |
| 7 | `shared_mem_bytes` | 动态 shared memory 大小 |
| 8 | `param_addr` | Kernel 参数地址 (global memory) |
| 9 | `param_size` | 参数大小 (bytes) |
| 10 | `fence_id` | Completion fence value |
| 11 | `flags` | bit[0]=barrier_all (等前一个完成) |
| 12-15 | `reserved` | 未来扩展 |

### 3.3 SM Interface (不变，保持兼容)

SM 的 kernel launch 接口不需要改动：
```verilog
input  kernel_start,
input  [31:0] kernel_pc,
input  [31:0] block_id_{x,y,z},
input  [31:0] block_dim_{x,y,z},
input  [31:0] grid_dim_{x,y,z},
output kernel_done
```

CP 只是替代 `ralph_gpu_top.v` 中的 inline scheduler FSM，通过相同信号驱动 SM。

### 3.4 Global Memory Interface (AXI4 Read)

CP 需要一个 AXI4 read port 来从 global memory 读取 kernel descriptors：
- 64-byte burst read (一个 descriptor)
- 可以共享现有 AXI arbiter，给 CP 一个额外的 port

---

## 4. 状态机设计

### 4.1 Command Queue FSM

```
IDLE ──(tail != head)──> FETCH_DESC ──(AXI read done)──> PARSE_DESC
  ^                                                          │
  │                      DISPATCH_CTA <──────────────────────┘
  │                          │
  │                      (all CTAs dispatched)
  │                          │
  │                      WAIT_KERNEL ──(all SMs done)──> COMPLETE
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### 4.2 CTA Dispatcher

每个 clock 尝试为一个 idle SM 分配一个 CTA：
1. 维护 `next_block_linear` 计数器 (0 到 total_blocks-1)
2. 线性 ID → (x, y, z) 分解 (与当前实现一致)
3. 找到 idle SM → assert `sm_kernel_start[i]`
4. 所有 block 分配完 → 进入 WAIT

### 4.3 Completion Engine

1. 监控 `sm_done` 信号
2. 当一个 kernel 的所有 CTA 完成 → 更新 `fence_value` 为 descriptor 的 `fence_id`
3. 如果 `fence_value == fence_signal` → 触发中断
4. 推进 queue head

---

## 5. 实现计划

### Phase 1: CP Core + Legacy 兼容 (本 Sprint 交付)
- [ ] `command_processor.v` — 主模块，包含:
  - Command queue ring buffer 管理
  - Kernel descriptor 解析
  - CTA dispatcher (替代 inline scheduler)
  - Completion tracking
- [ ] 修改 `ralph_gpu_top.v` — 将 inline scheduler FSM 替换为 CP 实例
  - 新增 CSR 地址映射
  - Legacy mode: 当 CP 未 enable 时，保持原有 CSR 直写行为
- [ ] Testbench: 验证 legacy 兼容 + 新 command queue 模式

### Phase 2: AXI-Lite Host Interface (后续)
- [ ] 将 CSR 接口升级为标准 AXI-Lite slave
- [ ] DMA 读取 kernel descriptor

### 决策点 (需要 Jerry 标注)
1. **Descriptor 来源**：Phase 1 是否先用 CSR 直写 descriptor (不走 global memory DMA)? 这样可以快速验证 queue 逻辑，DMA 留 Phase 2。
2. **Queue 深度**：是否 8 entries 够用？
3. **Fence 语义**：fence_id 递增还是 host 指定？当前设计是 host 在 descriptor 中指定。
4. **Legacy 兼容**：是否保留 CSR 直写 launch 作为后备模式？

---

## 6. 文件变更清单

| 文件 | 操作 | 描述 |
|------|------|------|
| `rtl/command_processor.v` | 新增 | CP 主模块 |
| `rtl/ralph_gpu_top.v` | 修改 | 集成 CP，替换 inline scheduler |
| `rtl/gpu_defines.vh` | 修改 | 新增 CP 相关定义 |
| `tb/tb_command_processor.v` | 新增 | CP 单元测试 |
| `tb/tb_ralph_gpu_top.v` | 修改 | 添加 CP 模式测试用例 |

---

## 7. 权衡分析

### 方案 A: 全 DMA (descriptor 在 global memory)
- ✅ 更接近真实 GPU
- ❌ 需要 AXI read port，增加 arbiter 复杂度
- ❌ 验证更复杂

### 方案 B: CSR 直写 descriptor (Phase 1 推荐)
- ✅ 实现快，容易验证
- ✅ 不需要修改 AXI arbiter
- ❌ 不如 DMA 模式真实
- 可以后续迁移到 DMA

### 推荐：Phase 1 用方案 B，Phase 2 迁移到方案 A

