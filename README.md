# RalphGPU - CUDA/PTX Compatible GPU IP

一个简洁、可扩展的开源GPU IP，兼容NVIDIA CUDA/PTX执行模型。

## 特性

- **SIMT架构**: 单指令多线程，兼容CUDA执行模型
- **PTX指令集子集**: 支持基本算术、逻辑、内存操作
- **可扩展设计**: 通过参数化SM数量扩展算力
- **AXI4接口**: 标准内存接口，易于集成
- **模块化**: 清晰的模块划分，便于理解和扩展

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     RalphGPU Top                            │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐      ┌───────────┐           │
│  │   SM 0    │  │   SM 1    │ ...  │  SM N-1   │           │
│  │           │  │           │      │           │           │
│  │ Warp Sched│  │ Warp Sched│      │ Warp Sched│           │
│  │ Decoder   │  │ Decoder   │      │ Decoder   │           │
│  │ SIMD ALU  │  │ SIMD ALU  │      │ SIMD ALU  │           │
│  │ Reg File  │  │ Reg File  │      │ Reg File  │           │
│  │ Shared Mem│  │ Shared Mem│      │ Shared Mem│           │
│  └───────────┘  └───────────┘      └───────────┘           │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Global Memory Interface (AXI4)             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
GPU/
├── rtl/                    # RTL源文件
│   ├── gpu_defines.vh      # 全局定义和参数
│   ├── alu.v               # ALU和SIMD ALU
│   ├── mul_unit.v          # 乘法单元
│   ├── register_file.v     # 寄存器文件
│   ├── decoder.v           # 指令解码器
│   ├── warp_scheduler.v    # Warp调度器
│   ├── shared_memory.v     # 共享内存
│   ├── memory_interface.v  # 全局内存接口
│   ├── streaming_multiprocessor_v2.v  # SM模块
│   └── ralph_gpu_top.v     # 顶层模块
├── tb/                     # Testbench
│   └── tb_ralph_gpu.v      # 主测试文件
├── tools/                  # 工具
│   └── ptx_assembler.py    # PTX汇编器
├── examples/               # 示例程序
│   ├── vector_add.ptx      # 向量加法
│   └── saxpy.ptx           # SAXPY
├── doc/                    # 文档
│   └── architecture.md     # 架构说明
├── Makefile               # 构建脚本
└── README.md              # 本文件
```

## 快速开始

### 依赖

- Icarus Verilog (仿真)
- Python 3 (汇编器)
- GTKWave (可选，波形查看)

### 构建和仿真

```bash
# 运行仿真
make sim

# 查看波形
make wave

# 汇编PTX示例
make assemble

# 清理
make clean
```

## 配置参数

在 `rtl/gpu_defines.vh` 中修改以下参数来扩展算力:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NUM_SM | 2 | SM数量 |
| THREADS_PER_WARP | 32 | 每Warp线程数 |
| WARPS_PER_SM | 4 | 每SM的Warp数 |
| NUM_REGS | 32 | 每线程寄存器数 |
| SHARED_MEM_KB | 16 | 共享内存大小(KB) |

### 算力扩展示例

```
配置1 (小型): NUM_SM=2  → 2×4×32 = 256 线程
配置2 (中型): NUM_SM=4  → 4×4×32 = 512 线程
配置3 (大型): NUM_SM=16 → 16×4×32 = 2048 线程
```

### Profile 选择 (Lite / Balanced / HPC)

可在构建时通过编译宏选择配置档位:

```bash
make GPU_PROFILE=LITE
make GPU_PROFILE=BALANCED
make GPU_PROFILE=HPC
```

Profile 由 `rtl/gpu_config.vh`、`rtl/gpu_defines.vh`、`rtl/memory_config.vh` 控制。
注意: 当前实现默认 `WARPS_PER_SM=4`，扩展到更高 warp 数需同步修改调度器与控制流逻辑。

## PTX指令集

### 支持的指令

**算术运算**
```
add.s32  rd, ra, rb    // rd = ra + rb
sub.s32  rd, ra, rb    // rd = ra - rb
mul.lo.s32 rd, ra, rb  // rd = (ra * rb)[31:0]
mad.lo.s32 rd, ra, rb, rc // rd = ra*rb + rc
```

**逻辑运算**
```
and.b32  rd, ra, rb    // rd = ra & rb
or.b32   rd, ra, rb    // rd = ra | rb
xor.b32  rd, ra, rb    // rd = ra ^ rb
shl.b32  rd, ra, rb    // rd = ra << rb
shr.u32  rd, ra, rb    // rd = ra >> rb
```

**内存操作**
```
ld.global.s32 rd, [addr]   // 全局内存加载
st.global.s32 [addr], rs   // 全局内存存储
ld.shared.s32 rd, [addr]   // 共享内存加载
st.shared.s32 [addr], rs   // 共享内存存储
```

**特殊寄存器**
```
mov.u32 rd, %tid.x     // 线程ID
mov.u32 rd, %ctaid.x   // Block ID
mov.u32 rd, %ntid.x    // Block维度
```

## CSR寄存器

| 地址 | 名称 | 说明 |
|------|------|------|
| 0x000 | GPU_STATUS | 状态 (RO) |
| 0x004 | GPU_CONTROL | 控制 (写1启动) |
| 0x008 | KERNEL_PC | Kernel入口地址 |
| 0x00C | GRID_DIM_X | Grid X维度 |
| 0x018 | BLOCK_DIM_X | Block X维度 |

## 扩展路线图

- [x] Phase 1: 整数运算、基本内存操作
- [ ] Phase 2: 浮点支持 (FP32)
- [ ] Phase 3: 向量化 (128-bit SIMD)
- [ ] Phase 4: Tensor Core
- [ ] Phase 5: 完整3D索引支持

## 许可证

MIT License
