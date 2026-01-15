# RalphGPU - CUDA/PTX Compatible GPU IP

## 1. 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                        RalphGPU Top                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐        │
│  │    SM 0     │  │    SM 1     │  ...  │   SM N-1    │        │
│  │             │  │             │       │             │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │        │
│  │ │Warp Sch │ │  │ │Warp Sch │ │       │ │Warp Sch │ │        │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │        │
│  │ │ Decoder │ │  │ │ Decoder │ │       │ │ Decoder │ │        │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │        │
│  │ │Exec Unit│ │  │ │Exec Unit│ │       │ │Exec Unit│ │        │
│  │ │(ALU×32) │ │  │ │(ALU×32) │ │       │ │(ALU×32) │ │        │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │        │
│  │ │Reg File │ │  │ │Reg File │ │       │ │Reg File │ │        │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │        │
│  │ │SharedMem│ │  │ │SharedMem│ │       │ │SharedMem│ │        │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │        │
│  └─────────────┘  └─────────────┘       └─────────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Global Memory Interface                  ││
│  │                    (AXI4 / Wishbone / Custom)               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 2. 设计参数 (可扩展)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NUM_SM | 2 | SM数量，增加可提升算力 |
| THREADS_PER_WARP | 32 | 每Warp线程数 |
| WARPS_PER_SM | 4 | 每SM的Warp数 |
| NUM_REGS | 32 | 每线程寄存器数 |
| SHARED_MEM_SIZE | 16KB | 每SM共享内存大小 |
| DATA_WIDTH | 32 | 数据位宽 |

## 3. PTX 指令集子集 (Phase 1)

### 3.1 算术指令
```
add.s32  rd, ra, rb    // rd = ra + rb
sub.s32  rd, ra, rb    // rd = ra - rb
mul.lo.s32 rd, ra, rb  // rd = (ra * rb)[31:0]
mul.hi.s32 rd, ra, rb  // rd = (ra * rb)[63:32]
mad.lo.s32 rd, ra, rb, rc // rd = ra*rb + rc
div.s32  rd, ra, rb    // rd = ra / rb
rem.s32  rd, ra, rb    // rd = ra % rb
```

### 3.2 逻辑指令
```
and.b32  rd, ra, rb    // rd = ra & rb
or.b32   rd, ra, rb    // rd = ra | rb
xor.b32  rd, ra, rb    // rd = ra ^ rb
not.b32  rd, ra        // rd = ~ra
shl.b32  rd, ra, rb    // rd = ra << rb
shr.u32  rd, ra, rb    // rd = ra >> rb (logical)
shr.s32  rd, ra, rb    // rd = ra >> rb (arithmetic)
```

### 3.3 比较与分支
```
setp.eq.s32 p, ra, rb  // p = (ra == rb)
setp.ne.s32 p, ra, rb  // p = (ra != rb)
setp.lt.s32 p, ra, rb  // p = (ra < rb)
setp.le.s32 p, ra, rb  // p = (ra <= rb)
setp.gt.s32 p, ra, rb  // p = (ra > rb)
setp.ge.s32 p, ra, rb  // p = (ra >= rb)
@p bra target          // conditional branch
bra target             // unconditional branch
```

### 3.4 内存指令
```
ld.global.s32 rd, [addr]   // 从全局内存加载
st.global.s32 [addr], rs   // 存储到全局内存
ld.shared.s32 rd, [addr]   // 从共享内存加载
st.shared.s32 [addr], rs   // 存储到共享内存
```

### 3.5 特殊寄存器
```
mov.u32 rd, %tid.x     // 线程ID (x维)
mov.u32 rd, %ctaid.x   // Block ID (x维)
mov.u32 rd, %ntid.x    // Block内线程数
mov.u32 rd, %nctaid.x  // Grid内Block数
```

### 3.6 同步
```
bar.sync 0             // Block内线程同步 (barrier)
```

## 4. 指令编码 (32-bit)

```
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ 31-26  │ 25-21  │ 20-16  │ 15-11  │ 10-6   │ 5-0    │
│ OPCODE │  RD    │  RA    │  RB    │  RC/P  │ FUNC   │
└────────┴────────┴────────┴────────┴────────┴────────┘
```

| OPCODE | 指令类型 |
|--------|----------|
| 000000 | ALU (add/sub/and/or/xor/shl/shr) |
| 000001 | MUL/MAD |
| 000010 | DIV/REM |
| 000011 | SETP (比较) |
| 000100 | BRANCH |
| 000101 | LD.GLOBAL |
| 000110 | ST.GLOBAL |
| 000111 | LD.SHARED |
| 001000 | ST.SHARED |
| 001001 | MOV (特殊寄存器) |
| 001010 | BAR.SYNC |

## 5. 执行流水线

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  FETCH  │─▶│ DECODE  │─▶│  EXEC   │─▶│  MEM    │─▶│   WB    │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

- **FETCH**: 从指令内存取指令
- **DECODE**: 解码指令，读取寄存器
- **EXEC**: ALU/FPU执行
- **MEM**: 内存访问
- **WB**: 写回寄存器

## 6. 扩展路线图

### Phase 2: 浮点支持
- add.f32, sub.f32, mul.f32, div.f32
- fma.rn.f32 (fused multiply-add)

### Phase 3: 向量化
- ld.v4.f32, st.v4.f32
- 128-bit宽向量操作

### Phase 4: Tensor Core
- mma.sync (矩阵乘累加)
- 适配深度学习加速

### Phase 5: 多维支持
- 完整3D thread/block索引
- 更复杂的调度器
