# RalphGPU - CUDA/PTX Compatible GPU IP

## 1. 架构概述

RalphGPU是一个完整的CUDA/PTX兼容GPU IP核，支持NVIDIA PTX ISA 8.5+指令集。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RalphGPU Top                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐       ┌─────────────────┐     │
│  │      SM 0       │  │      SM 1       │  ...  │    SM N-1       │     │
│  │                 │  │                 │       │                 │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ Warp Sched  │ │  │ │ Warp Sched  │ │       │ │ Warp Sched  │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │   Decoder   │ │  │ │   Decoder   │ │       │ │   Decoder   │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ SIMD ALU×32 │ │  │ │ SIMD ALU×32 │ │       │ │ SIMD ALU×32 │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ SIMD FPU×32 │ │  │ │ SIMD FPU×32 │ │       │ │ SIMD FPU×32 │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ SIMD SFU×32 │ │  │ │ SIMD SFU×32 │ │       │ │ SIMD SFU×32 │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ Tensor Core │ │  │ │ Tensor Core │ │       │ │ Tensor Core │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │  Reg File   │ │  │ │  Reg File   │ │       │ │  Reg File   │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │ Shared Mem  │ │  │ │ Shared Mem  │ │       │ │ Shared Mem  │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │ ┌─────────────┐ │     │
│  │ │Atomic Unit  │ │  │ │Atomic Unit  │ │       │ │Atomic Unit  │ │     │
│  │ └─────────────┘ │  │ └─────────────┘ │       │ └─────────────┘ │     │
│  └─────────────────┘  └─────────────────┘       └─────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Global Memory Interface (AXI4)                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. 设计参数 (可配置)

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| NUM_SM | 2 | 1-64 | SM数量，线性增加算力 |
| THREADS_PER_WARP | 32 | 32 | 每Warp线程数 (固定) |
| WARPS_PER_SM | 4 | 2-32 | 每SM的Warp数 |
| NUM_REGS | 32 | 32-256 | 每线程寄存器数 |
| SHARED_MEM_KB | 16 | 16-96 | 每SM共享内存大小 |
| DATA_WIDTH | 32 | 32/64 | 数据位宽 |

## 3. PTX 指令集支持 (完整PTX ISA 8.5+)

### 3.1 整数算术指令
```
// 基础运算
add.s32  rd, ra, rb       // rd = ra + rb
sub.s32  rd, ra, rb       // rd = ra - rb
mul.lo.s32 rd, ra, rb     // rd = (ra * rb)[31:0]
mul.hi.s32 rd, ra, rb     // rd = (ra * rb)[63:32]
mad.lo.s32 rd, ra, rb, rc // rd = ra*rb + rc
div.s32  rd, ra, rb       // rd = ra / rb
rem.s32  rd, ra, rb       // rd = ra % rb

// PTX扩展整数运算 (NEW)
abs.s32  rd, ra           // rd = |ra|
neg.s32  rd, ra           // rd = -ra
min.s32  rd, ra, rb       // rd = min(ra, rb) signed
min.u32  rd, ra, rb       // rd = min(ra, rb) unsigned
max.s32  rd, ra, rb       // rd = max(ra, rb) signed
max.u32  rd, ra, rb       // rd = max(ra, rb) unsigned
```

### 3.2 位操作指令 (NEW)
```
popc.b32  rd, ra          // rd = popcount(ra)
clz.b32   rd, ra          // rd = count_leading_zeros(ra)
bfind.s32 rd, ra          // rd = find_msb(ra)
brev.b32  rd, ra          // rd = bit_reverse(ra)
bfe.s32   rd, ra, rb      // rd = bit_field_extract(ra, pos, len)
bfe.u32   rd, ra, rb      // rd = bit_field_extract_unsigned(ra, pos, len)
bfi.b32   rd, ra, rb, rc  // rd = bit_field_insert(ra, rb, pos, len)
prmt.b32  rd, ra, rb, rc  // rd = permute_bytes(ra, rb, selector)
```

### 3.3 逻辑指令
```
and.b32  rd, ra, rb       // rd = ra & rb
or.b32   rd, ra, rb       // rd = ra | rb
xor.b32  rd, ra, rb       // rd = ra ^ rb
not.b32  rd, ra           // rd = ~ra
shl.b32  rd, ra, rb       // rd = ra << rb
shr.u32  rd, ra, rb       // rd = ra >> rb (logical)
shr.s32  rd, ra, rb       // rd = ra >> rb (arithmetic)
```

### 3.4 选择和比较指令
```
setp.eq.s32 p, ra, rb     // p = (ra == rb)
setp.ne.s32 p, ra, rb     // p = (ra != rb)
setp.lt.s32 p, ra, rb     // p = (ra < rb)
setp.le.s32 p, ra, rb     // p = (ra <= rb)
setp.gt.s32 p, ra, rb     // p = (ra > rb)
setp.ge.s32 p, ra, rb     // p = (ra >= rb)
selp.b32  rd, ra, rb, p   // rd = p ? ra : rb (NEW)
slct.s32  rd, ra, rb, rc  // rd = rc < 0 ? ra : rb (NEW)
sad.s32   rd, ra, rb, rc  // rd = |ra - rb| + rc (NEW)
```

### 3.5 浮点运算指令 (NEW - FP32)
```
add.f32   rd, ra, rb      // rd = ra + rb
sub.f32   rd, ra, rb      // rd = ra - rb
mul.f32   rd, ra, rb      // rd = ra * rb
div.f32   rd, ra, rb      // rd = ra / rb
fma.rn.f32 rd, ra, rb, rc // rd = ra * rb + rc (fused)
neg.f32   rd, ra          // rd = -ra
abs.f32   rd, ra          // rd = |ra|
min.f32   rd, ra, rb      // rd = min(ra, rb)
max.f32   rd, ra, rb      // rd = max(ra, rb)
```

### 3.6 特殊函数指令 (NEW - SFU)
```
rcp.f32     rd, ra        // rd = 1.0 / ra
sqrt.f32    rd, ra        // rd = sqrt(ra)
rsqrt.f32   rd, ra        // rd = 1.0 / sqrt(ra)
sin.f32     rd, ra        // rd = sin(ra)
cos.f32     rd, ra        // rd = cos(ra)
lg2.f32     rd, ra        // rd = log2(ra)
ex2.f32     rd, ra        // rd = 2^ra
tanh.f32    rd, ra        // rd = tanh(ra) (NEW for ML)
```

### 3.7 内存指令
```
// 全局内存
ld.global.s32 rd, [addr]  // 从全局内存加载
st.global.s32 [addr], rs  // 存储到全局内存

// 共享内存
ld.shared.s32 rd, [addr]  // 从共享内存加载
st.shared.s32 [addr], rs  // 存储到共享内存

// 扩展内存访问 (NEW)
ld.param.s32  rd, [addr]  // 从参数内存加载
ld.const.s32  rd, [addr]  // 从常量内存加载
ld.local.s32  rd, [addr]  // 从本地内存加载
st.local.s32  [addr], rs  // 存储到本地内存

// 向量加载/存储 (NEW)
ld.v2.f32    rd, [addr]   // 加载2个连续float
ld.v4.f32    rd, [addr]   // 加载4个连续float
st.v2.f32    [addr], rs   // 存储2个连续float
st.v4.f32    [addr], rs   // 存储4个连续float
```

### 3.8 原子操作指令 (NEW)
```
atom.add.s32    rd, [addr], ra    // rd = *addr; *addr += ra
atom.min.s32    rd, [addr], ra    // rd = *addr; *addr = min(*addr, ra)
atom.max.s32    rd, [addr], ra    // rd = *addr; *addr = max(*addr, ra)
atom.inc.u32    rd, [addr], ra    // rd = *addr; *addr = ((*addr >= ra) ? 0 : *addr+1)
atom.dec.u32    rd, [addr], ra    // rd = *addr; *addr = ((*addr==0 || *addr>ra) ? ra : *addr-1)
atom.and.b32    rd, [addr], ra    // rd = *addr; *addr &= ra
atom.or.b32     rd, [addr], ra    // rd = *addr; *addr |= ra
atom.xor.b32    rd, [addr], ra    // rd = *addr; *addr ^= ra
atom.exch.b32   rd, [addr], ra    // rd = *addr; *addr = ra
atom.cas.b32    rd, [addr], ra, rb // rd = *addr; if(*addr == ra) *addr = rb

// 归约操作 (不返回旧值)
red.add.s32     [addr], ra        // *addr += ra
red.min.s32     [addr], ra        // *addr = min(*addr, ra)
red.max.s32     [addr], ra        // *addr = max(*addr, ra)
```

### 3.9 Warp级指令 (NEW)
```
// Shuffle
shfl.sync.idx.b32   rd, ra, lane, mask   // rd = warp[lane].ra
shfl.sync.up.b32    rd, ra, delta, mask  // rd = warp[lane-delta].ra
shfl.sync.down.b32  rd, ra, delta, mask  // rd = warp[lane+delta].ra
shfl.sync.bfly.b32  rd, ra, delta, mask  // rd = warp[lane^delta].ra

// Vote
vote.sync.all.pred  p, mask, pred        // p = all threads have pred=1
vote.sync.any.pred  p, mask, pred        // p = any thread has pred=1
vote.sync.uni.pred  p, mask, pred        // p = all threads have same pred
vote.sync.ballot.b32 rd, mask, pred      // rd = ballot of pred across warp

// Reduction (NEW)
redux.sync.add.s32  rd, mask, ra         // rd = sum of ra across warp
redux.sync.min.s32  rd, mask, ra         // rd = min of ra across warp
redux.sync.max.s32  rd, mask, ra         // rd = max of ra across warp
```

### 3.10 Tensor Core指令 (NEW - WMMA/MMA)
```
// WMMA矩阵操作 (16x16x16)
wmma.load.a.sync.aligned.m16n16k16.{layout}.f16   frag_a, [addr], stride
wmma.load.b.sync.aligned.m16n16k16.{layout}.f16   frag_b, [addr], stride
wmma.load.c.sync.aligned.m16n16k16.{layout}.f32   frag_c, [addr], stride
wmma.store.d.sync.aligned.m16n16k16.{layout}.f32  [addr], frag_d, stride
wmma.mma.sync.aligned.m16n16k16.{layout}.f32.f16.f16.f32  frag_d, frag_a, frag_b, frag_c

// MMA指令 (低级矩阵操作)
mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32  d, a, b, c
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 d, a, b, c
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c
```

### 3.11 同步和控制流
```
// 同步
bar.sync 0            // Block内线程同步
membar.cta            // CTA内存屏障 (NEW)
membar.gl             // 全局内存屏障 (NEW)
membar.sys            // 系统内存屏障 (NEW)

// 控制流
@p bra target         // 条件分支
bra target            // 无条件分支
call target           // 函数调用 (NEW)
ret                   // 函数返回
exit                  // Kernel退出
```

### 3.12 半精度浮点指令 (NEW - FP16/BF16)
```
// FP16 运算
add.f16   rd, ra, rb      // rd = ra + rb (FP16)
sub.f16   rd, ra, rb      // rd = ra - rb
mul.f16   rd, ra, rb      // rd = ra * rb
fma.f16   rd, ra, rb, rc  // rd = ra * rb + rc (fused)
neg.f16   rd, ra          // rd = -ra
abs.f16   rd, ra          // rd = |ra|
min.f16   rd, ra, rb      // rd = min(ra, rb)
max.f16   rd, ra, rb      // rd = max(ra, rb)

// BF16 (Brain Float 16) for ML
add.bf16  rd, ra, rb      // BF16 加法
mul.bf16  rd, ra, rb      // BF16 乘法
fma.bf16  rd, ra, rb, rc  // BF16 FMA

// Packed FP16x2 (SIMD in register)
add.f16x2  rd, ra, rb     // 两个FP16并行加法
mul.f16x2  rd, ra, rb     // 两个FP16并行乘法
fma.f16x2  rd, ra, rb, rc // 两个FP16并行FMA
```

### 3.13 视频处理指令 (NEW)
```
// 32-bit 视频运算
vadd.s32     rd, ra, rb      // 视频加法
vsub.s32     rd, ra, rb      // 视频减法
vabsdiff.s32 rd, ra, rb      // 绝对差值
vmin.s32     rd, ra, rb      // 最小值
vmax.s32     rd, ra, rb      // 最大值
vmad.s32     rd, ra, rb, rc  // 乘加

// SIMD 4x8-bit (用于图像处理)
vadd4.u8     rd, ra, rb      // 4个字节并行加法
vsub4.u8     rd, ra, rb      // 4个字节并行减法
vabsdiff4.u8 rd, ra, rb      // 4个字节绝对差值

// SIMD 2x16-bit
vadd2.s16    rd, ra, rb      // 2个短整数并行加法
vmul2.s16    rd, ra, rb      // 2个短整数并行乘法

// 点积 (用于INT8 ML加速)
dp4a.s32.s32  rd, ra, rb, rc // 4元素INT8点积 + 累加
dp2a.s32.s32  rd, ra, rb, rc // 2元素INT16点积 + 累加
```

### 3.14 纹理/Surface指令 (NEW)
```
// 纹理采样
tex.1d.f32    rd, texref, coord    // 1D纹理采样
tex.2d.f32    rd, texref, coord    // 2D纹理采样
tex.3d.f32    rd, texref, coord    // 3D纹理采样
tex.cube.f32  rd, texref, coord    // Cube纹理采样
tex.level.2d.f32 rd, texref, coord, lod // 指定LOD采样

// 纹理查询
txq.width     rd, texref           // 查询纹理宽度
txq.height    rd, texref           // 查询纹理高度
txq.depth     rd, texref           // 查询纹理深度
txq.num_mipmap_levels rd, texref   // 查询Mipmap级别数

// Surface访问
suld.b.2d.f32 rd, surfref, coord   // Surface加载
sust.b.2d.f32 surfref, coord, rs   // Surface存储
sured.add.b.2d.s32 surfref, coord, ra // Surface原子加
```

### 3.15 特殊寄存器
```
mov.u32 rd, %tid.x    // 线程ID (x/y/z)
mov.u32 rd, %ctaid.x  // Block ID (x/y/z)
mov.u32 rd, %ntid.x   // Block维度 (x/y/z)
mov.u32 rd, %nctaid.x // Grid维度 (x/y/z)
mov.u32 rd, %laneid   // Lane ID (0-31) (NEW)
mov.u32 rd, %warpid   // Warp ID (NEW)
mov.u32 rd, %smid     // SM ID (NEW)
mov.u64 rd, %clock64  // 时钟计数器 (NEW)
```

## 4. 指令编码 (32-bit)

```
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ 31-26  │ 25-21  │ 20-16  │ 15-11  │ 10-6   │ 5-0    │
│ OPCODE │  RD    │  RA    │  RB    │  RC/P  │ FUNC   │
└────────┴────────┴────────┴────────┴────────┴────────┘
```

| OPCODE (6-bit) | 指令类型 |
|----------------|----------|
| 000000 | ALU (add/sub/and/or/xor/shl/shr/abs/neg/min/max/popc/clz/bfe/bfi/selp) |
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
| 001011 | EXIT |
| 001100 | RET |
| 001101 | FP32_ARITH (add/sub/mul/div/fma) |
| 001110 | FP32_SPECIAL (rcp/sqrt/rsqrt/sin/cos/lg2/ex2) |
| 001111 | FP64_ARITH |
| 010000 | FP16_ARITH |
| 010001 | CVT (类型转换) |
| 010010 | LD.PARAM |
| 010011 | LD.CONST |
| 010100 | LD.LOCAL |
| 010101 | ST.LOCAL |
| 010110 | LD.V2 |
| 010111 | LD.V4 |
| 011000 | ST.V2 |
| 011001 | ST.V4 |
| 011010 | ATOM |
| 011011 | RED |
| 011100 | SHFL |
| 011101 | VOTE |
| 011110 | REDUX |
| 011111 | WMMA.LOAD |
| 100000 | WMMA.STORE |
| 100001 | WMMA.MMA |
| 100010 | MMA |
| 100011 | CALL |
| 100100 | MEMBAR |
| 100101 | VIDEO (vadd/vsub/vabsdiff/vmin/vmax/dp4a) |
| 100110 | TEX (纹理采样) |
| 100111 | TXQ (纹理查询) |
| 101000 | SULD (Surface加载) |
| 101001 | SUST (Surface存储) |
| 101010 | SURED (Surface归约) |
| 111111 | NOP |

## 5. 执行流水线

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  FETCH  │─▶│ DECODE  │─▶│  EXEC   │─▶│  MEM    │─▶│   WB    │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
                              │
                              ├─▶ ALU (1 cycle)
                              ├─▶ FPU (1-4 cycles)
                              ├─▶ SFU (4-8 cycles)
                              ├─▶ Tensor Core (16+ cycles)
                              └─▶ Atomic (variable)
```

## 6. 功能单元

### 6.1 ALU (Arithmetic Logic Unit)
- 支持所有整数运算和位操作
- 单周期延迟
- 32个并行单元 (每Warp)

### 6.2 FPU (Floating-Point Unit)
- IEEE 754 FP32兼容
- 支持所有舍入模式 (RN/RZ/RM/RP)
- 支持FTZ (Flush-to-Zero)
- 1-4周期延迟

### 6.3 SFU (Special Function Unit)
- 查表+插值实现
- 支持rcp/sqrt/rsqrt/sin/cos/lg2/ex2/tanh
- 4-8周期延迟
- 精度可配置

### 6.4 Tensor Core
- WMMA 16x16x16 FP16矩阵乘累加
- MMA低级矩阵指令
- INT8 8x8x32配置
- 适用于深度学习加速

### 6.5 Atomic Unit
- 支持所有原子操作
- 共享内存和全局内存原子
- 读-改-写序列化

### 6.6 FP16 Unit (NEW)
- IEEE 754 FP16半精度运算
- BF16 (Brain Float 16) 支持
- FP16x2 packed SIMD操作
- 2-3周期延迟

### 6.7 Control Flow Unit (NEW)
- 分支预测和执行
- 函数调用/返回硬件栈
- SIMT分歧处理和重聚合
- 谓词寄存器管理

### 6.8 Video Unit (NEW)
- SIMD字节/半字操作
- DP4A/DP2A INT8点积加速
- 视频编解码优化指令
- 1-2周期延迟

### 6.9 Texture Unit (NEW)
- 1D/2D/3D/Cube纹理采样
- 双线性/三线性过滤
- LOD和各向异性过滤
- 纹理缓存集成
- Surface加载/存储/原子

## 7. 内存子系统

### 7.1 寄存器文件
- 32个32-bit寄存器/线程
- 3读1写端口
- 单周期访问

### 7.2 共享内存
- 16KB/SM (可配置到96KB)
- 32 banks
- 单周期访问 (无bank冲突时)

### 7.3 全局内存
- AXI4接口
- 地址合并
- 可变延迟

### 7.4 常量内存
- 只读
- 缓存优化
- 广播访问

## 8. RTL文件列表

| 文件 | 描述 | 行数 |
|------|------|------|
| rtl/gpu_defines.vh | 全局定义和参数 | ~350 |
| rtl/alu.v | ALU和SIMD ALU (扩展指令) | ~280 |
| rtl/mul_unit.v | 乘法单元 | ~180 |
| rtl/fpu.v | FP32浮点运算单元 | ~400 |
| rtl/fp16_unit.v | FP16/BF16半精度单元 (NEW) | ~450 |
| rtl/sfu.v | 特殊函数单元 | ~240 |
| rtl/tensor_core.v | Tensor Core (WMMA/MMA) | ~480 |
| rtl/atomic_unit.v | 原子操作单元 | ~200 |
| rtl/warp_shuffle.v | Warp Shuffle/Vote/Redux | ~350 |
| rtl/control_flow_unit.v | 控制流单元 (bra/call/ret) (NEW) | ~280 |
| rtl/video_unit.v | 视频处理单元 (NEW) | ~320 |
| rtl/texture_unit.v | 纹理/Surface单元 (NEW) | ~400 |
| rtl/decoder.v | 指令解码器 (完整PTX 8.5+) | ~460 |
| rtl/register_file.v | 寄存器文件 | ~200 |
| rtl/warp_scheduler.v | Warp调度器 | ~300 |
| rtl/shared_memory.v | 共享内存 | ~150 |
| rtl/memory_interface.v | 内存接口 | ~250 |
| rtl/streaming_multiprocessor.v | SM核心 | ~500 |
| rtl/ralph_gpu_top.v | 顶层模块 | ~400 |

## 9. 测试套件

| 测试 | 描述 | 状态 |
|------|------|------|
| tb_alu.v | ALU基础测试 | ✅ PASS |
| tb_alu_extended.v | ALU扩展指令测试 | ✅ PASS |
| tb_mul_unit.v | 乘法单元测试 | ✅ PASS |
| tb_fpu.v | FPU测试 | ✅ PASS |
| tb_decoder.v | 解码器测试 | ✅ PASS |
| tb_register_file.v | 寄存器文件测试 | ✅ PASS |
| tb_shared_memory.v | 共享内存测试 | ✅ PASS |
| tb_warp_scheduler.v | Warp调度测试 | ✅ PASS |
| tb_vector_add.v | 向量加法集成测试 | ✅ PASS |

## 10. 性能估算

| 配置 | 算力 (TOPS) | 说明 |
|------|-------------|------|
| 2 SM, 100MHz | 0.2 INT32 | 最小配置 |
| 8 SM, 500MHz | 4.0 INT32 | 中等配置 |
| 16 SM, 1GHz | 16.0 INT32 | 高性能配置 |
| 16 SM + TC | 64+ FP16 | 包含Tensor Core |

## 11. 参考文档

- [PTX ISA 9.1 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [PTX ISA 8.5 PDF](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)
- [NVIDIA Tensor Core Evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
