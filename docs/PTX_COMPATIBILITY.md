# PTX ISA 兼容性分析

**目标版本:** PTX ISA 9.1 (2024-2026 最新)  
**分析日期:** 2026-02-08  
**分析人:** Lily

---

## PTX ISA 9.1 新特性

1. `.volatile` qualifier for `.local` state space (ld/st)
2. `.f16x2` and `.bf16x2` source types for cvt instruction
3. `.scale_vec::4X` with `.ue8m0` for mma/mma.sp
4. `.s2f6x2` instruction type for cvt
5. `multimem.cp.async.bulk` and `multimem.cp.reduce.async.bulk`

---

## RalphGPU 当前实现状态

### ✅ 已实现指令 (基础集)

#### 算术/逻辑 (ALU)
- [x] ADD, SUB (整数)
- [x] AND, OR, XOR, NOT
- [x] SHL, SHR (signed/unsigned)
- [x] ABS, NEG
- [x] MIN, MAX

#### 乘法/除法
- [x] MUL.LO, MUL.HI
- [x] MAD.LO, MAD.HI
- [x] MUL24, MAD24 (24-bit variants)
- [x] DIV_S, DIV_U
- [x] REM_S, REM_U

#### 内存访问
- [x] LD.GLOBAL
- [x] ST.GLOBAL
- [x] LD.SHARED
- [x] ST.SHARED

#### 控制流
- [x] BRA (branch)
- [x] BAR.SYNC (barrier synchronization)
- [x] NOP

#### 特殊寄存器
- [x] %tid.x, %tid.y, %tid.z
- [x] %ctaid.x, %ctaid.y, %ctaid.z
- [x] %ntid.x, %ntid.y, %ntid.z
- [x] %nctaid.x, %nctaid.y, %nctaid.z

#### 浮点运算 (部分)
- [x] FP32 via FMA
- [x] DP4A (dot product)

---

### 🚧 部分实现 / 需验证

#### Tensor Core 指令
- [?] MMA (matrix-multiply-accumulate) — 需验证覆盖范围
- [?] WGMMA — 需检查是否完整
- [?] LDMATRIX / STMATRIX

#### 浮点扩展
- [?] FP16, BF16 support
- [?] TANH, COS, SIN, SQRT, RSQRT
- [?] RCP (reciprocal)
- [?] LG2, EX2 (log2, exp2)

#### 原子操作
- [?] ATOM.ADD, ATOM.MIN, ATOM.MAX
- [?] ATOM.CAS (compare-and-swap)
- [?] ATOM.EXCH (exchange)

---

### ❌ 未实现 / 缺失指令

#### PTX 9.x 新特性
- [ ] `multimem.cp.async.bulk`
- [ ] `multimem.cp.reduce.async.bulk`
- [ ] `.volatile` qualifier for `.local`
- [ ] `.f16x2`, `.bf16x2` cvt support

#### 高级内存操作
- [ ] PREFETCH, PREFETCHU
- [ ] FENCE (memory fence)
- [ ] MEMBAR (memory barrier variants)

#### 异步操作
- [ ] CP.ASYNC (async copy)
- [ ] MBARRIER (async barrier)

#### 图形/纹理
- [ ] TEX (texture fetch)
- [ ] SULD, SUST (surface load/store)
- [ ] TLD4 (texture load 4-component)

#### 视频/图像
- [ ] VABSDIFF, VADD, VSUB
- [ ] VMAD, VMAX, VMIN
- [ ] VSET (vector set)

#### 控制流扩展
- [ ] CALL (function call) — 部分支持？
- [ ] RET (return)
- [ ] EXIT (thread exit)
- [ ] BRX (indexed branch)

#### 位操作扩展
- [ ] BFE (bit field extract)
- [ ] BFI (bit field insert)
- [ ] BFIND (find first bit)
- [ ] POPC (population count)
- [ ] BREV (bit reverse)

#### 其他
- [ ] VOTE (warp vote)
- [ ] SHFL (warp shuffle)
- [ ] ACTIVEMASK
- [ ] REDUX (reduction)

---

## Transformer 所需指令优先级

### 高优先级 (立即需要)
1. **浮点运算扩展**
   - [ ] TANH (激活函数)
   - [ ] SQRT, RSQRT (layer norm)
   - [ ] RCP (1/x, attention scaling)
   - [ ] EX2 (softmax)

2. **更大 GEMM 支持**
   - [ ] 16x16x16 MMA
   - [ ] Multi-head attention matrix ops

3. **Reduction 操作**
   - [ ] REDUX (sum/max reduction)
   - [ ] Warp-level primitives

### 中优先级 (增强性能)
1. **Async 操作**
   - [ ] CP.ASYNC (hide memory latency)
   - [ ] MBARRIER

2. **原子操作**
   - [ ] ATOM.ADD (全局计数器)

3. **Warp 操作**
   - [ ] SHFL (数据交换)
   - [ ] VOTE (条件检查)

### 低优先级 (可选)
1. 图形/纹理指令 (非 ML 核心)
2. 视频处理指令
3. PTX 9.x 特性 (向后兼容优先)

---

## 下一步行动计划

### Phase 2A: 缺失指令调研 (进行中)
- [x] 获取 PTX ISA 9.1 文档
- [ ] 详细对比指令表
- [ ] 生成完整 gap analysis

### Phase 2B: Transformer 关键指令实现
1. 实现 TANH (优先级 #1)
2. 实现 SQRT/RSQRT (layer norm)
3. 实现 RCP (attention scaling)
4. 扩展 GEMM 到 16x16x16

### Phase 2C: 测试验证
1. 为每个新指令创建单元测试
2. 更新 transformer_block 使用新指令
3. 性能对比测试

---

**状态:** 进行中  
**预计完成 Phase 2A:** 30 分钟  
**预计完成 Phase 2B:** 2-3 小时
