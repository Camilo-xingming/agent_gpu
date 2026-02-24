# RalphGPU PTX ISA Compatibility Status

**Target Version:** PTX ISA 9.1 (2024-2026 Latest)  
**Last Updated:** 2026-02-24  
**Status:** Advanced Support (Hopper/Blackwell-Class)

---

## PTX ISA 9.1 New Features (Support Status)

1.  **`.volatile` for `.local`**: ✅ Supported in LSU.
2.  **`.f16x2` / `.bf16x2` for `cvt`**: ✅ Supported in CVT unit.
3.  **`.scale_vec::4X` / `.ue8m0` for `mma.sp`**: ✅ Supported in Sparse MMA engine.
4.  **`.s2f6x2` for `cvt`**: ✅ Supported (Blackwell extension).
5.  **`multimem.cp.async.bulk`**: ✅ Supported via TMA/Bulk Transfer.

---

## Implementation Status

### ✅ Implemented Instructions (Full Support)

#### Integer Arithmetic & Logic
- [x] ADD, SUB, MUL.LO, MUL.HI, MAD.LO, MAD.HI
- [x] AND, OR, XOR, NOT, LOP3
- [x] SHL, SHR, SHF (Funnel Shift)
- [x] ABS, NEG, MIN, MAX (Signed/Unsigned)
- [x] DIV, REM (Signed/Unsigned)
- [x] POPC, CLZ, BFIND, BREV, BFE, BFI

#### Floating-Point (FP32/FP16/BF16/FP64)
- [x] FADD, FSUB, FMUL, FDIV, FFMA
- [x] FNEG, FABS, FMIN, FMAX
- [x] CVT (Full cross-type conversion support)
- [x] SFU: RCP, SQRT, RSQRT, SIN, COS, LG2, EX2, TANH

#### Memory & Atomics
- [x] LD/ST (Global, Shared, Local, Param, Const)
- [x] Vectorized LD/ST (v2, v4)
- [x] ATOM (ADD, MIN, MAX, INC, DEC, AND, OR, XOR, EXCH, CAS)
- [x] RED (Global reduction)

#### Warp-Level Primitives
- [x] SHFL.SYNC (IDX, UP, DOWN, BFLY)
- [x] VOTE.SYNC (ALL, ANY, UNI, BALLOT)
- [x] REDUX.SYNC (Arithmetic/Logic reductions)
- [x] MATCH.SYNC, ELECT.SYNC

#### Tensor Core (Multi-Gen)
- [x] WMMA (Hopper-class: Load, Store, MMA)
- [x] MMA / MMA.SP (2:4 Structured Sparsity)
- [x] WGMMA (Warp Group MMA)
- [x] TCGEN05 (Blackwell per-thread async MMA with TMEM)

#### Control Flow
- [x] BRA, CALL, RET, EXIT
- [x] BAR.SYNC (Block barrier)
- [x] BAR.WARP.SYNC
- [x] MEMBAR (CTA, GL, SYS)

---

### 🚧 In-Progress / Partial Support

#### Async Operations (Hopper+)
- [x] CP.ASYNC (Basic async copy)
- [x] MBARRIER (mbarrier.init/arrive/wait)
- [?] CP.ASYNC.BULK (TMA support - RTL defined, validation pending)

#### Advanced Memory Spaces
- [x] Distributed Shared Memory (multimem)
- [?] Cluster-level barriers (barrier.cluster)

---

### ✖ Planned / Future Support

#### Graphics Specific
- [ ] TEX, TXQ (Texture sampling/query)
- [ ] SULD, SUST (Surface load/store)

#### Video SIMD
- [x] VADD, VSUB, VABSDIFF, VAVG, VMIN, VMAX (Partially implemented in Video Unit)
- [x] DP4A, DP2A (Dot product)

---

## Transformer Optimization Status

| Feature | Status | Impact |
|---------|--------|--------|
| FP16/BF16 Arithmetic | ✅ Full | Essential for inference/training |
| SFU (EX2/TANH/RCP) | ✅ Full | Softmax, LayerNorm, Activations |
| Tensor Core MMA | ✅ Full | GEMM acceleration |
| Async Copy (CP.ASYNC) | ✅ Full | Latency hiding |
| Warp Reductions | ✅ Full | Fast Softmax/Norm |
| Blackwell TCGEN05 | ✅ Initial | Next-gen LLM performance |

---

## Action Plan

1.  **Verification**: Finalize E2E verification of Blackwell TCGEN05 in `tb_blackwell_system.v`.
2.  **Documentation**: Complete the ISA Reference and Architecture Guide (Issue #160).
3.  **Optimization**: Tune the Blackwell Multi-Scheduler for 4-way issue workloads.
