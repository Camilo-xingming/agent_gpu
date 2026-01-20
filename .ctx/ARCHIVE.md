# RalphGPU Completed MVU Archive

## Summary
This file contains summaries of completed and verified Minimal Verifiable Units (MVUs).

---

## MVU-001: Basic ALU Operations
**Completed**: 2026-01-17
**Scope**: ADD, SUB, AND, OR, XOR, NOT, SHL, SHR
**RTL**: rtl/alu.v
**Tests**: tb/tb_alu.v (26/26 pass)
**Verification**: Bit-exact integer comparison

---

## MVU-002: Multiply Unit
**Completed**: 2026-01-17
**Scope**: MUL, MAD, DIV, REM (signed/unsigned)
**RTL**: rtl/mul_unit.v
**Tests**: tb/tb_mul_unit.v (27/27 pass)
**Verification**: Bit-exact integer comparison

---

## MVU-003: Floating Point Unit
**Completed**: 2026-01-17
**Scope**: FADD, FSUB, FMUL, FMA, FDIV, RCP, SQRT
**RTL**: rtl/fpu.v
**Tests**: tb/tb_fpu.v (26/26 pass)
**Verification**: Bit-exact FP comparison

---

## MVU-004: Decoder
**Completed**: 2026-01-17
**Scope**: All PTX opcode decoding
**RTL**: rtl/decoder.v
**Tests**: tb/tb_decoder.v (16/16 pass)
**Verification**: Field extraction correctness

---

## MVU-005: Shared Memory
**Completed**: 2026-01-17
**Scope**: LD.SHARED, ST.SHARED, bank conflict detection
**RTL**: rtl/shared_memory.v
**Tests**: tb/tb_shared_memory.v (11/11 pass)
**Verification**: Read-after-write, async write

---

## MVU-006: Warp Scheduler
**Completed**: 2026-01-17
**Scope**: Warp allocation, scheduling, sync
**RTL**: rtl/warp_scheduler.v
**Tests**: tb/tb_warp_scheduler.v (10/10 pass)
**Verification**: Multi-warp interleaving

---

## MVU-007: Extended ALU (DP4A/DP2A)
**Completed**: 2026-01-18
**Scope**: DP4A, DP2A, bmsk, szext, fns, shf, lop3, cnot
**RTL**: rtl/alu.v, rtl/video_unit.v
**Tests**: tb/tb_alu_extended.v (50/50 pass)
**Verification**: Bit-exact integer comparison

---

## MVU-008: SM V2 Integration
**Completed**: 2026-01-17
**Scope**: Scoreboard, FU tracking, WB arbitration
**RTL**: rtl/streaming_multiprocessor_v2.v
**Tests**: tb/tb_sm_v2_integration.v (4/4 core pass)
**Verification**: Multi-cycle hazard handling

---

## MVU-009: LLM Operators
**Completed**: 2026-01-19
**Scope**: DP4A dot product, GEMM, ReLU, Attention, Residual
**RTL**: Full GPU top level
**Tests**: tb/tb_llm_operators.v (5/5 pass)
**Verification**: End-to-end kernel execution

---

## MVU-010: Trigonometric Functions
**Completed**: 2026-01-19
**Scope**: sin.f32, cos.f32, tan (via sin/cos/div)
**RTL**: rtl/sfu.v
**Tests**: tb/tb_trig_operators.v (3/3 pass)
**Verification**: Tolerance-based SFU comparison

---

## MVU-011: B300 Features
**Completed**: 2026-01-20
**Scope**: mbarrier, st.async, multimem, barrier.cluster, WGMMA, cache policy
**RTL**: Multiple new files
**Tests**: tb/tb_b300_features.v (145/145 pass)
**Verification**: Decode and functional correctness

---

## MVU-012: RAS/ECC
**Completed**: 2026-01-20
**Scope**: SEC-DED ECC for register file
**RTL**: rtl/register_file_banked.v
**Tests**: Compilation + integration (pass)
**Verification**: ECC encode/decode functions

---

## MVU-013: Test Generator & FRM Comparison
**Completed**: 2026-01-20
**Scope**: Python test generator + RTL vs FRM comparison harness
**Tools**: tools/test_generator.py, tools/rtl_frm_compare.py
**Tests**: 75/75 pass (60 ALU + 10 FP32 + 5 edge cases)
**Verification**: FRM vs expected results comparison
**Process**: Codex implemented, Gemini reviewed (PASS)

---

## MVU-014: FRM Branch Instruction Support
**Completed**: 2026-01-20
**Scope**: BRANCH opcode in FRM (register + predicate based)
**Tools**: tools/gpu_simulator.py
**Features**:
- Unconditional branch (bra LABEL)
- Register-based conditional (bra.nz/bra.z rN, LABEL)
- Predicate-based conditional (@p bra LABEL)
**Tests**: 75/75 pass (existing tests continue to pass)
**Verification**: FRM vs expected results comparison
**Process**: Codex implemented, Gemini reviewed (PASS after fix)

---

## MVU-015: FRM Memory Operations
**Completed**: 2026-01-20
**Scope**: LD_PARAM, LD_CONST, ATOM opcodes in FRM
**Tools**: tools/gpu_simulator.py
**Features**:
- LD_PARAM: Load from parameter memory
- LD_CONST: Load from constant memory
- ATOM: Atomic operations (ADD, EXCH, CAS, AND, OR, XOR, MIN, MAX)
- AtomicFunc enum for atomic function codes
- param_memory and const_memory dictionaries
**Tests**: 75/75 pass (existing tests continue to pass)
**Verification**: Syntax check + regression pass
**Process**: Codex implemented, Gemini reviewed (found CAS bug, FIXED)

---

## MVU-016: FRM Core Memory & Sync
**Completed**: 2026-01-20
**Scope**: BAR_SYNC, MEMBAR, CTAState for block-level synchronization
**Tools**: tools/gpu_simulator.py
**Features**:
- CTAState class for CTA/block-level state tracking
- barrier_arrive() method for tracking thread arrivals at barriers
- flush_writes() for membar memory ordering
- BAR_SYNC handler: barrier_id, thread_count, warp stall support
- MEMBAR handler: scope (CTA/GL/SYS), sequencing point
- WarpState.at_barrier flag for multi-warp scheduling
**Tests**: 75/75 pass (existing tests continue to pass)
**Verification**: RTL-FRM comparison + Gemini review (PASS)
**Process**: Direct implementation, Gemini reviewed (PASS after enhancement)

---

## MVU-017: Test Generators for Branch/Memory
**Completed**: 2026-01-20
**Scope**: Test generators for LD/ST (global/shared) and branch operations
**Tools**: tools/test_generator.py, tools/ptx_assembler.py
**Features**:
- MemoryTestGenerator: gen_ld_st_global_tests(), gen_ld_st_shared_tests()
- BranchTestGenerator: gen_unconditional_branch_tests(), gen_conditional_setp_tests()
- Assembler fix: Branch offset now in instruction count (not bytes)
- Assembler fix: Predicated branches set rd[4]=1 for FRM compatibility
**Tests**: 103/103 pass (65 ALU + 10 FP32 + 20 Memory + 8 Branch)
**Verification**: RTL-FRM comparison + Gemini review (PASS)
**Process**: Implementation + bug fixes, Gemini reviewed (PASS)

---

## Regression Summary
- All archived MVUs continue to pass in regression testing
- Total unique test cases: 375+
- Coverage: Tier 1-7 RTL, Tier 1/3 verified with generators
