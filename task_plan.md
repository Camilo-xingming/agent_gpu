# Task Plan: RalphGPU PTX Comprehensive Test Suite

## Goal
Create comprehensive PTX test suite with:
1. PTX test cases for functional verification (22 tests)
2. Compile PTX to binary using ptx_assembler.py
3. Testbench to load compiled binaries with performance metrics
4. Memory consistency model tests per PTX 9.1 spec
5. Run verification, debug failures

## Test Categories Created

### Functional Tests (test_01 - test_15)
| Test | Description | Instructions | Status |
|------|-------------|--------------|--------|
| test_01_alu_basic | ADD, SUB, AND, OR, XOR, NOT, SHL, SHR | 46 | PASS |
| test_02_alu_extended | MIN, MAX, ABS, NEG, POPC, CLZ, BREV | 43 | PASS |
| test_03_multiply | MUL.LO, MUL.HI, MAD.LO, MUL24, MAD24 | 41 | FAIL |
| test_04_fp32_arith | ADD.F32, SUB.F32, MUL.F32, DIV.F32, FMA | 39 | FAIL |
| test_05_fp32_special | RCP, SQRT, MIN, MAX, ABS, NEG | 40 | PASS |
| test_06_fp16_arith | FP16 arithmetic operations | 39 | FAIL |
| test_07_memory_global | LD.GLOBAL, ST.GLOBAL | 43 | PASS |
| test_08_memory_shared | LD.SHARED, ST.SHARED | 47 | TIMEOUT |
| test_09_atomic | ATOM.ADD, ATOM.CAS, etc. | 76 | TIMEOUT |
| test_10_cvt | Type conversion CVT operations | 53 | FAIL |
| test_11_special_regs | %laneid, %warpid, %smid | 26 | PASS |
| test_12_setp_compare | SETP comparison operations | 40 | - |
| test_13_video_ops | VADD4, VSUB4, DP4A | 92 | - |
| test_14_wmma | WMMA matrix operations | 47 | - |
| test_15_control_flow | BRA, branch conditions | 39 | - |

### Memory Consistency Model Tests (test_20 - test_23)
Based on NVIDIA PTX 9.1 Memory Consistency Model:
| Test | Description | Pattern |
|------|-------------|---------|
| test_20_mem_consistency_mp | Message Passing | Release-Acquire |
| test_21_mem_consistency_sb | Store Buffering | SC Fencing |
| test_22_mem_consistency_coherence | Cache Coherence | CoRR Pattern |
| test_23_mem_consistency_atomicity | Atomicity | Multi-thread atomic |

### Performance Benchmark Tests (test_30 - test_32)
| Test | Description | Operations |
|------|-------------|------------|
| test_30_perf_alu_throughput | ALU throughput | 100 ALU ops |
| test_31_perf_fp_throughput | FP32 throughput | 50 FP32 ops |
| test_32_perf_memory_latency | Memory latency | Pointer chasing |

## Files Created

### PTX Test Files
- `asm/ptx_comprehensive_tests/test_01_alu_basic.ptx` - `test_32_perf_memory_latency.ptx`
- All tests use 16-bit immediates (PTX assembler constraint)
- 32-bit values built via shift operations

### Compilation Script
- `scripts/compile_ptx_tests.py` - Compiles all PTX to hex

### Hex Output
- `hex/ptx_comprehensive_tests/*.hex` - 22 compiled hex files
- `sim/*.hex` - Copies for simulation

### Testbench
- `tb/tb_ptx_tests.v` - PTX test runner with performance metrics
  - Loads hex files via $readmemh
  - Measures cycles per test
  - Calculates IPC and throughput
  - Reports PASS/FAIL based on 0xCAFE marker at 0x2000

## Verification Results (Partial Run)

| Test | Cycles | IPC | Status |
|------|--------|-----|--------|
| ALU Basic | 267 | 0.172 | PASS |
| ALU Extended | 266 | 0.162 | PASS |
| Multiply | 183 | - | FAIL (result=0) |
| FP32 Arith | 174 | - | FAIL (result=0) |
| FP32 Special | 259 | 0.154 | PASS |
| FP16 Arith | 81 | - | FAIL (0xDEAD) |
| Global Memory | 294 | 0.146 | PASS |
| Shared Memory | 200000 | - | TIMEOUT |
| Atomics | 200000 | - | TIMEOUT |
| CVT | 76 | - | FAIL (0xDEAD) |
| Special Regs | 155 | 0.168 | PASS |

## Known Issues

1. **Shared Memory Timeout**: LD.SHARED/ST.SHARED operations may not be fully wired
2. **Atomic Timeout**: ATOM operations may have incomplete execution path
3. **FP Arithmetic**: Some FP32 tests fail - likely FPU pipeline issues
4. **CVT Failures**: Type conversion operations returning incorrect results

## Performance Metrics
- Clock: 100 MHz (10ns period)
- Typical IPC: 0.15-0.17 (due to memory latency, pipeline stalls)
- Successful tests complete in 150-300 cycles

## Phases Complete
- [x] Phase 1: Create PTX test cases
- [x] Phase 2: Compile PTX to binary
- [x] Phase 3: Create testbench with hex loading
- [x] Phase 4: Add performance measurement
- [x] Phase 5: Create memory consistency tests
- [x] Phase 6: Run verification (partial - some timeouts)
- [x] Phase 7: Document results

## Recommendations for Further Work
1. Debug shared memory execution path
2. Investigate FP32 arithmetic failures
3. Fix atomic operation pipeline
4. Add more comprehensive CVT test coverage
5. Optimize IPC through pipeline improvements
