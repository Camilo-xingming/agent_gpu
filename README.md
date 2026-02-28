# RalphGPU - CUDA/PTX Compatible GPU IP

[![CI](https://github.com/ssql2014/RalphGPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ssql2014/RalphGPU/actions/workflows/ci.yml) [![Nightly](https://github.com/ssql2014/RalphGPU/actions/workflows/nightly.yml/badge.svg)](https://github.com/ssql2014/RalphGPU/actions/workflows/nightly.yml)

An open-source, synthesizable GPU IP core compatible with NVIDIA's CUDA/PTX execution model. 63 RTL modules, ~36K lines of SystemVerilog.

## Features

### Compute
- **SIMT Architecture**: 32-thread warps, CUDA-compatible execution model
- **Dual-Issue / Blackwell Multi-Scheduler**: 2-way (default) or 4-way (HPC) issue with I-Buffer
- **ALU**: add/sub/mul/mad/div/rem, logic, shifts, min/max, popc/clz
- **FP32**: add/sub/mul/div/fma/neg/abs/min/max
- **FP16/BF16**: add/sub/mul/fma/neg/abs/min/max (fp16_unit)
- **FP64**: add/sub/mul/div/fma/neg/abs/min/max (fpu64)
- **SFU**: sin/cos/rcp/sqrt/rsqrt/lg2/ex2/tanh
- **CVT**: f32↔s32, f32↔f16, f32↔f64, f64↔s64/u64
- **Tensor Core**: WGMMA (m64n8-256k16), sparse MMA, tile engine
- **DPX/Video**: dp4a, dp2a, vibmatch

### Memory
- **L1 Data Cache**: 16KB (Lite) / 64KB (Balanced), write-back, configurable policies
- **L2 Cache**: shared across SMs with interconnect
- **I-Cache**: instruction fetch pipeline with prefetch
- **Shared Memory**: 16–96KB per SM, banked
- **Memory Coalescing**: burst coalescing unit for global memory
- **TLB**: enhanced TLB with page table walk
- **HBM Controller**: multi-channel HBM2e with scheduling and QoS
- **Atomic Unit**: add/exch/cas/and/or/xor/min/max on global memory
- **cp.async**: async shared←global copy engine (CA/CG/BULK)
- **TMA**: tensor memory accelerator for bulk tensor data movement

### Control Flow
- **Warp Scheduler**: advanced scheduler with scoreboard + dependency tracking
- **Branch Predictor**: warp-level branch prediction
- **Reconvergence Stack**: SIMT stack for divergence/reconvergence
- **Warp Collectives**: SHFL (idx/up/down/bfly), VOTE (all/any/uni/ballot), REDUX
- **Barrier**: bar.sync with CTA-level tracking, mbarrier (Hopper-style)
- **Cluster Barrier**: cross-SM synchronization
- **Grid Dependencies**: griddepcontrol for multi-kernel orchestration

### Infrastructure
- **Performance Counters**: cycle/instruction/stall tracking
- **LZ4 Decompressor**: hardware decompression
- **CHI Controller**: AMBA CHI interface
- **Multi-Mem**: multicast store/reduce across SMs
- **Configurable Profiles**: Lite / Balanced / HPC

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        RalphGPU Top                              │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐      │
│  │     SM 0     │  │     SM 1     │  ...  │    SM N-1    │      │
│  │              │  │              │       │              │      │
│  │ Multi-Sched  │  │ Multi-Sched  │       │ Multi-Sched  │      │
│  │ I-Cache      │  │ I-Cache      │       │ I-Cache      │      │
│  │ Decoder ×2   │  │ Decoder ×2   │       │ Decoder ×2   │      │
│  │ SIMD ALU     │  │ SIMD ALU     │       │ SIMD ALU     │      │
│  │ FP32/FP16/64 │  │ FP32/FP16/64 │       │ FP32/FP16/64 │      │
│  │ SFU / CVT    │  │ SFU / CVT    │       │ SFU / CVT    │      │
│  │ Tensor Core  │  │ Tensor Core  │       │ Tensor Core  │      │
│  │ Reg File     │  │ Reg File     │       │ Reg File     │      │
│  │ Shared Mem   │  │ Shared Mem   │       │ Shared Mem   │      │
│  │ L1 D-Cache   │  │ L1 D-Cache   │       │ L1 D-Cache   │      │
│  │ TLB          │  │ TLB          │       │ TLB          │      │
│  └──────┬───────┘  └──────┬───────┘       └──────┬───────┘      │
│         │                  │                      │              │
│  ┌──────┴──────────────────┴──────────────────────┴──────┐       │
│  │              L2 Cache + Interconnect                   │       │
│  ├───────────────────────────────────────────────────────┤       │
│  │         Memory Coalescing + HBM Controller            │       │
│  └───────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

## Configuration Profiles

| Parameter | Lite | Balanced | HPC |
|-----------|------|----------|-----|
| NUM_SM | 2 | 8 | 16 |
| WARPS_PER_SM | 4 | 8 | 16 |
| SHARED_MEM_KB | 16 | 64 | 96 |
| Issue Width | 1 | 1 | 2 |
| Schedulers | 2 | 2 | 4 |
| Sched Lanes | 2 | 2 | 4 |
| Threads | 256 | 2,048 | 8,192 |

```bash
make GPU_PROFILE=LITE      # Small FPGA targets
make GPU_PROFILE=BALANCED  # Default
make GPU_PROFILE=HPC       # Maximum throughput
```

## PTX Instruction Set

### Integer
```
add.s32 / sub.s32 / mul.lo.s32 / mad.lo.s32 / div.s32 / div.u32 / rem.s32 / rem.u32
and.b32 / or.b32 / xor.b32 / not.b32 / shl.b32 / shr.u32 / shr.s32
min.s32 / min.u32 / max.s32 / max.u32 / popc / clz / abs / neg
```

### Floating-Point
```
add.f32 / sub.f32 / mul.f32 / div.f32 / fma.rn.f32         # FP32
add.f16 / sub.f16 / mul.f16 / fma.rn.f16                   # FP16
add.f64 / sub.f64 / mul.f64 / div.f64 / fma.rn.f64         # FP64
sin / cos / rcp / sqrt / rsqrt / lg2 / ex2 / tanh           # SFU
cvt.s32.f32 / cvt.f32.s32 / cvt.f32.f16 / cvt.f16.f32 ... # CVT
```

### Memory
```
ld.global.s32 / st.global.s32          # Global memory
ld.shared.s32 / st.shared.s32          # Shared memory
ld.param.s32 / ld.const.s32            # Parameter / constant
atom.global.add / atom.global.cas ...   # Atomics
cp.async.ca.shared.global              # Async copy
```

### Control Flow & Warp
```
bra / setp / @p bra                    # Branch + predicated
bar.sync / membar.cta / membar.gl      # Barriers
shfl.sync.idx / up / down / bfly       # Shuffle
vote.sync.all / any / uni / ballot     # Vote
redux.sync.add / min / max / and ...   # Reduction
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE_GUIDE.md) - Detailed technical overview of RalphGPU internal structure.
- [ISA Reference](docs/ISA_REFERENCE.md) - Complete instruction set reference and encoding details.
- [PTX Compatibility](docs/PTX_COMPATIBILITY.md) - Status of NVIDIA PTX instruction support.

## Verification

### Tools
| Tool | Description |
|------|-------------|
| `tools/gpu_simulator.py` | Functional Reference Model (FRM) — Python ISA simulator |
| `tools/rtl_frm_compare.py` | RTL vs FRM comparison framework |
| `tools/test_generator.py` | Automated test generation (ALU/FP/memory/branch/...) |
| `tools/ptx_assembler.py` | PTX assembler → hex |
| `tools/perf_report.py` | Performance analysis |

### Test Coverage
```
RTL-FRM Comparison:  204/204 pass
  ALU:     60    FP32:   10    FP16:   15    FP64:    9
  CVT:     12    SFU:    21    DIV:    32    Memory: 20
  Branch:   8    Atom:    5    Sync:    3    Membar:  3
  Param:    2    Special: 4

FRM Self-Tests:      10/10 pass
  ALU, vector_add, FP32, SHFL, VOTE, REDUX, cp.async, FP16, FP64, CVT
```

### Running Tests
```bash
make test                              # Quick regression (iverilog)
make lint                              # Verilator lint
make synth                             # Yosys synthesis
make test_tensor_fp4_fp8_frm           # Generated FP4/FP8 Tensor Core RTL vs FRM e2e
python3 tools/rtl_frm_compare.py --all # RTL vs FRM (204 tests)
python3 tools/gpu_simulator.py         # FRM self-tests (10 tests)
python3 tools/test_generator.py --gen all  # Regenerate test vectors
```

## Directory Structure

```
RalphGPU/
├── rtl/                    # 63 RTL modules (~36K lines)
│   ├── gpu_defines.vh      # Global defines and parameters
│   ├── gpu_config.vh       # Profile configuration
│   ├── streaming_multiprocessor_v2.v  # SM top-level
│   ├── ralph_gpu_top.v     # GPU top-level
│   ├── alu.v / mul_unit.v / fpu.v / fpu64.v / fp16_unit.v / sfu.v / cvt_unit.v
│   ├── tensor_core.v / wgmma.v / wgmma_tile_engine.v
│   ├── decoder.v / blackwell_scheduler.v / dual_issue_scheduler.v
│   ├── l1_data_cache.v / l2_cache.v / icache.v / tlb.v
│   ├── memory_coalescing_unit.v / memory_controller_hbm.v
│   ├── warp_shuffle.v / warp_collective_unit.v / reconvergence_stack.v
│   └── ...
├── tb/                     # Testbenches
├── tools/                  # Python verification tools
├── build/                  # Build artifacts + generated tests
├── examples/               # PTX example programs
├── docs/                   # Documentation
├── .ctx/                   # Project status tracking
├── .github/workflows/      # CI (lint + test)
└── Makefile               # Build system
```

## Quick Start

### Dependencies
- **Icarus Verilog** or **Verilator** (simulation/lint)
- **Yosys** (synthesis)
- **Python 3** (assembler, FRM, test tools)
- **GTKWave** (optional, waveform viewing)

### Build & Run
```bash
git clone https://github.com/ssql2014/RalphGPU.git
cd RalphGPU
make test      # Run regression tests
make lint      # Verilator lint check
make synth     # Yosys synthesis
```

## License

MIT License
