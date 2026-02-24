# RalphGPU ISA Reference (Hopper/Blackwell-Class)

## 1. Instruction Encoding

RalphGPU uses a fixed-length 32-bit instruction encoding.

```
| 31-26  | 25-21  | 20-16  | 15-11  | 10-6   | 5-0    |
| OPCODE |  RD    |  RA    |  RB    |  RC/P  | FUNC   |
```

*   **OPCODE (6 bits)**: Primary instruction operation code.
*   **RD (5 bits)**: Destination register index.
*   **RA (5 bits)**: First source register index.
*   **RB (5 bits)**: Second source register index (or immediate value, depending on instruction).
*   **RC/P (5 bits)**: Third source register index or Predicate register index.
*   **FUNC (6 bits)**: Sub-operation or modifier (e.g., specific ALU operation type).

**Immediate Fields**: 
*   **IMM16 (16 bits)**: [15:0] for immediate ALU/Memory operations.
*   **IMM21 (21 bits)**: [20:0] for branch offsets.

---

## 2. Operands and Registers

### 2.1 General Purpose Registers (GPRs)
*   **%r0 - %r31**: 32-bit registers per thread. %r0 is usually used as a zero-register by convention but is not hard-wired.

### 2.2 Predicate Registers
*   **%p0 - %p7**: 1-bit registers used for conditional execution (predication).

### 2.3 Special Registers (Read-only)
*   **%tid.x, %tid.y, %tid.z**: Thread ID within a block.
*   **%ctaid.x, %ctaid.y, %ctaid.z**: Block ID within a grid.
*   **%ntid.x, %ntid.y, %ntid.z**: Block dimensions.
*   **%nctaid.x, %nctaid.y, %nctaid.z**: Grid dimensions.
*   **%laneid**: Thread index within a warp (0-31).
*   **%warpid**: Warp index within an SM.
*   **%smid**: Streaming Multiprocessor ID.
*   **%activemask**: Mask of currently active threads in the warp.
*   **%clock64**: 64-bit cycle counter.

---

## 3. Instruction Set Summary

### 3.1 Integer Arithmetic & Logic (OPCODE 000000)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| add.s32     | rd = ra + rb | 000000 |
| sub.s32     | rd = ra - rb | 000001 |
| and.b32     | rd = ra & rb | 000010 |
| or.b32      | rd = ra | rb | 000011 |
| xor.b32     | rd = ra ^ rb | 000100 |
| not.b32     | rd = ~ra | 000101 |
| shl.b32     | rd = ra << rb | 000110 |
| shr.u32     | rd = ra >> rb (logical) | 000111 |
| shr.s32     | rd = ra >> rb (arithmetic) | 001000 |
| abs.s32     | rd = |ra| | 001001 |
| min.s32/u32 | rd = min(ra, rb) | 001011/001100 |
| max.s32/u32 | rd = max(ra, rb) | 001101/001110 |
| popc.b32    | Population count | 001111 |
| clz.b32     | Count leading zeros | 010000 |
| bfind.s32   | Bit find | 010001 |
| brev.b32    | Bit reverse | 010010 |
| bfe.s32/u32 | Bit field extract | 010011/010100 |
| bfi.b32     | Bit field insert | 010101 |
| selp.b32    | Predicate selection | 011000 |

### 3.2 Multiplication & Multiply-Add (OPCODE 000001)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| mul.lo.s32  | 32-bit mul (low bits) | 000000 |
| mul.hi.s32  | 32-bit mul (high bits) | 000001 |
| mad.lo.s32  | rd = ra * rb + rc | 000010 |
| mad.hi.s32  | rd = ra * rb + rc (high) | 000011 |
| mul24.lo.s32| 24-bit mul (low) | 000100 |
| mad24.lo.s32| 24-bit mad (low) | 000101 |

### 3.3 Division & Remainder (OPCODE 000010)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| div.s32     | Signed division | 000000 |
| div.u32     | Unsigned division | 000001 |
| rem.s32     | Signed remainder | 000010 |
| rem.u32     | Unsigned remainder | 000011 |

### 3.4 Floating Point (FP32) (OPCODE 001101)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| add.f32     | Floating-point add | 000000 |
| sub.f32     | Floating-point sub | 000001 |
| mul.f32     | Floating-point mul | 000010 |
| div.f32     | Floating-point div | 000011 |
| fma.f32     | Fused multiply-add | 000100 |
| neg.f32     | Floating-point neg | 000101 |
| abs.f32     | Floating-point abs | 000110 |
| min.f32     | Floating-point min | 000111 |
| max.f32     | Floating-point max | 001000 |

### 3.5 Floating Point Special Functions (SFU) (OPCODE 001110)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| rcp.f32     | Reciprocal (1/x) | 000000 |
| sqrt.f32    | Square Root | 000001 |
| rsqrt.f32   | Reciprocal Square Root | 000010 |
| sin.f32     | Sine | 000011 |
| cos.f32     | Cosine | 000100 |
| lg2.f32     | log2(x) | 000101 |
| ex2.f32     | 2^x | 000110 |

### 3.6 Floating Point (FP16/BF16) (OPCODE 010000)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| add.f16     | FP16 add | 000000 |
| add.bf16    | BF16 add | 010000 |
| add.f16x2   | Packed FP16 add | 100000 |

### 3.7 Floating Point (FP64) (OPCODE 001111)
| Instruction | Description | FUNC |
|-------------|-------------|------|
| add.f64     | FP64 add | 000000 |
| mul.f64     | FP64 mul | 000010 |
| fma.f64     | FP64 fma | 000100 |

### 3.8 Memory Operations
| OPCODE | Instruction | Description |
|--------|-------------|-------------|
| 000101 | ld.global   | Load from Global Memory |
| 000110 | st.global   | Store to Global Memory |
| 000111 | ld.shared   | Load from Shared Memory |
| 001000 | st.shared   | Store to Shared Memory |
| 010010 | ld.param    | Load kernel parameters |
| 010110 | ld.v2       | Vector load (2 elements) |
| 010111 | ld.v4       | Vector load (4 elements) |

### 3.9 Atomics & Reductions
| OPCODE | Instruction | Description |
|--------|-------------|-------------|
| 011010 | atom        | Atomic operations (add/min/max/exch/cas) |
| 011011 | red         | Global reduction operations |

### 3.10 Warp-Level Operations
| OPCODE | Instruction | Description |
|--------|-------------|-------------|
| 011100 | shfl.sync   | Warp shuffle (idx/up/down/bfly) |
| 011101 | vote.sync   | Warp vote (all/any/uni/ballot) |
| 011110 | redux.sync  | Warp-level reduction |

### 3.11 Tensor Core Operations
| OPCODE | Instruction | Description |
|--------|-------------|-------------|
| 011111 | wmma.load   | WMMA load matrix (M16N16K16, etc.) |
| 100000 | wmma.store  | WMMA store result matrix |
| 100001 | wmma.mma    | WMMA perform MMA |
| 100010 | mma         | Hopper/Blackwell mma / sparse mma |
| 101011 | cp.async    | Async copy global -> shared |
| 101111 | wgmma.mma   | Warp Group MMA (Hopper/Blackwell) |

### 3.12 Control Flow
| OPCODE | Instruction | Description |
|--------|-------------|-------------|
| 000100 | bra         | Branch (conditional/unconditional) |
| 100011 | call        | Function call |
| 001100 | ret         | Return from function |
| 001011 | exit        | Terminate thread |
| 001010 | bar.sync    | Block-level synchronization |

---

## 4. Cache Policies & Hints

Memory instructions support 3-bit cache hints:
*   **ca**: Cache at all levels (default).
*   **cg**: Cache at global level only (L2).
*   **cs**: Streaming (evict soon).
*   **lu**: Last use.
*   **cv**: Volatile.
*   **wb**: Write-back (for stores).
*   **wt**: Write-through (for stores).

---

## 5. Predication

Most instructions can be conditionally executed based on a predicate register. In assembly, this is denoted by `@%pX` before the instruction.
Example:
```
@%p0 add.s32 %r1, %r2, %r3;  // Executes only if %p0 is true
```

---

## 6. Blackwell/Hopper Extensions

### 6.1 TCGEN05 (SM100+)
Blackwell introduces per-thread asynchronous MMA via TMEM.
*   **tcgen05.mma**: Async MMA with TMEM accumulator.
*   **tcgen05.ld/st**: Move data between registers and TMEM.
*   **tcgen05.alloc/dealloc**: Manage TMEM space.

### 6.2 mbarrier (Hopper+)
*   **mbarrier.init/arrive/wait**: Hardware-accelerated barriers for async operations.

### 6.3 2:4 Structured Sparsity
*   **mma.sp**: Sparse matrix multiply-accumulate with 50% theoretical speedup.
