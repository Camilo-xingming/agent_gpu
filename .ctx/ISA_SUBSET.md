# RalphGPU PTX ISA Subset

## Implemented Opcodes

### Integer Arithmetic
| Opcode | Variants | Status |
|--------|----------|--------|
| add | .s32, .u32 | ✅ |
| sub | .s32, .u32 | ✅ |
| mul | .lo.s32, .lo.u32, .hi.s32, .hi.u32, .wide.s32, .wide.u32 | ✅ |
| mad | .lo.s32, .lo.u32, .hi.s32 | ✅ |
| mul24 | .lo.s32, .lo.u32 | ✅ |
| mad24 | .lo.s32, .lo.u32 | ✅ |
| div | .s32, .u32 | ✅ |
| rem | .s32, .u32 | ✅ |
| abs | .s32 | ✅ |
| neg | .s32 | ✅ |
| min | .s32, .u32 | ✅ |
| max | .s32, .u32 | ✅ |

### Bitwise Operations
| Opcode | Variants | Status |
|--------|----------|--------|
| and | .b32 | ✅ |
| or | .b32 | ✅ |
| xor | .b32 | ✅ |
| not | .b32 | ✅ |
| shl | .b32 | ✅ |
| shr | .u32, .s32 | ✅ |
| bfe | .s32, .u32 | ✅ |
| bfi | .b32 | ✅ |
| brev | .b32 | ✅ |
| popc | .b32 | ✅ |
| clz | .b32 | ✅ |
| fns | .b32 | ✅ |
| lop3 | .b32 | ✅ |
| shf | .l.clamp.b32, .r.clamp.b32 | ✅ |

### Comparison and Predicate
| Opcode | Variants | Status |
|--------|----------|--------|
| setp | .eq, .ne, .lt, .le, .gt, .ge (.s32, .u32, .f32) | ✅ |
| selp | .s32, .u32, .f32 | ✅ |

### Data Movement
| Opcode | Variants | Status |
|--------|----------|--------|
| mov | .b32, .u32, .s32, .f32 | ✅ |
| ld.global | .b32, .b64, .v2, .v4 | ✅ |
| st.global | .b32, .b64, .v2, .v4 | ✅ |
| ld.shared | .b32 | ✅ |
| st.shared | .b32 | ✅ |
| ld.param | .b32, .b64 | ✅ |
| ld.const | .b32 | ✅ |
| ld.local | .b32 | ✅ |
| st.local | .b32 | ✅ |

### Floating Point (FP32)
| Opcode | Variants | Status |
|--------|----------|--------|
| add | .f32 | ✅ |
| sub | .f32 | ✅ |
| mul | .f32 | ✅ |
| fma | .rn.f32 | ✅ |
| div | .approx.f32, .rn.f32 | ✅ |
| rcp | .approx.f32 | ✅ |
| sqrt | .approx.f32, .rn.f32 | ✅ |
| rsqrt | .approx.f32 | ✅ |
| abs | .f32 | ✅ |
| neg | .f32 | ✅ |
| min | .f32 | ✅ |
| max | .f32 | ✅ |
| copysign | .f32 | ✅ |
| testp | .f32 | ✅ |

### Special Functions (SFU)
| Opcode | Variants | Status |
|--------|----------|--------|
| sin | .approx.f32 | ✅ |
| cos | .approx.f32 | ✅ |
| ex2 | .approx.f32 | ✅ |
| lg2 | .approx.f32 | ✅ |

### Floating Point (FP64)
| Opcode | Variants | Status |
|--------|----------|--------|
| add | .f64 | ✅ |
| sub | .f64 | ✅ |
| mul | .f64 | ✅ |
| fma | .rn.f64 | ✅ |
| div | .rn.f64 | ✅ |
| rcp | .approx.f64 | ✅ |

### Floating Point (FP16)
| Opcode | Variants | Status |
|--------|----------|--------|
| add | .f16, .f16x2 | ✅ |
| sub | .f16, .f16x2 | ✅ |
| mul | .f16, .f16x2 | ✅ |
| fma | .f16, .f16x2 | ✅ |

### Type Conversion
| Opcode | Variants | Status |
|--------|----------|--------|
| cvt | .f32.s32, .s32.f32, .f32.f16, .f16.f32, etc. | ✅ |

### Control Flow
| Opcode | Variants | Status |
|--------|----------|--------|
| bra | (unconditional, predicated) | ✅ |
| call | .uni | ✅ |
| ret | | ✅ |
| exit | | ✅ |
| @p | (predication) | ✅ |

### Synchronization
| Opcode | Variants | Status |
|--------|----------|--------|
| bar.sync | | ✅ |
| bar.warp.sync | | ✅ |
| membar | .cta, .gl, .sys | ✅ |
| barrier.cluster | .arrive, .wait, .sync | ✅ |

### Warp-Level
| Opcode | Variants | Status |
|--------|----------|--------|
| shfl.sync | .up, .down, .bfly, .idx | ✅ |
| vote.sync | .all, .any, .uni, .ballot | ✅ |
| redux.sync | .add, .min, .max, .and, .or, .xor | ✅ |
| match.sync | .any, .all | ✅ |

### Atomic Operations
| Opcode | Variants | Status |
|--------|----------|--------|
| atom | .add, .cas, .exch, .min, .max, .and, .or, .xor | ✅ |
| red | .add, .min, .max | ✅ |

### Tensor Core
| Opcode | Variants | Status |
|--------|----------|--------|
| wmma.load | .a, .b, .c | ✅ |
| wmma.store | .d | ✅ |
| wmma.mma | .sync.aligned | ✅ |
| mma | .sync.aligned | ✅ |
| wgmma.load | | ✅ |
| wgmma.store | | ✅ |
| wgmma.mma | | ✅ |
| wgmma.fence | | ✅ |
| wgmma.commit_group | | ✅ |
| wgmma.wait_group | | ✅ |

### Video/DPX
| Opcode | Variants | Status |
|--------|----------|--------|
| dp4a | .s32.s32, .u32.u32 | ✅ |
| dp2a | .s32.s32, .u32.u32 | ✅ |
| vadd | | ✅ |
| vsub | | ✅ |
| vabsdiff | | ✅ |
| vmin | | ✅ |
| vmax | | ✅ |

### Async Copy (Hopper+)
| Opcode | Variants | Status |
|--------|----------|--------|
| cp.async | .ca, .cg | ✅ |
| cp.async.commit_group | | ✅ |
| cp.async.wait_group | | ✅ |
| st.async | .global, .shared | ✅ |
| mbarrier.init | | ✅ |
| mbarrier.arrive | | ✅ |
| mbarrier.test_wait | | ✅ |
| mbarrier.try_wait | | ✅ |

### Texture/Surface
| Opcode | Variants | Status |
|--------|----------|--------|
| tex | .1d, .2d | ✅ |
| txq | | ✅ |
| suld | | ✅ |
| sust | | ✅ |
| sured | | ✅ |

### Cache Policy (Blackwell)
| Opcode | Variants | Status |
|--------|----------|--------|
| createpolicy | | ✅ |
| applypriority | | ✅ |
| discard | | ✅ |

### Debug/Misc
| Opcode | Variants | Status |
|--------|----------|--------|
| brkpt | | ✅ |
| trap | | ✅ |
| pmevent | | ✅ |
| nanosleep | | ✅ |
| alloca | | ✅ |
| stacksave | | ✅ |
| stackrestore | | ✅ |
| setmaxnreg | | ✅ |

## Special Registers
| Register | Description | Status |
|----------|-------------|--------|
| %tid.x/y/z | Thread ID within block | ✅ |
| %ntid.x/y/z | Block dimensions | ✅ |
| %ctaid.x/y/z | Block ID within grid | ✅ |
| %nctaid.x/y/z | Grid dimensions | ✅ |
| %laneid | Lane within warp | ✅ |
| %warpid | Warp ID within SM | ✅ |
| %smid | SM ID | ✅ |
| %activemask | Active lane mask | ✅ |
| %clock | Cycle counter | ✅ |
