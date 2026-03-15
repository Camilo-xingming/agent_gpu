//============================================================================
// RalphGPU - CUDA/PTX Compatible GPU IP
// 全局定义和参数
//============================================================================

`ifndef GPU_DEFINES_VH
`define GPU_DEFINES_VH

`include "gpu_config.vh"
`include "memory_config.vh"

//============================================================================
// 可配置参数 - 修改这些参数来扩展算力
//============================================================================
`ifndef NUM_SM
`ifdef GPU_PROFILE_LITE
`define NUM_SM              2       // LITE: 2 SM
`elsif GPU_PROFILE_BALANCED
`define NUM_SM              8       // BALANCED: 8 SM
`elsif GPU_PROFILE_HPC
`define NUM_SM              16      // HPC: 16 SM (scale-up option)
`else
`define NUM_SM              2       // 默认: 保持现有配置
`endif
`endif

`ifndef THREADS_PER_WARP
`define THREADS_PER_WARP    32      // 每Warp线程数
`endif

`ifndef WARPS_PER_SM
`ifdef GPU_PROFILE_LITE
`define WARPS_PER_SM        4       // LITE: 4 warps/SM
`elsif GPU_PROFILE_BALANCED
`define WARPS_PER_SM        8       // BALANCED: 8 warps/SM
`elsif GPU_PROFILE_HPC
`define WARPS_PER_SM        16      // HPC: 16 warps/SM
`else
`define WARPS_PER_SM        4       // 默认: 4 warps/SM
`endif
`endif

`ifndef NUM_REGS
`define NUM_REGS            32      // 每线程寄存器数
`endif

`ifndef SHARED_MEM_KB
`ifdef GPU_PROFILE_LITE
`define SHARED_MEM_KB       16      // LITE: 16KB
`elsif GPU_PROFILE_BALANCED
`define SHARED_MEM_KB       64      // BALANCED: 64KB
`elsif GPU_PROFILE_HPC
`define SHARED_MEM_KB       96      // HPC: 96KB
`else
`define SHARED_MEM_KB       16      // 默认: 16KB
`endif
`endif

`ifndef DATA_WIDTH
`define DATA_WIDTH          32      // 数据位宽
`endif

// 调度/发射宽度
`ifndef SM_ISSUE_WIDTH
`ifdef GPU_PROFILE_HPC
`define SM_ISSUE_WIDTH      2
`else
`define SM_ISSUE_WIDTH      1
`endif
`endif

// Blackwell-style Multi-Scheduler Parameters
`ifndef NUM_SCHEDULERS
`ifdef GPU_PROFILE_HPC
`define NUM_SCHEDULERS      4       // Blackwell: 4 parallel schedulers
`else
`define NUM_SCHEDULERS      2       // Default: 2 schedulers
`endif
`endif

`ifndef IBUFFER_DEPTH
`define IBUFFER_DEPTH       4       // Instructions per warp I-Buffer (Blackwell: 4+)
`endif

// Scheduler Selection: 1 = Blackwell multi-scheduler, 0 = Advanced dual-issue
`ifndef USE_BLACKWELL_SCHEDULER
`define USE_BLACKWELL_SCHEDULER 1   // Default: Use Blackwell-style scheduler
`endif

// Issue Pipeline Width: Number of instructions issued per cycle
// Note: 4-way requires GPU_PROFILE_HPC and additional pipeline resources
`ifndef SCHED_LANES
`ifdef GPU_PROFILE_HPC
`define SCHED_LANES             4   // Blackwell: 4-way issue
`else
`define SCHED_LANES             2   // Default: 2-way issue (dual-issue)
`endif
`endif

//============================================================================
// 派生参数 (自动计算)
//============================================================================
`define THREADS_PER_SM      (`THREADS_PER_WARP * `WARPS_PER_SM)
`define TOTAL_THREADS       (`THREADS_PER_SM * `NUM_SM)
`define REG_ADDR_WIDTH      5       // log2(32) = 5
`define WARP_ID_WIDTH       ((`WARPS_PER_SM > 1) ? $clog2(`WARPS_PER_SM) : 1)
`define SM_ID_WIDTH         ((`NUM_SM > 1) ? $clog2(`NUM_SM) : 1)
`define THREAD_ID_WIDTH     $clog2(`THREADS_PER_WARP)
`define SHARED_MEM_SIZE     (`SHARED_MEM_KB * 1024)
`define SHARED_MEM_ADDR_W   $clog2(`SHARED_MEM_SIZE)

//============================================================================
// 指令编码 - OPCODE (6-bit)
//============================================================================
`define OP_ALU          6'b000000   // ALU运算 (add/sub/and/or/xor/shl/shr)
`define OP_MUL          6'b000001   // 乘法 (mul.lo/mul.hi/mad)
`define OP_DIV          6'b000010   // 除法 (div/rem)
`define OP_SETP         6'b000011   // 比较设置谓词
`define OP_BRANCH       6'b000100   // 分支
`define OP_LD_GLOBAL    6'b000101   // 全局内存加载
`define OP_ST_GLOBAL    6'b000110   // 全局内存存储
`define OP_LD_SHARED    6'b000111   // 共享内存加载
`define OP_ST_SHARED    6'b001000   // 共享内存存储
`define OP_MOV_SPECIAL  6'b001001   // 特殊寄存器移动
`define OP_BAR_SYNC     6'b001010   // 同步屏障 (block-level)
`define OP_BAR_WARP_SYNC 6'b110011  // Warp-level sync (bar.warp.sync)
`define OP_EXIT         6'b001011   // Kernel退出
`define OP_RET          6'b001100   // 函数返回 (当前等同EXIT)

// Phase 2: 浮点运算指令
`define OP_FP32_ARITH   6'b001101   // FP32算术 (add/sub/mul/div/fma)
`define OP_FP32_SPECIAL 6'b001110   // FP32特殊函数 (rcp/sqrt/rsqrt/sin/cos/lg2/ex2)
`define OP_SFU          6'b001110   // SFU alias for special functions
`define OP_FP64_ARITH   6'b001111   // FP64算术
`define OP_FP16_ARITH   6'b010000   // FP16/BF16算术
`define OP_CVT          6'b010001   // 类型转换

// Phase 3: 扩展内存操作
`define OP_LD_PARAM     6'b010010   // Kernel参数加载
`define OP_LD_CONST     6'b010011   // 常量内存加载
`define OP_LD_LOCAL     6'b010100   // 本地内存加载
`define OP_ST_LOCAL     6'b010101   // 本地内存存储
`define OP_LD_V2        6'b010110   // 向量加载(2元素)
`define OP_LD_V4        6'b010111   // 向量加载(4元素)
`define OP_ST_V2        6'b011000   // 向量存储(2元素)
`define OP_ST_V4        6'b011001   // 向量存储(4元素)

// Phase 4: 原子操作
`define OP_ATOM         6'b011010   // 原子操作
`define OP_RED          6'b011011   // 归约操作

// Phase 5: Warp级操作
`define OP_SHFL         6'b011100   // Warp shuffle
`define OP_VOTE         6'b011101   // Warp vote
`define OP_REDUX        6'b011110   // Warp reduction

// Phase 6: Tensor Core
`define OP_WMMA_LOAD    6'b011111   // WMMA矩阵加载
`define OP_WMMA_STORE   6'b100000   // WMMA矩阵存储
`define OP_WMMA_MMA     6'b100001   // WMMA矩阵乘累加
`define OP_MMA          6'b100010   // MMA指令

// Phase 7: 控制流扩展
`define OP_CALL         6'b100011   // 函数调用
`define OP_MEMBAR       6'b100100   // 内存屏障

// Phase 8: Video Instructions
`define OP_VIDEO        6'b100101   // Video processing instructions

// Phase 9: Texture/Surface Instructions
`define OP_TEX          6'b100110   // Texture sampling
`define OP_TXQ          6'b100111   // Texture query
`define OP_SULD         6'b101000   // Surface load
`define OP_SUST         6'b101001   // Surface store
`define OP_SURED        6'b101010   // Surface reduction

// Phase 10: mbarrier Instructions (Hopper+)
`define OP_MBARRIER     6'b110010   // mbarrier operations

// Phase 4.1: Cache Policy Instructions (Hopper+)
`define OP_CACHE_POLICY 6'b110100   // Cache policy operations

// Phase 4.2: Address Space Query Instructions (PTX ISA)
// Note: These use func codes within OP_CACHE_POLICY opcode to avoid collisions
`define CACHE_ISSPACEP      6'b000100   // isspacep - Test address space membership (under OP_CACHE_POLICY)
`define CACHE_MAPA          6'b000101   // mapa - Map address between spaces (under OP_CACHE_POLICY)
`define CACHE_GETCTARANK    6'b000110   // getctarank - Get CTA rank in cluster (under OP_CACHE_POLICY)

// Phase 4.3: Bulk Store and Grid Dependency (Hopper+)
`define OP_ST_BULK      6'b100000   // st.bulk - Bulk store operations
`define OP_GRIDDEPCTRL  6'b100001   // griddepcontrol - Grid dependency control

// isspacep space type codes (used in src_b[2:0] with CACHE_ISSPACEP)
// 0=global, 1=shared, 2=local, 3=const, 4=param

// mapa map type codes (used in src_b[2:0] with CACHE_MAPA)
// 0=to_global, 1=to_shared, 2=from_shared, 3=to_local

// Legacy defines for backward compatibility (deprecated)
`define MAPA_TO_GLOBAL   6'b000000   // mapa.global - Map to global address
`define MAPA_TO_SHARED   6'b000001   // mapa.shared - Map to shared address
`define MAPA_FROM_SHARED 6'b000010   // mapa.to_generic - Map shared to generic pointer
`define MAPA_TO_LOCAL    6'b000011   // mapa.local - Map to local address

// st.bulk function codes (for OP_ST_BULK)
`define ST_BULK_GLOBAL        6'b000000   // st.bulk.global - Bulk store to global memory
`define ST_BULK_SHARED        6'b000001   // st.bulk.shared - Bulk store to shared memory
`define ST_BULK_COMMIT        6'b000010   // st.bulk.commit - Commit bulk store group
`define ST_BULK_WAIT          6'b000011   // st.bulk.wait - Wait for bulk store completion

// griddepcontrol function codes (for OP_GRIDDEPCTRL)
`define GRIDDEP_WAIT          6'b000000   // griddepcontrol.wait - Wait for grid dependency
`define GRIDDEP_LAUNCH_DEP    6'b000001   // griddepcontrol.launch_dependent - Launch dependent grid
`define GRIDDEP_SIGNAL        6'b000010   // griddepcontrol.signal - Signal grid completion
`define GRIDDEP_GET_TOKEN     6'b000011   // griddepcontrol.get_token - Get dependency token

// Phase 1.2: Async Store and Multimem Instructions (Hopper+)
`define OP_ST_ASYNC     6'b111000   // st.async - Async store operations
`define OP_MULTIMEM     6'b111001   // multimem - Multi-target write operations

// st.async function codes (for OP_ST_ASYNC)
`define ST_ASYNC_GLOBAL       6'b000000   // st.async.global - Async store to global memory
`define ST_ASYNC_SHARED       6'b000001   // st.async.shared - Async store to shared memory
`define ST_ASYNC_COMMIT       6'b000010   // cp.async.commit_group - Commit async store group
`define ST_ASYNC_WAIT         6'b000011   // cp.async.wait_group - Wait for async store group

// multimem function codes (for OP_MULTIMEM)
`define MULTIMEM_LD           6'b000000   // multimem.ld - Load from distributed shared memory
`define MULTIMEM_ST           6'b000001   // multimem.st - Multicast store to multiple SM shared memories
`define MULTIMEM_RED          6'b000010   // multimem.red - Multicast reduction

// Phase 3.2: Barrier Cluster Instructions (Hopper+)
`define OP_BARRIER_CLUSTER    6'b111010   // barrier.cluster - Cross-SM cluster synchronization

// barrier.cluster function codes (for OP_BARRIER_CLUSTER)
`define CLUSTER_BARRIER_ARRIVE   6'b000000   // barrier.cluster.arrive - Signal arrival at cluster barrier
`define CLUSTER_BARRIER_WAIT     6'b000001   // barrier.cluster.wait - Wait for all cluster members
`define CLUSTER_BARRIER_SYNC     6'b000010   // barrier.cluster.sync - Combined arrive + wait
`define CLUSTER_BARRIER_INIT     6'b000011   // barrier.cluster.init - Initialize cluster barrier

// Phase 5.1: Warp-level Collective Operations (Hopper+)
`define OP_MATCH_SYNC   6'b111011   // match.sync - Warp-level predicate matching
`define OP_ELECT_SYNC   6'b111100   // elect.sync - Warp-level leader election
`define OP_RED_ASYNC    6'b111101   // red.async - Async reduction to shared memory

// match.sync function codes (for OP_MATCH_SYNC)
`define MATCH_ANY       6'b000000   // match.sync.any - Match any thread with same value
`define MATCH_ALL       6'b000001   // match.sync.all - Match all threads must have same value

// elect.sync function codes (for OP_ELECT_SYNC)
`define ELECT_SYNC_ONE  6'b000000   // elect.sync.one - Elect one thread (leader election)

// red.async function codes (for OP_RED_ASYNC)
`define RED_ASYNC_ADD   6'b000000   // red.async.add - Async reduction add
`define RED_ASYNC_MIN   6'b000001   // red.async.min - Async reduction min
`define RED_ASYNC_MAX   6'b000010   // red.async.max - Async reduction max
`define RED_ASYNC_AND   6'b000011   // red.async.and - Async reduction bitwise AND
`define RED_ASYNC_OR    6'b000100   // red.async.or  - Async reduction bitwise OR
`define RED_ASYNC_XOR   6'b000101   // red.async.xor - Async reduction bitwise XOR

// Phase 5.2: DPX Instructions (Blackwell Dynamic Programming Extensions)
`define OP_DPX          6'b111110   // DPX operations - Dynamic programming accelerator

// DPX function codes (for OP_DPX)
`define DPX_VIADDMIN    6'b000000   // viaddmin - add with min for Viterbi/DTW
`define DPX_VIADDMAX    6'b000001   // viaddmax - add with max for sequence alignment
`define DPX_VIMINABS    6'b000010   // viminabs - min of absolute values
`define DPX_VIMAXABS    6'b000011   // vimaxabs - max of absolute values
`define DPX_VIADDMINMAX 6'b000100   // viaddminmax - add with both min and max
`define DPX_VIBMATCH    6'b000101   // vibmatch - bit match for pattern matching
`define DPX_VIBSET      6'b000110   // vibset - bit set operations
`define DPX_RELU        6'b000111   // relu - ReLU activation (max(0, x))
`define DPX_TANH        6'b001000   // tanh approximation
`define DPX_EXP2        6'b001001   // fast exp2 approximation

// Phase 5.3: Sparse Tensor Operations (Blackwell 2:4 Structured Sparsity)
// Note: OP_SPARSE_MMA uses OP_MMA (6'b100010) with func codes 6'b1xxxxx
// This avoids conflict with OP_NOP (6'b111111)

// Sparse MMA function codes (under OP_MMA with high bit set)
`define SPARSE_MMA_FP16     6'b100000   // Sparse FP16 MMA with 2:4 sparsity
`define SPARSE_MMA_BF16     6'b100001   // Sparse BF16 MMA with 2:4 sparsity
`define SPARSE_MMA_TF32     6'b100010   // Sparse TF32 MMA with 2:4 sparsity
`define SPARSE_MMA_INT8     6'b100011   // Sparse INT8 MMA with 2:4 sparsity
`define SPARSE_MMA_FP8      6'b100100   // Sparse FP8 MMA with 2:4 sparsity
`define SPARSE_COMPRESS     6'b101000   // Compress dense to 2:4 sparse format
`define SPARSE_DECOMPRESS   6'b101001   // Decompress 2:4 sparse to dense

// Phase 6.2: Stack and Debug Instructions
`define OP_STACK        6'b110101   // Stack operations (alloca/stacksave/stackrestore)
`define OP_DEBUG        6'b110110   // Debug operations (brkpt/trap/pmevent)
`define OP_MISC         6'b110111   // Misc operations (nanosleep/setmaxnreg)

// Stack operation function codes (for OP_STACK)
`define STACK_ALLOCA          6'b000000   // alloca - dynamic stack allocation
`define STACK_SAVE            6'b000001   // stacksave - save stack pointer
`define STACK_RESTORE         6'b000010   // stackrestore - restore stack pointer

// Debug operation function codes (for OP_DEBUG)
`define DEBUG_BRKPT           6'b000000   // brkpt - breakpoint
`define DEBUG_TRAP            6'b000001   // trap - software trap
`define DEBUG_PMEVENT         6'b000010   // pmevent - performance monitoring event

// Misc operation function codes (for OP_MISC)
`define MISC_NANOSLEEP        6'b000000   // nanosleep - nanosecond delay
`define MISC_SETMAXNREG       6'b000001   // setmaxnreg - set maximum register count

// mbarrier function codes
`define MBAR_INIT           6'b000000   // mbarrier.init - initialize with expected count
`define MBAR_ARRIVE         6'b000001   // mbarrier.arrive - signal arrival
`define MBAR_ARRIVE_DROP    6'b000010   // mbarrier.arrive_drop - arrive and decrement expected
`define MBAR_ARRIVE_TX      6'b000011   // mbarrier.arrive_and_expect_tx - arrive with tx bytes
`define MBAR_TEST_WAIT      6'b000100   // mbarrier.test_wait - non-blocking test
`define MBAR_TRY_WAIT       6'b000101   // mbarrier.try_wait - non-blocking wait attempt
`define MBAR_INVALIDATE     6'b000110   // mbarrier.inval - invalidate barrier
`define MBAR_ARRIVE_NOCOMP  6'b000111   // mbarrier.arrive.noComplete - arrive without completion
`define MBAR_EXPECT_TX      6'b001000   // mbarrier.expect_tx - set expected transaction bytes

// Cache policy function codes (for OP_CACHE_POLICY)
`define CACHE_CREATEPOLICY  6'b000000   // createpolicy - create cache policy token
`define CACHE_APPLYPRIORITY 6'b000001   // applypriority - apply priority to cache lines
`define CACHE_DISCARD       6'b000010   // discard - mark cache lines for eviction (invalidate without writeback)

// 保留
`define OP_MOV_IMM      6'b110000   // Move immediate to register
`define OP_ALU_IMM      6'b110001   // ALU with 16-bit immediate: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:0]=imm16, func from lower bits
`define OP_NOP          6'b111111   // 空操作

//============================================================================
// ALU 功能码 - FUNC (6-bit)
//============================================================================
// 基础算术逻辑运算
`define FUNC_ADD        6'b000000   // 加法
`define FUNC_SUB        6'b000001   // 减法
`define FUNC_AND        6'b000010   // 按位与
`define FUNC_OR         6'b000011   // 按位或
`define FUNC_XOR        6'b000100   // 按位异或
`define FUNC_NOT        6'b000101   // 按位取反
`define FUNC_SHL        6'b000110   // 逻辑左移
`define FUNC_SHR_U      6'b000111   // 逻辑右移 (无符号)
`define FUNC_SHR_S      6'b001000   // 算术右移 (有符号)

// PTX扩展整数运算
`define FUNC_ABS        6'b001001   // 绝对值 abs.s32
`define FUNC_NEG        6'b001010   // 取反 neg.s32
`define FUNC_MIN_S      6'b001011   // 有符号最小值 min.s32
`define FUNC_MIN_U      6'b001100   // 无符号最小值 min.u32
`define FUNC_MAX_S      6'b001101   // 有符号最大值 max.s32
`define FUNC_MAX_U      6'b001110   // 无符号最大值 max.u32
`define FUNC_POPC       6'b001111   // 位计数 popc.b32
`define FUNC_CLZ        6'b010000   // 前导零计数 clz.b32
`define FUNC_BFIND      6'b010001   // 最高有效位查找 bfind.s32
`define FUNC_BREV       6'b010010   // 位反转 brev.b32
`define FUNC_BFE_S      6'b010011   // 位域提取(有符号) bfe.s32
`define FUNC_BFE_U      6'b010100   // 位域提取(无符号) bfe.u32
`define FUNC_BFI        6'b010101   // 位域插入 bfi.b32
`define FUNC_PRMT       6'b010110   // 字节排列 prmt.b32
`define FUNC_SAD        6'b010111   // 绝对差值和 sad.s32
`define FUNC_CNOT       6'b011010   // 取反后与 (cnot)
`define FUNC_BMSK       6'b011011   // 生成位掩码 bmsk
`define FUNC_SZEXT      6'b011100   // 按宽度符号扩展 szext
`define FUNC_FNS        6'b011101   // 找到首个置位 (from LSB) fns
`define FUNC_SHF_L      6'b011110   // 漏斗左移 shf.l
`define FUNC_SHF_R      6'b011111   // 漏斗右移 shf.r
`define FUNC_LOP3       6'b100101   // 三输入逻辑 lop3 (固定LUT)

// 选择操作
`define FUNC_SELP       6'b011000   // 谓词选择 selp.b32
`define FUNC_SLCT       6'b011001   // 符号选择 slct.{f32,s32}

//============================================================================
// MUL 功能码
//============================================================================
`define FUNC_MUL_LO     6'b000000   // 乘法低32位
`define FUNC_MUL_HI     6'b000001   // 乘法高32位
`define FUNC_MAD_LO     6'b000010   // 乘加低32位
`define FUNC_MAD_HI     6'b000011   // 乘加高32位
`define FUNC_MUL24      6'b000100   // 24-bit 乘法 (低32位)
`define FUNC_MAD24      6'b000101   // 24-bit 乘加 (低32位)
`define FUNC_MAD_LO_CC   6'b100101   // 乘加低32位，生成进位
`define FUNC_MADC_LO     6'b100110   // 乘加低32位，带进位输入

// 除法/取余 (OP_DIV) 功能码
`define DIV_FUNC_DIV_S  6'b000000   // 有符号除法
`define DIV_FUNC_DIV_U  6'b000001   // 无符号除法
`define DIV_FUNC_REM_S  6'b000010   // 有符号取余
`define DIV_FUNC_REM_U  6'b000011   // 无符号取余

//============================================================================
// 比较功能码
//============================================================================
`define CMP_EQ          6'b000000   // 等于
`define CMP_NE          6'b000001   // 不等于
`define CMP_LT          6'b000010   // 小于
`define CMP_LE          6'b000011   // 小于等于
`define CMP_GT          6'b000100   // 大于
`define CMP_GE          6'b000101   // 大于等于

//============================================================================
// FP32 功能码
//============================================================================
`define FP_ADD          6'b000000   // add.f32
`define FP_SUB          6'b000001   // sub.f32
`define FP_MUL          6'b000010   // mul.f32
`define FP_DIV          6'b000011   // div.f32
`define FP_FMA          6'b000100   // fma.f32
`define FP_NEG          6'b000101   // neg.f32
`define FP_ABS          6'b000110   // abs.f32
`define FP_MIN          6'b000111   // min.f32
`define FP_MAX          6'b001000   // max.f32

//============================================================================
// FP特殊函数功能码
//============================================================================
`define FP_RCP          6'b000000   // rcp.f32 (1/x)
`define FP_SQRT         6'b000001   // sqrt.f32
`define FP_RSQRT        6'b000010   // rsqrt.f32 (1/sqrt(x))
`define FP_SIN          6'b000011   // sin.f32
`define FP_COS          6'b000100   // cos.f32
`define FP_LG2          6'b000101   // lg2.f32 (log2)
`define FP_EX2          6'b000110   // ex2.f32 (2^x)
`define FP_TANH         6'b000111   // tanh.f32
`define FP_TESTP        6'b001000   // testp (simple NaN test)
`define FP_COPYSIGN     6'b001001   // copysign

//============================================================================
// 原子操作功能码
//============================================================================
`define ATOM_ADD        6'b000000   // atom.add
`define ATOM_MIN_S      6'b000001   // atom.min (signed)
`define ATOM_MIN_U      6'b000010   // atom.min (unsigned)
`define ATOM_MAX_S      6'b000011   // atom.max (signed)
`define ATOM_MAX_U      6'b000100   // atom.max (unsigned)
`define ATOM_INC        6'b000101   // atom.inc
`define ATOM_DEC        6'b000110   // atom.dec
`define ATOM_AND        6'b000111   // atom.and
`define ATOM_OR         6'b001000   // atom.or
`define ATOM_XOR        6'b001001   // atom.xor
`define ATOM_EXCH       6'b001010   // atom.exch
`define ATOM_CAS        6'b001011   // atom.cas

//============================================================================
// Warp Shuffle 功能码
//============================================================================
`define SHFL_IDX        6'b000000   // shfl.sync.idx
`define SHFL_UP         6'b000001   // shfl.sync.up
`define SHFL_DOWN       6'b000010   // shfl.sync.down
`define SHFL_BFLY       6'b000011   // shfl.sync.bfly

//============================================================================
// Warp Vote 功能码
//============================================================================
`define VOTE_ALL        6'b000000   // vote.sync.all
`define VOTE_ANY        6'b000001   // vote.sync.any
`define VOTE_UNI        6'b000010   // vote.sync.uni
`define VOTE_BALLOT     6'b000011   // vote.sync.ballot

//============================================================================
// FP16/BF16 功能码
//============================================================================
`define FP16_ADD        6'b000000   // add.f16/add.f16x2
`define FP16_SUB        6'b000001   // sub.f16/sub.f16x2
`define FP16_MUL        6'b000010   // mul.f16/mul.f16x2
`define FP16_FMA        6'b000011   // fma.f16/fma.f16x2
`define FP16_NEG        6'b000100   // neg.f16
`define FP16_ABS        6'b000101   // abs.f16
`define FP16_MIN        6'b000110   // min.f16
`define FP16_MAX        6'b000111   // max.f16
`define FP16_TANH       6'b001000   // tanh.f16 (for ML)
`define FP16_EX2        6'b001001   // ex2.f16
// BF16 (Brain Float 16)
`define BF16_ADD        6'b010000   // add.bf16
`define BF16_SUB        6'b010001   // sub.bf16
`define BF16_MUL        6'b010010   // mul.bf16
`define BF16_FMA        6'b010011   // fma.bf16
// Packed FP16x2 operations
`define FP16X2_ADD      6'b100000   // add.f16x2
`define FP16X2_SUB      6'b100001   // sub.f16x2
`define FP16X2_MUL      6'b100010   // mul.f16x2
`define FP16X2_FMA      6'b100011   // fma.f16x2
// FP16 compare operations (setp.f16)
`define FP16_CMP_EQ     6'b001010   // setp.eq.f16
`define FP16_CMP_NE     6'b001011   // setp.ne.f16
`define FP16_CMP_LT     6'b001100   // setp.lt.f16
`define FP16_CMP_LE     6'b001101   // setp.le.f16
`define FP16_CMP_GT     6'b001110   // setp.gt.f16
`define FP16_CMP_GE     6'b001111   // setp.ge.f16
`define FP16_CMP_NUM    6'b011000   // setp.num.f16 (ordered: both not NaN)
`define FP16_CMP_NAN    6'b011001   // setp.nan.f16 (unordered: either is NaN)
`define FP16_MUL_F32    6'b010010   // mul.f16.f32 - FP16 inputs, FP32 output (mixed precision)
// Mixed FP16-FP32 compare (convert FP16 to FP32, then compare)
`define FP16_CMP_EQ_F32 6'b011010   // setp.eq.f16.f32 (mixed precision)
`define FP16_CMP_LT_F32 6'b011011   // setp.lt.f16.f32 (mixed precision)
`define FP16_CMP_LE_F32 6'b011100   // setp.le.f16.f32 (mixed precision)
`define FP16_CMP_GT_F32 6'b011101   // setp.gt.f16.f32 (mixed precision)
`define FP16_CMP_GE_F32 6'b011110   // setp.ge.f16.f32 (mixed precision)
`define FP16_CMP_NE_F32 6'b011111   // setp.ne.f16.f32 (mixed precision)

//============================================================================
// FP64 功能码 (双精度浮点)
//============================================================================
`define FP64_ADD        6'b000000   // add.f64
`define FP64_SUB        6'b000001   // sub.f64
`define FP64_MUL        6'b000010   // mul.f64
`define FP64_DIV        6'b000011   // div.f64
`define FP64_FMA        6'b000100   // fma.f64
`define FP64_NEG        6'b000101   // neg.f64
`define FP64_ABS        6'b000110   // abs.f64
`define FP64_MIN        6'b000111   // min.f64
`define FP64_MAX        6'b001000   // max.f64
`define FP64_SQRT       6'b001001   // sqrt.f64
`define FP64_RSQRT      6'b001010   // rsqrt.f64
`define FP64_RCP        6'b001011   // rcp.f64
`define FP64_COPYSIGN   6'b001100   // copysign.f64
`define FP64_TESTP      6'b001101   // testp.f64

//============================================================================
// 类型转换功能码 (CVT指令)
//============================================================================
`define CVT_S32_F32     6'd42   // cvt.s32.f32
`define CVT_U32_F32     6'd43   // cvt.u32.f32
`define CVT_F32_S32     6'd44   // cvt.f32.s32
`define CVT_F32_U32     6'd45   // cvt.f32.u32
`define CVT_F32_F64     6'd46   // cvt.f32.f64
`define CVT_F64_F32     6'd47   // cvt.f64.f32
`define CVT_F32_F16     6'b101000   // cvt.f32.f16 (unique code 40)
`define CVT_F16_F32     6'b101001   // cvt.f16.f32 (unique code 41)
`define CVT_S64_F64     6'd48   // cvt.s64.f64
`define CVT_U64_F64     6'd49   // cvt.u64.f64
`define CVT_F64_S64     6'd50   // cvt.f64.s64
`define CVT_F64_U64     6'd51   // cvt.f64.u64
`define CVT_PACK        6'd52   // cvt.pack (pack two 16-bit values) - unique code

//============================================================================
// WMMA 功能码
//============================================================================
`define WMMA_M16N16K16  6'b000000   // 16x16x16 配置
`define WMMA_M8N8K4     6'b000001   // 8x8x4 配置
`define WMMA_M32N8K16   6'b000010   // 32x8x16 配置

//============================================================================
// Tensor Core 数据类型 (用于WMMA/MMA配置)
// Extended to 4-bit to support FP6 (5th-gen Tensor Core - Blackwell)
//============================================================================
`define TC_DATA_FP16        4'd0
`define TC_DATA_BF16        4'd1
`define TC_DATA_INT8        4'd2
`define TC_DATA_INT4        4'd3
`define TC_DATA_FP8_E4M3    4'd4
`define TC_DATA_FP8_E5M2    4'd5
`define TC_DATA_FP4_E2M1    4'd6
`define TC_DATA_FP4_E3M0    4'd7
// FP6 format (5th-gen Tensor Core - Blackwell)
// FP6 E3M2: 1-bit sign, 3-bit exponent (bias=3), 2-bit mantissa
// Range: ~0.0625 to 7.5, suitable for weight quantization in LLMs
`define TC_DATA_FP6_E3M2    4'd8
`define TC_DATA_FP8         `TC_DATA_FP8_E4M3
`define TC_DATA_FP4         `TC_DATA_FP4_E2M1
`define TC_DATA_FP6         `TC_DATA_FP6_E3M2

// FP4 格式选择 (默认E2M1)
`define TC_FP4_E2M1     2'd0
`define TC_FP4_E3M0     2'd1

// FP6 格式选择 (5th-gen Tensor Core)
`define TC_FP6_E3M2     2'd0        // Default E3M2 format (bias=3)

// FP8 格式选择 (默认E4M3)
`define TC_FP8_E4M3     2'd0
`define TC_FP8_E5M2     2'd1

//============================================================================
// Texture功能码
//============================================================================
`define TEX_1D          6'b000000   // tex.1d
`define TEX_2D          6'b000001   // tex.2d
`define TEX_3D          6'b000010   // tex.3d
`define TEX_CUBE        6'b000011   // tex.cube
`define TEX_A1D         6'b000100   // tex.a1d (array 1D)
`define TEX_A2D         6'b000101   // tex.a2d (array 2D)
`define TEX_LEVEL       6'b001000   // tex.level (explicit LOD)
`define TEX_GRAD        6'b001001   // tex.grad (gradient)
`define TEX_GATHER      6'b001010   // tld4/tex.gather
// Texture query
`define TXQ_WIDTH       6'b010000   // txq.width
`define TXQ_HEIGHT      6'b010001   // txq.height
`define TXQ_DEPTH       6'b010010   // txq.depth
`define TXQ_LEVELS      6'b010011   // txq.num_mipmap_levels
// Surface modes
`define SURF_1D         6'b000000   // suld/sust.1d
`define SURF_2D         6'b000001   // suld/sust.2d
`define SURF_3D         6'b000010   // suld/sust.3d
`define SURF_A1D        6'b000100   // suld/sust.a1d
`define SURF_A2D        6'b000101   // suld/sust.a2d

//============================================================================
// Video指令功能码
//============================================================================
`define VIDEO_VADD      6'b000000   // vadd.{s32,u32}
`define VIDEO_VSUB      6'b000001   // vsub.{s32,u32}
`define VIDEO_VABSDIFF  6'b000010   // vabsdiff.{s32,u32}
`define VIDEO_VMIN      6'b000011   // vmin.{s32,u32}
`define VIDEO_VMAX      6'b000100   // vmax.{s32,u32}
`define VIDEO_VSHL      6'b000101   // vshl.u32
`define VIDEO_VSHR      6'b000110   // vshr.{s32,u32}
`define VIDEO_VMAD      6'b000111   // vmad.{s32,u32}
`define VIDEO_VSET      6'b001000   // vset.{s32,u32}
// SIMD byte/short operations
`define VIDEO_VADD4     6'b010000   // vadd4.{s8,u8} - 4x8-bit
`define VIDEO_VSUB4     6'b010001   // vsub4.{s8,u8}
`define VIDEO_VABSDIFF4 6'b010010   // vabsdiff4.{s8,u8}
`define VIDEO_VADD2     6'b010100   // vadd2.{s16,u16} - 2x16-bit
`define VIDEO_VSUB2     6'b010101   // vsub2.{s16,u16}
`define VIDEO_VMUL2     6'b010110   // vmul2.{s16,u16}
// DP4A/DP2A for ML
`define VIDEO_DP4A      6'b100000   // dp4a.{s32,u32}.{s32,u32}
`define VIDEO_DP2A      6'b100001   // dp2a.{s32,u32}.{s32,u32}
`define VIDEO_DP4A_ALU   6'b100010   // dp4a routed via ALU path
`define VIDEO_DP2A_ALU   6'b100011   // dp2a routed via ALU path

//============================================================================
// 特殊寄存器编码
//============================================================================
`define SREG_TID_X      5'd0        // %tid.x - 线程ID
`define SREG_TID_Y      5'd1        // %tid.y
`define SREG_TID_Z      5'd2        // %tid.z
`define SREG_CTAID_X    5'd3        // %ctaid.x - Block ID
`define SREG_CTAID_Y    5'd4        // %ctaid.y
`define SREG_CTAID_Z    5'd5        // %ctaid.z
`define SREG_NTID_X     5'd6        // %ntid.x - Block维度
`define SREG_NTID_Y     5'd7        // %ntid.y
`define SREG_NTID_Z     5'd8        // %ntid.z
`define SREG_NCTAID_X   5'd9        // %nctaid.x - Grid维度
`define SREG_NCTAID_Y   5'd10       // %nctaid.y
`define SREG_NCTAID_Z   5'd11       // %nctaid.z
`define SREG_LANEID     5'd12       // %laneid
`define SREG_WARPID     5'd13       // %warpid
`define SREG_SMID       5'd14       // %smid
`define SREG_ACTIVEMASK 5'd15       // %activemask 当前活跃线程掩码

//============================================================================
// 流水线阶段编码
//============================================================================
`define STAGE_FETCH     3'd0
`define STAGE_DECODE    3'd1
`define STAGE_EXEC      3'd2
`define STAGE_MEM       3'd3
`define STAGE_WB        3'd4

//============================================================================
// 指令字段位置
//============================================================================
`define INST_OPCODE     31:26       // [31:26] OPCODE
`define INST_RD         25:21       // [25:21] 目标寄存器
`define INST_RA         20:16       // [20:16] 源寄存器A
`define INST_RB         15:11       // [15:11] 源寄存器B
`define INST_RC         10:6        // [10:6]  源寄存器C/谓词
`define INST_FUNC       5:0         // [5:0]   功能码
`define INST_IMM16      15:0        // [15:0]  16位立即数
`define INST_IMM21      20:0        // [20:0]  21位立即数(分支偏移)

//============================================================================
// 内存接口参数
//============================================================================
`define GLOBAL_ADDR_WIDTH   32      // 全局地址宽度
`define GLOBAL_DATA_WIDTH   32      // 全局数据宽度
`define CACHE_LINE_SIZE     32      // 缓存行大小(字节)

//============================================================================
// 缓存控制修饰符 (Cache Hints)
//============================================================================
`define CACHE_DEFAULT   3'b000      // 默认缓存行为
`define CACHE_CA        3'b001      // Cache at all levels (ld.ca)
`define CACHE_CG        3'b010      // Cache at global level (ld.cg)
`define CACHE_CS        3'b011      // Cache streaming (ld.cs)
`define CACHE_LU        3'b100      // Last use (ld.lu)
`define CACHE_CV        3'b101      // Cache as volatile (ld.cv)
`define CACHE_WB        3'b110      // Write-back (st.wb)
`define CACHE_WT        3'b111      // Write-through (st.wt)

//============================================================================
// 异步拷贝功能码 (cp.async)
//============================================================================
`define CPASYNC_CA      6'b000000   // cp.async.ca.shared.global
`define CPASYNC_CG      6'b000001   // cp.async.cg.shared.global
`define CPASYNC_COMMIT  6'b000010   // cp.async.commit_group
`define CPASYNC_WAIT    6'b000011   // cp.async.wait_group
`define CPASYNC_WAIT_ALL 6'b000100  // cp.async.wait_all
`define CPASYNC_BULK    6'b001000   // cp.async.bulk
`define CPASYNC_BULK_TENSOR 6'b001001   // cp.async.bulk.tensor (TMA - Tensor Memory Accelerator)

//============================================================================
// 整数进位操作功能码
//============================================================================
`define FUNC_ADD_CC     6'b100000   // add.cc (with carry out)
`define FUNC_ADDC       6'b100001   // addc (add with carry in)
`define FUNC_SUB_CC     6'b100010   // sub.cc (with borrow out)
`define FUNC_SUBC       6'b100011   // subc (sub with borrow in)
`define FUNC_MUL_WIDE   6'b100100   // mul.wide (32x32->64)

//============================================================================
// WGMMA (Hopper) 功能码
//============================================================================
`define WGMMA_M64N8K16      6'b000000   // wgmma.mma_async m64n8k16
`define WGMMA_M64N16K16     6'b000001   // wgmma.mma_async m64n16k16
`define WGMMA_M64N32K16     6'b000010   // wgmma.mma_async m64n32k16
`define WGMMA_M64N64K16     6'b000011   // wgmma.mma_async m64n64k16
`define WGMMA_M64N128K16    6'b000100   // wgmma.mma_async m64n128k16
`define WGMMA_M64N256K16    6'b000101   // wgmma.mma_async m64n256k16
`define WGMMA_FENCE         6'b010000   // wgmma.fence
`define WGMMA_COMMIT_GROUP  6'b010001   // wgmma.commit_group
`define WGMMA_WAIT_GROUP    6'b010010   // wgmma.wait_group

//============================================================================
// 预取指令功能码
//============================================================================
`define PREFETCH_L1     6'b000000   // prefetch.L1
`define PREFETCH_L2     6'b000001   // prefetch.L2
`define PREFETCHU_L1    6'b000010   // prefetchu.L1 (uniform)

//============================================================================
// 扩展操作码 (Phase 10+)
//============================================================================
`define OP_CPASYNC      6'b101011   // cp.async operations
`define OP_PREFETCH     6'b101100   // prefetch operations
`define OP_WGMMA_LOAD   6'b101101   // WGMMA load
`define OP_WGMMA_STORE  6'b101110   // WGMMA store
`define OP_WGMMA_MMA    6'b101111   // WGMMA mma_async

//============================================================================
// Blackwell 5th-gen Tensor Core (tcgen05) Instructions (SM100+)
// Per-thread tensor operations with Tensor Memory (TMEM) accumulator
// Replaces warp-synchronous WMMA/WGMMA with independent per-thread MMA
//============================================================================
// tcgen05 uses func[4]=1 under OP_MMA to distinguish from regular/sparse MMA
// Encoding: OP_MMA (6'b100010) + func[5:4]=01 for tcgen05 operations
// Regular MMA: func[5:4]=00, Sparse MMA: func[5:4]=10, TCGEN05: func[5:4]=01
`define OP_TCGEN05      `OP_MMA     // tcgen05 shares OP_MMA opcode

// tcgen05 function codes (for OP_TCGEN05 = OP_MMA with func[4]=1)
// Format: func[5]=0, func[4]=1, func[3:0]=operation
`define TCGEN05_MMA         6'b010000   // tcgen05.mma - Per-thread async MMA with TMEM accumulator
`define TCGEN05_LD          6'b010001   // tcgen05.ld - Load from TMEM to registers
`define TCGEN05_ST          6'b010010   // tcgen05.st - Store from registers to TMEM
`define TCGEN05_CP          6'b010011   // tcgen05.cp - Async tensor data transfer (TMA-like)
`define TCGEN05_ALLOC       6'b010100   // tcgen05.alloc - Allocate TMEM columns
`define TCGEN05_DEALLOC     6'b010101   // tcgen05.dealloc - Deallocate TMEM (required before kernel exit)
`define TCGEN05_COMMIT      6'b010110   // tcgen05.commit - Signal MMA completion via mbarrier
`define TCGEN05_WAIT        6'b010111   // tcgen05.wait - Wait for pending TMEM operations

// tcgen05.mma shape configurations (in idesc[5:0])
`define TCGEN05_M128N256K16 6'b000000   // m128n256k16 (largest Blackwell tile)
`define TCGEN05_M64N256K16  6'b000001   // m64n256k16 (Hopper-compatible)
`define TCGEN05_M64N128K16  6'b000010   // m64n128k16
`define TCGEN05_M64N64K16   6'b000011   // m64n64k16
`define TCGEN05_M64N32K16   6'b000100   // m64n32k16
`define TCGEN05_M64N16K16   6'b000101   // m64n16k16

// tcgen05 data type codes (in idesc[9:6])
`define TCGEN05_FP16        4'b0000     // FP16 input with FP32 accumulator
`define TCGEN05_BF16        4'b0001     // BF16 input with FP32 accumulator
`define TCGEN05_TF32        4'b0010     // TF32 input with FP32 accumulator
`define TCGEN05_FP8_E4M3    4'b0011     // FP8 E4M3 input
`define TCGEN05_FP8_E5M2    4'b0100     // FP8 E5M2 input
`define TCGEN05_FP6_E3M2    4'b0101     // FP6 E3M2 input (Blackwell new)
`define TCGEN05_FP4_E2M1    4'b0110     // FP4 E2M1 input (Blackwell new)
`define TCGEN05_INT8        4'b0111     // INT8 input with INT32 accumulator

// TMEM address encoding
// [31:16] = Row address (0-127 for 256KB TMEM)
// [15:0]  = Column address (0-511)
`define TMEM_ROW_BITS       7           // 128 rows
`define TMEM_COL_BITS       9           // 512 columns
`define TMEM_CELL_BITS      32          // 32-bit cells
`define TMEM_SIZE_KB        256         // 256KB per SM

`endif // GPU_DEFINES_VH
