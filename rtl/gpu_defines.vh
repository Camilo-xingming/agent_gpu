//============================================================================
// RalphGPU - CUDA/PTX Compatible GPU IP
// 全局定义和参数
//============================================================================

`ifndef GPU_DEFINES_VH
`define GPU_DEFINES_VH

//============================================================================
// 可配置参数 - 修改这些参数来扩展算力
//============================================================================
`define NUM_SM              2       // SM数量 (2, 4, 8, 16...)
`define THREADS_PER_WARP    32      // 每Warp线程数
`define WARPS_PER_SM        4       // 每SM的Warp数
`define NUM_REGS            32      // 每线程寄存器数
`define SHARED_MEM_KB       16      // 共享内存大小(KB)
`define DATA_WIDTH          32      // 数据位宽

//============================================================================
// 派生参数 (自动计算)
//============================================================================
`define THREADS_PER_SM      (`THREADS_PER_WARP * `WARPS_PER_SM)
`define TOTAL_THREADS       (`THREADS_PER_SM * `NUM_SM)
`define REG_ADDR_WIDTH      5       // log2(32) = 5
`define WARP_ID_WIDTH       2       // log2(4) = 2
`define SM_ID_WIDTH         1       // log2(2) = 1
`define THREAD_ID_WIDTH     5       // log2(32) = 5
`define SHARED_MEM_SIZE     (`SHARED_MEM_KB * 1024)
`define SHARED_MEM_ADDR_W   14      // log2(16K) = 14

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
`define OP_BAR_SYNC     6'b001010   // 同步屏障
`define OP_EXIT         6'b001011   // Kernel退出
`define OP_RET          6'b001100   // 函数返回 (当前等同EXIT)

// Phase 2: 浮点运算指令
`define OP_FP32_ARITH   6'b001101   // FP32算术 (add/sub/mul/div/fma)
`define OP_FP32_SPECIAL 6'b001110   // FP32特殊函数 (rcp/sqrt/rsqrt/sin/cos/lg2/ex2)
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

// 保留
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

// 选择操作
`define FUNC_SELP       6'b011000   // 谓词选择 selp.b32
`define FUNC_SLCT       6'b011001   // 符号选择 slct.{f32,s32}

//============================================================================
// MUL 功能码
//============================================================================
`define FUNC_MUL_LO     6'b000000   // 乘法低32位
`define FUNC_MUL_HI     6'b000001   // 乘法高32位
`define FUNC_MAD_LO     6'b000010   // 乘加低32位

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

//============================================================================
// 类型转换功能码 (CVT指令)
//============================================================================
`define CVT_S32_F32     6'b000000   // cvt.s32.f32
`define CVT_U32_F32     6'b000001   // cvt.u32.f32
`define CVT_F32_S32     6'b000010   // cvt.f32.s32
`define CVT_F32_U32     6'b000011   // cvt.f32.u32
`define CVT_F32_F64     6'b000100   // cvt.f32.f64
`define CVT_F64_F32     6'b000101   // cvt.f64.f32
`define CVT_F32_F16     6'b000110   // cvt.f32.f16
`define CVT_F16_F32     6'b000111   // cvt.f16.f32

//============================================================================
// WMMA 功能码
//============================================================================
`define WMMA_M16N16K16  6'b000000   // 16x16x16 配置
`define WMMA_M8N8K4     6'b000001   // 8x8x4 配置
`define WMMA_M32N8K16   6'b000010   // 32x8x16 配置

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

`endif // GPU_DEFINES_VH
