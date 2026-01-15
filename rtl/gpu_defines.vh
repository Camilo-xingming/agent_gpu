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
`define OP_NOP          6'b111111   // 空操作

//============================================================================
// ALU 功能码 - FUNC (6-bit)
//============================================================================
`define FUNC_ADD        6'b000000   // 加法
`define FUNC_SUB        6'b000001   // 减法
`define FUNC_AND        6'b000010   // 按位与
`define FUNC_OR         6'b000011   // 按位或
`define FUNC_XOR        6'b000100   // 按位异或
`define FUNC_NOT        6'b000101   // 按位取反
`define FUNC_SHL        6'b000110   // 逻辑左移
`define FUNC_SHR_U      6'b000111   // 逻辑右移 (无符号)
`define FUNC_SHR_S      6'b001000   // 算术右移 (有符号)

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
