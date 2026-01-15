//============================================================================
// RalphGPU - Instruction Decoder
// 解码PTX风格的32位指令
// 支持完整PTX ISA 8.5+指令集
//============================================================================

`include "gpu_defines.vh"

module decoder (
    input  wire        clk,
    input  wire        rst_n,

    // 指令输入
    input  wire [31:0] instruction,
    input  wire        valid_in,

    // 解码结果
    output reg         valid_out,
    output reg  [5:0]  opcode,
    output reg  [4:0]  rd,          // 目标寄存器
    output reg  [4:0]  ra,          // 源寄存器A
    output reg  [4:0]  rb,          // 源寄存器B
    output reg  [4:0]  rc,          // 源寄存器C / 谓词
    output reg  [5:0]  func,        // 功能码
    output reg  [15:0] imm16,       // 16位立即数
    output reg  [20:0] imm21,       // 21位立即数(分支偏移)

    // 控制信号 - 基础
    output reg         use_imm,     // 使用立即数
    output reg         alu_op,      // ALU操作
    output reg         mul_op,      // 乘法操作
    output reg         div_op,      // 除法操作
    output reg         mem_read,    // 内存读
    output reg         mem_write,   // 内存写
    output reg         mem_shared,  // 共享内存访问
    output reg         branch_op,   // 分支操作
    output reg         sync_op,     // 同步操作
    output reg         special_reg, // 特殊寄存器访问
    output reg         exit_op,     // EXIT/RET操作
    output reg         reg_write,   // 需要写寄存器
    output reg         pred_write,  // 需要写谓词
    output reg  [2:0]  pred_addr,   // 谓词寄存器地址

    // 控制信号 - FP操作
    output reg         fp32_op,     // FP32算术操作
    output reg         fp32_special,// FP32特殊函数
    output reg         fp64_op,     // FP64操作
    output reg         fp16_op,     // FP16/BF16操作
    output reg         cvt_op,      // 类型转换

    // 控制信号 - 扩展内存
    output reg         mem_param,   // 参数内存
    output reg         mem_const,   // 常量内存
    output reg         mem_local,   // 本地内存
    output reg         mem_vector,  // 向量内存访问
    output reg  [1:0]  vec_size,    // 向量大小: 00=1, 01=2, 10=4

    // 控制信号 - 原子操作
    output reg         atomic_op,   // 原子操作
    output reg         reduce_op,   // 归约操作

    // 控制信号 - Warp操作
    output reg         shfl_op,     // Warp shuffle
    output reg         vote_op,     // Warp vote
    output reg         redux_op,    // Warp reduction

    // 控制信号 - Tensor Core
    output reg         wmma_load,   // WMMA加载
    output reg         wmma_store,  // WMMA存储
    output reg         wmma_mma,    // WMMA MMA
    output reg         mma_op,      // MMA指令

    // 控制信号 - 其他
    output reg         call_op,     // 函数调用
    output reg         membar_op,   // 内存屏障

    // 控制信号 - Video
    output reg         video_op,    // Video处理

    // 控制信号 - Texture/Surface
    output reg         tex_op,      // 纹理采样
    output reg         txq_op,      // 纹理查询
    output reg         surf_ld,     // Surface加载
    output reg         surf_st,     // Surface存储
    output reg         surf_red     // Surface归约
);

    //------------------------------------------------------------------------
    // 指令字段提取 (组合逻辑)
    //------------------------------------------------------------------------
    wire [5:0]  inst_opcode = instruction[`INST_OPCODE];
    wire [4:0]  inst_rd     = instruction[`INST_RD];
    wire [4:0]  inst_ra     = instruction[`INST_RA];
    wire [4:0]  inst_rb     = instruction[`INST_RB];
    wire [4:0]  inst_rc     = instruction[`INST_RC];
    wire [5:0]  inst_func   = instruction[`INST_FUNC];
    wire [15:0] inst_imm16  = instruction[`INST_IMM16];
    wire [20:0] inst_imm21  = instruction[`INST_IMM21];

    //------------------------------------------------------------------------
    // 解码逻辑
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            opcode      <= 6'b0;
            rd          <= 5'b0;
            ra          <= 5'b0;
            rb          <= 5'b0;
            rc          <= 5'b0;
            func        <= 6'b0;
            imm16       <= 16'b0;
            imm21       <= 21'b0;
            use_imm     <= 1'b0;
            alu_op      <= 1'b0;
            mul_op      <= 1'b0;
            div_op      <= 1'b0;
            mem_read    <= 1'b0;
            mem_write   <= 1'b0;
            mem_shared  <= 1'b0;
            branch_op   <= 1'b0;
            sync_op     <= 1'b0;
            special_reg <= 1'b0;
            exit_op     <= 1'b0;
            reg_write   <= 1'b0;
            pred_write  <= 1'b0;
            pred_addr   <= 3'b0;
            // 新增控制信号复位
            fp32_op     <= 1'b0;
            fp32_special<= 1'b0;
            fp64_op     <= 1'b0;
            fp16_op     <= 1'b0;
            cvt_op      <= 1'b0;
            mem_param   <= 1'b0;
            mem_const   <= 1'b0;
            mem_local   <= 1'b0;
            mem_vector  <= 1'b0;
            vec_size    <= 2'b0;
            atomic_op   <= 1'b0;
            reduce_op   <= 1'b0;
            shfl_op     <= 1'b0;
            vote_op     <= 1'b0;
            redux_op    <= 1'b0;
            wmma_load   <= 1'b0;
            wmma_store  <= 1'b0;
            wmma_mma    <= 1'b0;
            mma_op      <= 1'b0;
            call_op     <= 1'b0;
            membar_op   <= 1'b0;
            video_op    <= 1'b0;
            tex_op      <= 1'b0;
            txq_op      <= 1'b0;
            surf_ld     <= 1'b0;
            surf_st     <= 1'b0;
            surf_red    <= 1'b0;
        end else if (valid_in) begin
            valid_out <= 1'b1;

            // 基本字段
            opcode <= inst_opcode;
            rd     <= inst_rd;
            ra     <= inst_ra;
            rb     <= inst_rb;
            rc     <= inst_rc;
            func   <= inst_func;
            imm16  <= inst_imm16;
            imm21  <= inst_imm21;

            // 默认控制信号
            use_imm     <= 1'b0;
            alu_op      <= 1'b0;
            mul_op      <= 1'b0;
            div_op      <= 1'b0;
            mem_read    <= 1'b0;
            mem_write   <= 1'b0;
            mem_shared  <= 1'b0;
            branch_op   <= 1'b0;
            sync_op     <= 1'b0;
            special_reg <= 1'b0;
            exit_op     <= 1'b0;
            reg_write   <= 1'b0;
            pred_write  <= 1'b0;
            pred_addr   <= inst_rc[2:0];
            // 新增控制信号默认值
            fp32_op     <= 1'b0;
            fp32_special<= 1'b0;
            fp64_op     <= 1'b0;
            fp16_op     <= 1'b0;
            cvt_op      <= 1'b0;
            mem_param   <= 1'b0;
            mem_const   <= 1'b0;
            mem_local   <= 1'b0;
            mem_vector  <= 1'b0;
            vec_size    <= 2'b0;
            atomic_op   <= 1'b0;
            reduce_op   <= 1'b0;
            shfl_op     <= 1'b0;
            vote_op     <= 1'b0;
            redux_op    <= 1'b0;
            wmma_load   <= 1'b0;
            wmma_store  <= 1'b0;
            wmma_mma    <= 1'b0;
            mma_op      <= 1'b0;
            call_op     <= 1'b0;
            membar_op   <= 1'b0;
            video_op    <= 1'b0;
            tex_op      <= 1'b0;
            txq_op      <= 1'b0;
            surf_ld     <= 1'b0;
            surf_st     <= 1'b0;
            surf_red    <= 1'b0;

            // 根据OPCODE设置控制信号
            case (inst_opcode)
                `OP_ALU: begin
                    alu_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_MUL: begin
                    mul_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_DIV: begin
                    div_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SETP: begin
                    alu_op     <= 1'b1;  // 比较用ALU
                    pred_write <= 1'b1;
                end

                `OP_BRANCH: begin
                    branch_op <= 1'b1;
                end

                `OP_LD_GLOBAL: begin
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_ST_GLOBAL: begin
                    mem_write <= 1'b1;
                end

                `OP_LD_SHARED: begin
                    mem_read   <= 1'b1;
                    mem_shared <= 1'b1;
                    reg_write  <= 1'b1;
                end

                `OP_ST_SHARED: begin
                    mem_write  <= 1'b1;
                    mem_shared <= 1'b1;
                end

                `OP_MOV_SPECIAL: begin
                    special_reg <= 1'b1;
                    reg_write   <= 1'b1;
                end

                `OP_BAR_SYNC: begin
                    sync_op <= 1'b1;
                end

                `OP_EXIT, `OP_RET: begin
                    exit_op <= 1'b1;
                end

                //============================================================
                // Phase 2: 浮点运算指令
                //============================================================
                `OP_FP32_ARITH: begin
                    fp32_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_FP32_SPECIAL: begin
                    fp32_special <= 1'b1;
                    reg_write    <= 1'b1;
                end

                `OP_FP64_ARITH: begin
                    fp64_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_FP16_ARITH: begin
                    fp16_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_CVT: begin
                    cvt_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 3: 扩展内存操作
                //============================================================
                `OP_LD_PARAM: begin
                    mem_read  <= 1'b1;
                    mem_param <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_LD_CONST: begin
                    mem_read  <= 1'b1;
                    mem_const <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_LD_LOCAL: begin
                    mem_read  <= 1'b1;
                    mem_local <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_ST_LOCAL: begin
                    mem_write <= 1'b1;
                    mem_local <= 1'b1;
                end

                `OP_LD_V2: begin
                    mem_read   <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b01;  // v2
                    reg_write  <= 1'b1;
                end

                `OP_LD_V4: begin
                    mem_read   <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b10;  // v4
                    reg_write  <= 1'b1;
                end

                `OP_ST_V2: begin
                    mem_write  <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b01;
                end

                `OP_ST_V4: begin
                    mem_write  <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b10;
                end

                //============================================================
                // Phase 4: 原子操作
                //============================================================
                `OP_ATOM: begin
                    atomic_op <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                    reg_write <= 1'b1;  // 返回旧值
                end

                `OP_RED: begin
                    reduce_op <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                end

                //============================================================
                // Phase 5: Warp级操作
                //============================================================
                `OP_SHFL: begin
                    shfl_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_VOTE: begin
                    vote_op    <= 1'b1;
                    reg_write  <= 1'b1;
                    pred_write <= 1'b1;
                end

                `OP_REDUX: begin
                    redux_op  <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 6: Tensor Core
                //============================================================
                `OP_WMMA_LOAD: begin
                    wmma_load <= 1'b1;
                    mem_read  <= 1'b1;
                end

                `OP_WMMA_STORE: begin
                    wmma_store <= 1'b1;
                    mem_write  <= 1'b1;
                end

                `OP_WMMA_MMA: begin
                    wmma_mma  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_MMA: begin
                    mma_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 7: 控制流扩展
                //============================================================
                `OP_CALL: begin
                    call_op   <= 1'b1;
                    branch_op <= 1'b1;
                end

                `OP_MEMBAR: begin
                    membar_op <= 1'b1;
                    sync_op   <= 1'b1;
                end

                //============================================================
                // Phase 8: Video处理指令
                //============================================================
                `OP_VIDEO: begin
                    video_op  <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 9: 纹理/Surface指令
                //============================================================
                `OP_TEX: begin
                    tex_op    <= 1'b1;
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_TXQ: begin
                    txq_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SULD: begin
                    surf_ld   <= 1'b1;
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SUST: begin
                    surf_st   <= 1'b1;
                    mem_write <= 1'b1;
                end

                `OP_SURED: begin
                    surf_red  <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_NOP: begin
                    // 空操作，不设置任何控制信号
                end

                default: begin
                    // 未知指令，当作NOP
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
