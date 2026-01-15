//============================================================================
// RalphGPU - Instruction Decoder
// 解码PTX风格的32位指令
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

    // 控制信号
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
    output reg  [2:0]  pred_addr    // 谓词寄存器地址
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
