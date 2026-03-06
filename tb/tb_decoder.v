//============================================================================
// RalphGPU - Instruction Decoder Test
// 验证所有指令解码正确性
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_decoder;

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT 信号
    //------------------------------------------------------------------------
    reg  [31:0] instruction;
    reg         valid_in;
    wire        valid_out;
    wire [5:0]  opcode;
    wire [4:0]  rd, ra, rb, rc;
    wire [5:0]  func;
    wire [15:0] imm16;
    wire [20:0] imm21;
    wire        use_imm;
    wire        alu_op, mul_op, div_op;
    wire        mem_read, mem_write, mem_shared;
    wire        branch_op, sync_op, special_reg;
    wire        reg_write, pred_write;
    wire [2:0]  pred_addr;
    wire        illegal_inst;

    //------------------------------------------------------------------------
    // DUT 实例化
    //------------------------------------------------------------------------
    decoder dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .instruction (instruction),
        .valid_in    (valid_in),
        .valid_out   (valid_out),
        .opcode      (opcode),
        .rd          (rd),
        .ra          (ra),
        .rb          (rb),
        .rc          (rc),
        .func        (func),
        .imm16       (imm16),
        .imm21       (imm21),
        .use_imm     (use_imm),
        .alu_op      (alu_op),
        .mul_op      (mul_op),
        .div_op      (div_op),
        .mem_read    (mem_read),
        .mem_write   (mem_write),
        .mem_shared  (mem_shared),
        .branch_op   (branch_op),
        .sync_op     (sync_op),
        .special_reg (special_reg),
        .reg_write   (reg_write),
        .pred_write  (pred_write),
        .pred_addr   (pred_addr),
        .illegal_inst(illegal_inst)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;

    //------------------------------------------------------------------------
    // 辅助函数 - 构造指令
    //------------------------------------------------------------------------
    function [31:0] make_alu_inst;
        input [5:0] op;
        input [4:0] dst, src_a, src_b, src_c;
        input [5:0] fn;
        begin
            make_alu_inst = {op, dst, src_a, src_b, src_c, fn};
        end
    endfunction

    //------------------------------------------------------------------------
    // 测试任务
    //------------------------------------------------------------------------
    task decode_and_check;
        input [31:0] inst;
        input        exp_alu_op;
        input        exp_mul_op;
        input        exp_mem_read;
        input        exp_mem_write;
        input        exp_mem_shared;
        input        exp_branch_op;
        input        exp_sync_op;
        input        exp_special_reg;
        input        exp_reg_write;
        input        exp_illegal_inst;
        input [127:0] test_name;
        reg all_match;
        begin
            @(posedge clk);
            instruction <= inst;
            valid_in    <= 1;

            @(posedge clk);
            valid_in <= 0;

            @(posedge clk);  // 等待输出

            all_match = (alu_op === exp_alu_op) &&
                       (mul_op === exp_mul_op) &&
                       (mem_read === exp_mem_read) &&
                       (mem_write === exp_mem_write) &&
                       (mem_shared === exp_mem_shared) &&
                       (branch_op === exp_branch_op) &&
                       (sync_op === exp_sync_op) &&
                       (special_reg === exp_special_reg) &&
                       (reg_write === exp_reg_write) &&
                       (illegal_inst === exp_illegal_inst);

            if (valid_out && all_match) begin
                $display("[PASS] %s: opcode=0x%02X, rd=%0d, ra=%0d, rb=%0d, func=0x%02X",
                         test_name, opcode, rd, ra, rb, func);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s", test_name);
                $display("       Got: alu=%b mul=%b mem_r=%b mem_w=%b shared=%b br=%b sync=%b spec=%b reg_wr=%b ill=%b",
                         alu_op, mul_op, mem_read, mem_write, mem_shared, branch_op, sync_op, special_reg, reg_write, illegal_inst);
                $display("       Exp: alu=%b mul=%b mem_r=%b mem_w=%b shared=%b br=%b sync=%b spec=%b reg_wr=%b ill=%b",
                         exp_alu_op, exp_mul_op, exp_mem_read, exp_mem_write, exp_mem_shared,
                         exp_branch_op, exp_sync_op, exp_special_reg, exp_reg_write, exp_illegal_inst);
                failed = failed + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Decoder Unit Test");
        $display("============================================================");

        rst_n = 0;
        instruction = 0;
        valid_in = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // ALU 指令测试
        //====================================================================
        $display("\n--- ALU Instruction Tests ---");

        // add.s32 r1, r2, r3 (opcode=000000, func=000000)
        decode_and_check(
            make_alu_inst(`OP_ALU, 5'd1, 5'd2, 5'd3, 5'd0, `FUNC_ADD),
            1, 0, 0, 0, 0, 0, 0, 0, 1,  // alu_op=1, reg_write=1
            0, "ADD r1,r2,r3"
        );

        // sub.s32 r4, r5, r6
        decode_and_check(
            make_alu_inst(`OP_ALU, 5'd4, 5'd5, 5'd6, 5'd0, `FUNC_SUB),
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, "SUB r4,r5,r6"
        );

        // and.b32 r7, r8, r9
        decode_and_check(
            make_alu_inst(`OP_ALU, 5'd7, 5'd8, 5'd9, 5'd0, `FUNC_AND),
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, "AND r7,r8,r9"
        );

        // shl.b32 r10, r11, r12
        decode_and_check(
            make_alu_inst(`OP_ALU, 5'd10, 5'd11, 5'd12, 5'd0, `FUNC_SHL),
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, "SHL r10,r11,r12"
        );

        //====================================================================
        // MUL 指令测试
        //====================================================================
        $display("\n--- MUL Instruction Tests ---");

        // mul.lo r1, r2, r3
        decode_and_check(
            make_alu_inst(`OP_MUL, 5'd1, 5'd2, 5'd3, 5'd0, `FUNC_MUL_LO),
            0, 1, 0, 0, 0, 0, 0, 0, 1,  // mul_op=1, reg_write=1
            0, "MUL.LO r1,r2,r3"
        );

        // mad.lo r4, r5, r6, r7
        decode_and_check(
            make_alu_inst(`OP_MUL, 5'd4, 5'd5, 5'd6, 5'd7, `FUNC_MAD_LO),
            0, 1, 0, 0, 0, 0, 0, 0, 1,
            0, "MAD.LO r4,r5,r6,r7"
        );

        //====================================================================
        // 内存指令测试
        //====================================================================
        $display("\n--- Memory Instruction Tests ---");

        // ld.global r1, [r2]
        decode_and_check(
            make_alu_inst(`OP_LD_GLOBAL, 5'd1, 5'd2, 5'd0, 5'd0, 6'd0),
            0, 0, 1, 0, 0, 0, 0, 0, 1,  // mem_read=1, reg_write=1
            0, "LD.GLOBAL r1,[r2]"
        );

        // st.global [r1], r2
        decode_and_check(
            make_alu_inst(`OP_ST_GLOBAL, 5'd0, 5'd1, 5'd2, 5'd0, 6'd0),
            0, 0, 0, 1, 0, 0, 0, 0, 0,  // mem_write=1
            0, "ST.GLOBAL [r1],r2"
        );

        // ld.shared r3, [r4]
        decode_and_check(
            make_alu_inst(`OP_LD_SHARED, 5'd3, 5'd4, 5'd0, 5'd0, 6'd0),
            0, 0, 1, 0, 1, 0, 0, 0, 1,  // mem_read=1, mem_shared=1, reg_write=1
            0, "LD.SHARED r3,[r4]"
        );

        // st.shared [r5], r6
        decode_and_check(
            make_alu_inst(`OP_ST_SHARED, 5'd0, 5'd5, 5'd6, 5'd0, 6'd0),
            0, 0, 0, 1, 1, 0, 0, 0, 0,  // mem_write=1, mem_shared=1
            0, "ST.SHARED [r5],r6"
        );

        //====================================================================
        // 特殊寄存器指令测试
        //====================================================================
        $display("\n--- Special Register Tests ---");

        // mov r0, %tid.x
        decode_and_check(
            make_alu_inst(`OP_MOV_SPECIAL, 5'd0, `SREG_TID_X, 5'd0, 5'd0, 6'd0),
            0, 0, 0, 0, 0, 0, 0, 1, 1,  // special_reg=1, reg_write=1
            0, "MOV r0,%tid.x"
        );

        // mov r1, %ctaid.x
        decode_and_check(
            make_alu_inst(`OP_MOV_SPECIAL, 5'd1, `SREG_CTAID_X, 5'd0, 5'd0, 6'd0),
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, "MOV r1,%ctaid.x"
        );

        //====================================================================
        // 控制流指令测试
        //====================================================================
        $display("\n--- Control Flow Tests ---");

        // branch
        decode_and_check(
            make_alu_inst(`OP_BRANCH, 5'd0, 5'd0, 5'd0, 5'd0, 6'b111111),
            0, 0, 0, 0, 0, 1, 0, 0, 0,  // branch_op=1
            0, "BRANCH"
        );

        // bar.sync
        decode_and_check(
            make_alu_inst(`OP_BAR_SYNC, 5'd0, 5'd0, 5'd0, 5'd0, 6'b111111),
            0, 0, 0, 0, 0, 0, 1, 0, 0,  // sync_op=1
            0, "BAR.SYNC"
        );

        //====================================================================
        // NOP 指令测试
        //====================================================================
        $display("\n--- NOP Test ---");

        decode_and_check(
            make_alu_inst(`OP_NOP, 5'd0, 5'd0, 5'd0, 5'd0, 6'b111111),
            0, 0, 0, 0, 0, 0, 0, 0, 0,  // 所有控制信号为0
            0, "NOP"
        );

        //====================================================================
        // 寄存器字段验证
        //====================================================================
        $display("\n--- Register Field Tests ---");

        @(posedge clk);

        //====================================================================
        // Illegal Opcode and Edge Cases Tests
        //====================================================================
        $display("\n--- Illegal Opcode Tests ---");

        // Undefined opcode 6'b110101
        decode_and_check(
            make_alu_inst(6'b110101, 5'd0, 5'd0, 5'd0, 5'd0, 6'b111111),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // illegal_inst=1
            "Illegal FUNC under OP_STACK"
        );

        // Illegal FUNC under OP_ALU
        decode_and_check(
            make_alu_inst(`OP_ALU, 5'd1, 5'd2, 5'd3, 5'd0, 6'b111111),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // illegal_inst=1
            "Illegal FUNC under OP_ALU"
        );

        instruction <= make_alu_inst(`OP_ALU, 5'd31, 5'd30, 5'd29, 5'd28, `FUNC_ADD);
        valid_in <= 1;
        @(posedge clk);
        valid_in <= 0;
        @(posedge clk);

        if (rd === 5'd31 && ra === 5'd30 && rb === 5'd29 && rc === 5'd28) begin
            $display("[PASS] Register fields: rd=%0d, ra=%0d, rb=%0d, rc=%0d", rd, ra, rb, rc);
            passed = passed + 1;
        end else begin
            $display("[FAIL] Register fields: got rd=%0d, ra=%0d, rb=%0d, rc=%0d", rd, ra, rb, rc);
            failed = failed + 1;
        end

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Decoder Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_decoder.vcd");
        $dumpvars(0, tb_decoder);
    end

    initial begin
        #20000;
        $display("ERROR: Timeout!");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule