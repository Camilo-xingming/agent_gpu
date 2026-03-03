//============================================================================
// RalphGPU - ALU Unit Test
// 验证所有ALU算术逻辑运算
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_alu;

    //------------------------------------------------------------------------
    // 测试信号
    //------------------------------------------------------------------------
    reg  [5:0]  func;
    reg  [31:0] operand_a;
    reg  [31:0] operand_b;
    wire [31:0] result;
    wire        zero;
    wire        negative;
    wire        overflow;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    alu dut (
        .func      (func),
        .operand_a (operand_a),
        .operand_b (operand_b),
        .result    (result),
        .zero      (zero),
        .negative  (negative),
        .overflow  (overflow)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;
    reg [31:0] expected;

    //------------------------------------------------------------------------
    // 测试任务
    //------------------------------------------------------------------------
    task check_result;
        input [31:0] exp_result;
        input        exp_zero;
        input        exp_neg;
        input        exp_ovf;
        input [63:0] test_name;
        begin
            #1;  // 等待组合逻辑稳定
            if (result === exp_result && zero === exp_zero && negative === exp_neg && overflow === exp_ovf) begin
                $display("[PASS] %s: result=0x%08X, zero=%b, neg=%b, ovf=%b",
                         test_name, result, zero, negative, overflow);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s: got 0x%08X (z=%b,n=%b,o=%b), expected 0x%08X (z=%b,n=%b,o=%b)",
                         test_name, result, zero, negative, overflow, exp_result, exp_zero, exp_neg, exp_ovf);
                failed = failed + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU ALU Unit Test");
        $display("============================================================");

        //====================================================================
        // ADD 测试
        //====================================================================
        $display("\n--- ADD Tests ---");

        // 正数加正数
        func = `FUNC_ADD; operand_a = 32'd100; operand_b = 32'd200;
        check_result(32'd300, 1'b0, 1'b0, 1'b0, "ADD 100+200");

        // 加零
        func = `FUNC_ADD; operand_a = 32'd50; operand_b = 32'd0;
        check_result(32'd50, 1'b0, 1'b0, 1'b0, "ADD 50+0");

        // 结果为零
        func = `FUNC_ADD; operand_a = 32'd0; operand_b = 32'd0;
        check_result(32'd0, 1'b1, 1'b0, 1'b0, "ADD 0+0");

        // 负数(补码)
        func = `FUNC_ADD; operand_a = 32'hFFFFFFFF; operand_b = 32'd1;
        check_result(32'd0, 1'b1, 1'b0, 1'b0, "ADD -1+1");

        // 大数相加
        func = `FUNC_ADD; operand_a = 32'h7FFFFFFF; operand_b = 32'd1;
        check_result(32'h80000000, 1'b0, 1'b1, 1'b1, "ADD MAX+1 (overflow)");

        //====================================================================
        // SUB 测试
        //====================================================================
        $display("\n--- SUB Tests ---");

        func = `FUNC_SUB; operand_a = 32'd300; operand_b = 32'd100;
        check_result(32'd200, 1'b0, 1'b0, 1'b0, "SUB 300-100");

        func = `FUNC_SUB; operand_a = 32'd100; operand_b = 32'd100;
        check_result(32'd0, 1'b1, 1'b0, 1'b0, "SUB 100-100");

        func = `FUNC_SUB; operand_a = 32'd50; operand_b = 32'd100;
        check_result(32'hFFFFFFCE, 1'b0, 1'b1, 1'b0, "SUB 50-100 (negative)");

        //====================================================================
        // AND 测试
        //====================================================================
        $display("\n--- AND Tests ---");

        func = `FUNC_AND; operand_a = 32'hFF00FF00; operand_b = 32'h0F0F0F0F;
        check_result(32'h0F000F00, 1'b0, 1'b0, 1'b0, "AND patterns");

        func = `FUNC_AND; operand_a = 32'hFFFFFFFF; operand_b = 32'h00000000;
        check_result(32'h00000000, 1'b1, 1'b0, 1'b0, "AND all-ones & zeros");

        //====================================================================
        // OR 测试
        //====================================================================
        $display("\n--- OR Tests ---");

        func = `FUNC_OR; operand_a = 32'hFF00FF00; operand_b = 32'h00FF00FF;
        check_result(32'hFFFFFFFF, 1'b0, 1'b1, 1'b0, "OR patterns");

        func = `FUNC_OR; operand_a = 32'h00000000; operand_b = 32'h00000000;
        check_result(32'h00000000, 1'b1, 1'b0, 1'b0, "OR zeros");

        //====================================================================
        // XOR 测试
        //====================================================================
        $display("\n--- XOR Tests ---");

        func = `FUNC_XOR; operand_a = 32'hAAAAAAAA; operand_b = 32'h55555555;
        check_result(32'hFFFFFFFF, 1'b0, 1'b1, 1'b0, "XOR alt patterns");

        func = `FUNC_XOR; operand_a = 32'h12345678; operand_b = 32'h12345678;
        check_result(32'h00000000, 1'b1, 1'b0, 1'b0, "XOR same value");

        //====================================================================
        // NOT 测试
        //====================================================================
        $display("\n--- NOT Tests ---");

        func = `FUNC_NOT; operand_a = 32'hFFFFFFFF; operand_b = 32'd0;
        check_result(32'h00000000, 1'b1, 1'b0, 1'b0, "NOT all-ones");

        func = `FUNC_NOT; operand_a = 32'h00000000; operand_b = 32'd0;
        check_result(32'hFFFFFFFF, 1'b0, 1'b1, 1'b0, "NOT zeros");

        //====================================================================
        // SHL 测试
        //====================================================================
        $display("\n--- SHL Tests ---");

        func = `FUNC_SHL; operand_a = 32'd1; operand_b = 32'd4;
        check_result(32'd16, 1'b0, 1'b0, 1'b0, "SHL 1<<4");

        func = `FUNC_SHL; operand_a = 32'h80000000; operand_b = 32'd1;
        check_result(32'h00000000, 1'b1, 1'b0, 1'b0, "SHL overflow");

        func = `FUNC_SHL; operand_a = 32'h00000001; operand_b = 32'd31;
        check_result(32'h80000000, 1'b0, 1'b1, 1'b0, "SHL to MSB");

        //====================================================================
        // SHR_U (逻辑右移) 测试
        //====================================================================
        $display("\n--- SHR_U Tests ---");

        func = `FUNC_SHR_U; operand_a = 32'd16; operand_b = 32'd4;
        check_result(32'd1, 1'b0, 1'b0, 1'b0, "SHR_U 16>>4");

        func = `FUNC_SHR_U; operand_a = 32'h80000000; operand_b = 32'd4;
        check_result(32'h08000000, 1'b0, 1'b0, 1'b0, "SHR_U sign bit");

        //====================================================================
        // SHR_S (算术右移) 测试
        //====================================================================
        $display("\n--- SHR_S Tests ---");

        func = `FUNC_SHR_S; operand_a = 32'h80000000; operand_b = 32'd4;
        check_result(32'hF8000000, 1'b0, 1'b1, 1'b0, "SHR_S sign extend");

        func = `FUNC_SHR_S; operand_a = 32'h7FFFFFFF; operand_b = 32'd4;
        check_result(32'h07FFFFFF, 1'b0, 1'b0, 1'b0, "SHR_S positive");

        //====================================================================
        // 边界条件测试
        //====================================================================
        $display("\n--- Boundary Tests ---");

        // 最大值
        func = `FUNC_ADD; operand_a = 32'hFFFFFFFF; operand_b = 32'hFFFFFFFF;
        check_result(32'hFFFFFFFE, 1'b0, 1'b1, 1'b0, "ADD max+max");

        // 移位边界
        func = `FUNC_SHL; operand_a = 32'hFFFFFFFF; operand_b = 32'd0;
        check_result(32'hFFFFFFFF, 1'b0, 1'b1, 1'b0, "SHL by 0");

        func = `FUNC_SHL; operand_a = 32'hFFFFFFFF; operand_b = 32'd32;
        check_result(32'hFFFFFFFF, 1'b0, 1'b1, 1'b0, "SHL by 32 (uses low 5 bits)");

        //====================================================================
        // 测试总结
        //====================================================================
        $display("\n============================================================");
        $display("ALU Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        $finish;
    end

endmodule
