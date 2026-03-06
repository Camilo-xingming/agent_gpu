//============================================================================
// RalphGPU - Multiply Unit Test
// 验证 mul.lo, mul.hi, mad 操作
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_mul_unit;

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
    end

    //------------------------------------------------------------------------
    // DUT 信号
    //------------------------------------------------------------------------
    reg         valid_in;
    reg  [5:0]  func;
    reg  [31:0] operand_a;
    reg  [31:0] operand_b;
    reg  [31:0] operand_c;
    reg         is_div;
    wire        valid_out;
    wire [31:0] result;

    //------------------------------------------------------------------------
    // DUT 实例化
    //------------------------------------------------------------------------
    mul_unit dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .is_div    (is_div),
        .func      (func),
        .operand_a (operand_a),
        .operand_b (operand_b),
        .operand_c (operand_c),
        .valid_out (valid_out),
        .result    (result)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;

    //------------------------------------------------------------------------
    // 测试任务
    //------------------------------------------------------------------------
    task test_mul;
        input [5:0]  test_func;
        input [31:0] a, b, c;
        input [31:0] expected;
        input [127:0] test_name;
        begin
            @(posedge clk);
            valid_in  <= 1;
            is_div    <= 0;
            func      <= test_func;
            operand_a <= a;
            operand_b <= b;
            operand_c <= c;

            @(posedge clk);
            valid_in <= 0;

            // 等待流水线输出 (2级流水线)
            @(posedge clk);
            @(posedge clk);

            if (valid_out && result === expected) begin
                $display("[PASS] %s: %0d * %0d = 0x%08X", test_name, a, b, result);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s: got 0x%08X, expected 0x%08X (valid=%b)",
                         test_name, result, expected, valid_out);
                failed = failed + 1;
            end
        end
    endtask

    task test_div;
        input [5:0]  div_func;
        input [31:0] a, b;
        input [31:0] expected;
        input [127:0] test_name;
        begin
            @(posedge clk);
            valid_in  <= 1;
            is_div    <= 1;
            func      <= div_func;
            operand_a <= a;
            operand_b <= b;
            operand_c <= 0;

            @(posedge clk);
            valid_in <= 0;
            is_div   <= 0;

            // 等待流水线输出 (2级流水线)
            @(posedge clk);
            @(posedge clk);

            if (valid_out && result === expected) begin
                $display("[PASS] %s: %0d / %0d = 0x%08X", test_name, a, b, result);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s: got 0x%08X, expected 0x%08X (valid=%b)",
                         test_name, result, expected, valid_out);
                failed = failed + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Multiply Unit Test");
        $display("============================================================");

        // 初始化
        rst_n     = 0;
        valid_in  = 0;
        func      = 0;
        operand_a = 0;
        operand_b = 0;
        operand_c = 0;
        is_div    = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // MUL.LO 测试 (取低32位)
        //====================================================================
        $display("\n--- MUL.LO Tests ---");

        // 基本乘法
        test_mul(`FUNC_MUL_LO, 32'd10, 32'd20, 32'd0, 32'd200, "MUL_LO 10*20");
        test_mul(`FUNC_MUL_LO, 32'd100, 32'd100, 32'd0, 32'd10000, "MUL_LO 100*100");

        // 乘0
        test_mul(`FUNC_MUL_LO, 32'd12345, 32'd0, 32'd0, 32'd0, "MUL_LO x*0");

        // 乘1
        test_mul(`FUNC_MUL_LO, 32'd12345, 32'd1, 32'd0, 32'd12345, "MUL_LO x*1");

        // 大数乘法 (溢出，只取低32位)
        // 0x10000 * 0x10000 = 0x100000000, 低32位 = 0
        test_mul(`FUNC_MUL_LO, 32'h00010000, 32'h00010000, 32'd0, 32'h00000000, "MUL_LO overflow");

        // 0xFFFF * 0xFFFF = 0xFFFE0001
        test_mul(`FUNC_MUL_LO, 32'h0000FFFF, 32'h0000FFFF, 32'd0, 32'hFFFE0001, "MUL_LO 0xFFFF^2");

        //====================================================================
        // MUL.HI 测试 (取高32位)
        //====================================================================
        $display("\n--- MUL.HI Tests ---");

        // 小数乘法，高位为0
        test_mul(`FUNC_MUL_HI, 32'd100, 32'd100, 32'd0, 32'd0, "MUL_HI 100*100");

        // 0x10000 * 0x10000 = 0x100000000, 高32位 = 1
        test_mul(`FUNC_MUL_HI, 32'h00010000, 32'h00010000, 32'd0, 32'h00000001, "MUL_HI overflow");

        // 有符号: -1 * 2 = -2, 高32位 = 0xFFFFFFFF (符号扩展)
        test_mul(`FUNC_MUL_HI, 32'hFFFFFFFF, 32'd2, 32'd0, 32'hFFFFFFFF, "MUL_HI -1*2 signed");

        // 有符号解释: -1 * -1 = 1 (高32位 = 0)
        test_mul(`FUNC_MUL_HI, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'd0, 32'h00000000, "MUL_HI signed -1*-1");

        //====================================================================
        // MAD.LO 测试 (乘加)
        //====================================================================
        $display("\n--- MAD.LO Tests ---");

        // 基本乘加: 10*20 + 5 = 205
        test_mul(`FUNC_MAD_LO, 32'd10, 32'd20, 32'd5, 32'd205, "MAD_LO 10*20+5");

        // 乘加为0: 0*100 + 0 = 0
        test_mul(`FUNC_MAD_LO, 32'd0, 32'd100, 32'd0, 32'd0, "MAD_LO 0*x+0");

        // SAXPY风格: a*x + y
        test_mul(`FUNC_MAD_LO, 32'd3, 32'd7, 32'd100, 32'd121, "MAD_LO 3*7+100");

        // 大数乘加
        test_mul(`FUNC_MAD_LO, 32'd1000, 32'd1000, 32'd999, 32'd1000999, "MAD_LO 1000*1000+999");

        //====================================================================
        // MAD.HI 测试 (高位乘加)
        //====================================================================
        $display("\n--- MAD.HI Tests ---");

        // 0x10000 * 0x10000 = 0x0001_0000_0000, 高位=1, 加上C=2 -> 3
        test_mul(`FUNC_MAD_HI, 32'h00010000, 32'h00010000, 32'd2, 32'd3, "MAD_HI 0x10000*0x10000 + 2");

        // 有符号: (-2 * 3) 高位=0xFFFFFFFF，再加1 -> 0
        test_mul(`FUNC_MAD_HI, 32'hFFFFFFFE, 32'd3, 32'd1, 32'h00000000, "MAD_HI signed -2*3 +1 high");

        //====================================================================
        // MUL24/MAD24 测试 (24-bit)
        //====================================================================
        $display("\n--- MUL24/MAD24 Tests ---");
        test_mul(`FUNC_MUL24, 32'h000000FF, 32'h00000010, 32'd0, 32'h00000FF0, "MUL24 0xFF * 0x10");
        test_mul(`FUNC_MUL24, 32'h00FFFF80, 32'h00000002, 32'd0, 32'hFFFFFF00, "MUL24 -128 * 2");
        test_mul(`FUNC_MAD24, 32'h00000100, 32'h00000010, 32'd5, 32'h00001005, "MAD24 0x100*0x10+5");
        test_mul(`FUNC_MAD24, 32'h00FFFFFF, 32'h00000002, 32'd1, 32'hFFFFFFFF, "MAD24 -1*2+1 (24-bit)");

        //====================================================================
        // 边界条件测试
        //====================================================================
        $display("\n--- Boundary Tests ---");

        // 最大正数 * 1
        test_mul(`FUNC_MUL_LO, 32'h7FFFFFFF, 32'd1, 32'd0, 32'h7FFFFFFF, "MUL_LO MAX_POS*1");

        // 负数乘法 (有符号) -2 * 3 = -6
        test_mul(`FUNC_MUL_LO, 32'hFFFFFFFE, 32'd3, 32'd0, 32'hFFFFFFFA, "MUL_LO -2*3");

        //====================================================================
        // DIV/REM 测试
        //====================================================================
        $display("\n--- DIV/REM Tests ---");

        test_div(`DIV_FUNC_DIV_S, 32'sd10, 32'sd3, 32'sd3, "DIV_S 10/3");
        test_div(`DIV_FUNC_DIV_S, -32'sd10, 32'sd3, -32'sd3, "DIV_S -10/3");
        test_div(`DIV_FUNC_DIV_U, 32'hFFFFFFFE, 32'd2, 32'h7FFFFFFF, "DIV_U 0xFFFFFFFE/2");
        test_div(`DIV_FUNC_REM_S, -32'sd10, 32'sd3, -32'sd1, "REM_S -10%3");
        test_div(`DIV_FUNC_REM_U, 32'd10, 32'd3, 32'd1, "REM_U 10%3");

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Multiply Unit Test Summary: %0d PASSED, %0d FAILED", passed, failed);
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
        $dumpfile("tb_mul_unit.vcd");
        $dumpvars(0, tb_mul_unit);
    end

    // 超时保护
    initial begin
        #10000;
        $display("ERROR: Timeout!");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
