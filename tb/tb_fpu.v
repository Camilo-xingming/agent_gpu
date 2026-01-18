//============================================================================
// RalphGPU - FPU Testbench
// 测试浮点运算单元
//============================================================================

`timescale 1ns/1ps
`include "../rtl/gpu_defines.vh"

module tb_fpu;

    //------------------------------------------------------------------------
    // 信号定义
    //------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg  [1:0]  rnd_mode;
    reg         ftz;
    reg  [31:0] operand_a;
    reg  [31:0] operand_b;
    reg  [31:0] operand_c;
    reg         valid_in;

    wire [31:0] result;
    wire        valid_out;
    wire        overflow;
    wire        underflow;
    wire        inexact;
    wire        invalid;
    wire        div_by_zero;

    //------------------------------------------------------------------------
    // DUT实例化
    //------------------------------------------------------------------------
    fpu u_fpu (
        .clk        (clk),
        .rst_n      (rst_n),
        .func       (func),
        .rnd_mode   (rnd_mode),
        .ftz        (ftz),
        .operand_a  (operand_a),
        .operand_b  (operand_b),
        .operand_c  (operand_c),
        .valid_in   (valid_in),
        .result     (result),
        .valid_out  (valid_out),
        .overflow   (overflow),
        .underflow  (underflow),
        .inexact    (inexact),
        .invalid    (invalid),
        .div_by_zero(div_by_zero)
    );

    //------------------------------------------------------------------------
    // 时钟生成
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // 测试计数器
    //------------------------------------------------------------------------
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // 辅助函数: IEEE 754 浮点数构造
    //------------------------------------------------------------------------
    function [31:0] make_fp32;
        input        sign;
        input [7:0]  exp;
        input [22:0] mantissa;
        begin
            make_fp32 = {sign, exp, mantissa};
        end
    endfunction

    // 常用浮点数常量
    localparam [31:0] FP_ZERO     = 32'h00000000;  // +0.0
    localparam [31:0] FP_NEG_ZERO = 32'h80000000;  // -0.0
    localparam [31:0] FP_ONE      = 32'h3F800000;  // 1.0
    localparam [31:0] FP_NEG_ONE  = 32'hBF800000;  // -1.0
    localparam [31:0] FP_TWO      = 32'h40000000;  // 2.0
    localparam [31:0] FP_HALF     = 32'h3F000000;  // 0.5
    localparam [31:0] FP_THREE    = 32'h40400000;  // 3.0
    localparam [31:0] FP_FOUR     = 32'h40800000;  // 4.0
    localparam [31:0] FP_NEG_TWO  = 32'hC0000000;  // -2.0
    localparam [31:0] FP_INF      = 32'h7F800000;  // +Inf
    localparam [31:0] FP_NEG_INF  = 32'hFF800000;  // -Inf
    localparam [31:0] FP_NAN      = 32'h7FC00000;  // NaN

    //------------------------------------------------------------------------
    // 测试任务
    //------------------------------------------------------------------------
    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            @(posedge clk);
            valid_in = 1'b1;
            @(posedge clk);
            valid_in = 1'b0;

            // 等待结果
            wait(valid_out);
            @(posedge clk);

            if (result === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%08h == 0x%08h", test_name, result, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got 0x%08h, expected 0x%08h", test_name, result, expected);
            end
        end
    endtask

    task check_result_approx;
        input [31:0] expected;
        input [7:0]  tolerance;  // 允许的ULP误差
        input [255:0] test_name;
        reg [31:0] diff;
        begin
            test_count = test_count + 1;
            @(posedge clk);
            valid_in = 1'b1;
            @(posedge clk);
            valid_in = 1'b0;

            // 等待结果
            wait(valid_out);
            @(posedge clk);

            // 计算ULP差异
            if (result > expected)
                diff = result - expected;
            else
                diff = expected - result;

            if (diff <= tolerance) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%08h ~ 0x%08h (diff=%0d ULP)", test_name, result, expected, diff);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got 0x%08h, expected 0x%08h (diff=%0d ULP)", test_name, result, expected, diff);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 主测试
    //------------------------------------------------------------------------
    initial begin
        $display("============================================");
        $display("RalphGPU FPU Testbench");
        $display("============================================");

        // 初始化
        rst_n     = 0;
        func      = 6'b0;
        rnd_mode  = 2'b00;  // RN
        ftz       = 1'b0;
        operand_a = 32'b0;
        operand_b = 32'b0;
        operand_c = 32'b0;
        valid_in  = 1'b0;

        // 复位
        #20;
        rst_n = 1;
        #10;

        //====================================================================
        // 测试 FP_ADD
        //====================================================================
        $display("\n--- Testing FP_ADD ---");
        func = `FP_ADD;

        operand_a = FP_ONE;   // 1.0
        operand_b = FP_ONE;   // 1.0
        check_result(FP_TWO, "1.0 + 1.0 = 2.0");

        operand_a = FP_ONE;
        operand_b = FP_NEG_ONE;
        check_result(FP_ZERO, "1.0 + (-1.0) = 0.0");

        operand_a = FP_INF;
        operand_b = FP_ONE;
        check_result(FP_INF, "Inf + 1.0 = Inf");

        operand_a = FP_INF;
        operand_b = FP_NEG_INF;
        check_result(FP_NAN, "Inf + (-Inf) = NaN");

        //====================================================================
        // 测试 FP_SUB
        //====================================================================
        $display("\n--- Testing FP_SUB ---");
        func = `FP_SUB;

        operand_a = FP_TWO;
        operand_b = FP_ONE;
        check_result(FP_ONE, "2.0 - 1.0 = 1.0");

        operand_a = FP_ONE;
        operand_b = FP_ONE;
        check_result(FP_ZERO, "1.0 - 1.0 = 0.0");

        //====================================================================
        // 测试 FP_MUL
        //====================================================================
        $display("\n--- Testing FP_MUL ---");
        func = `FP_MUL;

        operand_a = FP_TWO;
        operand_b = FP_TWO;
        check_result(FP_FOUR, "2.0 * 2.0 = 4.0");

        operand_a = FP_TWO;
        operand_b = FP_HALF;
        check_result(FP_ONE, "2.0 * 0.5 = 1.0");

        operand_a = FP_ONE;
        operand_b = FP_ZERO;
        check_result(FP_ZERO, "1.0 * 0.0 = 0.0");

        operand_a = FP_INF;
        operand_b = FP_ZERO;
        check_result(FP_NAN, "Inf * 0.0 = NaN");

        //====================================================================
        // 测试 FP_NEG
        //====================================================================
        $display("\n--- Testing FP_NEG ---");
        func = `FP_NEG;

        operand_a = FP_ONE;
        check_result(FP_NEG_ONE, "neg(1.0) = -1.0");

        operand_a = FP_NEG_ONE;
        check_result(FP_ONE, "neg(-1.0) = 1.0");

        operand_a = FP_ZERO;
        check_result(FP_NEG_ZERO, "neg(0.0) = -0.0");

        //====================================================================
        // 测试 FP_ABS
        //====================================================================
        $display("\n--- Testing FP_ABS ---");
        func = `FP_ABS;

        operand_a = FP_NEG_ONE;
        check_result(FP_ONE, "abs(-1.0) = 1.0");

        operand_a = FP_ONE;
        check_result(FP_ONE, "abs(1.0) = 1.0");

        operand_a = FP_NEG_ZERO;
        check_result(FP_ZERO, "abs(-0.0) = 0.0");

        //====================================================================
        // 测试 FP_MIN
        //====================================================================
        $display("\n--- Testing FP_MIN ---");
        func = `FP_MIN;

        operand_a = FP_ONE;
        operand_b = FP_TWO;
        check_result(FP_ONE, "min(1.0, 2.0) = 1.0");

        operand_a = FP_NEG_ONE;
        operand_b = FP_ONE;
        check_result(FP_NEG_ONE, "min(-1.0, 1.0) = -1.0");

        //====================================================================
        // 测试 FP_MAX
        //====================================================================
        $display("\n--- Testing FP_MAX ---");
        func = `FP_MAX;

        operand_a = FP_ONE;
        operand_b = FP_TWO;
        check_result(FP_TWO, "max(1.0, 2.0) = 2.0");

        operand_a = FP_NEG_ONE;
        operand_b = FP_ONE;
        check_result(FP_ONE, "max(-1.0, 1.0) = 1.0");

        //====================================================================
        // 测试 FP_FMA (Fused Multiply-Add)
        //====================================================================
        $display("\n--- Testing FP_FMA ---");
        func = `FP_FMA;

        operand_a = FP_TWO;
        operand_b = FP_THREE;
        operand_c = FP_ONE;
        check_result_approx(32'h40E00000, 8'd10, "fma(2.0, 3.0, 1.0) = 7.0");

        operand_a = FP_ONE;
        operand_b = FP_ONE;
        operand_c = FP_ONE;
        check_result_approx(FP_TWO, 8'd2, "fma(1.0, 1.0, 1.0) = 2.0");

        //====================================================================
        // 测试 COPYSIGN / TESTP
        //====================================================================
        $display("\n--- Testing COPYSIGN/TESTP ---");
        func = `FP_COPYSIGN;
        operand_a = FP_ONE; operand_b = FP_NEG_ZERO;
        check_result(FP_NEG_ONE, "copysign(1.0, -0.0) = -1.0");
        operand_a = FP_NEG_TWO; operand_b = FP_ZERO;
        check_result(FP_TWO, "copysign(-2.0, +0.0) = +2.0");

        func = `FP_TESTP;
        operand_a = FP_ONE;
        check_result(FP_ONE, "testp(1.0) -> 1.0");
        operand_a = FP_NAN;
        check_result(32'h00000000, "testp(NaN) -> 0");

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================");
        $display("Test Summary:");
        $display("  Total:  %0d", test_count);
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("============================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end

        $finish;
    end

endmodule
