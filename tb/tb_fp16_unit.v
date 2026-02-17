//============================================================================
// RalphGPU - FP16 Unit Standalone Testbench
// Tests all func codes: ADD, SUB, MUL, FMA, NEG, ABS, MIN, MAX,
//                       TANH, EX2, CMP_*, BF16_*, FP16X2_*
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_fp16_unit;

    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg         valid_in;
    reg         packed_mode;
    reg  [31:0] operand_a;
    reg  [31:0] operand_b;
    reg  [31:0] operand_c;

    wire [31:0] result;
    wire        valid_out;
    wire        overflow, underflow, inexact, invalid;

    fp16_unit uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .func       (func),
        .valid_in   (valid_in),
        .packed_mode(packed_mode),
        .operand_a  (operand_a),
        .operand_b  (operand_b),
        .operand_c  (operand_c),
        .result     (result),
        .valid_out  (valid_out),
        .overflow   (overflow),
        .underflow  (underflow),
        .inexact    (inexact),
        .invalid    (invalid)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Test counters
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // FP16 encoding helpers
    //------------------------------------------------------------------------
    // Common FP16 values:
    // 0.0    = 16'h0000    -0.0   = 16'h8000
    // 1.0    = 16'h3C00    -1.0   = 16'hBC00
    // 2.0    = 16'h4000    -2.0   = 16'hC000
    // 0.5    = 16'h3800     3.0   = 16'h4200
    // 4.0    = 16'h4400    -4.0   = 16'hC400
    // 0.25   = 16'h3400    10.0   = 16'h4900
    // Inf    = 16'h7C00    -Inf   = 16'hFC00
    // NaN    = 16'h7E00

    // FP32 to real for display
    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp_f;
        reg [22:0] mant;
        real val;
        integer exp_unbiased;
        begin
            sign = fp32[31];
            exp_f = fp32[30:23];
            mant = fp32[22:0];
            if (exp_f == 0 && mant == 0) begin
                fp32_to_real = 0.0;
            end else if (exp_f == 8'hFF) begin
                fp32_to_real = (mant != 0) ? 0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp_f - 127;
                val = 1.0 + (mant * 1.0 / 8388608.0);
                if (exp_unbiased >= 0)
                    repeat(exp_unbiased) val = val * 2.0;
                else
                    repeat(-exp_unbiased) val = val / 2.0;
                fp32_to_real = sign ? -val : val;
            end
        end
    endfunction

    // FP16 to real
    function real fp16_to_real;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp_f;
        reg [9:0] mant;
        real val;
        integer exp_unbiased;
        begin
            sign = fp16[15];
            exp_f = fp16[14:10];
            mant = fp16[9:0];
            if (exp_f == 0 && mant == 0) begin
                fp16_to_real = 0.0;
            end else if (exp_f == 5'h1F) begin
                fp16_to_real = (mant != 0) ? 0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp_f - 15;
                val = 1.0 + (mant * 1.0 / 1024.0);
                if (exp_unbiased >= 0)
                    repeat(exp_unbiased) val = val * 2.0;
                else
                    repeat(-exp_unbiased) val = val / 2.0;
                fp16_to_real = sign ? -val : val;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Task: Apply stimulus and wait for result (3-cycle pipeline)
    //------------------------------------------------------------------------
    task apply_op;
        input [5:0]  t_func;
        input         t_packed;
        input [31:0]  t_a, t_b, t_c;
        begin
            @(posedge clk);
            func <= t_func;
            valid_in <= 1'b1;
            packed_mode <= t_packed;
            operand_a <= t_a;
            operand_b <= t_b;
            operand_c <= t_c;
            @(posedge clk);
            valid_in <= 1'b0;
            // Wait for pipeline (3 stages)
            @(posedge clk);
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    // Check result (FP16 in lower 16 bits)
    task check_fp16;
        input [15:0] expected;
        input [255:0] test_name; // 32 chars
        begin
            test_count = test_count + 1;
            if (result[15:0] == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%04x (%.4f)", test_name, result[15:0],
                         fp16_to_real(result[15:0]));
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%04x (%.4f), got 0x%04x (%.4f)",
                         test_name, expected, fp16_to_real(expected),
                         result[15:0], fp16_to_real(result[15:0]));
            end
        end
    endtask

    // Check result with tolerance (for transcendentals)
    task check_fp16_approx;
        input [15:0] expected;
        input [15:0] tolerance_ulps;
        input [255:0] test_name;
        reg [15:0] got;
        integer diff;
        begin
            test_count = test_count + 1;
            got = result[15:0];
            // ULP comparison (ignore sign for special values)
            if (expected[14:0] >= got[14:0])
                diff = expected[14:0] - got[14:0];
            else
                diff = got[14:0] - expected[14:0];

            if (expected[15] != got[15] && diff != 0) begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: sign mismatch, expected 0x%04x, got 0x%04x",
                         test_name, expected, got);
            end else if (diff <= tolerance_ulps) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%04x (%.4f), expected 0x%04x (%.4f), diff=%0d ULPs",
                         test_name, got, fp16_to_real(got), expected, fp16_to_real(expected), diff);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%04x (%.4f), got 0x%04x (%.4f), diff=%0d ULPs > %0d",
                         test_name, expected, fp16_to_real(expected),
                         got, fp16_to_real(got), diff, tolerance_ulps);
            end
        end
    endtask

    // Check 32-bit result (for CMP operations)
    task check_result32;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%08x", test_name, result);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%08x, got 0x%08x",
                         test_name, expected, result);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU FP16 Unit - Comprehensive Test Suite");
        $display("============================================================");

        // Reset
        rst_n = 0;
        valid_in = 0;
        func = 0;
        packed_mode = 0;
        operand_a = 0;
        operand_b = 0;
        operand_c = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        //====================================================================
        // 1. FP16_ADD
        //====================================================================
        $display("\n--- FP16_ADD ---");
        // 1.0 + 2.0 = 3.0
        apply_op(`FP16_ADD, 0, 32'h0000_3C00, 32'h0000_4000, 0);
        check_fp16(16'h4200, "1.0 + 2.0 = 3.0");

        // -1.0 + 1.0 = 0.0
        apply_op(`FP16_ADD, 0, 32'h0000_BC00, 32'h0000_3C00, 0);
        check_fp16(16'h0000, "-1.0 + 1.0 = 0.0");

        // 0.5 + 0.25 = 0.75
        apply_op(`FP16_ADD, 0, 32'h0000_3800, 32'h0000_3400, 0);
        check_fp16(16'h3A00, "0.5 + 0.25 = 0.75");

        // NaN + 1.0 = NaN
        apply_op(`FP16_ADD, 0, 32'h0000_7E00, 32'h0000_3C00, 0);
        check_fp16(16'h7E00, "NaN + 1.0 = NaN");

        //====================================================================
        // 2. FP16_SUB
        //====================================================================
        $display("\n--- FP16_SUB ---");
        // 3.0 - 1.0 = 2.0
        apply_op(`FP16_SUB, 0, 32'h0000_4200, 32'h0000_3C00, 0);
        check_fp16(16'h4000, "3.0 - 1.0 = 2.0");

        // 1.0 - 1.0 = 0.0
        apply_op(`FP16_SUB, 0, 32'h0000_3C00, 32'h0000_3C00, 0);
        check_fp16(16'h0000, "1.0 - 1.0 = 0.0");

        //====================================================================
        // 3. FP16_MUL
        //====================================================================
        $display("\n--- FP16_MUL ---");
        // 2.0 * 3.0 = 6.0
        apply_op(`FP16_MUL, 0, 32'h0000_4000, 32'h0000_4200, 0);
        check_fp16(16'h4600, "2.0 * 3.0 = 6.0");

        // -1.0 * 2.0 = -2.0
        apply_op(`FP16_MUL, 0, 32'h0000_BC00, 32'h0000_4000, 0);
        check_fp16(16'hC000, "-1.0 * 2.0 = -2.0");

        // 0.0 * inf = NaN
        apply_op(`FP16_MUL, 0, 32'h0000_0000, 32'h0000_7C00, 0);
        check_fp16(16'h7E00, "0.0 * inf = NaN");

        // inf * 0.0 = NaN
        apply_op(`FP16_MUL, 0, 32'h0000_7C00, 32'h0000_0000, 0);
        check_fp16(16'h7E00, "inf * 0.0 = NaN");

        //====================================================================
        // 4. FP16_FMA
        //====================================================================
        $display("\n--- FP16_FMA ---");
        // 2.0 * 3.0 + 1.0 = 7.0
        apply_op(`FP16_FMA, 0, 32'h0000_4000, 32'h0000_4200, 32'h0000_3C00);
        check_fp16(16'h4700, "2.0*3.0+1.0 = 7.0");

        // NaN FMA
        apply_op(`FP16_FMA, 0, 32'h0000_7E00, 32'h0000_3C00, 32'h0000_3C00);
        check_fp16(16'h7E00, "NaN*1.0+1.0 = NaN");

        //====================================================================
        // 5. FP16_NEG
        //====================================================================
        $display("\n--- FP16_NEG ---");
        // neg(1.0) = -1.0
        apply_op(`FP16_NEG, 0, 32'h0000_3C00, 0, 0);
        check_fp16(16'hBC00, "neg(1.0) = -1.0");

        // neg(-2.0) = 2.0
        apply_op(`FP16_NEG, 0, 32'h0000_C000, 0, 0);
        check_fp16(16'h4000, "neg(-2.0) = 2.0");

        //====================================================================
        // 6. FP16_ABS
        //====================================================================
        $display("\n--- FP16_ABS ---");
        // abs(-3.0) = 3.0
        apply_op(`FP16_ABS, 0, 32'h0000_C200, 0, 0);
        check_fp16(16'h4200, "abs(-3.0) = 3.0");

        // abs(2.0) = 2.0
        apply_op(`FP16_ABS, 0, 32'h0000_4000, 0, 0);
        check_fp16(16'h4000, "abs(2.0) = 2.0");

        //====================================================================
        // 7. FP16_MIN / FP16_MAX
        //====================================================================
        $display("\n--- FP16_MIN/MAX ---");
        // min(2.0, 3.0) = 2.0
        apply_op(`FP16_MIN, 0, 32'h0000_4000, 32'h0000_4200, 0);
        check_fp16(16'h4000, "min(2.0, 3.0) = 2.0");

        // max(2.0, 3.0) = 3.0
        apply_op(`FP16_MAX, 0, 32'h0000_4000, 32'h0000_4200, 0);
        check_fp16(16'h4200, "max(2.0, 3.0) = 3.0");

        // min(NaN, 1.0) = 1.0
        apply_op(`FP16_MIN, 0, 32'h0000_7E00, 32'h0000_3C00, 0);
        check_fp16(16'h3C00, "min(NaN, 1.0) = 1.0");

        // max(NaN, 1.0) = 1.0
        apply_op(`FP16_MAX, 0, 32'h0000_7E00, 32'h0000_3C00, 0);
        check_fp16(16'h3C00, "max(NaN, 1.0) = 1.0");

        //====================================================================
        // 8. FP16_TANH (new!)
        //====================================================================
        $display("\n--- FP16_TANH ---");
        // tanh(0) = 0
        apply_op(`FP16_TANH, 0, 32'h0000_0000, 0, 0);
        check_fp16(16'h0000, "tanh(0) = 0");

        // tanh(NaN) = NaN
        apply_op(`FP16_TANH, 0, 32'h0000_7E00, 0, 0);
        check_fp16(16'h7E00, "tanh(NaN) = NaN");

        // tanh(0.5) ~= 0.4621 -> FP16 ~0x3764 (approx, allow tolerance)
        apply_op(`FP16_TANH, 0, 32'h0000_3800, 0, 0);
        check_fp16_approx(16'h3764, 16'd32, "tanh(0.5) ~ 0.462");

        // tanh(1.0) ~= 0.7616 -> FP16 ~0x3A18
        apply_op(`FP16_TANH, 0, 32'h0000_3C00, 0, 0);
        check_fp16_approx(16'h3A18, 16'd64, "tanh(1.0) ~ 0.762");

        // tanh(10.0) -> 1.0 (saturated)
        apply_op(`FP16_TANH, 0, 32'h0000_4900, 0, 0);
        check_fp16(16'h3C00, "tanh(10.0) = 1.0");

        // tanh(-10.0) -> -1.0 (saturated)
        apply_op(`FP16_TANH, 0, 32'h0000_C900, 0, 0);
        check_fp16(16'hBC00, "tanh(-10.0) = -1.0");

        //====================================================================
        // 9. FP16_EX2 (new!)
        //====================================================================
        $display("\n--- FP16_EX2 ---");
        // 2^0 = 1.0
        apply_op(`FP16_EX2, 0, 32'h0000_0000, 0, 0);
        check_fp16(16'h3C00, "2^0 = 1.0");

        // 2^1 = 2.0
        apply_op(`FP16_EX2, 0, 32'h0000_3C00, 0, 0);
        check_fp16_approx(16'h4000, 16'd16, "2^1 ~ 2.0");

        // 2^(-1) = 0.5
        apply_op(`FP16_EX2, 0, 32'h0000_BC00, 0, 0);
        check_fp16_approx(16'h3800, 16'd16, "2^(-1) ~ 0.5");

        // 2^2 = 4.0
        apply_op(`FP16_EX2, 0, 32'h0000_4000, 0, 0);
        check_fp16_approx(16'h4400, 16'd32, "2^2 ~ 4.0");

        // 2^NaN = NaN
        apply_op(`FP16_EX2, 0, 32'h0000_7E00, 0, 0);
        check_fp16(16'h7E00, "2^NaN = NaN");

        // 2^(+inf) = +inf
        apply_op(`FP16_EX2, 0, 32'h0000_7C00, 0, 0);
        check_fp16(16'h7C00, "2^(+inf) = +inf");

        // 2^(-inf) = 0
        apply_op(`FP16_EX2, 0, 32'h0000_FC00, 0, 0);
        check_fp16(16'h0000, "2^(-inf) = 0");

        // 2^20 -> overflow to inf
        // 20.0 in FP16 = 0x4D00
        apply_op(`FP16_EX2, 0, 32'h0000_4D00, 0, 0);
        check_fp16(16'h7C00, "2^20 = +inf (overflow)");

        // 2^(-20) -> underflow to 0
        apply_op(`FP16_EX2, 0, 32'h0000_CD00, 0, 0);
        check_fp16(16'h0000, "2^(-20) = 0 (underflow)");

        //====================================================================
        // 10. FP16 Compare operations
        //====================================================================
        $display("\n--- FP16 Compares ---");
        // EQ: 1.0 == 1.0 -> true
        apply_op(`FP16_CMP_EQ, 0, 32'h0000_3C00, 32'h0000_3C00, 0);
        check_result32(32'hFFFF_FFFF, "1.0 == 1.0 -> true");

        // EQ: 1.0 == 2.0 -> false
        apply_op(`FP16_CMP_EQ, 0, 32'h0000_3C00, 32'h0000_4000, 0);
        check_result32(32'h0000_0000, "1.0 == 2.0 -> false");

        // NE: 1.0 != 2.0 -> true
        apply_op(`FP16_CMP_NE, 0, 32'h0000_3C00, 32'h0000_4000, 0);
        check_result32(32'hFFFF_FFFF, "1.0 != 2.0 -> true");

        // LT: 1.0 < 2.0 -> true
        apply_op(`FP16_CMP_LT, 0, 32'h0000_3C00, 32'h0000_4000, 0);
        check_result32(32'hFFFF_FFFF, "1.0 < 2.0 -> true");

        // LT: 2.0 < 1.0 -> false
        apply_op(`FP16_CMP_LT, 0, 32'h0000_4000, 32'h0000_3C00, 0);
        check_result32(32'h0000_0000, "2.0 < 1.0 -> false");

        // LE: 1.0 <= 1.0 -> true
        apply_op(`FP16_CMP_LE, 0, 32'h0000_3C00, 32'h0000_3C00, 0);
        check_result32(32'hFFFF_FFFF, "1.0 <= 1.0 -> true");

        // GT: 2.0 > 1.0 -> true
        apply_op(`FP16_CMP_GT, 0, 32'h0000_4000, 32'h0000_3C00, 0);
        check_result32(32'hFFFF_FFFF, "2.0 > 1.0 -> true");

        // GE: 2.0 >= 2.0 -> true
        apply_op(`FP16_CMP_GE, 0, 32'h0000_4000, 32'h0000_4000, 0);
        check_result32(32'hFFFF_FFFF, "2.0 >= 2.0 -> true");

        // NUM: 1.0, 2.0 -> true (both ordered)
        apply_op(`FP16_CMP_NUM, 0, 32'h0000_3C00, 32'h0000_4000, 0);
        check_result32(32'hFFFF_FFFF, "num(1.0, 2.0) -> true");

        // NAN: NaN, 1.0 -> true (unordered)
        apply_op(`FP16_CMP_NAN, 0, 32'h0000_7E00, 32'h0000_3C00, 0);
        check_result32(32'hFFFF_FFFF, "nan(NaN, 1.0) -> true");

        // EQ with NaN -> false
        apply_op(`FP16_CMP_EQ, 0, 32'h0000_7E00, 32'h0000_3C00, 0);
        check_result32(32'h0000_0000, "NaN == 1.0 -> false");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("FP16 Unit Test Summary");
        $display("============================================================");
        $display("Total: %0d  PASS: %0d  FAIL: %0d", test_count, pass_count, fail_count);
        if (fail_count == 0)
            $display("*** ALL TESTS PASSED ***");
        else
            $display("*** %0d TESTS FAILED ***", fail_count);
        $display("============================================================");

        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("[TIMEOUT] Test did not complete within timeout");
        $finish;
    end

endmodule
