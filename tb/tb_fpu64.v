//============================================================================
// RalphGPU - FPU64 (Double-Precision) Standalone Testbench
// Tests: add, sub, mul, div, fma, neg, abs, min, max, sqrt, rcp, copysign, testp
// Special values: NaN, Inf, Zero, Denorm
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_fpu64;

    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg  [1:0]  rnd_mode;
    reg         ftz;
    reg         valid_in;
    reg  [63:0] operand_a;
    reg  [63:0] operand_b;
    reg  [63:0] operand_c;

    wire [63:0] result;
    wire        valid_out;
    wire        overflow, underflow, inexact, invalid, div_by_zero;

    fpu64 uut (
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

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Test counters
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // IEEE 754 FP64 constants
    //------------------------------------------------------------------------
    // +1.0  = 64'h3FF0_0000_0000_0000
    // -1.0  = 64'hBFF0_0000_0000_0000
    // +2.0  = 64'h4000_0000_0000_0000
    // -2.0  = 64'hC000_0000_0000_0000
    // +3.0  = 64'h4008_0000_0000_0000
    // +0.5  = 64'h3FE0_0000_0000_0000
    // +0.25 = 64'h3FD0_0000_0000_0000
    // +4.0  = 64'h4010_0000_0000_0000
    // +6.0  = 64'h4018_0000_0000_0000
    // +7.0  = 64'h401C_0000_0000_0000
    // +10.0 = 64'h4024_0000_0000_0000
    // +0.0  = 64'h0000_0000_0000_0000
    // -0.0  = 64'h8000_0000_0000_0000
    // +Inf  = 64'h7FF0_0000_0000_0000
    // -Inf  = 64'hFFF0_0000_0000_0000
    // QNaN  = 64'h7FF8_0000_0000_0000

    //------------------------------------------------------------------------
    // Task: Apply stimulus and wait for result (1-cycle registered output)
    //------------------------------------------------------------------------
    task apply_op;
        input [5:0]  t_func;
        input [63:0] t_a, t_b, t_c;
        begin
            @(posedge clk);
            func <= t_func;
            rnd_mode <= 2'b00; // Round to nearest
            ftz <= 1'b0;
            valid_in <= 1'b1;
            operand_a <= t_a;
            operand_b <= t_b;
            operand_c <= t_c;
            @(posedge clk);
            valid_in <= 1'b0;
            @(posedge clk); // result available
        end
    endtask

    // Check exact 64-bit result
    task check_fp64;
        input [63:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%016x", test_name, result);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%016x, got 0x%016x",
                         test_name, expected, result);
            end
        end
    endtask

    // Check result with ULP tolerance (mantissa-only comparison, same exponent expected)
    task check_fp64_approx;
        input [63:0] expected;
        input [31:0] tolerance_ulps;
        input [255:0] test_name;
        reg [62:0] exp_abs, got_abs;
        integer diff;
        begin
            test_count = test_count + 1;
            exp_abs = expected[62:0];
            got_abs = result[62:0];

            // Sign check
            if (expected[63] != result[63] && exp_abs != 0 && got_abs != 0) begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: sign mismatch, expected 0x%016x, got 0x%016x",
                         test_name, expected, result);
            end else begin
                if (exp_abs >= got_abs)
                    diff = exp_abs - got_abs;
                else
                    diff = got_abs - exp_abs;

                if (diff <= tolerance_ulps) begin
                    pass_count = pass_count + 1;
                    $display("[PASS] %0s: got 0x%016x (expected 0x%016x, diff=%0d ULPs)",
                             test_name, result, expected, diff);
                end else begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] %0s: expected 0x%016x, got 0x%016x, diff=%0d ULPs > %0d",
                             test_name, expected, result, diff, tolerance_ulps);
                end
            end
        end
    endtask

    // Check flags
    task check_flag;
        input flag_val;
        input flag_expected;
        input [255:0] flag_name;
        begin
            test_count = test_count + 1;
            if (flag_val == flag_expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s = %0b", flag_name, flag_val);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected %0b, got %0b", flag_name, flag_expected, flag_val);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU FPU64 - Comprehensive Test Suite");
        $display("============================================================");

        // Reset
        rst_n = 0;
        valid_in = 0;
        func = 0;
        rnd_mode = 0;
        ftz = 0;
        operand_a = 0;
        operand_b = 0;
        operand_c = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        //====================================================================
        // 1. FP64_ADD
        //====================================================================
        $display("\n--- FP64_ADD ---");

        // 1.0 + 2.0 = 3.0
        apply_op(`FP64_ADD,
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0);
        check_fp64(64'h4008_0000_0000_0000, "1.0 + 2.0 = 3.0");

        // -1.0 + 1.0 = 0.0
        apply_op(`FP64_ADD,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "-1.0 + 1.0 = 0.0");

        // 0.5 + 0.25 = 0.75
        apply_op(`FP64_ADD,
                 64'h3FE0_0000_0000_0000, // 0.5
                 64'h3FD0_0000_0000_0000, // 0.25
                 64'h0);
        check_fp64(64'h3FE8_0000_0000_0000, "0.5 + 0.25 = 0.75");

        // NaN + 1.0 = NaN
        apply_op(`FP64_ADD,
                 64'h7FF8_0000_0000_0000, // QNaN
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "NaN + 1.0 = NaN");
        check_flag(invalid, 1'b1, "NaN+1 invalid flag");

        // +Inf + (-Inf) = NaN
        apply_op(`FP64_ADD,
                 64'h7FF0_0000_0000_0000, // +Inf
                 64'hFFF0_0000_0000_0000, // -Inf
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "+Inf + (-Inf) = NaN");
        check_flag(invalid, 1'b1, "Inf-Inf invalid flag");

        // +Inf + 1.0 = +Inf
        apply_op(`FP64_ADD,
                 64'h7FF0_0000_0000_0000, // +Inf
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "+Inf + 1.0 = +Inf");

        // 0.0 + 0.0 = 0.0
        apply_op(`FP64_ADD,
                 64'h0000_0000_0000_0000,
                 64'h0000_0000_0000_0000,
                 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "0.0 + 0.0 = 0.0");

        //====================================================================
        // 2. FP64_SUB
        //====================================================================
        $display("\n--- FP64_SUB ---");

        // 3.0 - 1.0 = 2.0
        apply_op(`FP64_SUB,
                 64'h4008_0000_0000_0000, // 3.0
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h4000_0000_0000_0000, "3.0 - 1.0 = 2.0");

        // 1.0 - 1.0 = 0.0
        apply_op(`FP64_SUB,
                 64'h3FF0_0000_0000_0000,
                 64'h3FF0_0000_0000_0000,
                 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "1.0 - 1.0 = 0.0");

        // 0.5 - 1.0 = -0.5
        apply_op(`FP64_SUB,
                 64'h3FE0_0000_0000_0000, // 0.5
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'hBFE0_0000_0000_0000, "0.5 - 1.0 = -0.5");

        //====================================================================
        // 3. FP64_MUL
        //====================================================================
        $display("\n--- FP64_MUL ---");

        // 2.0 * 3.0 = 6.0
        apply_op(`FP64_MUL,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h4008_0000_0000_0000, // 3.0
                 64'h0);
        check_fp64(64'h4018_0000_0000_0000, "2.0 * 3.0 = 6.0");

        // -1.0 * 2.0 = -2.0
        apply_op(`FP64_MUL,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0);
        check_fp64(64'hC000_0000_0000_0000, "-1.0 * 2.0 = -2.0");

        // 0.5 * 4.0 = 2.0
        apply_op(`FP64_MUL,
                 64'h3FE0_0000_0000_0000, // 0.5
                 64'h4010_0000_0000_0000, // 4.0
                 64'h0);
        check_fp64(64'h4000_0000_0000_0000, "0.5 * 4.0 = 2.0");

        // 0.0 * Inf = NaN
        apply_op(`FP64_MUL,
                 64'h0000_0000_0000_0000, // 0.0
                 64'h7FF0_0000_0000_0000, // +Inf
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "0.0 * Inf = NaN");
        check_flag(invalid, 1'b1, "0*Inf invalid flag");

        // Inf * 2.0 = Inf
        apply_op(`FP64_MUL,
                 64'h7FF0_0000_0000_0000, // +Inf
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "Inf * 2.0 = Inf");

        // NaN * 1.0 = NaN
        apply_op(`FP64_MUL,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "NaN * 1.0 = NaN");

        //====================================================================
        // 4. FP64_DIV
        //====================================================================
        $display("\n--- FP64_DIV ---");

        // 6.0 / 2.0 = 3.0
        apply_op(`FP64_DIV,
                 64'h4018_0000_0000_0000, // 6.0
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0);
        check_fp64(64'h4008_0000_0000_0000, "6.0 / 2.0 = 3.0");

        // 1.0 / 0.0 = +Inf (div by zero)
        apply_op(`FP64_DIV,
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0000_0000_0000_0000, // 0.0
                 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "1.0 / 0.0 = +Inf");
        check_flag(div_by_zero, 1'b1, "1/0 div_by_zero flag");

        // 0.0 / 0.0 = NaN
        apply_op(`FP64_DIV,
                 64'h0000_0000_0000_0000,
                 64'h0000_0000_0000_0000,
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "0.0 / 0.0 = NaN");
        check_flag(invalid, 1'b1, "0/0 invalid flag");

        // Inf / Inf = NaN
        apply_op(`FP64_DIV,
                 64'h7FF0_0000_0000_0000,
                 64'h7FF0_0000_0000_0000,
                 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "Inf / Inf = NaN");

        // 0.0 / 1.0 = 0.0
        apply_op(`FP64_DIV,
                 64'h0000_0000_0000_0000,
                 64'h3FF0_0000_0000_0000,
                 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "0.0 / 1.0 = 0.0");

        // 1.0 / Inf = 0.0
        apply_op(`FP64_DIV,
                 64'h3FF0_0000_0000_0000,
                 64'h7FF0_0000_0000_0000,
                 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "1.0 / Inf = 0.0");

        //====================================================================
        // 5. FP64_FMA (a*b+c)
        //====================================================================
        $display("\n--- FP64_FMA ---");

        // 2.0 * 3.0 + 1.0 = 7.0
        apply_op(`FP64_FMA,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h4008_0000_0000_0000, // 3.0
                 64'h3FF0_0000_0000_0000); // 1.0
        check_fp64(64'h401C_0000_0000_0000, "2.0*3.0+1.0 = 7.0");

        // -1.0 * 2.0 + 3.0 = 1.0
        apply_op(`FP64_FMA,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h4000_0000_0000_0000, // 2.0
                 64'h4008_0000_0000_0000); // 3.0
        check_fp64(64'h3FF0_0000_0000_0000, "-1.0*2.0+3.0 = 1.0");

        // NaN FMA -> NaN
        apply_op(`FP64_FMA,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h3FF0_0000_0000_0000); // 1.0
        check_fp64(64'h7FF8_0000_0000_0000, "NaN*1.0+1.0 = NaN");

        //====================================================================
        // 6. FP64_NEG
        //====================================================================
        $display("\n--- FP64_NEG ---");

        // neg(1.0) = -1.0
        apply_op(`FP64_NEG,
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0, 64'h0);
        check_fp64(64'hBFF0_0000_0000_0000, "neg(1.0) = -1.0");

        // neg(-2.0) = 2.0
        apply_op(`FP64_NEG,
                 64'hC000_0000_0000_0000, // -2.0
                 64'h0, 64'h0);
        check_fp64(64'h4000_0000_0000_0000, "neg(-2.0) = 2.0");

        // neg(0.0) = -0.0
        apply_op(`FP64_NEG,
                 64'h0000_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h8000_0000_0000_0000, "neg(0.0) = -0.0");

        //====================================================================
        // 7. FP64_ABS
        //====================================================================
        $display("\n--- FP64_ABS ---");

        // abs(-3.0) = 3.0
        apply_op(`FP64_ABS,
                 64'hC008_0000_0000_0000, // -3.0
                 64'h0, 64'h0);
        check_fp64(64'h4008_0000_0000_0000, "abs(-3.0) = 3.0");

        // abs(2.0) = 2.0
        apply_op(`FP64_ABS,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0, 64'h0);
        check_fp64(64'h4000_0000_0000_0000, "abs(2.0) = 2.0");

        // abs(-0.0) = 0.0
        apply_op(`FP64_ABS,
                 64'h8000_0000_0000_0000, // -0.0
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "abs(-0.0) = 0.0");

        //====================================================================
        // 8. FP64_MIN / FP64_MAX
        //====================================================================
        $display("\n--- FP64_MIN/MAX ---");

        // min(2.0, 3.0) = 2.0
        apply_op(`FP64_MIN,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h4008_0000_0000_0000, // 3.0
                 64'h0);
        check_fp64(64'h4000_0000_0000_0000, "min(2.0, 3.0) = 2.0");

        // max(2.0, 3.0) = 3.0
        apply_op(`FP64_MAX,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h4008_0000_0000_0000, // 3.0
                 64'h0);
        check_fp64(64'h4008_0000_0000_0000, "max(2.0, 3.0) = 3.0");

        // min(-1.0, 1.0) = -1.0
        apply_op(`FP64_MIN,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'hBFF0_0000_0000_0000, "min(-1.0, 1.0) = -1.0");

        // max(-1.0, 1.0) = 1.0
        apply_op(`FP64_MAX,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h3FF0_0000_0000_0000, "max(-1.0, 1.0) = 1.0");

        // min(NaN, 1.0) = 1.0 (NaN is unordered)
        apply_op(`FP64_MIN,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h3FF0_0000_0000_0000, "min(NaN, 1.0) = 1.0");

        // max(NaN, 1.0) = 1.0
        apply_op(`FP64_MAX,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h3FF0_0000_0000_0000, "max(NaN, 1.0) = 1.0");

        //====================================================================
        // 9. FP64_COPYSIGN
        //====================================================================
        $display("\n--- FP64_COPYSIGN ---");

        // copysign(2.0, -1.0) = -2.0 (sign from B)
        apply_op(`FP64_COPYSIGN,
                 64'h4000_0000_0000_0000, // 2.0
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h0);
        check_fp64(64'hC000_0000_0000_0000, "copysign(2.0, -1.0) = -2.0");

        // copysign(-3.0, 1.0) = 3.0
        apply_op(`FP64_COPYSIGN,
                 64'hC008_0000_0000_0000, // -3.0
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0);
        check_fp64(64'h4008_0000_0000_0000, "copysign(-3.0, 1.0) = 3.0");

        //====================================================================
        // 10. FP64_TESTP
        //====================================================================
        $display("\n--- FP64_TESTP ---");

        // testp(NaN) = 0 (not a number)
        apply_op(`FP64_TESTP,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "testp(NaN) = 0 (is NaN)");

        // testp(1.0) = 1 (is a number)
        apply_op(`FP64_TESTP,
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0001, "testp(1.0) = 1 (is number)");

        // testp(Inf) = 1 (Inf is not NaN)
        apply_op(`FP64_TESTP,
                 64'h7FF0_0000_0000_0000, // Inf
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0001, "testp(Inf) = 1 (not NaN)");

        //====================================================================
        // 11. FP64_RCP
        //====================================================================
        $display("\n--- FP64_RCP ---");

        // rcp(1.0) = 1.0
        apply_op(`FP64_RCP,
                 64'h3FF0_0000_0000_0000, // 1.0
                 64'h0, 64'h0);
        check_fp64_approx(64'h3FF0_0000_0000_0000, 32'd2, "rcp(1.0) ~ 1.0");

        // rcp(2.0) = 0.5
        apply_op(`FP64_RCP,
                 64'h4000_0000_0000_0000, // 2.0
                 64'h0, 64'h0);
        check_fp64_approx(64'h3FE0_0000_0000_0000, 32'd2, "rcp(2.0) ~ 0.5");

        // rcp(0.0) = +Inf
        apply_op(`FP64_RCP,
                 64'h0000_0000_0000_0000, // 0.0
                 64'h0, 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "rcp(0.0) = +Inf");
        check_flag(div_by_zero, 1'b1, "rcp(0) div_by_zero flag");

        // rcp(Inf) = 0.0
        apply_op(`FP64_RCP,
                 64'h7FF0_0000_0000_0000, // +Inf
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "rcp(Inf) = 0.0");

        // rcp(NaN) = NaN
        apply_op(`FP64_RCP,
                 64'h7FF8_0000_0000_0000, // NaN
                 64'h0, 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "rcp(NaN) = NaN");

        //====================================================================
        // 12. FP64_SQRT (simplified - check special values)
        //====================================================================
        $display("\n--- FP64_SQRT ---");

        // sqrt(0.0) = 0.0
        apply_op(`FP64_SQRT,
                 64'h0000_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "sqrt(0.0) = 0.0");

        // sqrt(+Inf) = +Inf
        apply_op(`FP64_SQRT,
                 64'h7FF0_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "sqrt(+Inf) = +Inf");

        // sqrt(-1.0) = NaN
        apply_op(`FP64_SQRT,
                 64'hBFF0_0000_0000_0000, // -1.0
                 64'h0, 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "sqrt(-1.0) = NaN");
        check_flag(invalid, 1'b1, "sqrt(-1) invalid flag");

        // sqrt(NaN) = NaN
        apply_op(`FP64_SQRT,
                 64'h7FF8_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "sqrt(NaN) = NaN");

        //====================================================================
        // 13. FP64_RSQRT (special values)
        //====================================================================
        $display("\n--- FP64_RSQRT ---");

        // rsqrt(0.0) = +Inf
        apply_op(`FP64_RSQRT,
                 64'h0000_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h7FF0_0000_0000_0000, "rsqrt(0.0) = +Inf");

        // rsqrt(+Inf) = 0.0
        apply_op(`FP64_RSQRT,
                 64'h7FF0_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h0000_0000_0000_0000, "rsqrt(+Inf) = 0.0");

        // rsqrt(-1.0) = NaN
        apply_op(`FP64_RSQRT,
                 64'hBFF0_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "rsqrt(-1.0) = NaN");

        // rsqrt(NaN) = NaN
        apply_op(`FP64_RSQRT,
                 64'h7FF8_0000_0000_0000,
                 64'h0, 64'h0);
        check_fp64(64'h7FF8_0000_0000_0000, "rsqrt(NaN) = NaN");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("FPU64 Test Summary");
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
        #500000;
        $display("[TIMEOUT] Test did not complete within timeout");
        $finish;
    end

endmodule
