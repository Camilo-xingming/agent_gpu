//============================================================================
// RalphGPU - FP16 Unit Testbench
// Tests fp16_unit with 15 cases: add/sub/mul/fma + NaN/Inf special values
//============================================================================

`timescale 1ns/1ps
`include "../rtl/gpu_defines.vh"

module tb_fp16_unit;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
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
    wire        overflow;
    wire        underflow;
    wire        inexact;
    wire        invalid;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    fp16_unit u_fp16 (
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

    //------------------------------------------------------------------------
    // Clock: 10ns period
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Counters
    //------------------------------------------------------------------------
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // FP16 constants (IEEE 754 half-precision, lower 16 bits of 32-bit reg)
    //------------------------------------------------------------------------
    localparam [15:0] HP_ZERO     = 16'h0000;  // +0.0
    localparam [15:0] HP_ONE      = 16'h3C00;  // 1.0
    localparam [15:0] HP_NEG_ONE  = 16'hBC00;  // -1.0
    localparam [15:0] HP_TWO      = 16'h4000;  // 2.0
    localparam [15:0] HP_NEG_TWO  = 16'hC000;  // -2.0
    localparam [15:0] HP_THREE    = 16'h4200;  // 3.0
    localparam [15:0] HP_FOUR     = 16'h4400;  // 4.0
    localparam [15:0] HP_HALF     = 16'h3800;  // 0.5
    localparam [15:0] HP_QUARTER  = 16'h3400;  // 0.25
    localparam [15:0] HP_0P75     = 16'h3A00;  // 0.75 (approx)
    localparam [15:0] HP_SIX      = 16'h4600;  // 6.0
    localparam [15:0] HP_SEVEN    = 16'h4700;  // 7.0
    localparam [15:0] HP_INF      = 16'h7C00;  // +Inf
    localparam [15:0] HP_NEG_INF  = 16'hFC00;  // -Inf
    localparam [15:0] HP_NAN      = 16'h7E00;  // Quiet NaN

    //------------------------------------------------------------------------
    // Test task: apply inputs, wait for valid_out, check result
    //------------------------------------------------------------------------
    task run_test;
        input [5:0]   t_func;
        input [31:0]  t_a;
        input [31:0]  t_b;
        input [31:0]  t_c;
        input [15:0]  expected;
        input [255:0] test_name;
        input         check_nan;  // 1=check NaN (any NaN matches)
        begin
            test_count = test_count + 1;

            @(posedge clk);
            func      = t_func;
            operand_a = t_a;
            operand_b = t_b;
            operand_c = t_c;
            packed_mode = 1'b0;
            valid_in  = 1'b1;

            @(posedge clk);
            valid_in  = 1'b0;

            // Wait for valid_out (3-stage pipeline)
            wait(valid_out);
            @(negedge clk);  // sample on negedge for stability

            if (check_nan) begin
                // Any NaN result is acceptable (exp=0x1F, mant!=0)
                if (result[14:10] == 5'h1F && result[9:0] != 10'b0) begin
                    pass_count = pass_count + 1;
                    $display("[PASS] %0s: got NaN (0x%04h)", test_name, result[15:0]);
                end else begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] %0s: got 0x%04h, expected NaN", test_name, result[15:0]);
                end
            end else begin
                if (result[15:0] === expected) begin
                    pass_count = pass_count + 1;
                    $display("[PASS] %0s: 0x%04h == 0x%04h", test_name, result[15:0], expected);
                end else begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] %0s: got 0x%04h, expected 0x%04h", test_name, result[15:0], expected);
                end
            end

            @(posedge clk); // gap between tests
        end
    endtask

    //------------------------------------------------------------------------
    // Test sequence
    //------------------------------------------------------------------------
    initial begin
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

        $display("============================================================");
        $display("FP16 Unit Testbench — 15 tests");
        $display("============================================================");

        //--------------------------------------------------------------------
        // ADD tests (3)
        //--------------------------------------------------------------------
        // add_000: 1.0 + 2.0 = 3.0
        run_test(`FP16_ADD, {16'b0, HP_ONE}, {16'b0, HP_TWO}, 32'b0,
                 HP_THREE, "fp16_add_000: 1+2=3", 0);

        // add_001: 0.5 + 0.25 = 0.75
        run_test(`FP16_ADD, {16'b0, HP_HALF}, {16'b0, HP_QUARTER}, 32'b0,
                 HP_0P75, "fp16_add_001: 0.5+0.25=0.75", 0);

        // add_002: -1.0 + 1.0 = 0.0
        run_test(`FP16_ADD, {16'b0, HP_NEG_ONE}, {16'b0, HP_ONE}, 32'b0,
                 HP_ZERO, "fp16_add_002: -1+1=0", 0);

        //--------------------------------------------------------------------
        // SUB tests (3)
        //--------------------------------------------------------------------
        // sub_000: 3.0 - 1.0 = 2.0
        run_test(`FP16_SUB, {16'b0, HP_THREE}, {16'b0, HP_ONE}, 32'b0,
                 HP_TWO, "fp16_sub_000: 3-1=2", 0);

        // sub_001: 1.0 - 0.5 = 0.5
        run_test(`FP16_SUB, {16'b0, HP_ONE}, {16'b0, HP_HALF}, 32'b0,
                 HP_HALF, "fp16_sub_001: 1-0.5=0.5", 0);

        // sub_002: 0.0 - 1.0 = -1.0
        run_test(`FP16_SUB, {16'b0, HP_ZERO}, {16'b0, HP_ONE}, 32'b0,
                 HP_NEG_ONE, "fp16_sub_002: 0-1=-1", 0);

        //--------------------------------------------------------------------
        // MUL tests (3)
        //--------------------------------------------------------------------
        // mul_000: 2.0 * 3.0 = 6.0
        run_test(`FP16_MUL, {16'b0, HP_TWO}, {16'b0, HP_THREE}, 32'b0,
                 HP_SIX, "fp16_mul_000: 2*3=6", 0);

        // mul_001: 0.5 * 4.0 = 2.0
        run_test(`FP16_MUL, {16'b0, HP_HALF}, {16'b0, HP_FOUR}, 32'b0,
                 HP_TWO, "fp16_mul_001: 0.5*4=2", 0);

        // mul_002: -1.0 * 2.0 = -2.0
        run_test(`FP16_MUL, {16'b0, HP_NEG_ONE}, {16'b0, HP_TWO}, 32'b0,
                 HP_NEG_TWO, "fp16_mul_002: -1*2=-2", 0);

        //--------------------------------------------------------------------
        // FMA tests (2)
        //--------------------------------------------------------------------
        // fma_000: 2*3+1 = 7
        run_test(`FP16_FMA, {16'b0, HP_TWO}, {16'b0, HP_THREE}, {16'b0, HP_ONE},
                 HP_SEVEN, "fp16_fma_000: 2*3+1=7", 0);

        // fma_001: -1*2+3 = 1
        run_test(`FP16_FMA, {16'b0, HP_NEG_ONE}, {16'b0, HP_TWO}, {16'b0, HP_THREE},
                 HP_ONE, "fp16_fma_001: -1*2+3=1", 0);

        //--------------------------------------------------------------------
        // Special value tests (4)
        //--------------------------------------------------------------------
        // NaN + 1.0 = NaN (NaN propagation)
        run_test(`FP16_ADD, {16'b0, HP_NAN}, {16'b0, HP_ONE}, 32'b0,
                 HP_NAN, "fp16_add_nan: NaN+1=NaN", 1);

        // Inf * 2.0 = Inf
        run_test(`FP16_MUL, {16'b0, HP_INF}, {16'b0, HP_TWO}, 32'b0,
                 HP_INF, "fp16_mul_inf: Inf*2=Inf", 0);

        // +Inf + (-Inf) = NaN (invalid operation)
        run_test(`FP16_ADD, {16'b0, HP_INF}, {16'b0, HP_NEG_INF}, 32'b0,
                 HP_NAN, "fp16_add_inf_neg: Inf+(-Inf)=NaN", 1);

        // 0 * Inf = NaN (invalid operation)
        run_test(`FP16_MUL, {16'b0, HP_ZERO}, {16'b0, HP_INF}, 32'b0,
                 HP_NAN, "fp16_mul_zero_inf: 0*Inf=NaN", 1);

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("============================================================");
        $display("Results: %0d/%0d PASSED, %0d FAILED",
                 pass_count, test_count, fail_count);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

    //------------------------------------------------------------------------
    // VCD dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_fp16_unit.vcd");
        $dumpvars(0, tb_fp16_unit);
    end

    //------------------------------------------------------------------------
    // Timeout
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("TIMEOUT: simulation exceeded 100us");
        $finish;
    end

endmodule
