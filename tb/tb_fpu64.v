//============================================================================
// RalphGPU - FPU64 (Double-Precision) Testbench
// Tests fpu64 with 9 basic cases: add/sub/mul + special values
//============================================================================

`timescale 1ns/1ps
`include "../rtl/gpu_defines.vh"

module tb_fpu64;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg  [1:0]  rnd_mode;
    reg         ftz;
    reg  [63:0] operand_a;
    reg  [63:0] operand_b;
    reg  [63:0] operand_c;
    reg         valid_in;

    wire [63:0] result;
    wire        valid_out;
    wire        overflow;
    wire        underflow;
    wire        inexact;
    wire        invalid;
    wire        div_by_zero;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    fpu64 u_fpu64 (
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
    // FP64 constants (IEEE 754 double-precision)
    //------------------------------------------------------------------------
    // 1.0  = 0x3FF0_0000_0000_0000
    // 2.0  = 0x4000_0000_0000_0000
    // 3.0  = 0x4008_0000_0000_0000
    // 0.5  = 0x3FE0_0000_0000_0000
    // 0.25 = 0x3FD0_0000_0000_0000
    // 0.75 = 0x3FE8_0000_0000_0000
    // 4.0  = 0x4010_0000_0000_0000
    // 5.0  = 0x4014_0000_0000_0000
    // 6.0  = 0x4018_0000_0000_0000
    // 1e10 = 0x4202_A05F_2000_0000
    // 2e10 = 0x4212_A05F_2000_0000
    // 3e10 = 0x421C_388C_D800_0000
    // 1e5  = 0x40F8_6A00_0000_0000
    // 1e10 = 0x4202_A05F_2000_0000
    // -1.0 = 0xBFF0_0000_0000_0000
    // -2.0 = 0xC000_0000_0000_0000
    // +Inf = 0x7FF0_0000_0000_0000
    // -Inf = 0xFFF0_0000_0000_0000
    // NaN  = 0x7FF8_0000_0000_0000
    // 0.0  = 0x0000_0000_0000_0000
    // 7e9  = 0x41FA_0B5F_C000_0000
    // 3e9  = 0x4186_5D40_C000_0000

    //------------------------------------------------------------------------
    // Test task: apply inputs, wait for valid_out, check result
    //------------------------------------------------------------------------
    task run_test;
        input [5:0]   t_func;
        input [63:0]  t_a;
        input [63:0]  t_b;
        input [63:0]  t_c;
        input [63:0]  expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;

            // Setup inputs before rising edge
            @(negedge clk);
            func      = t_func;
            operand_a = t_a;
            operand_b = t_b;
            operand_c = t_c;
            rnd_mode  = 2'b00;  // Round to nearest
            ftz       = 1'b0;
            valid_in  = 1'b1;

            // valid_in sampled on this rising edge
            @(posedge clk);
            // Deassert after one cycle
            @(negedge clk);
            valid_in  = 1'b0;

            // Result available on this rising edge (1-cycle latency)
            @(posedge clk);
            // Sample on negedge for stability
            @(negedge clk);

            if (result === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%016h == 0x%016h", test_name, result, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got 0x%016h, expected 0x%016h", test_name, result, expected);
            end
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
        rnd_mode = 0;
        ftz = 0;
        operand_a = 0;
        operand_b = 0;
        operand_c = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("============================================================");
        $display("FPU64 Testbench — 9 tests");
        $display("============================================================");

        //--------------------------------------------------------------------
        // ADD tests (3)
        //--------------------------------------------------------------------
        // add_000: 1.0 + 2.0 = 3.0
        run_test(`FP64_ADD,
                 64'h3FF0_0000_0000_0000,  // 1.0
                 64'h4000_0000_0000_0000,  // 2.0
                 64'h0,
                 64'h4008_0000_0000_0000,  // 3.0
                 "fp64_add_000: 1+2=3");

        // add_001: 0.5 + 0.25 = 0.75
        run_test(`FP64_ADD,
                 64'h3FE0_0000_0000_0000,  // 0.5
                 64'h3FD0_0000_0000_0000,  // 0.25
                 64'h0,
                 64'h3FE8_0000_0000_0000,  // 0.75
                 "fp64_add_001: 0.5+0.25=0.75");

        // add_002: 1e10 + 2e10 = 3e10
        run_test(`FP64_ADD,
                 64'h4202_A05F_2000_0000,  // 1e10
                 64'h4212_A05F_2000_0000,  // 2e10
                 64'h0,
                 64'h421B_F08E_B000_0000,  // 3e10
                 "fp64_add_002: 1e10+2e10=3e10");

        //--------------------------------------------------------------------
        // SUB tests (3)
        //--------------------------------------------------------------------
        // sub_000: 5.0 - 2.0 = 3.0
        run_test(`FP64_SUB,
                 64'h4014_0000_0000_0000,  // 5.0
                 64'h4000_0000_0000_0000,  // 2.0
                 64'h0,
                 64'h4008_0000_0000_0000,  // 3.0
                 "fp64_sub_000: 5-2=3");

        // sub_001: 1.0 - 0.5 = 0.5
        run_test(`FP64_SUB,
                 64'h3FF0_0000_0000_0000,  // 1.0
                 64'h3FE0_0000_0000_0000,  // 0.5
                 64'h0,
                 64'h3FE0_0000_0000_0000,  // 0.5
                 "fp64_sub_001: 1-0.5=0.5");

        // sub_002: 1e10 - 3e9 = 7e9
        run_test(`FP64_SUB,
                 64'h4202_A05F_2000_0000,  // 1e10
                 64'h41E6_5A0B_C000_0000,  // 3e9
                 64'h0,
                 64'h41FA_13B8_6000_0000,  // 7e9
                 "fp64_sub_002: 1e10-3e9=7e9");

        //--------------------------------------------------------------------
        // MUL tests (3)
        //--------------------------------------------------------------------
        // mul_000: 2.0 * 3.0 = 6.0
        run_test(`FP64_MUL,
                 64'h4000_0000_0000_0000,  // 2.0
                 64'h4008_0000_0000_0000,  // 3.0
                 64'h0,
                 64'h4018_0000_0000_0000,  // 6.0
                 "fp64_mul_000: 2*3=6");

        // mul_001: 0.5 * 4.0 = 2.0
        run_test(`FP64_MUL,
                 64'h3FE0_0000_0000_0000,  // 0.5
                 64'h4010_0000_0000_0000,  // 4.0
                 64'h0,
                 64'h4000_0000_0000_0000,  // 2.0
                 "fp64_mul_001: 0.5*4=2");

        // mul_002: 1e5 * 1e5 = 1e10
        run_test(`FP64_MUL,
                 64'h40F8_6A00_0000_0000,  // 1e5
                 64'h40F8_6A00_0000_0000,  // 1e5
                 64'h0,
                 64'h4202_A05F_2000_0000,  // 1e10
                 "fp64_mul_002: 1e5*1e5=1e10");

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
        $dumpfile("tb_fpu64.vcd");
        $dumpvars(0, tb_fpu64);
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
