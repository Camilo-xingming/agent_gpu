//============================================================================
// RalphGPU - SFU (Special Function Unit) Testbench
// Tests: rcp, sqrt, rsqrt, sin, cos, lg2, ex2, tanh
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_sfu;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;
    localparam LATENCY = 8;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg  [31:0] operand;
    reg         valid_in;
    wire [31:0] result;
    wire        valid_out;
    wire        ready;
    wire        invalid;
    wire        div_by_zero;

    //------------------------------------------------------------------------
    // Clock generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    sfu #(
        .LATENCY(LATENCY)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .func       (func),
        .operand    (operand),
        .valid_in   (valid_in),
        .result     (result),
        .valid_out  (valid_out),
        .ready      (ready),
        .invalid    (invalid),
        .div_by_zero(div_by_zero)
    );

    //------------------------------------------------------------------------
    // Test helpers
    //------------------------------------------------------------------------
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    // IEEE 754 constants
    localparam [31:0] FP_ZERO     = 32'h00000000;
    localparam [31:0] FP_ONE      = 32'h3F800000;
    localparam [31:0] FP_TWO      = 32'h40000000;
    localparam [31:0] FP_FOUR     = 32'h40800000;
    localparam [31:0] FP_HALF     = 32'h3F000000;
    localparam [31:0] FP_NEG_ONE  = 32'hBF800000;
    localparam [31:0] FP_INF      = 32'h7F800000;
    localparam [31:0] FP_NEG_INF  = 32'hFF800000;
    localparam [31:0] FP_NAN      = 32'h7FC00000;
    localparam [31:0] FP_PI_2     = 32'h3FC90FDB;  // π/2

    // Convert hex to real for display
    function real hex_to_real;
        input [31:0] fp;
        reg sign;
        reg [7:0] exp;
        reg [22:0] man;
        real result;
        begin
            sign = fp[31];
            exp = fp[30:23];
            man = fp[22:0];

            if (exp == 0 && man == 0) begin
                result = 0.0;
            end else if (exp == 255 && man == 0) begin
                result = sign ? -1e38 : 1e38;  // Represent Inf
            end else if (exp == 255) begin
                result = 0.0;  // NaN
            end else begin
                result = (1.0 + man / 8388608.0) * $pow(2.0, exp - 127);
                if (sign) result = -result;
            end
            hex_to_real = result;
        end
    endfunction

    // Check if result is within tolerance
    function check_result;
        input [31:0] actual;
        input [31:0] expected;
        input real tolerance;
        real actual_r, expected_r, diff;
        begin
            // Special case handling
            if (expected == FP_NAN) begin
                check_result = (actual[30:23] == 8'hFF) && (actual[22:0] != 0);
            end else if (expected == FP_INF) begin
                check_result = (actual == FP_INF);
            end else if (expected == FP_NEG_INF) begin
                check_result = (actual == FP_NEG_INF);
            end else if (expected == FP_ZERO) begin
                check_result = (actual[30:0] == 31'h0);  // Allow +0 or -0
            end else begin
                // Exact match check first
                if (actual == expected) begin
                    check_result = 1;
                end else begin
                    actual_r = hex_to_real(actual);
                    expected_r = hex_to_real(expected);
                    if (expected_r == 0.0) begin
                        check_result = (actual_r <= tolerance) && (actual_r >= -tolerance);
                    end else begin
                        diff = (actual_r - expected_r) / expected_r;
                        if (diff < 0) diff = -diff;
                        check_result = (diff <= tolerance);
                    end
                end
            end
        end
    endfunction

    // Send a test and wait for result
    task test_sfu;
        input [5:0] test_func;
        input [31:0] test_operand;
        input [31:0] expected;
        input [255:0] test_name;
        input real tolerance;
        integer i;
        begin
            test_count = test_count + 1;

            @(posedge clk);
            func <= test_func;
            operand <= test_operand;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for result (LATENCY cycles)
            for (i = 0; i < LATENCY + 2; i = i + 1) begin
                @(posedge clk);
                if (valid_out) begin
                    if (check_result(result, expected, tolerance)) begin
                        $display("[PASS] %0s: input=%h result=%h expected=%h",
                                 test_name, test_operand, result, expected);
                        pass_count = pass_count + 1;
                    end else begin
                        $display("[FAIL] %0s: input=%h result=%h expected=%h",
                                 test_name, test_operand, result, expected);
                        fail_count = fail_count + 1;
                    end
                    i = LATENCY + 10;  // Exit loop
                end
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Test sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU SFU Test");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        func = 0;
        operand = 0;
        valid_in = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        //--------------------------------------------------------------------
        // RCP (1/x) Tests
        //--------------------------------------------------------------------
        $display("\n--- RCP (1/x) Tests ---");

        // 1/1 = 1
        test_sfu(`FP_RCP, FP_ONE, FP_ONE, "RCP(1.0)", 0.1);

        // 1/2 = 0.5
        test_sfu(`FP_RCP, FP_TWO, FP_HALF, "RCP(2.0)", 0.1);

        // 1/0.5 = 2
        test_sfu(`FP_RCP, FP_HALF, FP_TWO, "RCP(0.5)", 0.1);

        // 1/0 = Inf
        test_sfu(`FP_RCP, FP_ZERO, FP_INF, "RCP(0.0)", 0.0);

        // 1/Inf = 0
        test_sfu(`FP_RCP, FP_INF, FP_ZERO, "RCP(Inf)", 0.0);

        // 1/NaN = NaN
        test_sfu(`FP_RCP, FP_NAN, FP_NAN, "RCP(NaN)", 0.0);

        //--------------------------------------------------------------------
        // SQRT Tests
        //--------------------------------------------------------------------
        $display("\n--- SQRT Tests ---");

        // sqrt(1) = 1
        test_sfu(`FP_SQRT, FP_ONE, FP_ONE, "SQRT(1.0)", 0.1);

        // sqrt(4) = 2
        test_sfu(`FP_SQRT, FP_FOUR, FP_TWO, "SQRT(4.0)", 0.1);

        // sqrt(0) = 0
        test_sfu(`FP_SQRT, FP_ZERO, FP_ZERO, "SQRT(0.0)", 0.0);

        // sqrt(Inf) = Inf
        test_sfu(`FP_SQRT, FP_INF, FP_INF, "SQRT(Inf)", 0.0);

        // sqrt(-1) = NaN
        test_sfu(`FP_SQRT, FP_NEG_ONE, FP_NAN, "SQRT(-1.0)", 0.0);

        //--------------------------------------------------------------------
        // RSQRT Tests
        //--------------------------------------------------------------------
        $display("\n--- RSQRT (1/sqrt) Tests ---");

        // 1/sqrt(1) = 1
        test_sfu(`FP_RSQRT, FP_ONE, FP_ONE, "RSQRT(1.0)", 0.1);

        // 1/sqrt(4) = 0.5
        test_sfu(`FP_RSQRT, FP_FOUR, FP_HALF, "RSQRT(4.0)", 0.1);

        // 1/sqrt(0) = Inf
        test_sfu(`FP_RSQRT, FP_ZERO, FP_INF, "RSQRT(0.0)", 0.0);

        // 1/sqrt(Inf) = 0
        test_sfu(`FP_RSQRT, FP_INF, FP_ZERO, "RSQRT(Inf)", 0.0);

        // 1/sqrt(-1) = NaN
        test_sfu(`FP_RSQRT, FP_NEG_ONE, FP_NAN, "RSQRT(-1.0)", 0.0);

        //--------------------------------------------------------------------
        // SIN Tests
        //--------------------------------------------------------------------
        $display("\n--- SIN Tests ---");

        // sin(0) = 0
        test_sfu(`FP_SIN, FP_ZERO, FP_ZERO, "SIN(0.0)", 0.0);

        // sin(Inf) = NaN
        test_sfu(`FP_SIN, FP_INF, FP_NAN, "SIN(Inf)", 0.0);

        // sin(NaN) = NaN
        test_sfu(`FP_SIN, FP_NAN, FP_NAN, "SIN(NaN)", 0.0);

        //--------------------------------------------------------------------
        // COS Tests
        //--------------------------------------------------------------------
        $display("\n--- COS Tests ---");

        // cos(0) = 1
        test_sfu(`FP_COS, FP_ZERO, FP_ONE, "COS(0.0)", 0.0);

        // cos(Inf) = NaN
        test_sfu(`FP_COS, FP_INF, FP_NAN, "COS(Inf)", 0.0);

        // cos(NaN) = NaN
        test_sfu(`FP_COS, FP_NAN, FP_NAN, "COS(NaN)", 0.0);

        //--------------------------------------------------------------------
        // LG2 (log2) Tests
        //--------------------------------------------------------------------
        $display("\n--- LG2 (log2) Tests ---");

        // log2(1) = 0
        test_sfu(`FP_LG2, FP_ONE, FP_ZERO, "LG2(1.0)", 0.1);

        // log2(2) = 1
        test_sfu(`FP_LG2, FP_TWO, FP_ONE, "LG2(2.0)", 0.1);

        // log2(0) = -Inf
        test_sfu(`FP_LG2, FP_ZERO, FP_NEG_INF, "LG2(0.0)", 0.0);

        // log2(Inf) = Inf
        test_sfu(`FP_LG2, FP_INF, FP_INF, "LG2(Inf)", 0.0);

        // log2(-1) = NaN
        test_sfu(`FP_LG2, FP_NEG_ONE, FP_NAN, "LG2(-1.0)", 0.0);

        //--------------------------------------------------------------------
        // EX2 (2^x) Tests
        //--------------------------------------------------------------------
        $display("\n--- EX2 (2^x) Tests ---");

        // 2^0 = 1
        test_sfu(`FP_EX2, FP_ZERO, FP_ONE, "EX2(0.0)", 0.0);

        // 2^1 = 2
        test_sfu(`FP_EX2, FP_ONE, FP_TWO, "EX2(1.0)", 0.1);

        // 2^Inf = Inf
        test_sfu(`FP_EX2, FP_INF, FP_INF, "EX2(Inf)", 0.0);

        // 2^(-Inf) = 0
        test_sfu(`FP_EX2, FP_NEG_INF, FP_ZERO, "EX2(-Inf)", 0.0);

        // 2^NaN = NaN
        test_sfu(`FP_EX2, FP_NAN, FP_NAN, "EX2(NaN)", 0.0);

        //--------------------------------------------------------------------
        // TANH Tests
        //--------------------------------------------------------------------
        $display("\n--- TANH Tests ---");

        // tanh(0) = 0
        test_sfu(`FP_TANH, FP_ZERO, FP_ZERO, "TANH(0.0)", 0.0);

        // tanh(Inf) = 1
        test_sfu(`FP_TANH, FP_INF, FP_ONE, "TANH(Inf)", 0.0);

        // tanh(-Inf) = -1
        test_sfu(`FP_TANH, FP_NEG_INF, FP_NEG_ONE, "TANH(-Inf)", 0.0);

        // tanh(NaN) = NaN
        test_sfu(`FP_TANH, FP_NAN, FP_NAN, "TANH(NaN)", 0.0);

        //--------------------------------------------------------------------
        // Pipeline test - multiple operations in flight
        //--------------------------------------------------------------------
        $display("\n--- Pipeline Test ---");
        @(posedge clk);

        // Issue 4 operations back-to-back
        func <= `FP_RCP; operand <= FP_TWO; valid_in <= 1'b1;
        @(posedge clk);
        func <= `FP_SQRT; operand <= FP_FOUR; valid_in <= 1'b1;
        @(posedge clk);
        func <= `FP_LG2; operand <= FP_TWO; valid_in <= 1'b1;
        @(posedge clk);
        func <= `FP_EX2; operand <= FP_ONE; valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Wait for all results
        repeat(LATENCY + 5) @(posedge clk);

        $display("[INFO] Pipeline test completed");

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        repeat(10) @(posedge clk);

        $display("\n============================================================");
        $display("SFU Test Summary: %0d PASSED, %0d FAILED out of %0d",
                 pass_count, fail_count, test_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        $finish;
    end

endmodule
