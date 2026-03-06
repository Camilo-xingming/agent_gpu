//============================================================================
// RalphGPU - Tensor Core FP4 Functional Verification
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_tensor_core_fp4;

    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    reg op_valid;
    wire op_ready;
    reg [3:0] op_type;

    reg  [31:0] frag_a;
    reg  [31:0] frag_b;
    reg  [31:0] frag_c;
    wire [31:0] result_data;
    wire        result_valid;
    wire        result_ready;

    // 1-lane tensor core for focused FP4 functional checks.
    tensor_core #(
        .NUM_LANES(1),
        .DATA_WIDTH(32),
        .TC_NUM_CORES(1),
        .TC_LATENCY(2),
        .TC_DATA_DEFAULT(`TC_DATA_FP16),
        .TC_USE_OP_TYPE(1),
        .TC_FP4_FORMAT(`TC_FP4_E2M1)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .op_valid    (op_valid),
        .op_ready    (op_ready),
        .op_type     (op_type),
        .frag_a      (frag_a),
        .frag_b      (frag_b),
        .frag_c      (frag_c),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_data (result_data)
    );

    assign result_ready = 1'b1;

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    integer timeout;
    integer pass_count;
    integer fail_count;
    integer test_count;
    reg [31:0] diff;

    task run_fp4_case_tol;
        input [3:0] dtype;
        input [31:0] a_val;
        input [31:0] b_val;
        input [31:0] c_val;
        input [31:0] expected;
        input [31:0] tolerance_ulp;
        input [8*96-1:0] label;
        begin
            test_count = test_count + 1;
            op_type = dtype;
            frag_a = a_val;
            frag_b = b_val;
            frag_c = c_val;

            @(posedge clk);
            while (!op_ready) begin
                @(posedge clk);
            end
            op_valid = 1'b1;
            @(posedge clk);
            op_valid = 1'b0;

            timeout = 50;
            while (!result_valid && timeout > 0) begin
                @(posedge clk);
                timeout = timeout - 1;
            end

            if (!result_valid) begin
                $display("FAIL test %0d: %0s timeout waiting for result", test_count, label);
                fail_count = fail_count + 1;
            end else begin
                if (result_data > expected)
                    diff = result_data - expected;
                else
                    diff = expected - result_data;

                if (result_data === expected || diff <= tolerance_ulp) begin
                    $display("PASS test %0d: %0s (got 0x%08x)", test_count, label, result_data);
                    pass_count = pass_count + 1;
                end else begin
                    $display("FAIL test %0d: %0s expected 0x%08x got 0x%08x diff=%0d", test_count, label, expected, result_data, diff);
                    fail_count = fail_count + 1;
                end
            end
        end
    endtask

    initial begin
        $display("========================================");
        $display("Tensor Core FP4 Functional Verification");
        $display("========================================");

        rst_n = 1'b0;
        op_valid = 1'b0;
        op_type = 4'b0;
        frag_a = 32'b0;
        frag_b = 32'b0;
        frag_c = 32'b0;
        pass_count = 0;
        fail_count = 0;
        test_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // E2M1 data path vectors.
        run_fp4_case_tol(`TC_DATA_FP4_E2M1, 32'h2222_2222, 32'h2222_2222, 32'h0000_0000,
                         32'h4100_0000, 32'd16, "E2M1 all 1.0 => dot8 8.0");
        run_fp4_case_tol(`TC_DATA_FP4_E2M1, 32'h2222_2222, 32'h4444_4444, 32'h0000_0000,
                         32'h4180_0000, 32'd16, "E2M1 1.0x2.0 => dot8 16.0");
        run_fp4_case_tol(`TC_DATA_FP4_E2M1, 32'h2222_2222, 32'h2222_2222, 32'h3f80_0000,
                         32'h4110_0000, 32'd16, "E2M1 dot8 + C(1.0) => 9.0");
        run_fp4_case_tol(`TC_DATA_FP4_E2M1, 32'hAAAA_AAAA, 32'h2222_2222, 32'h0000_0000,
                         32'hc100_0000, 32'd16, "E2M1 -1.0x1.0 => dot8 -8.0");
        run_fp4_case_tol(`TC_DATA_FP4_E2M1, 32'h1111_1111, 32'h2222_2222, 32'h0000_0000,
                         32'h4080_0000, 32'd16, "E2M1 0.5x1.0 => dot8 4.0");

        // E3M0 data path vectors.
        run_fp4_case_tol(`TC_DATA_FP4_E3M0, 32'h2222_2222, 32'h2222_2222, 32'h0000_0000,
                         32'h4000_0000, 32'd16, "E3M0 all 0.5 => dot8 2.0");
        run_fp4_case_tol(`TC_DATA_FP4_E3M0, 32'h3333_3333, 32'h3333_3333, 32'h0000_0000,
                         32'h4100_0000, 32'd16, "E3M0 all 1.0 => dot8 8.0");
        run_fp4_case_tol(`TC_DATA_FP4_E3M0, 32'h3333_3333, 32'h3333_3333, 32'h40a0_0000,
                         32'h4150_0000, 32'd16, "E3M0 dot8 + C(5.0) => 13.0");
        run_fp4_case_tol(`TC_DATA_FP4_E3M0, 32'hBBBB_BBBB, 32'h3333_3333, 32'h0000_0000,
                         32'hc100_0000, 32'd16, "E3M0 -1.0x1.0 => dot8 -8.0");
        run_fp4_case_tol(`TC_DATA_FP4_E3M0, 32'h0000_0000, 32'h3333_3333, 32'h0000_0000,
                         32'h0000_0000, 32'd16, "E3M0 zero multiplicand => 0.0");

        $display("========================================");
        $display("Tensor Core FP4 Functional Tests: %0d/%0d passed", pass_count, test_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED (%0d failures)", fail_count);

        if (fail_count > 0)
            $fatal(1, "Tensor Core FP4 functional test failed");

        $finish;
    end

    // Timeout watchdog: never allow hang to be reported as success.
    initial begin
        #500000;
        $fatal(1, "TIMEOUT: tb_tensor_core_fp4");
    end

endmodule
