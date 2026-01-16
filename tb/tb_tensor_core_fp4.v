//============================================================================
// RalphGPU - Tensor Core FP4 Sanity Test
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_tensor_core_fp4;

    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    reg op_valid;
    wire op_ready;
    reg [2:0] op_type;

    reg  [31:0] frag_a;
    reg  [31:0] frag_b;
    reg  [31:0] frag_c;
    wire [31:0] result_data;
    wire        result_valid;

    // 1-lane tensor core for a focused FP4 check
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
        .result_data (result_data)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    integer timeout;

    task run_fp4_case;
        input [2:0] dtype;
        input [31:0] expected;
        input [31:0] a_val;
        input [31:0] b_val;
        begin
            op_type = dtype;
            frag_a = a_val;
            frag_b = b_val;
            frag_c = 32'h0000_0000;

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
                $display("FAIL: timeout waiting for result");
            end else if (result_data !== expected) begin
                $display("FAIL: expected 0x%08x, got 0x%08x", expected, result_data);
            end else begin
                $display("PASS: dtype=%0d produced expected result", dtype);
            end
        end
    endtask
    initial begin
        $display("========================================");
        $display("Tensor Core FP4 Sanity Test");
        $display("========================================");

        rst_n = 0;
        op_valid = 0;
        op_type = 0;
        frag_a = 32'b0;
        frag_b = 32'b0;
        frag_c = 32'b0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // FP4 E2M1: 0x2 encodes ~1.0 -> sum(1*1) = 8.0
        run_fp4_case(`TC_DATA_FP4_E2M1, 32'h4100_0000, 32'h2222_2222, 32'h2222_2222);

        // FP4 E3M0: 0x2 encodes 0.5 -> sum(0.5*0.5) = 2.0
        run_fp4_case(`TC_DATA_FP4_E3M0, 32'h4000_0000, 32'h2222_2222, 32'h2222_2222);

        $finish;
    end

endmodule
