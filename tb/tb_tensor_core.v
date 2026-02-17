//============================================================================
// RalphGPU - Tensor Core RTL Testbench
// Tests tensor_core wrapper: FP16 dot2, INT8 dot4 per lane
// Pipeline: op_valid → TC_LATENCY cycles → result_valid
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_core;

    parameter NUM_LANES = 32;
    parameter DATA_WIDTH = 32;
    parameter TC_LATENCY = 8;

    reg clk, rst_n;

    // Pipeline interface
    reg                              op_valid;
    wire                             op_ready;
    reg  [3:0]                       op_type;

    // Fragment inputs
    reg  [NUM_LANES*DATA_WIDTH-1:0]  frag_a;
    reg  [NUM_LANES*DATA_WIDTH-1:0]  frag_b;
    reg  [NUM_LANES*DATA_WIDTH-1:0]  frag_c;

    // Result output
    wire                             result_valid;
    reg                              result_ready;
    wire [NUM_LANES*DATA_WIDTH-1:0]  result_data;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    tensor_core #(
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TC_NUM_CORES(4),
        .TC_LATENCY(TC_LATENCY),
        .TC_DATA_DEFAULT(`TC_DATA_FP16),
        .TC_USE_OP_TYPE(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .op_valid(op_valid),
        .op_ready(op_ready),
        .op_type(op_type),
        .frag_a(frag_a),
        .frag_b(frag_b),
        .frag_c(frag_c),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_data(result_data)
    );

    //------------------------------------------------------------------------
    // Clock
    //------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // Test infrastructure
    //------------------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    // FP16 encoding helpers (IEEE 754 half-precision)
    // FP16: sign(1) + exp(5) + man(10), bias=15
    // 0.0  = 16'h0000
    // 1.0  = 16'h3C00
    // 2.0  = 16'h4000
    // 3.0  = 16'h4200
    // 4.0  = 16'h4400
    // 5.0  = 16'h4500
    // 10.0 = 16'h4900

    // FP32 encoding helpers
    // 0.0  = 32'h00000000
    // 5.0  = 32'h40A00000
    // 7.0  = 32'h40E00000
    // 10.0 = 32'h41200000
    // 14.0 = 32'h41600000
    // 30.0 = 32'h41F00000
    // 70.0 = 32'h428C0000
    // 100.0 = 32'h42C80000

    task submit_op;
        input [3:0] t_type;
        begin
            @(negedge clk);
            op_valid <= 1'b1;
            op_type <= t_type;
            @(negedge clk);
            op_valid <= 1'b0;
        end
    endtask

    task wait_result;
        input integer max_cycles;
        integer i;
        begin
            result_ready <= 1'b1;
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (result_valid) begin
                    i = max_cycles;
                end
            end
        end
    endtask

    task check_lane_fp32;
        input integer lane;
        input [31:0] expected;
        reg [31:0] actual;
        begin
            actual = result_data[lane*32 +: 32];
            if (actual == expected) begin
                $display("  PASS: lane[%0d] = 0x%08x", lane, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: lane[%0d] = 0x%08x, expected 0x%08x", lane, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_lane_int32;
        input integer lane;
        input [31:0] expected;
        reg [31:0] actual;
        begin
            actual = result_data[lane*32 +: 32];
            if (actual == expected) begin
                $display("  PASS: lane[%0d] = %0d (0x%08x)", lane, $signed(actual), actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: lane[%0d] = %0d (0x%08x), expected %0d (0x%08x)",
                         lane, $signed(actual), actual, $signed(expected), expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helpers to set all lanes to the same value
    //------------------------------------------------------------------------
    integer lane_i;

    task set_all_lanes;
        input [31:0] a_val;
        input [31:0] b_val;
        input [31:0] c_val;
        begin
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                frag_a[lane_i*32 +: 32] = a_val;
                frag_b[lane_i*32 +: 32] = b_val;
                frag_c[lane_i*32 +: 32] = c_val;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_tensor_core.vcd");
        $dumpvars(0, tb_tensor_core);

        // Init
        rst_n = 0;
        op_valid = 0;
        op_type = 0;
        frag_a = 0;
        frag_b = 0;
        frag_c = 0;
        result_ready = 1;

        #20;
        rst_n = 1;
        #20;

        $display("========================================");
        $display("Tensor Core RTL Testbench");
        $display("========================================");

        //--------------------------------------------------------------------
        // Test 1: FP16 dot2 — 1.0*2.0 + 3.0*4.0 + 0.0 = 14.0
        //   a = {FP16(3.0), FP16(1.0)} = {16'h4200, 16'h3C00}
        //   b = {FP16(4.0), FP16(2.0)} = {16'h4400, 16'h4000}
        //   c = FP32(0.0) = 32'h00000000
        //   result = 1.0*2.0 + 3.0*4.0 + 0.0 = 2.0 + 12.0 = 14.0
        //   FP32(14.0) = 32'h41600000
        //--------------------------------------------------------------------
        test_num = 1;
        $display("\nTest %0d: FP16 dot2 (1*2 + 3*4 + 0 = 14.0)", test_num);
        set_all_lanes({16'h4200, 16'h3C00}, {16'h4400, 16'h4000}, 32'h00000000);
        submit_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane_fp32(0, 32'h41600000);

        //--------------------------------------------------------------------
        // Test 2: FP16 dot2 with accumulate — 1.0*1.0 + 1.0*1.0 + 5.0 = 7.0
        //   a = {FP16(1.0), FP16(1.0)} = {16'h3C00, 16'h3C00}
        //   b = {FP16(1.0), FP16(1.0)} = {16'h3C00, 16'h3C00}
        //   c = FP32(5.0) = 32'h40A00000
        //   result = 1+1+5 = 7.0 = 32'h40E00000
        //--------------------------------------------------------------------
        test_num = 2;
        $display("\nTest %0d: FP16 dot2 with accum (1*1 + 1*1 + 5 = 7.0)", test_num);
        set_all_lanes({16'h3C00, 16'h3C00}, {16'h3C00, 16'h3C00}, 32'h40A00000);
        submit_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane_fp32(0, 32'h40E00000);

        //--------------------------------------------------------------------
        // Test 3: FP16 dot2 zero inputs — 0*X + 0*X + 10.0 = 10.0
        //   a = {FP16(0.0), FP16(0.0)} = {16'h0000, 16'h0000}
        //   b = {FP16(5.0), FP16(3.0)} = {16'h4500, 16'h4200}
        //   c = FP32(10.0) = 32'h41200000
        //   result = 0+0+10 = 10.0 = 32'h41200000
        //--------------------------------------------------------------------
        test_num = 3;
        $display("\nTest %0d: FP16 dot2 zero inputs (0*5 + 0*3 + 10 = 10.0)", test_num);
        set_all_lanes({16'h0000, 16'h0000}, {16'h4500, 16'h4200}, 32'h41200000);
        submit_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane_fp32(0, 32'h41200000);

        //--------------------------------------------------------------------
        // Test 4: INT8 dot4 unsigned — [1,2,3,4]·[5,6,7,8] + 0 = 70
        //   a = {8'd4, 8'd3, 8'd2, 8'd1} = 32'h04030201
        //   b = {8'd8, 8'd7, 8'd6, 8'd5} = 32'h08070605
        //   c = 0
        //   result = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        //--------------------------------------------------------------------
        test_num = 4;
        $display("\nTest %0d: INT8 dot4 unsigned ([1,2,3,4]·[5,6,7,8]+0 = 70)", test_num);
        set_all_lanes(32'h04030201, 32'h08070605, 32'h00000000);
        submit_op(`TC_DATA_INT8);
        wait_result(TC_LATENCY + 10);
        check_lane_int32(0, 32'd70);

        //--------------------------------------------------------------------
        // Test 5: INT8 dot4 signed — [-1,2,-3,4]·[5,-6,7,-8] + 100 = 30
        //   a = {8'sd4, 8'sd(-3), 8'sd2, 8'sd(-1)}
        //     = {8'h04, 8'hFD, 8'h02, 8'hFF} = 32'h04FD02FF
        //   b = {8'sd(-8), 8'sd7, 8'sd(-6), 8'sd5}
        //     = {8'hF8, 8'h07, 8'hFA, 8'h05} = 32'hF807FA05
        //   c = 100 = 32'h00000064
        //   result = (-1)*5 + 2*(-6) + (-3)*7 + 4*(-8) + 100
        //          = -5 + -12 + -21 + -32 + 100 = 30
        //--------------------------------------------------------------------
        test_num = 5;
        $display("\nTest %0d: INT8 dot4 signed ([-1,2,-3,4]·[5,-6,7,-8]+100 = 30)", test_num);
        set_all_lanes(32'h04FD02FF, 32'hF807FA05, 32'h00000064);
        submit_op(`TC_DATA_INT8);
        wait_result(TC_LATENCY + 10);
        check_lane_int32(0, 32'd30);

        //--------------------------------------------------------------------
        // Results
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Results: %0d/%0d PASSED", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("%0d TESTS FAILED", fail_count);
        $display("========================================");

        #100;
        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT: simulation exceeded 50000ns");
        $finish;
    end

endmodule
