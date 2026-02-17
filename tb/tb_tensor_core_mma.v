//============================================================================
// RalphGPU - Tensor Core MMA RTL Testbench
// Tests tensor_core wrapper: FP16 dot2 and INT8 dot4 per-lane operations
// Pipeline: op_valid → TC_LATENCY cycles → result_valid
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_core_mma;

    parameter NUM_LANES = 32;
    parameter DATA_WIDTH = 32;
    parameter TC_LATENCY = 8;

    reg         clk, rst_n;
    reg         op_valid;
    wire        op_ready;
    reg  [3:0]  op_type;

    reg  [NUM_LANES*DATA_WIDTH-1:0] frag_a;
    reg  [NUM_LANES*DATA_WIDTH-1:0] frag_b;
    reg  [NUM_LANES*DATA_WIDTH-1:0] frag_c;

    wire        result_valid;
    reg         result_ready;
    wire [NUM_LANES*DATA_WIDTH-1:0] result_data;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    tensor_core #(
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TC_NUM_CORES(4),
        .TC_LATENCY(TC_LATENCY),
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

    // FP16 encoding helpers
    // FP16: sign(1) + exp(5) + man(10), bias=15
    // 1.0  = 0x3C00 (exp=15, man=0)
    // 2.0  = 0x4000 (exp=16, man=0)
    // 3.0  = 0x4200 (exp=16, man=0x200)
    // 4.0  = 0x4400 (exp=17, man=0)
    // 5.0  = 0x4500 (exp=17, man=0x100)
    // 0.0  = 0x0000
    // 10.0 = 0x4900 (exp=18, man=0x100)

    // FP32 encoding:
    // 0.0  = 0x00000000
    // 5.0  = 0x40A00000
    // 7.0  = 0x40E00000
    // 10.0 = 0x41200000
    // 14.0 = 0x41600000
    // 30.0 = 0x41F00000
    // 70.0 = 0x428C0000
    // 100.0 = 0x42C80000

    task issue_op;
        input [3:0] dtype;
        begin
            @(negedge clk);
            op_valid <= 1'b1;
            op_type <= dtype;
            @(negedge clk);
            op_valid <= 1'b0;
        end
    endtask

    task wait_result;
        input integer max_cycles;
        integer i;
        begin
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (result_valid) begin
                    i = max_cycles;
                end
            end
        end
    endtask

    task check_lane0_fp32;
        input [31:0] expected;
        reg [31:0] actual;
        begin
            actual = result_data[31:0];
            if (actual == expected) begin
                $display("  PASS: lane0 = 0x%08x", actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: lane0 = 0x%08x, expected 0x%08x", actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_lane0_int32;
        input [31:0] expected;
        reg [31:0] actual;
        begin
            actual = result_data[31:0];
            if (actual == expected) begin
                $display("  PASS: lane0 = %0d (0x%08x)", $signed(actual), actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: lane0 = %0d (0x%08x), expected %0d (0x%08x)",
                         $signed(actual), actual, $signed(expected), expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    integer lane;

    //------------------------------------------------------------------------
    // Main test
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_tensor_core_mma.vcd");
        $dumpvars(0, tb_tensor_core_mma);

        rst_n = 0;
        op_valid = 0;
        op_type = 0;
        frag_a = 0;
        frag_b = 0;
        frag_c = 0;
        result_ready = 1'b1;  // Always ready to consume

        #20;
        rst_n = 1;
        #20;

        $display("========================================");
        $display("Tensor Core MMA RTL Testbench");
        $display("========================================");

        //--------------------------------------------------------------------
        // Test 1: FP16 dot2: 1.0*2.0 + 3.0*4.0 + 0.0 = 14.0
        // a = {3.0, 1.0} = {0x4200, 0x3C00} = 0x42003C00
        // b = {4.0, 2.0} = {0x4400, 0x4000} = 0x44004000
        // c = 0.0 = 0x00000000
        // result = 1.0*2.0 + 3.0*4.0 + 0.0 = 2.0 + 12.0 = 14.0 = 0x41600000
        //--------------------------------------------------------------------
        test_num = 1;
        $display("\nTest %0d: FP16 dot2 (1*2 + 3*4 + 0 = 14.0)", test_num);
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
            frag_a[lane*32 +: 32] = 32'h4200_3C00;
            frag_b[lane*32 +: 32] = 32'h4400_4000;
            frag_c[lane*32 +: 32] = 32'h0000_0000;
        end
        issue_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane0_fp32(32'h41600000);
        // Consume result
        @(posedge clk);

        //--------------------------------------------------------------------
        // Test 2: FP16 dot2 with accumulate: 1.0*1.0 + 1.0*1.0 + 5.0 = 7.0
        // a = {1.0, 1.0} = {0x3C00, 0x3C00} = 0x3C003C00
        // b = {1.0, 1.0} = 0x3C003C00
        // c = 5.0 = 0x40A00000
        // result = 1+1+5 = 7.0 = 0x40E00000
        //--------------------------------------------------------------------
        test_num = 2;
        $display("\nTest %0d: FP16 dot2 (1*1 + 1*1 + 5.0 = 7.0)", test_num);
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
            frag_a[lane*32 +: 32] = 32'h3C00_3C00;
            frag_b[lane*32 +: 32] = 32'h3C00_3C00;
            frag_c[lane*32 +: 32] = 32'h40A0_0000;
        end
        issue_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane0_fp32(32'h40E00000);
        @(posedge clk);

        //--------------------------------------------------------------------
        // Test 3: FP16 dot2 zero inputs: 0*X + 0*X + 10.0 = 10.0
        // a = {0.0, 0.0} = 0x00000000
        // b = {2.0, 3.0} = 0x40004200
        // c = 10.0 = 0x41200000
        // result = 0+0+10 = 10.0 = 0x41200000
        //--------------------------------------------------------------------
        test_num = 3;
        $display("\nTest %0d: FP16 dot2 (0*X + 0*X + 10.0 = 10.0)", test_num);
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
            frag_a[lane*32 +: 32] = 32'h0000_0000;
            frag_b[lane*32 +: 32] = 32'h4000_4200;
            frag_c[lane*32 +: 32] = 32'h4120_0000;
        end
        issue_op(`TC_DATA_FP16);
        wait_result(TC_LATENCY + 10);
        check_lane0_fp32(32'h41200000);
        @(posedge clk);

        //--------------------------------------------------------------------
        // Test 4: INT8 dot4 unsigned: [1,2,3,4]·[5,6,7,8] + 0 = 70
        // a = {4, 3, 2, 1} packed as bytes = 0x04030201
        // b = {8, 7, 6, 5} packed as bytes = 0x08070605
        // c = 0
        // result = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        //--------------------------------------------------------------------
        test_num = 4;
        $display("\nTest %0d: INT8 dot4 unsigned ([1,2,3,4]·[5,6,7,8] + 0 = 70)", test_num);
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
            frag_a[lane*32 +: 32] = 32'h04030201;
            frag_b[lane*32 +: 32] = 32'h08070605;
            frag_c[lane*32 +: 32] = 32'h00000000;
        end
        issue_op(`TC_DATA_INT8);
        wait_result(TC_LATENCY + 10);
        check_lane0_int32(32'd70);
        @(posedge clk);

        //--------------------------------------------------------------------
        // Test 5: INT8 dot4 signed: [-1,2,-3,4]·[5,-6,7,-8] + 100 = 30
        // a = {4, -3, 2, -1} = {0x04, 0xFD, 0x02, 0xFF} = 0x04FD02FF
        // b = {-8, 7, -6, 5} = {0xF8, 0x07, 0xFA, 0x05} = 0xF807FA05
        // c = 100 = 0x00000064
        // result = (-1)*5 + 2*(-6) + (-3)*7 + 4*(-8) + 100
        //        = -5 + -12 + -21 + -32 + 100 = 30
        //--------------------------------------------------------------------
        test_num = 5;
        $display("\nTest %0d: INT8 dot4 signed ([-1,2,-3,4]·[5,-6,7,-8] + 100 = 30)", test_num);
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
            frag_a[lane*32 +: 32] = 32'h04FD02FF;
            frag_b[lane*32 +: 32] = 32'hF807FA05;
            frag_c[lane*32 +: 32] = 32'h00000064;
        end
        issue_op(`TC_DATA_INT8);
        wait_result(TC_LATENCY + 10);
        check_lane0_int32(32'd30);
        @(posedge clk);

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
        #100000;
        $display("TIMEOUT: simulation exceeded 100000ns");
        $finish;
    end

endmodule
