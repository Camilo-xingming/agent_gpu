//============================================================================
// RalphGPU - Cluster Barrier Unit Testbench
// Tests barrier.cluster multi-SM synchronization
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_cluster_barrier_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter NUM_SM = 4;
    parameter NUM_BARRIERS = 16;
    parameter BARRIER_ID_W = 4;
    parameter THREAD_COUNT_W = 16;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    //------------------------------------------------------------------------
    // DUT Interface
    //------------------------------------------------------------------------
    reg  [NUM_SM-1:0]            sm_arrive_valid;
    reg  [NUM_SM-1:0][BARRIER_ID_W-1:0] sm_arrive_barrier_id;
    reg  [NUM_SM-1:0][THREAD_COUNT_W-1:0] sm_arrive_count;

    reg  [NUM_SM-1:0]            sm_wait_valid;
    reg  [NUM_SM-1:0][BARRIER_ID_W-1:0] sm_wait_barrier_id;
    wire [NUM_SM-1:0]            sm_wait_complete;

    reg  [NUM_SM-1:0]            sm_init_valid;
    reg  [NUM_SM-1:0][BARRIER_ID_W-1:0] sm_init_barrier_id;
    reg  [NUM_SM-1:0][THREAD_COUNT_W-1:0] sm_init_count;

    wire [NUM_BARRIERS-1:0]      barrier_active;
    wire [NUM_BARRIERS-1:0]      barrier_complete;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    cluster_barrier_unit #(
        .NUM_SM(NUM_SM),
        .NUM_BARRIERS(NUM_BARRIERS),
        .BARRIER_ID_W(BARRIER_ID_W),
        .THREAD_COUNT_W(THREAD_COUNT_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sm_arrive_valid(sm_arrive_valid),
        .sm_arrive_barrier_id(sm_arrive_barrier_id),
        .sm_arrive_count(sm_arrive_count),
        .sm_wait_valid(sm_wait_valid),
        .sm_wait_barrier_id(sm_wait_barrier_id),
        .sm_wait_complete(sm_wait_complete),
        .sm_init_valid(sm_init_valid),
        .sm_init_barrier_id(sm_init_barrier_id),
        .sm_init_count(sm_init_count),
        .barrier_active(barrier_active),
        .barrier_complete(barrier_complete)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;
    integer i;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        sm_arrive_valid = 0;
        sm_wait_valid = 0;
        sm_init_valid = 0;
        for (i = 0; i < NUM_SM; i = i + 1) begin
            sm_arrive_barrier_id[i] = 0;
            sm_arrive_count[i] = 0;
            sm_wait_barrier_id[i] = 0;
            sm_init_barrier_id[i] = 0;
            sm_init_count[i] = 0;
        end
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task init_barrier;
        input [BARRIER_ID_W-1:0] barrier_id;
        input [THREAD_COUNT_W-1:0] thread_count;
        input integer sm_id;
    begin
        @(posedge clk);
        sm_init_valid[sm_id] = 1;
        sm_init_barrier_id[sm_id] = barrier_id;
        sm_init_count[sm_id] = thread_count;
        @(posedge clk);
        sm_init_valid = 0;
        #10;
    end
    endtask

    task arrive_barrier;
        input [BARRIER_ID_W-1:0] barrier_id;
        input [THREAD_COUNT_W-1:0] count;
        input integer sm_id;
    begin
        @(posedge clk);
        sm_arrive_valid[sm_id] = 1;
        sm_arrive_barrier_id[sm_id] = barrier_id;
        sm_arrive_count[sm_id] = count;
        @(posedge clk);
        sm_arrive_valid = 0;
        #10;
    end
    endtask

    task check_wait;
        input [BARRIER_ID_W-1:0] barrier_id;
        input integer sm_id;
        output reg complete;
    begin
        @(posedge clk);
        sm_wait_valid[sm_id] = 1;
        sm_wait_barrier_id[sm_id] = barrier_id;
        @(posedge clk);
        complete = sm_wait_complete[sm_id];
        sm_wait_valid = 0;
        #10;
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    reg wait_result;

    initial begin
        $display("============================================================");
        $display("RalphGPU Cluster Barrier Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // Test 1: Simple barrier with 4 SMs, 32 threads each
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Simple 4-SM barrier (128 threads total)", test_num);

        init_barrier(4'd0, 16'd128, 0);
        $display("  Barrier 0 initialized for 128 threads");

        arrive_barrier(4'd0, 16'd32, 0);
        $display("  SM0 arrived with 32 threads");

        check_wait(4'd0, 0, wait_result);
        if (!wait_result) begin
            $display("  Wait incomplete (expected - only 32/128)");
        end

        arrive_barrier(4'd0, 16'd32, 1);
        arrive_barrier(4'd0, 16'd32, 2);
        arrive_barrier(4'd0, 16'd32, 3);
        $display("  All SMs arrived");

        #10;
        check_wait(4'd0, 0, wait_result);
        if (wait_result) begin
            $display("  [PASS] Barrier complete after all SMs arrived");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Barrier should be complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 2: Multiple barriers concurrently
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Multiple concurrent barriers", test_num);

        init_barrier(4'd1, 16'd64, 0);
        init_barrier(4'd2, 16'd64, 0);

        arrive_barrier(4'd1, 16'd32, 0);
        arrive_barrier(4'd1, 16'd32, 1);
        arrive_barrier(4'd2, 16'd32, 2);
        arrive_barrier(4'd2, 16'd32, 3);

        #10;
        check_wait(4'd1, 0, wait_result);
        if (wait_result) begin
            $display("  [PASS] Barrier 1 complete");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Barrier 1 should be complete");
            fail_count = fail_count + 1;
        end

        test_num = test_num + 1;
        check_wait(4'd2, 2, wait_result);
        if (wait_result) begin
            $display("  [PASS] Barrier 2 complete");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Barrier 2 should be complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 3: Partial arrival
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Partial arrival - incomplete barrier", test_num);

        init_barrier(4'd3, 16'd100, 0);
        arrive_barrier(4'd3, 16'd50, 0);

        check_wait(4'd3, 0, wait_result);
        if (!wait_result) begin
            $display("  [PASS] Barrier correctly incomplete (50/100)");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Barrier should not be complete yet");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 4: Simultaneous arrivals
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Simultaneous arrivals from all SMs", test_num);

        init_barrier(4'd4, 16'd128, 0);

        @(posedge clk);
        sm_arrive_valid = 4'b1111;
        for (i = 0; i < NUM_SM; i = i + 1) begin
            sm_arrive_barrier_id[i] = 4'd4;
            sm_arrive_count[i] = 16'd32;
        end
        @(posedge clk);
        sm_arrive_valid = 0;

        #10;
        check_wait(4'd4, 0, wait_result);
        if (wait_result) begin
            $display("  [PASS] Barrier complete after simultaneous arrival");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Barrier should be complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 5: Single-thread barrier
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Single-thread barrier", test_num);

        init_barrier(4'd5, 16'd1, 0);
        arrive_barrier(4'd5, 16'd1, 0);

        check_wait(4'd5, 0, wait_result);
        if (wait_result) begin
            $display("  [PASS] Single-thread barrier complete");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Single-thread barrier should be complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 6: Large thread count
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Large thread count barrier", test_num);

        init_barrier(4'd6, 16'd4096, 0);
        arrive_barrier(4'd6, 16'd1024, 0);
        arrive_barrier(4'd6, 16'd1024, 1);
        arrive_barrier(4'd6, 16'd1024, 2);
        arrive_barrier(4'd6, 16'd1024, 3);

        #10;
        check_wait(4'd6, 0, wait_result);
        if (wait_result) begin
            $display("  [PASS] Large barrier complete");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Large barrier should be complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 7: Wait from different SM
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Wait from different SM than arrive", test_num);

        init_barrier(4'd7, 16'd64, 0);
        arrive_barrier(4'd7, 16'd32, 0);
        arrive_barrier(4'd7, 16'd32, 1);

        check_wait(4'd7, 2, wait_result);
        if (wait_result) begin
            $display("  [PASS] SM2 can wait on barrier it didn't arrive at");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] SM2 should see barrier complete");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Summary
        //====================================================================
        #50;
        $display("\n============================================================");
        $display("Test Summary: %0d passed, %0d failed out of %0d tests",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED!");
        else
            $display("SOME TESTS FAILED!");

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #50000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
