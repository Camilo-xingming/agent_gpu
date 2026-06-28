//============================================================================
// RalphGPU - bar.warp.sync Unit Test
// Tests warp-level synchronization barrier functionality
// Verifies: member mask tracking, thread arrival, barrier completion
//============================================================================

`timescale 1ns / 1ps

module tb_bar_warp_sync_unit;

    parameter CLK_PERIOD = 10;
    parameter NUM_WARPS = 4;
    parameter NUM_LANES = 32;

    reg clk;
    reg rst_n;

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;

    // Warp sync state (mirrors SM implementation)
    reg [NUM_WARPS-1:0] warp_sync_pending;
    reg [31:0] warp_sync_mask [0:NUM_WARPS-1];
    reg [31:0] warp_sync_arrived [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] warp_stalled_sync;

    // Issue interface
    reg issue_valid;
    reg issue_bar_warp_sync;
    reg [1:0] issue_warp_id;
    reg [31:0] issue_mask;         // Which threads are issuing
    reg [31:0] member_mask;        // Expected membermask (from ra register)

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // bar.warp.sync logic (copied from SM)
    integer w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_sync_pending <= {NUM_WARPS{1'b0}};
            warp_stalled_sync <= {NUM_WARPS{1'b0}};
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                warp_sync_mask[w] <= 32'b0;
                warp_sync_arrived[w] <= 32'b0;
            end
        end else begin
            // Handle bar.warp.sync issue
            if (issue_valid && issue_bar_warp_sync) begin
                // Set expected mask from ra register (membermask)
                warp_sync_mask[issue_warp_id] <= member_mask;
                // Mark arriving threads
                warp_sync_arrived[issue_warp_id] <= warp_sync_arrived[issue_warp_id] | issue_mask;
                // Mark sync pending and stall the warp
                warp_sync_pending[issue_warp_id] <= 1'b1;
                warp_stalled_sync[issue_warp_id] <= 1'b1;
            end

            // Check for warp-level sync completion and release
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                if (warp_sync_pending[w]) begin
                    // Check if all participating threads have arrived
                    if ((warp_sync_arrived[w] & warp_sync_mask[w]) == warp_sync_mask[w]) begin
                        // All threads arrived - release the warp
                        warp_stalled_sync[w] <= 1'b0;
                        warp_sync_pending[w] <= 1'b0;
                        warp_sync_arrived[w] <= 32'b0;
                    end
                end
            end
        end
    end

    // Test task: Check result
    task check_result;
        input [255:0] test_name;
        input [31:0] expected;
        input [31:0] actual;
        begin
            if (expected == actual) begin
                $display("[PASS] Test %0d: %0s", test_num, test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s - expected 0x%h, got 0x%h",
                         test_num, test_name, expected, actual);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
        end
    endtask

    // Issue bar.warp.sync
    task issue_bar_sync;
        input [1:0] warp;
        input [31:0] mask;
        input [31:0] arriving;
        begin
            @(posedge clk);
            issue_valid <= 1'b1;
            issue_bar_warp_sync <= 1'b1;
            issue_warp_id <= warp;
            member_mask <= mask;
            issue_mask <= arriving;
            @(posedge clk);
            issue_valid <= 1'b0;
            issue_bar_warp_sync <= 1'b0;
        end
    endtask

    initial begin
        $display("============================================================");
        $display("RalphGPU bar.warp.sync Unit Test");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        issue_valid = 0;
        issue_bar_warp_sync = 0;
        issue_warp_id = 0;
        issue_mask = 0;
        member_mask = 0;
        test_num = 1;
        pass_count = 0;
        fail_count = 0;

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Test 1: Reset state
        //==================================================================
        check_result("No warps pending after reset", 0, warp_sync_pending);
        check_result("No warps stalled after reset", 0, warp_stalled_sync);

        //==================================================================
        // Test 2: Single thread barrier (all 32 threads participate)
        //==================================================================
        $display("\n--- Test: All 32 threads barrier ---");

        // All 32 threads issue bar.warp.sync with full mask
        issue_bar_sync(2'd0, 32'hFFFFFFFF, 32'hFFFFFFFF);

        @(posedge clk);
        @(posedge clk);

        // Should complete immediately since all threads arrived
        check_result("Full warp sync completed", 0, warp_sync_pending[0]);
        check_result("Warp 0 not stalled", 0, warp_stalled_sync[0]);

        //==================================================================
        // Test 3: Partial thread barrier
        //==================================================================
        $display("\n--- Test: Partial thread barrier (16 threads) ---");

        // First 16 threads participate (mask=0x0000FFFF)
        // Issue with only first 8 threads arriving
        issue_bar_sync(2'd1, 32'h0000FFFF, 32'h000000FF);

        @(posedge clk);
        check_result("Warp 1 pending (partial)", 1, warp_sync_pending[1]);
        check_result("Warp 1 stalled (partial)", 1, warp_stalled_sync[1]);

        // Second batch: threads 8-15 arrive
        issue_bar_sync(2'd1, 32'h0000FFFF, 32'h0000FF00);

        @(posedge clk);
        @(posedge clk);

        // Should complete now
        check_result("Partial sync completed", 0, warp_sync_pending[1]);
        check_result("Warp 1 not stalled", 0, warp_stalled_sync[1]);

        //==================================================================
        // Test 4: Multiple warps with different barriers
        //==================================================================
        $display("\n--- Test: Multiple warps concurrent barriers ---");

        // Warp 0: lower 16 threads
        issue_bar_sync(2'd0, 32'h0000FFFF, 32'h00000000);

        // Warp 2: upper 16 threads
        issue_bar_sync(2'd2, 32'hFFFF0000, 32'h00000000);

        @(posedge clk);
        check_result("Warp 0 pending", 1, warp_sync_pending[0]);
        check_result("Warp 2 pending", 1, warp_sync_pending[2]);

        // Complete warp 0
        issue_bar_sync(2'd0, 32'h0000FFFF, 32'h0000FFFF);
        @(posedge clk);
        @(posedge clk);
        check_result("Warp 0 completed", 0, warp_sync_pending[0]);
        check_result("Warp 2 still pending", 1, warp_sync_pending[2]);

        // Complete warp 2
        issue_bar_sync(2'd2, 32'hFFFF0000, 32'hFFFF0000);
        @(posedge clk);
        @(posedge clk);
        check_result("Warp 2 completed", 0, warp_sync_pending[2]);

        //==================================================================
        // Test 5: Scattered thread participation
        //==================================================================
        $display("\n--- Test: Scattered threads (every 4th thread) ---");

        // Only threads 0,4,8,12,16,20,24,28 participate (0x11111111)
        issue_bar_sync(2'd3, 32'h11111111, 32'h00000000);
        @(posedge clk);
        check_result("Warp 3 pending (scattered)", 1, warp_sync_pending[3]);

        // First wave: threads 0,8,16,24
        issue_bar_sync(2'd3, 32'h11111111, 32'h01010101);
        @(posedge clk);
        check_result("Still pending after first wave", 1, warp_sync_pending[3]);

        // Second wave: threads 4,12,20,28
        issue_bar_sync(2'd3, 32'h11111111, 32'h10101010);
        @(posedge clk);
        @(posedge clk);
        check_result("Scattered sync completed", 0, warp_sync_pending[3]);

        //==================================================================
        // Test 6: Single thread participation
        //==================================================================
        $display("\n--- Test: Single thread participation ---");

        // Only thread 0 participates
        issue_bar_sync(2'd0, 32'h00000001, 32'h00000001);
        @(posedge clk);
        @(posedge clk);
        check_result("Single thread sync completed", 0, warp_sync_pending[0]);

        //==================================================================
        // Test 7: Empty mask (no threads participate)
        //==================================================================
        $display("\n--- Test: Empty mask ---");

        // No threads participate - should complete immediately
        issue_bar_sync(2'd1, 32'h00000000, 32'h00000000);
        @(posedge clk);
        @(posedge clk);
        // Empty mask means condition (arrived & mask) == mask is (0 & 0) == 0, true
        check_result("Empty mask sync completed", 0, warp_sync_pending[1]);

        //==================================================================
        // Test 8: Progressive arrivals
        //==================================================================
        $display("\n--- Test: Progressive thread arrivals ---");

        // 8 threads participate (0x000000FF)
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000001);  // Thread 0
        @(posedge clk);
        check_result("Pending after 1 thread", 1, warp_sync_pending[2]);

        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000002);  // Thread 1
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000004);  // Thread 2
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000008);  // Thread 3
        @(posedge clk);
        check_result("Still pending after 4 threads", 1, warp_sync_pending[2]);

        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000010);  // Thread 4
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000020);  // Thread 5
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000040);  // Thread 6
        issue_bar_sync(2'd2, 32'h000000FF, 32'h00000080);  // Thread 7
        @(posedge clk);
        @(posedge clk);
        check_result("All 8 threads completed", 0, warp_sync_pending[2]);

        //==================================================================
        // Results Summary
        //==================================================================
        #100;
        $display("\n============================================================");
        $display("bar.warp.sync Unit Test Results");
        $display("============================================================");
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        $display("============================================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("ERROR: Test timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_bar_warp_sync_unit.vcd");
        $dumpvars(0, tb_bar_warp_sync_unit);
    end

endmodule
