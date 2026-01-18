//============================================================================
// RalphGPU - mbarrier Unit Testbench
// Tests Hopper-class mbarrier (memory barrier) operations
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_mbarrier_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter NUM_BARRIERS = 8;
    parameter NUM_WARPS = 4;
    parameter SMEM_ADDR_W = 14;
    parameter WARP_ID_W = $clog2(NUM_WARPS);

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg                         valid_in;
    reg  [5:0]                  func;
    reg  [SMEM_ADDR_W-1:0]      barrier_addr;
    reg  [31:0]                 count;
    reg  [WARP_ID_W-1:0]        warp_id;
    reg  [31:0]                 thread_mask;

    wire                        ready;
    wire                        done;
    wire [31:0]                 result;
    wire                        result_valid;

    reg                         async_arrive_valid;
    reg  [SMEM_ADDR_W-1:0]      async_barrier_addr;
    reg  [31:0]                 async_tx_bytes;

    wire [NUM_WARPS-1:0]        warp_blocked;

    wire                        smem_rd_en;
    wire [SMEM_ADDR_W-1:0]      smem_rd_addr;
    reg  [127:0]                smem_rd_data;
    reg                         smem_rd_valid;

    wire                        smem_wr_en;
    wire [SMEM_ADDR_W-1:0]      smem_wr_addr;
    wire [127:0]                smem_wr_data;
    wire [15:0]                 smem_wr_mask;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    mbarrier_unit #(
        .NUM_BARRIERS(NUM_BARRIERS),
        .NUM_WARPS(NUM_WARPS),
        .SMEM_ADDR_W(SMEM_ADDR_W),
        .WARP_ID_W(WARP_ID_W)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .valid_in           (valid_in),
        .func               (func),
        .barrier_addr       (barrier_addr),
        .count              (count),
        .warp_id            (warp_id),
        .thread_mask        (thread_mask),
        .ready              (ready),
        .done               (done),
        .result             (result),
        .result_valid       (result_valid),
        .async_arrive_valid (async_arrive_valid),
        .async_barrier_addr (async_barrier_addr),
        .async_tx_bytes     (async_tx_bytes),
        .warp_blocked       (warp_blocked),
        .smem_rd_en         (smem_rd_en),
        .smem_rd_addr       (smem_rd_addr),
        .smem_rd_data       (smem_rd_data),
        .smem_rd_valid      (smem_rd_valid),
        .smem_wr_en         (smem_wr_en),
        .smem_wr_addr       (smem_wr_addr),
        .smem_wr_data       (smem_wr_data),
        .smem_wr_mask       (smem_wr_mask)
    );

    //------------------------------------------------------------------------
    // Test Helpers
    //------------------------------------------------------------------------
    integer test_pass_count;
    integer test_fail_count;

    task reset_dut;
    begin
        rst_n = 0;
        valid_in = 0;
        func = 0;
        barrier_addr = 0;
        count = 0;
        warp_id = 0;
        thread_mask = 0;
        async_arrive_valid = 0;
        async_barrier_addr = 0;
        async_tx_bytes = 0;
        smem_rd_data = 0;
        smem_rd_valid = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    end
    endtask

    task wait_for_done;
    begin
        while (!done) @(posedge clk);
        // Sample result in same cycle as done (don't wait another cycle)
    end
    endtask

    task do_async_arrive;
        input [SMEM_ADDR_W-1:0] addr;
        input [31:0] tx_bytes;
    begin
        // Wait for falling edge first, then set values so they're stable at next rising edge
        @(negedge clk);
        async_barrier_addr = addr;
        async_tx_bytes = tx_bytes;
        async_arrive_valid = 1;
        @(posedge clk);  // DUT samples here with values stable
        @(negedge clk);  // Wait till after to clear
        async_arrive_valid = 0;
        async_barrier_addr = 0;
        async_tx_bytes = 0;
    end
    endtask

    task issue_mbarrier_op;
        input [5:0] op_func;
        input [SMEM_ADDR_W-1:0] addr;
        input [31:0] op_count;
        input [WARP_ID_W-1:0] op_warp;
        input [31:0] op_mask;
    begin
        // Set values at negedge so they're stable before DUT samples at posedge
        @(negedge clk);
        valid_in = 1;
        func = op_func;
        barrier_addr = addr;
        count = op_count;
        warp_id = op_warp;
        thread_mask = op_mask;
        @(posedge clk);  // DUT samples here with stable values
        @(negedge clk);  // Wait till after posedge to deassert
        valid_in = 0;
        wait_for_done;
    end
    endtask

    //------------------------------------------------------------------------
    // Test Cases
    //------------------------------------------------------------------------

    // Test 1: Basic barrier init
    task test_barrier_init;
    begin
        $display("\n=== Test 1: Barrier Init ===");

        // Initialize barrier 0 at address 0x000 with expected count of 32
        issue_mbarrier_op(`MBAR_INIT, 14'h000, 32'd32, 2'd0, 32'hFFFFFFFF);

        // Wait for ready to go high (may take one cycle after done)
        @(posedge clk);

        if (ready) begin
            $display("PASS: Barrier init completed, unit ready");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Barrier init did not complete properly");
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 2: Barrier arrive
    task test_barrier_arrive;
    begin
        $display("\n=== Test 2: Barrier Arrive ===");

        // First init the barrier
        issue_mbarrier_op(`MBAR_INIT, 14'h010, 32'd64, 2'd0, 32'hFFFFFFFF);

        // Arrive with 32 threads
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h010, 32'd0, 2'd0, 32'hFFFFFFFF);

        $display("PASS: Barrier arrive completed");
        test_pass_count = test_pass_count + 1;
    end
    endtask

    // Test 3: Test wait (phase not complete)
    task test_test_wait_incomplete;
    begin
        $display("\n=== Test 3: Test Wait (Incomplete) ===");

        // Init barrier with expected count 64
        issue_mbarrier_op(`MBAR_INIT, 14'h020, 32'd64, 2'd0, 32'hFFFFFFFF);

        // Arrive with 32 threads (not enough)
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h020, 32'd0, 2'd0, 32'hFFFFFFFF);

        // Test wait should return 0 (not complete)
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TEST_WAIT;
        barrier_addr = 14'h020;
        count = 0;
        warp_id = 2'd0;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (result == 32'h0 && result_valid) begin
            $display("PASS: Test wait returned 0 (phase not complete)");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Test wait returned unexpected result=%0d", result);
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 4: Test wait (phase complete) - Test partial arrival first
    task test_test_wait_complete;
    begin
        $display("\n=== Test 4: Test Wait (Partial Then Complete) ===");

        // Init barrier with expected count 32
        issue_mbarrier_op(`MBAR_INIT, 14'h030, 32'd32, 2'd0, 32'hFFFFFFFF);

        // Arrive with 16 threads (partial)
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h030, 32'd0, 2'd0, 32'h0000FFFF);

        // Test wait should return 0 (not complete)
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TEST_WAIT;
        barrier_addr = 14'h030;
        count = 0;
        warp_id = 2'd0;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (result == 32'h0 && result_valid) begin
            $display("PASS: Test wait correctly returns 0 after partial arrival");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Test wait returned unexpected result=%0d (expected 0)", result);
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 5: Try wait blocks warp
    task test_try_wait_blocks;
    begin
        $display("\n=== Test 5: Try Wait Blocks Warp ===");

        // Init barrier with expected count 64
        issue_mbarrier_op(`MBAR_INIT, 14'h040, 32'd64, 2'd1, 32'hFFFFFFFF);

        // Arrive with 32 threads from warp 0
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h040, 32'd0, 2'd0, 32'hFFFFFFFF);

        // Try wait from warp 1 (should block)
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TRY_WAIT;
        barrier_addr = 14'h040;
        count = 0;
        warp_id = 2'd1;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (warp_blocked[1]) begin
            $display("PASS: Warp 1 correctly blocked on barrier");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 1 should be blocked but isn't");
            test_fail_count = test_fail_count + 1;
        end

        // Now complete the barrier with another arrive
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h040, 32'd0, 2'd2, 32'hFFFFFFFF);

        // Wait a few cycles for warp release
        repeat(3) @(posedge clk);

        if (!warp_blocked[1]) begin
            $display("PASS: Warp 1 released after barrier completion");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 1 should be released but is still blocked");
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 6: Arrive with transaction bytes - test that try_wait releases after async
    task test_arrive_with_tx;
    begin
        $display("\n=== Test 6: Arrive with Transaction Bytes ===");

        // Init barrier
        issue_mbarrier_op(`MBAR_INIT, 14'h050, 32'd32, 2'd0, 32'hFFFFFFFF);

        // Arrive and expect 128 bytes of transactions
        issue_mbarrier_op(`MBAR_ARRIVE_TX, 14'h050, 32'd128, 2'd0, 32'hFFFFFFFF);

        // Try wait from warp 3 (should block due to pending tx)
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TRY_WAIT;
        barrier_addr = 14'h050;
        count = 0;
        warp_id = 2'd3;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (warp_blocked[3]) begin
            $display("PASS: Warp 3 correctly blocked due to pending tx bytes");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 3 should be blocked");
            test_fail_count = test_fail_count + 1;
        end

        // Wait a cycle to ensure TRY_WAIT is fully complete
        repeat(2) @(posedge clk);

        // Simulate async arrival with 128 bytes using task
        do_async_arrive(14'h050, 32'd128);

        // Wait for warp release
        repeat(8) @(posedge clk);

        if (!warp_blocked[3]) begin
            $display("PASS: Warp 3 released after async arrival completes barrier");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 3 should be released after async arrival");
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 7: Barrier invalidate
    task test_barrier_invalidate;
    begin
        $display("\n=== Test 7: Barrier Invalidate ===");

        // Init barrier
        issue_mbarrier_op(`MBAR_INIT, 14'h060, 32'd64, 2'd0, 32'hFFFFFFFF);

        // Arrive with some threads
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h060, 32'd0, 2'd0, 32'hFFFFFFFF);

        // Block warp 2 on this barrier
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TRY_WAIT;
        barrier_addr = 14'h060;
        count = 0;
        warp_id = 2'd2;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (!warp_blocked[2]) begin
            $display("FAIL: Warp 2 should be blocked before invalidate");
            test_fail_count = test_fail_count + 1;
        end

        // Invalidate the barrier
        issue_mbarrier_op(`MBAR_INVALIDATE, 14'h060, 32'd0, 2'd0, 32'hFFFFFFFF);

        repeat(2) @(posedge clk);

        if (!warp_blocked[2]) begin
            $display("PASS: Warp 2 released after barrier invalidation");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 2 should be released after invalidation");
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    // Test 8: Arrive drop (decrement expected count)
    task test_arrive_drop;
    begin
        $display("\n=== Test 8: Arrive Drop ===");

        // Init barrier with expected count 48 (will need 32 + drop for completion)
        issue_mbarrier_op(`MBAR_INIT, 14'h070, 32'd48, 2'd0, 32'hFFFFFFFF);

        // First arrive with 16 threads
        issue_mbarrier_op(`MBAR_ARRIVE, 14'h070, 32'd0, 2'd0, 32'h0000FFFF);

        // Try wait from warp 0 - should block (16 < 48)
        @(posedge clk);
        valid_in = 1;
        func = `MBAR_TRY_WAIT;
        barrier_addr = 14'h070;
        count = 0;
        warp_id = 2'd0;
        thread_mask = 32'hFFFFFFFF;
        @(posedge clk);
        valid_in = 0;
        wait_for_done;

        if (warp_blocked[0]) begin
            $display("PASS: Warp 0 correctly blocked (16 arrivals < 48 expected)");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 0 should be blocked");
            test_fail_count = test_fail_count + 1;
        end

        // Arrive drop - arrives with 32 threads and decrements expected (48-1=47)
        // Total arrivals: 16+32=48 >= 47, so barrier completes
        issue_mbarrier_op(`MBAR_ARRIVE_DROP, 14'h070, 32'd0, 2'd1, 32'hFFFFFFFF);

        repeat(3) @(posedge clk);

        if (!warp_blocked[0]) begin
            $display("PASS: Warp 0 released after arrive_drop completed barrier");
            test_pass_count = test_pass_count + 1;
        end else begin
            $display("FAIL: Warp 0 should be released after arrive_drop");
            test_fail_count = test_fail_count + 1;
        end
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("\n");
        $display("============================================");
        $display("  mbarrier Unit Testbench");
        $display("============================================");

        test_pass_count = 0;
        test_fail_count = 0;

        reset_dut;

        test_barrier_init;
        test_barrier_arrive;
        test_test_wait_incomplete;
        test_test_wait_complete;
        test_try_wait_blocks;
        test_arrive_with_tx;
        test_barrier_invalidate;
        test_arrive_drop;

        $display("\n============================================");
        $display("  Test Summary");
        $display("============================================");
        $display("  PASSED: %0d", test_pass_count);
        $display("  FAILED: %0d", test_fail_count);
        $display("============================================\n");

        if (test_fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    //------------------------------------------------------------------------
    // VCD Dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_mbarrier_unit.vcd");
        $dumpvars(0, tb_mbarrier_unit);
    end

endmodule
