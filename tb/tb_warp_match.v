//============================================================================
// RalphGPU - Warp Match.Sync Testbench
// Focused tests for match.sync.any and match.sync.all operations
// in the warp_collective_unit module.
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_warp_match;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD        = 10;
    parameter NUM_WARPS         = 4;
    parameter NUM_LANES         = 32;
    parameter SHARED_MEM_ADDR_W = 14;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg                             clk;
    reg                             rst_n;

    // Issue interface
    reg                             valid_in;
    reg  [5:0]                      opcode;
    reg  [5:0]                      func;
    reg  [1:0]                      warp_id;
    reg  [31:0]                     thread_mask;
    reg  [31:0]                     membermask;
    reg  [31:0]                     src_data;
    reg  [SHARED_MEM_ADDR_W-1:0]   dst_addr;
    reg  [NUM_LANES*32-1:0]        lane_data_packed;

    // Outputs
    wire                            done;
    wire                            stall_warp;
    wire [31:0]                     result_mask;
    wire [31:0]                     result_data;
    wire                            pred_result;

    // Shared memory interface (unused for match, but must be connected)
    wire                            smem_red_valid;
    wire [SHARED_MEM_ADDR_W-1:0]   smem_red_addr;
    wire [31:0]                     smem_red_data;
    wire [2:0]                      smem_red_op;
    reg                             smem_red_done;

    // mbarrier interface (unused for match)
    wire                            mbarrier_arrive_trigger;
    wire [3:0]                      mbarrier_id;

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;

    //------------------------------------------------------------------------
    // Clock Generation: 10ns period (100 MHz)
    //------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    warp_collective_unit #(
        .NUM_WARPS        (NUM_WARPS),
        .NUM_LANES        (NUM_LANES),
        .SHARED_MEM_ADDR_W(SHARED_MEM_ADDR_W)
    ) dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .valid_in               (valid_in),
        .opcode                 (opcode),
        .func                   (func),
        .warp_id                (warp_id),
        .thread_mask            (thread_mask),
        .membermask             (membermask),
        .src_data               (src_data),
        .dst_addr               (dst_addr),
        .lane_data_packed       (lane_data_packed),
        .done                   (done),
        .stall_warp             (stall_warp),
        .result_mask            (result_mask),
        .result_data            (result_data),
        .pred_result            (pred_result),
        .smem_red_valid         (smem_red_valid),
        .smem_red_addr          (smem_red_addr),
        .smem_red_data          (smem_red_data),
        .smem_red_op            (smem_red_op),
        .smem_red_done          (smem_red_done),
        .mbarrier_arrive_trigger(mbarrier_arrive_trigger),
        .mbarrier_id            (mbarrier_id)
    );

    //------------------------------------------------------------------------
    // Helper task: set all lane data to a single value
    //------------------------------------------------------------------------
    task set_all_lanes;
        input [31:0] value;
        integer k;
        begin
            for (k = 0; k < NUM_LANES; k = k + 1) begin
                lane_data_packed[k*32 +: 32] = value;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper task: set lane[i] = i for all lanes
    //------------------------------------------------------------------------
    task set_lanes_unique;
        integer k;
        begin
            for (k = 0; k < NUM_LANES; k = k + 1) begin
                lane_data_packed[k*32 +: 32] = k;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper task: set lanes 0-15 = val_lo, lanes 16-31 = val_hi
    //------------------------------------------------------------------------
    task set_lanes_half;
        input [31:0] val_lo;
        input [31:0] val_hi;
        integer k;
        begin
            for (k = 0; k < 16; k = k + 1) begin
                lane_data_packed[k*32 +: 32] = val_lo;
            end
            for (k = 16; k < 32; k = k + 1) begin
                lane_data_packed[k*32 +: 32] = val_hi;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper task: issue a match.sync operation and wait for done
    // Sets inputs on negedge, asserts valid_in for 1 cycle, waits for done,
    // then samples result_mask on the following negedge.
    //------------------------------------------------------------------------
    task issue_match_and_wait;
        input [5:0]  t_func;
        input [31:0] t_src_data;
        begin
            // Set inputs on negedge
            @(negedge clk);
            opcode      <= `OP_MATCH_SYNC;
            func        <= t_func;
            warp_id     <= 2'd0;
            thread_mask <= 32'hFFFFFFFF;
            membermask  <= 32'hFFFFFFFF;
            src_data    <= t_src_data;
            dst_addr    <= {SHARED_MEM_ADDR_W{1'b0}};
            valid_in    <= 1'b1;

            // Deassert valid_in after 1 cycle
            @(negedge clk);
            valid_in <= 1'b0;

            // Wait for done=1 (with timeout)
            begin : wait_done
                integer timeout;
                timeout = 0;
                while (!done && timeout < 20) begin
                    @(posedge clk);
                    timeout = timeout + 1;
                end
                if (timeout >= 20) begin
                    $display("[FAIL] Test %0d: Timed out waiting for done", test_num);
                    fail_count = fail_count + 1;
                end
            end

            // Sample on negedge after done
            @(negedge clk);
        end
    endtask

    //------------------------------------------------------------------------
    // Helper task: check result_mask and print PASS/FAIL
    //------------------------------------------------------------------------
    task check_result;
        input [31:0] expected_mask;
        input [255:0] test_name;  // wide enough for description string
        begin
            test_num = test_num + 1;
            if (result_mask === expected_mask) begin
                $display("[PASS] Test %0d: %0s | result_mask=0x%08h (expected 0x%08h)",
                         test_num, test_name, result_mask, expected_mask);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s | result_mask=0x%08h (expected 0x%08h)",
                         test_num, test_name, result_mask, expected_mask);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_warp_match.vcd");
        $dumpvars(0, tb_warp_match);

        // Initialize
        test_num   = 0;
        pass_count = 0;
        fail_count = 0;

        clk           = 0;
        rst_n         = 0;
        valid_in      = 0;
        opcode        = 6'b0;
        func          = 6'b0;
        warp_id       = 2'b0;
        thread_mask   = 32'b0;
        membermask    = 32'b0;
        src_data      = 32'b0;
        dst_addr      = {SHARED_MEM_ADDR_W{1'b0}};
        lane_data_packed = {(NUM_LANES*32){1'b0}};
        smem_red_done = 1'b0;

        $display("============================================================");
        $display(" RalphGPU - Warp match.sync Testbench");
        $display("============================================================");

        // Reset for 5 cycles
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        //====================================================================
        // Test 1: match.any - all lanes have same value (42)
        //   Expected: result_mask = 0xFFFFFFFF (all lanes match)
        //====================================================================
        $display("\n--- Test 1: match.any, all lanes = 42, src_data = 42 ---");
        set_all_lanes(32'd42);
        issue_match_and_wait(`MATCH_ANY, 32'd42);
        check_result(32'hFFFFFFFF, "match.any all same (42)");

        //====================================================================
        // Test 2: match.any - each lane has unique value (lane[i]=i),
        //   src_data=15 -> only lane 15 matches
        //   Expected: result_mask = 0x00008000
        //====================================================================
        $display("\n--- Test 2: match.any, lane[i]=i, src_data=15 ---");
        set_lanes_unique();
        issue_match_and_wait(`MATCH_ANY, 32'd15);
        check_result(32'h00008000, "match.any unique lanes, src=15");

        //====================================================================
        // Test 3: match.any - half lanes match
        //   lanes 0-15 = 100, lanes 16-31 = 200, src_data = 100
        //   Expected: result_mask = 0x0000FFFF
        //====================================================================
        $display("\n--- Test 3: match.any, half lanes = 100, half = 200, src = 100 ---");
        set_lanes_half(32'd100, 32'd200);
        issue_match_and_wait(`MATCH_ANY, 32'd100);
        check_result(32'h0000FFFF, "match.any half match (100)");

        //====================================================================
        // Test 4: match.all - all lanes same (42)
        //   Expected: result_mask = membermask (0xFFFFFFFF)
        //====================================================================
        $display("\n--- Test 4: match.all, all lanes = 42, src_data = 42 ---");
        set_all_lanes(32'd42);
        issue_match_and_wait(`MATCH_ALL, 32'd42);
        check_result(32'hFFFFFFFF, "match.all all same (42)");

        //====================================================================
        // Test 5: match.all - not all same (lanes differ)
        //   lanes 0-15 = 100, lanes 16-31 = 200, src_data = 100
        //   Expected: result_mask = 0 (not all match)
        //====================================================================
        $display("\n--- Test 5: match.all, lanes differ, src = 100 ---");
        set_lanes_half(32'd100, 32'd200);
        issue_match_and_wait(`MATCH_ALL, 32'd100);
        check_result(32'h00000000, "match.all lanes differ");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display(" SUMMARY: %0d / %0d PASSED, %0d FAILED",
                 pass_count, pass_count + fail_count, fail_count);
        $display("============================================================");

        if (fail_count == 0)
            $display(" ALL TESTS PASSED");
        else
            $display(" SOME TESTS FAILED");

        $display("");
        #100;
        $finish;
    end

    //------------------------------------------------------------------------
    // Watchdog timer
    //------------------------------------------------------------------------
    initial begin
        #10000;
        $display("[ERROR] Watchdog timeout - simulation aborted");
        $finish;
    end

endmodule
