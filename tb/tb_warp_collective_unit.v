//============================================================================
// RalphGPU - Warp Collective Unit Testbench
// Tests match.sync, elect.sync, and red.async operations
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_warp_collective_unit;

    parameter CLK_PERIOD = 10;
    parameter NUM_WARPS = 4;
    parameter NUM_LANES = 32;
    parameter SHARED_MEM_ADDR_W = 14;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    // Issue interface
    reg                         valid_in;
    reg [5:0]                   opcode;
    reg [5:0]                   func;
    reg [1:0]                   warp_id;
    reg [31:0]                  thread_mask;
    reg [31:0]                  membermask;
    reg [31:0]                  src_data;
    reg [SHARED_MEM_ADDR_W-1:0] dst_addr;

    // Per-lane data (packed format)
    reg [NUM_LANES*32-1:0]      lane_data_packed;

    // Outputs
    wire                        done;
    wire                        stall_warp;
    wire [31:0]                 result_mask;
    wire [31:0]                 result_data;
    wire                        pred_result;

    // Shared memory interface
    wire                        smem_red_valid;
    wire [SHARED_MEM_ADDR_W-1:0] smem_red_addr;
    wire [31:0]                 smem_red_data;
    wire [2:0]                  smem_red_op;
    reg                         smem_red_done;

    // mbarrier interface
    wire                        mbarrier_arrive_trigger;
    wire [3:0]                  mbarrier_id;

    // Test tracking
    integer test_count;
    integer pass_count;
    integer fail_count;

    //------------------------------------------------------------------------
    // Shared memory simulation
    //------------------------------------------------------------------------
    reg [31:0] smem [0:1023];

    always @(posedge clk) begin
        if (smem_red_valid && !smem_red_done) begin
            smem[smem_red_addr[9:0]] <= smem_red_data;
            smem_red_done <= 1'b1;
            `ifdef SIMULATION
            $display("[SMEM] Write: addr=0x%04x data=0x%08x", smem_red_addr, smem_red_data);
            `endif
        end else begin
            smem_red_done <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    warp_collective_unit #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .SHARED_MEM_ADDR_W(SHARED_MEM_ADDR_W)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .opcode(opcode),
        .func(func),
        .warp_id(warp_id),
        .thread_mask(thread_mask),
        .membermask(membermask),
        .src_data(src_data),
        .dst_addr(dst_addr),
        .lane_data_packed(lane_data_packed),
        .done(done),
        .stall_warp(stall_warp),
        .result_mask(result_mask),
        .result_data(result_data),
        .pred_result(pred_result),
        .smem_red_valid(smem_red_valid),
        .smem_red_addr(smem_red_addr),
        .smem_red_data(smem_red_data),
        .smem_red_op(smem_red_op),
        .smem_red_done(smem_red_done),
        .mbarrier_arrive_trigger(mbarrier_arrive_trigger),
        .mbarrier_id(mbarrier_id)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_inputs;
        begin
            valid_in <= 1'b0;
            opcode <= 6'b0;
            func <= 6'b0;
            warp_id <= 2'b0;
            thread_mask <= 32'hFFFFFFFF;
            membermask <= 32'hFFFFFFFF;
            src_data <= 32'b0;
            dst_addr <= 14'b0;
            smem_red_done <= 1'b0;
        end
    endtask

    task test_match_any;
        input [31:0] mask;
        input [31:0] match_value;
        input [31:0] expected_mask;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: match.sync.any ---", test_count);
            $display("  membermask=0x%08x match_value=0x%08x", mask, match_value);

            @(posedge clk);
            opcode <= `OP_MATCH_SYNC;
            func <= `MATCH_ANY;
            valid_in <= 1'b1;
            membermask <= mask;
            src_data <= match_value;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for done
            wait (done);
            @(posedge clk);

            if (result_mask == expected_mask) begin
                $display("[PASS] result_mask=0x%08x matches expected", result_mask);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] result_mask=0x%08x, expected=0x%08x", result_mask, expected_mask);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_match_all;
        input [31:0] mask;
        input [31:0] match_value;
        input        expected_pred;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: match.sync.all ---", test_count);
            $display("  membermask=0x%08x match_value=0x%08x", mask, match_value);

            @(posedge clk);
            opcode <= `OP_MATCH_SYNC;
            func <= `MATCH_ALL;
            valid_in <= 1'b1;
            membermask <= mask;
            src_data <= match_value;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for done
            wait (done);
            @(posedge clk);

            if (pred_result == expected_pred) begin
                $display("[PASS] pred_result=%b matches expected", pred_result);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] pred_result=%b, expected=%b", pred_result, expected_pred);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_elect_sync;
        input [31:0] mask;
        input [4:0]  expected_lane;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: elect.sync ---", test_count);
            $display("  membermask=0x%08x", mask);

            @(posedge clk);
            opcode <= `OP_ELECT_SYNC;
            func <= `ELECT_SYNC_ONE;
            valid_in <= 1'b1;
            membermask <= mask;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for done
            wait (done);
            @(posedge clk);

            if (result_data[4:0] == expected_lane && result_mask == (32'b1 << expected_lane)) begin
                $display("[PASS] elected lane=%0d, mask=0x%08x", result_data[4:0], result_mask);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] elected lane=%0d (expected %0d), mask=0x%08x",
                         result_data[4:0], expected_lane, result_mask);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_red_async_add;
        input [31:0] mask;
        input [SHARED_MEM_ADDR_W-1:0] addr;
        input [31:0] expected_sum;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: red.async.add ---", test_count);
            $display("  membermask=0x%08x dst_addr=0x%04x", mask, addr);

            @(posedge clk);
            opcode <= `OP_RED_ASYNC;
            func <= `RED_ASYNC_ADD;
            valid_in <= 1'b1;
            membermask <= mask;
            dst_addr <= addr;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for done (async but we wait in test)
            wait (done);
            @(posedge clk);
            @(posedge clk);

            if (smem[addr[9:0]] == expected_sum) begin
                $display("[PASS] smem[0x%04x]=0x%08x matches expected", addr, smem[addr[9:0]]);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] smem[0x%04x]=0x%08x, expected=0x%08x",
                         addr, smem[addr[9:0]], expected_sum);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_red_async_max;
        input [31:0] mask;
        input [SHARED_MEM_ADDR_W-1:0] addr;
        input [31:0] expected_max;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: red.async.max ---", test_count);
            $display("  membermask=0x%08x dst_addr=0x%04x", mask, addr);

            @(posedge clk);
            opcode <= `OP_RED_ASYNC;
            func <= `RED_ASYNC_MAX;
            valid_in <= 1'b1;
            membermask <= mask;
            dst_addr <= addr;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for done
            wait (done);
            @(posedge clk);
            @(posedge clk);

            if (smem[addr[9:0]] == expected_max) begin
                $display("[PASS] smem[0x%04x]=0x%08x matches expected", addr, smem[addr[9:0]]);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] smem[0x%04x]=0x%08x, expected=0x%08x",
                         addr, smem[addr[9:0]], expected_max);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper task to set lane data
    //------------------------------------------------------------------------
    task set_lane_data;
        input integer lane_idx;
        input [31:0] value;
        begin
            lane_data_packed[lane_idx*32 +: 32] = value;
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer i;
    initial begin
        $display("============================================================");
        $display("RalphGPU Warp Collective Unit Testbench");
        $display("============================================================");

        $dumpfile("tb_warp_collective.vcd");
        $dumpvars(0, tb_warp_collective_unit);

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Initialize
        rst_n = 0;
        reset_inputs();

        // Initialize lane data with sequential values
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            set_lane_data(i, i);
        end

        // Reset
        #100;
        rst_n = 1;
        #20;

        //--------------------------------------------------------------------
        // Test 1: match.sync.any - lanes 0-7 match value 5
        //--------------------------------------------------------------------
        // Lane 5 has value 5, so only lane 5 should match
        test_match_any(32'h000000FF, 32'd5, 32'h00000020);  // Lane 5 = bit 5

        //--------------------------------------------------------------------
        // Test 2: match.sync.any - no matching lanes
        //--------------------------------------------------------------------
        test_match_any(32'h0000000F, 32'd100, 32'h00000000);  // No lane has 100

        //--------------------------------------------------------------------
        // Test 3: match.sync.all - all lanes have same value
        //--------------------------------------------------------------------
        // Set all lanes to same value for this test
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            set_lane_data(i, 32'hDEADBEEF);
        end
        @(posedge clk);
        test_match_all(32'hFFFFFFFF, 32'hDEADBEEF, 1'b1);

        //--------------------------------------------------------------------
        // Test 4: match.sync.all - not all lanes match
        //--------------------------------------------------------------------
        set_lane_data(0, 32'hCAFEBABE);  // Different value in lane 0
        @(posedge clk);
        test_match_all(32'hFFFFFFFF, 32'hDEADBEEF, 1'b0);

        // Restore sequential values
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            set_lane_data(i, i);
        end
        @(posedge clk);

        //--------------------------------------------------------------------
        // Test 5: elect.sync - full mask, should elect lane 0
        //--------------------------------------------------------------------
        test_elect_sync(32'hFFFFFFFF, 5'd0);

        //--------------------------------------------------------------------
        // Test 6: elect.sync - mask starting at lane 4
        //--------------------------------------------------------------------
        test_elect_sync(32'hFFFFFFF0, 5'd4);

        //--------------------------------------------------------------------
        // Test 7: elect.sync - only lane 16
        //--------------------------------------------------------------------
        test_elect_sync(32'h00010000, 5'd16);

        //--------------------------------------------------------------------
        // Test 8: elect.sync - sparse mask (lanes 3, 7, 15)
        //--------------------------------------------------------------------
        test_elect_sync(32'h00008088, 5'd3);  // Lowest is lane 3

        //--------------------------------------------------------------------
        // Test 9: red.async.add - sum lanes 0-3 (0+1+2+3=6)
        //--------------------------------------------------------------------
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            set_lane_data(i, i);
        end
        @(posedge clk);
        test_red_async_add(32'h0000000F, 14'h100, 32'd6);

        //--------------------------------------------------------------------
        // Test 10: red.async.add - sum all 32 lanes (0+1+...+31 = 496)
        //--------------------------------------------------------------------
        test_red_async_add(32'hFFFFFFFF, 14'h200, 32'd496);

        //--------------------------------------------------------------------
        // Test 11: red.async.max - find max of lanes 0-7 (max=7)
        //--------------------------------------------------------------------
        test_red_async_max(32'h000000FF, 14'h300, 32'd7);

        //--------------------------------------------------------------------
        // Test 12: red.async.max - non-sequential values
        //--------------------------------------------------------------------
        set_lane_data(0, 100);
        set_lane_data(1, 50);
        set_lane_data(2, 200);
        set_lane_data(3, 75);
        @(posedge clk);
        test_red_async_max(32'h0000000F, 14'h400, 32'd200);

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        #100;
        $display("\n============================================================");
        $display("Test Summary: %0d/%0d tests passed", pass_count, test_count);
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("FAILURES: %0d", fail_count);
        end
        $display("============================================================");

        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
