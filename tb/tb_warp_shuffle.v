//============================================================================
// RalphGPU - Warp Shuffle Unit Testbench
// Tests: shfl.idx, shfl.up, shfl.down, shfl.bfly
// Combinational module — no clock, just settle time
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_warp_shuffle;

    parameter LANES = 32;

    //------------------------------------------------------------------------
    // DUT signals
    //------------------------------------------------------------------------
    reg  [5:0]            func;
    reg  [LANES*32-1:0]   src_data;
    reg  [LANES*5-1:0]    src_lane;
    reg  [LANES*5-1:0]    offset;
    reg  [LANES-1:0]      lane_mask;
    reg  [4:0]            width;
    reg  [LANES-1:0]      membermask;
    wire [LANES*32-1:0]   result;
    wire [LANES-1:0]      valid_out;

    //------------------------------------------------------------------------
    // DUT instantiation
    //------------------------------------------------------------------------
    warp_shuffle #(.LANES(LANES)) dut (
        .func       (func),
        .src_data   (src_data),
        .src_lane   (src_lane),
        .offset     (offset),
        .lane_mask  (lane_mask),
        .width      (width),
        .membermask (membermask),
        .result     (result),
        .valid_out  (valid_out)
    );

    //------------------------------------------------------------------------
    // Test tracking
    //------------------------------------------------------------------------
    integer test_count;
    integer pass_count;
    integer fail_count;
    integer i;
    integer errors;

    // Expected values per lane
    reg [31:0] expected_data [0:LANES-1];
    reg        expected_valid [0:LANES-1];

    //------------------------------------------------------------------------
    // Helper: pack source data — lane[i] = i*100 + 42
    //------------------------------------------------------------------------
    task pack_src_data;
        integer k;
        begin
            for (k = 0; k < LANES; k = k + 1) begin
                src_data[k*32 +: 32] = k * 100 + 42;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: clear inputs
    //------------------------------------------------------------------------
    task clear_inputs;
        integer k;
        begin
            func      = 6'b0;
            src_data  = {(LANES*32){1'b0}};
            src_lane  = {(LANES*5){1'b0}};
            offset    = {(LANES*5){1'b0}};
            lane_mask = {LANES{1'b1}};
            width     = 5'd31;
            membermask = {LANES{1'b1}};
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: check results
    //------------------------------------------------------------------------
    task check_results;
        input [48*8-1:0] test_name;
        integer k;
        reg [31:0] got_data;
        reg        got_valid;
        begin
            errors = 0;
            for (k = 0; k < LANES; k = k + 1) begin
                got_data  = result[k*32 +: 32];
                got_valid = valid_out[k];
                if (got_data !== expected_data[k] || got_valid !== expected_valid[k]) begin
                    $display("  ERROR lane %0d: data=%0d (exp %0d), valid=%b (exp %b)",
                             k, got_data, expected_data[k], got_valid, expected_valid[k]);
                    errors = errors + 1;
                end
            end
            test_count = test_count + 1;
            if (errors == 0) begin
                $display("[PASS] %0s", test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s — %0d lane errors", test_name, errors);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        $display("============================================================");
        $display("  Warp Shuffle Unit Testbench");
        $display("============================================================");

        //====================================================================
        // Test 1: shfl.idx — all lanes read from lane 7
        //====================================================================
        clear_inputs;
        pack_src_data;
        func = `SHFL_IDX;
        for (i = 0; i < LANES; i = i + 1) begin
            src_lane[i*5 +: 5] = 5'd7;
        end
        // Expected: all lanes get lane 7's data (7*100+42=742)
        // valid: idx_lane(7) < width(31) && membermask[7](1) => true for all
        for (i = 0; i < LANES; i = i + 1) begin
            expected_data[i]  = 32'd742;
            expected_valid[i] = 1'b1;
        end
        #10;
        check_results("shfl.idx: all lanes read lane 7");

        //====================================================================
        // Test 2: shfl.up offset=1 — each lane reads from lane i-1
        //====================================================================
        clear_inputs;
        pack_src_data;
        func = `SHFL_UP;
        for (i = 0; i < LANES; i = i + 1) begin
            offset[i*5 +: 5] = 5'd1;
        end
        // Expected:
        //   lane 0: my_lane(0) >= offset(1)? No => returns own data, valid=0
        //   lane i (i>=1): src = (i-1)*100+42, valid=1
        for (i = 0; i < LANES; i = i + 1) begin
            if (i == 0) begin
                expected_data[i]  = i * 100 + 42;  // own data
                expected_valid[i] = 1'b0;
            end else begin
                expected_data[i]  = (i - 1) * 100 + 42;
                expected_valid[i] = 1'b1;
            end
        end
        #10;
        check_results("shfl.up offset=1");

        //====================================================================
        // Test 3: shfl.down offset=4 — each lane reads from lane i+4
        //====================================================================
        clear_inputs;
        pack_src_data;
        func = `SHFL_DOWN;
        for (i = 0; i < LANES; i = i + 1) begin
            offset[i*5 +: 5] = 5'd4;
        end
        // Expected:
        //   RTL uses 5-bit arithmetic: (my_lane + lane_offset) wraps at 32
        //   lane i: sum5 = (i+4) & 5'h1F (mod 32)
        //           if sum5 < width(31) => src = sum5*100+42, valid=1
        //           else => own data, valid=0
        //   Lanes 0-26: sum5 = i+4 (4..30), all < 31 => valid
        //   Lane 27: sum5 = 31, 31 < 31? No => own data, valid=0
        //   Lanes 28-31: sum5 wraps to 0-3, all < 31 => valid
        for (i = 0; i < LANES; i = i + 1) begin : gen_exp_down
            reg [4:0] sum5;
            sum5 = i[4:0] + 5'd4;
            if (sum5 < 31) begin
                expected_data[i]  = sum5 * 100 + 42;
                expected_valid[i] = 1'b1;
            end else begin
                expected_data[i]  = i * 100 + 42;  // own data
                expected_valid[i] = 1'b0;
            end
        end
        #10;
        check_results("shfl.down offset=4");

        //====================================================================
        // Test 4: shfl.bfly offset=1 — adjacent pair swap (lane i ^ 1)
        //====================================================================
        clear_inputs;
        pack_src_data;
        func = `SHFL_BFLY;
        for (i = 0; i < LANES; i = i + 1) begin
            offset[i*5 +: 5] = 5'd1;
        end
        // Expected:
        //   src_lane_id = i ^ 1
        //   valid = (src_lane_id < width(31)) && membermask[src_lane_id]
        //   lane 30: 30^1=31, 31 < 31? No => valid=0
        //   lane 31: 31^1=30, 30 < 31? Yes => valid=1
        //   All others: i^1 < 31 => valid=1
        // Note: result data always comes from src_lane_id regardless of valid
        for (i = 0; i < LANES; i = i + 1) begin
            expected_data[i] = (i ^ 1) * 100 + 42;
            if ((i ^ 1) < 31) begin
                expected_valid[i] = 1'b1;
            end else begin
                expected_valid[i] = 1'b0;
            end
        end
        #10;
        check_results("shfl.bfly offset=1 (pair swap)");

        //====================================================================
        // Test 5: shfl.bfly offset=16 — half-warp swap (lane i ^ 16)
        //====================================================================
        clear_inputs;
        pack_src_data;
        func = `SHFL_BFLY;
        for (i = 0; i < LANES; i = i + 1) begin
            offset[i*5 +: 5] = 5'd16;
        end
        // Expected:
        //   src_lane_id = i ^ 16
        //   valid = (src_lane_id < width(31)) && membermask[src_lane_id]
        //   lane 15: 15^16=31, 31 < 31? No => valid=0
        //   lane 31: 31^16=15, 15 < 31? Yes => valid=1
        //   All others: i^16 < 31 => valid=1
        for (i = 0; i < LANES; i = i + 1) begin
            expected_data[i] = (i ^ 16) * 100 + 42;
            if ((i ^ 16) < 31) begin
                expected_valid[i] = 1'b1;
            end else begin
                expected_valid[i] = 1'b0;
            end
        end
        #10;
        check_results("shfl.bfly offset=16 (half-warp swap)");

        //====================================================================
        // Summary
        //====================================================================
        $display("");
        $display("============================================================");
        $display("  SUMMARY: %0d tests, %0d passed, %0d failed", test_count, pass_count, fail_count);
        $display("============================================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        if (errors > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
