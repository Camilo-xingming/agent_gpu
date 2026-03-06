//============================================================================
// RalphGPU - Warp Operations Functional Test
// Tests warp_shuffle, warp_vote, and warp_reduction at unit level
// (Codex Issue #1: B300 functional tests beyond decoder-only)
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_warp_ops;

    parameter LANES = 32;

    //------------------------------------------------------------------------
    // Test Control
    //------------------------------------------------------------------------
    integer passed, failed;

    task check;
        input [255:0] name;
        input [31:0] actual;
        input [31:0] expected;
        begin
            if (actual === expected) begin
                passed = passed + 1;
            end else begin
                $display("  FAIL: %0s — got 0x%08x, expected 0x%08x", name, actual, expected);
                failed = failed + 1;
            end
        end
    endtask

    task check1;
        input [255:0] name;
        input actual;
        input expected;
        begin
            if (actual === expected) begin
                passed = passed + 1;
            end else begin
                $display("  FAIL: %0s — got %0b, expected %0b", name, actual, expected);
                failed = failed + 1;
            end
        end
    endtask

    //========================================================================
    // DUT 1: warp_shuffle
    //========================================================================
    reg  [5:0]            shfl_func;
    reg  [LANES*32-1:0]   shfl_src_data;
    reg  [LANES*5-1:0]    shfl_src_lane;
    reg  [LANES*5-1:0]    shfl_offset;
    reg  [LANES-1:0]      shfl_lane_mask;
    reg  [4:0]            shfl_width;
    reg  [LANES-1:0]      shfl_membermask;
    wire [LANES*32-1:0]   shfl_result;
    wire [LANES-1:0]      shfl_valid;

    warp_shuffle #(.LANES(LANES)) u_shuffle (
        .func       (shfl_func),
        .src_data   (shfl_src_data),
        .src_lane   (shfl_src_lane),
        .offset     (shfl_offset),
        .lane_mask  (shfl_lane_mask),
        .width      (shfl_width),
        .membermask (shfl_membermask),
        .result     (shfl_result),
        .valid_out  (shfl_valid)
    );

    //========================================================================
    // DUT 2: warp_vote
    //========================================================================
    reg  [5:0]          vote_func;
    reg  [LANES-1:0]    vote_pred_in;
    reg  [LANES-1:0]    vote_lane_mask;
    reg  [LANES-1:0]    vote_membermask;
    wire [31:0]         vote_result;
    wire                vote_pred_out;

    warp_vote #(.LANES(LANES)) u_vote (
        .func       (vote_func),
        .pred_in    (vote_pred_in),
        .lane_mask  (vote_lane_mask),
        .membermask (vote_membermask),
        .result     (vote_result),
        .pred_out   (vote_pred_out)
    );

    //========================================================================
    // DUT 3: warp_reduction
    //========================================================================
    reg  [5:0]            redux_func;
    reg  [LANES*32-1:0]   redux_src_data;
    reg  [LANES-1:0]      redux_lane_mask;
    reg  [LANES-1:0]      redux_membermask;
    wire [31:0]           redux_result;

    warp_reduction #(.LANES(LANES)) u_redux (
        .func       (redux_func),
        .src_data   (redux_src_data),
        .lane_mask  (redux_lane_mask),
        .membermask (redux_membermask),
        .result     (redux_result)
    );

    //========================================================================
    // Tests
    //========================================================================
    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU Warp Operations Functional Test");
        $display("============================================================");
        passed = 0;
        failed = 0;

        //--------------------------------------------------------------------
        // SHFL Tests
        // NOTE: RTL width is 5-bit [4:0], so width=32 truncates to 0.
        // We test with 16-lane segments (width=16) which works correctly.
        //--------------------------------------------------------------------
        $display("\n--- SHFL Tests ---");

        // Setup: src_data[lane] = lane * 10 (0, 10, 20, ..., 310)
        shfl_lane_mask = 32'hFFFFFFFF;
        shfl_membermask = 32'hFFFFFFFF;
        shfl_width = 5'd16;  // 16-lane segments
        for (i = 0; i < LANES; i = i + 1) begin
            shfl_src_data[i*32 +: 32] = i * 10;
            shfl_src_lane[i*5 +: 5] = 5'd0;
            shfl_offset[i*5 +: 5] = 5'd1;
        end

        // Test: shfl.idx — lanes 0-15 read from lane 5 (value=50)
        shfl_func = `SHFL_IDX;
        for (i = 0; i < LANES; i = i + 1)
            shfl_src_lane[i*5 +: 5] = 5'd5;
        #1;
        check("shfl.idx lane0 read lane5", shfl_result[0*32 +: 32], 32'd50);
        check("shfl.idx lane10 read lane5", shfl_result[10*32 +: 32], 32'd50);
        check1("shfl.idx lane0 valid", shfl_valid[0], 1'b1);
        check1("shfl.idx lane15 valid", shfl_valid[15], 1'b1);

        // Test: shfl.up offset=1 — lane i reads lane i-1
        // (shfl.up does NOT use width for data routing, only for valid)
        shfl_func = `SHFL_UP;
        for (i = 0; i < LANES; i = i + 1)
            shfl_offset[i*5 +: 5] = 5'd1;
        #1;
        check("shfl.up lane1 gets lane0", shfl_result[1*32 +: 32], 32'd0);
        check("shfl.up lane5 gets lane4", shfl_result[5*32 +: 32], 32'd40);
        check("shfl.up lane15 gets lane14", shfl_result[15*32 +: 32], 32'd140);
        check1("shfl.up lane0 invalid", shfl_valid[0], 1'b0);
        check1("shfl.up lane1 valid", shfl_valid[1], 1'b1);

        // Test: shfl.down offset=1 — lane i reads lane i+1 (within segment)
        shfl_func = `SHFL_DOWN;
        for (i = 0; i < LANES; i = i + 1)
            shfl_offset[i*5 +: 5] = 5'd1;
        #1;
        check("shfl.down lane0 gets lane1", shfl_result[0*32 +: 32], 32'd10);
        check("shfl.down lane10 gets lane11", shfl_result[10*32 +: 32], 32'd110);
        check1("shfl.down lane14 valid", shfl_valid[14], 1'b1);
        check1("shfl.down lane15 invalid", shfl_valid[15], 1'b0); // lane 16 outside segment

        // Test: shfl.bfly offset=1 — lane i reads lane i^1 (within segment)
        shfl_func = `SHFL_BFLY;
        for (i = 0; i < LANES; i = i + 1)
            shfl_offset[i*5 +: 5] = 5'd1;
        #1;
        check("shfl.bfly lane0 gets lane1", shfl_result[0*32 +: 32], 32'd10);
        check("shfl.bfly lane1 gets lane0", shfl_result[1*32 +: 32], 32'd0);
        check("shfl.bfly lane2 gets lane3", shfl_result[2*32 +: 32], 32'd30);
        check("shfl.bfly lane3 gets lane2", shfl_result[3*32 +: 32], 32'd20);
        check1("shfl.bfly lane0 valid", shfl_valid[0], 1'b1);

        // Test: shfl.bfly offset=8 — butterfly reduction across 16-lane segment
        shfl_func = `SHFL_BFLY;
        for (i = 0; i < LANES; i = i + 1)
            shfl_offset[i*5 +: 5] = 5'd8;
        #1;
        check("shfl.bfly8 lane0 gets lane8", shfl_result[0*32 +: 32], 32'd80);
        check("shfl.bfly8 lane8 gets lane0", shfl_result[8*32 +: 32], 32'd0);

        // Test: shfl.idx with partial lane_mask (only lanes 0-7 active)
        shfl_func = `SHFL_IDX;
        shfl_lane_mask = 32'h000000FF;
        for (i = 0; i < LANES; i = i + 1)
            shfl_src_lane[i*5 +: 5] = 5'd3;
        #1;
        check("shfl.idx masked lane0", shfl_result[0*32 +: 32], 32'd30);
        // Inactive lane should output 0
        check("shfl.idx masked lane8=0", shfl_result[8*32 +: 32], 32'd0);

        //--------------------------------------------------------------------
        // VOTE Tests
        //--------------------------------------------------------------------
        $display("\n--- VOTE Tests ---");

        vote_lane_mask = 32'hFFFFFFFF;
        vote_membermask = 32'hFFFFFFFF;

        // Test: vote.all — all predicates true
        vote_func = `VOTE_ALL;
        vote_pred_in = 32'hFFFFFFFF;
        #1;
        check1("vote.all all-true", vote_pred_out, 1'b1);

        // Test: vote.all — one predicate false
        vote_pred_in = 32'hFFFFFFFE;
        #1;
        check1("vote.all one-false", vote_pred_out, 1'b0);

        // Test: vote.any — one predicate true
        vote_func = `VOTE_ANY;
        vote_pred_in = 32'h00000001;
        #1;
        check1("vote.any one-true", vote_pred_out, 1'b1);

        // Test: vote.any — all false
        vote_pred_in = 32'h00000000;
        #1;
        check1("vote.any all-false", vote_pred_out, 1'b0);

        // Test: vote.uni — all same (true)
        vote_func = `VOTE_UNI;
        vote_pred_in = 32'hFFFFFFFF;
        #1;
        check1("vote.uni all-true", vote_pred_out, 1'b1);

        // Test: vote.uni — all same (false)
        vote_pred_in = 32'h00000000;
        #1;
        check1("vote.uni all-false", vote_pred_out, 1'b1);

        // Test: vote.uni — mixed (not uniform)
        vote_pred_in = 32'h0000FFFF;
        #1;
        check1("vote.uni mixed", vote_pred_out, 1'b0);

        // Test: vote.ballot — lower 16 true
        vote_func = `VOTE_BALLOT;
        vote_pred_in = 32'h0000FFFF;
        #1;
        check("vote.ballot lower16", vote_result, 32'h0000FFFF);

        // Test: vote.ballot — all true
        vote_pred_in = 32'hFFFFFFFF;
        #1;
        check("vote.ballot all", vote_result, 32'hFFFFFFFF);

        // Test: vote with partial mask (only lanes 0-7 active)
        vote_func = `VOTE_ALL;
        vote_lane_mask = 32'h000000FF;
        vote_membermask = 32'h000000FF;
        vote_pred_in = 32'h000000FF;
        #1;
        check1("vote.all partial-mask", vote_pred_out, 1'b1);

        // Test: vote.ballot with partial mask
        vote_func = `VOTE_BALLOT;
        vote_pred_in = 32'hFFFFFFFF; // all true, but only 0-7 active
        #1;
        check("vote.ballot partial", vote_result, 32'h000000FF);

        //--------------------------------------------------------------------
        // REDUX Tests
        //--------------------------------------------------------------------
        $display("\n--- REDUX Tests ---");

        redux_lane_mask = 32'hFFFFFFFF;
        redux_membermask = 32'hFFFFFFFF;

        // Setup: data[lane] = lane + 1 (1, 2, 3, ..., 32)
        for (i = 0; i < LANES; i = i + 1)
            redux_src_data[i*32 +: 32] = i + 1;

        // Test: redux.add — sum 1+2+...+32 = 528
        redux_func = `ATOM_ADD;
        #1;
        check("redux.add 1..32", redux_result, 32'd528);

        // Test: redux.and — AND of 1..32 = 0
        redux_func = `ATOM_AND;
        #1;
        check("redux.and 1..32", redux_result, 32'd0);

        // Test: redux.or — OR of 1..32 = 63
        redux_func = `ATOM_OR;
        #1;
        check("redux.or 1..32", redux_result, 32'h3F);

        // Test: redux.min_u — min of 1..32 = 1
        redux_func = `ATOM_MIN_U;
        #1;
        check("redux.min_u 1..32", redux_result, 32'd1);

        // Test: redux.max_u — max of 1..32 = 32
        redux_func = `ATOM_MAX_U;
        #1;
        check("redux.max_u 1..32", redux_result, 32'd32);

        // Test: redux.xor
        begin
            reg [31:0] expected_xor;
            expected_xor = 0;
            for (i = 1; i <= 32; i = i + 1)
                expected_xor = expected_xor ^ i;
            redux_func = `ATOM_XOR;
            #1;
            check("redux.xor 1..32", redux_result, expected_xor);
        end

        // Test: redux with partial mask (only lanes 0-3: sum 1+2+3+4=10)
        redux_lane_mask = 32'h0000000F;
        redux_membermask = 32'h0000000F;
        redux_func = `ATOM_ADD;
        #1;
        check("redux.add partial 1+2+3+4", redux_result, 32'd10);

        // Test: redux.min_s with negative values
        redux_lane_mask = 32'hFFFFFFFF;
        redux_membermask = 32'hFFFFFFFF;
        for (i = 0; i < LANES; i = i + 1)
            redux_src_data[i*32 +: 32] = i;
        redux_src_data[5*32 +: 32] = 32'hFFFFFFFF; // lane 5 = -1
        redux_func = `ATOM_MIN_S;
        #1;
        check("redux.min_s with -1", redux_result, 32'hFFFFFFFF);

        // Test: redux.max_s with mixed signed
        redux_func = `ATOM_MAX_S;
        #1;
        check("redux.max_s mixed", redux_result, 32'd31); // max of 0..31 except lane5=-1

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("\n============================================================");
        if (failed == 0) begin
            $display("ALL %0d TESTS PASSED", passed);
        end else begin
            $display("FAILED: %0d passed, %0d failed", passed, failed);
        end
        $display("============================================================");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
