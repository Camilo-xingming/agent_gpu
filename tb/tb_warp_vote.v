//============================================================================
// RalphGPU - Warp Vote Testbench
// Combinational tests for warp_vote module
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_warp_vote;

    parameter LANES = 32;

    // Inputs
    reg  [5:0]       func;
    reg  [LANES-1:0] pred_in;
    reg  [LANES-1:0] lane_mask;
    reg  [LANES-1:0] membermask;

    // Outputs
    wire [31:0]      result;
    wire             pred_out;

    // Counters
    integer pass_count;
    integer fail_count;

    // DUT
    warp_vote #(.LANES(LANES)) uut (
        .func       (func),
        .pred_in    (pred_in),
        .lane_mask  (lane_mask),
        .membermask (membermask),
        .result     (result),
        .pred_out   (pred_out)
    );

    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("============================================================");
        $display("  Warp Vote Testbench");
        $display("============================================================");

        //------------------------------------------------------------------
        // Test 1: vote.all — all predicates true
        //   pred_in = 0xFFFFFFFF, all lanes active & participating
        //   Expected: result = 1, pred_out = 1
        //------------------------------------------------------------------
        func       = `VOTE_ALL;
        pred_in    = 32'hFFFFFFFF;
        lane_mask  = 32'hFFFFFFFF;
        membermask = 32'hFFFFFFFF;
        #10;

        if (result == 32'd1 && pred_out == 1'b1) begin
            $display("[PASS] Test 1: vote.all (all true)  result=%0d pred_out=%0b", result, pred_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test 1: vote.all (all true)  result=%0d (exp 1) pred_out=%0b (exp 1)", result, pred_out);
            fail_count = fail_count + 1;
        end

        //------------------------------------------------------------------
        // Test 2: vote.all — one lane false (lane 0)
        //   pred_in = 0xFFFFFFFE, all lanes active & participating
        //   Expected: result = 0, pred_out = 0
        //------------------------------------------------------------------
        func       = `VOTE_ALL;
        pred_in    = 32'hFFFFFFFE;
        lane_mask  = 32'hFFFFFFFF;
        membermask = 32'hFFFFFFFF;
        #10;

        if (result == 32'd0 && pred_out == 1'b0) begin
            $display("[PASS] Test 2: vote.all (one false) result=%0d pred_out=%0b", result, pred_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test 2: vote.all (one false) result=%0d (exp 0) pred_out=%0b (exp 0)", result, pred_out);
            fail_count = fail_count + 1;
        end

        //------------------------------------------------------------------
        // Test 3: vote.any — single lane true (lane 0)
        //   pred_in = 0x00000001, all lanes active & participating
        //   Expected: result = 1, pred_out = 1
        //------------------------------------------------------------------
        func       = `VOTE_ANY;
        pred_in    = 32'h00000001;
        lane_mask  = 32'hFFFFFFFF;
        membermask = 32'hFFFFFFFF;
        #10;

        if (result == 32'd1 && pred_out == 1'b1) begin
            $display("[PASS] Test 3: vote.any (single lane) result=%0d pred_out=%0b", result, pred_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test 3: vote.any (single lane) result=%0d (exp 1) pred_out=%0b (exp 1)", result, pred_out);
            fail_count = fail_count + 1;
        end

        //------------------------------------------------------------------
        // Test 4: vote.uni — all predicates zero (uniform = all agree on 0)
        //   pred_in = 0x00000000, all lanes active & participating
        //   Expected: result = 1, pred_out = 1
        //------------------------------------------------------------------
        func       = `VOTE_UNI;
        pred_in    = 32'h00000000;
        lane_mask  = 32'hFFFFFFFF;
        membermask = 32'hFFFFFFFF;
        #10;

        if (result == 32'd1 && pred_out == 1'b1) begin
            $display("[PASS] Test 4: vote.uni (all zero)  result=%0d pred_out=%0b", result, pred_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test 4: vote.uni (all zero)  result=%0d (exp 1) pred_out=%0b (exp 1)", result, pred_out);
            fail_count = fail_count + 1;
        end

        //------------------------------------------------------------------
        // Test 5: vote.ballot — even lanes have pred=1
        //   pred_in = 0x55555555 (lanes 0,2,4,...,30 = 1), all active
        //   Expected: result = 0x55555555, pred_out = 1 (any active pred true)
        //------------------------------------------------------------------
        func       = `VOTE_BALLOT;
        pred_in    = 32'h55555555;
        lane_mask  = 32'hFFFFFFFF;
        membermask = 32'hFFFFFFFF;
        #10;

        if (result == 32'h55555555 && pred_out == 1'b1) begin
            $display("[PASS] Test 5: vote.ballot (even)   result=0x%08h pred_out=%0b", result, pred_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test 5: vote.ballot (even)   result=0x%08h (exp 0x55555555) pred_out=%0b (exp 1)", result, pred_out);
            fail_count = fail_count + 1;
        end

        //------------------------------------------------------------------
        // Summary
        //------------------------------------------------------------------
        $display("============================================================");
        $display("  Summary: %0d / %0d PASSED", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d TESTS FAILED", fail_count);
        $display("============================================================");

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
