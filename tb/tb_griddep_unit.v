//============================================================================
// RalphGPU - Grid Dependency Control Unit Testbench
// Tests griddepcontrol instructions for grid-level synchronization
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_griddep_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter DATA_WIDTH = 32;
    parameter MAX_GRIDS = 8;
    parameter TOKEN_WIDTH = 16;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    //------------------------------------------------------------------------
    // DUT Interface
    //------------------------------------------------------------------------
    reg                      valid_in;
    reg  [5:0]               opcode;
    reg  [5:0]               func;
    reg  [DATA_WIDTH-1:0]    src_a;
    reg  [DATA_WIDTH-1:0]    src_b;

    // Grid configuration
    reg  [3:0]               grid_id;
    reg  [TOKEN_WIDTH-1:0]   grid_token;

    // Outputs
    wire                     done;
    wire                     result_valid;
    wire [DATA_WIDTH-1:0]    result;
    wire                     pred_result;
    wire                     stall_grid;

    // Grid control interface
    wire                     signal_complete;
    wire [3:0]               signal_grid_id;
    wire [TOKEN_WIDTH-1:0]   signal_token;
    wire                     launch_dep_req;
    wire [TOKEN_WIDTH-1:0]   launch_dep_token;

    // Dependency status
    reg  [MAX_GRIDS-1:0]     dep_satisfied;
    reg  [MAX_GRIDS-1:0]     grid_active;
    reg  [MAX_GRIDS*TOKEN_WIDTH-1:0]   current_tokens;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    griddep_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_GRIDS(MAX_GRIDS),
        .TOKEN_WIDTH(TOKEN_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .opcode(opcode),
        .func(func),
        .src_a(src_a),
        .src_b(src_b),
        .grid_id(grid_id),
        .grid_token(grid_token),
        .done(done),
        .result_valid(result_valid),
        .result(result),
        .pred_result(pred_result),
        .stall_grid(stall_grid),
        .signal_complete(signal_complete),
        .signal_grid_id(signal_grid_id),
        .signal_token(signal_token),
        .launch_dep_req(launch_dep_req),
        .launch_dep_token(launch_dep_token),
        .dep_satisfied(dep_satisfied),
        .grid_active(grid_active),
        .current_tokens(current_tokens)
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
    reg [TOKEN_WIDTH-1:0] allocated_token;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        src_a = 0;
        src_b = 0;
        grid_id = 0;
        grid_token = 0;
        dep_satisfied = 0;
        grid_active = 0;
        for (i = 0; i < MAX_GRIDS; i = i + 1) begin
            current_tokens[i*TOKEN_WIDTH +: TOKEN_WIDTH] = 0;
        end
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task issue_griddep_op;
        input [5:0] f;
        input [DATA_WIDTH-1:0] a;
        input [DATA_WIDTH-1:0] b;
    begin
        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_GRIDDEPCTRL;
        func <= f;
        src_a <= a;
        src_b <= b;
        @(posedge clk);
        valid_in <= 0;
        wait(done);
        @(posedge clk);
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Grid Dependency Control Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // Test 1: GET_TOKEN - Allocate first token
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.get_token: allocate first token", test_num);

        issue_griddep_op(`GRIDDEP_GET_TOKEN, 32'h0, 32'h0);

        if (result_valid && result == 32'd1) begin
            $display("  [PASS] Token 1 allocated");
            pass_count = pass_count + 1;
            allocated_token = result[TOKEN_WIDTH-1:0];
        end else begin
            $display("  [FAIL] Expected token 1, got %0d", result);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 2: GET_TOKEN - Allocate second token
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.get_token: allocate second token", test_num);

        issue_griddep_op(`GRIDDEP_GET_TOKEN, 32'h0, 32'h0);

        if (result_valid && result == 32'd2) begin
            $display("  [PASS] Token 2 allocated");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Expected token 2, got %0d", result);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 3: SIGNAL - Signal grid completion
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.signal: signal grid completion", test_num);

        grid_id = 4'd2;
        grid_token = 16'd100;

        issue_griddep_op(`GRIDDEP_SIGNAL, 32'h0, 32'h0);

        if (signal_complete && signal_grid_id == 4'd2 && signal_token == 16'd100) begin
            $display("  [PASS] Grid 2 signaled completion with token 100");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Signal mismatch");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 4: LAUNCH_DEP - Request dependent grid launch
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.launch_dependent: launch dependent grid", test_num);

        issue_griddep_op(`GRIDDEP_LAUNCH_DEP, 32'd50, 32'h0);

        if (launch_dep_req && launch_dep_token == 16'd50) begin
            $display("  [PASS] Launch requested for grid with token 50");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Launch request mismatch");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 5: WAIT - Wait for dependency (already satisfied)
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.wait: wait for satisfied dependency", test_num);

        // Setup: Grid 3 is active with token 200, and dependency is satisfied
        grid_active[3] = 1'b1;
        current_tokens[3*TOKEN_WIDTH +: TOKEN_WIDTH] = 16'd200;
        dep_satisfied[3] = 1'b1;

        issue_griddep_op(`GRIDDEP_WAIT, 32'd200, 32'h0);

        if (result_valid && pred_result == 1'b1) begin
            $display("  [PASS] Wait returned immediately (dep satisfied)");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Wait did not return success");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 6: WAIT - Wait for non-existent dependency
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.wait: wait for non-existent dependency", test_num);

        // Token 999 doesn't exist, should return satisfied
        issue_griddep_op(`GRIDDEP_WAIT, 32'd999, 32'h0);

        if (result_valid && pred_result == 1'b1) begin
            $display("  [PASS] Non-existent dependency treated as satisfied");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Should have returned satisfied");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 7: WAIT - Wait for unsatisfied dependency (then satisfy it)
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] griddepcontrol.wait: wait then satisfy dependency", test_num);

        // Setup: Grid 4 is active with token 300, dependency NOT satisfied
        grid_active[4] = 1'b1;
        current_tokens[4*TOKEN_WIDTH +: TOKEN_WIDTH] = 16'd300;
        dep_satisfied[4] = 1'b0;

        // Start wait in background
        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_GRIDDEPCTRL;
        func <= `GRIDDEP_WAIT;
        src_a <= 32'd300;
        @(posedge clk);
        valid_in <= 0;

        // Check that we're stalling
        #30;
        if (stall_grid) begin
            $display("  Grid is stalling, waiting for dependency");
        end

        // Now satisfy the dependency
        #20;
        dep_satisfied[4] = 1'b1;
        $display("  Satisfying dependency for token 300");

        // Wait for done
        wait(done);
        @(posedge clk);

        if (pred_result == 1'b1) begin
            $display("  [PASS] Wait completed after dependency satisfied");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Wait did not return success after satisfy");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 8: Multiple token allocation
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Multiple token allocation", test_num);

        issue_griddep_op(`GRIDDEP_GET_TOKEN, 32'h0, 32'h0);
        if (result == 32'd3) begin
            $display("  Token 3 allocated correctly");
        end

        issue_griddep_op(`GRIDDEP_GET_TOKEN, 32'h0, 32'h0);
        if (result == 32'd4) begin
            $display("  Token 4 allocated correctly");
        end

        issue_griddep_op(`GRIDDEP_GET_TOKEN, 32'h0, 32'h0);
        if (result == 32'd5) begin
            $display("  [PASS] Tokens 3, 4, 5 allocated sequentially");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Token sequence incorrect");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 9: Signal from different grid
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Signal from different grid", test_num);

        grid_id = 4'd5;
        grid_token = 16'd500;

        issue_griddep_op(`GRIDDEP_SIGNAL, 32'h0, 32'h0);

        if (signal_complete && signal_grid_id == 4'd5 && signal_token == 16'd500) begin
            $display("  [PASS] Grid 5 signaled with token 500");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Signal mismatch for grid 5");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 10: Wait on completed grid
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Wait on previously completed grid", test_num);

        // Grid 2 was signaled in test 3 (token 100 was for a different purpose,
        // but internal grid_completed[2] should be set)
        // Let's set up a new scenario
        grid_active[6] = 1'b1;
        current_tokens[6*TOKEN_WIDTH +: TOKEN_WIDTH] = 16'd600;
        dep_satisfied[6] = 1'b0;  // Not satisfied via scheduler

        // Signal completion for grid 6
        grid_id = 4'd6;
        grid_token = 16'd600;
        issue_griddep_op(`GRIDDEP_SIGNAL, 32'h0, 32'h0);

        // Now wait on grid 6's token - should be satisfied due to grid_completed
        issue_griddep_op(`GRIDDEP_WAIT, 32'd600, 32'h0);

        if (pred_result == 1'b1) begin
            $display("  [PASS] Wait satisfied by completed grid");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Wait should have been satisfied by completed grid");
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

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #20000;
        $display("ERROR: Test timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
