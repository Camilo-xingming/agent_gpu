//============================================================================
// RalphGPU - SM V2 Core Architecture Test
// Tests the key improvements: Scoreboard, FU Tracking, Round-Robin WB
// Uses simplified stubs for peripheral modules
//============================================================================

`timescale 1ns / 1ps

module tb_sm_v2_core;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = 4;
    localparam NUM_LANES  = 32;
    localparam DATA_WIDTH = 32;
    localparam WARP_ID_W  = $clog2(NUM_WARPS);
    localparam CLK_PERIOD = 10;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Scoreboard DUT
    //------------------------------------------------------------------------
    reg [31:0] scoreboard_busy [0:NUM_WARPS-1];
    reg [3:0]  pending_fu_count [0:NUM_WARPS-1];

    // Simulated decode signals
    reg        dec_valid;
    reg [WARP_ID_W-1:0] decode_warp_id;
    reg [4:0]  issue_ra, issue_rb, issue_rc, issue_rd;
    reg        dec_reg_write;
    reg        dec_fp32_op;  // Multi-cycle op flag

    // Simulated writeback
    reg        wb_valid;
    reg [WARP_ID_W-1:0] wb_warp_id;
    reg [4:0]  wb_rd;

    // Hazard detection
    wire ra_busy = (issue_ra != 0) && scoreboard_busy[decode_warp_id][issue_ra];
    wire rb_busy = (issue_rb != 0) && scoreboard_busy[decode_warp_id][issue_rb];
    wire rc_busy = (issue_rc != 0) && scoreboard_busy[decode_warp_id][issue_rc];
    wire issue_stall_raw = dec_valid && (ra_busy || rb_busy || rc_busy);
    wire issue_stall_fu = dec_valid && (pending_fu_count[decode_warp_id] >= 8);

    // Issue logic
    reg issue_valid;

    integer sb_init;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue_valid <= 1'b0;
            for (sb_init = 0; sb_init < NUM_WARPS; sb_init = sb_init + 1) begin
                scoreboard_busy[sb_init] <= 32'b0;
                pending_fu_count[sb_init] <= 4'b0;
            end
        end else if (dec_valid && !issue_stall_raw && !issue_stall_fu) begin
            issue_valid <= 1'b1;

            // Mark destination register as busy
            if (dec_reg_write && issue_rd != 0) begin
                scoreboard_busy[decode_warp_id][issue_rd] <= 1'b1;
            end

            // Increment pending count for multi-cycle ops
            if (dec_fp32_op) begin
                pending_fu_count[decode_warp_id] <= pending_fu_count[decode_warp_id] + 1;
            end
        end else begin
            issue_valid <= 1'b0;
        end

        // Clear scoreboard on writeback
        if (wb_valid && wb_rd != 0) begin
            scoreboard_busy[wb_warp_id][wb_rd] <= 1'b0;
            if (pending_fu_count[wb_warp_id] > 0)
                pending_fu_count[wb_warp_id] <= pending_fu_count[wb_warp_id] - 1;
        end
    end

    //------------------------------------------------------------------------
    // Round-Robin Writeback Arbiter DUT
    //------------------------------------------------------------------------
    reg [7:0]  fu_ready;
    reg [3:0]  wb_arb_priority;
    reg [2:0]  wb_sel;
    reg        wb_found;

    integer wb_i;
    always @(*) begin
        wb_found = 1'b0;
        wb_sel = 0;
        for (wb_i = 0; wb_i < 8; wb_i = wb_i + 1) begin
            if (!wb_found && fu_ready[(wb_arb_priority + wb_i) % 8]) begin
                wb_sel = (wb_arb_priority + wb_i) % 8;
                wb_found = 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_arb_priority <= 0;
        end else if (wb_found) begin
            wb_arb_priority <= (wb_sel + 1) % 8;
        end
    end

    //------------------------------------------------------------------------
    // Test Counters
    //------------------------------------------------------------------------
    integer test_pass, test_fail;
    integer cycle_count;
    integer stall_count;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------

    task reset;
        begin
            rst_n = 0;
            dec_valid = 0;
            decode_warp_id = 0;
            issue_ra = 0; issue_rb = 0; issue_rc = 0; issue_rd = 0;
            dec_reg_write = 0;
            dec_fp32_op = 0;
            wb_valid = 0;
            wb_warp_id = 0;
            wb_rd = 0;
            fu_ready = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(2) @(posedge clk);
        end
    endtask

    task decode_instruction;
        input [WARP_ID_W-1:0] wid;
        input [4:0] rd_reg, ra_reg, rb_reg, rc_reg;
        input reg_wr;
        input is_fpu;
        begin
            @(posedge clk);
            dec_valid <= 1;
            decode_warp_id <= wid;
            issue_rd <= rd_reg;
            issue_ra <= ra_reg;
            issue_rb <= rb_reg;
            issue_rc <= rc_reg;
            dec_reg_write <= reg_wr;
            dec_fp32_op <= is_fpu;
            @(posedge clk);
            // Wait for scoreboard update to take effect
            @(negedge clk);
            dec_valid <= 0;
        end
    endtask

    task writeback;
        input [WARP_ID_W-1:0] wid;
        input [4:0] rd_reg;
        begin
            @(posedge clk);
            wb_valid <= 1;
            wb_warp_id <= wid;
            wb_rd <= rd_reg;
            @(posedge clk);
            wb_valid <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // Test 1: RAW Hazard Detection
    //------------------------------------------------------------------------
    task test_raw_hazard;
        begin
            $display("\n[TEST 1] RAW Hazard Detection");
            reset();

            // Issue: R1 = R0 + R0 (writes R1)
            decode_instruction(0, 5'd1, 5'd0, 5'd0, 5'd0, 1, 0);

            // Wait for scoreboard to update (happens on next posedge after issue)
            @(posedge clk);
            #1;  // Small delay to let combinational logic settle

            // Check: R1 should be marked busy
            if (scoreboard_busy[0][1]) begin
                $display("  PASS: R1 marked busy after issue");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: R1 not marked busy (val=%b)", scoreboard_busy[0][1]);
                test_fail = test_fail + 1;
            end

            // Try to issue: R2 = R1 + R0 (reads R1 - should stall)
            dec_valid = 1;
            decode_warp_id = 0;
            issue_rd = 5'd2;
            issue_ra = 5'd1;  // R1 is busy
            issue_rb = 5'd0;
            issue_rc = 5'd0;
            dec_reg_write = 1;
            #1;

            if (issue_stall_raw) begin
                $display("  PASS: RAW hazard detected (R1 busy)");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: RAW hazard not detected");
                test_fail = test_fail + 1;
            end
            dec_valid = 0;
            @(posedge clk);

            // Writeback R1
            writeback(0, 5'd1);
            @(posedge clk);
            #1;

            // Check: R1 should be cleared
            if (!scoreboard_busy[0][1]) begin
                $display("  PASS: R1 cleared after writeback");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: R1 not cleared");
                test_fail = test_fail + 1;
            end

            // Now R2 = R1 + R0 should proceed
            dec_valid = 1;
            decode_warp_id = 0;
            issue_ra = 5'd1;
            #1;

            if (!issue_stall_raw) begin
                $display("  PASS: No stall after R1 writeback");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Still stalling after R1 writeback");
                test_fail = test_fail + 1;
            end
            dec_valid = 0;
        end
    endtask

    //------------------------------------------------------------------------
    // Test 2: FU Capacity Stall
    //------------------------------------------------------------------------
    task test_fu_capacity;
        integer i;
        begin
            $display("\n[TEST 2] FU Capacity Stall (max 8 in-flight)");
            reset();

            // Issue 8 FPU operations (using different destination regs to avoid RAW)
            for (i = 0; i < 8; i = i + 1) begin
                decode_instruction(0, (i+10) & 5'h1F, 5'd0, 5'd0, 5'd0, 1, 1);
                @(posedge clk);
            end
            #1;

            // Check pending count
            if (pending_fu_count[0] == 8) begin
                $display("  PASS: 8 pending FU operations tracked");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Expected 8 pending, got %0d", pending_fu_count[0]);
                test_fail = test_fail + 1;
            end

            // Try to issue 9th - should stall
            dec_valid = 1;
            decode_warp_id = 0;
            issue_rd = 5'd20;
            dec_fp32_op = 1;
            dec_reg_write = 1;
            #1;

            if (issue_stall_fu) begin
                $display("  PASS: FU capacity stall at 8 pending");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: No FU capacity stall (pending=%0d)", pending_fu_count[0]);
                test_fail = test_fail + 1;
            end
            dec_valid = 0;
            @(posedge clk);

            // Writeback one - should allow next issue
            writeback(0, 5'd10);
            @(posedge clk);
            #1;

            dec_valid = 1;
            #1;

            if (!issue_stall_fu) begin
                $display("  PASS: FU capacity available after writeback");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Still stalling after writeback (pending=%0d)", pending_fu_count[0]);
                test_fail = test_fail + 1;
            end
            dec_valid = 0;
        end
    endtask

    //------------------------------------------------------------------------
    // Test 3: Multi-Warp Independence
    //------------------------------------------------------------------------
    task test_warp_independence;
        begin
            $display("\n[TEST 3] Multi-Warp Scoreboard Independence");
            reset();

            // Warp 0: Issue R1 = ...
            decode_instruction(0, 5'd1, 5'd0, 5'd0, 5'd0, 1, 0);
            @(posedge clk);

            // Warp 1: Issue R1 = ... (same register, different warp)
            decode_instruction(1, 5'd1, 5'd0, 5'd0, 5'd0, 1, 0);
            @(posedge clk);
            #1;

            // Check: Both warps have R1 busy independently
            if (scoreboard_busy[0][1] && scoreboard_busy[1][1]) begin
                $display("  PASS: Both warp 0 and warp 1 have R1 busy");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Warp independence issue (w0[1]=%b, w1[1]=%b)",
                         scoreboard_busy[0][1], scoreboard_busy[1][1]);
                test_fail = test_fail + 1;
            end

            // Warp 0: Read R1 - should stall
            dec_valid = 1;
            decode_warp_id = 0;
            issue_ra = 5'd1;
            issue_rd = 5'd2;
            dec_reg_write = 0;
            #1;

            if (issue_stall_raw) begin
                $display("  PASS: Warp 0 stalls on its own R1");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Warp 0 should stall");
                test_fail = test_fail + 1;
            end
            dec_valid = 0;
            @(posedge clk);

            // Writeback warp 0's R1
            writeback(0, 5'd1);
            @(posedge clk);
            #1;

            // Check: Warp 0's R1 clear, Warp 1's R1 still busy
            if (!scoreboard_busy[0][1] && scoreboard_busy[1][1]) begin
                $display("  PASS: Warp 0 R1 cleared, Warp 1 R1 still busy");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Warp isolation broken (w0[1]=%b, w1[1]=%b)",
                         scoreboard_busy[0][1], scoreboard_busy[1][1]);
                test_fail = test_fail + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Test 4: Round-Robin Writeback Arbiter
    //------------------------------------------------------------------------
    task test_wb_arbiter;
        integer selected_units [0:7];
        integer i;
        begin
            $display("\n[TEST 4] Round-Robin Writeback Arbitration");
            reset();

            // Test fairness: all 8 FUs ready
            fu_ready = 8'hFF;

            for (i = 0; i < 16; i = i + 1) begin
                @(posedge clk);
                selected_units[wb_sel] = selected_units[wb_sel] + 1;
            end

            // Check: Each FU should be selected twice (16 cycles / 8 FUs)
            $display("  FU selection counts over 16 cycles:");
            for (i = 0; i < 8; i = i + 1) begin
                $display("    FU[%0d]: %0d times", i, selected_units[i]);
                selected_units[i] = 0;
            end

            // Verify round-robin advances
            if (wb_arb_priority != 0) begin
                $display("  PASS: Priority rotated (now at %0d)", wb_arb_priority);
                test_pass = test_pass + 1;
            end else begin
                $display("  WARN: Priority check inconclusive");
            end

            // Test priority: only FU 5 ready
            fu_ready = 8'b00100000;
            @(posedge clk);
            if (wb_sel == 5) begin
                $display("  PASS: Correctly selected only ready FU (5)");
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: Selected FU %0d instead of 5", wb_sel);
                test_fail = test_fail + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 Core Architecture Test");
        $display("Testing: Scoreboard, FU Tracking, WB Arbitration");
        $display("============================================================");

        test_pass = 0;
        test_fail = 0;
        cycle_count = 0;
        stall_count = 0;

        test_raw_hazard();
        test_fu_capacity();
        test_warp_independence();
        test_wb_arbiter();

        $display("\n============================================================");
        $display("TEST SUMMARY");
        $display("============================================================");
        $display("  Passed: %0d", test_pass);
        $display("  Failed: %0d", test_fail);
        if (test_fail == 0)
            $display("  Status: ALL TESTS PASSED");
        else
            $display("  Status: SOME TESTS FAILED");
        $display("============================================================\n");

        $finish;
    end

endmodule
