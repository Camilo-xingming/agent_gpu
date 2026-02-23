//============================================================================
// RalphGPU - Control Flow Unit Testbench
// Tests branch, call/return, divergence, and reconvergence
//============================================================================

`timescale 1ns / 1ps

module tb_control_flow_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam STACK_DEPTH = 8;
    localparam NUM_WARPS   = 4;
    localparam WARP_ID_W   = 2;  // $clog2(4)
    localparam CLK_PERIOD  = 10;

    //------------------------------------------------------------------------
    // Branch type definitions (must match RTL)
    //------------------------------------------------------------------------
    localparam [5:0] BR_UNCONDITIONAL = 6'b000000;
    localparam [5:0] BR_IF_TRUE       = 6'b000001;
    localparam [5:0] BR_IF_FALSE      = 6'b000010;
    localparam [5:0] BR_UNIFORM       = 6'b000011;
    localparam [5:0] BR_INDIRECT      = 6'b000100;

    //------------------------------------------------------------------------
    // DUT signals
    //------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [31:0] pc_current;
    reg  [WARP_ID_W-1:0] warp_id;
    reg  [31:0] active_mask;
    reg         branch_valid;
    reg  [5:0]  branch_type;
    reg  [31:0] branch_target;
    reg  [31:0] branch_cond;
    reg         is_uniform;
    reg         call_valid;
    reg  [31:0] call_target;
    reg         ret_valid;
    reg  [31:0] diverge_mask;

    wire [31:0] next_pc;
    wire [31:0] next_active_mask;
    wire        pc_valid;
    wire        stall;
    wire [31:0] reconverge_pc;
    wire        at_reconverge;
    wire        stack_overflow;
    wire        stack_underflow;

    //------------------------------------------------------------------------
    // DUT instantiation
    //------------------------------------------------------------------------
    control_flow_unit #(
        .STACK_DEPTH(STACK_DEPTH),
        .NUM_WARPS(NUM_WARPS),
        .WARP_ID_W(WARP_ID_W)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .pc_current      (pc_current),
        .warp_id         (warp_id),
        .active_mask     (active_mask),
        .branch_valid    (branch_valid),
        .branch_type     (branch_type),
        .branch_target   (branch_target),
        .branch_cond     (branch_cond),
        .is_uniform      (is_uniform),
        .call_valid      (call_valid),
        .call_target     (call_target),
        .ret_valid       (ret_valid),
        .diverge_mask    (diverge_mask),
        .next_pc         (next_pc),
        .next_active_mask(next_active_mask),
        .pc_valid        (pc_valid),
        .stall           (stall),
        .reconverge_pc   (reconverge_pc),
        .at_reconverge   (at_reconverge),
        .stack_overflow  (stack_overflow),
        .stack_underflow (stack_underflow)
    );

    //------------------------------------------------------------------------
    // Clock generation
    //------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    //------------------------------------------------------------------------
    // Test tracking
    //------------------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    //------------------------------------------------------------------------
    // Helper tasks
    //------------------------------------------------------------------------
    task clear_inputs;
        begin
            branch_valid  <= 1'b0;
            branch_type   <= 6'b0;
            branch_target <= 32'b0;
            branch_cond   <= 32'b0;
            is_uniform    <= 1'b0;
            call_valid    <= 1'b0;
            call_target   <= 32'b0;
            ret_valid     <= 1'b0;
            diverge_mask  <= 32'b0;
        end
    endtask

    task check(
        input [255:0] test_name,
        input         condition
    );
        begin
            test_num = test_num + 1;
            if (condition) begin
                $display("[PASS] Test %0d: %0s", test_num, test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s", test_num, test_name);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task wait_clk(input integer n);
        integer i;
        begin
            for (i = 0; i < n; i = i + 1)
                @(posedge clk);
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Control Flow Unit Testbench");
        $display("============================================================");
        $display("STACK_DEPTH = %0d, NUM_WARPS = %0d", STACK_DEPTH, NUM_WARPS);
        $display("");

        // Initialize all inputs
        rst_n        = 1'b0;
        pc_current   = 32'h0000_0100;
        warp_id      = 2'b00;
        active_mask  = 32'hFFFF_FFFF;
        branch_valid  = 1'b0;
        branch_type   = 6'b0;
        branch_target = 32'b0;
        branch_cond   = 32'b0;
        is_uniform    = 1'b0;
        call_valid    = 1'b0;
        call_target   = 32'b0;
        ret_valid     = 1'b0;
        diverge_mask  = 32'b0;

        //====================================================================
        // Test 1: Reset behavior
        //====================================================================
        $display("--- Test Group 1: Reset Behavior ---");
        #(CLK_PERIOD * 3);
        check("next_pc zeroed after reset",
              next_pc === 32'b0);
        check("next_active_mask is all-ones after reset",
              next_active_mask === 32'hFFFF_FFFF);
        check("pc_valid deasserted after reset",
              pc_valid === 1'b0);
        check("stall deasserted after reset",
              stall === 1'b0);
        check("at_reconverge deasserted after reset",
              at_reconverge === 1'b0);
        check("stack_overflow deasserted after reset",
              stack_overflow === 1'b0);
        check("stack_underflow deasserted after reset",
              stack_underflow === 1'b0);

        // Release reset
        @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        //====================================================================
        // Test 2: Unconditional branch
        //====================================================================
        $display("\n--- Test Group 2: Unconditional Branch ---");
        clear_inputs;
        pc_current    <= 32'h0000_0100;
        warp_id       <= 2'b00;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_UNCONDITIONAL;
        branch_target <= 32'h0000_0200;
        @(posedge clk);
        #1;
        check("Unconditional branch: next_pc = target",
              next_pc === 32'h0000_0200);
        check("Unconditional branch: pc_valid asserted",
              pc_valid === 1'b1);
        check("Unconditional branch: active_mask unchanged",
              next_active_mask === 32'hFFFF_FFFF);
        clear_inputs;
        @(posedge clk);

        // Wait for FSM to return to IDLE
        wait_clk(2);

        //====================================================================
        // Test 3: Uniform branch
        //====================================================================
        $display("\n--- Test Group 3: Uniform Branch ---");
        clear_inputs;
        pc_current    <= 32'h0000_0300;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_UNIFORM;
        branch_target <= 32'h0000_0400;
        is_uniform    <= 1'b1;
        @(posedge clk);
        #1;
        check("Uniform branch: next_pc = target",
              next_pc === 32'h0000_0400);
        check("Uniform branch: pc_valid asserted",
              pc_valid === 1'b1);
        check("Uniform branch: active_mask unchanged",
              next_active_mask === 32'hFFFF_FFFF);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 4: BR_IF_TRUE, all threads take branch (no divergence)
        //====================================================================
        $display("\n--- Test Group 4: BR_IF_TRUE, No Divergence ---");
        clear_inputs;
        pc_current    <= 32'h0000_0500;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_IF_TRUE;
        branch_target <= 32'h0000_0600;
        diverge_mask  <= 32'hFFFF_FFFF;  // all threads want to take branch
        is_uniform    <= 1'b0;
        @(posedge clk);
        #1;
        check("BR_IF_TRUE no diverge: next_pc = target",
              next_pc === 32'h0000_0600);
        check("BR_IF_TRUE no diverge: pc_valid asserted",
              pc_valid === 1'b1);
        check("BR_IF_TRUE no diverge: all threads active",
              next_active_mask === 32'hFFFF_FFFF);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 5: BR_IF_TRUE with divergence
        //====================================================================
        $display("\n--- Test Group 5: BR_IF_TRUE with Divergence ---");
        clear_inputs;
        pc_current    <= 32'h0000_0700;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_IF_TRUE;
        branch_target <= 32'h0000_0800;
        diverge_mask  <= 32'h0000_00FF;  // lower 8 threads take, upper 24 don't
        is_uniform    <= 1'b0;
        @(posedge clk);
        #1;
        check("BR_IF_TRUE diverge: next_pc = target (taken path first)",
              next_pc === 32'h0000_0800);
        check("BR_IF_TRUE diverge: pc_valid asserted",
              pc_valid === 1'b1);
        check("BR_IF_TRUE diverge: active_mask = taken threads",
              next_active_mask === 32'h0000_00FF);
        check("BR_IF_TRUE diverge: reconverge_pc = pc+4",
              reconverge_pc === 32'h0000_0704);
        clear_inputs;
        @(posedge clk);

        // Wait for FSM to go through DIVERGE -> IDLE
        wait_clk(2);

        //====================================================================
        // Test 6: BR_IF_FALSE with divergence
        //====================================================================
        $display("\n--- Test Group 6: BR_IF_FALSE with Divergence ---");
        clear_inputs;
        pc_current    <= 32'h0000_0900;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_IF_FALSE;
        branch_target <= 32'h0000_0A00;
        diverge_mask  <= 32'hFFFF_0000;  // upper 16 set in diverge mask
        is_uniform    <= 1'b0;
        @(posedge clk);
        #1;
        // BR_IF_FALSE sends not_taken_mask path to target
        check("BR_IF_FALSE diverge: next_pc = target",
              next_pc === 32'h0000_0A00);
        check("BR_IF_FALSE diverge: pc_valid asserted",
              pc_valid === 1'b1);
        check("BR_IF_FALSE diverge: active_mask = not-taken threads",
              next_active_mask === 32'h0000_FFFF);
        check("BR_IF_FALSE diverge: reconverge_pc = pc+4",
              reconverge_pc === 32'h0000_0904);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 7: Indirect branch
        //====================================================================
        $display("\n--- Test Group 7: Indirect Branch ---");
        clear_inputs;
        pc_current    <= 32'h0000_0B00;
        active_mask   <= 32'hFFFF_FFFF;
        branch_valid  <= 1'b1;
        branch_type   <= BR_INDIRECT;
        branch_target <= 32'hDEAD_BEE0;
        @(posedge clk);
        #1;
        check("Indirect branch: next_pc = register target",
              next_pc === 32'hDEAD_BEE0);
        check("Indirect branch: pc_valid asserted",
              pc_valid === 1'b1);
        check("Indirect branch: active_mask unchanged",
              next_active_mask === 32'hFFFF_FFFF);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 8: Call/Return
        //====================================================================
        $display("\n--- Test Group 8: Call/Return ---");
        // Issue a CALL
        clear_inputs;
        pc_current   <= 32'h0000_1000;
        warp_id      <= 2'b00;
        active_mask  <= 32'hFFFF_FFFF;
        call_valid   <= 1'b1;
        call_target  <= 32'h0000_2000;
        @(posedge clk);
        #1;
        check("Call: next_pc = call_target",
              next_pc === 32'h0000_2000);
        check("Call: pc_valid asserted",
              pc_valid === 1'b1);
        check("Call: active_mask unchanged",
              next_active_mask === 32'hFFFF_FFFF);
        clear_inputs;
        @(posedge clk);

        // Wait for FSM to return to IDLE (CALL_RET -> IDLE)
        wait_clk(2);

        // Issue a RET
        clear_inputs;
        pc_current  <= 32'h0000_2050;
        warp_id     <= 2'b00;
        active_mask <= 32'hFFFF_FFFF;
        ret_valid   <= 1'b1;
        @(posedge clk);
        #1;
        check("Return: next_pc = return address (call_pc + 4)",
              next_pc === 32'h0000_1004);
        check("Return: pc_valid asserted",
              pc_valid === 1'b1);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 9: Nested calls (verify stack depth)
        //====================================================================
        $display("\n--- Test Group 9: Nested Calls ---");
        // Push multiple return addresses
        clear_inputs;
        warp_id     <= 2'b01;  // use warp 1 for clean stack
        active_mask <= 32'hFFFF_FFFF;

        // Call 1: PC=0x100 -> target 0x1000
        pc_current  <= 32'h0000_0100;
        call_valid  <= 1'b1;
        call_target <= 32'h0000_1000;
        @(posedge clk);
        #1;
        check("Nested call 1: next_pc = 0x1000",
              next_pc === 32'h0000_1000);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        // Call 2: PC=0x1000 -> target 0x2000
        pc_current  <= 32'h0000_1000;
        call_valid  <= 1'b1;
        call_target <= 32'h0000_2000;
        @(posedge clk);
        #1;
        check("Nested call 2: next_pc = 0x2000",
              next_pc === 32'h0000_2000);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        // Call 3: PC=0x2000 -> target 0x3000
        pc_current  <= 32'h0000_2000;
        call_valid  <= 1'b1;
        call_target <= 32'h0000_3000;
        @(posedge clk);
        #1;
        check("Nested call 3: next_pc = 0x3000",
              next_pc === 32'h0000_3000);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        // Return 3: should go back to 0x2004
        pc_current <= 32'h0000_3050;
        ret_valid  <= 1'b1;
        @(posedge clk);
        #1;
        check("Nested ret 3: next_pc = 0x2004",
              next_pc === 32'h0000_2004);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        // Return 2: should go back to 0x1004
        pc_current <= 32'h0000_2004;
        ret_valid  <= 1'b1;
        @(posedge clk);
        #1;
        check("Nested ret 2: next_pc = 0x1004",
              next_pc === 32'h0000_1004);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        // Return 1: should go back to 0x104
        pc_current <= 32'h0000_1004;
        ret_valid  <= 1'b1;
        @(posedge clk);
        #1;
        check("Nested ret 1: next_pc = 0x0104",
              next_pc === 32'h0000_0104);
        clear_inputs;
        @(posedge clk);
        wait_clk(1);

        //====================================================================
        // Test 10: Return underflow detection
        //====================================================================
        $display("\n--- Test Group 10: Return Underflow ---");
        // Warp 1 stack is now empty after all returns above
        clear_inputs;
        warp_id    <= 2'b01;
        pc_current <= 32'h0000_5000;
        ret_valid  <= 1'b1;
        @(posedge clk);
        #1;
        check("Return underflow: stack_underflow asserted",
              stack_underflow === 1'b1);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Test 11: Reconvergence after divergence
        //====================================================================
        $display("\n--- Test Group 11: Reconvergence ---");
        // Use warp 2 for clean state
        clear_inputs;
        warp_id     <= 2'b10;
        active_mask <= 32'hFFFF_FFFF;

        // Create divergence: BR_IF_TRUE at PC=0x3000 with half threads
        pc_current    <= 32'h0000_3000;
        branch_valid  <= 1'b1;
        branch_type   <= BR_IF_TRUE;
        branch_target <= 32'h0000_3100;
        diverge_mask  <= 32'h0000_FFFF;  // lower 16 take branch
        is_uniform    <= 1'b0;
        @(posedge clk);
        #1;
        check("Diverge setup: next_pc = taken target 0x3100",
              next_pc === 32'h0000_3100);
        check("Diverge setup: active_mask = taken mask",
              next_active_mask === 32'h0000_FFFF);
        clear_inputs;
        @(posedge clk);

        // Wait for FSM DIVERGE -> IDLE
        wait_clk(2);

        // Now simulate arriving at the reconvergence PC (pc_current + 4 = 0x3004)
        clear_inputs;
        warp_id     <= 2'b10;
        pc_current  <= 32'h0000_3004;  // This matches the div_stack_pc
        active_mask <= 32'h0000_FFFF;  // Only taken threads still active
        @(posedge clk);
        #1;
        check("Reconvergence: at_reconverge asserted",
              at_reconverge === 1'b1);
        check("Reconvergence: active_mask restored to not-taken threads",
              next_active_mask === 32'hFFFF_0000);
        check("Reconvergence: pc_valid asserted",
              pc_valid === 1'b1);
        clear_inputs;
        @(posedge clk);

        wait_clk(2);

        //====================================================================
        // Summary
        //====================================================================
        $display("");
        $display("============================================================");
        $display("Test Summary: %0d PASSED, %0d FAILED out of %0d",
                 pass_count, fail_count, test_num);
        $display("============================================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("");

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * 2000);
        $display("[TIMEOUT] Simulation exceeded maximum cycles");
        $finish;
    end

    //------------------------------------------------------------------------
    // VCD dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_control_flow_unit.vcd");
        $dumpvars(0, tb_control_flow_unit);
    end

endmodule
