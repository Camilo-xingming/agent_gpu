`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_simt_stack_controller;
    localparam NUM_WARPS   = 2;
    localparam STACK_DEPTH = 4;
    localparam NUM_THREADS = 32;
    localparam ADDR_WIDTH  = 32;
    localparam WARP_W      = $clog2(NUM_WARPS);

    reg                     clk;
    reg                     rst_n;

    reg  [NUM_WARPS-1:0]    warp_valid;
    reg  [ADDR_WIDTH*NUM_WARPS-1:0]  warp_pc;
    reg  [NUM_THREADS*NUM_WARPS-1:0] warp_active_mask;

    reg                     branch_valid;
    reg  [WARP_W-1:0]       branch_warp_id;
    reg  [ADDR_WIDTH-1:0]   branch_target;
    reg  [ADDR_WIDTH-1:0]   branch_fallthrough;
    reg  [NUM_THREADS-1:0]  branch_taken_mask;
    reg                     branch_uniform;

    wire [NUM_WARPS-1:0]    warp_diverged;
    wire [NUM_WARPS-1:0]    warp_at_barrier;
    wire                    pc_update_valid;
    wire [WARP_W-1:0]       pc_update_warp;
    wire [ADDR_WIDTH-1:0]   pc_update_value;
    wire [NUM_THREADS-1:0]  mask_update_value;

    integer pass_count;
    integer fail_count;
    integer test_num;

    simt_stack_controller #(
        .NUM_WARPS(NUM_WARPS),
        .STACK_DEPTH(STACK_DEPTH),
        .NUM_THREADS(NUM_THREADS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_pc(warp_pc),
        .warp_active_mask(warp_active_mask),
        .branch_valid(branch_valid),
        .branch_warp_id(branch_warp_id),
        .branch_target(branch_target),
        .branch_fallthrough(branch_fallthrough),
        .branch_taken_mask(branch_taken_mask),
        .branch_uniform(branch_uniform),
        .warp_diverged(warp_diverged),
        .warp_at_barrier(warp_at_barrier),
        .pc_update_valid(pc_update_valid),
        .pc_update_warp(pc_update_warp),
        .pc_update_value(pc_update_value),
        .mask_update_value(mask_update_value)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task clear_branch;
        begin
            branch_valid = 1'b0;
            branch_warp_id = {WARP_W{1'b0}};
            branch_target = {ADDR_WIDTH{1'b0}};
            branch_fallthrough = {ADDR_WIDTH{1'b0}};
            branch_taken_mask = {NUM_THREADS{1'b0}};
            branch_uniform = 1'b0;
        end
    endtask

    task set_warp_pc;
        input integer wid;
        input [ADDR_WIDTH-1:0] pc;
        begin
            warp_pc[wid*ADDR_WIDTH +: ADDR_WIDTH] = pc;
        end
    endtask

    task set_warp_mask;
        input integer wid;
        input [NUM_THREADS-1:0] mask;
        begin
            warp_active_mask[wid*NUM_THREADS +: NUM_THREADS] = mask;
        end
    endtask

    task apply_update_if_valid;
        begin
            if (pc_update_valid) begin
                set_warp_pc(pc_update_warp, pc_update_value);
                set_warp_mask(pc_update_warp, mask_update_value);
            end
        end
    endtask

    task check;
        input [255:0] msg;
        input cond;
        begin
            test_num = test_num + 1;
            if (cond) begin
                pass_count = pass_count + 1;
                $display("PASS: %s", msg);
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: %s", msg);
            end
        end
    endtask

    task branch_and_check;
        input [WARP_W-1:0] wid;
        input [ADDR_WIDTH-1:0] target;
        input [ADDR_WIDTH-1:0] fallthrough;
        input [NUM_THREADS-1:0] taken_mask;
        input [NUM_THREADS-1:0] expected_not_taken;
        input [NUM_THREADS-1:0] expected_active_before;
        begin
            branch_warp_id = wid;
            branch_target = target;
            branch_fallthrough = fallthrough;
            branch_taken_mask = taken_mask;
            branch_uniform = 1'b0;
            branch_valid = 1'b1;

            @(posedge clk);
            #1;

            check("branch update valid", pc_update_valid === 1'b1);
            check("branch update warp id", pc_update_warp === wid);
            check("branch update target pc", pc_update_value === target);
            check("branch update active mask", mask_update_value === taken_mask);
            if (wid == {WARP_W{1'b0}}) begin
                check("branch saved not-taken mask",
                      dut.gen_stacks[0].u_stack.top_mask === expected_not_taken);
                check("lane integrity across divergence",
                      (mask_update_value | dut.gen_stacks[0].u_stack.top_mask) === expected_active_before);
            end else begin
                check("branch saved not-taken mask",
                      dut.gen_stacks[1].u_stack.top_mask === expected_not_taken);
                check("lane integrity across divergence",
                      (mask_update_value | dut.gen_stacks[1].u_stack.top_mask) === expected_active_before);
            end

            branch_valid = 1'b0;
            clear_branch();
            apply_update_if_valid();
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        warp_valid = {NUM_WARPS{1'b1}};
        warp_pc = {NUM_WARPS*ADDR_WIDTH{1'b0}};
        warp_active_mask = {NUM_WARPS*NUM_THREADS{1'b1}};
        clear_branch();

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        #1;

        // Initialize warp contexts
        set_warp_pc(0, 32'h0000_3000);
        set_warp_mask(0, 32'hFFFF_FFFF);
        set_warp_pc(1, 32'h0000_5000);
        set_warp_mask(1, 32'hFFFF_FFFF);

        //====================================================================
        // Test 1: Reset status
        //====================================================================
        check("reset: no pending pc update", pc_update_valid === 1'b0);
        check("reset: warp0 not diverged", warp_diverged[0] === 1'b0);
        check("reset: warp1 not diverged", warp_diverged[1] === 1'b0);

        //====================================================================
        // Test 2: Warp0 outer divergence
        //====================================================================
        branch_and_check(1'b0,
                         32'h0000_3100,
                         32'h0000_3004,
                         32'h0000_FFFF,
                         32'hFFFF_0000,
                         32'hFFFF_FFFF);
        check("warp0 diverged after outer branch", warp_diverged[0] === 1'b1);

        //====================================================================
        // Test 3: Warp0 nested divergence
        //====================================================================
        set_warp_pc(0, 32'h0000_3100);
        set_warp_mask(0, 32'h0000_FFFF);
        branch_and_check(1'b0,
                         32'h0000_3200,
                         32'h0000_3104,
                         32'h0000_000F,
                         32'h0000_FFF0,
                         32'h0000_FFFF);
        check("warp0 nested diverged remains high", warp_diverged[0] === 1'b1);

        //====================================================================
        // Test 4: Warp0 inner reconvergence boundary
        //====================================================================
        set_warp_pc(0, 32'h0000_3104);
        set_warp_mask(0, 32'h0000_000F);
        @(posedge clk);
        #1;
        check("inner reconv: pc_update_valid", pc_update_valid === 1'b1);
        check("inner reconv: warp0 selected", pc_update_warp === 1'b0);
        check("inner reconv: pc set to saved path", pc_update_value === 32'h0000_3104);
        check("inner reconv: mask switches to not-taken", mask_update_value === 32'h0000_FFF0);
        apply_update_if_valid();
        @(posedge clk);
        #1;

        //====================================================================
        // Test 5: Warp1 divergence independent from warp0 state
        //====================================================================
        // Move warp0 away from reconvergence boundary to avoid arbitration tie.
        set_warp_pc(0, 32'h0000_3208);
        set_warp_mask(0, 32'h0000_FFF0);
        branch_and_check(1'b1,
                         32'h0000_5100,
                         32'h0000_5004,
                         32'h00FF_00FF,
                         32'hFF00_FF00,
                         32'hFFFF_FFFF);
        check("warp1 diverged high", warp_diverged[1] === 1'b1);
        check("warp0 context not clobbered",
              warp_pc[0*ADDR_WIDTH +: ADDR_WIDTH] == 32'h0000_3208);

        //====================================================================
        // Test 6: Warp1 reconvergence boundary behavior
        //====================================================================
        set_warp_pc(1, 32'h0000_5004);
        set_warp_mask(1, 32'h00FF_00FF);
        @(posedge clk);
        #1;
        check("warp1 reconv: pc_update_valid", pc_update_valid === 1'b1);
        check("warp1 reconv: warp id", pc_update_warp === 1'b1);
        check("warp1 reconv: pc to saved path", pc_update_value === 32'h0000_5004);
        check("warp1 reconv: mask to not-taken", mask_update_value === 32'hFF00_FF00);
        apply_update_if_valid();
        @(posedge clk);
        #1;

        //====================================================================
        // Test 7: Stack depth limit / overflow visibility
        //====================================================================
        dut.gen_stacks[1].u_stack.stack_ptr[0] = STACK_DEPTH;
        #1;
        check("overflow flag asserted at depth limit", dut.stack_overflow[1] === 1'b1);
        dut.gen_stacks[1].u_stack.stack_ptr[0] = 0;
        #1;

        $display("============================================================");
        $display("tb_simt_stack_controller Summary: %0d PASSED, %0d FAILED, total %0d",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        if (fail_count == 0)
            $finish;
        else
            $fatal(1, "tb_simt_stack_controller failed");
    end

    initial begin
        #200000;
        $fatal(1, "tb_simt_stack_controller timeout");
    end

    initial begin
        $dumpfile("tb_simt_stack_controller.vcd");
        $dumpvars(0, tb_simt_stack_controller);
    end
endmodule
