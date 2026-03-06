`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_reconvergence_stack;
    localparam NUM_WARPS   = 4;
    localparam STACK_DEPTH = 4;
    localparam NUM_THREADS = 32;
    localparam ADDR_WIDTH  = 32;
    localparam WARP_W      = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1;

    reg                     clk;
    reg                     rst_n;

    reg  [WARP_W-1:0]       warp_id;
    reg                     branch_valid;
    reg  [ADDR_WIDTH-1:0]   branch_target;
    reg  [ADDR_WIDTH-1:0]   fallthrough_pc;
    reg  [NUM_THREADS-1:0]  branch_taken_mask;
    reg  [NUM_THREADS-1:0]  active_mask;
    reg                     is_uniform;
    reg  [ADDR_WIDTH-1:0]   current_pc;

    wire [ADDR_WIDTH-1:0]   next_pc;
    wire [NUM_THREADS-1:0]  next_active_mask;
    wire                    pc_valid;
    wire                    at_reconvergence;
    wire                    diverged;
    wire                    stack_overflow;
    wire                    stack_empty;
    wire [$clog2(STACK_DEPTH):0] stack_depth;

    integer pass_count;
    integer fail_count;
    integer test_num;
    integer warp_iter;
    integer depth_iter;

    reg [31:0] lane_active_pattern;
    reg [31:0] lane_waiting_pattern;
    reg [31:0] expected_rpc;

    reconvergence_stack #(
        .NUM_WARPS(NUM_WARPS),
        .STACK_DEPTH(STACK_DEPTH),
        .NUM_THREADS(NUM_THREADS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_id(warp_id),
        .branch_valid(branch_valid),
        .branch_target(branch_target),
        .fallthrough_pc(fallthrough_pc),
        .branch_taken_mask(branch_taken_mask),
        .active_mask(active_mask),
        .is_uniform(is_uniform),
        .current_pc(current_pc),
        .next_pc(next_pc),
        .next_active_mask(next_active_mask),
        .pc_valid(pc_valid),
        .at_reconvergence(at_reconvergence),
        .diverged(diverged),
        .stack_overflow(stack_overflow),
        .stack_empty(stack_empty),
        .stack_depth(stack_depth)
    );

    function [31:0] warp_active_mask_pattern;
        input [WARP_W-1:0] wid;
        begin
            case (wid)
                2'd0: warp_active_mask_pattern = 32'h0000_00F0;
                2'd1: warp_active_mask_pattern = 32'h0000_0F00;
                2'd2: warp_active_mask_pattern = 32'h0000_F000;
                default: warp_active_mask_pattern = 32'h000F_0000;
            endcase
        end
    endfunction

    function [31:0] warp_waiting_mask_pattern;
        input [WARP_W-1:0] wid;
        begin
            case (wid)
                2'd0: warp_waiting_mask_pattern = 32'h0000_0300;
                2'd1: warp_waiting_mask_pattern = 32'h0000_3000;
                2'd2: warp_waiting_mask_pattern = 32'h0003_0000;
                default: warp_waiting_mask_pattern = 32'h0030_0000;
            endcase
        end
    endfunction

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task clear_inputs;
        begin
            branch_valid = 1'b0;
            branch_target = 0;
            fallthrough_pc = 0;
            branch_taken_mask = 0;
            active_mask = 32'hFFFF_FFFF;
            is_uniform = 1'b0;
            current_pc = 0;
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

    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        warp_id = 0;
        clear_inputs();

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 1: Reset status
        //====================================================================
        warp_id = 0;
        #1;
        check("reset: stack empty", stack_empty === 1'b1);
        check("reset: stack depth 0", stack_depth === 0);
        check("reset: diverged low", diverged === 1'b0);

        //====================================================================
        // Test 2: Divergence push (outer)
        //====================================================================
        clear_inputs();
        warp_id = 0;
        current_pc = 32'h0000_3000;
        branch_valid = 1'b1;
        branch_target = 32'h0000_3100;
        fallthrough_pc = 32'h0000_3004;
        branch_taken_mask = 32'h0000_FFFF;
        active_mask = 32'hFFFF_FFFF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("outer push: pc_valid", pc_valid === 1'b1);
        check("outer push: next_pc=branch_target", next_pc === 32'h0000_3100);
        check("outer push: next_mask=taken", next_active_mask === 32'h0000_FFFF);
        check("outer push: stack_depth=1", stack_depth === 1);
        check("outer push: diverged high", diverged === 1'b1);

        //====================================================================
        // Test 3: Nested divergence push
        //====================================================================
        clear_inputs();
        warp_id = 0;
        current_pc = 32'h0000_3100;
        branch_valid = 1'b1;
        branch_target = 32'h0000_3200;
        fallthrough_pc = 32'h0000_3104;
        branch_taken_mask = 32'h0000_000F;
        active_mask = 32'h0000_FFFF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested push: pc_valid", pc_valid === 1'b1);
        check("nested push: next_pc=0x3200", next_pc === 32'h0000_3200);
        check("nested push: next_mask=0x0000000F", next_active_mask === 32'h0000_000F);
        check("nested push: stack_depth=2", stack_depth === 2);

        //====================================================================
        // Test 4: Reconvergence point behavior in diverged state
        // (current RTL keeps stack entry and switches to saved path mask)
        //====================================================================
        clear_inputs();
        warp_id = 0;
        current_pc = 32'h0000_3104;   // top RPC from nested push
        active_mask = 32'h0000_000F;

        @(posedge clk);
        #1;
        check("at rpc (diverged): pc_valid", pc_valid === 1'b1);
        check("at rpc (diverged): next_pc=top_pc", next_pc === 32'h0000_3104);
        check("at rpc (diverged): next_mask=not_taken", next_active_mask === 32'h0000_FFF0);
        check("at rpc (diverged): stack depth unchanged", stack_depth === 2);

        //====================================================================
        // Test 5: Forced pop path in ST_NORMAL (unit-level pop verification)
        //====================================================================
        clear_inputs();
        warp_id = 1;
        active_mask = 32'h0000_00F0;
        current_pc = 32'h0000_4004;

        // Seed internal state to exercise ST_NORMAL pop path deterministically
        dut.stack_ptr[1] = 1;
        dut.state[1] = 3'd0; // ST_NORMAL
        // Entry = {rpc, mask, pc}
        dut.stack_mem[1][0] = {32'h0000_4004, 32'h0000_0F00, 32'h0000_4004};

        @(posedge clk);
        #1;
        check("forced pop: at_reconvergence asserted", at_reconvergence === 1'b1);
        check("forced pop: pc_valid asserted", pc_valid === 1'b1);
        check("forced pop: stack depth decremented", stack_depth === 0);
        check("forced pop: merged active mask", next_active_mask === 32'h0000_0FF0);

        //====================================================================
        // Test 6: Overflow edge case
        //====================================================================
        clear_inputs();
        warp_id = 1;
        dut.stack_ptr[1] = STACK_DEPTH;
        #1;
        check("overflow: stack_overflow asserted", stack_overflow === 1'b1);

        //====================================================================
        // Test 7: Underflow edge case (empty stack, reconverge-like PC)
        //====================================================================
        clear_inputs();
        warp_id = 1;
        dut.stack_ptr[1] = 0;
        dut.state[1] = 3'd0;
        current_pc = 32'h0000_4004;
        active_mask = 32'h0000_00F0;

        @(posedge clk);
        #1;
        check("underflow edge: stack_empty stays asserted", stack_empty === 1'b1);
        check("underflow edge: no reconvergence pulse", at_reconvergence === 1'b0);
        check("underflow edge: no pc_valid pulse", pc_valid === 1'b0);

        //====================================================================
        // Test 8: All-warps reconvergence PC + lane-mask integrity
        //====================================================================
        for (warp_iter = 0; warp_iter < NUM_WARPS; warp_iter = warp_iter + 1) begin
            clear_inputs();
            warp_id = warp_iter[WARP_W-1:0];
            lane_active_pattern = warp_active_mask_pattern(warp_id);
            lane_waiting_pattern = warp_waiting_mask_pattern(warp_id);
            expected_rpc = 32'h0000_5004 + (warp_iter * 32'h100);
            current_pc = expected_rpc;
            active_mask = lane_active_pattern;

            dut.stack_ptr[warp_iter] = 1;
            dut.state[warp_iter] = 3'd0; // ST_NORMAL
            dut.stack_mem[warp_iter][0] = {expected_rpc, lane_waiting_pattern, expected_rpc};

            @(posedge clk);
            #1;
            check("all-warps reconv: at_reconvergence asserted", at_reconvergence === 1'b1);
            check("all-warps reconv: next_pc == expected RPC", next_pc === expected_rpc);
            check("all-warps reconv: merged mask preserved",
                  next_active_mask === (lane_waiting_pattern | lane_active_pattern));
            check("all-warps reconv: stack popped", stack_depth === 0);
        end

        //====================================================================
        // Test 9: Dynamic depth fill to limit + overflow guard
        //====================================================================
        dut.stack_ptr[3] = 0;
        dut.state[3] = 3'd0;

        for (depth_iter = 0; depth_iter < STACK_DEPTH; depth_iter = depth_iter + 1) begin
            clear_inputs();
            warp_id = 2'd3;
            current_pc = 32'h0000_7000 + (depth_iter * 32'h0100);
            branch_valid = 1'b1;
            branch_target = current_pc + 32'h0000_0040;
            fallthrough_pc = current_pc + 32'h0000_0004;
            branch_taken_mask = depth_iter[0] ? 32'hAAAA_AAAA : 32'h5555_5555;
            active_mask = 32'hFFFF_FFFF;
            is_uniform = 1'b0;

            @(posedge clk);
            #1;
            check("depth fill: pc_valid asserted", pc_valid === 1'b1);
            check("depth fill: stack depth increments", stack_depth === (depth_iter + 1));
        end

        #1;
        check("depth fill: stack_overflow asserted at full depth", stack_overflow === 1'b1);

        clear_inputs();
        warp_id = 2'd3;
        current_pc = 32'h0000_7F00;
        branch_valid = 1'b1;
        branch_target = 32'h0000_7F40;
        fallthrough_pc = 32'h0000_7F04;
        branch_taken_mask = 32'h0F0F_F0F0;
        active_mask = 32'hFFFF_FFFF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("depth guard: stack depth clamped at limit", stack_depth === STACK_DEPTH);
        check("depth guard: stack_overflow remains asserted", stack_overflow === 1'b1);
        check("depth guard: no extra pc_valid pulse", pc_valid === 1'b0);

        //====================================================================
        // Test 10: 4-level nested divergence + overflow block
        //====================================================================
        clear_inputs();
        warp_id = 2'd2;
        dut.stack_ptr[2] = 0;
        dut.state[2] = 3'd0;

        // Level 1
        current_pc = 32'h0000_8000;
        branch_valid = 1'b1;
        branch_target = 32'h0000_8100;
        fallthrough_pc = 32'h0000_8004;
        branch_taken_mask = 32'h00FF_00FF;
        active_mask = 32'hFFFF_FFFF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested-4 lvl1: pc_valid", pc_valid === 1'b1);
        check("nested-4 lvl1: stack_depth=1", stack_depth === 1);
        check("nested-4 lvl1: next_mask=taken", next_active_mask === 32'h00FF_00FF);
        check("nested-4 lvl1: pushed not-taken entry",
              dut.stack_mem[2][0] === {32'h0000_8004, 32'hFF00_FF00, 32'h0000_8004});

        // Level 2
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8100;
        branch_valid = 1'b1;
        branch_target = 32'h0000_8200;
        fallthrough_pc = 32'h0000_8104;
        branch_taken_mask = 32'h000F_000F;
        active_mask = 32'h00FF_00FF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested-4 lvl2: pc_valid", pc_valid === 1'b1);
        check("nested-4 lvl2: stack_depth=2", stack_depth === 2);
        check("nested-4 lvl2: next_mask=taken", next_active_mask === 32'h000F_000F);
        check("nested-4 lvl2: pushed not-taken entry",
              dut.stack_mem[2][1] === {32'h0000_8104, 32'h00F0_00F0, 32'h0000_8104});

        // Level 3
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8200;
        branch_valid = 1'b1;
        branch_target = 32'h0000_8300;
        fallthrough_pc = 32'h0000_8204;
        branch_taken_mask = 32'h0003_0003;
        active_mask = 32'h000F_000F;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested-4 lvl3: pc_valid", pc_valid === 1'b1);
        check("nested-4 lvl3: stack_depth=3", stack_depth === 3);
        check("nested-4 lvl3: next_mask=taken", next_active_mask === 32'h0003_0003);
        check("nested-4 lvl3: pushed not-taken entry",
              dut.stack_mem[2][2] === {32'h0000_8204, 32'h000C_000C, 32'h0000_8204});

        // Level 4 (fill to hardware depth)
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8300;
        branch_valid = 1'b1;
        branch_target = 32'h0000_8400;
        fallthrough_pc = 32'h0000_8304;
        branch_taken_mask = 32'h0001_0001;
        active_mask = 32'h0003_0003;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested-4 lvl4: pc_valid", pc_valid === 1'b1);
        check("nested-4 lvl4: stack_depth=4", stack_depth === STACK_DEPTH);
        check("nested-4 lvl4: stack_overflow asserted at limit", stack_overflow === 1'b1);
        check("nested-4 lvl4: pushed not-taken entry",
              dut.stack_mem[2][3] === {32'h0000_8304, 32'h0002_0002, 32'h0000_8304});

        // One more divergent branch attempt must be blocked
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8400;
        branch_valid = 1'b1;
        branch_target = 32'h0000_8500;
        fallthrough_pc = 32'h0000_8404;
        branch_taken_mask = 32'h0000_0001;
        active_mask = 32'h0001_0001;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("nested-4 overflow: no extra push", stack_depth === STACK_DEPTH);
        check("nested-4 overflow: pc_valid suppressed", pc_valid === 1'b0);
        check("nested-4 overflow: overflow stays asserted", stack_overflow === 1'b1);

        //====================================================================
        // Test 11: LIFO reconvergence ordering + early-exit-like recovery
        //====================================================================
        // First hit top RPC in diverged mode: should switch to saved path mask
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8304;
        active_mask = 32'h0001_0001;

        @(posedge clk);
        #1;
        check("lifo step1: pc_valid at top rpc", pc_valid === 1'b1);
        check("lifo step1: next_pc selects top entry", next_pc === 32'h0000_8304);
        check("lifo step1: next_mask selects top waiting mask", next_active_mask === 32'h0002_0002);
        check("lifo step1: depth unchanged before pop", stack_depth === STACK_DEPTH);

        // Simulate early-exit completion of saved path by clearing top waiting mask
        dut.stack_mem[2][3] = {32'h0000_8304, 32'h0000_0000, 32'h0000_8304};

        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8304;
        active_mask = 32'h0002_0002;

        @(posedge clk);
        #1;
        check("lifo step2: reconvergence pulse when waiting mask drained", at_reconvergence === 1'b1);
        check("lifo step2: popped one level", stack_depth === 3);
        check("lifo step2: returns to ST_NORMAL", dut.state[2] === 3'd0);

        // Continue LIFO pops in ST_NORMAL and verify mask merges level-by-level
        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8204;
        active_mask = 32'h0002_0002;

        @(posedge clk);
        #1;
        check("lifo step3: pop level3", stack_depth === 2);
        check("lifo step3: merge mask level3", next_active_mask === 32'h000E_000E);

        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8104;
        active_mask = 32'h000E_000E;

        @(posedge clk);
        #1;
        check("lifo step4: pop level2", stack_depth === 1);
        check("lifo step4: merge mask level2", next_active_mask === 32'h00FE_00FE);

        clear_inputs();
        warp_id = 2'd2;
        current_pc = 32'h0000_8004;
        active_mask = 32'h00FE_00FE;

        @(posedge clk);
        #1;
        check("lifo step5: pop level1", stack_depth === 0);
        check("lifo step5: final merge with outer waiting mask", next_active_mask === 32'hFFFE_FFFE);
        check("lifo step5: diverged cleared", diverged === 1'b0);

        //====================================================================
        // Test 12: Sparse vote-like masks + branch-in-diverged interaction
        //====================================================================
        clear_inputs();
        warp_id = 2'd3;
        dut.stack_ptr[3] = 0;
        dut.state[3] = 3'd0;

        // Non-contiguous branch mask (vote-like) to create divergence
        current_pc = 32'h0000_9000;
        branch_valid = 1'b1;
        branch_target = 32'h0000_9100;
        fallthrough_pc = 32'h0000_9004;
        branch_taken_mask = 32'h8421_8421;
        active_mask = 32'hFFFF_FFFF;
        is_uniform = 1'b0;

        @(posedge clk);
        #1;
        check("sparse vote: divergence issued", pc_valid === 1'b1);
        check("sparse vote: next mask keeps sparse taken lanes", next_active_mask === 32'h8421_8421);
        check("sparse vote: waiting mask stored exactly",
              dut.stack_mem[3][0] === {32'h0000_9004, 32'h7BDE_7BDE, 32'h0000_9004});

        // Uniform branch inside diverged mode must keep stack stable
        clear_inputs();
        warp_id = 2'd3;
        current_pc = 32'h0000_9100;
        branch_valid = 1'b1;
        branch_target = 32'h0000_9200;
        fallthrough_pc = 32'h0000_9104;
        branch_taken_mask = 32'hFFFF_FFFF;
        active_mask = 32'h8421_8421;
        is_uniform = 1'b1;

        @(posedge clk);
        #1;
        check("sparse vote: uniform branch in diverged state allowed", pc_valid === 1'b1);
        check("sparse vote: uniform branch keeps active mask", next_active_mask === 32'h8421_8421);
        check("sparse vote: stack depth unchanged", stack_depth === 1);

        // Returning to RPC should restore saved sparse complement mask
        clear_inputs();
        warp_id = 2'd3;
        current_pc = 32'h0000_9004;
        active_mask = 32'h8421_8421;

        @(posedge clk);
        #1;
        check("sparse vote: rpc emits saved waiting mask", next_active_mask === 32'h7BDE_7BDE);
        check("sparse vote: depth unchanged before saved-path completion", stack_depth === 1);

        $display("============================================================");
        $display("tb_reconvergence_stack Summary: %0d PASSED, %0d FAILED, total %0d",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        if (fail_count == 0)
            $finish;
        else
            $fatal(1, "tb_reconvergence_stack failed");
    end

    initial begin
        #200000;
        $fatal(1, "tb_reconvergence_stack timeout");
    end

    initial begin
        $dumpfile("tb_reconvergence_stack.vcd");
        $dumpvars(0, tb_reconvergence_stack);
    end
endmodule
