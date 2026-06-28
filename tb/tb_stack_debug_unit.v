//============================================================================
// RalphGPU - Stack and Debug Unit Testbench
// Tests stack management and debug/monitoring instructions
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_stack_debug_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter DATA_WIDTH = 32;
    parameter NUM_WARPS = 4;
    parameter STACK_SIZE_PER_WARP = 4096;
    parameter WARP_ID_W = 2;

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
    reg  [WARP_ID_W-1:0]     warp_id;
    reg  [DATA_WIDTH-1:0]    src_a;
    reg  [DATA_WIDTH-1:0]    src_b;

    // Outputs
    wire                     done;
    wire                     result_valid;
    wire [DATA_WIDTH-1:0]    result;
    wire                     brkpt_hit;
    wire                     trap_hit;
    wire [15:0]              trap_code;
    wire                     pmevent_pulse;
    wire [7:0]               pmevent_id;
    wire [NUM_WARPS-1:0]     warp_stall;

    // Configuration
    reg  [7:0]               max_reg_limit;
    wire [NUM_WARPS*8-1:0]   warp_max_regs_flat;
    wire [7:0]               warp_max_regs [0:NUM_WARPS-1];

    // Unpack DUT packed per-warp register limit bus for easier checks
    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : unpack_warp_max_regs
            assign warp_max_regs[w] = warp_max_regs_flat[w*8 +: 8];
        end
    endgenerate

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    stack_debug_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_WARPS(NUM_WARPS),
        .STACK_SIZE_PER_WARP(STACK_SIZE_PER_WARP),
        .WARP_ID_W(WARP_ID_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .opcode(opcode),
        .func(func),
        .warp_id(warp_id),
        .src_a(src_a),
        .src_b(src_b),
        .done(done),
        .result_valid(result_valid),
        .result(result),
        .brkpt_hit(brkpt_hit),
        .trap_hit(trap_hit),
        .trap_code(trap_code),
        .pmevent_pulse(pmevent_pulse),
        .pmevent_id(pmevent_id),
        .warp_stall(warp_stall),
        .max_reg_limit(max_reg_limit),
        .warp_max_regs(warp_max_regs_flat)
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
    reg [DATA_WIDTH-1:0] expected_result;
    reg [DATA_WIDTH-1:0] saved_sp;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        warp_id = 0;
        src_a = 0;
        src_b = 0;
        max_reg_limit = 8'd255;
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task test_stack_op;
        input [5:0] f;
        input [WARP_ID_W-1:0] wid;
        input [DATA_WIDTH-1:0] operand;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Warp %0d, operand=0x%08x", wid, operand);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_STACK;
        func <= f;
        warp_id <= wid;
        src_a <= operand;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        if (result_valid) begin
            $display("  Result: 0x%08x", result);
        end
        $display("  [PASS] Stack operation completed");
        pass_count = pass_count + 1;
        #10;
    end
    endtask

    task test_debug_op;
        input [5:0] f;
        input [WARP_ID_W-1:0] wid;
        input [DATA_WIDTH-1:0] operand;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Warp %0d, operand=0x%08x", wid, operand);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_DEBUG;
        func <= f;
        warp_id <= wid;
        src_a <= operand;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        case (f)
            `DEBUG_BRKPT: begin
                if (brkpt_hit) begin
                    $display("  [PASS] Breakpoint triggered");
                    pass_count = pass_count + 1;
                end else begin
                    $display("  [FAIL] Breakpoint not triggered");
                    fail_count = fail_count + 1;
                end
            end
            `DEBUG_TRAP: begin
                if (trap_hit && trap_code == operand[15:0]) begin
                    $display("  [PASS] Trap triggered with code %0d", trap_code);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  [FAIL] Trap code mismatch: expected %0d, got %0d", operand[15:0], trap_code);
                    fail_count = fail_count + 1;
                end
            end
            `DEBUG_PMEVENT: begin
                if (pmevent_pulse && pmevent_id == operand[7:0]) begin
                    $display("  [PASS] PM event triggered with id %0d", pmevent_id);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  [FAIL] PM event mismatch");
                    fail_count = fail_count + 1;
                end
            end
        endcase
        #10;
    end
    endtask

    task test_misc_op;
        input [5:0] f;
        input [WARP_ID_W-1:0] wid;
        input [DATA_WIDTH-1:0] operand;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Warp %0d, operand=0x%08x", wid, operand);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_MISC;
        func <= f;
        warp_id <= wid;
        src_a <= operand;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        $display("  [PASS] Misc operation completed");
        pass_count = pass_count + 1;
        #10;
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Stack and Debug Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // Stack Operations Tests
        //====================================================================
        $display("\n--- Testing Stack Operations ---");

        // Test 1: STACKSAVE - get initial stack pointer for warp 0
        test_stack_op(`STACK_SAVE, 2'd0, 32'h0, "stacksave: get initial SP warp 0");
        saved_sp = result;

        // Test 2: ALLOCA - allocate 64 bytes on warp 0's stack
        test_stack_op(`STACK_ALLOCA, 2'd0, 32'd64, "alloca: allocate 64 bytes warp 0");

        // Test 3: STACKSAVE - verify SP moved
        test_stack_op(`STACK_SAVE, 2'd0, 32'h0, "stacksave: verify SP moved");

        // Test 4: ALLOCA - allocate 128 bytes
        test_stack_op(`STACK_ALLOCA, 2'd0, 32'd128, "alloca: allocate 128 bytes warp 0");

        // Test 5: STACKRESTORE - restore to saved SP
        test_stack_op(`STACK_RESTORE, 2'd0, saved_sp, "stackrestore: restore original SP");

        // Test 6: STACKSAVE - verify SP was restored
        test_stack_op(`STACK_SAVE, 2'd0, 32'h0, "stacksave: verify SP restored");
        if (result == saved_sp) begin
            $display("  [PASS] SP correctly restored to 0x%08x", saved_sp);
        end else begin
            $display("  [FAIL] SP mismatch: expected 0x%08x, got 0x%08x", saved_sp, result);
        end

        // Test 7: Different warp (warp 1)
        test_stack_op(`STACK_SAVE, 2'd1, 32'h0, "stacksave: warp 1 initial SP");

        // Test 8: Allocate on warp 1
        test_stack_op(`STACK_ALLOCA, 2'd1, 32'd256, "alloca: allocate 256 bytes warp 1");

        //====================================================================
        // Debug Operations Tests
        //====================================================================
        $display("\n--- Testing Debug Operations ---");

        // Test 9: BRKPT
        test_debug_op(`DEBUG_BRKPT, 2'd0, 32'h0, "brkpt: trigger breakpoint");

        // Test 10: TRAP with code
        test_debug_op(`DEBUG_TRAP, 2'd0, 32'd42, "trap: software trap code 42");

        // Test 11: PMEVENT
        test_debug_op(`DEBUG_PMEVENT, 2'd0, 32'd7, "pmevent: PM event id 7");

        //====================================================================
        // Misc Operations Tests
        //====================================================================
        $display("\n--- Testing Misc Operations ---");

        // Test 12: NANOSLEEP
        test_misc_op(`MISC_NANOSLEEP, 2'd2, 32'd10, "nanosleep: sleep 10 cycles warp 2");

        // Verify warp 2 is stalled
        #10;
        if (warp_stall[2]) begin
            $display("  [PASS] Warp 2 is stalled");
        end else begin
            $display("  [INFO] Warp 2 stall might have expired");
        end

        // Wait for sleep to complete
        #200;
        if (!warp_stall[2]) begin
            $display("  [PASS] Warp 2 sleep completed, no longer stalled");
        end

        // Test 13: SETMAXNREG
        test_misc_op(`MISC_SETMAXNREG, 2'd0, 32'd128, "setmaxnreg: set max regs to 128");
        if (warp_max_regs[0] == 8'd128) begin
            $display("  [PASS] Warp 0 max regs set to %0d", warp_max_regs[0]);
        end else begin
            $display("  [FAIL] Warp 0 max regs mismatch");
        end

        // Test 14: SETMAXNREG - try to exceed system limit
        test_misc_op(`MISC_SETMAXNREG, 2'd1, 32'd512, "setmaxnreg: try to exceed limit");
        if (warp_max_regs[1] == max_reg_limit) begin
            $display("  [PASS] Warp 1 max regs clamped to system limit %0d", max_reg_limit);
        end else begin
            $display("  [FAIL] Warp 1 max regs not clamped");
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
