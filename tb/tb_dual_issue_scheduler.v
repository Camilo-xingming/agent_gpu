`timescale 1ns/1ps

module tb_dual_issue_scheduler();

    parameter NUM_WARPS    = 4;
    parameter THREADS      = 32;
    parameter INST_WIDTH   = 32;
    parameter DATA_WIDTH   = 32;
    parameter WARP_ID_W    = 2;

    reg clk;
    reg rst_n;

    // Inputs
    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0]            warp_valid;
    reg [NUM_WARPS-1:0]            warp_ready;

    reg [NUM_WARPS*5-1:0]          warp_rd;
    reg [NUM_WARPS*5-1:0]          warp_rs1;
    reg [NUM_WARPS*5-1:0]          warp_rs2;
    reg [NUM_WARPS-1:0]            warp_writes_reg;
    reg [NUM_WARPS-1:0]            warp_reads_mem;
    reg [NUM_WARPS-1:0]            warp_writes_mem;

    reg alu_ready;
    reg fma_ready;
    reg mem_ready;
    reg branch_ready;

    // Outputs
    wire                  issue0_valid;
    wire  [WARP_ID_W-1:0] issue0_warp_id;
    wire  [INST_WIDTH-1:0] issue0_inst;
    wire  [2:0]           issue0_unit;

    wire                  issue1_valid;
    wire  [WARP_ID_W-1:0] issue1_warp_id;
    wire  [INST_WIDTH-1:0] issue1_inst;
    wire  [2:0]           issue1_unit;

    wire  [NUM_WARPS-1:0] warp_consumed;
    wire  [31:0]          stat_single_issue;
    wire  [31:0]          stat_dual_issue;
    wire  [31:0]          stat_stall_cycles;

    // Instantiate DUT
    dual_issue_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .THREADS(THREADS),
        .INST_WIDTH(INST_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .WARP_ID_W(WARP_ID_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_inst(warp_inst),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_rd(warp_rd),
        .warp_rs1(warp_rs1),
        .warp_rs2(warp_rs2),
        .warp_writes_reg(warp_writes_reg),
        .warp_reads_mem(warp_reads_mem),
        .warp_writes_mem(warp_writes_mem),
        .alu_ready(alu_ready),
        .fma_ready(fma_ready),
        .mem_ready(mem_ready),
        .branch_ready(branch_ready),
        .issue0_valid(issue0_valid),
        .issue0_warp_id(issue0_warp_id),
        .issue0_inst(issue0_inst),
        .issue0_unit(issue0_unit),
        .issue1_valid(issue1_valid),
        .issue1_warp_id(issue1_warp_id),
        .issue1_inst(issue1_inst),
        .issue1_unit(issue1_unit),
        .warp_consumed(warp_consumed),
        .stat_single_issue(stat_single_issue),
        .stat_dual_issue(stat_dual_issue),
        .stat_stall_cycles(stat_stall_cycles)
    );

    // Clock gen
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test sequence
    integer errors = 0;

    task check_issue(input integer slot0_valid, input integer slot0_wid, input integer slot1_valid, input integer slot1_wid);
        begin
            #1; // wait for comb logic
            if (issue0_valid !== slot0_valid) begin
                $display("ERROR: Time %0t | Expected issue0_valid=%0d, got %0d", $time, slot0_valid, issue0_valid);
                errors = errors + 1;
            end
            if (slot0_valid && (issue0_warp_id !== slot0_wid)) begin
                $display("ERROR: Time %0t | Expected issue0_warp_id=%0d, got %0d", $time, slot0_wid, issue0_warp_id);
                errors = errors + 1;
            end
            if (issue1_valid !== slot1_valid) begin
                $display("ERROR: Time %0t | Expected issue1_valid=%0d, got %0d", $time, slot1_valid, issue1_valid);
                errors = errors + 1;
            end
            if (slot1_valid && (issue1_warp_id !== slot1_wid)) begin
                $display("ERROR: Time %0t | Expected issue1_warp_id=%0d, got %0d", $time, slot1_wid, issue1_warp_id);
                errors = errors + 1;
            end
        end
    endtask

    // Instruction opcodes
    localparam OPCODE_ALU_R = 7'b0110011;
    localparam OPCODE_LOAD  = 7'b0000011;

    initial begin
        $dumpfile("tb_dual_issue_scheduler.vcd");
        $dumpvars(0, tb_dual_issue_scheduler);

        // Initialize inputs
        rst_n = 0;
        warp_inst = 0;
        warp_valid = 0;
        warp_ready = 0;
        warp_rd = 0; warp_rs1 = 0; warp_rs2 = 0;
        warp_writes_reg = 0; warp_reads_mem = 0; warp_writes_mem = 0;
        alu_ready = 1; fma_ready = 1; mem_ready = 1; branch_ready = 1;

        #20;
        rst_n = 1;
        #10;

        $display("--- Test 1: Independent instructions (Dual issue) ---");
        // Warp 0: ALU (R-type)
        warp_inst[0*32 +: 32] = {7'b0000000, 5'd2, 5'd1, 3'b000, 5'd3, OPCODE_ALU_R};
        warp_valid[0] = 1; warp_ready[0] = 1;
        warp_rd[0*5+:5] = 3; warp_rs1[0*5+:5] = 1; warp_rs2[0*5+:5] = 2; warp_writes_reg[0] = 1;
        
        // Warp 1: LOAD
        warp_inst[1*32 +: 32] = {12'b0, 5'd4, 3'b010, 5'd5, OPCODE_LOAD};
        warp_valid[1] = 1; warp_ready[1] = 1;
        warp_rd[1*5+:5] = 5; warp_rs1[1*5+:5] = 4; warp_reads_mem[1] = 1; warp_writes_reg[1] = 1;

        #10;
        // issue0 should be warp0 (ALU), issue1 should be warp1 (LOAD)
        if (issue0_valid && issue1_valid)
            $display("Test 1 PASS");
        else
            errors = errors + 1;

        #10;
        warp_valid = 0;
        #10;

        $display("--- Test 2: Dependency HAZARD (RAW) ---");
        // Warp 0: ALU writes to R5
        warp_inst[0*32 +: 32] = {7'b0000000, 5'd2, 5'd1, 3'b000, 5'd5, OPCODE_ALU_R};
        warp_valid[0] = 1; warp_ready[0] = 1;
        warp_rd[0*5+:5] = 5; warp_rs1[0*5+:5] = 1; warp_rs2[0*5+:5] = 2; warp_writes_reg[0] = 1;
        
        // Warp 1: ALU reads from R5
        warp_inst[1*32 +: 32] = {7'b0000000, 5'd3, 5'd5, 3'b000, 5'd6, OPCODE_ALU_R};
        warp_valid[1] = 1; warp_ready[1] = 1;
        warp_rd[1*5+:5] = 6; warp_rs1[1*5+:5] = 5; warp_rs2[1*5+:5] = 3; warp_writes_reg[1] = 1;

        #10;
        // Wait, the scheduler checks dependency between warp_a and warp_b.
        // It should single-issue warp 0, warp 1 will not be issued due to dependency.
        if (issue0_valid && !issue1_valid && issue0_warp_id == 0)
            $display("Test 2 PASS");
        else
            errors = errors + 1;

        #10;
        warp_valid = 0;
        #10;

        $display("--- Test 3: Structural Hazard (Both ALU) ---");
        // Both want ALU, only 1 ALU unit.
        warp_inst[0*32 +: 32] = {7'b0000000, 5'd2, 5'd1, 3'b000, 5'd3, OPCODE_ALU_R};
        warp_valid[0] = 1; warp_ready[0] = 1;
        warp_rd[0*5+:5] = 3; warp_rs1[0*5+:5] = 1; warp_rs2[0*5+:5] = 2; warp_writes_reg[0] = 1;

        warp_inst[1*32 +: 32] = {7'b0000000, 5'd5, 5'd4, 3'b000, 5'd6, OPCODE_ALU_R};
        warp_valid[1] = 1; warp_ready[1] = 1;
        warp_rd[1*5+:5] = 6; warp_rs1[1*5+:5] = 4; warp_rs2[1*5+:5] = 5; warp_writes_reg[1] = 1;
        // No dependency, but both ALU
        
        #10;
        if (issue0_valid && !issue1_valid && issue0_warp_id == 0)
            $display("Test 3 PASS");
        else
            errors = errors + 1;

        #10;
        warp_valid = 0;
        #10;

        $display("--- Test 4: Execution unit not ready ---");
        // Warp 0 wants ALU (not ready), Warp 1 wants LOAD (ready)
        warp_inst[0*32 +: 32] = {7'b0000000, 5'd2, 5'd1, 3'b000, 5'd3, OPCODE_ALU_R};
        warp_valid[0] = 1; warp_ready[0] = 1;
        warp_rd[0*5+:5] = 3; warp_rs1[0*5+:5] = 1; warp_rs2[0*5+:5] = 2; warp_writes_reg[0] = 1;
        
        warp_inst[1*32 +: 32] = {12'b0, 5'd4, 3'b010, 5'd5, OPCODE_LOAD};
        warp_valid[1] = 1; warp_ready[1] = 1;
        warp_rd[1*5+:5] = 5; warp_rs1[1*5+:5] = 4; warp_reads_mem[1] = 1; warp_writes_reg[1] = 1;

        alu_ready = 0;
        
        #10;
        // Warp 0 cannot issue, warp 1 should issue on slot 0
        if (issue0_valid && !issue1_valid && issue0_warp_id == 1)
            $display("Test 4 PASS");
        else
            errors = errors + 1;

        alu_ready = 1;
        #10;
        warp_valid = 0;
        #10;

        if (errors == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("FAILED with %0d errors", errors);
        end

        $finish;
    end

endmodule
