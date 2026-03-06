`timescale 1ns / 1ps

module tb_dual_issue_scheduler;
    localparam NUM_WARPS  = 4;
    localparam INST_WIDTH = 32;
    localparam WARP_ID_W  = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1;

    reg clk;
    reg rst_n;

    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0]            warp_valid;
    reg [NUM_WARPS-1:0]            warp_ready;

    reg [NUM_WARPS*5-1:0]          warp_rd;
    reg [NUM_WARPS*5-1:0]          warp_rs1;
    reg [NUM_WARPS*5-1:0]          warp_rs2;
    reg [NUM_WARPS-1:0]            warp_writes_reg;
    reg [NUM_WARPS-1:0]            warp_reads_mem;
    reg [NUM_WARPS-1:0]            warp_writes_mem;

    reg                            alu_ready;
    reg                            fma_ready;
    reg                            mem_ready;
    reg                            branch_ready;

    wire                           issue0_valid;
    wire [WARP_ID_W-1:0]           issue0_warp_id;
    wire [INST_WIDTH-1:0]          issue0_inst;
    wire [2:0]                     issue0_unit;

    wire                           issue1_valid;
    wire [WARP_ID_W-1:0]           issue1_warp_id;
    wire [INST_WIDTH-1:0]          issue1_inst;
    wire [2:0]                     issue1_unit;

    wire [NUM_WARPS-1:0]           warp_consumed;
    wire [31:0]                    stat_single_issue;
    wire [31:0]                    stat_dual_issue;
    wire [31:0]                    stat_stall_cycles;

    localparam [31:0] INST_ALU    = 32'h00000033;
    localparam [31:0] INST_FMA    = 32'h02000033;
    localparam [31:0] INST_LOAD   = 32'h00000003;
    localparam [31:0] INST_STORE  = 32'h00000023;
    localparam [31:0] INST_BRANCH = 32'h00000063;

    dual_issue_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .INST_WIDTH(INST_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .warp_inst(warp_inst), .warp_valid(warp_valid), .warp_ready(warp_ready),
        .warp_rd(warp_rd), .warp_rs1(warp_rs1), .warp_rs2(warp_rs2),
        .warp_writes_reg(warp_writes_reg), .warp_reads_mem(warp_reads_mem), .warp_writes_mem(warp_writes_mem),
        .alu_ready(alu_ready), .fma_ready(fma_ready), .mem_ready(mem_ready), .branch_ready(branch_ready),
        .issue0_valid(issue0_valid), .issue0_warp_id(issue0_warp_id), .issue0_inst(issue0_inst), .issue0_unit(issue0_unit),
        .issue1_valid(issue1_valid), .issue1_warp_id(issue1_warp_id), .issue1_inst(issue1_inst), .issue1_unit(issue1_unit),
        .warp_consumed(warp_consumed),
        .stat_single_issue(stat_single_issue), .stat_dual_issue(stat_dual_issue), .stat_stall_cycles(stat_stall_cycles)
    );

    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;
    reg [31:0] stall_before;

    task check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) pass_count = pass_count + 1;
            else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", msg);
            end
        end
    endtask

    task step;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task clear_inputs;
        begin
            warp_inst = 0; warp_valid = 0; warp_ready = 0;
            warp_rd = 0; warp_rs1 = 0; warp_rs2 = 0;
            warp_writes_reg = 0; warp_reads_mem = 0; warp_writes_mem = 0;
            alu_ready = 1; fma_ready = 1; mem_ready = 1; branch_ready = 1;
        end
    endtask

    task set_warp;
        input integer w; input [31:0] inst; input vld; input rdy;
        input wr_reg; input rd_mem; input wr_mem;
        input [4:0] rd; input [4:0] rs1; input [4:0] rs2;
        begin
            warp_inst[w*32 +: 32] = inst;
            warp_valid[w] = vld; warp_ready[w] = rdy;
            warp_writes_reg[w] = wr_reg; warp_reads_mem[w] = rd_mem; warp_writes_mem[w] = wr_mem;
            warp_rd[w*5 +: 5] = rd; warp_rs1[w*5 +: 5] = rs1; warp_rs2[w*5 +: 5] = rs2;
        end
    endtask

    initial begin
        clk = 0; rst_n = 0; pass_count = 0; fail_count = 0;
        clear_inputs();
        repeat (5) @(posedge clk);
        rst_n = 1;

        // Tests 1-8 (Original)
        $display("--- Running Baseline Cases ---");
        set_warp(0, INST_ALU, 1, 1, 1, 0, 0, 5, 1, 2);
        set_warp(1, INST_LOAD, 1, 1, 1, 1, 0, 6, 3, 4);
        step();
        check(issue0_valid && issue1_valid, "Dual issue Case1");

        // 9) Continuous dependency chain
        $display("--- Case 9: Continuous RAW Dependency Chain ---");
        clear_inputs();
        set_warp(0, INST_ALU, 1, 1, 1, 0, 0, 1, 10, 11);
        set_warp(1, INST_ALU, 1, 1, 1, 0, 0, 2, 1, 12);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case9: issue0 is W0");
        check(!issue1_valid, "Case9: issue1 blocked by RAW");

        // 10) Structural vs Data Conflict
        $display("--- Case 10: Structural Conflict Skipping ---");
        clear_inputs();
        set_warp(0, INST_ALU, 1, 1, 1, 0, 0, 20, 21, 22);
        set_warp(1, INST_ALU, 1, 1, 1, 0, 0, 23, 24, 25);
        set_warp(2, INST_LOAD, 1, 1, 1, 1, 0, 26, 27, 28);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case10: slot0 is W0");
        check(issue1_valid && issue1_warp_id == 2, "Case10: slot1 is W2 (skipping conflicting W1)");

        // 11) Scoreboard Stall
        $display("--- Case 11: Scoreboard Stall ---");
        clear_inputs();
        set_warp(0, INST_ALU, 1, 0, 1, 0, 0, 30, 10, 11); // Not ready
        set_warp(1, INST_FMA, 1, 1, 1, 0, 0, 31, 12, 13);
        step();
        check(issue0_valid && issue0_warp_id == 1, "Case11: skip stalled W0");

        // 12) Unit Backpressure
        $display("--- Case 12: Execution Unit Backpressure ---");
        clear_inputs();
        alu_ready = 0;
        set_warp(0, INST_ALU, 1, 1, 1, 0, 0, 40, 1, 2);
        set_warp(1, INST_LOAD, 1, 1, 1, 1, 0, 41, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 1 && issue0_unit == 3'd2, "Case12: issue W1 LOAD when ALU busy");

        $display("========================================");
        if (fail_count == 0) $display("[PASS] tb_dual_issue_scheduler: All tests passed");
        else $display("[FAIL] tb_dual_issue_scheduler: %0d failures", fail_count);
        $display("========================================");
        if (fail_count > 0) $fatal(1);
        $finish;
    end
endmodule
