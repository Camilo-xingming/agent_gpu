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

    localparam [31:0] INST_ALU    = 32'h00000033;  // opcode 0110011
    localparam [31:0] INST_FMA    = 32'h02000033;  // funct7=0000001 + opcode 0110011
    localparam [31:0] INST_LOAD   = 32'h00000003;  // opcode 0000011
    localparam [31:0] INST_STORE  = 32'h00000023;  // opcode 0100011
    localparam [31:0] INST_BRANCH = 32'h00000063;  // opcode 1100011

    dual_issue_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .INST_WIDTH(INST_WIDTH)
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

    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;
    reg [31:0] stall_before;

    task check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
            end else begin
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
            warp_inst       = {NUM_WARPS*INST_WIDTH{1'b0}};
            warp_valid      = {NUM_WARPS{1'b0}};
            warp_ready      = {NUM_WARPS{1'b0}};
            warp_rd         = {NUM_WARPS*5{1'b0}};
            warp_rs1        = {NUM_WARPS*5{1'b0}};
            warp_rs2        = {NUM_WARPS*5{1'b0}};
            warp_writes_reg = {NUM_WARPS{1'b0}};
            warp_reads_mem  = {NUM_WARPS{1'b0}};
            warp_writes_mem = {NUM_WARPS{1'b0}};

            alu_ready       = 1'b1;
            fma_ready       = 1'b1;
            mem_ready       = 1'b1;
            branch_ready    = 1'b1;
        end
    endtask

    task set_warp;
        input integer w;
        input [31:0] inst;
        input        vld;
        input        rdy;
        input        wr_reg;
        input        rd_mem;
        input        wr_mem;
        input [4:0]  rd;
        input [4:0]  rs1;
        input [4:0]  rs2;
        begin
            warp_inst[w*INST_WIDTH +: INST_WIDTH] = inst;
            warp_valid[w] = vld;
            warp_ready[w] = rdy;
            warp_writes_reg[w] = wr_reg;
            warp_reads_mem[w] = rd_mem;
            warp_writes_mem[w] = wr_mem;
            warp_rd[w*5 +: 5] = rd;
            warp_rs1[w*5 +: 5] = rs1;
            warp_rs2[w*5 +: 5] = rs2;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        pass_count = 0;
        fail_count = 0;

        clear_inputs();

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        // 1) Dual-issue eligibility: independent ALU + MEM should pair.
        clear_inputs();
        set_warp(0, INST_ALU,  1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 5, 1, 2);
        set_warp(1, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 6, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0 && issue0_unit == 3'd0, "Case1: slot0 should issue warp0 ALU");
        check(issue1_valid && issue1_warp_id == 1 && issue1_unit == 3'd2, "Case1: slot1 should issue warp1 MEM");
        check(warp_consumed == 4'b0011, "Case1: both warps should be consumed");
        check(stat_dual_issue == 32'd1 && stat_single_issue == 32'd0, "Case1: dual issue counter should increment");

        // 2) Structural hazard: same unit conflict (ALU + ALU) => single issue only.
        clear_inputs();
        set_warp(0, INST_ALU, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 7, 1, 2);
        set_warp(1, INST_ALU, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 8, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case2: slot0 should pick warp0");
        check(!issue1_valid, "Case2: slot1 should be blocked by structural conflict");

        // 3) RAW hazard: warp0 writes R10, warp1 reads R10 => block pairing.
        clear_inputs();
        set_warp(0, INST_ALU,  1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 10, 1, 2);
        set_warp(1, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 11, 10, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case3: slot0 should pick warp0");
        check(!issue1_valid, "Case3: slot1 should be blocked by RAW hazard");

        // 4) WAW hazard: both write same RD => block pairing.
        clear_inputs();
        set_warp(0, INST_ALU, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 12, 1, 2);
        set_warp(1, INST_FMA, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 12, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case4: slot0 should pick warp0");
        check(!issue1_valid, "Case4: slot1 should be blocked by WAW hazard");

        // 5) Memory dependency (scoreboard-like conservative MEM conflict) blocks pairing.
        // Use different units to isolate dependency logic from unit conflict.
        clear_inputs();
        set_warp(0, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 13, 1, 2);
        set_warp(1, INST_ALU,  1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 14, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case5: slot0 should pick warp0");
        check(!issue1_valid, "Case5: slot1 should be blocked by MEM dependency");

        // 6) Stall/resource case: warp0 needs MEM but MEM unit not ready; warp1 ALU should issue.
        clear_inputs();
        mem_ready = 1'b0;
        set_warp(0, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 15, 1, 2);
        set_warp(1, INST_ALU,  1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 16, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 1 && issue0_unit == 3'd0, "Case6: scheduler should skip blocked warp0 and issue warp1");
        check(!issue1_valid, "Case6: no second issue expected");

        // 7) Barrier/sync proxy: branch unit unavailable should skip branch-like warp.
        clear_inputs();
        branch_ready = 1'b0;
        set_warp(0, INST_BRANCH, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0);
        set_warp(1, INST_ALU,    1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 19, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 1 && issue0_unit == 3'd0, "Case7: branch-unready warp should be skipped");
        check(!issue1_valid, "Case7: no second issue expected when first ready warp is skipped");

        // 8a) Predication proxy: warp_ready=0 models predicated-off instruction.
        clear_inputs();
        set_warp(0, INST_ALU,    1'b1, 1'b0, 1'b1, 1'b0, 1'b0, 17, 1, 2);
        set_warp(1, INST_BRANCH, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0);
        step();
        check(issue0_valid && issue0_warp_id == 1, "Case8a: predicated-off warp0 should not issue; warp1 should issue");
        check(!warp_consumed[0] && warp_consumed[1], "Case8a: only warp1 consumed");

        // 8b) Replay-like behavior: unstall warp0 next cycle and ensure it issues.
        clear_inputs();
        set_warp(0, INST_ALU, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 18, 1, 2);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case8b: previously blocked warp0 should issue when ready");
        check(warp_consumed[0], "Case8b: warp0 consumed after ready");

        // 9) WAR hazard: warp0 reads R21, warp1 writes R21 => block pairing.
        clear_inputs();
        set_warp(0, INST_ALU, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 23, 21, 2);
        set_warp(1, INST_FMA, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 21, 3, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case9: slot0 should pick warp0");
        check(!issue1_valid, "Case9: slot1 should be blocked by WAR hazard");

        // 10a) Back-to-back dependency: producer issues first, dependent blocked.
        clear_inputs();
        set_warp(0, INST_ALU,  1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 24, 1, 2);
        set_warp(1, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 25, 24, 4);
        step();
        check(issue0_valid && issue0_warp_id == 0, "Case10a: producer warp should issue first");
        check(!issue1_valid, "Case10a: dependent warp should be blocked in same cycle");

        // 10b) Next cycle, dependent warp can issue once producer is gone.
        clear_inputs();
        set_warp(1, INST_LOAD, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 25, 24, 4);
        step();
        check(issue0_valid && issue0_warp_id == 1 && issue0_unit == 3'd2, "Case10b: dependent warp should issue in follow-up cycle");
        check(warp_consumed[1], "Case10b: warp1 consumed in follow-up cycle");

        // 11) Full stall cycle: no eligible warp => stall counter increments.
        stall_before = stat_stall_cycles;
        clear_inputs();
        step();
        check(!issue0_valid && !issue1_valid, "Case11: no issue expected in full stall cycle");

        // Final counter checks across all cases.
        check(stat_dual_issue == 32'd1, "Final: stat_dual_issue should be 1");
        check(stat_single_issue == 32'd11, "Final: stat_single_issue should be 11");
        check(stat_stall_cycles == stall_before + 1, "Final: stat_stall_cycles should increment by 1 in Case11");

        if (fail_count == 0) begin
            $display("========================================");
            $display("[PASS] tb_dual_issue_scheduler: %0d checks passed", pass_count);
            $display("========================================");
            $finish;
        end else begin
            $display("========================================");
            $display("[FAIL] tb_dual_issue_scheduler: %0d passed, %0d failed", pass_count, fail_count);
            $display("========================================");
            $fatal(1, "tb_dual_issue_scheduler failed");
        end
    end
endmodule
