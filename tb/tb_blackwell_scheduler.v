`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_blackwell_scheduler;
    localparam NUM_WARPS = 4;
    localparam NUM_SCHEDULERS = 2;
    localparam INST_WIDTH = 32;
    localparam WARP_W = $clog2(NUM_WARPS);

    localparam PIPE_COMPUTE0 = 3'd0;
    localparam PIPE_MEMORY   = 3'd3;

    reg clk;
    reg rst_n;

    reg [NUM_WARPS-1:0] warp_valid;
    reg [NUM_WARPS-1:0] warp_ready;
    reg [NUM_WARPS-1:0] warp_diverged;
    reg [NUM_WARPS-1:0] warp_at_barrier;

    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0] warp_inst_valid;
    wire [NUM_WARPS-1:0] warp_inst_consume;

    reg [NUM_WARPS*5-1:0] warp_rd;
    reg [NUM_WARPS*5-1:0] warp_rs1;
    reg [NUM_WARPS*5-1:0] warp_rs2;
    reg [NUM_WARPS*5-1:0] warp_rs3;
    reg [NUM_WARPS-1:0] warp_reads_rs3;

    reg [NUM_WARPS-1:0] warp_is_compute;
    reg [NUM_WARPS-1:0] warp_is_tensor;
    reg [NUM_WARPS-1:0] tensor_push_locked;
    reg [NUM_WARPS-1:0] warp_is_memory;
    reg [NUM_WARPS-1:0] warp_is_branch;

    reg [NUM_WARPS-1:0] warp_is_alu;
    reg [NUM_WARPS-1:0] warp_is_mul;
    reg [NUM_WARPS-1:0] warp_is_fp32;
    reg [NUM_WARPS-1:0] warp_is_fp16;
    reg [NUM_WARPS-1:0] warp_is_sfu;
    reg [NUM_WARPS-1:0] warp_is_shfl;
    reg [NUM_WARPS-1:0] warp_is_video;
    reg [NUM_WARPS-1:0] warp_writes_reg;

    reg [NUM_WARPS-1:0] warp_is_tcgen05;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_mma;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_alloc;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_ld;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_st;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_commit;
    reg [NUM_WARPS-1:0] warp_is_tcgen05_wait;

    reg [NUM_WARPS-1:0] tmem_alloc_valid;
    reg tmem_pipe_ready;

    reg async_mma_complete;
    reg [WARP_W-1:0] async_mma_warp_id;
    reg [3:0] async_mma_op_id;

    reg compute_pipe0_ready;
    reg compute_pipe1_ready;
    reg tensor_pipe_ready;
    reg memory_pipe_ready;
    reg branch_unit_ready;

    wire [NUM_SCHEDULERS-1:0] issue_valid;
    wire [NUM_SCHEDULERS*WARP_W-1:0] issue_warp_id;
    wire [NUM_SCHEDULERS*INST_WIDTH-1:0] issue_inst;
    wire [NUM_SCHEDULERS*3-1:0] issue_pipe;
    wire [NUM_SCHEDULERS-1:0] issue_is_async_mma;
    wire [NUM_SCHEDULERS*4-1:0] issue_async_mma_id;

    reg pipeline_stall;
    reg fu_conflict_sb_clr_valid;
    reg [WARP_W-1:0] fu_conflict_sb_clr_warp;
    reg [4:0] fu_conflict_sb_clr_rd;
    reg pipeline_stall_slot1;
    reg [NUM_SCHEDULERS-1:0] dispatch_fire;

    reg branch_flush_sb_clr0_valid;
    reg [WARP_W-1:0] branch_flush_sb_clr0_warp;
    reg [4:0] branch_flush_sb_clr0_rd;
    reg branch_flush_sb_clr1_valid;
    reg [WARP_W-1:0] branch_flush_sb_clr1_warp;
    reg [4:0] branch_flush_sb_clr1_rd;

    reg tensor_sb_set_valid;
    reg [WARP_W-1:0] tensor_sb_set_warp;
    reg [4:0] tensor_sb_set_rd;
    reg tensor_issue_conflict;

    reg wgmma_sb_clr_valid;
    reg [WARP_W-1:0] wgmma_sb_clr_warp;
    reg [4:0] wgmma_sb_clr_rd;

    reg replay_sb_clr_valid;
    reg [WARP_W-1:0] replay_sb_clr_warp;
    reg [4:0] replay_sb_clr_rd;

    reg wb_valid;
    reg [WARP_W-1:0] wb_warp_id;
    reg [4:0] wb_rd;

    wire [31:0] stat_cycles;
    wire [31:0] stat_single_issue;
    wire [31:0] stat_dual_issue;
    wire [31:0] stat_stalls;
    wire [31:0] stat_async_mma_issued;
    wire [31:0] stat_async_mma_completed;
    wire [31:0] stat_tcgen05_issued;
    wire perf_sched_stall_ifetch;
    wire [NUM_WARPS*4-1:0] issue_seq_out;
    wire [NUM_WARPS*32-1:0] scoreboard_out;

    wire [WARP_W-1:0] issue_warp0 = issue_warp_id[WARP_W-1:0];
    wire [WARP_W-1:0] issue_warp1 = issue_warp_id[WARP_W +: WARP_W];
    wire [2:0] issue_pipe0 = issue_pipe[2:0];
    wire [2:0] issue_pipe1 = issue_pipe[5:3];
    wire [31:0] scoreboard_w0 = scoreboard_out[31:0];

    integer pass_count;
    integer fail_count;
    reg [31:0] stall_before;
    reg [WARP_W-1:0] first_pick;
    reg [WARP_W-1:0] second_pick;

    blackwell_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_SCHEDULERS(NUM_SCHEDULERS),
        .INST_WIDTH(INST_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_diverged(warp_diverged),
        .warp_at_barrier(warp_at_barrier),
        .warp_inst(warp_inst),
        .warp_inst_valid(warp_inst_valid),
        .warp_inst_consume(warp_inst_consume),
        .warp_rd(warp_rd),
        .warp_rs1(warp_rs1),
        .warp_rs2(warp_rs2),
        .warp_rs3(warp_rs3),
        .warp_reads_rs3(warp_reads_rs3),
        .warp_is_compute(warp_is_compute),
        .warp_is_tensor(warp_is_tensor),
        .tensor_push_locked(tensor_push_locked),
        .warp_is_memory(warp_is_memory),
        .warp_is_branch(warp_is_branch),
        .warp_is_alu(warp_is_alu),
        .warp_is_mul(warp_is_mul),
        .warp_is_fp32(warp_is_fp32),
        .warp_is_fp16(warp_is_fp16),
        .warp_is_sfu(warp_is_sfu),
        .warp_is_shfl(warp_is_shfl),
        .warp_is_video(warp_is_video),
        .warp_writes_reg(warp_writes_reg),
        .warp_is_tcgen05(warp_is_tcgen05),
        .warp_is_tcgen05_mma(warp_is_tcgen05_mma),
        .warp_is_tcgen05_alloc(warp_is_tcgen05_alloc),
        .warp_is_tcgen05_ld(warp_is_tcgen05_ld),
        .warp_is_tcgen05_st(warp_is_tcgen05_st),
        .warp_is_tcgen05_commit(warp_is_tcgen05_commit),
        .warp_is_tcgen05_wait(warp_is_tcgen05_wait),
        .tmem_alloc_valid(tmem_alloc_valid),
        .tmem_pipe_ready(tmem_pipe_ready),
        .async_mma_complete(async_mma_complete),
        .async_mma_warp_id(async_mma_warp_id),
        .async_mma_op_id(async_mma_op_id),
        .compute_pipe0_ready(compute_pipe0_ready),
        .compute_pipe1_ready(compute_pipe1_ready),
        .tensor_pipe_ready(tensor_pipe_ready),
        .memory_pipe_ready(memory_pipe_ready),
        .branch_unit_ready(branch_unit_ready),
        .issue_valid(issue_valid),
        .issue_warp_id(issue_warp_id),
        .issue_inst(issue_inst),
        .issue_pipe(issue_pipe),
        .issue_is_async_mma(issue_is_async_mma),
        .issue_async_mma_id(issue_async_mma_id),
        .pipeline_stall(pipeline_stall),
        .fu_conflict_sb_clr_valid(fu_conflict_sb_clr_valid),
        .fu_conflict_sb_clr_warp(fu_conflict_sb_clr_warp),
        .fu_conflict_sb_clr_rd(fu_conflict_sb_clr_rd),
        .pipeline_stall_slot1(pipeline_stall_slot1),
        .dispatch_fire(dispatch_fire),
        .branch_flush_sb_clr0_valid(branch_flush_sb_clr0_valid),
        .branch_flush_sb_clr0_warp(branch_flush_sb_clr0_warp),
        .branch_flush_sb_clr0_rd(branch_flush_sb_clr0_rd),
        .branch_flush_sb_clr1_valid(branch_flush_sb_clr1_valid),
        .branch_flush_sb_clr1_warp(branch_flush_sb_clr1_warp),
        .branch_flush_sb_clr1_rd(branch_flush_sb_clr1_rd),
        .tensor_sb_set_valid(tensor_sb_set_valid),
        .tensor_sb_set_warp(tensor_sb_set_warp),
        .tensor_sb_set_rd(tensor_sb_set_rd),
        .tensor_issue_conflict(tensor_issue_conflict),
        .wgmma_sb_clr_valid(wgmma_sb_clr_valid),
        .wgmma_sb_clr_warp(wgmma_sb_clr_warp),
        .wgmma_sb_clr_rd(wgmma_sb_clr_rd),
        .replay_sb_clr_valid(replay_sb_clr_valid),
        .replay_sb_clr_warp(replay_sb_clr_warp),
        .replay_sb_clr_rd(replay_sb_clr_rd),
        .wb_valid(wb_valid),
        .wb_warp_id(wb_warp_id),
        .wb_rd(wb_rd),
        .stat_cycles(stat_cycles),
        .stat_single_issue(stat_single_issue),
        .stat_dual_issue(stat_dual_issue),
        .stat_stalls(stat_stalls),
        .stat_async_mma_issued(stat_async_mma_issued),
        .stat_async_mma_completed(stat_async_mma_completed),
        .stat_tcgen05_issued(stat_tcgen05_issued),
        .perf_sched_stall_ifetch(perf_sched_stall_ifetch),
        .issue_seq_out(issue_seq_out),
        .scoreboard_out(scoreboard_out)
    );

    always #5 clk = ~clk;

    task clear_inputs;
        begin
            warp_valid = 0;
            warp_ready = 0;
            warp_diverged = 0;
            warp_at_barrier = 0;
            warp_inst = 0;
            warp_inst_valid = 0;
            warp_rd = 0;
            warp_rs1 = 0;
            warp_rs2 = 0;
            warp_rs3 = 0;
            warp_reads_rs3 = 0;
            warp_is_compute = 0;
            warp_is_tensor = 0;
            tensor_push_locked = 0;
            warp_is_memory = 0;
            warp_is_branch = 0;
            warp_is_alu = 0;
            warp_is_mul = 0;
            warp_is_fp32 = 0;
            warp_is_fp16 = 0;
            warp_is_sfu = 0;
            warp_is_shfl = 0;
            warp_is_video = 0;
            warp_writes_reg = 0;
            warp_is_tcgen05 = 0;
            warp_is_tcgen05_mma = 0;
            warp_is_tcgen05_alloc = 0;
            warp_is_tcgen05_ld = 0;
            warp_is_tcgen05_st = 0;
            warp_is_tcgen05_commit = 0;
            warp_is_tcgen05_wait = 0;
            tmem_alloc_valid = 0;
            tmem_pipe_ready = 1'b1;
            async_mma_complete = 1'b0;
            async_mma_warp_id = 0;
            async_mma_op_id = 0;
            compute_pipe0_ready = 1'b1;
            compute_pipe1_ready = 1'b1;
            tensor_pipe_ready = 1'b1;
            memory_pipe_ready = 1'b1;
            branch_unit_ready = 1'b1;
            pipeline_stall = 1'b0;
            fu_conflict_sb_clr_valid = 1'b0;
            fu_conflict_sb_clr_warp = 0;
            fu_conflict_sb_clr_rd = 0;
            pipeline_stall_slot1 = 1'b0;
            dispatch_fire = {NUM_SCHEDULERS{1'b1}};
            branch_flush_sb_clr0_valid = 1'b0;
            branch_flush_sb_clr0_warp = 0;
            branch_flush_sb_clr0_rd = 0;
            branch_flush_sb_clr1_valid = 1'b0;
            branch_flush_sb_clr1_warp = 0;
            branch_flush_sb_clr1_rd = 0;
            tensor_sb_set_valid = 1'b0;
            tensor_sb_set_warp = 0;
            tensor_sb_set_rd = 0;
            tensor_issue_conflict = 1'b0;
            wgmma_sb_clr_valid = 1'b0;
            wgmma_sb_clr_warp = 0;
            wgmma_sb_clr_rd = 0;
            replay_sb_clr_valid = 1'b0;
            replay_sb_clr_warp = 0;
            replay_sb_clr_rd = 0;
            wb_valid = 1'b0;
            wb_warp_id = 0;
            wb_rd = 0;
        end
    endtask

    task set_warp_compute;
        input integer w;
        input [4:0] rd;
        input [4:0] rs1;
        input [4:0] rs2;
        input wr_reg;
        begin
            warp_valid[w] = 1'b1;
            warp_ready[w] = 1'b1;
            warp_inst_valid[w] = 1'b1;
            warp_is_compute[w] = 1'b1;
            warp_is_alu[w] = 1'b1;
            warp_writes_reg[w] = wr_reg;
            warp_inst[w*INST_WIDTH +: INST_WIDTH] = 32'h0000_0013;
            warp_rd[w*5 +: 5] = rd;
            warp_rs1[w*5 +: 5] = rs1;
            warp_rs2[w*5 +: 5] = rs2;
        end
    endtask

    task set_warp_memory;
        input integer w;
        begin
            warp_valid[w] = 1'b1;
            warp_ready[w] = 1'b1;
            warp_inst_valid[w] = 1'b1;
            warp_is_memory[w] = 1'b1;
            warp_inst[w*INST_WIDTH +: INST_WIDTH] = 32'h0000_2003;
            warp_rs1[w*5 +: 5] = 5'd1;
            warp_rs2[w*5 +: 5] = 5'd2;
            warp_rd[w*5 +: 5] = 5'd3;
        end
    endtask

    task step;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: %0s", msg);
            end
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        pass_count = 0;
        fail_count = 0;
        stall_before = 0;

        clear_inputs;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Case 1: round-robin fairness within scheduler 0 (warps 0 and 2)
        clear_inputs;
        set_warp_compute(0, 5'd10, 5'd1, 5'd2, 1'b0);
        set_warp_compute(2, 5'd11, 5'd3, 5'd4, 1'b0);
        step;
        first_pick = issue_warp0;
        check(issue_valid[0] && (first_pick == 0 || first_pick == 2),
              "Case1.1 scheduler0 should pick from {warp0, warp2}");
        step;
        second_pick = issue_warp0;
        check(issue_valid[0] && (second_pick == 0 || second_pick == 2),
              "Case1.2 scheduler0 should continue selecting eligible assigned warp");

        // Case 2: slot arbitration (FU conflict ALU vs ALU suppresses slot1)
        clear_inputs;
        set_warp_compute(0, 5'd12, 5'd1, 5'd2, 1'b0);
        set_warp_compute(1, 5'd13, 5'd3, 5'd4, 1'b0);
        step;
        check(issue_valid[0], "Case2 slot0 should issue");
        check(!issue_valid[1], "Case2 slot1 should be suppressed by FU conflict");
        check(!warp_inst_consume[1], "Case2 slot1 warp should not be consumed");

        // Case 3: memory swap arbitration (slot1 memory moves to slot0)
        clear_inputs;
        set_warp_compute(0, 5'd14, 5'd1, 5'd2, 1'b0);
        set_warp_memory(1);
        step;
        check(issue_valid[0] && issue_warp0 == 1 && issue_pipe0 == PIPE_MEMORY,
              "Case3 slot0 should carry memory warp after swap");
        check(issue_valid[1] && issue_warp1 == 0 && issue_pipe1 == PIPE_COMPUTE0,
              "Case3 slot1 should carry original compute warp after swap");

        // Case 4: scoreboard set -> hazard stall -> writeback clear
        clear_inputs;
        set_warp_compute(0, 5'd5, 5'd1, 5'd2, 1'b1);
        step;
        check(scoreboard_w0[5], "Case4.1 scoreboard bit R5 should set after issue");

        clear_inputs;
        set_warp_compute(0, 5'd0, 5'd5, 5'd2, 1'b0);
        step;
        check(!issue_valid[0], "Case4.2 RAW hazard on busy R5 should block issue");

        wb_valid = 1'b1;
        wb_warp_id = 0;
        wb_rd = 5'd5;
        step;
        wb_valid = 1'b0;

        step;
        check(!scoreboard_w0[5], "Case4.3 scoreboard bit R5 should clear on writeback");
        check(issue_valid[0], "Case4.4 issue should resume after writeback clear");

        // Case 5: full scoreboard edge blocks issue
        clear_inputs;
        dut.scoreboard[0] = 32'hFFFF_FFFF;
        set_warp_compute(0, 5'd0, 5'd1, 5'd2, 1'b0);
        step;
        check(!issue_valid[0], "Case5 full scoreboard should block issue");
        dut.scoreboard[0] = 32'h0;

        // Case 6: all warps stalled edge increments stall counter
        clear_inputs;
        warp_valid = 4'b1111;
        warp_ready = 4'b0000;
        warp_inst_valid = 4'b1111;
        stall_before = stat_stalls;
        step;
        check(!issue_valid[0] && !issue_valid[1], "Case6 all stalled should produce no issue");
        check(stat_stalls == stall_before + 1, "Case6 stall counter should increment by 1");

        $display("============================================================");
        $display("tb_blackwell_scheduler Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_blackwell_scheduler failed");
        end
    end

endmodule
