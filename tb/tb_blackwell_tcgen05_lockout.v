`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_blackwell_tcgen05_lockout;
    localparam NUM_WARPS = 4;
    localparam NUM_SCHEDULERS = 2;

    reg clk;
    reg rst_n;

    reg [NUM_WARPS-1:0] warp_valid;
    reg [NUM_WARPS-1:0] warp_ready;
    reg [NUM_WARPS-1:0] warp_diverged;
    reg [NUM_WARPS-1:0] warp_at_barrier;
    reg [31:0] warp_inst [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] warp_inst_valid;
    wire [NUM_WARPS-1:0] warp_inst_consume;

    reg [4:0] warp_rd [0:NUM_WARPS-1];
    reg [4:0] warp_rs1 [0:NUM_WARPS-1];
    reg [4:0] warp_rs2 [0:NUM_WARPS-1];
    reg [4:0] warp_rs3 [0:NUM_WARPS-1];

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
    reg [$clog2(NUM_WARPS)-1:0] async_mma_warp_id;
    reg [3:0] async_mma_op_id;

    reg compute_pipe0_ready;
    reg compute_pipe1_ready;
    reg tensor_pipe_ready;
    reg memory_pipe_ready;
    reg branch_unit_ready;

    wire [NUM_SCHEDULERS-1:0] issue_valid;
    wire [$clog2(NUM_WARPS)-1:0] issue_warp_id [0:NUM_SCHEDULERS-1];
    wire [31:0] issue_inst [0:NUM_SCHEDULERS-1];
    wire [2:0] issue_pipe [0:NUM_SCHEDULERS-1];
    wire [NUM_SCHEDULERS-1:0] issue_is_async_mma;
    wire [3:0] issue_async_mma_id [0:NUM_SCHEDULERS-1];

    reg pipeline_stall;
    reg fu_conflict_sb_clr_valid;
    reg [$clog2(NUM_WARPS)-1:0] fu_conflict_sb_clr_warp;
    reg [4:0] fu_conflict_sb_clr_rd;
    reg pipeline_stall_slot1;
    reg tensor_sb_set_valid;
    reg [$clog2(NUM_WARPS)-1:0] tensor_sb_set_warp;
    reg [4:0] tensor_sb_set_rd;
    reg tensor_issue_conflict;

    reg wgmma_sb_clr_valid;
    reg [$clog2(NUM_WARPS)-1:0] wgmma_sb_clr_warp;
    reg [4:0] wgmma_sb_clr_rd;

    reg wb_valid;
    reg [$clog2(NUM_WARPS)-1:0] wb_warp_id;
    reg [4:0] wb_rd;

    wire [31:0] stat_cycles;
    wire [31:0] stat_single_issue;
    wire [31:0] stat_dual_issue;
    wire [31:0] stat_stalls;
    wire [31:0] stat_async_mma_issued;
    wire [31:0] stat_async_mma_completed;
    wire [31:0] stat_tcgen05_issued;
    wire perf_sched_stall_ifetch;
    wire [31:0] scoreboard_out [0:NUM_WARPS-1];

    reg [31:0] shadow_pc;
    reg [31:0] pc_before_check;
    integer i;

    blackwell_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_SCHEDULERS(NUM_SCHEDULERS)
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
        .tensor_sb_set_valid(tensor_sb_set_valid),
        .tensor_sb_set_warp(tensor_sb_set_warp),
        .tensor_sb_set_rd(tensor_sb_set_rd),
        .tensor_issue_conflict(tensor_issue_conflict),
        .wgmma_sb_clr_valid(wgmma_sb_clr_valid),
        .wgmma_sb_clr_warp(wgmma_sb_clr_warp),
        .wgmma_sb_clr_rd(wgmma_sb_clr_rd),
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
        .scoreboard_out(scoreboard_out)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shadow_pc <= 32'h0;
        end else begin
            // Mirrors SM PC gate: advance only when issue_valid && warp_inst_consume.
            if (issue_valid[0] && warp_inst_consume[issue_warp_id[0]]) begin
                shadow_pc <= shadow_pc + 32'd4;
            end
        end
    end

    initial begin
        rst_n = 1'b0;

        warp_valid = '0;
        warp_ready = '0;
        warp_diverged = '0;
        warp_at_barrier = '0;
        warp_inst_valid = '0;

        warp_is_compute = '0;
        warp_is_tensor = '0;
        tensor_push_locked = '0;
        warp_is_memory = '0;
        warp_is_branch = '0;
        warp_is_alu = '0;
        warp_is_mul = '0;
        warp_is_fp32 = '0;
        warp_is_fp16 = '0;
        warp_is_sfu = '0;
        warp_is_shfl = '0;
        warp_is_video = '0;
        warp_writes_reg = '0;

        warp_is_tcgen05 = '0;
        warp_is_tcgen05_mma = '0;
        warp_is_tcgen05_alloc = '0;
        warp_is_tcgen05_ld = '0;
        warp_is_tcgen05_st = '0;
        warp_is_tcgen05_commit = '0;
        warp_is_tcgen05_wait = '0;

        tmem_alloc_valid = '0;
        tmem_pipe_ready = 1'b1;
        async_mma_complete = 1'b0;
        async_mma_warp_id = '0;
        async_mma_op_id = 4'b0;

        compute_pipe0_ready = 1'b1;
        compute_pipe1_ready = 1'b1;
        tensor_pipe_ready = 1'b1;
        memory_pipe_ready = 1'b1;
        branch_unit_ready = 1'b1;

        pipeline_stall = 1'b0;
        fu_conflict_sb_clr_valid = 1'b0;
        fu_conflict_sb_clr_warp = '0;
        fu_conflict_sb_clr_rd = 5'b0;
        pipeline_stall_slot1 = 1'b0;
        tensor_sb_set_valid = 1'b0;
        tensor_sb_set_warp = '0;
        tensor_sb_set_rd = 5'b0;
        tensor_issue_conflict = 1'b0;

        wgmma_sb_clr_valid = 1'b0;
        wgmma_sb_clr_warp = '0;
        wgmma_sb_clr_rd = 5'b0;

        wb_valid = 1'b0;
        wb_warp_id = '0;
        wb_rd = 5'b0;

        for (i = 0; i < NUM_WARPS; i = i + 1) begin
            warp_inst[i] = 32'h0;
            warp_rd[i] = 5'd0;
            warp_rs1[i] = 5'd0;
            warp_rs2[i] = 5'd0;
            warp_rs3[i] = 5'd0;
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Precondition for tcgen05.mma: TMEM must be allocated for warp0.
        force dut.warp_has_tmem_alloc[0] = 1'b1;

        // Directed setup: only warp0 is active tcgen05.mma and should be issuable.
        warp_valid[0] = 1'b1;
        warp_ready[0] = 1'b1;
        warp_inst_valid[0] = 1'b1;
        warp_is_tcgen05[0] = 1'b1;
        warp_is_tcgen05_mma[0] = 1'b1;

        @(posedge clk);
        #1;

        if (!issue_valid[0] || issue_warp_id[0] != 0) begin
            $display("FAIL: expected issue slot0 to pick warp0 tcgen05");
            $finish(1);
        end

        if (!warp_inst_consume[0]) begin
            $display("FAIL: baseline consume should be 1 without lockout");
            $finish(1);
        end

        pc_before_check = shadow_pc;
        @(posedge clk);
        #1;
        if (shadow_pc != (pc_before_check + 32'd4)) begin
            $display("FAIL: baseline PC gate should advance by +4 (before=%h after=%h)",
                     pc_before_check, shadow_pc);
            $finish(1);
        end

        // Enable lockout: consume must be suppressed for tcgen05 and PC must hold.
        tensor_push_locked[0] = 1'b1;

        @(posedge clk);
        #1;
        if (!issue_valid[0]) begin
            $display("FAIL: lockout case lost issue_valid unexpectedly");
            $finish(1);
        end
        if (warp_inst_consume[0]) begin
            $display("FAIL: lockout should suppress tcgen05 consume");
            $finish(1);
        end

        pc_before_check = shadow_pc;
        @(posedge clk);
        #1;
        if (shadow_pc != pc_before_check) begin
            $display("FAIL: lockout PC gate should hold (before=%h after=%h)",
                     pc_before_check, shadow_pc);
            $finish(1);
        end

        $display("Directed check: tcgen05 + lockout consume/PC guard");
        $display("PASS: consume suppressed and PC held under lockout (shadow_pc=%h)", shadow_pc);
        $finish(0);
    end
endmodule
