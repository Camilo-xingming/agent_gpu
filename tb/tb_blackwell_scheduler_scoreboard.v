`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_blackwell_scheduler_scoreboard;
    localparam NUM_WARPS = 2;
    localparam NUM_SCHEDULERS = 2;
    localparam INST_WIDTH = 32;
    localparam WARP_W = $clog2(NUM_WARPS);

    reg clk;
    reg rst_n;

    // Warp status / instruction inputs
    reg [NUM_WARPS-1:0] warp_valid;
    reg [NUM_WARPS-1:0] warp_ready;
    reg [NUM_WARPS-1:0] warp_diverged;
    reg [NUM_WARPS-1:0] warp_at_barrier;
    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0] warp_inst_valid;

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

    reg pipeline_stall;
    reg fu_conflict_sb_clr_valid;
    reg [WARP_W-1:0] fu_conflict_sb_clr_warp;
    reg [4:0] fu_conflict_sb_clr_rd;
    reg pipeline_stall_slot1;
    reg tensor_sb_set_valid;
    reg [WARP_W-1:0] tensor_sb_set_warp;
    reg [4:0] tensor_sb_set_rd;
    reg tensor_issue_conflict;
    reg wgmma_sb_clr_valid;
    reg [WARP_W-1:0] wgmma_sb_clr_warp;
    reg [4:0] wgmma_sb_clr_rd;

    reg wb_valid;
    reg [WARP_W-1:0] wb_warp_id;
    reg [4:0] wb_rd;

    wire [NUM_SCHEDULERS-1:0] issue_valid;
    wire [NUM_SCHEDULERS*WARP_W-1:0] issue_warp_id;
    wire [NUM_SCHEDULERS*INST_WIDTH-1:0] issue_inst;
    wire [NUM_SCHEDULERS*3-1:0] issue_pipe;
    wire [NUM_WARPS-1:0] warp_inst_consume;

    wire [NUM_SCHEDULERS-1:0] issue_is_async_mma;
    wire [NUM_SCHEDULERS*4-1:0] issue_async_mma_id;
    wire [31:0] stat_cycles;
    wire [31:0] stat_single_issue;
    wire [31:0] stat_dual_issue;
    wire [31:0] stat_stalls;
    wire [31:0] stat_async_mma_issued;
    wire [31:0] stat_async_mma_completed;
    wire [31:0] stat_tcgen05_issued;
    wire perf_sched_stall_ifetch;
    wire [NUM_WARPS*32-1:0] scoreboard_out;

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

    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;

    task check_expect;
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
        clk = 1'b0;
        rst_n = 1'b0;
        pass_count = 0;
        fail_count = 0;

        warp_valid = 1'b1;
        warp_ready = 1'b1;
        warp_diverged = 1'b0;
        warp_at_barrier = 1'b0;
        warp_inst = 32'h0;
        warp_inst_valid = 1'b1;

        warp_rd = 5'd5;
        warp_rs1 = 5'd1;
        warp_rs2 = 5'd2;
        warp_rs3 = 5'd3;
        warp_reads_rs3 = 1'b1;
        warp_is_compute = 1'b1;
        warp_is_tensor = 1'b0;
        tensor_push_locked = 1'b0;
        warp_is_memory = 1'b0;
        warp_is_branch = 1'b0;
        warp_is_alu = 1'b1;
        warp_is_mul = 1'b0;
        warp_is_fp32 = 1'b0;
        warp_is_fp16 = 1'b0;
        warp_is_sfu = 1'b0;
        warp_is_shfl = 1'b0;
        warp_is_video = 1'b0;
        warp_writes_reg = 1'b1;

        warp_is_tcgen05 = 1'b0;
        warp_is_tcgen05_mma = 1'b0;
        warp_is_tcgen05_alloc = 1'b0;
        warp_is_tcgen05_ld = 1'b0;
        warp_is_tcgen05_st = 1'b0;
        warp_is_tcgen05_commit = 1'b0;
        warp_is_tcgen05_wait = 1'b0;

        tmem_alloc_valid = 1'b0;
        tmem_pipe_ready = 1'b1;

        async_mma_complete = 1'b0;
        async_mma_warp_id = {WARP_W{1'b0}};
        async_mma_op_id = 4'b0;

        compute_pipe0_ready = 1'b1;
        compute_pipe1_ready = 1'b1;
        tensor_pipe_ready = 1'b1;
        memory_pipe_ready = 1'b1;
        branch_unit_ready = 1'b1;

        pipeline_stall = 1'b0;
        fu_conflict_sb_clr_valid = 1'b0;
        fu_conflict_sb_clr_warp = {WARP_W{1'b0}};
        fu_conflict_sb_clr_rd = 5'b0;
        pipeline_stall_slot1 = 1'b0;
        tensor_sb_set_valid = 1'b0;
        tensor_sb_set_warp = {WARP_W{1'b0}};
        tensor_sb_set_rd = 5'b0;
        tensor_issue_conflict = 1'b0;
        wgmma_sb_clr_valid = 1'b0;
        wgmma_sb_clr_warp = {WARP_W{1'b0}};
        wgmma_sb_clr_rd = 5'b0;

        wb_valid = 1'b0;
        wb_warp_id = {WARP_W{1'b0}};
        wb_rd = 5'b0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Seed scoreboard by issuing an instruction that writes R5.
        repeat (3) @(posedge clk);
        #1;
        check_expect(dut.scoreboard[0][5], "scoreboard R5 should be set after issue");

        // Prepare non-writing probe instruction with rs3=R5.
        warp_writes_reg = 1'b0;
        warp_rd = 5'd6;
        warp_rs1 = 5'd1;
        warp_rs2 = 5'd2;
        warp_rs3 = 5'd5;

        // Case A: rs3 is architecturally unused -> should NOT stall.
        warp_reads_rs3 = 1'b0;
        #1;
        check_expect(issue_valid[0], "issue should proceed when warp_reads_rs3=0");

        // Case B: rs3 is used -> should stall on RAW hazard (R5 busy).
        warp_reads_rs3 = 1'b1;
        #1;
        check_expect(!issue_valid[0], "issue should stall when warp_reads_rs3=1 and rs3 busy");
        check_expect(dut.warp_has_hazard[0], "hazard flag should assert when rs3 dependency is enabled");

        // Clear scoreboard via writeback and verify issue resumes.
        @(posedge clk);
        wb_valid <= 1'b1;
        wb_rd <= 5'd5;

        @(posedge clk);
        wb_valid <= 1'b0;
        #1;
        check_expect(!dut.scoreboard[0][5], "scoreboard R5 should clear on writeback");
        check_expect(issue_valid[0], "issue should resume after writeback clears dependency");

        $display("============================================================");
        $display("Blackwell Scheduler RS3 Mask Test");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        if (fail_count == 0)
            $display("  STATUS: PASS");
        else
            $display("  STATUS: FAIL");
        $display("============================================================");

        if (fail_count != 0)
            $fatal(1);

        $finish;
    end
endmodule
