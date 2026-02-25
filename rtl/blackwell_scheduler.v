//============================================================================
// RalphGPU - Blackwell-Style Multi-Scheduler with Per-Thread Tensor Support
// Configurable number of parallel schedulers (1-4)
// Each scheduler manages a subset of warps: warp_id % NUM_SCHEDULERS
//
// Blackwell Enhancements (SM100+):
// - Per-thread tensor operation tracking (tcgen05 support)
// - Async MMA scoreboard for non-blocking tensor operations
// - Independent compute/tensor/memory pipeline scheduling
// - Reduced synchronization overhead for tensor ops
// - TMEM operation tracking for tcgen05.alloc/ld/st/mma
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module blackwell_scheduler #(
    parameter NUM_WARPS      = 8,
    parameter NUM_SCHEDULERS = `NUM_SCHEDULERS,  // Configurable: 1, 2, or 4 (Blackwell uses 4)
    parameter INST_WIDTH     = 32,
    parameter SCOREBOARD_DEPTH = 16,             // For compatibility
    parameter NUM_THREADS_PER_WARP = 32,         // Threads per warp
    parameter MAX_ASYNC_MMA_OPS = 8              // Max outstanding async MMA per warp
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Warp Status Interface (same as advanced_warp_scheduler)
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]     warp_valid,
    input  wire [NUM_WARPS-1:0]     warp_ready,         // Not stalled
    input  wire [NUM_WARPS-1:0]     warp_diverged,      // In divergent execution
    input  wire [NUM_WARPS-1:0]     warp_at_barrier,

    //------------------------------------------------------------------------
    // Instruction Buffer Interface (same as advanced_warp_scheduler)
    //------------------------------------------------------------------------
    input wire [NUM_WARPS*(INST_WIDTH)-1:0] warp_inst,
    input  wire [NUM_WARPS-1:0]     warp_inst_valid,
    output wire [NUM_WARPS-1:0]     warp_inst_consume,

    //------------------------------------------------------------------------
    // Decoded Instruction Info (same as advanced_warp_scheduler)
    //------------------------------------------------------------------------
    input wire [NUM_WARPS*5-1:0] warp_rd,
    input wire [NUM_WARPS*5-1:0] warp_rs1,
    input wire [NUM_WARPS*5-1:0] warp_rs2,
    input wire [NUM_WARPS*5-1:0] warp_rs3,
    input  wire [NUM_WARPS-1:0]     warp_reads_rs3,
    input  wire [NUM_WARPS-1:0]     warp_is_compute,
    input  wire [NUM_WARPS-1:0]     warp_is_tensor,
    input  wire [NUM_WARPS-1:0]     tensor_push_locked,  // 2-cycle lockout from SM
    input  wire [NUM_WARPS-1:0]     warp_is_memory,
    input  wire [NUM_WARPS-1:0]     warp_is_branch,
    // Fine-grained FU types for conflict detection
    input  wire [NUM_WARPS-1:0]     warp_is_alu,
    input  wire [NUM_WARPS-1:0]     warp_is_mul,
    input  wire [NUM_WARPS-1:0]     warp_is_fp32,
    input  wire [NUM_WARPS-1:0]     warp_is_fp16,
    input  wire [NUM_WARPS-1:0]     warp_is_sfu,
    input  wire [NUM_WARPS-1:0]     warp_is_shfl,
    input  wire [NUM_WARPS-1:0]     warp_is_video,
    input  wire [NUM_WARPS-1:0]     warp_writes_reg,

    //------------------------------------------------------------------------
    // Blackwell Per-Thread Tensor Info (tcgen05 support)
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05,       // tcgen05 instruction
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_mma,   // tcgen05.mma (async MMA)
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_alloc, // tcgen05.alloc
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_ld,    // tcgen05.ld
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_st,    // tcgen05.st
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_commit,// tcgen05.commit
    input  wire [NUM_WARPS-1:0]     warp_is_tcgen05_wait,  // tcgen05.wait

    //------------------------------------------------------------------------
    // TMEM Status Interface (from tensor_memory module)
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]     tmem_alloc_valid,      // TMEM allocated for warp
    input  wire                     tmem_pipe_ready,       // TMEM port available

    //------------------------------------------------------------------------
    // Async MMA Completion Interface (from tensor_core)
    //------------------------------------------------------------------------
    input  wire                     async_mma_complete,    // An async MMA completed
    input  wire [$clog2(NUM_WARPS)-1:0] async_mma_warp_id, // Warp ID of completed MMA
    input  wire [3:0]               async_mma_op_id,       // Operation ID within warp

    //------------------------------------------------------------------------
    // Execution Unit Availability (compatible interface)
    //------------------------------------------------------------------------
    input  wire                     compute_pipe0_ready,
    input  wire                     compute_pipe1_ready,
    input  wire                     tensor_pipe_ready,
    input  wire                     memory_pipe_ready,
    input  wire                     branch_unit_ready,

    //------------------------------------------------------------------------
    // Issue Outputs (NUM_SCHEDULERS outputs, compatible with NUM_ISSUE=2)
    //------------------------------------------------------------------------
    output wire [NUM_SCHEDULERS-1:0]              issue_valid,
    output wire [NUM_SCHEDULERS*($clog2(NUM_WARPS))-1:0] issue_warp_id,
    output wire [NUM_SCHEDULERS*(INST_WIDTH)-1:0] issue_inst,
    output wire [NUM_SCHEDULERS*3-1:0] issue_pipe,

    //------------------------------------------------------------------------
    // Blackwell-specific Issue Outputs
    //------------------------------------------------------------------------
    output reg  [NUM_SCHEDULERS-1:0]              issue_is_async_mma, // Issued op is async MMA
    output reg [NUM_SCHEDULERS*4-1:0] issue_async_mma_id, // Async MMA op ID

    //------------------------------------------------------------------------
    // Writeback Interface (for scoreboard clearing)
    //------------------------------------------------------------------------
    input  wire                     pipeline_stall,
    // Issue-stage FU conflict scoreboard rollback
    input  wire                     fu_conflict_sb_clr_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] fu_conflict_sb_clr_warp,
    input  wire [4:0]              fu_conflict_sb_clr_rd,
    input  wire                     pipeline_stall_slot1,
    // Tensor scoreboard deferred SET: SM signals when tensor push actually succeeds
    input  wire                     tensor_sb_set_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] tensor_sb_set_warp,
    input  wire [4:0]               tensor_sb_set_rd,
    input  wire                     tensor_issue_conflict, // lane1 tensor suppressed by lane0

    // WGMMA completion (clears scoreboard for async MMA ops)
    input  wire                     wgmma_sb_clr_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] wgmma_sb_clr_warp,
    input  wire [4:0]               wgmma_sb_clr_rd,

    // Pipeline replay scoreboard rollback (L1 miss → PC rollback)
    input  wire                     replay_sb_clr_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] replay_sb_clr_warp,
    input  wire [4:0]               replay_sb_clr_rd,

    input  wire                     wb_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] wb_warp_id,
    input  wire [4:0]               wb_rd,

    //------------------------------------------------------------------------
    // Statistics (compatible + Blackwell extensions)
    //------------------------------------------------------------------------
    output wire [31:0]              stat_cycles,
    output wire [31:0]              stat_single_issue,
    output wire [31:0]              stat_dual_issue,
    output wire [31:0]              stat_stalls,
    output wire [31:0]              stat_async_mma_issued,    // Async MMA ops issued
    output wire [31:0]              stat_async_mma_completed, // Async MMA ops completed
    output wire [31:0]              stat_tcgen05_issued,      // Total tcgen05 ops issued

    //------------------------------------------------------------------------
    // Scoreboard Visibility (eliminates hierarchical references)
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    // Scheduler-centric IFetch stall (true when IFetch is the bottleneck)
    //------------------------------------------------------------------------
    output wire                     perf_sched_stall_ifetch,

    output wire [NUM_WARPS*4-1:0] issue_seq_out,
    output wire [NUM_WARPS*32-1:0] scoreboard_out
);

    localparam WARP_W = $clog2(NUM_WARPS);
    localparam WARPS_PER_SCHED = (NUM_WARPS + NUM_SCHEDULERS - 1) / NUM_SCHEDULERS;

    //------------------------------------------------------------------------
    // Pipe Encoding (compatible with advanced_warp_scheduler)
    //------------------------------------------------------------------------
    localparam PIPE_COMPUTE0 = 3'd0;
    localparam PIPE_COMPUTE1 = 3'd1;
    localparam PIPE_TENSOR   = 3'd2;
    localparam PIPE_MEMORY   = 3'd3;
    localparam PIPE_BRANCH   = 3'd4;
    localparam PIPE_TCGEN05  = 3'd5;    // Blackwell tcgen05 dedicated pipe
    localparam PIPE_TMEM     = 3'd6;    // TMEM operations pipe

    //------------------------------------------------------------------------
    // Per-Warp Scoreboard (register dependencies)
    //------------------------------------------------------------------------
    reg [31:0] scoreboard [0:NUM_WARPS-1];

    // Expose scoreboard to parent module (replaces hierarchical references)
    genvar sb_gi;
    generate
        for (sb_gi = 0; sb_gi < NUM_WARPS; sb_gi = sb_gi + 1) begin : gen_sb_out
            assign scoreboard_out[sb_gi*32 +: 32] = scoreboard[sb_gi];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Per-Warp Issue Sequence Number (ISN) — stale tensor push elimination
    //------------------------------------------------------------------------
    reg [3:0] issue_seq [0:NUM_WARPS-1];

    genvar isn_gi;
    generate
        for (isn_gi = 0; isn_gi < NUM_WARPS; isn_gi = isn_gi + 1) begin : gen_isn_out
            assign issue_seq_out[isn_gi*4 +: 4] = issue_seq[isn_gi];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Async MMA Scoreboard (Blackwell per-thread tensor tracking)
    // Tracks outstanding async MMA operations per warp
    // Each warp can have up to MAX_ASYNC_MMA_OPS in-flight
    //------------------------------------------------------------------------
    reg [MAX_ASYNC_MMA_OPS-1:0] async_mma_pending [0:NUM_WARPS-1];  // Bitmask of pending ops
    reg [3:0] async_mma_next_id [0:NUM_WARPS-1];                    // Next op ID to allocate
    reg [3:0] async_mma_count [0:NUM_WARPS-1];                      // Count of pending ops

    // Per-warp TMEM allocation status (for tcgen05.alloc dependency)
    reg [NUM_WARPS-1:0] warp_has_tmem_alloc;

    // tcgen05.commit/wait tracking - these need to wait for async_mma to complete
    wire [NUM_WARPS-1:0] warp_async_mma_pending_any;
    genvar aw;
    generate
        for (aw = 0; aw < NUM_WARPS; aw = aw + 1) begin : gen_async_pending
            assign warp_async_mma_pending_any[aw] = |async_mma_pending[aw];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Per-Warp Hazard Detection (Enhanced for Blackwell)
    //------------------------------------------------------------------------
    wire [NUM_WARPS-1:0] warp_has_hazard;
    wire [NUM_WARPS-1:0] warp_has_async_hazard;  // Async MMA specific hazards
    wire [NUM_WARPS-1:0] warp_has_tmem_hazard;   // TMEM allocation hazards

    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : gen_hazard
            // Standard RAW/WAW hazards for register operands
            wire raw_hazard = scoreboard[w][warp_rs1[w*5 +: 5]] ||
                             scoreboard[w][warp_rs2[w*5 +: 5]] ||
                             (warp_reads_rs3[w] && scoreboard[w][warp_rs3[w*5 +: 5]]);
            wire waw_hazard = warp_writes_reg[w] && scoreboard[w][warp_rd[w*5 +: 5]];

            // Async MMA hazards:
            // - tcgen05.commit/wait must wait for all pending async MMA to complete
            // - tcgen05.mma can proceed if async queue not full
            wire async_commit_wait_hazard = (warp_is_tcgen05_commit[w] || warp_is_tcgen05_wait[w]) &&
                                            warp_async_mma_pending_any[w];
            wire async_queue_full = (async_mma_count[w] >= MAX_ASYNC_MMA_OPS);
            wire async_mma_hazard = warp_is_tcgen05_mma[w] && async_queue_full;

            assign warp_has_async_hazard[w] = async_commit_wait_hazard || async_mma_hazard;

            // TMEM hazards:
            // - tcgen05.ld/st/mma require TMEM to be allocated first
            // - tcgen05.dealloc must wait for all TMEM operations to complete
            wire tmem_not_allocated = (warp_is_tcgen05_ld[w] || warp_is_tcgen05_st[w] ||
                                       warp_is_tcgen05_mma[w]) && !warp_has_tmem_alloc[w];
            assign warp_has_tmem_hazard[w] = tmem_not_allocated;

            // Combined hazard
            assign warp_has_hazard[w] = raw_hazard || waw_hazard ||
                                        warp_has_async_hazard[w] || warp_has_tmem_hazard[w];
        end
    endgenerate

    // Standard eligibility: valid, ready, has instruction, no hazard, not diverged, not at barrier
    wire [NUM_WARPS-1:0] warp_eligible_base = warp_valid & warp_ready & warp_inst_valid &
                                               ~warp_has_hazard & ~warp_diverged & ~warp_at_barrier;

    // Blackwell enhancement: tcgen05 ops don't require warp synchronization
    // Per-thread tensor ops can be issued even if warp is diverged (only active threads execute)
    wire [NUM_WARPS-1:0] warp_tcgen05_eligible = warp_valid & warp_inst_valid &
                                                  warp_is_tcgen05 & ~warp_has_hazard &
                                                  ~warp_at_barrier;  // No divergence check for tcgen05

    // Combined eligibility
    wire [NUM_WARPS-1:0] warp_eligible = warp_eligible_base | warp_tcgen05_eligible;

    // Scheduler-centric IFetch stall: no warp eligible, but at least one warp
    // WOULD be eligible if it had a valid instruction (IFetch is the bottleneck)
    wire [NUM_WARPS-1:0] warp_ifetch_blocked = warp_valid & warp_ready & ~warp_inst_valid
                                              & ~warp_has_hazard & ~warp_diverged & ~warp_at_barrier;
    assign perf_sched_stall_ifetch = (warp_eligible == {NUM_WARPS{1'b0}}) && (|warp_ifetch_blocked);

    //------------------------------------------------------------------------
    // Per-Scheduler Warp Selection
    // Blackwell: Each scheduler handles warps where (warp_id % NUM_SCHEDULERS == scheduler_id)
    //------------------------------------------------------------------------
    reg [NUM_SCHEDULERS-1:0] issue_valid_r;
    reg [WARP_W-1:0]         issue_warp_r [0:NUM_SCHEDULERS-1];
    reg [INST_WIDTH-1:0]     issue_inst_r [0:NUM_SCHEDULERS-1];
    reg [2:0]                issue_pipe_r [0:NUM_SCHEDULERS-1];
    reg [NUM_WARPS-1:0]      issue_consume_r;
    reg sched_fu_conflict;

    // Round-robin pointer per scheduler (for fairness within assigned warps)
    reg [WARP_W-1:0] sched_rr_ptr [0:NUM_SCHEDULERS-1];

    // Compute pipe assignment: scheduler 0 -> pipe0, scheduler 1 -> pipe1, etc.
    wire [NUM_SCHEDULERS-1:0] compute_pipe_ready;
    generate
        if (NUM_SCHEDULERS >= 1) begin : gen_pipe_ready_0
            assign compute_pipe_ready[0] = compute_pipe0_ready;
        end
        if (NUM_SCHEDULERS >= 2) begin : gen_pipe_ready_1
            assign compute_pipe_ready[1] = compute_pipe1_ready;
        end
        // For NUM_SCHEDULERS > 2, share pipes (round-robin or priority)
        if (NUM_SCHEDULERS >= 3) begin : gen_pipe_ready_2
            assign compute_pipe_ready[2] = compute_pipe0_ready; // Share with pipe0
        end
        if (NUM_SCHEDULERS >= 4) begin : gen_pipe_ready_3
            assign compute_pipe_ready[3] = compute_pipe1_ready; // Share with pipe1
        end
    endgenerate

    // Async MMA ID allocation per scheduler
    reg [3:0] issue_async_mma_id_r [0:NUM_SCHEDULERS-1];
    reg [NUM_SCHEDULERS-1:0] issue_is_async_mma_r;

    // Scheduler selection logic (enhanced for Blackwell tcgen05)
    wire [NUM_WARPS-1:0] tensor_like_warp_mask = warp_is_tensor | warp_is_tcgen05;
    wire [NUM_WARPS-1:0] tensor_like_lockout_mask = tensor_push_locked & tensor_like_warp_mask;
    integer s, sw;
    always @(*) begin
        issue_valid_r = 0;
        sched_fu_conflict = 1'b0;
        issue_consume_r = 0;
        issue_is_async_mma_r = 0;
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            issue_warp_r[s] = 0;
            issue_inst_r[s] = 0;
            issue_pipe_r[s] = PIPE_COMPUTE0;
            issue_async_mma_id_r[s] = 0;
        end

        // Each scheduler selects from its assigned warps
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            // Iterate through warps assigned to this scheduler
            for (sw = 0; sw < WARPS_PER_SCHED; sw = sw + 1) begin
                // Calculate warp ID for this scheduler
                // Warp assignment: scheduler s handles warps s, s+NUM_SCHEDULERS, s+2*NUM_SCHEDULERS, ...
                if (!issue_valid_r[s]) begin
                    // Use round-robin starting point for fairness
                    integer warp_idx;
                    /* verilator lint_off WIDTHEXPAND */ // integer RR math; width-normalized by warp_idx type
                    warp_idx = s + ((sched_rr_ptr[s] + sw) % WARPS_PER_SCHED) * NUM_SCHEDULERS;
                    /* verilator lint_on WIDTHEXPAND */

                    if (warp_idx < NUM_WARPS && warp_eligible[warp_idx]) begin
                        // Priority order: Branch > tcgen05 > Memory > Tensor > Compute
                        // tcgen05 ops have high priority to maximize tensor core utilization
                        if (warp_is_branch[warp_idx] && branch_unit_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                            issue_pipe_r[s] = PIPE_BRANCH;
                            issue_consume_r[warp_idx] = 1'b1;

                        // Blackwell tcgen05 instructions (per-thread tensor ops)
                        end else if (warp_is_tcgen05[warp_idx]) begin
                            // tcgen05.mma -> tensor pipe (async, per-thread)
                            if (warp_is_tcgen05_mma[warp_idx] && tensor_pipe_ready) begin
                                issue_valid_r[s] = 1'b1;
                                issue_warp_r[s] = warp_idx[WARP_W-1:0];
                                issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                                issue_pipe_r[s] = PIPE_TCGEN05;
                                issue_consume_r[warp_idx] = 1'b1;
                                issue_is_async_mma_r[s] = 1'b1;
                                issue_async_mma_id_r[s] = async_mma_next_id[warp_idx];

                            // tcgen05.ld/st -> TMEM pipe
                            end else if ((warp_is_tcgen05_ld[warp_idx] || warp_is_tcgen05_st[warp_idx]) &&
                                         tmem_pipe_ready) begin
                                issue_valid_r[s] = 1'b1;
                                issue_warp_r[s] = warp_idx[WARP_W-1:0];
                                issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                                issue_pipe_r[s] = PIPE_TMEM;
                                issue_consume_r[warp_idx] = 1'b1;

                            // tcgen05.alloc/dealloc/commit/wait -> TMEM pipe (control ops)
                            end else if ((warp_is_tcgen05_alloc[warp_idx] ||
                                          warp_is_tcgen05_commit[warp_idx] ||
                                          warp_is_tcgen05_wait[warp_idx]) &&
                                         tmem_pipe_ready) begin
                                issue_valid_r[s] = 1'b1;
                                issue_warp_r[s] = warp_idx[WARP_W-1:0];
                                issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                                issue_pipe_r[s] = PIPE_TMEM;
                                issue_consume_r[warp_idx] = 1'b1;
                            end

                        end else if (warp_is_memory[warp_idx] && memory_pipe_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                            issue_pipe_r[s] = PIPE_MEMORY;
                            issue_consume_r[warp_idx] = 1'b1;

                        // Legacy warp-synchronous tensor ops (WMMA/WGMMA)
                        end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                            issue_pipe_r[s] = PIPE_TENSOR;
                            issue_consume_r[warp_idx] = 1'b1;

                        end else if (warp_is_compute[warp_idx] && compute_pipe_ready[s]) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx*INST_WIDTH +: INST_WIDTH];
                            // Map scheduler to compute pipe
                            issue_pipe_r[s] = (s % 2 == 0) ? PIPE_COMPUTE0 : PIPE_COMPUTE1;
                            issue_consume_r[warp_idx] = 1'b1;
                        end
                    end
                end
            end
        end
    
        // Compute FU conflict after selection (suppress slot 1 if same FU as slot 0)
        if (NUM_SCHEDULERS >= 2 && issue_valid_r[0] && issue_valid_r[1]) begin
            if ((warp_is_alu[issue_warp_r[0]] && warp_is_alu[issue_warp_r[1]]) ||
                (warp_is_mul[issue_warp_r[0]] && warp_is_mul[issue_warp_r[1]]) ||
                (warp_is_fp32[issue_warp_r[0]] && warp_is_fp32[issue_warp_r[1]]) ||
                (warp_is_fp16[issue_warp_r[0]] && warp_is_fp16[issue_warp_r[1]]) ||
                (warp_is_sfu[issue_warp_r[0]] && warp_is_sfu[issue_warp_r[1]]) ||
                (warp_is_shfl[issue_warp_r[0]] && warp_is_shfl[issue_warp_r[1]]) ||
                (warp_is_video[issue_warp_r[0]] && warp_is_video[issue_warp_r[1]]) ||
                (warp_is_memory[issue_warp_r[0]] && warp_is_memory[issue_warp_r[1]]) ||
                warp_is_branch[issue_warp_r[0]]) begin
                sched_fu_conflict = 1'b1;
                issue_valid_r[1] = 1'b0;  // Suppress slot 1
                issue_consume_r[issue_warp_r[1]] = 1'b0;  // Don't consume slot 1's instruction
            end
        end

        // Tensor-like lockout guard (slot1 only): preserve slot0 forward progress.
        // Slot1 still gets consume suppression to avoid stale replay in the 2-cycle
        // tensor_push_locked window.
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            if (s > 0 && issue_valid_r[s] && tensor_like_lockout_mask[issue_warp_r[s]]) begin
                issue_consume_r[issue_warp_r[s]] = 1'b0;
            end
        end
end

    // Output async MMA info
    always @(*) begin
        issue_is_async_mma = issue_is_async_mma_r;
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            issue_async_mma_id[s*4 +: 4] = issue_async_mma_id_r[s];
        end
    end

    // Suppress consume when stalled OR when slot 1 tensor was suppressed
    wire [NUM_WARPS-1:0] conflict_mask = tensor_issue_conflict ? (1 << issue_warp_r[1]) : {NUM_WARPS{1'b0}};
    // Per-slot stall: even warps (0,2) use slot 0 stall, odd warps (1,3) use slot 1 stall
    wire [NUM_WARPS-1:0] stall_mask;
    genvar sm_w;
    generate
        for (sm_w = 0; sm_w < NUM_WARPS; sm_w = sm_w + 1) begin : gen_stall_mask
            assign stall_mask[sm_w] = (sm_w % 2 == 0) ? pipeline_stall : pipeline_stall_slot1;
        end
    endgenerate
    assign warp_inst_consume = (issue_consume_r & ~conflict_mask & ~stall_mask);

    //------------------------------------------------------------------------
    // Scoreboard Update (Enhanced for Blackwell async operations)
    //------------------------------------------------------------------------
    integer sb_w, sb_s;
    /* verilator lint_on SELRANGE */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sb_w = 0; sb_w < NUM_WARPS; sb_w = sb_w + 1) begin
                scoreboard[sb_w] <= 0;
                async_mma_pending[sb_w] <= 0;
                async_mma_next_id[sb_w] <= 0;
                async_mma_count[sb_w] <= 0;
                issue_seq[sb_w] <= 4'd0;
            end
            for (sb_s = 0; sb_s < NUM_SCHEDULERS; sb_s = sb_s + 1) begin
                sched_rr_ptr[sb_s] <= 0;
            end
            warp_has_tmem_alloc <= 0;
        end else begin
            // Set scoreboard on issue
            // Note: R0 is a normal register in GPU (not hardwired zero like RISC-V)
            // NOP/BAR_SYNC/etc. won't set scoreboard because warp_writes_reg=0 for them
            for (sb_s = 0; sb_s < NUM_SCHEDULERS; sb_s = sb_s + 1) begin
                // Skip scoreboard SET when pipeline stalled or when slot 1 tensor was suppressed
                if (issue_valid_r[sb_s] &&
                    !((issue_warp_r[sb_s][0]) ? pipeline_stall_slot1 : pipeline_stall) &&
                    !(sb_s > 0 && tensor_issue_conflict) &&
                    !(sb_s > 0 && sched_fu_conflict)) begin
                    // Standard register scoreboard update
                    // For tensor ops: defer scoreboard SET to tensor_sb_set feedback
                    // (scoreboard set happens when SM's tensor push actually succeeds)
                    if (warp_writes_reg[issue_warp_r[sb_s]] &&
                        issue_pipe_r[sb_s] != PIPE_TENSOR) begin
                        scoreboard[issue_warp_r[sb_s]][warp_rd[issue_warp_r[sb_s]*5 +: 5]] <= 1'b1;
                    end

                    // Async MMA scoreboard update (Blackwell)
                    if (issue_is_async_mma_r[sb_s]) begin
                        // Mark this async MMA op as pending
                        async_mma_pending[issue_warp_r[sb_s]][issue_async_mma_id_r[sb_s][2:0]] <= 1'b1;
                        // Increment next ID (wrap around)
                        begin : async_mma_id_wrap
                            reg [3:0] next_mma_id;
                            next_mma_id = async_mma_next_id[issue_warp_r[sb_s]] + 1'b1;
                            async_mma_next_id[issue_warp_r[sb_s]] <=
                                (next_mma_id < MAX_ASYNC_MMA_OPS) ? next_mma_id : 4'd0;
                        end
                        async_mma_count[issue_warp_r[sb_s]] <= async_mma_count[issue_warp_r[sb_s]] + 1;
                    end

                    // TMEM allocation tracking
                    if (warp_is_tcgen05_alloc[issue_warp_r[sb_s]]) begin
                        warp_has_tmem_alloc[issue_warp_r[sb_s]] <= 1'b1;
                    end
                    // Note: dealloc is implicit on kernel exit, but tracked for safety
                    // tcgen05.dealloc is required before kernel exit

                    // Update round-robin pointer for fairness (gated by stall and conflict)
                    if (!((issue_warp_r[sb_s][0]) ? pipeline_stall_slot1 : pipeline_stall) &&
                        !(sb_s > 0 && tensor_issue_conflict) &&
                        !(sb_s > 0 && sched_fu_conflict))
                        begin : rr_ptr_wrap
                            reg [WARP_W-1:0] next_rr;
                            next_rr = sched_rr_ptr[sb_s] + 1'b1;
                            sched_rr_ptr[sb_s] <= (next_rr < WARPS_PER_SCHED[WARP_W-1:0]) ? next_rr : {WARP_W{1'b0}};
                        end
                end
            end


            // Deferred scoreboard SET for tensor ops
            // (tensor ops skip SET at issue; SET happens when SM tensor push succeeds)
            if (tensor_sb_set_valid && tensor_sb_set_rd != 5'b0) begin
                scoreboard[tensor_sb_set_warp][tensor_sb_set_rd] <= 1'b1;
            end
            // ISN increment: fires when instruction is truly consumed
            for (sb_w = 0; sb_w < NUM_WARPS; sb_w = sb_w + 1) begin
                if (warp_inst_consume[sb_w])
                    issue_seq[sb_w] <= issue_seq[sb_w] + 4'd1;
            end

            // Clear scoreboard on writeback
            if (wb_valid) begin
                scoreboard[wb_warp_id][wb_rd] <= 1'b0;
            end

            // Rollback scoreboard SET for issue-stage FU conflict (slot 1 dropped)
            if (fu_conflict_sb_clr_valid) begin
                scoreboard[fu_conflict_sb_clr_warp][fu_conflict_sb_clr_rd] <= 1'b0;
            end

            // Clear scoreboard on WGMMA completion
            if (wgmma_sb_clr_valid) begin
                scoreboard[wgmma_sb_clr_warp][wgmma_sb_clr_rd] <= 1'b0;
            end

            // Clear scoreboard on pipeline replay (L1 miss rollback)
            if (replay_sb_clr_valid) begin
                scoreboard[replay_sb_clr_warp][replay_sb_clr_rd] <= 1'b0;
            end

            // Clear async MMA pending on completion (from tensor_core)
            if (async_mma_complete) begin
                async_mma_pending[async_mma_warp_id][async_mma_op_id[2:0]] <= 1'b0;
                if (async_mma_count[async_mma_warp_id] > 0) begin
                    async_mma_count[async_mma_warp_id] <= async_mma_count[async_mma_warp_id] - 1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics (Enhanced for Blackwell)
    //------------------------------------------------------------------------
    reg [31:0] cycle_count;
    reg [31:0] single_issue_count;
    reg [31:0] dual_issue_count;
    reg [31:0] stall_count;
    reg [31:0] async_mma_issued_count;
    reg [31:0] async_mma_completed_count;
    reg [31:0] tcgen05_issued_count;

    /* verilator lint_off SELRANGE */
    wire [3:0] num_issued = {3'b0, issue_valid_r[0]} + {3'b0, issue_valid_r[1]} +
                            ((NUM_SCHEDULERS > 2) ? {3'b0, issue_valid_r[2]} : 4'b0) +
                            ((NUM_SCHEDULERS > 3) ? {3'b0, issue_valid_r[3]} : 4'b0);

    // Count async MMA and tcgen05 issues this cycle
    wire [3:0] num_async_mma_issued = {3'b0, issue_is_async_mma_r[0]} +
                                       ((NUM_SCHEDULERS > 1) ? {3'b0, issue_is_async_mma_r[1]} : 4'b0) +
                                       ((NUM_SCHEDULERS > 2) ? {3'b0, issue_is_async_mma_r[2]} : 4'b0) +
                                       ((NUM_SCHEDULERS > 3) ? {3'b0, issue_is_async_mma_r[3]} : 4'b0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            single_issue_count <= 0;
            dual_issue_count <= 0;
            stall_count <= 0;
            async_mma_issued_count <= 0;
            async_mma_completed_count <= 0;
            tcgen05_issued_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            // Count issue patterns
            if (num_issued == 0) begin
                stall_count <= stall_count + 1;
            end else if (num_issued == 1) begin
                single_issue_count <= single_issue_count + 1;
            end else begin
                // 2 or more issues = dual/multi issue
                dual_issue_count <= dual_issue_count + 1;
            end

            // Blackwell-specific statistics
            /* verilator lint_off WIDTHEXPAND */ // 4-bit per-cycle tally into 32-bit counter
            async_mma_issued_count <= async_mma_issued_count + num_async_mma_issued;
            /* verilator lint_on WIDTHEXPAND */

            if (async_mma_complete) begin
                async_mma_completed_count <= async_mma_completed_count + 1;
            end

            // Count tcgen05 instructions issued
            for (sb_s = 0; sb_s < NUM_SCHEDULERS; sb_s = sb_s + 1) begin
                if (issue_valid_r[sb_s] && warp_is_tcgen05[issue_warp_r[sb_s]]) begin
                    tcgen05_issued_count <= tcgen05_issued_count + 1;
                end
            end

            // Debug output
            `ifdef SIMULATION
            if (cycle_count < 200) begin
                if(0) $display("[%0t BLACKWELL_SCHED] cycle=%0d issue_valid=%b consume=%b num_issued=%0d async_mma=%0d",
                         $time, cycle_count, issue_valid_r, issue_consume_r, num_issued, num_async_mma_issued);
                if(0) $display("  eligible=%b inst_valid=%b hazard=%b async_hazard=%b tmem_hazard=%b",
                         warp_eligible, warp_inst_valid, warp_has_hazard, warp_has_async_hazard, warp_has_tmem_hazard);
                if(0) $display("  valid=%b ready=%b diverged=%b barrier=%b tcgen05=%b",
                         warp_valid, warp_ready, warp_diverged, warp_at_barrier, warp_is_tcgen05);
                if (issue_valid_r[0])
                    if(0) $display("  sched0: warp=%0d pipe=%0d inst=0x%08x async_mma=%b",
                             issue_warp_r[0], issue_pipe_r[0], issue_inst_r[0], issue_is_async_mma_r[0]);
                if (NUM_SCHEDULERS > 1 && issue_valid_r[1])
                    if(0) $display("  sched1: warp=%0d pipe=%0d inst=0x%08x async_mma=%b",
                             issue_warp_r[1], issue_pipe_r[1], issue_inst_r[1], issue_is_async_mma_r[1]);
            end
            `endif
        end
    end

    assign stat_cycles = cycle_count;
    assign stat_single_issue = single_issue_count;
    assign stat_dual_issue = dual_issue_count;
    assign stat_stalls = stall_count;
    assign stat_async_mma_issued = async_mma_issued_count;
    assign stat_async_mma_completed = async_mma_completed_count;
    assign stat_tcgen05_issued = tcgen05_issued_count;

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign issue_valid = issue_valid_r;

    generate
        genvar i;
        for (i = 0; i < NUM_SCHEDULERS; i = i + 1) begin : gen_issue_out
            assign issue_warp_id[i*($clog2(NUM_WARPS)) +: ($clog2(NUM_WARPS))] = issue_warp_r[i];
            assign issue_inst[i*INST_WIDTH +: INST_WIDTH] = issue_inst_r[i];
            assign issue_pipe[i*3 +: 3] = issue_pipe_r[i];
        end
    endgenerate

    always @(posedge clk) begin 
        if (warp_valid[0]) 
            $display("[%0t SM0-SB] scoreboard[0]=%h fu_clr_v=%b fu_clr_rd=%d", $time, scoreboard[0], fu_conflict_sb_clr_valid, fu_conflict_sb_clr_rd); 
    end 

    always @(posedge clk) begin 
        for (integer ds_s = 0; ds_s < NUM_SCHEDULERS; ds_s = ds_s + 1) begin 
            if (issue_valid_r[ds_s] && warp_writes_reg[issue_warp_r[ds_s]]) 
                $display("[%0t SM0-SCHED] Issue warp=%d pc=%h rd=%d pipe=%d", $time, issue_warp_r[ds_s], warp_inst[issue_warp_r[ds_s]*INST_WIDTH +: INST_WIDTH], warp_rd[issue_warp_r[ds_s]*5 +: 5], issue_pipe_r[ds_s]); 
        end 
    end 

endmodule
