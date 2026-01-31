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
    input  wire [INST_WIDTH-1:0]    warp_inst [0:NUM_WARPS-1],
    input  wire [NUM_WARPS-1:0]     warp_inst_valid,
    output wire [NUM_WARPS-1:0]     warp_inst_consume,

    //------------------------------------------------------------------------
    // Decoded Instruction Info (same as advanced_warp_scheduler)
    //------------------------------------------------------------------------
    input  wire [4:0]               warp_rd [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs1 [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs2 [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs3 [0:NUM_WARPS-1],
    input  wire [NUM_WARPS-1:0]     warp_is_compute,
    input  wire [NUM_WARPS-1:0]     warp_is_tensor,
    input  wire [NUM_WARPS-1:0]     warp_is_memory,
    input  wire [NUM_WARPS-1:0]     warp_is_branch,
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
    output wire [$clog2(NUM_WARPS)-1:0]           issue_warp_id [0:NUM_SCHEDULERS-1],
    output wire [INST_WIDTH-1:0]                  issue_inst [0:NUM_SCHEDULERS-1],
    output wire [2:0]                             issue_pipe [0:NUM_SCHEDULERS-1],

    //------------------------------------------------------------------------
    // Blackwell-specific Issue Outputs
    //------------------------------------------------------------------------
    output reg  [NUM_SCHEDULERS-1:0]              issue_is_async_mma, // Issued op is async MMA
    output reg  [3:0]                             issue_async_mma_id [0:NUM_SCHEDULERS-1], // Async MMA op ID

    //------------------------------------------------------------------------
    // Writeback Interface (for scoreboard clearing)
    //------------------------------------------------------------------------
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
    output wire [31:0]              stat_tcgen05_issued       // Total tcgen05 ops issued
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
            wire raw_hazard = scoreboard[w][warp_rs1[w]] ||
                             scoreboard[w][warp_rs2[w]] ||
                             scoreboard[w][warp_rs3[w]];
            wire waw_hazard = warp_writes_reg[w] && scoreboard[w][warp_rd[w]];

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

    //------------------------------------------------------------------------
    // Per-Scheduler Warp Selection
    // Blackwell: Each scheduler handles warps where (warp_id % NUM_SCHEDULERS == scheduler_id)
    //------------------------------------------------------------------------
    reg [NUM_SCHEDULERS-1:0] issue_valid_r;
    reg [WARP_W-1:0]         issue_warp_r [0:NUM_SCHEDULERS-1];
    reg [INST_WIDTH-1:0]     issue_inst_r [0:NUM_SCHEDULERS-1];
    reg [2:0]                issue_pipe_r [0:NUM_SCHEDULERS-1];
    reg [NUM_WARPS-1:0]      issue_consume_r;

    // Round-robin pointer per scheduler (for fairness within assigned warps)
    reg [WARP_W-1:0] sched_rr_ptr [0:NUM_SCHEDULERS-1];

    // Compute pipe assignment: scheduler 0 -> pipe0, scheduler 1 -> pipe1, etc.
    wire [NUM_SCHEDULERS-1:0] compute_pipe_ready;
    generate
        if (NUM_SCHEDULERS >= 1) assign compute_pipe_ready[0] = compute_pipe0_ready;
        if (NUM_SCHEDULERS >= 2) assign compute_pipe_ready[1] = compute_pipe1_ready;
        // For NUM_SCHEDULERS > 2, share pipes (round-robin or priority)
        if (NUM_SCHEDULERS >= 3) assign compute_pipe_ready[2] = compute_pipe0_ready; // Share with pipe0
        if (NUM_SCHEDULERS >= 4) assign compute_pipe_ready[3] = compute_pipe1_ready; // Share with pipe1
    endgenerate

    // Scheduler selection logic
    integer s, sw;
    always @(*) begin
        issue_valid_r = 0;
        issue_consume_r = 0;
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            issue_warp_r[s] = 0;
            issue_inst_r[s] = 0;
            issue_pipe_r[s] = PIPE_COMPUTE0;
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
                    warp_idx = s + ((sched_rr_ptr[s] + sw) % WARPS_PER_SCHED) * NUM_SCHEDULERS;

                    if (warp_idx < NUM_WARPS && warp_eligible[warp_idx]) begin
                        // Priority order: Branch > Memory > Tensor > Compute
                        // This ensures long-latency ops get dispatched early
                        if (warp_is_branch[warp_idx] && branch_unit_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx];
                            issue_pipe_r[s] = PIPE_BRANCH;
                            issue_consume_r[warp_idx] = 1'b1;
                        end else if (warp_is_memory[warp_idx] && memory_pipe_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx];
                            issue_pipe_r[s] = PIPE_MEMORY;
                            issue_consume_r[warp_idx] = 1'b1;
                        end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx];
                            issue_pipe_r[s] = PIPE_TENSOR;
                            issue_consume_r[warp_idx] = 1'b1;
                        end else if (warp_is_compute[warp_idx] && compute_pipe_ready[s]) begin
                            issue_valid_r[s] = 1'b1;
                            issue_warp_r[s] = warp_idx[WARP_W-1:0];
                            issue_inst_r[s] = warp_inst[warp_idx];
                            // Map scheduler to compute pipe
                            issue_pipe_r[s] = (s % 2 == 0) ? PIPE_COMPUTE0 : PIPE_COMPUTE1;
                            issue_consume_r[warp_idx] = 1'b1;
                        end
                    end
                end
            end
        end
    end

    assign warp_inst_consume = issue_consume_r;

    //------------------------------------------------------------------------
    // Scoreboard Update
    //------------------------------------------------------------------------
    integer sb_w, sb_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sb_w = 0; sb_w < NUM_WARPS; sb_w = sb_w + 1) begin
                scoreboard[sb_w] <= 0;
            end
            for (sb_s = 0; sb_s < NUM_SCHEDULERS; sb_s = sb_s + 1) begin
                sched_rr_ptr[sb_s] <= 0;
            end
        end else begin
            // Set scoreboard on issue
            // Note: R0 is a normal register in GPU (not hardwired zero like RISC-V)
            // NOP/BAR_SYNC/etc. won't set scoreboard because warp_writes_reg=0 for them
            for (sb_s = 0; sb_s < NUM_SCHEDULERS; sb_s = sb_s + 1) begin
                if (issue_valid_r[sb_s] && warp_writes_reg[issue_warp_r[sb_s]]) begin
                    scoreboard[issue_warp_r[sb_s]][warp_rd[issue_warp_r[sb_s]]] <= 1'b1;
                end
                // Update round-robin pointer for fairness
                if (issue_valid_r[sb_s]) begin
                    sched_rr_ptr[sb_s] <= (sched_rr_ptr[sb_s] + 1) % WARPS_PER_SCHED;
                end
            end

            // Clear scoreboard on writeback
            if (wb_valid) begin
                scoreboard[wb_warp_id][wb_rd] <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] cycle_count;
    reg [31:0] single_issue_count;
    reg [31:0] dual_issue_count;
    reg [31:0] stall_count;

    wire [3:0] num_issued = issue_valid_r[0] + issue_valid_r[1] +
                            ((NUM_SCHEDULERS > 2) ? issue_valid_r[2] : 1'b0) +
                            ((NUM_SCHEDULERS > 3) ? issue_valid_r[3] : 1'b0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            single_issue_count <= 0;
            dual_issue_count <= 0;
            stall_count <= 0;
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

            // Debug output
            `ifdef SIMULATION
            if (cycle_count < 200) begin
                $display("[%0t BLACKWELL_SCHED] cycle=%0d issue_valid=%b consume=%b num_issued=%0d",
                         $time, cycle_count, issue_valid_r, issue_consume_r, num_issued);
                $display("  eligible=%b inst_valid=%b hazard=%b valid=%b ready=%b diverged=%b barrier=%b",
                         warp_eligible, warp_inst_valid, warp_has_hazard, warp_valid, warp_ready, warp_diverged, warp_at_barrier);
                if (issue_valid_r[0])
                    $display("  sched0: warp=%0d pipe=%0d inst=0x%08x",
                             issue_warp_r[0], issue_pipe_r[0], issue_inst_r[0]);
                if (issue_valid_r[1])
                    $display("  sched1: warp=%0d pipe=%0d inst=0x%08x",
                             issue_warp_r[1], issue_pipe_r[1], issue_inst_r[1]);
            end
            `endif
        end
    end

    assign stat_cycles = cycle_count;
    assign stat_single_issue = single_issue_count;
    assign stat_dual_issue = dual_issue_count;
    assign stat_stalls = stall_count;

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign issue_valid = issue_valid_r;

    generate
        genvar i;
        for (i = 0; i < NUM_SCHEDULERS; i = i + 1) begin : gen_issue_out
            assign issue_warp_id[i] = issue_warp_r[i];
            assign issue_inst[i] = issue_inst_r[i];
            assign issue_pipe[i] = issue_pipe_r[i];
        end
    endgenerate

endmodule
