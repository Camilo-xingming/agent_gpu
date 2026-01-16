//============================================================================
// RalphGPU - Advanced Warp Scheduler
// Split schedulers: Compute (ALU/FPU) + Tensor + Memory
// Supports dual-issue and multi-warp scheduling
//============================================================================

`include "gpu_defines.vh"

module advanced_warp_scheduler #(
    parameter NUM_WARPS     = 8,
    parameter INST_WIDTH    = 32,
    parameter NUM_ISSUE     = 2,                // Dual-issue
    parameter SCOREBOARD_DEPTH = 16             // Outstanding instructions tracked
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Warp Status Interface
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]     warp_valid,
    input  wire [NUM_WARPS-1:0]     warp_ready,         // Not stalled
    input  wire [NUM_WARPS-1:0]     warp_diverged,      // In divergent execution
    input  wire [NUM_WARPS-1:0]     warp_at_barrier,

    //------------------------------------------------------------------------
    // Instruction Buffer Interface (per warp)
    //------------------------------------------------------------------------
    input  wire [INST_WIDTH-1:0]    warp_inst [0:NUM_WARPS-1],
    input  wire [NUM_WARPS-1:0]     warp_inst_valid,
    output wire [NUM_WARPS-1:0]     warp_inst_consume,

    //------------------------------------------------------------------------
    // Decoded Instruction Info (for scheduling decisions)
    //------------------------------------------------------------------------
    input  wire [4:0]               warp_rd [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs1 [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs2 [0:NUM_WARPS-1],
    input  wire [4:0]               warp_rs3 [0:NUM_WARPS-1],
    input  wire [NUM_WARPS-1:0]     warp_is_compute,    // ALU/FPU/SFU
    input  wire [NUM_WARPS-1:0]     warp_is_tensor,     // Tensor Core
    input  wire [NUM_WARPS-1:0]     warp_is_memory,     // Load/Store
    input  wire [NUM_WARPS-1:0]     warp_is_branch,     // Control flow
    input  wire [NUM_WARPS-1:0]     warp_writes_reg,

    //------------------------------------------------------------------------
    // Execution Unit Availability
    //------------------------------------------------------------------------
    input  wire                     compute_pipe0_ready,
    input  wire                     compute_pipe1_ready,
    input  wire                     tensor_pipe_ready,
    input  wire                     memory_pipe_ready,
    input  wire                     branch_unit_ready,

    //------------------------------------------------------------------------
    // Issue Outputs
    //------------------------------------------------------------------------
    output wire [NUM_ISSUE-1:0]     issue_valid,
    output wire [$clog2(NUM_WARPS)-1:0] issue_warp_id [0:NUM_ISSUE-1],
    output wire [INST_WIDTH-1:0]    issue_inst [0:NUM_ISSUE-1],
    output wire [2:0]               issue_pipe [0:NUM_ISSUE-1],   // 0=compute0, 1=compute1, 2=tensor, 3=memory, 4=branch

    //------------------------------------------------------------------------
    // Scoreboard Interface (dependency tracking)
    //------------------------------------------------------------------------
    input  wire                     wb_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] wb_warp_id,
    input  wire [4:0]               wb_rd,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output wire [31:0]              stat_cycles,
    output wire [31:0]              stat_single_issue,
    output wire [31:0]              stat_dual_issue,
    output wire [31:0]              stat_stalls
);

    localparam WARP_W = $clog2(NUM_WARPS);

    //------------------------------------------------------------------------
    // Execution Pipe Encoding
    //------------------------------------------------------------------------
    localparam PIPE_COMPUTE0 = 3'd0;
    localparam PIPE_COMPUTE1 = 3'd1;
    localparam PIPE_TENSOR   = 3'd2;
    localparam PIPE_MEMORY   = 3'd3;
    localparam PIPE_BRANCH   = 3'd4;

    //------------------------------------------------------------------------
    // Scoreboard - Track in-flight register writes
    //------------------------------------------------------------------------
    // Per-warp scoreboard: which registers have pending writes
    reg [31:0] scoreboard [0:NUM_WARPS-1];  // Bit per register

    // Check RAW hazard
    function check_raw_hazard;
        input [WARP_W-1:0] warp_id;
        input [4:0] rs1;
        input [4:0] rs2;
        input [4:0] rs3;
        begin
            check_raw_hazard = scoreboard[warp_id][rs1] ||
                              scoreboard[warp_id][rs2] ||
                              scoreboard[warp_id][rs3];
        end
    endfunction

    // Check WAW hazard
    function check_waw_hazard;
        input [WARP_W-1:0] warp_id;
        input [4:0] rd;
        begin
            check_waw_hazard = scoreboard[warp_id][rd];
        end
    endfunction

    //------------------------------------------------------------------------
    // Warp Scheduling Priority
    //------------------------------------------------------------------------
    // Greedy-then-oldest (GTO) scheduler with split priorities
    reg [WARP_W-1:0] compute_priority [0:NUM_WARPS-1];
    reg [WARP_W-1:0] tensor_priority [0:NUM_WARPS-1];
    reg [WARP_W-1:0] memory_priority [0:NUM_WARPS-1];

    // Round-robin pointers for each scheduler
    reg [WARP_W-1:0] compute_rr_ptr;
    reg [WARP_W-1:0] tensor_rr_ptr;
    reg [WARP_W-1:0] memory_rr_ptr;

    //------------------------------------------------------------------------
    // Eligible Warp Detection
    //------------------------------------------------------------------------
    wire [NUM_WARPS-1:0] warp_schedulable = warp_valid & warp_ready & warp_inst_valid;

    // Per-warp hazard check
    wire [NUM_WARPS-1:0] warp_has_hazard;
    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : gen_hazard
            assign warp_has_hazard[w] =
                check_raw_hazard(w[WARP_W-1:0], warp_rs1[w], warp_rs2[w], warp_rs3[w]) ||
                (warp_writes_reg[w] && check_waw_hazard(w[WARP_W-1:0], warp_rd[w]));
        end
    endgenerate

    wire [NUM_WARPS-1:0] warp_eligible = warp_schedulable & ~warp_has_hazard;

    // Split by instruction type
    wire [NUM_WARPS-1:0] compute_eligible = warp_eligible & warp_is_compute;
    wire [NUM_WARPS-1:0] tensor_eligible  = warp_eligible & warp_is_tensor;
    wire [NUM_WARPS-1:0] memory_eligible  = warp_eligible & warp_is_memory;
    wire [NUM_WARPS-1:0] branch_eligible  = warp_eligible & warp_is_branch;

    //------------------------------------------------------------------------
    // Scheduler Selection Logic
    //------------------------------------------------------------------------
    reg [WARP_W-1:0] selected_compute0;
    reg [WARP_W-1:0] selected_compute1;
    reg [WARP_W-1:0] selected_tensor;
    reg [WARP_W-1:0] selected_memory;
    reg [WARP_W-1:0] selected_branch;

    reg found_compute0, found_compute1, found_tensor, found_memory, found_branch;

    // Round-robin selection for each category
    integer sel_i, sel_idx;
    always @(*) begin
        selected_compute0 = 0;
        selected_compute1 = 0;
        selected_tensor = 0;
        selected_memory = 0;
        selected_branch = 0;
        found_compute0 = 0;
        found_compute1 = 0;
        found_tensor = 0;
        found_memory = 0;
        found_branch = 0;

        // Compute scheduler 0 (starts from compute_rr_ptr)
        for (sel_i = 0; sel_i < NUM_WARPS; sel_i = sel_i + 1) begin
            sel_idx = (compute_rr_ptr + sel_i) % NUM_WARPS;
            if (compute_eligible[sel_idx] && !found_compute0) begin
                selected_compute0 = sel_idx[WARP_W-1:0];
                found_compute0 = 1;
            end else if (compute_eligible[sel_idx] && !found_compute1 && sel_idx != selected_compute0) begin
                selected_compute1 = sel_idx[WARP_W-1:0];
                found_compute1 = 1;
            end
        end

        // Tensor scheduler
        for (sel_i = 0; sel_i < NUM_WARPS; sel_i = sel_i + 1) begin
            sel_idx = (tensor_rr_ptr + sel_i) % NUM_WARPS;
            if (tensor_eligible[sel_idx] && !found_tensor) begin
                selected_tensor = sel_idx[WARP_W-1:0];
                found_tensor = 1;
            end
        end

        // Memory scheduler
        for (sel_i = 0; sel_i < NUM_WARPS; sel_i = sel_i + 1) begin
            sel_idx = (memory_rr_ptr + sel_i) % NUM_WARPS;
            if (memory_eligible[sel_idx] && !found_memory) begin
                selected_memory = sel_idx[WARP_W-1:0];
                found_memory = 1;
            end
        end

        // Branch scheduler (single issue, prioritize)
        for (sel_i = 0; sel_i < NUM_WARPS; sel_i = sel_i + 1) begin
            if (branch_eligible[sel_i] && !found_branch) begin
                selected_branch = sel_i[WARP_W-1:0];
                found_branch = 1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Issue Arbitration - Select 2 non-conflicting instructions
    //------------------------------------------------------------------------
    // Priority: 1) Branch (critical), 2) Memory (latency hiding), 3) Tensor, 4) Compute
    reg [NUM_ISSUE-1:0] issue_valid_r;
    reg [WARP_W-1:0]    issue_warp_r [0:NUM_ISSUE-1];
    reg [INST_WIDTH-1:0] issue_inst_r [0:NUM_ISSUE-1];
    reg [2:0]           issue_pipe_r [0:NUM_ISSUE-1];

    // Check for inter-issue dependencies
    function check_issue_conflict;
        input [WARP_W-1:0] warp_a;
        input [WARP_W-1:0] warp_b;
        input [4:0] rd_a;
        input [4:0] rs1_b, rs2_b, rs3_b;
        begin
            // Same warp conflict (can't dual-issue from same warp usually)
            // Or RAW between the two instructions
            check_issue_conflict = (warp_a == warp_b) ||
                                  (warp_writes_reg[warp_a] &&
                                   (rd_a == rs1_b || rd_a == rs2_b || rd_a == rs3_b));
        end
    endfunction

    always @(*) begin
        issue_valid_r = 0;
        issue_warp_r[0] = 0;
        issue_warp_r[1] = 0;
        issue_inst_r[0] = 0;
        issue_inst_r[1] = 0;
        issue_pipe_r[0] = PIPE_COMPUTE0;
        issue_pipe_r[1] = PIPE_COMPUTE0;

        // Slot 0 selection (highest priority first)
        if (found_branch && branch_unit_ready) begin
            issue_valid_r[0] = 1'b1;
            issue_warp_r[0] = selected_branch;
            issue_inst_r[0] = warp_inst[selected_branch];
            issue_pipe_r[0] = PIPE_BRANCH;
        end else if (found_memory && memory_pipe_ready) begin
            issue_valid_r[0] = 1'b1;
            issue_warp_r[0] = selected_memory;
            issue_inst_r[0] = warp_inst[selected_memory];
            issue_pipe_r[0] = PIPE_MEMORY;
        end else if (found_tensor && tensor_pipe_ready) begin
            issue_valid_r[0] = 1'b1;
            issue_warp_r[0] = selected_tensor;
            issue_inst_r[0] = warp_inst[selected_tensor];
            issue_pipe_r[0] = PIPE_TENSOR;
        end else if (found_compute0 && compute_pipe0_ready) begin
            issue_valid_r[0] = 1'b1;
            issue_warp_r[0] = selected_compute0;
            issue_inst_r[0] = warp_inst[selected_compute0];
            issue_pipe_r[0] = PIPE_COMPUTE0;
        end

        // Slot 1 selection (dual-issue if possible)
        if (issue_valid_r[0]) begin
            // Try to issue to a different pipe
            if (issue_pipe_r[0] != PIPE_COMPUTE0 && found_compute0 && compute_pipe0_ready &&
                !check_issue_conflict(issue_warp_r[0], selected_compute0, warp_rd[issue_warp_r[0]],
                                     warp_rs1[selected_compute0], warp_rs2[selected_compute0], warp_rs3[selected_compute0])) begin
                issue_valid_r[1] = 1'b1;
                issue_warp_r[1] = selected_compute0;
                issue_inst_r[1] = warp_inst[selected_compute0];
                issue_pipe_r[1] = PIPE_COMPUTE0;
            end else if (issue_pipe_r[0] != PIPE_COMPUTE1 && found_compute1 && compute_pipe1_ready &&
                        !check_issue_conflict(issue_warp_r[0], selected_compute1, warp_rd[issue_warp_r[0]],
                                             warp_rs1[selected_compute1], warp_rs2[selected_compute1], warp_rs3[selected_compute1])) begin
                issue_valid_r[1] = 1'b1;
                issue_warp_r[1] = selected_compute1;
                issue_inst_r[1] = warp_inst[selected_compute1];
                issue_pipe_r[1] = PIPE_COMPUTE1;
            end else if (issue_pipe_r[0] != PIPE_TENSOR && found_tensor && tensor_pipe_ready &&
                        !check_issue_conflict(issue_warp_r[0], selected_tensor, warp_rd[issue_warp_r[0]],
                                             warp_rs1[selected_tensor], warp_rs2[selected_tensor], warp_rs3[selected_tensor])) begin
                issue_valid_r[1] = 1'b1;
                issue_warp_r[1] = selected_tensor;
                issue_inst_r[1] = warp_inst[selected_tensor];
                issue_pipe_r[1] = PIPE_TENSOR;
            end else if (issue_pipe_r[0] != PIPE_MEMORY && found_memory && memory_pipe_ready &&
                        !check_issue_conflict(issue_warp_r[0], selected_memory, warp_rd[issue_warp_r[0]],
                                             warp_rs1[selected_memory], warp_rs2[selected_memory], warp_rs3[selected_memory])) begin
                issue_valid_r[1] = 1'b1;
                issue_warp_r[1] = selected_memory;
                issue_inst_r[1] = warp_inst[selected_memory];
                issue_pipe_r[1] = PIPE_MEMORY;
            end
        end
    end

    //------------------------------------------------------------------------
    // Warp Consume Signals
    //------------------------------------------------------------------------
    reg [NUM_WARPS-1:0] warp_consume_r;
    integer cons_i;
    always @(*) begin
        warp_consume_r = 0;
        for (cons_i = 0; cons_i < NUM_ISSUE; cons_i = cons_i + 1) begin
            if (issue_valid_r[cons_i]) begin
                warp_consume_r[issue_warp_r[cons_i]] = 1'b1;
            end
        end
    end

    assign warp_inst_consume = warp_consume_r;

    //------------------------------------------------------------------------
    // Scoreboard Update
    //------------------------------------------------------------------------
    integer sb_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sb_w = 0; sb_w < NUM_WARPS; sb_w = sb_w + 1) begin
                scoreboard[sb_w] <= 0;
            end
            compute_rr_ptr <= 0;
            tensor_rr_ptr <= 0;
            memory_rr_ptr <= 0;
        end else begin
            // Set scoreboard bits on issue
            for (sb_w = 0; sb_w < NUM_ISSUE; sb_w = sb_w + 1) begin
                if (issue_valid_r[sb_w] && warp_writes_reg[issue_warp_r[sb_w]]) begin
                    scoreboard[issue_warp_r[sb_w]][warp_rd[issue_warp_r[sb_w]]] <= 1'b1;
                end
            end

            // Clear scoreboard bits on writeback
            if (wb_valid) begin
                scoreboard[wb_warp_id][wb_rd] <= 1'b0;
            end

            // Update round-robin pointers
            if (issue_valid_r[0]) begin
                case (issue_pipe_r[0])
                    PIPE_COMPUTE0, PIPE_COMPUTE1: compute_rr_ptr <= (issue_warp_r[0] + 1) % NUM_WARPS;
                    PIPE_TENSOR: tensor_rr_ptr <= (issue_warp_r[0] + 1) % NUM_WARPS;
                    PIPE_MEMORY: memory_rr_ptr <= (issue_warp_r[0] + 1) % NUM_WARPS;
                endcase
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] cycle_count;
    reg [31:0] single_count;
    reg [31:0] dual_count;
    reg [31:0] stall_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            single_count <= 0;
            dual_count <= 0;
            stall_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (issue_valid_r[0] && issue_valid_r[1]) begin
                dual_count <= dual_count + 1;
            end else if (issue_valid_r[0]) begin
                single_count <= single_count + 1;
            end else begin
                stall_count <= stall_count + 1;
            end
        end
    end

    assign stat_cycles = cycle_count;
    assign stat_single_issue = single_count;
    assign stat_dual_issue = dual_count;
    assign stat_stalls = stall_count;

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign issue_valid = issue_valid_r;

    generate
        genvar i;
        for (i = 0; i < NUM_ISSUE; i = i + 1) begin : gen_issue_out
            assign issue_warp_id[i] = issue_warp_r[i];
            assign issue_inst[i] = issue_inst_r[i];
            assign issue_pipe[i] = issue_pipe_r[i];
        end
    endgenerate

endmodule


//============================================================================
// Greedy-Then-Oldest (GTO) Warp Scheduler
// Alternative scheduler optimized for throughput
//============================================================================
module gto_scheduler #(
    parameter NUM_WARPS = 8,
    parameter INST_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Warp status
    input  wire [NUM_WARPS-1:0]     warp_valid,
    input  wire [NUM_WARPS-1:0]     warp_ready,
    input  wire [NUM_WARPS-1:0]     warp_inst_valid,

    // Age tracking (cycles since last issued)
    output wire [31:0]              warp_age [0:NUM_WARPS-1],

    // Selected warp
    output wire [$clog2(NUM_WARPS)-1:0] selected_warp,
    output wire                     selected_valid,
    input  wire                     issue_ack
);

    localparam WARP_W = $clog2(NUM_WARPS);

    // Per-warp age counters
    reg [31:0] age_counter [0:NUM_WARPS-1];

    // Track currently executing warp (greedy mode)
    reg [WARP_W-1:0] greedy_warp;
    reg greedy_valid;

    wire [NUM_WARPS-1:0] eligible = warp_valid & warp_ready & warp_inst_valid;

    // Find oldest eligible warp
    reg [WARP_W-1:0] oldest_warp;
    reg [31:0] oldest_age;
    reg found_oldest;

    integer age_i;
    always @(*) begin
        oldest_warp = 0;
        oldest_age = 0;
        found_oldest = 0;

        for (age_i = 0; age_i < NUM_WARPS; age_i = age_i + 1) begin
            if (eligible[age_i] && (!found_oldest || age_counter[age_i] > oldest_age)) begin
                oldest_warp = age_i[WARP_W-1:0];
                oldest_age = age_counter[age_i];
                found_oldest = 1;
            end
        end
    end

    // Greedy-then-oldest logic
    wire use_greedy = greedy_valid && eligible[greedy_warp];
    assign selected_warp = use_greedy ? greedy_warp : oldest_warp;
    assign selected_valid = use_greedy || found_oldest;

    // Age counter and greedy warp update
    integer upd_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            greedy_warp <= 0;
            greedy_valid <= 0;
            for (upd_i = 0; upd_i < NUM_WARPS; upd_i = upd_i + 1) begin
                age_counter[upd_i] <= 0;
            end
        end else begin
            // Increment age for all eligible warps
            for (upd_i = 0; upd_i < NUM_WARPS; upd_i = upd_i + 1) begin
                if (eligible[upd_i] && upd_i != selected_warp) begin
                    age_counter[upd_i] <= age_counter[upd_i] + 1;
                end else if (issue_ack && upd_i == selected_warp) begin
                    age_counter[upd_i] <= 0;  // Reset age on issue
                end
            end

            // Update greedy warp
            if (issue_ack && selected_valid) begin
                greedy_warp <= selected_warp;
                greedy_valid <= 1'b1;
            end else if (!eligible[greedy_warp]) begin
                greedy_valid <= 1'b0;  // Greedy warp became ineligible
            end
        end
    end

    // Output age for monitoring
    generate
        genvar g;
        for (g = 0; g < NUM_WARPS; g = g + 1) begin : gen_age
            assign warp_age[g] = age_counter[g];
        end
    endgenerate

endmodule
