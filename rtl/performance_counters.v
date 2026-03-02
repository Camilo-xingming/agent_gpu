//============================================================================
// RalphGPU - Comprehensive Performance Counters
// Per-SM throughput, memory stalls, and detailed metrics
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module performance_counters #(
    parameter NUM_SM        = `NUM_SM,
    parameter NUM_WARPS     = `WARPS_PER_SM,
    parameter NUM_COUNTERS  = 64,               // Total counter slots
    parameter COUNTER_WIDTH = 48                // 48-bit counters
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Control Interface
    //------------------------------------------------------------------------
    input  wire                     enable,             // Global enable
    input  wire                     clear,              // Clear all counters
    input  wire [5:0]               select,             // Counter select for read
    output wire [COUNTER_WIDTH-1:0] counter_value,

    //------------------------------------------------------------------------
    // Per-SM Event Inputs
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]        sm_active,
    input  wire [NUM_SM-1:0]        sm_issue_valid,     // Instruction issued
    input  wire [NUM_SM-1:0]        sm_dual_issue,      // Dual issue occurred
    input  wire [NUM_SM-1:0]        sm_stall_scoreboard,
    input  wire [NUM_SM-1:0]        sm_stall_ifetch,
    input  wire [NUM_SM-1:0]        sm_stall_mem,
    input  wire [NUM_SM-1:0]        sm_stall_sync,
    input  wire [NUM_SM-1:0]        sm_stall_other,

    //------------------------------------------------------------------------
    // Functional Unit Events (aggregated)
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]        fu_alu_active,
    input  wire [NUM_SM-1:0]        fu_fpu_active,
    input  wire [NUM_SM-1:0]        fu_sfu_active,
    input  wire [NUM_SM-1:0]        fu_tensor_active,
    input  wire [NUM_SM-1:0]        fu_ldst_active,

    //------------------------------------------------------------------------
    // Memory System Events
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]        l1_hit,
    input  wire [NUM_SM-1:0]        l1_miss,
    input  wire                     l2_hit,
    input  wire                     l2_miss,
    input  wire                     dram_access,

    //------------------------------------------------------------------------
    // Warp Events (per SM, packed)
    //------------------------------------------------------------------------
    input  wire [NUM_SM*NUM_WARPS-1:0] warp_issued,
    input  wire [NUM_SM*NUM_WARPS-1:0] warp_active,
    input  wire [NUM_SM*NUM_WARPS-1:0] warp_stalled,
    input  wire [NUM_SM*NUM_WARPS-1:0] warp_diverged,

    //------------------------------------------------------------------------
    // Branch Events
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]        branch_taken,
    input  wire [NUM_SM-1:0]        branch_divergent,
    input  wire [NUM_SM-1:0]        branch_reconverge,

    //------------------------------------------------------------------------
    // Tensor Core Events
    //------------------------------------------------------------------------
    input  wire                     tensor_mma_issued,
    input  wire                     tensor_mma_completed,
    input  wire [15:0]              tensor_flops,       // FLOPs this cycle

    //------------------------------------------------------------------------
    // Occupancy Metrics
    //------------------------------------------------------------------------
    output wire [NUM_SM*8-1:0] sm_occupancy,  // % of max warps
    output wire [31:0]              achieved_ipc,               // Instructions/cycle * 100
    output wire [31:0]              memory_throughput,          // GB/s * 100

    //------------------------------------------------------------------------
    // Summary Statistics
    //------------------------------------------------------------------------
    output wire [63:0]              total_instructions,
    output wire [63:0]              total_cycles,
    output wire [63:0]              total_memory_bytes
);

    //------------------------------------------------------------------------
    // Counter Index Definitions
    //------------------------------------------------------------------------
    localparam CTR_CYCLES           = 6'd0;
    localparam CTR_INSTRUCTIONS     = 6'd1;
    localparam CTR_DUAL_ISSUED      = 6'd2;
    localparam CTR_STALL_SCOREBOARD = 6'd3;
    localparam CTR_STALL_IFETCH     = 6'd4;
    localparam CTR_STALL_MEM        = 6'd5;
    localparam CTR_STALL_SYNC       = 6'd6;
    localparam CTR_STALL_OTHER      = 6'd7;

    localparam CTR_ALU_CYCLES       = 6'd8;
    localparam CTR_FPU_CYCLES       = 6'd9;
    localparam CTR_SFU_CYCLES       = 6'd10;
    localparam CTR_TENSOR_CYCLES    = 6'd11;
    localparam CTR_LDST_CYCLES      = 6'd12;

    localparam CTR_L1_HITS          = 6'd16;
    localparam CTR_L1_MISSES        = 6'd17;
    localparam CTR_L2_HITS          = 6'd18;
    localparam CTR_L2_MISSES        = 6'd19;
    localparam CTR_DRAM_ACCESSES    = 6'd20;

    localparam CTR_BRANCH_TAKEN     = 6'd24;
    localparam CTR_BRANCH_DIVERGENT = 6'd25;
    localparam CTR_BRANCH_RECONVERGE= 6'd26;

    localparam CTR_TENSOR_MMA       = 6'd28;
    localparam CTR_TENSOR_FLOPS     = 6'd29;

    localparam CTR_WARP_ISSUED      = 6'd32;
    localparam CTR_WARP_STALLED     = 6'd33;
    localparam CTR_WARP_DIVERGED    = 6'd34;

    // Per-SM counters: CTR_SM0_xxx = 40 + sm_id
    localparam CTR_SM_BASE          = 6'd40;

    //------------------------------------------------------------------------
    // Counter Storage
    //------------------------------------------------------------------------
    reg [COUNTER_WIDTH-1:0] counters [0:NUM_COUNTERS-1];

    //------------------------------------------------------------------------
    // Event Aggregation
    //------------------------------------------------------------------------
    // Sum SM-level events
    function [15:0] count_ones;
        input [NUM_SM-1:0] vec;
        integer i;
        begin
            count_ones = 0;
            for (i = 0; i < NUM_SM; i = i + 1) begin
                count_ones = count_ones + {15'b0, vec[i]};
            end
        end
    endfunction

    function [15:0] count_warp_ones;
        input [NUM_SM*NUM_WARPS-1:0] vec;
        integer i;
        begin
            count_warp_ones = 0;
            for (i = 0; i < NUM_SM*NUM_WARPS; i = i + 1) begin
                count_warp_ones = count_warp_ones + {15'b0, vec[i]};
            end
        end
    endfunction

    wire [15:0] total_issue = count_ones(sm_issue_valid);
    wire [15:0] total_dual = count_ones(sm_dual_issue);
    wire [15:0] total_stall_sb = count_ones(sm_stall_scoreboard);
    wire [15:0] total_stall_if = count_ones(sm_stall_ifetch);
    wire [15:0] total_stall_mem = count_ones(sm_stall_mem);
    wire [15:0] total_stall_sync = count_ones(sm_stall_sync);
    wire [15:0] total_stall_other = count_ones(sm_stall_other);

    wire [15:0] total_alu = count_ones(fu_alu_active);
    wire [15:0] total_fpu = count_ones(fu_fpu_active);
    wire [15:0] total_sfu = count_ones(fu_sfu_active);
    wire [15:0] total_tensor = count_ones(fu_tensor_active);
    wire [15:0] total_ldst = count_ones(fu_ldst_active);

    wire [15:0] total_l1_hit = count_ones(l1_hit);
    wire [15:0] total_l1_miss = count_ones(l1_miss);

    wire [15:0] total_br_taken = count_ones(branch_taken);
    wire [15:0] total_br_div = count_ones(branch_divergent);
    wire [15:0] total_br_reconv = count_ones(branch_reconverge);

    wire [15:0] total_warp_issue = count_warp_ones(warp_issued);
    wire [15:0] total_warp_stall = count_warp_ones(warp_stalled);
    wire [15:0] total_warp_div = count_warp_ones(warp_diverged);

    //------------------------------------------------------------------------
    // Counter Update Logic
    //------------------------------------------------------------------------
    integer rst_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            for (rst_i = 0; rst_i < NUM_COUNTERS; rst_i = rst_i + 1) begin
                counters[rst_i] <= 0;
            end
        end else if (enable) begin
            // Safe width expansions: narrow per-cycle tallies into 64-bit counters.
            // Keep arithmetic unchanged; waive noisy WIDTHEXPAND in this accumulation block.
            /* verilator lint_off WIDTHEXPAND */
            // Core counters
            counters[CTR_CYCLES] <= counters[CTR_CYCLES] + 1;
            counters[CTR_INSTRUCTIONS] <= counters[CTR_INSTRUCTIONS] + total_issue;
            counters[CTR_DUAL_ISSUED] <= counters[CTR_DUAL_ISSUED] + total_dual;

            // Stall counters
            counters[CTR_STALL_SCOREBOARD] <= counters[CTR_STALL_SCOREBOARD] + total_stall_sb;
            counters[CTR_STALL_IFETCH] <= counters[CTR_STALL_IFETCH] + total_stall_if;
            counters[CTR_STALL_MEM] <= counters[CTR_STALL_MEM] + total_stall_mem;
            counters[CTR_STALL_SYNC] <= counters[CTR_STALL_SYNC] + total_stall_sync;
            counters[CTR_STALL_OTHER] <= counters[CTR_STALL_OTHER] + total_stall_other;

            // FU utilization
            counters[CTR_ALU_CYCLES] <= counters[CTR_ALU_CYCLES] + total_alu;
            counters[CTR_FPU_CYCLES] <= counters[CTR_FPU_CYCLES] + total_fpu;
            counters[CTR_SFU_CYCLES] <= counters[CTR_SFU_CYCLES] + total_sfu;
            counters[CTR_TENSOR_CYCLES] <= counters[CTR_TENSOR_CYCLES] + total_tensor;
            counters[CTR_LDST_CYCLES] <= counters[CTR_LDST_CYCLES] + total_ldst;

            // Memory counters
            counters[CTR_L1_HITS] <= counters[CTR_L1_HITS] + total_l1_hit;
            counters[CTR_L1_MISSES] <= counters[CTR_L1_MISSES] + total_l1_miss;
            counters[CTR_L2_HITS] <= counters[CTR_L2_HITS] + l2_hit;
            counters[CTR_L2_MISSES] <= counters[CTR_L2_MISSES] + l2_miss;
            counters[CTR_DRAM_ACCESSES] <= counters[CTR_DRAM_ACCESSES] + dram_access;

            // Branch counters
            counters[CTR_BRANCH_TAKEN] <= counters[CTR_BRANCH_TAKEN] + total_br_taken;
            counters[CTR_BRANCH_DIVERGENT] <= counters[CTR_BRANCH_DIVERGENT] + total_br_div;
            counters[CTR_BRANCH_RECONVERGE] <= counters[CTR_BRANCH_RECONVERGE] + total_br_reconv;

            // Tensor counters
            counters[CTR_TENSOR_MMA] <= counters[CTR_TENSOR_MMA] + tensor_mma_issued;
            counters[CTR_TENSOR_FLOPS] <= counters[CTR_TENSOR_FLOPS] + tensor_flops;

            // Warp counters
            counters[CTR_WARP_ISSUED] <= counters[CTR_WARP_ISSUED] + total_warp_issue;
            counters[CTR_WARP_STALLED] <= counters[CTR_WARP_STALLED] + total_warp_stall;
            counters[CTR_WARP_DIVERGED] <= counters[CTR_WARP_DIVERGED] + total_warp_div;

            // Per-SM instruction counts
            for (rst_i = 0; rst_i < NUM_SM && rst_i < (NUM_COUNTERS - CTR_SM_BASE); rst_i = rst_i + 1) begin
                counters[CTR_SM_BASE + rst_i] <= counters[CTR_SM_BASE + rst_i] + sm_issue_valid[rst_i];
            end
            /* verilator lint_on WIDTHEXPAND */
        end
    end

    //------------------------------------------------------------------------
    // Counter Read Interface
    //------------------------------------------------------------------------
    assign counter_value = counters[select];

    //------------------------------------------------------------------------
    // Derived Metrics
    //------------------------------------------------------------------------
    // Occupancy: active warps / max warps per SM
    genvar sm;
    generate
        for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : gen_occ
            // Count active warps for this SM
            reg [7:0] active_warps;
            integer w;
            always @(*) begin
                active_warps = 0;
                for (w = 0; w < NUM_WARPS; w = w + 1) begin
                    if (warp_active[sm*NUM_WARPS + w]) begin
                        active_warps = active_warps + 1;
                    end
                end
            end
            assign sm_occupancy[sm*8 +: 8] = (active_warps * 100) / NUM_WARPS;
        end
    endgenerate

    // IPC calculation (instructions / cycles * 100)
    wire [63:0] instructions = {16'b0, counters[CTR_INSTRUCTIONS]};
    wire [63:0] cycles = {16'b0, counters[CTR_CYCLES]};
    wire [63:0] ipc_raw = (cycles > 0) ? ((instructions * 100) / cycles) : 64'd0;
    assign achieved_ipc = ipc_raw[31:0];

    // Memory throughput (bytes/cycle * 100, assuming 128B cache lines)
    wire [63:0] mem_accesses = {16'b0, counters[CTR_L1_HITS]} + {16'b0, counters[CTR_L1_MISSES]};
    wire [63:0] mem_bytes = mem_accesses * 128;  // 128B per access
    wire [63:0] throughput_raw = (cycles > 0) ? ((mem_bytes * 100) / cycles) : 64'd0;
    assign memory_throughput = throughput_raw[31:0];

    //------------------------------------------------------------------------
    // Summary Outputs
    //------------------------------------------------------------------------
    assign total_instructions = instructions;
    assign total_cycles = cycles;
    assign total_memory_bytes = mem_bytes;

endmodule


//============================================================================
// Per-SM Performance Monitor
// Detailed per-SM metrics for profiling
//============================================================================
module sm_performance_monitor #(
    parameter SM_ID         = 0,
    parameter NUM_WARPS     = `WARPS_PER_SM,
    parameter SAMPLE_INTERVAL = 1000    // Sample every N cycles
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // SM Status Inputs
    //------------------------------------------------------------------------
    input  wire                     sm_active,
    input  wire [NUM_WARPS-1:0]     warp_valid,
    input  wire [NUM_WARPS-1:0]     warp_ready,
    input  wire [NUM_WARPS-1:0]     warp_issued,
    input  wire [NUM_WARPS-1:0]     warp_stalled_mem,
    input  wire [NUM_WARPS-1:0]     warp_stalled_sync,
    input  wire [NUM_WARPS-1:0]     warp_stalled_fu,

    //------------------------------------------------------------------------
    // Pipeline Inputs
    //------------------------------------------------------------------------
    input  wire                     issue_valid,
    input  wire                     dual_issue,
    input  wire                     stall_ifetch,
    input  wire                     stall_decode,
    input  wire                     stall_issue,

    //------------------------------------------------------------------------
    // FU Utilization
    //------------------------------------------------------------------------
    input  wire                     alu_busy,
    input  wire                     fpu_busy,
    input  wire                     sfu_busy,
    input  wire                     tensor_busy,
    input  wire                     ldst_busy,

    //------------------------------------------------------------------------
    // Memory Events
    //------------------------------------------------------------------------
    input  wire                     l1_access,
    input  wire                     l1_hit,
    input  wire                     smem_access,
    input  wire                     smem_bank_conflict,

    //------------------------------------------------------------------------
    // Output Metrics
    //------------------------------------------------------------------------
    output wire [31:0]              ipc_sampled,        // IPC * 1000
    output wire [31:0]              issue_efficiency,   // % of cycles with issue
    output wire [31:0]              warp_occupancy,     // % of max warps active
    output wire [31:0]              mem_stall_rate,     // % cycles stalled on memory
    output wire [31:0]              l1_hit_rate,        // % L1 hits
    output wire [31:0]              fu_utilization      // % FU busy
);

    //------------------------------------------------------------------------
    // Interval Counters
    //------------------------------------------------------------------------
    reg [31:0] cycle_counter;
    reg [31:0] issue_counter;
    reg [31:0] dual_issue_counter;
    reg [31:0] stall_mem_counter;
    reg [31:0] l1_access_counter;
    reg [31:0] l1_hit_counter;
    reg [31:0] fu_busy_counter;

    reg [31:0] active_warp_sum;

    // Sampled outputs
    reg [31:0] ipc_sampled_r;
    reg [31:0] issue_eff_r;
    reg [31:0] warp_occ_r;
    reg [31:0] mem_stall_r;
    reg [31:0] l1_hit_r;
    reg [31:0] fu_util_r;

    //------------------------------------------------------------------------
    // Active Warp Count
    //------------------------------------------------------------------------
    function [4:0] count_active_warps;
        input [NUM_WARPS-1:0] valid;
        input [NUM_WARPS-1:0] ready;
        integer i;
        begin
            count_active_warps = 0;
            for (i = 0; i < NUM_WARPS; i = i + 1) begin
                if (valid[i] && ready[i]) begin
                    count_active_warps = count_active_warps + 1;
                end
            end
        end
    endfunction

    wire [4:0] active_warps = count_active_warps(warp_valid, warp_ready);

    //------------------------------------------------------------------------
    // FU Utilization
    //------------------------------------------------------------------------
    wire any_fu_busy = alu_busy | fpu_busy | sfu_busy | tensor_busy | ldst_busy;

    //------------------------------------------------------------------------
    // Memory Stall Detection
    //------------------------------------------------------------------------
    wire mem_stall = |warp_stalled_mem;

    //------------------------------------------------------------------------
    // Counter Update and Sampling
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 0;
            issue_counter <= 0;
            dual_issue_counter <= 0;
            stall_mem_counter <= 0;
            l1_access_counter <= 0;
            l1_hit_counter <= 0;
            fu_busy_counter <= 0;
            active_warp_sum <= 0;
            ipc_sampled_r <= 0;
            issue_eff_r <= 0;
            warp_occ_r <= 0;
            mem_stall_r <= 0;
            l1_hit_r <= 0;
            fu_util_r <= 0;
        end else if (sm_active) begin
            cycle_counter <= cycle_counter + 1;

            if (issue_valid) begin
                issue_counter <= issue_counter + 1;
                if (dual_issue) begin
                    dual_issue_counter <= dual_issue_counter + 1;
                end
            end

            if (mem_stall) begin
                stall_mem_counter <= stall_mem_counter + 1;
            end

            if (l1_access) begin
                l1_access_counter <= l1_access_counter + 1;
                if (l1_hit) begin
                    l1_hit_counter <= l1_hit_counter + 1;
                end
            end

            if (any_fu_busy) begin
                fu_busy_counter <= fu_busy_counter + 1;
            end

            active_warp_sum <= active_warp_sum + {{27{1'b0}}, active_warps};

            // Sample at interval
            if (cycle_counter >= SAMPLE_INTERVAL - 1) begin
                // IPC = (issue + dual_issue) / cycles * 1000
                ipc_sampled_r <= ((issue_counter + dual_issue_counter) * 1000) / SAMPLE_INTERVAL;

                // Issue efficiency = cycles_with_issue / cycles * 100
                issue_eff_r <= (issue_counter * 100) / SAMPLE_INTERVAL;

                // Warp occupancy = avg_active_warps / max_warps * 100
                warp_occ_r <= (active_warp_sum * 100) / (SAMPLE_INTERVAL * NUM_WARPS);

                // Memory stall rate
                mem_stall_r <= (stall_mem_counter * 100) / SAMPLE_INTERVAL;

                // L1 hit rate
                l1_hit_r <= (l1_access_counter > 0) ?
                           ((l1_hit_counter * 100) / l1_access_counter) : 0;

                // FU utilization
                fu_util_r <= (fu_busy_counter * 100) / SAMPLE_INTERVAL;

                // Reset interval counters
                cycle_counter <= 0;
                issue_counter <= 0;
                dual_issue_counter <= 0;
                stall_mem_counter <= 0;
                l1_access_counter <= 0;
                l1_hit_counter <= 0;
                fu_busy_counter <= 0;
                active_warp_sum <= 0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Outputs
    //------------------------------------------------------------------------
    assign ipc_sampled = ipc_sampled_r;
    assign issue_efficiency = issue_eff_r;
    assign warp_occupancy = warp_occ_r;
    assign mem_stall_rate = mem_stall_r;
    assign l1_hit_rate = l1_hit_r;
    assign fu_utilization = fu_util_r;

endmodule
