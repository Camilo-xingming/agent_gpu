//============================================================================
// RalphGPU - WGMMA Tile Engine with SMEM Staging
// Implements Hopper-style WGMMA dataflow optimization
// - Warpgroup-level matrix operations (128 threads)
// - Asynchronous SMEM staging
// - Software-managed tile scheduling
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module wgmma_tile_engine #(
    parameter TILE_M        = 64,               // Output tile M dimension
    parameter TILE_N        = 64,               // Output tile N dimension
    parameter TILE_K        = 16,               // Reduction tile K dimension
    parameter NUM_STAGES    = 4,                // Double/multi-buffering stages
    parameter SMEM_SIZE_KB  = 16,               // Shared memory size
    parameter WARPGROUP_SIZE = 4,               // Warps per warpgroup
    parameter THREADS_PER_WARP = 32,
    parameter DATA_WIDTH    = 16                // FP16 default
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Tile Control Interface
    //------------------------------------------------------------------------
    input  wire                     tile_start,
    input  wire [31:0]              tile_m_offset,      // Global M offset
    input  wire [31:0]              tile_n_offset,      // Global N offset
    input  wire [31:0]              k_tiles,            // Number of K tiles
    output wire                     tile_done,
    output wire                     tile_ready,

    //------------------------------------------------------------------------
    // Global Memory Interface (for prefetch)
    //------------------------------------------------------------------------
    output wire                     gmem_req_valid,
    output wire [31:0]              gmem_req_addr,
    output wire [8:0]               gmem_req_size,      // Bytes to fetch
    output wire                     gmem_req_is_a,      // 1=Matrix A, 0=Matrix B
    input  wire                     gmem_req_ready,
    input  wire [511:0]             gmem_resp_data,
    input  wire                     gmem_resp_valid,

    //------------------------------------------------------------------------
    // Shared Memory Interface (SMEM staging)
    //------------------------------------------------------------------------
    output wire                     smem_wr_en,
    output wire [13:0]              smem_wr_addr,       // 16KB addressing
    output wire [511:0]             smem_wr_data,
    output wire [63:0]              smem_wr_mask,       // Byte mask

    output wire                     smem_rd_en,
    output wire [13:0]              smem_rd_addr,
    input  wire [511:0]             smem_rd_data,
    input  wire                     smem_rd_valid,

    //------------------------------------------------------------------------
    // MMA Core Interface
    //------------------------------------------------------------------------
    output wire                     mma_valid,
    output wire [511:0]             mma_frag_a,         // Matrix A fragment
    output wire [511:0]             mma_frag_b,         // Matrix B fragment
    output wire [1023:0]            mma_accum_in,       // Accumulator input
    input  wire                     mma_ready,
    input  wire [1023:0]            mma_accum_out,      // Accumulator output
    input  wire                     mma_done,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output wire [31:0]              stat_tiles_computed,
    output wire [31:0]              stat_smem_stalls,
    output wire [31:0]              stat_mma_stalls
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam SMEM_SIZE_BYTES = SMEM_SIZE_KB * 1024;
    localparam STAGE_A_SIZE = TILE_M * TILE_K * (DATA_WIDTH / 8);   // Bytes per A stage
    localparam STAGE_B_SIZE = TILE_K * TILE_N * (DATA_WIDTH / 8);   // Bytes per B stage
    localparam STAGE_SIZE = STAGE_A_SIZE + STAGE_B_SIZE;

    localparam STAGE_PTR_W = $clog2(NUM_STAGES);
    localparam K_PTR_W = 16;

    // SMEM layout: [Stage0_A][Stage0_B][Stage1_A][Stage1_B]...
    localparam SMEM_A_BASE = 0;
    localparam SMEM_B_BASE = STAGE_A_SIZE;
    localparam SMEM_STAGE_STRIDE = STAGE_SIZE;

    //------------------------------------------------------------------------
    // Stage State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 4'd0;
    localparam ST_PREFETCH_A    = 4'd1;
    localparam ST_PREFETCH_B    = 4'd2;
    localparam ST_WAIT_PREFETCH = 4'd3;
    localparam ST_LOAD_A        = 4'd4;
    localparam ST_LOAD_B        = 4'd5;
    localparam ST_COMPUTE       = 4'd6;
    localparam ST_ADVANCE       = 4'd7;
    localparam ST_WRITEBACK     = 4'd8;
    localparam ST_DONE          = 4'd9;

    reg [3:0] state;
    reg [3:0] next_state;

    //------------------------------------------------------------------------
    // Tile Tracking
    //------------------------------------------------------------------------
    reg [K_PTR_W-1:0] k_tile_idx;
    reg [K_PTR_W-1:0] k_tiles_total;
    reg [31:0] m_offset, n_offset;

    // Stage tracking
    reg [STAGE_PTR_W-1:0] load_stage;       // Stage being loaded
    reg [STAGE_PTR_W-1:0] compute_stage;    // Stage being computed
    reg [NUM_STAGES-1:0]  stage_valid;      // Stage has valid data
    reg [NUM_STAGES-1:0]  stage_computing;  // Stage is being used for MMA

    // Prefetch tracking
    reg prefetch_a_done;
    reg prefetch_b_done;
    reg [8:0] prefetch_offset;

    //------------------------------------------------------------------------
    // Accumulator
    //------------------------------------------------------------------------
    reg [1023:0] accumulator;

    //------------------------------------------------------------------------
    // Statistics Counters
    //------------------------------------------------------------------------
    reg [31:0] tiles_computed;
    reg [31:0] smem_stall_cycles;
    reg [31:0] mma_stall_cycles;

    //------------------------------------------------------------------------
    // SMEM Address Calculation
    //------------------------------------------------------------------------
    function [13:0] calc_smem_addr_a;
        input [STAGE_PTR_W-1:0] stage;
        input [8:0] offset;
        begin
            calc_smem_addr_a = (stage * SMEM_STAGE_STRIDE) + SMEM_A_BASE + {5'b0, offset};
        end
    endfunction

    function [13:0] calc_smem_addr_b;
        input [STAGE_PTR_W-1:0] stage;
        input [8:0] offset;
        begin
            calc_smem_addr_b = (stage * SMEM_STAGE_STRIDE) + SMEM_B_BASE + {5'b0, offset};
        end
    endfunction

    //------------------------------------------------------------------------
    // Global Memory Address Calculation
    //------------------------------------------------------------------------
    function [31:0] calc_gmem_addr_a;
        input [31:0] m_off;
        input [K_PTR_W-1:0] k_tile;
        input [8:0] offset;
        begin
            // A[m_off : m_off + TILE_M, k_tile * TILE_K : (k_tile+1) * TILE_K]
            calc_gmem_addr_a = m_off * 1024 + (k_tile * TILE_K * (DATA_WIDTH/8)) + {23'b0, offset};
        end
    endfunction

    function [31:0] calc_gmem_addr_b;
        input [31:0] n_off;
        input [K_PTR_W-1:0] k_tile;
        input [8:0] offset;
        begin
            // B[k_tile * TILE_K : (k_tile+1) * TILE_K, n_off : n_off + TILE_N]
            calc_gmem_addr_b = n_off * 1024 + (k_tile * TILE_K * (DATA_WIDTH/8)) + {23'b0, offset};
        end
    endfunction

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    reg gmem_req_valid_r;
    reg [31:0] gmem_req_addr_r;
    reg [8:0] gmem_req_size_r;
    reg gmem_req_is_a_r;
    reg smem_wr_en_r;
    reg [13:0] smem_wr_addr_r;
    reg [511:0] smem_wr_data_r;
    reg smem_rd_en_r;
    reg [13:0] smem_rd_addr_r;
    reg mma_valid_r;
    reg [511:0] mma_frag_a_r;
    reg [511:0] mma_frag_b_r;
    reg tile_done_r;
    reg tile_ready_r;

    integer rst_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            k_tile_idx <= 0;
            k_tiles_total <= 0;
            m_offset <= 0;
            n_offset <= 0;
            load_stage <= 0;
            compute_stage <= 0;
            stage_valid <= 0;
            stage_computing <= 0;
            prefetch_a_done <= 0;
            prefetch_b_done <= 0;
            prefetch_offset <= 0;
            accumulator <= 0;
            tiles_computed <= 0;
            smem_stall_cycles <= 0;
            mma_stall_cycles <= 0;
            gmem_req_valid_r <= 0;
            gmem_req_addr_r <= 0;
            gmem_req_size_r <= 0;
            gmem_req_is_a_r <= 0;
            smem_wr_en_r <= 0;
            smem_wr_addr_r <= 0;
            smem_wr_data_r <= 0;
            smem_rd_en_r <= 0;
            smem_rd_addr_r <= 0;
            mma_valid_r <= 0;
            mma_frag_a_r <= 0;
            mma_frag_b_r <= 0;
            tile_done_r <= 0;
            tile_ready_r <= 1;
        end else begin
            gmem_req_valid_r <= 0;
            smem_wr_en_r <= 0;
            smem_rd_en_r <= 0;
            mma_valid_r <= 0;
            tile_done_r <= 0;

            case (state)
                ST_IDLE: begin
                    tile_ready_r <= 1;
                    if (tile_start) begin
                        tile_ready_r <= 0;
                        m_offset <= tile_m_offset;
                        n_offset <= tile_n_offset;
                        k_tiles_total <= k_tiles[K_PTR_W-1:0];
                        k_tile_idx <= 0;
                        load_stage <= 0;
                        compute_stage <= 0;
                        stage_valid <= 0;
                        accumulator <= 0;
                        prefetch_a_done <= 0;
                        prefetch_b_done <= 0;
                        prefetch_offset <= 0;
                        state <= ST_PREFETCH_A;
                    end
                end

                ST_PREFETCH_A: begin
                    // Issue prefetch for matrix A tile
                    gmem_req_valid_r <= 1;
                    gmem_req_addr_r <= calc_gmem_addr_a(m_offset, k_tile_idx, prefetch_offset);
                    gmem_req_size_r <= 64;  // 64 bytes per request
                    gmem_req_is_a_r <= 1;

                    if (gmem_req_ready) begin
                        state <= ST_WAIT_PREFETCH;
                    end else begin
                        smem_stall_cycles <= smem_stall_cycles + 1;
                    end
                end

                ST_PREFETCH_B: begin
                    // Issue prefetch for matrix B tile
                    gmem_req_valid_r <= 1;
                    gmem_req_addr_r <= calc_gmem_addr_b(n_offset, k_tile_idx, prefetch_offset);
                    gmem_req_size_r <= 64;
                    gmem_req_is_a_r <= 0;

                    if (gmem_req_ready) begin
                        state <= ST_WAIT_PREFETCH;
                    end else begin
                        smem_stall_cycles <= smem_stall_cycles + 1;
                    end
                end

                ST_WAIT_PREFETCH: begin
                    if (gmem_resp_valid) begin
                        // Write to SMEM
                        smem_wr_en_r <= 1;
                        smem_wr_data_r <= gmem_resp_data;

                        if (gmem_req_is_a_r) begin
                            smem_wr_addr_r <= calc_smem_addr_a(load_stage, prefetch_offset);
                            prefetch_offset <= prefetch_offset + 64;

                            if (prefetch_offset + 64 >= STAGE_A_SIZE) begin
                                prefetch_a_done <= 1;
                                prefetch_offset <= 0;
                                state <= ST_PREFETCH_B;
                            end else begin
                                state <= ST_PREFETCH_A;
                            end
                        end else begin
                            smem_wr_addr_r <= calc_smem_addr_b(load_stage, prefetch_offset);
                            prefetch_offset <= prefetch_offset + 64;

                            if (prefetch_offset + 64 >= STAGE_B_SIZE) begin
                                prefetch_b_done <= 1;
                                prefetch_offset <= 0;
                                stage_valid[load_stage] <= 1;

                                // Start next prefetch if possible, or go to compute
                                if (k_tile_idx + 1 < k_tiles_total &&
                                    !stage_valid[load_stage + 1'b1]) begin
                                    load_stage <= load_stage + 1'b1;
                                    k_tile_idx <= k_tile_idx + 1;
                                    prefetch_a_done <= 0;
                                    prefetch_b_done <= 0;
                                    state <= ST_PREFETCH_A;
                                end else begin
                                    state <= ST_LOAD_A;
                                end
                            end else begin
                                state <= ST_PREFETCH_B;
                            end
                        end
                    end
                end

                ST_LOAD_A: begin
                    // Load A fragment from SMEM
                    if (stage_valid[compute_stage]) begin
                        smem_rd_en_r <= 1;
                        smem_rd_addr_r <= calc_smem_addr_a(compute_stage, 0);
                        state <= ST_LOAD_B;
                    end else begin
                        smem_stall_cycles <= smem_stall_cycles + 1;
                    end
                end

                ST_LOAD_B: begin
                    // Store A and load B
                    if (smem_rd_valid) begin
                        mma_frag_a_r <= smem_rd_data;
                        smem_rd_en_r <= 1;
                        smem_rd_addr_r <= calc_smem_addr_b(compute_stage, 0);
                        state <= ST_COMPUTE;
                    end
                end

                ST_COMPUTE: begin
                    // Issue MMA operation
                    if (smem_rd_valid) begin
                        mma_frag_b_r <= smem_rd_data;
                    end

                    if (mma_ready) begin
                        mma_valid_r <= 1;
                        stage_computing[compute_stage] <= 1;
                        state <= ST_ADVANCE;
                    end else begin
                        mma_stall_cycles <= mma_stall_cycles + 1;
                    end
                end

                ST_ADVANCE: begin
                    // Wait for MMA completion and advance
                    if (mma_done) begin
                        accumulator <= mma_accum_out;
                        stage_valid[compute_stage] <= 0;
                        stage_computing[compute_stage] <= 0;
                        tiles_computed <= tiles_computed + 1;

                        // Check if more tiles to process
                        if (compute_stage != load_stage || k_tile_idx >= k_tiles_total) begin
                            compute_stage <= compute_stage + 1'b1;

                            if (compute_stage + 1'b1 == load_stage &&
                                k_tile_idx >= k_tiles_total) begin
                                state <= ST_WRITEBACK;
                            end else begin
                                state <= ST_LOAD_A;
                            end
                        end else begin
                            // More prefetching needed
                            state <= ST_PREFETCH_A;
                        end
                    end
                end

                ST_WRITEBACK: begin
                    // Write back final accumulator (handled externally)
                    state <= ST_DONE;
                end

                ST_DONE: begin
                    tile_done_r <= 1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign gmem_req_valid = gmem_req_valid_r;
    assign gmem_req_addr  = gmem_req_addr_r;
    assign gmem_req_size  = gmem_req_size_r;
    assign gmem_req_is_a  = gmem_req_is_a_r;

    assign smem_wr_en   = smem_wr_en_r;
    assign smem_wr_addr = smem_wr_addr_r;
    assign smem_wr_data = smem_wr_data_r;
    assign smem_wr_mask = 64'hFFFFFFFFFFFFFFFF;

    assign smem_rd_en   = smem_rd_en_r;
    assign smem_rd_addr = smem_rd_addr_r;

    assign mma_valid    = mma_valid_r;
    assign mma_frag_a   = mma_frag_a_r;
    assign mma_frag_b   = mma_frag_b_r;
    assign mma_accum_in = accumulator;

    assign tile_done    = tile_done_r;
    assign tile_ready   = tile_ready_r;

    assign stat_tiles_computed = tiles_computed;
    assign stat_smem_stalls    = smem_stall_cycles;
    assign stat_mma_stalls     = mma_stall_cycles;

endmodule


//============================================================================
// Tensor Pipeline Scheduler
// Optimizes MMA instruction scheduling with data reuse
//============================================================================
module tensor_pipeline_scheduler #(
    parameter NUM_TC_UNITS  = 8,        // Tensor core units
    parameter ISSUE_DEPTH   = 16,       // Issue queue depth
    parameter REUSE_DEPTH   = 4         // Data reuse buffer depth
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // MMA Request Interface
    //------------------------------------------------------------------------
    input  wire                     mma_req_valid,
    input  wire [511:0]             mma_req_frag_a,
    input  wire [511:0]             mma_req_frag_b,
    input  wire [1023:0]            mma_req_accum,
    input  wire [2:0]               mma_req_op,         // Operation type
    input  wire [4:0]               mma_req_rd,         // Destination
    input  wire [1:0]               mma_req_warp,       // Warp ID
    output wire                     mma_req_ready,

    //------------------------------------------------------------------------
    // Tensor Core Interface
    //------------------------------------------------------------------------
    output wire [NUM_TC_UNITS-1:0]  tc_valid,
    output wire [511:0]             tc_frag_a [0:NUM_TC_UNITS-1],
    output wire [511:0]             tc_frag_b [0:NUM_TC_UNITS-1],
    output wire [1023:0]            tc_accum [0:NUM_TC_UNITS-1],
    input  wire [NUM_TC_UNITS-1:0]  tc_ready,
    input  wire [NUM_TC_UNITS-1:0]  tc_done,
    input  wire [1023:0]            tc_result [0:NUM_TC_UNITS-1],

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output wire                     result_valid,
    output wire [1023:0]            result_data,
    output wire [4:0]               result_rd,
    output wire [1:0]               result_warp,

    //------------------------------------------------------------------------
    // Reuse Statistics
    //------------------------------------------------------------------------
    output wire [31:0]              stat_reuse_hits,
    output wire [31:0]              stat_total_mmas
);

    //------------------------------------------------------------------------
    // Issue Queue Entry
    //------------------------------------------------------------------------
    localparam ENTRY_W = 512 + 512 + 1024 + 3 + 5 + 2;

    reg [ENTRY_W-1:0] issue_queue [0:ISSUE_DEPTH-1];
    reg [ISSUE_DEPTH-1:0] queue_valid;
    reg [$clog2(ISSUE_DEPTH)-1:0] queue_head;
    reg [$clog2(ISSUE_DEPTH)-1:0] queue_tail;
    reg [$clog2(ISSUE_DEPTH+1)-1:0] queue_count;

    wire queue_full = (queue_count == ISSUE_DEPTH);
    wire queue_empty = (queue_count == 0);

    //------------------------------------------------------------------------
    // Reuse Buffer for A and B fragments
    //------------------------------------------------------------------------
    reg [511:0] reuse_a [0:REUSE_DEPTH-1];
    reg [511:0] reuse_b [0:REUSE_DEPTH-1];
    reg [REUSE_DEPTH-1:0] reuse_valid_a;
    reg [REUSE_DEPTH-1:0] reuse_valid_b;
    reg [$clog2(REUSE_DEPTH)-1:0] reuse_ptr_a;
    reg [$clog2(REUSE_DEPTH)-1:0] reuse_ptr_b;

    // Check for reuse hit
    wire [REUSE_DEPTH-1:0] reuse_hit_a;
    wire [REUSE_DEPTH-1:0] reuse_hit_b;

    genvar r;
    generate
        for (r = 0; r < REUSE_DEPTH; r = r + 1) begin : gen_reuse_check
            assign reuse_hit_a[r] = reuse_valid_a[r] && (reuse_a[r] == mma_req_frag_a);
            assign reuse_hit_b[r] = reuse_valid_b[r] && (reuse_b[r] == mma_req_frag_b);
        end
    endgenerate

    wire found_reuse_a = |reuse_hit_a;
    wire found_reuse_b = |reuse_hit_b;

    //------------------------------------------------------------------------
    // TC Unit Assignment
    //------------------------------------------------------------------------
    reg [NUM_TC_UNITS-1:0] tc_busy;
    reg [$clog2(NUM_TC_UNITS)-1:0] tc_assign_ptr;

    // Find free TC unit
    reg [$clog2(NUM_TC_UNITS)-1:0] free_tc;
    reg found_free_tc;

    integer tc_i;
    always @(*) begin
        free_tc = 0;
        found_free_tc = 0;
        for (tc_i = 0; tc_i < NUM_TC_UNITS; tc_i = tc_i + 1) begin
            if (!tc_busy[tc_i] && tc_ready[tc_i] && !found_free_tc) begin
                free_tc = tc_i[$clog2(NUM_TC_UNITS)-1:0];
                found_free_tc = 1;
            end
        end
    end

    //------------------------------------------------------------------------
    // TC Output Registers
    //------------------------------------------------------------------------
    reg [NUM_TC_UNITS-1:0] tc_valid_r;
    reg [511:0] tc_frag_a_r [0:NUM_TC_UNITS-1];
    reg [511:0] tc_frag_b_r [0:NUM_TC_UNITS-1];
    reg [1023:0] tc_accum_r [0:NUM_TC_UNITS-1];

    // Track pending operations per TC
    reg [4:0] tc_pending_rd [0:NUM_TC_UNITS-1];
    reg [1:0] tc_pending_warp [0:NUM_TC_UNITS-1];

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] reuse_hit_count;
    reg [31:0] mma_count;

    //------------------------------------------------------------------------
    // Main Logic
    //------------------------------------------------------------------------
    integer rst_q, rst_tc;
    integer upd_tc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            queue_valid <= 0;
            queue_head <= 0;
            queue_tail <= 0;
            queue_count <= 0;
            reuse_valid_a <= 0;
            reuse_valid_b <= 0;
            reuse_ptr_a <= 0;
            reuse_ptr_b <= 0;
            tc_busy <= 0;
            tc_assign_ptr <= 0;
            tc_valid_r <= 0;
            reuse_hit_count <= 0;
            mma_count <= 0;

            for (rst_tc = 0; rst_tc < NUM_TC_UNITS; rst_tc = rst_tc + 1) begin
                tc_frag_a_r[rst_tc] <= 0;
                tc_frag_b_r[rst_tc] <= 0;
                tc_accum_r[rst_tc] <= 0;
                tc_pending_rd[rst_tc] <= 0;
                tc_pending_warp[rst_tc] <= 0;
            end

            for (rst_q = 0; rst_q < REUSE_DEPTH; rst_q = rst_q + 1) begin
                reuse_a[rst_q] <= 0;
                reuse_b[rst_q] <= 0;
            end
        end else begin
            tc_valid_r <= 0;

            // Enqueue new request
            if (mma_req_valid && !queue_full) begin
                issue_queue[queue_tail] <= {mma_req_warp, mma_req_rd, mma_req_op,
                                           mma_req_accum, mma_req_frag_b, mma_req_frag_a};
                queue_valid[queue_tail] <= 1'b1;
                queue_tail <= queue_tail + 1'b1;
                queue_count <= queue_count + 1;

                // Update reuse buffer
                if (!found_reuse_a) begin
                    reuse_a[reuse_ptr_a] <= mma_req_frag_a;
                    reuse_valid_a[reuse_ptr_a] <= 1'b1;
                    reuse_ptr_a <= reuse_ptr_a + 1'b1;
                end else begin
                    reuse_hit_count <= reuse_hit_count + 1;
                end

                if (!found_reuse_b) begin
                    reuse_b[reuse_ptr_b] <= mma_req_frag_b;
                    reuse_valid_b[reuse_ptr_b] <= 1'b1;
                    reuse_ptr_b <= reuse_ptr_b + 1'b1;
                end else begin
                    reuse_hit_count <= reuse_hit_count + 1;
                end
            end

            // Issue from queue to TC
            if (!queue_empty && found_free_tc) begin
                tc_valid_r[free_tc] <= 1'b1;
                tc_frag_a_r[free_tc] <= issue_queue[queue_head][511:0];
                tc_frag_b_r[free_tc] <= issue_queue[queue_head][1023:512];
                tc_accum_r[free_tc] <= issue_queue[queue_head][2047:1024];
                tc_pending_rd[free_tc] <= issue_queue[queue_head][2052:2048];
                tc_pending_warp[free_tc] <= issue_queue[queue_head][2054:2053];

                tc_busy[free_tc] <= 1'b1;
                queue_valid[queue_head] <= 1'b0;
                queue_head <= queue_head + 1'b1;
                queue_count <= queue_count - 1;
                mma_count <= mma_count + 1;
            end

            // Handle TC completion
            for (upd_tc = 0; upd_tc < NUM_TC_UNITS; upd_tc = upd_tc + 1) begin
                if (tc_done[upd_tc]) begin
                    tc_busy[upd_tc] <= 1'b0;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Result Output
    //------------------------------------------------------------------------
    // Priority encoder for completed TC
    reg [$clog2(NUM_TC_UNITS)-1:0] result_tc;
    reg found_result;

    integer res_i;
    always @(*) begin
        result_tc = 0;
        found_result = 0;
        for (res_i = 0; res_i < NUM_TC_UNITS; res_i = res_i + 1) begin
            if (tc_done[res_i] && !found_result) begin
                result_tc = res_i[$clog2(NUM_TC_UNITS)-1:0];
                found_result = 1;
            end
        end
    end

    assign result_valid = found_result;
    assign result_data  = tc_result[result_tc];
    assign result_rd    = tc_pending_rd[result_tc];
    assign result_warp  = tc_pending_warp[result_tc];

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign mma_req_ready = !queue_full;

    generate
        genvar tc;
        for (tc = 0; tc < NUM_TC_UNITS; tc = tc + 1) begin : gen_tc_out
            assign tc_valid[tc] = tc_valid_r[tc];
            assign tc_frag_a[tc] = tc_frag_a_r[tc];
            assign tc_frag_b[tc] = tc_frag_b_r[tc];
            assign tc_accum[tc] = tc_accum_r[tc];
        end
    endgenerate

    assign stat_reuse_hits = reuse_hit_count;
    assign stat_total_mmas = mma_count;

endmodule
