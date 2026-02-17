//============================================================================
// RalphGPU - SM Fetch Pipeline (Extracted from streaming_multiprocessor_v2)
//
// RALPH-6 Phase 2: Fetch arbitration, instruction buffer, and icache interface.
// PC management remains in the parent SM module.
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module sm_fetch_pipeline #(
    parameter NUM_WARPS     = `WARPS_PER_SM,
    parameter ICACHE_BYPASS = 0,
    parameter FETCH_PIPE_DEPTH = 2
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     kernel_start,

    // Warp state (from SM)
    input  wire [NUM_WARPS-1:0]     warp_valid,
    input  wire [NUM_WARPS-1:0]     warp_exit_pending,
    input  wire [NUM_WARPS-1:0]     warp_inst_consume,
    input  wire [NUM_WARPS-1:0]     decode_stalled_per_warp,

    // Branch flush: SM sets bit when warp takes a branch (clear buffer + pending)
    input  wire [NUM_WARPS-1:0]     branch_flush_mask,

    // Per-warp fetch PC (from SM, read-only here)
    input  wire [32*NUM_WARPS-1:0]  warp_fetch_pc_flat,

    // Instruction memory interface
    output wire                     imem_req,
    output wire [31:0]              imem_addr,
    input  wire                     imem_ready,
    input  wire [63:0]              imem_data,
    input  wire                     imem_valid,

    // Outputs to SM
    output reg  [NUM_WARPS-1:0]     warp_inst_buf_valid,
    output reg  [NUM_WARPS-1:0]     warp_inst_valid_d1,
    output wire [NUM_WARPS-1:0]     warp_inst_consume_gated,
    output wire [32*NUM_WARPS-1:0]  warp_inst_buf_flat,
    output wire                     fetch_req,
    output wire                     fetch_fire,
    output wire [WARP_ID_W-1:0]     fetch_warp_id_out,

    // PC advance requests (SM applies these)
    output reg  [NUM_WARPS-1:0]     fetch_pc_advance,      // advance by 4 on fetch_fire
    output reg  [NUM_WARPS-1:0]     nib_pc_advance,        // advance by 4 on NIB hit

    // Status outputs
    output wire [NUM_WARPS-1:0]     warp_next_inst_hit,
    output wire [NUM_WARPS-1:0]     warp_buf_will_be_empty,
    output wire [NUM_WARPS-1:0]     warp_fill,
    output wire [NUM_WARPS-1:0]     warp_fetch_pending_out
);

    localparam WARP_ID_W = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1;

    // Unpack flat PC array
    wire [31:0] warp_fetch_pc [0:NUM_WARPS-1];
    genvar upi;
    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_unpack_pc
            assign warp_fetch_pc[upi] = warp_fetch_pc_flat[32*upi +: 32];
        end
    endgenerate

    // Pack instruction buffer output
    reg [31:0] warp_inst_buf [0:NUM_WARPS-1];
    genvar pki;
    generate
        for (pki = 0; pki < NUM_WARPS; pki = pki + 1) begin : gen_pack_buf
            assign warp_inst_buf_flat[32*pki +: 32] = warp_inst_buf[pki];
        end
    endgenerate

    // ---- Gated consume ----
    assign warp_inst_consume_gated = warp_inst_consume & ~decode_stalled_per_warp;
    assign warp_buf_will_be_empty = ~warp_inst_buf_valid | warp_inst_consume_gated;

    // ---- Per-warp next-instruction buffer (RALPH-8 P0) ----
    reg [31:0] warp_next_inst [0:NUM_WARPS-1];
    reg [31:0] warp_next_inst_pc [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] warp_next_inst_valid;

    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_next_inst_hit
            assign warp_next_inst_hit[upi] = warp_next_inst_valid[upi] &&
                                             (warp_fetch_pc[upi] == warp_next_inst_pc[upi]);
        end
    endgenerate

    // ---- Fetch pending tracking ----
    reg [NUM_WARPS-1:0] warp_fetch_pending;
    assign warp_fetch_pending_out = warp_fetch_pending;

    wire [NUM_WARPS-1:0] warp_needs_fetch = warp_buf_will_be_empty & ~warp_fetch_pending & ~warp_next_inst_hit;

    // ---- Fetch arbitration (round-robin) ----
    reg [WARP_ID_W-1:0] fetch_arb_ptr;
    reg [WARP_ID_W-1:0] fetch_warp_id_r;
    reg                  fetch_valid_arb;

    integer f_i;
    always @(*) begin
        fetch_valid_arb = 0;
        fetch_warp_id_r = 0;
        for (f_i = 0; f_i < NUM_WARPS; f_i = f_i + 1) begin
            if (!fetch_valid_arb) begin
                begin : fetch_arb_check
                    reg [WARP_ID_W-1:0] f_idx;
                    f_idx = fetch_arb_ptr + f_i[WARP_ID_W-1:0];
                    if (warp_valid[f_idx] &&
                        warp_needs_fetch[f_idx] &&
                        !warp_exit_pending[f_idx]) begin
                        fetch_valid_arb = 1;
                        fetch_warp_id_r = f_idx;
                    end
                end
            end
        end
    end

    // ---- ICache interface ----
    wire icache_ready;
    wire [31:0] icache_data;
    wire icache_valid;
    wire [63:0] icache_line_data;  // RALPH-8 P1: full cache line for NIB

    // RALPH-8 P2: Hit-bypass during miss
    wire icache_hit_bypass;
    wire [31:0] icache_hit_bypass_data;
    wire [63:0] icache_hit_bypass_line_data;

    generate
    if (ICACHE_BYPASS) begin : gen_icache_bypass
        assign imem_req  = fetch_req;
        assign imem_addr = warp_fetch_pc[fetch_warp_id_r];
        assign icache_ready = imem_ready;
        assign icache_valid = imem_valid;
        assign icache_data  = imem_data[31:0];
        assign icache_line_data = imem_data[63:0];
        // No bypass needed in bypass mode (always ready)
        assign icache_hit_bypass = 1'b0;
        assign icache_hit_bypass_data = 32'b0;
        assign icache_hit_bypass_line_data = 64'b0;
    end else begin : gen_icache_normal
        icache #(
            .SIZE_KB(4),
            .LINE_SIZE(8),
            .NUM_WAYS(2)
        ) u_icache (
            .clk(clk),
            .rst_n(rst_n),
            .fetch_req(fetch_req),
            .fetch_addr(warp_fetch_pc[fetch_warp_id_r]),
            .fetch_ready(icache_ready),
            .fetch_data(icache_data),
            .fetch_line_data(icache_line_data),
            .fetch_valid(icache_valid),
            .fetch_hit_bypass(icache_hit_bypass),
            .fetch_hit_bypass_data(icache_hit_bypass_data),
            .fetch_hit_bypass_line_data(icache_hit_bypass_line_data),
            .invalidate_req(1'b0),
            .invalidate_addr(32'b0),
            .invalidate_all(1'b0),
            .invalidate_done(),
            .mem_req_valid(imem_req),
            .mem_req_addr(imem_addr),
            .mem_req_ready(imem_ready),
            .mem_resp_data(imem_data),
            .mem_resp_valid(imem_valid),
            .stat_hits(),
            .stat_misses(),
            .stat_prefetch_hits()
        );
    end
    endgenerate

    // ---- Fetch request / fire ----
    assign fetch_req = fetch_valid_arb;
    assign fetch_fire = fetch_req && icache_ready;
    // RALPH-8 P2: Bypass fire — serves a cache hit while miss is in flight
    wire bypass_fire = fetch_req && icache_hit_bypass;
    assign fetch_warp_id_out = fetch_warp_id_r;

    // ---- Fetch pipeline tracking ----
    reg [WARP_ID_W-1:0] fetch_pipe_warp [0:FETCH_PIPE_DEPTH-1];
    reg [31:0] fetch_pipe_pc [0:FETCH_PIPE_DEPTH-1];  // RALPH-8 P1: track PC at fetch time
    reg [FETCH_PIPE_DEPTH-1:0] fetch_pipe_valid;

    wire same_cycle_hit = fetch_fire && icache_valid && (fetch_pipe_valid == 0);

    // Round-robin pointer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fetch_arb_ptr <= 0;
        else if (fetch_fire || bypass_fire)
            fetch_arb_ptr <= fetch_warp_id_r + 1'b1;
    end

    // Fetch pipeline shift register
    integer fp_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || kernel_start) begin
            warp_fetch_pending <= 0;
            fetch_pipe_valid <= 0;
            for (fp_i = 0; fp_i < FETCH_PIPE_DEPTH; fp_i = fp_i + 1) begin
                fetch_pipe_warp[fp_i] <= 0;
                fetch_pipe_pc[fp_i] <= 0;
            end
        end else begin
            // Branch flush clears pending
            warp_fetch_pending <= warp_fetch_pending & ~branch_flush_mask;

            if (icache_ready) begin
                fetch_pipe_valid[FETCH_PIPE_DEPTH-1:1] <= fetch_pipe_valid[FETCH_PIPE_DEPTH-2:0];
                for (fp_i = FETCH_PIPE_DEPTH-1; fp_i > 0; fp_i = fp_i - 1) begin
                    fetch_pipe_warp[fp_i] <= fetch_pipe_warp[fp_i-1];
                    fetch_pipe_pc[fp_i] <= fetch_pipe_pc[fp_i-1];
                end
            end

            if (fetch_fire && !same_cycle_hit) begin
                fetch_pipe_valid[0] <= 1'b1;
                fetch_pipe_warp[0] <= fetch_warp_id_r;
                fetch_pipe_pc[0] <= warp_fetch_pc[fetch_warp_id_r];
                warp_fetch_pending[fetch_warp_id_r] <= 1'b1;
            end else if (icache_ready) begin
                fetch_pipe_valid[0] <= 1'b0;
                fetch_pipe_warp[0] <= 0;
                fetch_pipe_pc[0] <= 0;
            end

            // Clear pending on response
            if (icache_valid && fetch_pipe_valid[0] && !fetch_pipe_valid[FETCH_PIPE_DEPTH-1])
                warp_fetch_pending[fetch_pipe_warp[0]] <= 1'b0;
            else if (icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1])
                warp_fetch_pending[fetch_pipe_warp[FETCH_PIPE_DEPTH-1]] <= 1'b0;
        end
    end

    // ---- Fill logic ----
    wire delayed_response_valid = icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1];
    wire early_response_valid   = icache_valid && fetch_pipe_valid[0] && !fetch_pipe_valid[FETCH_PIPE_DEPTH-1];
    wire [WARP_ID_W-1:0] fill_warp_id = bypass_fire ? fetch_warp_id_r :
                                         same_cycle_hit ? fetch_warp_id_r :
                                         early_response_valid ? fetch_pipe_warp[0] :
                                         fetch_pipe_warp[FETCH_PIPE_DEPTH-1];
    // RALPH-8 P1: PC at fetch time (for NIB address calculation)
    wire [31:0] fill_fetch_pc = (bypass_fire || same_cycle_hit) ? warp_fetch_pc[fetch_warp_id_r] :
                                early_response_valid ? fetch_pipe_pc[0] :
                                fetch_pipe_pc[FETCH_PIPE_DEPTH-1];
    wire fill_valid = bypass_fire || same_cycle_hit || delayed_response_valid || early_response_valid;

    // RALPH-8 P2: Select data source — bypass uses separate data path
    wire [31:0] fill_data = bypass_fire ? icache_hit_bypass_data : icache_data;
    wire [63:0] fill_line_data = bypass_fire ? icache_hit_bypass_line_data : icache_line_data;

    genvar fill_w;
    generate
        for (fill_w = 0; fill_w < NUM_WARPS; fill_w = fill_w + 1) begin : gen_warp_fill
            assign warp_fill[fill_w] = fill_valid && (fill_warp_id == fill_w);
        end
    endgenerate

    // ---- Instruction buffer + valid management ----
    integer w_buf;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1)
                warp_inst_buf[w_buf] <= 0;
            warp_inst_buf_valid <= 0;
            warp_next_inst_valid <= 0;
        end else if (kernel_start) begin
            warp_inst_buf_valid <= 0;
            warp_next_inst_valid <= 0;
        end else begin
            // Branch flush clears buffer valid
            warp_inst_buf_valid <= warp_inst_buf_valid & ~branch_flush_mask;

            // Valid bit management (fill/consume, after flush)
            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                if (warp_fill[w_buf])
                    warp_inst_buf_valid[w_buf] <= 1'b1;
                else if (warp_inst_consume_gated[w_buf] && !branch_flush_mask[w_buf])
                    warp_inst_buf_valid[w_buf] <= 1'b0;
            end

            // Fill instruction data (RALPH-8 P2: uses fill_data for bypass support)
            if (bypass_fire || same_cycle_hit)
                warp_inst_buf[fetch_warp_id_r] <= fill_data;
            else if (early_response_valid)
                warp_inst_buf[fetch_pipe_warp[0]] <= fill_data;
            else if (delayed_response_valid)
                warp_inst_buf[fetch_pipe_warp[FETCH_PIPE_DEPTH-1]] <= fill_data;

            // RALPH-8 P0+P1+P2: Buffer upper word from 64-bit cache line
            // Only populate NIB when fetched address is first word of line (bit[2]==0),
            // meaning the second word (PC+4) is in the same line.
            if (fill_valid && !fill_fetch_pc[2]) begin
                warp_next_inst[fill_warp_id] <= fill_line_data[63:32];
                warp_next_inst_pc[fill_warp_id] <= fill_fetch_pc + 32'd4;
                warp_next_inst_valid[fill_warp_id] <= 1'b1;
            end

            // NIB hit: serve from buffer
            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                if (warp_next_inst_hit[w_buf] && warp_buf_will_be_empty[w_buf]
                    && !warp_fill[w_buf] && !warp_fetch_pending[w_buf]
                    && !branch_flush_mask[w_buf]) begin
                    warp_inst_buf[w_buf] <= warp_next_inst[w_buf];
                    warp_inst_buf_valid[w_buf] <= 1'b1;
                    warp_next_inst_valid[w_buf] <= 1'b0;
                end
            end

            // Branch flush invalidates NIB too
            warp_next_inst_valid <= warp_next_inst_valid & ~branch_flush_mask;
        end
    end

    // ---- d1 delay for scheduler ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            warp_inst_valid_d1 <= 0;
        else
            warp_inst_valid_d1 <= warp_inst_buf_valid;
    end

    // ---- PC advance requests (combinational outputs to SM) ----
    always @(*) begin
        fetch_pc_advance = {NUM_WARPS{1'b0}};
        nib_pc_advance   = {NUM_WARPS{1'b0}};

        if (fetch_fire || bypass_fire)
            fetch_pc_advance[fetch_warp_id_r] = 1'b1;

        for (f_i = 0; f_i < NUM_WARPS; f_i = f_i + 1) begin
            if (warp_next_inst_hit[f_i] && warp_buf_will_be_empty[f_i]
                && !warp_fill[f_i] && !warp_fetch_pending[f_i]
                && !branch_flush_mask[f_i])
                nib_pc_advance[f_i] = 1'b1;
        end
    end

endmodule
