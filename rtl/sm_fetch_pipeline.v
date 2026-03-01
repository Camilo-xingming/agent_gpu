//============================================================================
// RalphGPU - SM Fetch Pipeline (Dual-Port for True Dual-Issue)
//
// #148: Port A serves even warps (scheduler 0), Port B serves odd warps (scheduler 1).
// Each port has independent round-robin arbitration and fetch tracking.
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

    // Branch flush
    input  wire [NUM_WARPS-1:0]     branch_flush_mask,

    // Per-warp fetch PC (from SM, read-only here)
    input  wire [32*NUM_WARPS-1:0]  warp_fetch_pc_flat,

    // Instruction memory interface — Port A (even warps)
    output wire                     imem_req,
    output wire [31:0]              imem_addr,
    input  wire                     imem_ready,
    input  wire [63:0]              imem_data,
    input  wire                     imem_valid,

    // Outputs to SM
    output reg  [NUM_WARPS-1:0]     warp_inst_buf_valid,
    output reg  [NUM_WARPS-1:0]     warp_inst_valid_d1,
    output wire [NUM_WARPS-1:0]     warp_inst_consume_gated,
    output wire [NUM_WARPS-1:0]     warp_inst_valid_fast,
    output wire [32*NUM_WARPS-1:0]  warp_inst_buf_fast_flat,
    output wire [32*NUM_WARPS-1:0]  warp_inst_buf_flat,
    output wire                     fetch_req,
    output wire                     fetch_fire,
    output wire [WARP_ID_W-1:0]     fetch_warp_id_out,

    // PC advance requests (SM applies these)
    output reg  [NUM_WARPS-1:0]     fetch_pc_advance,
    output reg  [NUM_WARPS-1:0]     nib_pc_advance,

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

    // ---- NIB take logic (combinational) ----
    wire [NUM_WARPS-1:0] nib_take;
    wire [NUM_WARPS-1:0] nib_will_serve;
    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_nib_serve
            assign nib_take[upi] = warp_next_inst_hit[upi]
                                && !warp_inst_buf_valid[upi]
                                && !warp_fill[upi]
                                && !warp_fetch_pending[upi]
                                && !branch_flush_mask[upi];
            assign nib_will_serve[upi] = nib_take[upi];
        end
    endgenerate

    // ---- #148: Even/odd warp masks for dual-port partitioning ----
    wire [NUM_WARPS-1:0] even_warp_mask;
    wire [NUM_WARPS-1:0] odd_warp_mask;
    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_warp_parity
            assign even_warp_mask[upi] = (upi % 2 == 0);
            assign odd_warp_mask[upi]  = (upi % 2 == 1);
        end
    endgenerate

    // ---- Port A fetch arbitration (even warps only) ----
    reg [WARP_ID_W-1:0] fetch_arb_ptr_a;
    reg [WARP_ID_W-1:0] fetch_warp_id_a;
    reg                  fetch_valid_arb_a;

    wire [NUM_WARPS-1:0] warp_needs_fetch_a = ICACHE_BYPASS ? warp_needs_fetch : (warp_needs_fetch & even_warp_mask);

    integer fa_i;
    always @(*) begin
        fetch_valid_arb_a = 0;
        fetch_warp_id_a = 0;
        for (fa_i = 0; fa_i < NUM_WARPS; fa_i = fa_i + 1) begin
            if (!fetch_valid_arb_a) begin
                begin : fetch_arb_check_a
                    reg [WARP_ID_W-1:0] f_idx;
                    f_idx = fetch_arb_ptr_a + fa_i[WARP_ID_W-1:0];
                    if (warp_valid[f_idx] &&
                        warp_needs_fetch_a[f_idx] &&
                        !warp_exit_pending[f_idx]) begin
                        fetch_valid_arb_a = 1;
                        fetch_warp_id_a = f_idx;
                    end
                end
            end
        end
    end

    // ---- Port B fetch arbitration (odd warps only) ----
    reg [WARP_ID_W-1:0] fetch_arb_ptr_b;
    reg [WARP_ID_W-1:0] fetch_warp_id_b;
    reg                  fetch_valid_arb_b;

    wire [NUM_WARPS-1:0] warp_needs_fetch_b = warp_needs_fetch & odd_warp_mask;

    integer fb_i;
    always @(*) begin
        fetch_valid_arb_b = 0;
        fetch_warp_id_b = 0;
        for (fb_i = 0; fb_i < NUM_WARPS; fb_i = fb_i + 1) begin
            if (!fetch_valid_arb_b) begin
                begin : fetch_arb_check_b
                    reg [WARP_ID_W-1:0] f_idx;
                    f_idx = fetch_arb_ptr_b + fb_i[WARP_ID_W-1:0];
                    if (warp_valid[f_idx] &&
                        warp_needs_fetch_b[f_idx] &&
                        !warp_exit_pending[f_idx]) begin
                        fetch_valid_arb_b = 1;
                        fetch_warp_id_b = f_idx;
                    end
                end
            end
        end
    end

    // ---- ICache interface (dual-port) ----
    wire icache_ready_a;
    wire [31:0] icache_data_a;
    wire icache_valid_a;
    wire [63:0] icache_line_data_a;
    wire icache_hit_bypass_a;
    wire [31:0] icache_hit_bypass_data_a;
    wire [63:0] icache_hit_bypass_line_data_a;

    wire icache_ready_b;
    wire [31:0] icache_data_b;
    wire icache_valid_b;
    wire [63:0] icache_line_data_b;

    wire fetch_req_a = fetch_valid_arb_a;
    wire fetch_req_b_int = fetch_valid_arb_b;

    generate
    if (ICACHE_BYPASS) begin : gen_icache_bypass
        assign imem_req  = fetch_req_a;
        assign imem_addr = warp_fetch_pc[fetch_warp_id_a];
        assign icache_ready_a = imem_ready;
        assign icache_valid_a = imem_valid;
        assign icache_data_a  = imem_data[31:0];
        assign icache_line_data_a = imem_data[63:0];
        assign icache_hit_bypass_a = 1'b0;
        assign icache_hit_bypass_data_a = 32'b0;
        assign icache_hit_bypass_line_data_a = 64'b0;
        // Bypass mode: Port B gets same-cycle valid if different address
        assign icache_ready_b = 1'b0; // Bypass mode uses only Port A fetch path
        assign icache_valid_b = 1'b0; // No Port B responses in bypass mode
        assign icache_data_b = 32'b0;
        assign icache_line_data_b = 64'b0;
    end else begin : gen_icache_normal
        icache #(
            .SIZE_KB(4),
            .LINE_SIZE(8),
            .NUM_WAYS(2)
        ) u_icache (
            .clk(clk),
            .rst_n(rst_n),
            // Port A
            .fetch_req(fetch_req_a),
            .fetch_addr(warp_fetch_pc[fetch_warp_id_a]),
            .fetch_ready(icache_ready_a),
            .fetch_data(icache_data_a),
            .fetch_line_data(icache_line_data_a),
            .fetch_valid(icache_valid_a),
            .fetch_hit_bypass(icache_hit_bypass_a),
            .fetch_hit_bypass_data(icache_hit_bypass_data_a),
            .fetch_hit_bypass_line_data(icache_hit_bypass_line_data_a),
            // Port B (#148)
            .fetch_req_b(fetch_req_b_int),
            .fetch_addr_b(warp_fetch_pc[fetch_warp_id_b]),
            .fetch_ready_b(icache_ready_b),
            .fetch_data_b(icache_data_b),
            .fetch_line_data_b(icache_line_data_b),
            .fetch_valid_b(icache_valid_b),
            // Shared
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

    // ---- Port A fetch fire ----
    assign fetch_req = fetch_req_a;
    wire fetch_fire_a = fetch_req_a && icache_ready_a;
    wire bypass_fire_a = fetch_req_a && icache_hit_bypass_a;
    assign fetch_fire = fetch_fire_a;
    assign fetch_warp_id_out = fetch_warp_id_a;

    // ---- Port B fetch fire ----
    wire fetch_fire_b = fetch_req_b_int && icache_ready_b;

    // ---- Port A fetch pipeline tracking ----
    reg [WARP_ID_W-1:0] fetch_pipe_warp_a [0:FETCH_PIPE_DEPTH-1];
    reg [31:0] fetch_pipe_pc_a [0:FETCH_PIPE_DEPTH-1];
    reg [FETCH_PIPE_DEPTH-1:0] fetch_pipe_valid_a;

    wire same_cycle_hit_a = fetch_fire_a && icache_valid_a && (fetch_pipe_valid_a == 0);

    // Port A round-robin pointer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fetch_arb_ptr_a <= 0;
        else if (fetch_fire_a || bypass_fire_a) begin
            if (ICACHE_BYPASS)
                fetch_arb_ptr_a <= fetch_warp_id_a + 1'b1;
            else
                fetch_arb_ptr_a <= fetch_warp_id_a + 2'd2; // Skip to next even warp
        end
    end

    // Port B round-robin pointer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fetch_arb_ptr_b <= 1; // Start at warp 1 (first odd)
        else if (fetch_fire_b)
            fetch_arb_ptr_b <= fetch_warp_id_b + 2'd2; // Skip to next odd warp
    end

    // ---- Port A fetch pipeline shift register ----
    wire early_response_valid_a = icache_valid_a && fetch_pipe_valid_a[0] && !fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1];
    wire delayed_response_valid_a = icache_valid_a && fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1];

    integer fp_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || kernel_start) begin
            warp_fetch_pending <= 0;
            fetch_pipe_valid_a <= 0;
            for (fp_i = 0; fp_i < FETCH_PIPE_DEPTH; fp_i = fp_i + 1) begin
                fetch_pipe_warp_a[fp_i] <= 0;
                fetch_pipe_pc_a[fp_i] <= 0;
            end
        end else begin
            // Branch flush clears pending
            warp_fetch_pending <= warp_fetch_pending & ~branch_flush_mask;

            // Port A pipeline
            if (icache_ready_a) begin
                if (early_response_valid_a) begin
                    fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1:1] <= {(FETCH_PIPE_DEPTH-1){1'b0}};
                    for (fp_i = FETCH_PIPE_DEPTH-1; fp_i > 0; fp_i = fp_i - 1) begin
                        fetch_pipe_warp_a[fp_i] <= 0;
                        fetch_pipe_pc_a[fp_i] <= 0;
                    end
                end else begin
                    fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1:1] <= fetch_pipe_valid_a[FETCH_PIPE_DEPTH-2:0];
                    for (fp_i = FETCH_PIPE_DEPTH-1; fp_i > 0; fp_i = fp_i - 1) begin
                        fetch_pipe_warp_a[fp_i] <= fetch_pipe_warp_a[fp_i-1];
                        fetch_pipe_pc_a[fp_i] <= fetch_pipe_pc_a[fp_i-1];
                    end
                end
            end

            if (fetch_fire_a && !same_cycle_hit_a) begin
                fetch_pipe_valid_a[0] <= 1'b1;
                fetch_pipe_warp_a[0] <= fetch_warp_id_a;
                fetch_pipe_pc_a[0] <= warp_fetch_pc[fetch_warp_id_a];
                warp_fetch_pending[fetch_warp_id_a] <= 1'b1;
            end else if (icache_ready_a) begin
                fetch_pipe_valid_a[0] <= 1'b0;
                fetch_pipe_warp_a[0] <= 0;
                fetch_pipe_pc_a[0] <= 0;
            end

            // Clear pending on Port A response
            if (icache_valid_a && fetch_pipe_valid_a[0] && !fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1])
                warp_fetch_pending[fetch_pipe_warp_a[0]] <= 1'b0;
            else if (icache_valid_a && fetch_pipe_valid_a[FETCH_PIPE_DEPTH-1])
                warp_fetch_pending[fetch_pipe_warp_a[FETCH_PIPE_DEPTH-1]] <= 1'b0;

            // Port B: same-cycle hit sets and clears pending in same cycle
            if (fetch_fire_b && icache_valid_b) begin
                // Same-cycle hit on Port B — no pending needed
            end else if (fetch_fire_b) begin
                warp_fetch_pending[fetch_warp_id_b] <= 1'b1;
            end

            // Port B response clears pending (for misses served later)
            if (icache_valid_b && !fetch_fire_b) begin
                // Port B miss response — pending was already set
                // The icache returns valid_b when a queued Port B miss completes
                warp_fetch_pending[fetch_warp_id_b] <= 1'b0;
            end
        end
    end

    // ---- Port A fill logic ----
    wire [WARP_ID_W-1:0] fill_warp_id_a = bypass_fire_a ? fetch_warp_id_a :
                                           same_cycle_hit_a ? fetch_warp_id_a :
                                           early_response_valid_a ? fetch_pipe_warp_a[0] :
                                           fetch_pipe_warp_a[FETCH_PIPE_DEPTH-1];
    wire [31:0] fill_fetch_pc_a = (bypass_fire_a || same_cycle_hit_a) ? warp_fetch_pc[fetch_warp_id_a] :
                                  early_response_valid_a ? fetch_pipe_pc_a[0] :
                                  fetch_pipe_pc_a[FETCH_PIPE_DEPTH-1];
    wire fill_valid_a = bypass_fire_a || same_cycle_hit_a || delayed_response_valid_a || early_response_valid_a;
    wire [31:0] fill_data_a = bypass_fire_a ? icache_hit_bypass_data_a : icache_data_a;
    wire [63:0] fill_line_data_a = bypass_fire_a ? icache_hit_bypass_line_data_a : icache_line_data_a;

    // ---- Port B fill logic (#148) ----
    wire same_cycle_hit_b = fetch_fire_b && icache_valid_b;
    wire fill_valid_b = same_cycle_hit_b || (icache_valid_b && !fetch_fire_b);
    wire [WARP_ID_W-1:0] fill_warp_id_b = fetch_warp_id_b; // Port B always fills the warp it requested
    wire [31:0] fill_fetch_pc_b = warp_fetch_pc[fetch_warp_id_b];
    wire [31:0] fill_data_b = icache_data_b;
    wire [63:0] fill_line_data_b = icache_line_data_b;

    // ---- Combined fill (Port A + Port B) ----
    // fill_data used by fast mux needs to be valid for whichever port is filling
    wire [31:0] fill_data;
    wire [63:0] fill_line_data;
    wire fill_valid;
    wire [WARP_ID_W-1:0] fill_warp_id;
    wire [31:0] fill_fetch_pc;

    // Port A has priority; both can fill different warps in same cycle
    assign fill_valid = fill_valid_a || fill_valid_b;
    assign fill_warp_id = fill_valid_a ? fill_warp_id_a : fill_warp_id_b;
    assign fill_data = fill_valid_a ? fill_data_a : fill_data_b;
    assign fill_line_data = fill_valid_a ? fill_line_data_a : fill_line_data_b;
    assign fill_fetch_pc = fill_valid_a ? fill_fetch_pc_a : fill_fetch_pc_b;

    genvar fill_w;
    generate
        for (fill_w = 0; fill_w < NUM_WARPS; fill_w = fill_w + 1) begin : gen_warp_fill
            // A warp can be filled by Port A or Port B (they target different warps)
            assign warp_fill[fill_w] = (fill_valid_a && (fill_warp_id_a == fill_w)) ||
                                       (fill_valid_b && (fill_warp_id_b == fill_w));
        end
    endgenerate

    // ---- Fast valid/data (combinational bypass for scheduler) ----
    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_fast_valid
            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            assign warp_inst_valid_fast[upi] = warp_inst_buf_valid[upi]
                                             | nib_will_serve[upi]
                                             | fill_into_empty;
        end
    endgenerate

    generate
        for (upi = 0; upi < NUM_WARPS; upi = upi + 1) begin : gen_fast_data
            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            // Select fill data from correct port
            wire [31:0] this_fill_data = (fill_valid_a && fill_warp_id_a == upi) ? fill_data_a :
                                         (fill_valid_b && fill_warp_id_b == upi) ? fill_data_b :
                                         32'b0;
            assign warp_inst_buf_fast_flat[32*upi +: 32] =
                fill_into_empty      ? this_fill_data :
                nib_will_serve[upi]  ? warp_next_inst[upi] :
                                       warp_inst_buf[upi];
        end
    endgenerate

    // ---- Instruction buffer management ----
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

            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                if (warp_fill[w_buf] && !warp_inst_buf_valid[w_buf] && warp_inst_consume_gated[w_buf])
                    warp_inst_buf_valid[w_buf] <= 1'b0;
                else if (warp_fill[w_buf])
                    warp_inst_buf_valid[w_buf] <= 1'b1;
                else if (warp_inst_consume_gated[w_buf] && !branch_flush_mask[w_buf])
                    warp_inst_buf_valid[w_buf] <= 1'b0;
            end

            // Port A fill data
            if (bypass_fire_a || same_cycle_hit_a)
                warp_inst_buf[fetch_warp_id_a] <= fill_data_a;
            else if (early_response_valid_a)
                warp_inst_buf[fetch_pipe_warp_a[0]] <= fill_data_a;
            else if (delayed_response_valid_a)
                warp_inst_buf[fetch_pipe_warp_a[FETCH_PIPE_DEPTH-1]] <= fill_data_a;

            // Port B fill data (#148)
            if (fill_valid_b)
                warp_inst_buf[fill_warp_id_b] <= fill_data_b;

            // NIB: populate from both ports
            if (fill_valid_a && !fill_fetch_pc_a[2]) begin
                warp_next_inst[fill_warp_id_a] <= fill_line_data_a[63:32];
                warp_next_inst_pc[fill_warp_id_a] <= fill_fetch_pc_a + 32'd4;
                warp_next_inst_valid[fill_warp_id_a] <= 1'b1;
            end
            if (fill_valid_b && !fill_fetch_pc_b[2]) begin
                warp_next_inst[fill_warp_id_b] <= fill_line_data_b[63:32];
                warp_next_inst_pc[fill_warp_id_b] <= fill_fetch_pc_b + 32'd4;
                warp_next_inst_valid[fill_warp_id_b] <= 1'b1;
            end

            // NIB hit: serve from buffer
            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                if (nib_take[w_buf]) begin
                    warp_inst_buf[w_buf] <= warp_next_inst[w_buf];
                    warp_inst_buf_valid[w_buf] <= 1'b1;
                    warp_next_inst_valid[w_buf] <= 1'b0;
                end
            end

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

    // ---- PC advance (both ports) ----
    integer adv_i;
    always @(*) begin
        fetch_pc_advance = {NUM_WARPS{1'b0}};
        nib_pc_advance   = {NUM_WARPS{1'b0}};

        // Port A fills
        for (adv_i = 0; adv_i < NUM_WARPS; adv_i = adv_i + 1) begin
            if (fill_valid_a && fill_warp_id_a == adv_i[WARP_ID_W-1:0])
                fetch_pc_advance[adv_i] = 1'b1;
            if (fill_valid_b && fill_warp_id_b == adv_i[WARP_ID_W-1:0])
                fetch_pc_advance[adv_i] = 1'b1;
            if (nib_take[adv_i])
                nib_pc_advance[adv_i] = 1'b1;
        end
    end
endmodule
