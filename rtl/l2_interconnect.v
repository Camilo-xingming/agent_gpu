//============================================================================
// RalphGPU - L2 Cache Slice Interconnect
// Multi-slice L2 topology with crossbar arbitration
// Based on NVIDIA crossbar NOC design
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module l2_interconnect #(
    parameter NUM_SM            = `NUM_SM,
`ifdef SYNTHESIS
    parameter NUM_L2_SLICES     = 2,
    parameter ADDR_WIDTH        = 32,
    parameter DATA_WIDTH        = 512,
    parameter ID_WIDTH          = 4,
    parameter MAX_OUTSTANDING   = 2
`else
    parameter NUM_L2_SLICES     = 4,            // L2 cache slices
    parameter ADDR_WIDTH        = 32,
    parameter DATA_WIDTH        = 512,          // Cache line width
    parameter ID_WIDTH          = 4,
    parameter MAX_OUTSTANDING   = 16            // Per-SM outstanding requests
`endif
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // SM Interfaces (Requestors)
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]                    sm_req_valid,
    input  wire [NUM_SM-1:0]                    sm_req_write,
    input  wire [NUM_SM*ADDR_WIDTH-1:0]         sm_req_addr,
    input  wire [NUM_SM*DATA_WIDTH-1:0]         sm_req_wdata,
    input  wire [NUM_SM*ID_WIDTH-1:0]           sm_req_id,
    output wire [NUM_SM-1:0]                    sm_req_ready,
    output wire [NUM_SM-1:0]                    sm_resp_valid,
    output wire [NUM_SM*DATA_WIDTH-1:0]         sm_resp_rdata,
    output wire [NUM_SM*ID_WIDTH-1:0]           sm_resp_id,

    //------------------------------------------------------------------------
    // L2 Slice Interfaces
    //------------------------------------------------------------------------
    output wire [NUM_L2_SLICES-1:0]             l2_req_valid,
    output wire [NUM_L2_SLICES-1:0]             l2_req_write,
    output wire [NUM_L2_SLICES*ADDR_WIDTH-1:0]  l2_req_addr,
    output wire [NUM_L2_SLICES*DATA_WIDTH-1:0]  l2_req_wdata,
    output wire [NUM_L2_SLICES*ID_WIDTH-1:0]    l2_req_id,
    output wire [NUM_L2_SLICES*$clog2(NUM_SM)-1:0] l2_req_sm_id,
    input  wire [NUM_L2_SLICES-1:0]             l2_req_ready,
    input  wire [NUM_L2_SLICES-1:0]             l2_resp_valid,
    input  wire [NUM_L2_SLICES*DATA_WIDTH-1:0]  l2_resp_rdata,
    input  wire [NUM_L2_SLICES*ID_WIDTH-1:0]    l2_resp_id,
    input  wire [NUM_L2_SLICES*$clog2(NUM_SM)-1:0] l2_resp_sm_id,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output wire [31:0]              stat_total_requests,
    output wire [31:0]              stat_xbar_conflicts,
    output wire [31:0]              stat_avg_latency
);

    localparam SM_W = $clog2(NUM_SM);
    localparam SLICE_W = $clog2(NUM_L2_SLICES);
    localparam OST_W = $clog2(MAX_OUTSTANDING);

    //------------------------------------------------------------------------
    // Address to L2 Slice Mapping
    // Uses address hashing for balanced distribution
    //------------------------------------------------------------------------
    function [SLICE_W-1:0] addr_to_slice;
        input [ADDR_WIDTH-1:0] addr;
        begin
            // XOR-based hash for better distribution
            // Use bits [12:6] XOR bits [18:12] for slice selection
            addr_to_slice = (addr[6 +: SLICE_W] ^ addr[12 +: SLICE_W]) % NUM_L2_SLICES;
        end
    endfunction

    //------------------------------------------------------------------------
    // Per-SM Target Slice Calculation
    //------------------------------------------------------------------------
    wire [SLICE_W-1:0] sm_target_slice [0:NUM_SM-1];
    genvar s;
    generate
        for (s = 0; s < NUM_SM; s = s + 1) begin : gen_target
            assign sm_target_slice[s] = addr_to_slice(sm_req_addr[s*ADDR_WIDTH +: ADDR_WIDTH]);
        end
    endgenerate

    //------------------------------------------------------------------------
    // Request Crossbar - SM to L2 Slice Arbitration
    //------------------------------------------------------------------------
    // Per-slice: which SMs are requesting this slice
    wire [NUM_SM-1:0] slice_requestors [0:NUM_L2_SLICES-1];
    generate
        genvar sl, sm;
        for (sl = 0; sl < NUM_L2_SLICES; sl = sl + 1) begin : gen_slice_req
            for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : gen_sm_to_slice
                assign slice_requestors[sl][sm] = sm_req_valid[sm] && (sm_target_slice[sm] == sl);
            end
        end
    endgenerate

    // Round-robin arbitration per slice
    reg [SM_W-1:0] slice_rr_ptr [0:NUM_L2_SLICES-1];
    reg [NUM_SM-1:0] slice_grant [0:NUM_L2_SLICES-1];

    // Granted SM per slice (one-hot to binary)
    reg [SM_W-1:0] slice_granted_sm [0:NUM_L2_SLICES-1];
    reg [NUM_L2_SLICES-1:0] slice_has_grant;

    integer arb_sl, arb_sm, arb_idx;
    always @(*) begin
        for (arb_sl = 0; arb_sl < NUM_L2_SLICES; arb_sl = arb_sl + 1) begin
            slice_grant[arb_sl] = 0;
            slice_granted_sm[arb_sl] = 0;
            slice_has_grant[arb_sl] = 0;

            for (arb_sm = 0; arb_sm < NUM_SM; arb_sm = arb_sm + 1) begin
                arb_idx = (slice_rr_ptr[arb_sl] + arb_sm) % NUM_SM;
                if (slice_requestors[arb_sl][arb_idx] && !slice_has_grant[arb_sl]) begin
                    slice_grant[arb_sl][arb_idx] = 1'b1;
                    slice_granted_sm[arb_sl] = arb_idx[SM_W-1:0];
                    slice_has_grant[arb_sl] = 1'b1;
                end
            end
        end
    end

    // Update round-robin pointers
    integer rr_sl;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (rr_sl = 0; rr_sl < NUM_L2_SLICES; rr_sl = rr_sl + 1) begin
                slice_rr_ptr[rr_sl] <= 0;
            end
        end else begin
            for (rr_sl = 0; rr_sl < NUM_L2_SLICES; rr_sl = rr_sl + 1) begin
                if (slice_has_grant[rr_sl] && l2_req_ready[rr_sl]) begin
                    slice_rr_ptr[rr_sl] <= (slice_granted_sm[rr_sl] + 1) % NUM_SM;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Generate L2 Request Outputs
    //------------------------------------------------------------------------
    reg [NUM_L2_SLICES-1:0] l2_req_valid_r;
    reg [NUM_L2_SLICES-1:0] l2_req_write_r;
    reg [ADDR_WIDTH-1:0] l2_req_addr_r [0:NUM_L2_SLICES-1];
    reg [DATA_WIDTH-1:0] l2_req_wdata_r [0:NUM_L2_SLICES-1];
    reg [ID_WIDTH-1:0] l2_req_id_r [0:NUM_L2_SLICES-1];
    reg [SM_W-1:0] l2_req_sm_id_r [0:NUM_L2_SLICES-1];

    integer req_sl;
    always @(*) begin
        for (req_sl = 0; req_sl < NUM_L2_SLICES; req_sl = req_sl + 1) begin
            l2_req_valid_r[req_sl] = slice_has_grant[req_sl];
            l2_req_write_r[req_sl] = sm_req_write[slice_granted_sm[req_sl]];
            l2_req_addr_r[req_sl] = sm_req_addr[slice_granted_sm[req_sl]*ADDR_WIDTH +: ADDR_WIDTH];
            l2_req_wdata_r[req_sl] = sm_req_wdata[slice_granted_sm[req_sl]*DATA_WIDTH +: DATA_WIDTH];
            l2_req_id_r[req_sl] = sm_req_id[slice_granted_sm[req_sl]*ID_WIDTH +: ID_WIDTH];
            l2_req_sm_id_r[req_sl] = slice_granted_sm[req_sl];
        end
    end

    // Pack outputs
    generate
        for (sl = 0; sl < NUM_L2_SLICES; sl = sl + 1) begin : gen_l2_out
            assign l2_req_valid[sl] = l2_req_valid_r[sl];
            assign l2_req_write[sl] = l2_req_write_r[sl];
            assign l2_req_addr[sl*ADDR_WIDTH +: ADDR_WIDTH] = l2_req_addr_r[sl];
            assign l2_req_wdata[sl*DATA_WIDTH +: DATA_WIDTH] = l2_req_wdata_r[sl];
            assign l2_req_id[sl*ID_WIDTH +: ID_WIDTH] = l2_req_id_r[sl];
            assign l2_req_sm_id[sl*SM_W +: SM_W] = l2_req_sm_id_r[sl];
        end
    endgenerate

    //------------------------------------------------------------------------
    // SM Ready Signal Generation
    //------------------------------------------------------------------------
    // SM is ready if its target slice can accept and it wins arbitration
    reg [NUM_SM-1:0] sm_req_ready_r;

    integer rdy_sm;
    always @(*) begin
        for (rdy_sm = 0; rdy_sm < NUM_SM; rdy_sm = rdy_sm + 1) begin
            sm_req_ready_r[rdy_sm] = slice_grant[sm_target_slice[rdy_sm]][rdy_sm] &&
                                    l2_req_ready[sm_target_slice[rdy_sm]];
        end
    end

    assign sm_req_ready = sm_req_ready_r;

    //------------------------------------------------------------------------
    // Response Crossbar - L2 Slice to SM Routing
    //------------------------------------------------------------------------
    // Route responses back to originating SM based on sm_id tag
    reg [NUM_SM-1:0] sm_resp_valid_r;
    reg [DATA_WIDTH-1:0] sm_resp_rdata_r [0:NUM_SM-1];
    reg [ID_WIDTH-1:0] sm_resp_id_r [0:NUM_SM-1];

    integer resp_sm, resp_sl;
    always @(*) begin
        for (resp_sm = 0; resp_sm < NUM_SM; resp_sm = resp_sm + 1) begin
            sm_resp_valid_r[resp_sm] = 0;
            sm_resp_rdata_r[resp_sm] = 0;
            sm_resp_id_r[resp_sm] = 0;

            for (resp_sl = 0; resp_sl < NUM_L2_SLICES; resp_sl = resp_sl + 1) begin
                if (l2_resp_valid[resp_sl] &&
                    l2_resp_sm_id[resp_sl*SM_W +: SM_W] == resp_sm) begin
                    sm_resp_valid_r[resp_sm] = 1'b1;
                    sm_resp_rdata_r[resp_sm] = l2_resp_rdata[resp_sl*DATA_WIDTH +: DATA_WIDTH];
                    sm_resp_id_r[resp_sm] = l2_resp_id[resp_sl*ID_WIDTH +: ID_WIDTH];
                end
            end
        end
    end

    // Pack response outputs
    generate
        for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : gen_sm_resp
            assign sm_resp_valid[sm] = sm_resp_valid_r[sm];
            assign sm_resp_rdata[sm*DATA_WIDTH +: DATA_WIDTH] = sm_resp_rdata_r[sm];
            assign sm_resp_id[sm*ID_WIDTH +: ID_WIDTH] = sm_resp_id_r[sm];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Statistics Tracking
    //------------------------------------------------------------------------
    reg [31:0] total_requests;
    reg [31:0] xbar_conflicts;
    reg [63:0] latency_sum;
    reg [31:0] latency_count;

    // Track per-request latency (simplified)
    reg [15:0] pending_latency [0:NUM_SM-1][0:MAX_OUTSTANDING-1];
    reg [OST_W-1:0] pending_head [0:NUM_SM-1];
    reg [OST_W-1:0] pending_tail [0:NUM_SM-1];

    integer stat_sm, stat_sl, stat_i;
    integer reqs_this_cycle;
    integer conflicts_this_cycle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_requests <= 0;
            xbar_conflicts <= 0;
            latency_sum <= 0;
            latency_count <= 0;
            for (stat_sm = 0; stat_sm < NUM_SM; stat_sm = stat_sm + 1) begin
                pending_head[stat_sm] <= 0;
                pending_tail[stat_sm] <= 0;
                for (stat_i = 0; stat_i < MAX_OUTSTANDING; stat_i = stat_i + 1) begin
                    pending_latency[stat_sm][stat_i] <= 0;
                end
            end
        end else begin
                        // Count requests and conflicts for this cycle.
            reqs_this_cycle = 0;
            conflicts_this_cycle = 0;
            for (stat_sm = 0; stat_sm < NUM_SM; stat_sm = stat_sm + 1) begin
                if (sm_req_valid[stat_sm] && sm_req_ready_r[stat_sm]) begin
                    reqs_this_cycle = reqs_this_cycle + 1;
                    // Track latency start
                    pending_latency[stat_sm][pending_tail[stat_sm]] <= 0;
                    pending_tail[stat_sm] <= (pending_tail[stat_sm] + 1) % MAX_OUTSTANDING;
                end
            end
            total_requests <= total_requests + reqs_this_cycle;

            for (stat_sl = 0; stat_sl < NUM_L2_SLICES; stat_sl = stat_sl + 1) begin
                if (($countones(slice_requestors[stat_sl])) > 1) begin
                    conflicts_this_cycle = conflicts_this_cycle + 1;
                end
            end
            xbar_conflicts <= xbar_conflicts + conflicts_this_cycle;

            // Update latency counters and track responses
            for (stat_sm = 0; stat_sm < NUM_SM; stat_sm = stat_sm + 1) begin
                // Increment all pending latencies
                for (stat_i = 0; stat_i < MAX_OUTSTANDING; stat_i = stat_i + 1) begin
                    if (pending_head[stat_sm] != pending_tail[stat_sm]) begin
                        pending_latency[stat_sm][stat_i] <= pending_latency[stat_sm][stat_i] + 1;
                    end
                end

                // On response, record latency
                if (sm_resp_valid_r[stat_sm]) begin
                    latency_sum <= latency_sum + pending_latency[stat_sm][pending_head[stat_sm]];
                    latency_count <= latency_count + 1;
                    pending_head[stat_sm] <= (pending_head[stat_sm] + 1) % MAX_OUTSTANDING;
                end
            end
        end
    end

    assign stat_total_requests = total_requests;
    assign stat_xbar_conflicts = xbar_conflicts;
    assign stat_avg_latency = (latency_count > 0) ? (latency_sum[31:0] / latency_count) : 0;

endmodule
