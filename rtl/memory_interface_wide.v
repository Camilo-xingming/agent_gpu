//============================================================================
// RalphGPU - Wide Memory Interface with Deep Memory-Level Parallelism
// Features:
//   - Wide memory lanes (512-bit)
//   - Multiple outstanding requests (MSHR-style tracking)
//   - Request coalescing across warps
//   - Deep MLP with request buffering
//   - Cross-lane aggregation
//============================================================================

`include "gpu_defines.vh"
`include "memory_config.vh"

module memory_interface_wide #(
    parameter NUM_LANES         = 4,            // Memory lanes
    parameter LANE_WIDTH        = 128,          // Bits per lane
    parameter NUM_WARPS         = 8,
    parameter THREADS_PER_WARP  = 32,
    parameter ADDR_WIDTH        = 32,
    parameter DATA_WIDTH        = 32,
    parameter MSHR_ENTRIES      = 32,           // Miss Status Holding Registers (NVIDIA-comparable)
    parameter MAX_OUTSTANDING   = 64,           // Max in-flight requests
    parameter COALESCE_WINDOW   = 4             // Cycles to coalesce
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // SM Load/Store Interface (per warp)
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]         warp_req_valid,
    input  wire [NUM_WARPS-1:0]         warp_req_write,
    input  wire [ADDR_WIDTH*NUM_WARPS-1:0] warp_req_addr,
    input  wire [DATA_WIDTH*THREADS_PER_WARP*NUM_WARPS-1:0] warp_req_wdata,
    input  wire [THREADS_PER_WARP*NUM_WARPS-1:0] warp_req_mask,
    output wire [NUM_WARPS-1:0]         warp_req_ready,

    output wire [NUM_WARPS-1:0]         warp_resp_valid,
    output wire [DATA_WIDTH*THREADS_PER_WARP*NUM_WARPS-1:0] warp_resp_rdata,

    //------------------------------------------------------------------------
    // L1 Cache / Coalescing Interface
    //------------------------------------------------------------------------
    output wire [NUM_LANES-1:0]         lane_req_valid,
    output wire [NUM_LANES-1:0]         lane_req_write,
    output wire [ADDR_WIDTH*NUM_LANES-1:0] lane_req_addr,
    output wire [LANE_WIDTH*NUM_LANES-1:0] lane_req_wdata,
    output wire [(LANE_WIDTH/8)*NUM_LANES-1:0] lane_req_wmask,
    input  wire [NUM_LANES-1:0]         lane_req_ready,

    input  wire [NUM_LANES-1:0]         lane_resp_valid,
    input  wire [LANE_WIDTH*NUM_LANES-1:0] lane_resp_rdata,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_requests,
    output wire [31:0]                  stat_coalesced,
    output wire [31:0]                  stat_outstanding_peak
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam WARP_DATA_WIDTH = DATA_WIDTH * THREADS_PER_WARP;
    localparam TOTAL_REQ_WIDTH = 1 + ADDR_WIDTH + WARP_DATA_WIDTH + THREADS_PER_WARP;
    localparam LANE_BYTES = LANE_WIDTH / 8;
    localparam COALESCE_BITS = $clog2(LANE_BYTES);
    localparam MSHR_IDX_WIDTH = $clog2(MSHR_ENTRIES);

    //------------------------------------------------------------------------
    // MSHR Entry Structure
    //------------------------------------------------------------------------
    // Tracks pending memory requests waiting for data
    reg [NUM_WARPS-1:0] mshr_warp_mask [0:MSHR_ENTRIES-1];
    reg [ADDR_WIDTH-1:0] mshr_base_addr [0:MSHR_ENTRIES-1];
    reg [THREADS_PER_WARP-1:0] mshr_thread_mask [0:MSHR_ENTRIES-1][0:NUM_WARPS-1];
    reg [MSHR_ENTRIES-1:0] mshr_valid;
    reg [MSHR_ENTRIES-1:0] mshr_waiting;  // Waiting for response
    reg [7:0] mshr_lane_idx [0:MSHR_ENTRIES-1];

    //------------------------------------------------------------------------
    // Request Coalescing
    //------------------------------------------------------------------------
    // Coalescing buffer - collects requests within window
    reg [NUM_WARPS-1:0] coalesce_warp_pending;
    reg [ADDR_WIDTH-1:0] coalesce_base [0:NUM_LANES-1];
    reg [NUM_LANES-1:0] coalesce_valid;
    reg [3:0] coalesce_counter;
    reg coalesce_active;

    // Per-lane request aggregation
    reg [LANE_WIDTH-1:0] lane_wdata_buf [0:NUM_LANES-1];
    reg [LANE_BYTES-1:0] lane_wmask_buf [0:NUM_LANES-1];
    reg [NUM_WARPS-1:0] lane_warp_contrib [0:NUM_LANES-1];

    //------------------------------------------------------------------------
    // Outstanding Request Tracking
    //------------------------------------------------------------------------
    reg [6:0] outstanding_count;
    reg [31:0] outstanding_peak;

    //------------------------------------------------------------------------
    // Address Coalescing Logic
    //------------------------------------------------------------------------
    // Determine which lane an address maps to
    function [1:0] addr_to_lane;
        input [ADDR_WIDTH-1:0] addr;
        begin
            addr_to_lane = addr[COALESCE_BITS +: 2];  // Use bits above cache line
        end
    endfunction

    // Get coalesced base address (cache line aligned)
    function [ADDR_WIDTH-1:0] align_addr;
        input [ADDR_WIDTH-1:0] addr;
        begin
            align_addr = {addr[ADDR_WIDTH-1:COALESCE_BITS], {COALESCE_BITS{1'b0}}};
        end
    endfunction

    // Check if address can coalesce with existing request
    function can_coalesce;
        input [ADDR_WIDTH-1:0] new_addr;
        input [ADDR_WIDTH-1:0] base_addr;
        begin
            can_coalesce = (align_addr(new_addr) == align_addr(base_addr));
        end
    endfunction

    //------------------------------------------------------------------------
    // Request Acceptance and Coalescing
    //------------------------------------------------------------------------
    wire [NUM_WARPS-1:0] warp_can_accept;
    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : gen_warp_ready
            assign warp_can_accept[w] = (outstanding_count < MAX_OUTSTANDING - 4) &&
                                        (|(~mshr_valid));
        end
    endgenerate
    assign warp_req_ready = warp_can_accept;

    // Coalescing state machine
    integer coal_w, coal_l, coal_t, coal_m;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            coalesce_active <= 0;
            coalesce_counter <= 0;
            coalesce_valid <= 0;
            coalesce_warp_pending <= 0;
            for (coal_l = 0; coal_l < NUM_LANES; coal_l = coal_l + 1) begin
                coalesce_base[coal_l] <= 0;
                lane_wdata_buf[coal_l] <= 0;
                lane_wmask_buf[coal_l] <= 0;
                lane_warp_contrib[coal_l] <= 0;
            end
        end else begin
            // Process incoming requests
            for (coal_w = 0; coal_w < NUM_WARPS; coal_w = coal_w + 1) begin
                if (warp_req_valid[coal_w] && warp_req_ready[coal_w]) begin
                    // Get base address for this warp's request
                    begin
                        reg [ADDR_WIDTH-1:0] warp_addr;
                        reg [1:0] lane_idx;
                        reg [ADDR_WIDTH-1:0] aligned;

                        warp_addr = warp_req_addr[coal_w*ADDR_WIDTH +: ADDR_WIDTH];
                        lane_idx = addr_to_lane(warp_addr);
                        aligned = align_addr(warp_addr);

                        if (!coalesce_active) begin
                            // Start new coalescing window
                            coalesce_active <= 1;
                            coalesce_counter <= COALESCE_WINDOW;
                            coalesce_base[lane_idx] <= aligned;
                            coalesce_valid[lane_idx] <= 1;
                            lane_warp_contrib[lane_idx][coal_w] <= 1;
                        end else if (coalesce_valid[lane_idx] &&
                                   can_coalesce(warp_addr, coalesce_base[lane_idx])) begin
                            // Coalesce with existing request
                            lane_warp_contrib[lane_idx][coal_w] <= 1;
                        end else if (!coalesce_valid[lane_idx]) begin
                            // Use empty lane
                            coalesce_base[lane_idx] <= aligned;
                            coalesce_valid[lane_idx] <= 1;
                            lane_warp_contrib[lane_idx][coal_w] <= 1;
                        end
                    end
                    coalesce_warp_pending[coal_w] <= 1;
                end
            end

            // Countdown coalescing window
            if (coalesce_active) begin
                if (coalesce_counter > 0) begin
                    coalesce_counter <= coalesce_counter - 1;
                end else begin
                    // End of window - issue coalesced requests
                    coalesce_active <= 0;
                    coalesce_valid <= 0;
                    coalesce_warp_pending <= 0;
                    for (coal_l = 0; coal_l < NUM_LANES; coal_l = coal_l + 1) begin
                        lane_warp_contrib[coal_l] <= 0;
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // MSHR Allocation and Management
    //------------------------------------------------------------------------
    // Find free MSHR entry
    reg [MSHR_IDX_WIDTH-1:0] free_mshr;
    reg free_mshr_valid;

    integer mshr_i;
    always @(*) begin
        free_mshr = 0;
        free_mshr_valid = 0;
        for (mshr_i = 0; mshr_i < MSHR_ENTRIES; mshr_i = mshr_i + 1) begin
            if (!mshr_valid[mshr_i] && !free_mshr_valid) begin
                free_mshr = mshr_i;
                free_mshr_valid = 1;
            end
        end
    end

    // MSHR allocation
    integer alloc_l, alloc_m, alloc_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mshr_valid <= 0;
            mshr_waiting <= 0;
            for (alloc_m = 0; alloc_m < MSHR_ENTRIES; alloc_m = alloc_m + 1) begin
                mshr_warp_mask[alloc_m] <= 0;
                mshr_base_addr[alloc_m] <= 0;
                mshr_lane_idx[alloc_m] <= 0;
                for (alloc_w = 0; alloc_w < NUM_WARPS; alloc_w = alloc_w + 1) begin
                    mshr_thread_mask[alloc_m][alloc_w] <= 0;
                end
            end
        end else begin
            // Allocate MSHRs when coalescing window ends
            if (coalesce_active && coalesce_counter == 0) begin
                for (alloc_l = 0; alloc_l < NUM_LANES; alloc_l = alloc_l + 1) begin
                    if (coalesce_valid[alloc_l] && free_mshr_valid) begin
                        mshr_valid[free_mshr] <= 1;
                        mshr_warp_mask[free_mshr] <= lane_warp_contrib[alloc_l];
                        mshr_base_addr[free_mshr] <= coalesce_base[alloc_l];
                        mshr_lane_idx[free_mshr] <= alloc_l;
                    end
                end
            end

            // Handle responses - deallocate MSHRs
            for (alloc_l = 0; alloc_l < NUM_LANES; alloc_l = alloc_l + 1) begin
                if (lane_resp_valid[alloc_l]) begin
                    for (alloc_m = 0; alloc_m < MSHR_ENTRIES; alloc_m = alloc_m + 1) begin
                        if (mshr_valid[alloc_m] && mshr_waiting[alloc_m] &&
                            mshr_lane_idx[alloc_m] == alloc_l) begin
                            mshr_valid[alloc_m] <= 0;
                            mshr_waiting[alloc_m] <= 0;
                        end
                    end
                end
            end

            // Mark MSHRs as waiting when request sent
            for (alloc_l = 0; alloc_l < NUM_LANES; alloc_l = alloc_l + 1) begin
                if (lane_req_valid[alloc_l] && lane_req_ready[alloc_l]) begin
                    for (alloc_m = 0; alloc_m < MSHR_ENTRIES; alloc_m = alloc_m + 1) begin
                        if (mshr_valid[alloc_m] && !mshr_waiting[alloc_m] &&
                            mshr_lane_idx[alloc_m] == alloc_l) begin
                            mshr_waiting[alloc_m] <= 1;
                        end
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Lane Request Generation
    //------------------------------------------------------------------------
    reg [NUM_LANES-1:0] lane_req_valid_r;
    reg [NUM_LANES-1:0] lane_req_write_r;
    reg [ADDR_WIDTH-1:0] lane_req_addr_r [0:NUM_LANES-1];
    reg [LANE_WIDTH-1:0] lane_req_wdata_r [0:NUM_LANES-1];
    reg [LANE_BYTES-1:0] lane_req_wmask_r [0:NUM_LANES-1];

    integer lane_i, lane_m;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lane_req_valid_r <= 0;
            lane_req_write_r <= 0;
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                lane_req_addr_r[lane_i] <= 0;
                lane_req_wdata_r[lane_i] <= 0;
                lane_req_wmask_r[lane_i] <= 0;
            end
        end else begin
            lane_req_valid_r <= 0;

            // Issue requests from MSHRs
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                for (lane_m = 0; lane_m < MSHR_ENTRIES; lane_m = lane_m + 1) begin
                    if (mshr_valid[lane_m] && !mshr_waiting[lane_m] &&
                        mshr_lane_idx[lane_m] == lane_i && !lane_req_valid_r[lane_i]) begin
                        lane_req_valid_r[lane_i] <= 1;
                        lane_req_write_r[lane_i] <= 0;  // Simplified: reads only for now
                        lane_req_addr_r[lane_i] <= mshr_base_addr[lane_m];
                    end
                end
            end
        end
    end

    // Output assignments
    genvar li;
    generate
        for (li = 0; li < NUM_LANES; li = li + 1) begin : gen_lane_out
            assign lane_req_valid[li] = lane_req_valid_r[li];
            assign lane_req_write[li] = lane_req_write_r[li];
            assign lane_req_addr[li*ADDR_WIDTH +: ADDR_WIDTH] = lane_req_addr_r[li];
            assign lane_req_wdata[li*LANE_WIDTH +: LANE_WIDTH] = lane_req_wdata_r[li];
            assign lane_req_wmask[li*LANE_BYTES +: LANE_BYTES] = lane_req_wmask_r[li];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Response Distribution to Warps
    //------------------------------------------------------------------------
    reg [NUM_WARPS-1:0] warp_resp_valid_r;
    reg [WARP_DATA_WIDTH-1:0] warp_resp_rdata_r [0:NUM_WARPS-1];

    integer resp_w, resp_l, resp_m, resp_t;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_resp_valid_r <= 0;
            for (resp_w = 0; resp_w < NUM_WARPS; resp_w = resp_w + 1) begin
                warp_resp_rdata_r[resp_w] <= 0;
            end
        end else begin
            warp_resp_valid_r <= 0;

            // Distribute lane responses to warps
            for (resp_l = 0; resp_l < NUM_LANES; resp_l = resp_l + 1) begin
                if (lane_resp_valid[resp_l]) begin
                    for (resp_m = 0; resp_m < MSHR_ENTRIES; resp_m = resp_m + 1) begin
                        if (mshr_valid[resp_m] && mshr_waiting[resp_m] &&
                            mshr_lane_idx[resp_m] == resp_l) begin
                            // Distribute to all warps that contributed
                            for (resp_w = 0; resp_w < NUM_WARPS; resp_w = resp_w + 1) begin
                                if (mshr_warp_mask[resp_m][resp_w]) begin
                                    warp_resp_valid_r[resp_w] <= 1;
                                    // Extract relevant data for this warp
                                    for (resp_t = 0; resp_t < THREADS_PER_WARP; resp_t = resp_t + 1) begin
                                        warp_resp_rdata_r[resp_w][resp_t*DATA_WIDTH +: DATA_WIDTH] <=
                                            lane_resp_rdata[resp_l*LANE_WIDTH +: DATA_WIDTH];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    // Output assignments
    assign warp_resp_valid = warp_resp_valid_r;
    genvar wi;
    generate
        for (wi = 0; wi < NUM_WARPS; wi = wi + 1) begin : gen_warp_resp
            assign warp_resp_rdata[wi*WARP_DATA_WIDTH +: WARP_DATA_WIDTH] = warp_resp_rdata_r[wi];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Outstanding Request Tracking
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            outstanding_count <= 0;
            outstanding_peak <= 0;
        end else begin
            // Count changes
            begin
                reg [3:0] new_reqs;
                reg [3:0] completed_reqs;
                integer oi;

                new_reqs = 0;
                completed_reqs = 0;

                for (oi = 0; oi < NUM_LANES; oi = oi + 1) begin
                    if (lane_req_valid[oi] && lane_req_ready[oi])
                        new_reqs = new_reqs + 1;
                    if (lane_resp_valid[oi])
                        completed_reqs = completed_reqs + 1;
                end

                outstanding_count <= outstanding_count + new_reqs - completed_reqs;

                if (outstanding_count + new_reqs - completed_reqs > outstanding_peak)
                    outstanding_peak <= outstanding_count + new_reqs - completed_reqs;
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] total_requests;
    reg [31:0] coalesced_requests;

    assign stat_requests = total_requests;
    assign stat_coalesced = coalesced_requests;
    assign stat_outstanding_peak = outstanding_peak;

    integer stat_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_requests <= 0;
            coalesced_requests <= 0;
        end else begin
            for (stat_w = 0; stat_w < NUM_WARPS; stat_w = stat_w + 1) begin
                if (warp_req_valid[stat_w] && warp_req_ready[stat_w])
                    total_requests <= total_requests + 1;
            end
        end
    end

endmodule
