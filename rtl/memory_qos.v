//============================================================================
// RalphGPU - Memory QoS and Bandwidth Management
// Features:
//   - Per-SM bandwidth allocation
//   - Priority-based arbitration
//   - Fairness enforcement
//   - Latency-sensitive request prioritization
//   - Bandwidth throttling for power management
//============================================================================

`include "gpu_defines.vh"

module memory_qos #(
    parameter NUM_SMS           = 4,
    parameter NUM_CHANNELS      = 8,
    parameter ADDR_WIDTH        = 32,
    parameter DATA_WIDTH        = 512,
    parameter REQ_QUEUE_DEPTH   = 16,
    parameter BANDWIDTH_BITS    = 16,           // Bandwidth counter width
    parameter PRIORITY_LEVELS   = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // SM Request Interfaces (NUM_SMS ports)
    //------------------------------------------------------------------------
    input  wire [NUM_SMS-1:0]           sm_req_valid,
    input  wire [NUM_SMS-1:0]           sm_req_write,
    input  wire [ADDR_WIDTH*NUM_SMS-1:0] sm_req_addr,
    input  wire [DATA_WIDTH*NUM_SMS-1:0] sm_req_wdata,
    input  wire [2*NUM_SMS-1:0]         sm_req_priority,    // 2-bit priority per SM
    input  wire [NUM_SMS-1:0]           sm_req_latency_sensitive,
    output wire [NUM_SMS-1:0]           sm_req_ready,

    output wire [NUM_SMS-1:0]           sm_resp_valid,
    output wire [DATA_WIDTH*NUM_SMS-1:0] sm_resp_rdata,

    //------------------------------------------------------------------------
    // Memory Channel Interfaces (NUM_CHANNELS ports)
    //------------------------------------------------------------------------
    output wire [NUM_CHANNELS-1:0]      ch_req_valid,
    output wire [NUM_CHANNELS-1:0]      ch_req_write,
    output wire [ADDR_WIDTH*NUM_CHANNELS-1:0] ch_req_addr,
    output wire [DATA_WIDTH*NUM_CHANNELS-1:0] ch_req_wdata,
    output wire [$clog2(NUM_SMS)*NUM_CHANNELS-1:0] ch_req_source,
    input  wire [NUM_CHANNELS-1:0]      ch_req_ready,

    input  wire [NUM_CHANNELS-1:0]      ch_resp_valid,
    input  wire [DATA_WIDTH*NUM_CHANNELS-1:0] ch_resp_rdata,
    input  wire [$clog2(NUM_SMS)*NUM_CHANNELS-1:0] ch_resp_source,

    //------------------------------------------------------------------------
    // QoS Configuration
    //------------------------------------------------------------------------
    input  wire [BANDWIDTH_BITS*NUM_SMS-1:0] cfg_bandwidth_limit,  // Per-SM limit
    input  wire [7:0]                   cfg_fairness_window,        // Cycles per window
    input  wire                         cfg_throttle_enable,
    input  wire [7:0]                   cfg_throttle_level,         // 0-255

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_total_requests,
    output wire [31:0]                  stat_throttled_requests,
    output wire [31:0]                  stat_priority_inversions,
    output wire [BANDWIDTH_BITS*NUM_SMS-1:0] stat_sm_bandwidth
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam SM_WIDTH = (NUM_SMS > 1) ? $clog2(NUM_SMS) : 1;
    localparam CH_WIDTH = (NUM_CHANNELS > 1) ? $clog2(NUM_CHANNELS) : 1;
    localparam PTR_WIDTH = (REQ_QUEUE_DEPTH > 1) ? $clog2(REQ_QUEUE_DEPTH) : 1;

    //------------------------------------------------------------------------
    // Per-SM Request Queues
    //------------------------------------------------------------------------
    localparam REQ_ENTRY_WIDTH = 1 + ADDR_WIDTH + DATA_WIDTH + 2 + 1;
    // [write, addr, wdata, priority, latency_sensitive]

    reg [REQ_ENTRY_WIDTH-1:0] sm_queue [0:NUM_SMS-1][0:REQ_QUEUE_DEPTH-1];
    reg [PTR_WIDTH:0] sm_queue_count [0:NUM_SMS-1];
    reg [PTR_WIDTH-1:0] sm_queue_head [0:NUM_SMS-1];
    reg [PTR_WIDTH-1:0] sm_queue_tail [0:NUM_SMS-1];

    //------------------------------------------------------------------------
    // Bandwidth Tracking (per SM, per window)
    //------------------------------------------------------------------------
    reg [BANDWIDTH_BITS-1:0] sm_bandwidth [0:NUM_SMS-1];
    reg [7:0] window_counter;
    wire window_reset = (window_counter >= cfg_fairness_window);

    //------------------------------------------------------------------------
    // Fairness Tracking
    //------------------------------------------------------------------------
    reg [15:0] sm_service_count [0:NUM_SMS-1];  // Requests serviced this window
    reg [SM_WIDTH-1:0] last_serviced_sm;

    //------------------------------------------------------------------------
    // Throttling State
    //------------------------------------------------------------------------
    reg [7:0] throttle_counter;
    wire throttle_active = cfg_throttle_enable && (throttle_counter < cfg_throttle_level);

    //------------------------------------------------------------------------
    // Request Acceptance
    //------------------------------------------------------------------------
    wire [NUM_SMS-1:0] queue_has_space;
    wire [NUM_SMS-1:0] bandwidth_available;
    wire [NUM_SMS-1:0] can_accept;

    genvar sm;
    generate
        for (sm = 0; sm < NUM_SMS; sm = sm + 1) begin : gen_accept
            assign queue_has_space[sm] = (sm_queue_count[sm] < REQ_QUEUE_DEPTH);
            assign bandwidth_available[sm] = (sm_bandwidth[sm] <
                cfg_bandwidth_limit[sm*BANDWIDTH_BITS +: BANDWIDTH_BITS]);
            assign can_accept[sm] = queue_has_space[sm] && bandwidth_available[sm] &&
                                   !throttle_active;
        end
    endgenerate

    assign sm_req_ready = can_accept;

    // Enqueue requests
    integer enq_sm;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (enq_sm = 0; enq_sm < NUM_SMS; enq_sm = enq_sm + 1) begin
                sm_queue_count[enq_sm] <= 0;
                sm_queue_head[enq_sm] <= 0;
                sm_queue_tail[enq_sm] <= 0;
            end
        end else begin
            for (enq_sm = 0; enq_sm < NUM_SMS; enq_sm = enq_sm + 1) begin
                if (sm_req_valid[enq_sm] && can_accept[enq_sm]) begin
                    sm_queue[enq_sm][sm_queue_tail[enq_sm]] <= {
                        sm_req_write[enq_sm],
                        sm_req_addr[enq_sm*ADDR_WIDTH +: ADDR_WIDTH],
                        sm_req_wdata[enq_sm*DATA_WIDTH +: DATA_WIDTH],
                        sm_req_priority[enq_sm*2 +: 2],
                        sm_req_latency_sensitive[enq_sm]
                    };
                    sm_queue_tail[enq_sm] <= (sm_queue_tail[enq_sm] + 1) % REQ_QUEUE_DEPTH;
                    sm_queue_count[enq_sm] <= sm_queue_count[enq_sm] + 1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Priority-based Arbitration with Fairness
    //------------------------------------------------------------------------
    // Find highest priority request across all SMs
    reg [SM_WIDTH-1:0] selected_sm;
    reg selected_valid;
    reg [1:0] selected_priority;
    reg selected_latency_sensitive;

    // Temporary variables for combinational logic (declared at module level)
    reg [REQ_ENTRY_WIDTH-1:0] arb_entry;
    reg [1:0] arb_priority;
    reg arb_lat_sens;
    reg arb_fair_turn;
    reg [SM_WIDTH-1:0] arb_check_sm;

    integer arb_sm, arb_p;
    always @(*) begin
        selected_sm = 0;
        selected_valid = 0;
        selected_priority = 0;
        selected_latency_sensitive = 0;
        arb_entry = 0;
        arb_priority = 0;
        arb_lat_sens = 0;
        arb_fair_turn = 0;
        arb_check_sm = 0;

        // First pass: find latency-sensitive requests
        for (arb_sm = 0; arb_sm < NUM_SMS; arb_sm = arb_sm + 1) begin
            if (sm_queue_count[arb_sm] > 0) begin
                arb_entry = sm_queue[arb_sm][sm_queue_head[arb_sm]];
                arb_priority = arb_entry[DATA_WIDTH +: 2];
                arb_lat_sens = arb_entry[0];

                if (arb_lat_sens && !selected_latency_sensitive) begin
                    selected_sm = arb_sm[SM_WIDTH-1:0];
                    selected_valid = 1;
                    selected_priority = arb_priority;
                    selected_latency_sensitive = 1;
                end else if (arb_lat_sens && arb_priority > selected_priority) begin
                    selected_sm = arb_sm[SM_WIDTH-1:0];
                    selected_priority = arb_priority;
                end
            end
        end

        // Second pass: if no latency-sensitive, use priority + fairness
        if (!selected_latency_sensitive) begin
            for (arb_p = PRIORITY_LEVELS - 1; arb_p >= 0; arb_p = arb_p - 1) begin
                for (arb_sm = 0; arb_sm < NUM_SMS; arb_sm = arb_sm + 1) begin
                    if (sm_queue_count[arb_sm] > 0 && !selected_valid) begin
                        arb_entry = sm_queue[arb_sm][sm_queue_head[arb_sm]];
                        arb_priority = arb_entry[DATA_WIDTH +: 2];

                        // Fairness: prefer SM that hasn't been serviced recently
                        arb_fair_turn = (arb_sm[SM_WIDTH-1:0] != last_serviced_sm) ||
                                   (sm_service_count[arb_sm] <
                                    sm_service_count[(arb_sm + 1) % NUM_SMS]);

                        if (arb_priority == arb_p[1:0] && arb_fair_turn) begin
                            selected_sm = arb_sm[SM_WIDTH-1:0];
                            selected_valid = 1;
                            selected_priority = arb_priority;
                        end
                    end
                end
            end

            // Fallback: round-robin if no fair selection
            if (!selected_valid) begin
                for (arb_sm = 0; arb_sm < NUM_SMS; arb_sm = arb_sm + 1) begin
                    arb_check_sm = (last_serviced_sm + 1 + arb_sm[SM_WIDTH-1:0]) % NUM_SMS;
                    if (sm_queue_count[arb_check_sm] > 0 && !selected_valid) begin
                        selected_sm = arb_check_sm;
                        selected_valid = 1;
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Channel Mapping and Request Issue
    //------------------------------------------------------------------------
    // Map address to channel (interleaved)
    function [CH_WIDTH-1:0] addr_to_channel;
        input [ADDR_WIDTH-1:0] addr;
        begin
            addr_to_channel = addr[6 +: CH_WIDTH];  // Interleave at cache line granularity
        end
    endfunction

    reg [NUM_CHANNELS-1:0] ch_req_valid_r;
    reg [NUM_CHANNELS-1:0] ch_req_write_r;
    reg [ADDR_WIDTH-1:0] ch_req_addr_r [0:NUM_CHANNELS-1];
    reg [DATA_WIDTH-1:0] ch_req_wdata_r [0:NUM_CHANNELS-1];
    reg [SM_WIDTH-1:0] ch_req_source_r [0:NUM_CHANNELS-1];

    // Temporary variables for sequential logic
    reg [REQ_ENTRY_WIDTH-1:0] iss_entry;
    reg [ADDR_WIDTH-1:0] iss_addr;
    reg [CH_WIDTH-1:0] iss_channel;

    integer iss_ch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch_req_valid_r <= 0;
            ch_req_write_r <= 0;
            for (iss_ch = 0; iss_ch < NUM_CHANNELS; iss_ch = iss_ch + 1) begin
                ch_req_addr_r[iss_ch] <= 0;
                ch_req_wdata_r[iss_ch] <= 0;
                ch_req_source_r[iss_ch] <= 0;
            end
            last_serviced_sm <= 0;
            iss_entry <= 0;
            iss_addr <= 0;
            iss_channel <= 0;
        end else begin
            ch_req_valid_r <= 0;

            if (selected_valid && !throttle_active) begin
                iss_entry = sm_queue[selected_sm][sm_queue_head[selected_sm]];
                iss_addr = iss_entry[REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                iss_channel = addr_to_channel(iss_addr);

                if (ch_req_ready[iss_channel]) begin
                    ch_req_valid_r[iss_channel] <= 1;
                    ch_req_write_r[iss_channel] <= iss_entry[REQ_ENTRY_WIDTH-1];
                    ch_req_addr_r[iss_channel] <= iss_addr;
                    ch_req_wdata_r[iss_channel] <= iss_entry[REQ_ENTRY_WIDTH-2-ADDR_WIDTH -: DATA_WIDTH];
                    ch_req_source_r[iss_channel] <= selected_sm;

                    // Dequeue
                    sm_queue_head[selected_sm] <=
                        (sm_queue_head[selected_sm] + 1) % REQ_QUEUE_DEPTH;
                    sm_queue_count[selected_sm] <= sm_queue_count[selected_sm] - 1;

                    // Update tracking
                    last_serviced_sm <= selected_sm;
                    sm_service_count[selected_sm] <= sm_service_count[selected_sm] + 1;
                    sm_bandwidth[selected_sm] <= sm_bandwidth[selected_sm] + 1;
                end
            end
        end
    end

    // Output assignments
    genvar chi;
    generate
        for (chi = 0; chi < NUM_CHANNELS; chi = chi + 1) begin : gen_ch_out
            assign ch_req_valid[chi] = ch_req_valid_r[chi];
            assign ch_req_write[chi] = ch_req_write_r[chi];
            assign ch_req_addr[chi*ADDR_WIDTH +: ADDR_WIDTH] = ch_req_addr_r[chi];
            assign ch_req_wdata[chi*DATA_WIDTH +: DATA_WIDTH] = ch_req_wdata_r[chi];
            assign ch_req_source[chi*SM_WIDTH +: SM_WIDTH] = ch_req_source_r[chi];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Response Routing
    //------------------------------------------------------------------------
    reg [NUM_SMS-1:0] sm_resp_valid_r;
    reg [DATA_WIDTH-1:0] sm_resp_rdata_r [0:NUM_SMS-1];

    // Temporary variable for response routing
    reg [SM_WIDTH-1:0] resp_source;

    integer resp_ch, resp_sm;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_resp_valid_r <= 0;
            for (resp_sm = 0; resp_sm < NUM_SMS; resp_sm = resp_sm + 1) begin
                sm_resp_rdata_r[resp_sm] <= 0;
            end
            resp_source <= 0;
        end else begin
            sm_resp_valid_r <= 0;

            for (resp_ch = 0; resp_ch < NUM_CHANNELS; resp_ch = resp_ch + 1) begin
                if (ch_resp_valid[resp_ch]) begin
                    resp_source = ch_resp_source[resp_ch*SM_WIDTH +: SM_WIDTH];
                    sm_resp_valid_r[resp_source] <= 1;
                    sm_resp_rdata_r[resp_source] <= ch_resp_rdata[resp_ch*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

    assign sm_resp_valid = sm_resp_valid_r;
    genvar smi;
    generate
        for (smi = 0; smi < NUM_SMS; smi = smi + 1) begin : gen_sm_resp
            assign sm_resp_rdata[smi*DATA_WIDTH +: DATA_WIDTH] = sm_resp_rdata_r[smi];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Window and Throttle Management
    //------------------------------------------------------------------------
    integer win_sm;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_counter <= 0;
            throttle_counter <= 0;
            for (win_sm = 0; win_sm < NUM_SMS; win_sm = win_sm + 1) begin
                sm_bandwidth[win_sm] <= 0;
                sm_service_count[win_sm] <= 0;
            end
        end else begin
            // Window counter
            if (window_reset) begin
                window_counter <= 0;
                for (win_sm = 0; win_sm < NUM_SMS; win_sm = win_sm + 1) begin
                    sm_bandwidth[win_sm] <= 0;
                    sm_service_count[win_sm] <= 0;
                end
            end else begin
                window_counter <= window_counter + 1;
            end

            // Throttle counter (wraps at 256)
            throttle_counter <= throttle_counter + 1;
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] total_req_count;
    reg [31:0] throttled_count;
    reg [31:0] priority_inversion_count;

    assign stat_total_requests = total_req_count;
    assign stat_throttled_requests = throttled_count;
    assign stat_priority_inversions = priority_inversion_count;

    genvar stat_sm;
    generate
        for (stat_sm = 0; stat_sm < NUM_SMS; stat_sm = stat_sm + 1) begin : gen_bw_stat
            assign stat_sm_bandwidth[stat_sm*BANDWIDTH_BITS +: BANDWIDTH_BITS] =
                sm_bandwidth[stat_sm];
        end
    endgenerate

    integer stat_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_req_count <= 0;
            throttled_count <= 0;
            priority_inversion_count <= 0;
        end else begin
            for (stat_s = 0; stat_s < NUM_SMS; stat_s = stat_s + 1) begin
                if (sm_req_valid[stat_s] && can_accept[stat_s])
                    total_req_count <= total_req_count + 1;
                if (sm_req_valid[stat_s] && !can_accept[stat_s] && throttle_active)
                    throttled_count <= throttled_count + 1;
            end
        end
    end

endmodule
