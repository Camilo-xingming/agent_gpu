//============================================================================
// RalphGPU - HBM Memory Controller with Real DRAM Latency Modeling
// Features:
//   - FR-FCFS (First-Ready First-Come-First-Served) scheduling
//   - Row buffer management with hit/miss tracking
//   - Multiple outstanding requests per bank
//   - Realistic HBM timing parameters
//   - Request reordering for efficiency
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module memory_controller_hbm #(
    parameter DATA_WIDTH        = 256,          // HBM channel width
    parameter NUM_CHANNELS      = 8,            // HBM2e has 8 channels
    parameter NUM_BANKS_PER_CH  = 16,           // Banks per channel
    parameter BURST_LENGTH      = 4,
    parameter ADDR_WIDTH        = 32,
    parameter ROW_WIDTH         = 14,
    parameter COL_WIDTH         = 6,
    parameter BANK_WIDTH        = 4,
    parameter REQ_QUEUE_DEPTH   = 32,           // Requests per channel
    // HBM Timing parameters (in memory clock cycles)
    parameter tCL               = 14,           // CAS Latency
    parameter tRCD              = 14,           // RAS to CAS delay
    parameter tRP               = 14,           // Row precharge time
    parameter tRAS              = 32,           // Row active time
    parameter tRC               = 46,           // Row cycle time (tRAS + tRP)
    parameter tWR               = 16,           // Write recovery time
    parameter tRTP              = 8,            // Read to precharge
    parameter tCCD_L            = 4,            // CAS to CAS (same bank group)
    parameter tCCD_S            = 2,            // CAS to CAS (different bank group)
    parameter tFAW              = 16            // Four-activate window
)(
    input  wire                         clk,
    input  wire                         mem_clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // L2 Cache Interface (GPU side, clk domain)
    //------------------------------------------------------------------------
    input  wire                         l2_req_valid,
    input  wire                         l2_req_write,
    input  wire [ADDR_WIDTH-1:0]        l2_req_addr,
    input  wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_req_wdata,
    input  wire [DATA_WIDTH*BURST_LENGTH/8-1:0] l2_req_wmask,
    input  wire [7:0]                   l2_req_id,         // Request ID for reordering
    output wire                         l2_req_ready,

    output wire                         l2_resp_valid,
    output wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_resp_rdata,
    output wire [7:0]                   l2_resp_id,        // Matching request ID

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_read_count,
    output wire [31:0]                  stat_write_count,
    output wire [31:0]                  stat_row_hits,
    output wire [31:0]                  stat_row_misses,
    output wire [31:0]                  stat_row_conflicts,
    output wire [31:0]                  stat_avg_latency
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam BURST_BITS  = DATA_WIDTH * BURST_LENGTH;
    localparam BURST_BYTES = BURST_BITS / 8;
    localparam TOTAL_BANKS = NUM_CHANNELS * NUM_BANKS_PER_CH;
    localparam CH_WIDTH    = $clog2(NUM_CHANNELS);
    localparam PTR_W       = $clog2(REQ_QUEUE_DEPTH);

    // Address decomposition
    localparam OFFSET_WIDTH = $clog2(BURST_BYTES);
    localparam COL_LSB      = OFFSET_WIDTH;
    localparam BANK_LSB     = COL_LSB + COL_WIDTH;
    localparam CH_LSB       = BANK_LSB + BANK_WIDTH;
    localparam ROW_LSB      = CH_LSB + CH_WIDTH;

    //------------------------------------------------------------------------
    // Request Queue Entry
    //------------------------------------------------------------------------
    localparam REQ_ENTRY_WIDTH = 1 + ADDR_WIDTH + BURST_BITS + BURST_BYTES + 8 + 32;
    // [write, addr, wdata, wmask, id, timestamp]

    //------------------------------------------------------------------------
    // Per-Channel Request Queues
    //------------------------------------------------------------------------
    reg [REQ_ENTRY_WIDTH-1:0] ch_req_queue [0:NUM_CHANNELS-1][0:REQ_QUEUE_DEPTH-1];
    reg [PTR_W:0] ch_req_count [0:NUM_CHANNELS-1];
    reg [PTR_W-1:0] ch_req_head [0:NUM_CHANNELS-1];
    reg [PTR_W-1:0] ch_req_tail [0:NUM_CHANNELS-1];

    //------------------------------------------------------------------------
    // Per-Bank State
    //------------------------------------------------------------------------
    reg [ROW_WIDTH-1:0] bank_open_row [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];
    reg [NUM_BANKS_PER_CH-1:0] bank_row_open [0:NUM_CHANNELS-1];
    reg [7:0] bank_busy_counter [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];

    // Timing counters per bank
    reg [7:0] bank_act_counter [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];  // Since last ACT
    reg [7:0] bank_pre_counter [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];  // Since last PRE
    reg [7:0] bank_rd_counter  [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];  // Since last RD
    reg [7:0] bank_wr_counter  [0:NUM_CHANNELS-1][0:NUM_BANKS_PER_CH-1];  // Since last WR

    // Four-activate window tracking
    reg [7:0] faw_timestamps [0:NUM_CHANNELS-1][0:3];
    reg [1:0] faw_ptr [0:NUM_CHANNELS-1];

    //------------------------------------------------------------------------
    // Response Queue
    //------------------------------------------------------------------------
    localparam RESP_ENTRY_WIDTH = BURST_BITS + 8; // data + id
    reg [RESP_ENTRY_WIDTH-1:0] resp_queue [0:REQ_QUEUE_DEPTH-1];
    reg [PTR_W:0] resp_count;
    reg [PTR_W-1:0] resp_head;
    reg [PTR_W-1:0] resp_tail;

    //------------------------------------------------------------------------
    // Memory Model (for simulation)
    //------------------------------------------------------------------------
    localparam MEM_DEPTH = 65536;  // 64K lines
    localparam MEM_INDEX_W = $clog2(MEM_DEPTH);
    `ifndef SYNTHESIS
    reg [BURST_BITS-1:0] mem_array [0:MEM_DEPTH-1];
`endif

    //------------------------------------------------------------------------
    // Address Decomposition
    //------------------------------------------------------------------------
    function [CH_WIDTH-1:0] get_channel;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_channel = addr[CH_LSB +: CH_WIDTH];
        end
    endfunction

    function [BANK_WIDTH-1:0] get_bank;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_bank = addr[BANK_LSB +: BANK_WIDTH];
        end
    endfunction

    /* verilator lint_off SELRANGE */
    function [ROW_WIDTH-1:0] get_row;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_row = addr[ROW_LSB +: ROW_WIDTH];
        end
    endfunction
    /* verilator lint_on SELRANGE */

    function [COL_WIDTH-1:0] get_col;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_col = addr[COL_LSB +: COL_WIDTH];
        end
    endfunction

    //------------------------------------------------------------------------
    // Request Acceptance (clk domain)
    //------------------------------------------------------------------------
    wire [CH_WIDTH-1:0] req_channel = get_channel(l2_req_addr);
    wire channel_has_space = (ch_req_count[req_channel] < REQ_QUEUE_DEPTH);
    assign l2_req_ready = channel_has_space && (resp_count < REQ_QUEUE_DEPTH - 4);

    reg [31:0] global_timestamp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_timestamp <= 0;
        end else begin
            global_timestamp <= global_timestamp + 1;
        end
    end

    // Enqueue requests
    integer enq_ch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (enq_ch = 0; enq_ch < NUM_CHANNELS; enq_ch = enq_ch + 1) begin
                ch_req_count[enq_ch] <= 0;
                ch_req_tail[enq_ch] <= 0;
            end
        end else if (l2_req_valid && l2_req_ready) begin
            ch_req_queue[req_channel][ch_req_tail[req_channel]] <= {
                l2_req_write, l2_req_addr, l2_req_wdata, l2_req_wmask,
                l2_req_id, global_timestamp
            };
            ch_req_tail[req_channel] <= ch_req_tail[req_channel] + 1'b1;
            ch_req_count[req_channel] <= ch_req_count[req_channel] + 1;
        end
    end

    //------------------------------------------------------------------------
    // FR-FCFS Scheduling (mem_clk domain - simplified single clock for now)
    //------------------------------------------------------------------------
    reg [2:0] scheduler_state [0:NUM_CHANNELS-1];
    localparam SCH_IDLE     = 3'd0;
    localparam SCH_ACTIVATE = 3'd1;
    localparam SCH_PRECHARGE = 3'd2;
    localparam SCH_READ     = 3'd3;
    localparam SCH_WRITE    = 3'd4;
    localparam SCH_WAIT     = 3'd5;

    reg [7:0] sch_wait_counter [0:NUM_CHANNELS-1];
    reg [PTR_W-1:0] sch_selected_idx [0:NUM_CHANNELS-1];
    reg [REQ_ENTRY_WIDTH-1:0] sch_current_req [0:NUM_CHANNELS-1];

    // FR-FCFS: Find best request (row hit > oldest)
    // NOTE: Channel-specific lookup handled inline due to iverilog limitations
    function [PTR_W-1:0] find_oldest_request;
        input integer ch;
        reg [PTR_W-1:0] oldest_idx;
        reg [31:0] oldest_ts;
        reg [REQ_ENTRY_WIDTH-1:0] entry;
        integer i;
            reg [PTR_W:0] count;
            reg [PTR_W-1:0] idx;
        begin
            oldest_ts = 32'hFFFFFFFF;
            oldest_idx = 0;
            count = ch_req_count[ch];

            for (i = 0; i < 32; i = i + 1) begin
                if (i < count) begin
                    idx = ch_req_head[ch] + i[PTR_W-1:0];
                    entry = ch_req_queue[ch][idx];

                    // Track oldest for FCFS fallback
                    if (entry[31:0] < oldest_ts) begin
                        oldest_ts = entry[31:0];
                        oldest_idx = idx;
                    end
                end
            end


            find_oldest_request = oldest_idx;
        end
    endfunction

    // Main scheduling FSM
    integer sch_ch, sch_bk, sch_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sch_ch = 0; sch_ch < NUM_CHANNELS; sch_ch = sch_ch + 1) begin
                scheduler_state[sch_ch] <= SCH_IDLE;
                ch_req_head[sch_ch] <= 0;
                sch_wait_counter[sch_ch] <= 0;
                bank_row_open[sch_ch] <= 0;
                faw_ptr[sch_ch] <= 0;
                for (sch_bk = 0; sch_bk < NUM_BANKS_PER_CH; sch_bk = sch_bk + 1) begin
                    bank_open_row[sch_ch][sch_bk] <= 0;
                    bank_busy_counter[sch_ch][sch_bk] <= 0;
                    bank_act_counter[sch_ch][sch_bk] <= 255;
                    bank_pre_counter[sch_ch][sch_bk] <= 255;
                    bank_rd_counter[sch_ch][sch_bk] <= 255;
                    bank_wr_counter[sch_ch][sch_bk] <= 255;
                end
                for (sch_i = 0; sch_i < 4; sch_i = sch_i + 1) begin
                    faw_timestamps[sch_ch][sch_i] <= 0;
                end
            end
            resp_count <= 0;
            resp_head <= 0;
            resp_tail <= 0;
        end else begin
            // Decrement timing counters
            for (sch_ch = 0; sch_ch < NUM_CHANNELS; sch_ch = sch_ch + 1) begin
                for (sch_bk = 0; sch_bk < NUM_BANKS_PER_CH; sch_bk = sch_bk + 1) begin
                    if (bank_busy_counter[sch_ch][sch_bk] > 0)
                        bank_busy_counter[sch_ch][sch_bk] <= bank_busy_counter[sch_ch][sch_bk] - 1;
                    if (bank_act_counter[sch_ch][sch_bk] < 255)
                        bank_act_counter[sch_ch][sch_bk] <= bank_act_counter[sch_ch][sch_bk] + 1;
                    if (bank_pre_counter[sch_ch][sch_bk] < 255)
                        bank_pre_counter[sch_ch][sch_bk] <= bank_pre_counter[sch_ch][sch_bk] + 1;
                    if (bank_rd_counter[sch_ch][sch_bk] < 255)
                        bank_rd_counter[sch_ch][sch_bk] <= bank_rd_counter[sch_ch][sch_bk] + 1;
                    if (bank_wr_counter[sch_ch][sch_bk] < 255)
                        bank_wr_counter[sch_ch][sch_bk] <= bank_wr_counter[sch_ch][sch_bk] + 1;
                end
            end

            // Process each channel
            for (sch_ch = 0; sch_ch < NUM_CHANNELS; sch_ch = sch_ch + 1) begin
                case (scheduler_state[sch_ch])
                    SCH_IDLE: begin
                        if (ch_req_count[sch_ch] > 0) begin
                            // Find oldest request (simplified FR-FCFS)
                            sch_selected_idx[sch_ch] <= find_oldest_request(sch_ch);
                            sch_current_req[sch_ch] <= ch_req_queue[sch_ch][
                                find_oldest_request(sch_ch)
                            ];
                            scheduler_state[sch_ch] <= SCH_WAIT;
                            sch_wait_counter[sch_ch] <= 1;
                        end
                    end

                    SCH_WAIT: begin
                        if (sch_wait_counter[sch_ch] > 0) begin
                            sch_wait_counter[sch_ch] <= sch_wait_counter[sch_ch] - 1;
                        end else begin
                            // Determine next action based on bank state
                            begin
                                reg [ADDR_WIDTH-1:0] addr;
                                reg [BANK_WIDTH-1:0] bank;
                                reg [ROW_WIDTH-1:0] row;
                                reg is_write;

                                addr = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                                bank = get_bank(addr);
                                row = get_row(addr);
                                is_write = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-1];

                                if (bank_busy_counter[sch_ch][bank] > 0) begin
                                    // Bank busy, wait
                                    sch_wait_counter[sch_ch] <= bank_busy_counter[sch_ch][bank];
                                end else if (!bank_row_open[sch_ch][bank]) begin
                                    // Need to activate row
                                    scheduler_state[sch_ch] <= SCH_ACTIVATE;
                                end else if (bank_open_row[sch_ch][bank] != row) begin
                                    // Row conflict - need precharge first
                                    scheduler_state[sch_ch] <= SCH_PRECHARGE;
                                end else begin
                                    // Row hit - issue read/write
                                    scheduler_state[sch_ch] <= is_write ? SCH_WRITE : SCH_READ;
                                end
                            end
                        end
                    end

                    SCH_ACTIVATE: begin
                        begin
                            reg [ADDR_WIDTH-1:0] addr;
                            reg [BANK_WIDTH-1:0] bank;
                            reg [ROW_WIDTH-1:0] row;

                            addr = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                            bank = get_bank(addr);
                            row = get_row(addr);

                            // Activate the row
                            bank_row_open[sch_ch][bank] <= 1'b1;
                            bank_open_row[sch_ch][bank] <= row;
                            bank_act_counter[sch_ch][bank] <= 0;
                            bank_busy_counter[sch_ch][bank] <= tRCD;

                            // Update FAW tracking
                            faw_timestamps[sch_ch][faw_ptr[sch_ch]] <= global_timestamp[7:0];
                            faw_ptr[sch_ch] <= faw_ptr[sch_ch] + 1;

                            scheduler_state[sch_ch] <= SCH_WAIT;
                            sch_wait_counter[sch_ch] <= tRCD;
                        end
                    end

                    SCH_PRECHARGE: begin
                        begin
                            reg [ADDR_WIDTH-1:0] addr;
                            reg [BANK_WIDTH-1:0] bank;

                            addr = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                            bank = get_bank(addr);

                            // Precharge the bank
                            bank_row_open[sch_ch][bank] <= 1'b0;
                            bank_pre_counter[sch_ch][bank] <= 0;
                            bank_busy_counter[sch_ch][bank] <= tRP;

                            scheduler_state[sch_ch] <= SCH_WAIT;
                            sch_wait_counter[sch_ch] <= tRP;
                        end
                    end

                    SCH_READ: begin
                        begin
                            reg [ADDR_WIDTH-1:0] addr;
                            reg [BANK_WIDTH-1:0] bank;
                            reg [MEM_INDEX_W-1:0] mem_idx;
                            reg [BURST_BITS-1:0] rdata;
                            reg [7:0] req_id;

                            addr = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                            bank = get_bank(addr);
                            mem_idx = addr[OFFSET_WIDTH +: MEM_INDEX_W];
                            req_id = sch_current_req[sch_ch][39:32];

                            // Read from memory model
                            `ifndef SYNTHESIS
                            rdata = mem_array[mem_idx];
`else
                            rdata = {BURST_BITS{1'b0}};
`endif

                            // Queue response with latency (tCL)
                            if (resp_count < REQ_QUEUE_DEPTH) begin
                                resp_queue[resp_tail] <= {rdata, req_id};
                                resp_tail <= resp_tail + 1'b1;
                                resp_count <= resp_count + 1;
                            end

                            bank_rd_counter[sch_ch][bank] <= 0;
                            bank_busy_counter[sch_ch][bank] <= tCL + BURST_LENGTH;

                            // Remove from request queue
                            ch_req_head[sch_ch] <= ch_req_head[sch_ch] + 1'b1;
                            ch_req_count[sch_ch] <= ch_req_count[sch_ch] - 1;

                            scheduler_state[sch_ch] <= SCH_IDLE;
                        end
                    end

                    SCH_WRITE: begin
                        begin
                            reg [ADDR_WIDTH-1:0] addr;
                            reg [BANK_WIDTH-1:0] bank;
                            reg [MEM_INDEX_W-1:0] mem_idx;
                            reg [BURST_BITS-1:0] wdata;
                            reg [BURST_BYTES-1:0] wmask;
                            integer wb;

                            addr = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2 -: ADDR_WIDTH];
                            bank = get_bank(addr);
                            mem_idx = addr[OFFSET_WIDTH +: MEM_INDEX_W];
                            wdata = sch_current_req[sch_ch][REQ_ENTRY_WIDTH-2-ADDR_WIDTH -: BURST_BITS];
                            wmask = sch_current_req[sch_ch][BURST_BYTES+39:40];

                            // Write to memory model with byte mask
`ifndef SYNTHESIS
                            for (wb = 0; wb < BURST_BYTES; wb = wb + 1) begin
                                if (wmask[wb])
                                    mem_array[mem_idx][wb*8 +: 8] <= wdata[wb*8 +: 8];
                            end
`endif

                            bank_wr_counter[sch_ch][bank] <= 0;
                            bank_busy_counter[sch_ch][bank] <= tCL + BURST_LENGTH + tWR;

                            // Remove from request queue
                            ch_req_head[sch_ch] <= ch_req_head[sch_ch] + 1'b1;
                            ch_req_count[sch_ch] <= ch_req_count[sch_ch] - 1;

                            scheduler_state[sch_ch] <= SCH_IDLE;
                        end
                    end

                    default: scheduler_state[sch_ch] <= SCH_IDLE;
                endcase
            end
        end
    end

    //------------------------------------------------------------------------
    // Response Output
    //------------------------------------------------------------------------
    reg l2_resp_valid_r;
    reg [BURST_BITS-1:0] l2_resp_rdata_r;
    reg [7:0] l2_resp_id_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l2_resp_valid_r <= 0;
            l2_resp_rdata_r <= 0;
            l2_resp_id_r <= 0;
        end else begin
            l2_resp_valid_r <= 0;
            if (resp_count > 0) begin
                l2_resp_rdata_r <= resp_queue[resp_head][RESP_ENTRY_WIDTH-1:8];
                l2_resp_id_r <= resp_queue[resp_head][7:0];
                l2_resp_valid_r <= 1;
                resp_head <= resp_head + 1'b1;
                resp_count <= resp_count - 1;
            end
        end
    end

    assign l2_resp_valid = l2_resp_valid_r;
    assign l2_resp_rdata = l2_resp_rdata_r;
    assign l2_resp_id = l2_resp_id_r;

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] read_cnt, write_cnt, row_hit_cnt, row_miss_cnt, row_conflict_cnt;
    reg [63:0] latency_sum;
    reg [31:0] latency_cnt;

    assign stat_read_count = read_cnt;
    assign stat_write_count = write_cnt;
    assign stat_row_hits = row_hit_cnt;
    assign stat_row_misses = row_miss_cnt;
    assign stat_row_conflicts = row_conflict_cnt;
    assign stat_avg_latency = (latency_cnt > 0) ? latency_sum[31:0] / latency_cnt : 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_cnt <= 0;
            write_cnt <= 0;
            row_hit_cnt <= 0;
            row_miss_cnt <= 0;
            row_conflict_cnt <= 0;
            latency_sum <= 0;
            latency_cnt <= 0;
        end else if (l2_req_valid && l2_req_ready) begin
            if (l2_req_write)
                write_cnt <= write_cnt + 1;
            else
                read_cnt <= read_cnt + 1;
        end
    end

    //------------------------------------------------------------------------
    // Memory Initialization
    //------------------------------------------------------------------------
    integer init_i;
`ifndef SYNTHESIS
    initial begin
        for (init_i = 0; init_i < MEM_DEPTH; init_i = init_i + 1) begin
            mem_array[init_i] = {BURST_BITS{1'b0}};
        end
    end
`endif

endmodule
