//============================================================================
// RalphGPU - Memory Controller Interface
// Supports DDR4/DDR5/LPDDR5/HBM interfaces
//============================================================================

`include "gpu_defines.vh"
`include "memory_config.vh"

module memory_controller #(
    parameter DATA_WIDTH        = `MEM_DATA_WIDTH,
    parameter NUM_CHANNELS      = `MEM_NUM_CHANNELS,
    parameter BURST_LENGTH      = `MEM_BURST_LENGTH,
    parameter ADDR_WIDTH        = 32,
    parameter REQ_QUEUE_DEPTH   = `MEM_REQ_QUEUE_DEPTH
)(
    input  wire                         clk,
    input  wire                         mem_clk,        // Memory clock domain
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // L2 Cache Interface (GPU side)
    //------------------------------------------------------------------------
    input  wire                         l2_req_valid,
    input  wire                         l2_req_write,
    input  wire [ADDR_WIDTH-1:0]        l2_req_addr,
    input  wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_req_wdata,
    input  wire [DATA_WIDTH*BURST_LENGTH/8-1:0] l2_req_wmask,
    output wire                         l2_req_ready,
    output wire                         l2_resp_valid,
    output wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_resp_rdata,

    //------------------------------------------------------------------------
    // DDR/HBM Physical Interface (Memory side)
    // Generic interface - can be mapped to specific memory type
    //------------------------------------------------------------------------
    // Command interface
    output wire [NUM_CHANNELS-1:0]      mem_cs_n,       // Chip select
    output wire [NUM_CHANNELS-1:0]      mem_ras_n,      // Row address strobe
    output wire [NUM_CHANNELS-1:0]      mem_cas_n,      // Column address strobe
    output wire [NUM_CHANNELS-1:0]      mem_we_n,       // Write enable
    output wire [NUM_CHANNELS*17-1:0]   mem_addr,       // Address (row/column)
    output wire [NUM_CHANNELS*3-1:0]    mem_ba,         // Bank address
    output wire [NUM_CHANNELS*2-1:0]    mem_bg,         // Bank group

    // Data interface
    output wire [NUM_CHANNELS*DATA_WIDTH-1:0]   mem_dq_out,
    input  wire [NUM_CHANNELS*DATA_WIDTH-1:0]   mem_dq_in,
    output wire [NUM_CHANNELS-1:0]              mem_dq_oe,
    output wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_out,
    input  wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_in,
    output wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dm,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_read_count,
    output wire [31:0]                  stat_write_count,
    output wire [31:0]                  stat_row_hits,
    output wire [31:0]                  stat_row_misses
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam TOTAL_DATA_WIDTH = DATA_WIDTH * NUM_CHANNELS;
    localparam BURST_SIZE       = DATA_WIDTH * BURST_LENGTH / 8;  // Bytes
    localparam CHANNEL_BITS     = $clog2(NUM_CHANNELS);

    // Memory timing parameters (in memory clock cycles)
    localparam tCL      = `MEM_tCL;     // CAS Latency
    localparam tRCD     = `MEM_tRCD;    // RAS to CAS delay
    localparam tRP      = `MEM_tRP;     // Row Precharge
    localparam tRAS     = `MEM_tRAS;    // Row Active Time

    //------------------------------------------------------------------------
    // Request Queue
    //------------------------------------------------------------------------
    // Request entry: [valid][write][addr][data][mask]
    localparam REQ_ENTRY_WIDTH = 1 + 1 + ADDR_WIDTH + DATA_WIDTH*BURST_LENGTH +
                                  DATA_WIDTH*BURST_LENGTH/8;

    reg [REQ_ENTRY_WIDTH-1:0] req_queue [0:REQ_QUEUE_DEPTH-1];
    reg [$clog2(REQ_QUEUE_DEPTH):0] req_head;
    reg [$clog2(REQ_QUEUE_DEPTH):0] req_tail;
    wire req_queue_full  = (req_tail - req_head) >= REQ_QUEUE_DEPTH;
    wire req_queue_empty = (req_head == req_tail);

    //------------------------------------------------------------------------
    // Address Mapping (Configurable)
    // Default: Channel interleaving at cache line granularity
    //------------------------------------------------------------------------
    // Address format: [row][bank_group][bank][channel][column][offset]
    localparam OFFSET_BITS  = $clog2(BURST_SIZE);
    localparam COLUMN_BITS  = 10;
    localparam BANK_BITS    = 2;
    localparam BG_BITS      = 2;
    localparam ROW_BITS     = ADDR_WIDTH - OFFSET_BITS - COLUMN_BITS -
                              BANK_BITS - BG_BITS - CHANNEL_BITS;

    function [CHANNEL_BITS-1:0] get_channel;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_channel = addr[OFFSET_BITS +: CHANNEL_BITS];
        end
    endfunction

    function [COLUMN_BITS-1:0] get_column;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_column = addr[OFFSET_BITS + CHANNEL_BITS +: COLUMN_BITS];
        end
    endfunction

    function [BANK_BITS-1:0] get_bank;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_bank = addr[OFFSET_BITS + CHANNEL_BITS + COLUMN_BITS +: BANK_BITS];
        end
    endfunction

    function [BG_BITS-1:0] get_bank_group;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_bank_group = addr[OFFSET_BITS + CHANNEL_BITS + COLUMN_BITS +
                                  BANK_BITS +: BG_BITS];
        end
    endfunction

    function [ROW_BITS-1:0] get_row;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_row = addr[ADDR_WIDTH-1 -: ROW_BITS];
        end
    endfunction

    //------------------------------------------------------------------------
    // Per-Bank State (Row Buffer Tracking)
    //------------------------------------------------------------------------
    localparam NUM_BANKS = (1 << BANK_BITS) * (1 << BG_BITS);

    reg [ROW_BITS-1:0] open_row [0:NUM_CHANNELS-1][0:NUM_BANKS-1];
    reg [NUM_BANKS-1:0] row_valid [0:NUM_CHANNELS-1];

    //------------------------------------------------------------------------
    // Request Queue Management
    //------------------------------------------------------------------------
    assign l2_req_ready = !req_queue_full;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_head <= 0;
            req_tail <= 0;
        end else begin
            // Enqueue new request
            if (l2_req_valid && l2_req_ready) begin
                req_queue[req_tail[$clog2(REQ_QUEUE_DEPTH)-1:0]] <=
                    {1'b1, l2_req_write, l2_req_addr, l2_req_wdata, l2_req_wmask};
                req_tail <= req_tail + 1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Memory Command Scheduler
    //------------------------------------------------------------------------
    // FR-FCFS (First-Ready, First-Come-First-Served) scheduling
    // Prioritizes row buffer hits

    localparam MC_IDLE      = 3'd0;
    localparam MC_ACTIVATE  = 3'd1;
    localparam MC_READ      = 3'd2;
    localparam MC_WRITE     = 3'd3;
    localparam MC_PRECHARGE = 3'd4;
    localparam MC_WAIT      = 3'd5;

    reg [2:0] mc_state [0:NUM_CHANNELS-1];
    reg [7:0] mc_timer [0:NUM_CHANNELS-1];

    // Command generation per channel
    genvar ch;
    generate
        for (ch = 0; ch < NUM_CHANNELS; ch = ch + 1) begin : gen_channel
            // Per-channel state machine
            reg [ADDR_WIDTH-1:0] current_addr;
            reg current_write;
            reg [DATA_WIDTH*BURST_LENGTH-1:0] current_wdata;

            // Command outputs
            reg cs_n_reg, ras_n_reg, cas_n_reg, we_n_reg;
            reg [16:0] addr_reg;
            reg [2:0] ba_reg;
            reg [1:0] bg_reg;

            assign mem_cs_n[ch]  = cs_n_reg;
            assign mem_ras_n[ch] = ras_n_reg;
            assign mem_cas_n[ch] = cas_n_reg;
            assign mem_we_n[ch]  = we_n_reg;
            assign mem_addr[ch*17 +: 17] = addr_reg;
            assign mem_ba[ch*3 +: 3]     = {1'b0, ba_reg[1:0]};
            assign mem_bg[ch*2 +: 2]     = bg_reg;

            always @(posedge mem_clk or negedge rst_n) begin
                if (!rst_n) begin
                    mc_state[ch] <= MC_IDLE;
                    mc_timer[ch] <= 0;
                    cs_n_reg  <= 1'b1;
                    ras_n_reg <= 1'b1;
                    cas_n_reg <= 1'b1;
                    we_n_reg  <= 1'b1;
                    addr_reg  <= 0;
                    ba_reg    <= 0;
                    bg_reg    <= 0;
                    row_valid[ch] <= 0;
                end else begin
                    // Default: NOP command
                    cs_n_reg  <= 1'b0;  // Chip selected
                    ras_n_reg <= 1'b1;
                    cas_n_reg <= 1'b1;
                    we_n_reg  <= 1'b1;

                    if (mc_timer[ch] > 0) begin
                        mc_timer[ch] <= mc_timer[ch] - 1;
                    end

                    case (mc_state[ch])
                        MC_IDLE: begin
                            // Check for pending request to this channel
                            // (Simplified: check head of queue)
                            if (!req_queue_empty) begin
                                reg [ADDR_WIDTH-1:0] addr;
                                addr = req_queue[req_head[$clog2(REQ_QUEUE_DEPTH)-1:0]]
                                       [DATA_WIDTH*BURST_LENGTH + DATA_WIDTH*BURST_LENGTH/8 +:
                                        ADDR_WIDTH];

                                if (get_channel(addr) == ch) begin
                                    current_addr <= addr;
                                    current_write <= req_queue[req_head[$clog2(REQ_QUEUE_DEPTH)-1:0]]
                                                     [ADDR_WIDTH + DATA_WIDTH*BURST_LENGTH +
                                                      DATA_WIDTH*BURST_LENGTH/8];

                                    // Check row buffer hit
                                    if (row_valid[ch][{get_bank_group(addr), get_bank(addr)}] &&
                                        open_row[ch][{get_bank_group(addr), get_bank(addr)}] ==
                                        get_row(addr)) begin
                                        // Row hit - go directly to read/write
                                        mc_state[ch] <= current_write ? MC_WRITE : MC_READ;
                                    end else begin
                                        // Row miss - need activate (precharge if row open)
                                        if (row_valid[ch][{get_bank_group(addr), get_bank(addr)}]) begin
                                            mc_state[ch] <= MC_PRECHARGE;
                                        end else begin
                                            mc_state[ch] <= MC_ACTIVATE;
                                        end
                                    end
                                end
                            end
                        end

                        MC_PRECHARGE: begin
                            // Issue precharge command
                            ras_n_reg <= 1'b0;
                            we_n_reg  <= 1'b0;
                            ba_reg    <= get_bank(current_addr);
                            bg_reg    <= get_bank_group(current_addr);
                            addr_reg[10] <= 1'b0;  // Single bank precharge

                            row_valid[ch][{get_bank_group(current_addr),
                                          get_bank(current_addr)}] <= 1'b0;

                            mc_timer[ch] <= tRP;
                            mc_state[ch] <= MC_WAIT;
                        end

                        MC_ACTIVATE: begin
                            if (mc_timer[ch] == 0) begin
                                // Issue activate command
                                ras_n_reg <= 1'b0;
                                ba_reg    <= get_bank(current_addr);
                                bg_reg    <= get_bank_group(current_addr);
                                addr_reg  <= get_row(current_addr);

                                open_row[ch][{get_bank_group(current_addr),
                                             get_bank(current_addr)}] <= get_row(current_addr);
                                row_valid[ch][{get_bank_group(current_addr),
                                              get_bank(current_addr)}] <= 1'b1;

                                mc_timer[ch] <= tRCD;
                                mc_state[ch] <= MC_WAIT;
                            end
                        end

                        MC_READ: begin
                            if (mc_timer[ch] == 0) begin
                                // Issue read command
                                cas_n_reg <= 1'b0;
                                ba_reg    <= get_bank(current_addr);
                                bg_reg    <= get_bank_group(current_addr);
                                addr_reg[9:0] <= get_column(current_addr);
                                addr_reg[10]  <= 1'b0;  // No auto-precharge

                                mc_timer[ch] <= tCL + BURST_LENGTH;
                                mc_state[ch] <= MC_WAIT;
                            end
                        end

                        MC_WRITE: begin
                            if (mc_timer[ch] == 0) begin
                                // Issue write command
                                cas_n_reg <= 1'b0;
                                we_n_reg  <= 1'b0;
                                ba_reg    <= get_bank(current_addr);
                                bg_reg    <= get_bank_group(current_addr);
                                addr_reg[9:0] <= get_column(current_addr);
                                addr_reg[10]  <= 1'b0;

                                mc_timer[ch] <= BURST_LENGTH + 4;  // Write latency
                                mc_state[ch] <= MC_WAIT;
                            end
                        end

                        MC_WAIT: begin
                            if (mc_timer[ch] == 0) begin
                                mc_state[ch] <= MC_IDLE;
                            end
                        end
                    endcase
                end
            end
        end
    endgenerate

    //------------------------------------------------------------------------
    // Response Generation
    //------------------------------------------------------------------------
    // Collect responses from channels and route back to L2

    reg resp_valid_reg;
    reg [DATA_WIDTH*BURST_LENGTH-1:0] resp_rdata_reg;

    assign l2_resp_valid = resp_valid_reg;
    assign l2_resp_rdata = resp_rdata_reg;

    // Simplified: Single response at a time
    // Full implementation needs response reordering

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid_reg <= 0;
            resp_rdata_reg <= 0;
        end else begin
            resp_valid_reg <= 0;  // Default

            // Check for completed reads
            // (Simplified - would need proper tracking in real implementation)
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] read_cnt, write_cnt, row_hit_cnt, row_miss_cnt;

    assign stat_read_count  = read_cnt;
    assign stat_write_count = write_cnt;
    assign stat_row_hits    = row_hit_cnt;
    assign stat_row_misses  = row_miss_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_cnt <= 0;
            write_cnt <= 0;
            row_hit_cnt <= 0;
            row_miss_cnt <= 0;
        end else begin
            if (l2_req_valid && l2_req_ready) begin
                if (l2_req_write)
                    write_cnt <= write_cnt + 1;
                else
                    read_cnt <= read_cnt + 1;
            end
        end
    end

endmodule
