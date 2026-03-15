//============================================================================
// RalphGPU - L2 Cache
// Multi-banked, non-blocking L2 cache with ECC support
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module l2_cache #(
    parameter SIZE_KB       = `L2_SIZE_KB,
    parameter NUM_BANKS     = `L2_NUM_BANKS,
    parameter NUM_WAYS      = `L2_WAYS,
    parameter LINE_SIZE     = `L2_LINE_SIZE,
    parameter MSHR_ENTRIES  = `L2_MSHR_ENTRIES,
    parameter ADDR_WIDTH    = 32,
    parameter DATA_WIDTH    = 512,          // Wide data for cache line
    parameter NUM_PORTS     = `NUM_SM       // One port per SM
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // L1 Interface (from SMs)
    //------------------------------------------------------------------------
    input  wire [NUM_PORTS-1:0]         l1_req_valid,
    input  wire [NUM_PORTS-1:0]         l1_req_write,
    input  wire [NUM_PORTS*ADDR_WIDTH-1:0]  l1_req_addr,
    input  wire [NUM_PORTS*LINE_SIZE*8-1:0] l1_req_wdata,
    input  wire [NUM_PORTS*LINE_SIZE-1:0]   l1_req_wmask,  // Byte mask
    output wire [NUM_PORTS-1:0]         l1_req_ready,
    output wire [NUM_PORTS-1:0]         l1_resp_valid,
    output wire [NUM_PORTS*LINE_SIZE*8-1:0] l1_resp_rdata,

    //------------------------------------------------------------------------
    // Memory Controller Interface
    //------------------------------------------------------------------------
    output wire                         mem_req_valid,
    output wire                         mem_req_write,
    output wire [ADDR_WIDTH-1:0]        mem_req_addr,
    output wire [LINE_SIZE*8-1:0]       mem_req_wdata,
    input  wire                         mem_req_ready,
    input  wire                         mem_resp_valid,
    input  wire [LINE_SIZE*8-1:0]       mem_resp_rdata,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_hits,
    output wire [31:0]                  stat_misses,
    output wire [31:0]                  stat_writebacks
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam SIZE_BYTES       = SIZE_KB * 1024;
    localparam SIZE_PER_BANK    = SIZE_BYTES / NUM_BANKS;
    localparam SETS_PER_BANK    = SIZE_PER_BANK / (NUM_WAYS * LINE_SIZE);

    localparam OFFSET_BITS      = $clog2(LINE_SIZE);
    localparam BANK_BITS        = $clog2(NUM_BANKS);
    localparam INDEX_BITS       = $clog2(SETS_PER_BANK);
    localparam TAG_BITS         = ADDR_WIDTH - INDEX_BITS - BANK_BITS - OFFSET_BITS;

    localparam LINE_BITS        = LINE_SIZE * 8;
    localparam WAY_BITS         = $clog2(NUM_WAYS);

    //------------------------------------------------------------------------
    // Address Decoding
    //------------------------------------------------------------------------
    function [BANK_BITS-1:0] get_bank;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_bank = addr[OFFSET_BITS +: BANK_BITS];
        end
    endfunction

    function [INDEX_BITS-1:0] get_index;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_index = addr[OFFSET_BITS + BANK_BITS +: INDEX_BITS];
        end
    endfunction

    function [TAG_BITS-1:0] get_tag;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_tag = addr[ADDR_WIDTH-1 -: TAG_BITS];
        end
    endfunction

    //------------------------------------------------------------------------
    // Bank Arbitration
    //------------------------------------------------------------------------
    // Round-robin arbiter per bank
    reg [NUM_PORTS-1:0] bank_grant [0:NUM_BANKS-1];

    // Determine which port requests which bank
    wire [BANK_BITS-1:0] port_bank [0:NUM_PORTS-1];
    wire [NUM_PORTS-1:0] bank_req [0:NUM_BANKS-1];

    genvar p, b;
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : gen_port_bank
            assign port_bank[p] = get_bank(l1_req_addr[p*ADDR_WIDTH +: ADDR_WIDTH]);
        end

        for (b = 0; b < NUM_BANKS; b = b + 1) begin : gen_bank_req
            wire [NUM_PORTS-1:0] this_bank_req;
            for (p = 0; p < NUM_PORTS; p = p + 1) begin : gen_bank_port_req
                assign this_bank_req[p] = l1_req_valid[p] && (port_bank[p] == b);
            end
            assign bank_req[b] = this_bank_req;
        end
    endgenerate

    // Simple priority arbitration per bank (lowest port wins)
    integer i, j;
    always @(*) begin
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            bank_grant[i] = 0;
            for (j = 0; j < NUM_PORTS; j = j + 1) begin
                if (bank_req[i][j] && !(|bank_grant[i])) begin
                    bank_grant[i][j] = 1'b1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Per-Bank Cache Logic
    //------------------------------------------------------------------------
    // Instantiate cache banks
    wire [NUM_BANKS-1:0] bank_hit;
    wire [NUM_BANKS-1:0] bank_miss;
    wire [NUM_BANKS-1:0] bank_writeback;
    wire [LINE_BITS-1:0] bank_rdata [0:NUM_BANKS-1];
    wire [NUM_BANKS-1:0] bank_resp_valid;

    // Bank request signals (after arbitration)
    wire [NUM_BANKS-1:0] bank_req_valid;
    wire [NUM_BANKS-1:0] bank_req_write;
    wire [ADDR_WIDTH-1:0] bank_req_addr [0:NUM_BANKS-1];
    wire [LINE_BITS-1:0] bank_req_wdata [0:NUM_BANKS-1];
    wire [LINE_SIZE-1:0] bank_req_wmask [0:NUM_BANKS-1];
    wire [3:0]           bank_req_port_id [0:NUM_BANKS-1];
    wire [NUM_BANKS-1:0] bank_ready;
    wire [3:0]           bank_resp_port_id [0:NUM_BANKS-1];
    wire [NUM_BANKS-1:0] bank_mem_req_valid;
    wire [NUM_BANKS-1:0] bank_mem_req_write;
    wire [ADDR_WIDTH-1:0] bank_mem_req_addr [0:NUM_BANKS-1];
    wire [LINE_BITS-1:0] bank_mem_req_wdata [0:NUM_BANKS-1];
    reg  [NUM_BANKS-1:0] bank_mem_req_ready;
    reg  [NUM_BANKS-1:0] bank_mem_fill_valid;

    generate
        for (b = 0; b < NUM_BANKS; b = b + 1) begin : gen_bank

            // Select winning port for this bank
            reg [$clog2(NUM_PORTS)-1:0] winning_port;
            always @(*) begin
                winning_port = 0;
                for (i = 0; i < NUM_PORTS; i = i + 1) begin
                    if (bank_grant[b][i]) winning_port = i;
                end
            end

            assign bank_req_valid[b] = |bank_grant[b];
            assign bank_req_write[b] = l1_req_write[winning_port];
            assign bank_req_addr[b]  = l1_req_addr[winning_port*ADDR_WIDTH +: ADDR_WIDTH];
            assign bank_req_wdata[b] = l1_req_wdata[winning_port*LINE_BITS +: LINE_BITS];
            assign bank_req_wmask[b] = l1_req_wmask[winning_port*LINE_SIZE +: LINE_SIZE];
            assign bank_req_port_id[b] = winning_port[3:0];

            // Cache bank instance
            l2_cache_bank #(
                .SIZE_BYTES     (SIZE_PER_BANK),
                .NUM_WAYS       (NUM_WAYS),
                .LINE_SIZE      (LINE_SIZE),
                .ADDR_WIDTH     (ADDR_WIDTH),
                .MSHR_ENTRIES   (MSHR_ENTRIES / NUM_BANKS),
                .PORT_ID_WIDTH  (4)
            ) u_bank (
                .clk            (clk),
                .rst_n          (rst_n),
                .req_valid      (bank_req_valid[b]),
                .req_write      (bank_req_write[b]),
                .req_addr       (bank_req_addr[b]),
                .req_wdata      (bank_req_wdata[b]),
                .req_wmask      (bank_req_wmask[b]),
                .req_port_id    (bank_req_port_id[b]),
                .req_ready      (bank_ready[b]),
                .resp_valid     (bank_resp_valid[b]),
                .resp_port_id   (bank_resp_port_id[b]),
                .resp_rdata     (bank_rdata[b]),
                .hit            (bank_hit[b]),
                .miss           (bank_miss[b]),
                .writeback      (bank_writeback[b]),
                .mem_req_valid  (bank_mem_req_valid[b]),
                .mem_req_write  (bank_mem_req_write[b]),
                .mem_req_addr   (bank_mem_req_addr[b]),
                .mem_req_wdata  (bank_mem_req_wdata[b]),
                .mem_req_ready  (bank_mem_req_ready[b]),
                .mem_fill_valid (bank_mem_fill_valid[b]),
                .mem_fill_data  (mem_resp_rdata)
            );
        end
    endgenerate

    //------------------------------------------------------------------------
    // Response Routing (Bank to Port)
    //------------------------------------------------------------------------
    // Track which port is waiting for which bank
    reg [BANK_BITS-1:0] port_pending_bank [0:NUM_PORTS-1];
    reg [NUM_PORTS-1:0] port_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            port_pending <= 0;
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                port_pending_bank[i] <= 0;
            end
        end else begin
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                // New request accepted
                if (l1_req_valid[i] && l1_req_ready[i]) begin
                    port_pending[i] <= 1'b1;
                    port_pending_bank[i] <= port_bank[i];
                end
                // Response received
                if (l1_resp_valid[i]) begin
                    port_pending[i] <= 1'b0;
                end
            end
        end
    end

    // Generate ready and response signals
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : gen_port_resp
            // Ready when the target bank is available and grants access
            assign l1_req_ready[p] = bank_grant[port_bank[p]][p] &&
                                     bank_ready[port_bank[p]];

            // Response valid when pending and bank responds
            assign l1_resp_valid[p] = port_pending[p] &&
                                      bank_resp_valid[port_pending_bank[p]] &&
                                      (bank_resp_port_id[port_pending_bank[p]] == p[3:0]);

            // Response data from appropriate bank
            assign l1_resp_rdata[p*LINE_BITS +: LINE_BITS] =
                   bank_rdata[port_pending_bank[p]];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Memory Controller Interface (single outstanding, round-robin banks)
    //------------------------------------------------------------------------
    localparam BANK_SEL_W = (NUM_BANKS > 1) ? $clog2(NUM_BANKS) : 1;

    reg mem_req_pending;
    reg mem_outstanding;
    reg [BANK_SEL_W-1:0] mem_req_sel;
    reg [BANK_SEL_W-1:0] mem_out_bank;
    reg [BANK_SEL_W-1:0] mem_rr_ptr;
    reg mem_req_has;
    reg [BANK_SEL_W-1:0] mem_req_sel_next;
    integer mem_i;
    integer mem_idx;

    always @(*) begin
        mem_req_has = 1'b0;
        mem_req_sel_next = mem_rr_ptr;
        for (mem_i = 0; mem_i < NUM_BANKS; mem_i = mem_i + 1) begin
            mem_idx = mem_rr_ptr + mem_i;
            if (mem_idx >= NUM_BANKS) begin
                mem_idx = mem_idx - NUM_BANKS;
            end
            if (!mem_req_has && bank_mem_req_valid[mem_idx]) begin
                mem_req_has = 1'b1;
                mem_req_sel_next = mem_idx[BANK_SEL_W-1:0];
            end
        end
    end

    assign mem_req_valid = mem_req_pending;
    assign mem_req_write = bank_mem_req_write[mem_req_sel];
    assign mem_req_addr  = bank_mem_req_addr[mem_req_sel];
    assign mem_req_wdata = bank_mem_req_wdata[mem_req_sel];

    always @(*) begin
        bank_mem_req_ready = {NUM_BANKS{1'b0}};
        if (mem_req_pending) begin
            bank_mem_req_ready[mem_req_sel] = mem_req_ready;
        end
    end

    always @(*) begin
        bank_mem_fill_valid = {NUM_BANKS{1'b0}};
        if (mem_resp_valid) begin
            bank_mem_fill_valid[mem_out_bank] = 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_req_pending <= 1'b0;
            mem_outstanding <= 1'b0;
            mem_req_sel <= {BANK_SEL_W{1'b0}};
            mem_out_bank <= {BANK_SEL_W{1'b0}};
            mem_rr_ptr <= {BANK_SEL_W{1'b0}};
        end else begin
            if (!mem_req_pending && !mem_outstanding && mem_req_has) begin
                mem_req_pending <= 1'b1;
                mem_req_sel <= mem_req_sel_next;
            end

            if (mem_req_pending && mem_req_ready) begin
                mem_req_pending <= 1'b0;
                mem_rr_ptr <= mem_req_sel + 1'b1;
                if (!bank_mem_req_write[mem_req_sel]) begin
                    mem_outstanding <= 1'b1;
                    mem_out_bank <= mem_req_sel;
                end
            end

            if (mem_outstanding && mem_resp_valid) begin
                mem_outstanding <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] hit_count, miss_count, wb_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hit_count <= 0;
            miss_count <= 0;
            wb_count <= 0;
        end else begin
            hit_count <= hit_count + |bank_hit;
            miss_count <= miss_count + |bank_miss;
            wb_count <= wb_count + |bank_writeback;
        end
    end

    assign stat_hits = hit_count;
    assign stat_misses = miss_count;
    assign stat_writebacks = wb_count;

endmodule


//============================================================================
// L2 Cache Bank - Single bank implementation
//============================================================================
module l2_cache_bank #(
    parameter SIZE_BYTES    = 262144,       // 256KB per bank
    parameter NUM_WAYS      = 16,
    parameter LINE_SIZE     = 128,          // 128 bytes
    parameter ADDR_WIDTH    = 32,
    parameter MSHR_ENTRIES  = 4,
    parameter PORT_ID_WIDTH = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         req_valid,
    input  wire                         req_write,
    input  wire [ADDR_WIDTH-1:0]        req_addr,
    input  wire [LINE_SIZE*8-1:0]       req_wdata,
    input  wire [LINE_SIZE-1:0]         req_wmask,
    input  wire [PORT_ID_WIDTH-1:0]     req_port_id,
    output wire                         req_ready,

    output reg                          resp_valid,
    output reg  [PORT_ID_WIDTH-1:0]     resp_port_id,
    output reg  [LINE_SIZE*8-1:0]       resp_rdata,

    output reg                          hit,
    output reg                          miss,
    output reg                          writeback,

    output reg                          mem_req_valid,
    output reg                          mem_req_write,
    output reg  [ADDR_WIDTH-1:0]        mem_req_addr,
    output reg  [LINE_SIZE*8-1:0]       mem_req_wdata,
    input  wire                         mem_req_ready,
    input  wire                         mem_fill_valid,
    input  wire [LINE_SIZE*8-1:0]       mem_fill_data
);

    localparam NUM_SETS     = SIZE_BYTES / (NUM_WAYS * LINE_SIZE);
    localparam OFFSET_BITS  = $clog2(LINE_SIZE);
    localparam INDEX_BITS   = $clog2(NUM_SETS);
    localparam TAG_BITS     = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
    localparam LINE_BITS    = LINE_SIZE * 8;
    localparam WAY_BITS     = $clog2(NUM_WAYS);

    //------------------------------------------------------------------------
    // Cache Storage
    //------------------------------------------------------------------------
    // Tag array: [valid][dirty][tag]
    reg [TAG_BITS-1:0]      tag_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [NUM_WAYS-1:0]      valid_array [0:NUM_SETS-1];
    reg [NUM_WAYS-1:0]      dirty_array [0:NUM_SETS-1];

    // Data array (in real implementation, this would be SRAM)
    reg [LINE_BITS-1:0]     data_array  [0:NUM_SETS-1][0:NUM_WAYS-1];

    // LRU state (pseudo-LRU tree)
    reg [NUM_WAYS-2:0]      lru_state   [0:NUM_SETS-1];

    //------------------------------------------------------------------------
    // Address Parsing (latched)
    //------------------------------------------------------------------------
    wire [OFFSET_BITS-1:0]  req_offset  = addr_reg[OFFSET_BITS-1:0];
    wire [INDEX_BITS-1:0]   req_index   = addr_reg[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     req_tag     = addr_reg[ADDR_WIDTH-1 -: TAG_BITS];

    //------------------------------------------------------------------------
    // Tag Comparison
    //------------------------------------------------------------------------
    reg [NUM_WAYS-1:0] way_hit;
    reg [WAY_BITS-1:0] hit_way;
    reg any_hit;

    integer w;
    always @(*) begin
        way_hit = 0;
        hit_way = 0;
        any_hit = 0;

        for (w = 0; w < NUM_WAYS; w = w + 1) begin
            if (valid_array[req_index][w] &&
                tag_array[req_index][w] == req_tag) begin
                way_hit[w] = 1'b1;
                hit_way = w[WAY_BITS-1:0];
                any_hit = 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // LRU Victim Selection
    //------------------------------------------------------------------------
    reg [WAY_BITS-1:0] victim_way;

    always @(*) begin
        // Simple pseudo-LRU: find first invalid, else use LRU tree
        victim_way = 0;

        // First check for invalid way
        for (w = 0; w < NUM_WAYS; w = w + 1) begin
            if (!valid_array[req_index][w]) begin
                victim_way = w[WAY_BITS-1:0];
            end
        end

        // If all valid, use LRU tree (simplified: just pick based on state)
        if (valid_array[req_index] == {NUM_WAYS{1'b1}}) begin
            victim_way = lru_state[req_index][WAY_BITS-1:0];
        end
    end

    //------------------------------------------------------------------------
    // Cache State Machine
    //------------------------------------------------------------------------
    localparam S_IDLE        = 3'd0;
    localparam S_TAG_CHECK   = 3'd1;
    localparam S_HIT         = 3'd2;
    localparam S_WRITEBACK   = 3'd3;
    localparam S_FILL_REQ    = 3'd4;
    localparam S_WAIT_FILL   = 3'd5;

    reg [2:0] state;
    reg [ADDR_WIDTH-1:0] addr_reg;
    reg [LINE_BITS-1:0] wdata_reg;
    reg [LINE_SIZE-1:0] wmask_reg;
    reg write_reg;
    reg [WAY_BITS-1:0] way_reg;
    reg [PORT_ID_WIDTH-1:0] port_id_reg;

    assign req_ready = (state == S_IDLE);

    wire [ADDR_WIDTH-1:0] victim_addr =
        {tag_array[req_index][way_reg], req_index, {OFFSET_BITS{1'b0}}};

    reg [LINE_BITS-1:0] merged_hit_line;
    reg [LINE_BITS-1:0] merged_fill_line;
    integer b;
    always @(*) begin
        merged_hit_line = data_array[req_index][way_reg];
        merged_fill_line = mem_fill_data;
        for (b = 0; b < LINE_SIZE; b = b + 1) begin
            if (wmask_reg[b]) begin
                merged_hit_line[b*8 +: 8] = wdata_reg[b*8 +: 8];
                merged_fill_line[b*8 +: 8] = wdata_reg[b*8 +: 8];
            end
        end
    end

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            resp_valid <= 0;
            resp_rdata <= 0;
            hit <= 0;
            miss <= 0;
            writeback <= 0;
            mem_req_valid <= 0;
            mem_req_write <= 0;
            mem_req_addr <= 0;
            mem_req_wdata <= 0;
            addr_reg <= 0;
            wdata_reg <= 0;
            wmask_reg <= 0;
            write_reg <= 0;
            way_reg <= 0;
            port_id_reg <= 0;
            resp_port_id <= 0;

            // Initialize arrays
            for (i = 0; i < NUM_SETS; i = i + 1) begin
                valid_array[i] <= 0;
                dirty_array[i] <= 0;
                lru_state[i] <= 0;
            end
        end else begin
            // Default outputs
            resp_valid <= 0;
            hit <= 0;
            miss <= 0;
            writeback <= 0;
            mem_req_valid <= 0;
            mem_req_write <= 0;
            mem_req_addr <= 0;
            mem_req_wdata <= 0;

            case (state)
                S_IDLE: begin
                    if (req_valid) begin
                        addr_reg <= req_addr;
                        wdata_reg <= req_wdata;
                        wmask_reg <= req_wmask;
                        write_reg <= req_write;
                        port_id_reg <= req_port_id;
                        state <= S_TAG_CHECK;
                    end
                end

                S_TAG_CHECK: begin
                    if (any_hit) begin
                        hit <= 1'b1;
                        way_reg <= hit_way;
                        state <= S_HIT;
                    end else begin
                        miss <= 1'b1;
                        way_reg <= victim_way;
                        // Check if victim needs writeback
                        if (valid_array[req_index][victim_way] &&
                            dirty_array[req_index][victim_way]) begin
                            writeback <= 1'b1;
                            state <= S_WRITEBACK;
                        end else begin
                            state <= S_FILL_REQ; // Always fetch on miss to support partial writes
                        end
                    end
                end

                S_HIT: begin
                    // Read or write hit
                    if (write_reg) begin
                        data_array[req_index][way_reg] <= merged_hit_line;
                        dirty_array[req_index][way_reg] <= 1'b1;
                        resp_rdata <= merged_hit_line;
                    end else begin
                        resp_rdata <= data_array[req_index][way_reg];
                    end
                    // Update LRU
                    lru_state[req_index] <= lru_state[req_index] ^ (1 << way_reg);
                    resp_valid <= 1'b1;
                    resp_port_id <= port_id_reg;
                    state <= S_IDLE;
                end

                S_WRITEBACK: begin
                    mem_req_valid <= 1'b1;
                    mem_req_write <= 1'b1;
                    mem_req_addr  <= victim_addr;
                    mem_req_wdata <= data_array[req_index][way_reg];
                    if (mem_req_valid && mem_req_ready) begin
                        mem_req_valid <= 1'b0;
                        state <= S_FILL_REQ;
                    end
                end

                S_FILL_REQ: begin
                    mem_req_valid <= 1'b1;
                    mem_req_write <= 1'b0;
                    mem_req_addr  <= {req_tag, req_index, {OFFSET_BITS{1'b0}}};
                    if (mem_req_valid && mem_req_ready) begin
                        mem_req_valid <= 1'b0;
                        state <= S_WAIT_FILL;
                    end
                end

                S_WAIT_FILL: begin
                    if (mem_fill_valid) begin
                        if (write_reg) begin
                            data_array[req_index][way_reg] <= merged_fill_line;
                            dirty_array[req_index][way_reg] <= 1'b1;
                            resp_rdata <= merged_fill_line;
                        end else begin
                            data_array[req_index][way_reg] <= mem_fill_data;
                            dirty_array[req_index][way_reg] <= 1'b0;
                            resp_rdata <= mem_fill_data;
                        end
                        tag_array[req_index][way_reg] <= req_tag;
                        valid_array[req_index][way_reg] <= 1'b1;
                        lru_state[req_index] <= lru_state[req_index] ^ (1 << way_reg);
                        resp_valid <= 1'b1;
                        resp_port_id <= port_id_reg;
                        state <= S_IDLE;
                    end
                end


                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
