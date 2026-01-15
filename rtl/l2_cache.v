//============================================================================
// RalphGPU - L2 Cache
// Multi-banked, non-blocking L2 cache with ECC support
//============================================================================

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
    reg [$clog2(NUM_PORTS)-1:0] bank_rr_ptr [0:NUM_BANKS-1];

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

    // Round-robin arbitration per bank
    integer i, j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_BANKS; i = i + 1) begin
                bank_grant[i] <= 0;
                bank_rr_ptr[i] <= 0;
            end
        end else begin
            for (i = 0; i < NUM_BANKS; i = i + 1) begin
                bank_grant[i] <= 0;
                // Round-robin selection
                for (j = 0; j < NUM_PORTS; j = j + 1) begin
                    if (bank_req[i][(bank_rr_ptr[i] + j) % NUM_PORTS] && !bank_grant[i]) begin
                        bank_grant[i][(bank_rr_ptr[i] + j) % NUM_PORTS] <= 1'b1;
                        bank_rr_ptr[i] <= (bank_rr_ptr[i] + j + 1) % NUM_PORTS;
                    end
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

            // Cache bank instance
            l2_cache_bank #(
                .SIZE_BYTES     (SIZE_PER_BANK),
                .NUM_WAYS       (NUM_WAYS),
                .LINE_SIZE      (LINE_SIZE),
                .ADDR_WIDTH     (ADDR_WIDTH),
                .MSHR_ENTRIES   (MSHR_ENTRIES / NUM_BANKS)
            ) u_bank (
                .clk            (clk),
                .rst_n          (rst_n),
                .req_valid      (bank_req_valid[b]),
                .req_write      (bank_req_write[b]),
                .req_addr       (bank_req_addr[b]),
                .req_wdata      (bank_req_wdata[b]),
                .resp_valid     (bank_resp_valid[b]),
                .resp_rdata     (bank_rdata[b]),
                .hit            (bank_hit[b]),
                .miss           (bank_miss[b]),
                .writeback      (bank_writeback[b])
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
            // Ready when not pending and bank grants access
            assign l1_req_ready[p] = !port_pending[p] &&
                                     bank_grant[port_bank[p]][p];

            // Response valid when pending and bank responds
            assign l1_resp_valid[p] = port_pending[p] &&
                                      bank_resp_valid[port_pending_bank[p]];

            // Response data from appropriate bank
            assign l1_resp_rdata[p*LINE_BITS +: LINE_BITS] =
                   bank_rdata[port_pending_bank[p]];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Memory Controller Interface (unified for all banks)
    //------------------------------------------------------------------------
    // Simple FIFO for memory requests from banks
    // In full implementation, this would be more sophisticated

    reg mem_req_pending;
    reg [ADDR_WIDTH-1:0] mem_req_addr_reg;
    reg [LINE_BITS-1:0] mem_req_wdata_reg;
    reg mem_req_write_reg;

    assign mem_req_valid = mem_req_pending;
    assign mem_req_addr = mem_req_addr_reg;
    assign mem_req_wdata = mem_req_wdata_reg;
    assign mem_req_write = mem_req_write_reg;

    // Simplified: Banks would queue their miss requests
    // Full implementation needs per-bank MSHR and memory arbiter

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
    parameter MSHR_ENTRIES  = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         req_valid,
    input  wire                         req_write,
    input  wire [ADDR_WIDTH-1:0]        req_addr,
    input  wire [LINE_SIZE*8-1:0]       req_wdata,

    output reg                          resp_valid,
    output reg  [LINE_SIZE*8-1:0]       resp_rdata,

    output reg                          hit,
    output reg                          miss,
    output reg                          writeback
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
    // Address Parsing
    //------------------------------------------------------------------------
    wire [OFFSET_BITS-1:0]  req_offset  = req_addr[OFFSET_BITS-1:0];
    wire [INDEX_BITS-1:0]   req_index   = req_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     req_tag     = req_addr[ADDR_WIDTH-1 -: TAG_BITS];

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
    localparam S_IDLE       = 3'd0;
    localparam S_TAG_CHECK  = 3'd1;
    localparam S_HIT        = 3'd2;
    localparam S_MISS       = 3'd3;
    localparam S_WRITEBACK  = 3'd4;
    localparam S_FILL       = 3'd5;
    localparam S_RESP       = 3'd6;

    reg [2:0] state;
    reg [ADDR_WIDTH-1:0] addr_reg;
    reg [LINE_BITS-1:0] wdata_reg;
    reg write_reg;
    reg [WAY_BITS-1:0] way_reg;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            resp_valid <= 0;
            resp_rdata <= 0;
            hit <= 0;
            miss <= 0;
            writeback <= 0;
            addr_reg <= 0;
            wdata_reg <= 0;
            write_reg <= 0;
            way_reg <= 0;

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

            case (state)
                S_IDLE: begin
                    if (req_valid) begin
                        addr_reg <= req_addr;
                        wdata_reg <= req_wdata;
                        write_reg <= req_write;
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
                            state <= S_FILL;
                        end
                    end
                end

                S_HIT: begin
                    // Read or write hit
                    if (write_reg) begin
                        data_array[req_index][way_reg] <= wdata_reg;
                        dirty_array[req_index][way_reg] <= 1'b1;
                    end else begin
                        resp_rdata <= data_array[req_index][way_reg];
                    end
                    // Update LRU
                    lru_state[req_index] <= lru_state[req_index] ^ (1 << way_reg);
                    resp_valid <= 1'b1;
                    state <= S_IDLE;
                end

                S_WRITEBACK: begin
                    // In real implementation: send writeback to memory
                    // Simplified: just move to fill state
                    state <= S_FILL;
                end

                S_FILL: begin
                    // In real implementation: wait for memory response
                    // Simplified: immediate fill with write data or zeros
                    if (write_reg) begin
                        data_array[req_index][way_reg] <= wdata_reg;
                        dirty_array[req_index][way_reg] <= 1'b1;
                    end else begin
                        data_array[req_index][way_reg] <= 0; // Would come from memory
                        dirty_array[req_index][way_reg] <= 1'b0;
                    end
                    tag_array[req_index][way_reg] <= req_tag;
                    valid_array[req_index][way_reg] <= 1'b1;
                    lru_state[req_index] <= lru_state[req_index] ^ (1 << way_reg);

                    resp_rdata <= write_reg ? wdata_reg : 0;
                    resp_valid <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
