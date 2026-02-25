//============================================================================
// RalphGPU - Memory Coalescing Unit
// Memory Coalescing Unit: Coalesce memory accesses from 32 threads into minimum 
// memory transactions. This is key to high performance in GPUs.
//============================================================================

`timescale 1ns / 1ps

module memory_coalescing_unit #(
    parameter THREADS         = 32,
    parameter DATA_WIDTH      = 32,
    parameter ADDR_WIDTH      = 32,
    parameter CACHE_LINE_SIZE = 128,    // 128 bytes per cache line
    parameter MAX_COALESCED   = 4       // Max unique transactions per warp request
)(
    input  wire                 clk,
    input  wire                 rst_n,

    //------------------------------------------------------------------------
    // Warp-level Request (from SM)
    //------------------------------------------------------------------------
    input  wire                         req_valid,
    input  wire                         req_write,
    input  wire [THREADS*ADDR_WIDTH-1:0] req_addr,
    input  wire [THREADS*DATA_WIDTH-1:0] req_wdata,
    input  wire [THREADS-1:0]           req_mask,
    output reg                          ready,

    //------------------------------------------------------------------------
    // Memory Interface (to L1 Cache)
    //------------------------------------------------------------------------
    output reg                          mem_req_valid,
    output reg                          mem_req_write,
    output reg  [ADDR_WIDTH-1:0]        mem_req_addr,
    output reg  [CACHE_LINE_SIZE*8-1:0] mem_req_wdata,
    output reg  [CACHE_LINE_SIZE-1:0]   mem_req_wmask,
    input  wire [CACHE_LINE_SIZE*8-1:0] mem_resp_rdata,
    input  wire                         mem_resp_valid,

    //------------------------------------------------------------------------
    // Response to Warp
    //------------------------------------------------------------------------
    output reg [THREADS*DATA_WIDTH-1:0]  resp_rdata,
    output reg                          resp_valid,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output reg [31:0]                   stat_requests,
    output reg [31:0]                   stat_transactions,
    output reg [31:0]                   stat_coalesce_ratio
);

    localparam OFFSET_BITS = $clog2(CACHE_LINE_SIZE);
    localparam LINE_ADDR_BITS = ADDR_WIDTH - OFFSET_BITS;

    // Internal storage for the current warp request
    reg [ADDR_WIDTH-1:0] saved_addr [0:THREADS-1];
    reg [DATA_WIDTH-1:0] saved_wdata [0:THREADS-1];
    reg [THREADS-1:0]    saved_mask;
    reg                  saved_write;

    // Coalescing analysis results
    reg [LINE_ADDR_BITS-1:0] unique_lines [0:MAX_COALESCED-1];
    reg [MAX_COALESCED-1:0]  line_valid;
    reg [2:0]                num_unique_lines; // Supports up to 4, so 3 bits is enough
    reg [1:0]                thread_to_line [0:THREADS-1];

    // State Machine
    localparam ST_IDLE    = 3'd0;
    localparam ST_ANALYZE = 3'd1;
    localparam ST_REQUEST = 3'd2;
    localparam ST_WAIT    = 3'd3;
    localparam ST_COLLECT = 3'd4;
    localparam ST_DONE    = 3'd5;

    reg [2:0] state;
    reg [2:0] current_line_idx;
    reg [CACHE_LINE_SIZE*8-1:0] line_data_buf [0:MAX_COALESCED-1];

    integer i, j;
    reg line_found;

    // Combinationally analyze coalescing
    always @(*) begin
        num_unique_lines = 0;
        for (i = 0; i < MAX_COALESCED; i = i + 1) begin
            unique_lines[i] = 0;
            line_valid[i] = 0;
        end
        for (i = 0; i < THREADS; i = i + 1) begin
            thread_to_line[i] = 0;
        end

        for (i = 0; i < THREADS; i = i + 1) begin
            if (saved_mask[i]) begin
                line_found = 0;
                for (j = 0; j < MAX_COALESCED; j = j + 1) begin
                    if (line_valid[j] && unique_lines[j] == saved_addr[i][ADDR_WIDTH-1:OFFSET_BITS]) begin
                        line_found = 1;
                        thread_to_line[i] = j[1:0];
                    end
                end

                if (!line_found && num_unique_lines < MAX_COALESCED) begin
                    unique_lines[num_unique_lines[1:0]] = saved_addr[i][ADDR_WIDTH-1:OFFSET_BITS];
                    line_valid[num_unique_lines[1:0]] = 1;
                    thread_to_line[i] = num_unique_lines[1:0];
                    num_unique_lines = num_unique_lines + 1;
                end
            end
        end
    end

    // FSM and Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ready <= 1;
            mem_req_valid <= 0;
            mem_req_write <= 0;
            mem_req_addr <= 0;
            mem_req_wdata <= 0;
            mem_req_wmask <= 0;
            resp_valid <= 0;
            resp_rdata <= 0;
            current_line_idx <= 0;
            stat_requests <= 0;
            stat_transactions <= 0;
            stat_coalesce_ratio <= 0;
            saved_mask <= 0;
            saved_write <= 0;
            for (i = 0; i < THREADS; i = i + 1) begin
                saved_addr[i] <= 0;
                saved_wdata[i] <= 0;
            end
            for (i = 0; i < MAX_COALESCED; i = i + 1) begin
                line_data_buf[i] <= 0;
            end
        end else begin
            mem_req_valid <= 0;
            resp_valid <= 0;

            case (state)
                ST_IDLE: begin
                    ready <= 1;
                    if (req_valid) begin
                        ready <= 0;
                        saved_mask <= req_mask;
                        saved_write <= req_write;
                        for (i = 0; i < THREADS; i = i + 1) begin
                            saved_addr[i] <= req_addr[i*ADDR_WIDTH +: ADDR_WIDTH];
                            saved_wdata[i] <= req_wdata[i*DATA_WIDTH +: DATA_WIDTH];
                        end
                        state <= ST_ANALYZE;
                        stat_requests <= stat_requests + 1;
                    end
                end

                ST_ANALYZE: begin
                    current_line_idx <= 0;
                    state <= ST_REQUEST;
                end

                ST_REQUEST: begin
                    if (current_line_idx < num_unique_lines) begin
                        mem_req_valid <= 1;
                        mem_req_write <= saved_write;
                        mem_req_addr <= {unique_lines[current_line_idx[1:0]], {OFFSET_BITS{1'b0}}};
                        
                        // Construct write data and mask
                        mem_req_wdata <= 0;
                        mem_req_wmask <= 0;
                        if (saved_write) begin
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i] && thread_to_line[i] == current_line_idx[1:0]) begin
                                    // Byte offset within cache line
                                    // saved_addr[i][OFFSET_BITS-1:0] is byte offset
                                    // We assume 4-byte (32-bit) aligned access for simplicity here
                                    mem_req_wdata[saved_addr[i][OFFSET_BITS-1:0]*8 +: DATA_WIDTH] <= saved_wdata[i];
                                    mem_req_wmask[saved_addr[i][OFFSET_BITS-1:0] +: (DATA_WIDTH/8)] <= {(DATA_WIDTH/8){1'b1}};
                                end
                            end
                        end
                        state <= ST_WAIT;
                        stat_transactions <= stat_transactions + 1;
                    end else begin
                        state <= ST_COLLECT;
                    end
                end

                ST_WAIT: begin
                    if (mem_resp_valid) begin
                        line_data_buf[current_line_idx[1:0]] <= mem_resp_rdata;
                        current_line_idx <= current_line_idx + 1;
                        state <= ST_REQUEST;
                    end
                end

                ST_COLLECT: begin
                    for (i = 0; i < THREADS; i = i + 1) begin
                        if (saved_mask[i]) begin
                            // Extract data from the correct line buffer and correct offset
                            resp_rdata[i*DATA_WIDTH +: DATA_WIDTH] <= 
                                line_data_buf[thread_to_line[i]][saved_addr[i][OFFSET_BITS-1:0]*8 +: DATA_WIDTH];
                        end else begin
                            resp_rdata[i*DATA_WIDTH +: DATA_WIDTH] <= 0;
                        end
                    end
                    resp_valid <= 1;
                    
                    // Update stats ratio (fixed point x100)
                    if (stat_transactions > 0) begin
                        stat_coalesce_ratio <= (stat_requests * 100) / stat_transactions;
                    end
                    
                    state <= ST_DONE;
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
