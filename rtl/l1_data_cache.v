//============================================================================
// RalphGPU - L1 Data Cache (Replay-Enabled)
//============================================================================

`timescale 1ns / 1ps

module l1_data_cache #(
    parameter CACHE_SIZE_KB   = 16,
    parameter LINE_SIZE_BYTES = 128,
    parameter NUM_WAYS        = 4,
    parameter HIT_LATENCY     = 2,
    parameter THREADS         = 32,
    parameter DATA_WIDTH      = 32
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Warp Request Interface
    input  wire                 req_valid,
    input  wire                 req_write,
    input  wire [THREADS*32-1:0] req_addr,
    input  wire [THREADS*32-1:0] req_wdata,
    input  wire [THREADS-1:0]   req_mask,
    input  wire [31:0]          req_pc,
    input  wire [4:0]           req_warp_id,
    input  wire [4:0]           req_rd,

    output reg [THREADS*32-1:0] resp_rdata,
    output wire                 resp_valid,
    output reg                  resp_hit,
    output wire                 resp_replay,
    output wire [31:0]          resp_pc,
    output wire [4:0]           resp_warp_id,
    output wire [4:0]           resp_rd,

    // Memory Interface
    output reg                  mem_req,
    output reg                  mem_write,
    output reg  [31:0]          mem_addr,
    output reg  [LINE_SIZE_BYTES*8-1:0] mem_wdata,
    input  wire [LINE_SIZE_BYTES*8-1:0] mem_rdata,
    input  wire                 mem_valid,
    input  wire                 mem_ready,

    output reg  [31:0]          stat_hits,
    output reg  [31:0]          stat_misses,

    // Policy Interface (Stubs)
    input  wire                 policy_create_valid,
    input  wire [2:0]           policy_id,
    input  wire [7:0]           policy_priority,
    output reg  [31:0]          policy_token_out,
    output reg                  policy_token_valid,
    input  wire                 policy_apply_valid,
    input  wire [31:0]          policy_apply_addr,
    input  wire [2:0]           policy_apply_id,
    input  wire                 policy_discard_valid,
    input  wire [31:0]          policy_discard_addr
);

    localparam NUM_LINES      = (CACHE_SIZE_KB * 1024) / LINE_SIZE_BYTES;
    localparam NUM_SETS       = NUM_LINES / NUM_WAYS;
    localparam WORDS_PER_LINE = LINE_SIZE_BYTES / 4;
    localparam OFFSET_BITS    = $clog2(LINE_SIZE_BYTES);
    localparam INDEX_BITS     = $clog2(NUM_SETS);
    localparam TAG_BITS       = 32 - OFFSET_BITS - INDEX_BITS;

    reg [31:0] data_array [0:NUM_WAYS-1][0:NUM_SETS-1][0:WORDS_PER_LINE-1];
    reg [TAG_BITS-1:0] tag_array [0:NUM_WAYS-1][0:NUM_SETS-1];
    reg valid_array [0:NUM_WAYS-1][0:NUM_SETS-1];
    reg dirty_array [0:NUM_WAYS-1][0:NUM_SETS-1];
    reg [1:0] lru_array [0:NUM_SETS-1];

    localparam ST_IDLE      = 3'd0;
    localparam ST_HIT       = 3'd2;
    localparam ST_MISS      = 3'd3;
    localparam ST_WRITEBACK = 3'd4;
    localparam ST_FILL      = 3'd5;
    localparam ST_FILL_WAIT = 3'd6;

    reg [2:0] state;
    reg [31:0] saved_pc;
    reg [4:0]  saved_warp_id;
    reg [4:0]  saved_rd;
    reg [31:0] saved_addr [0:THREADS-1];
    reg [THREADS-1:0] saved_mask;
    reg [INDEX_BITS-1:0] saved_index;
    reg [TAG_BITS-1:0] saved_tag;
    reg [1:0] saved_way;
    reg [2:0] latency_counter;

    wire [31:0] primary_addr = req_addr[31:0];
    wire [INDEX_BITS-1:0] primary_index = primary_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0] primary_tag = primary_addr[31 : 32-TAG_BITS];

    reg cache_hit_comb;
    reg [1:0] hit_way_comb;
    integer way_idx;
    always @(*) begin
        cache_hit_comb = 0;
        hit_way_comb = 0;
        for (way_idx = 0; way_idx < NUM_WAYS; way_idx = way_idx + 1) begin
            if (valid_array[way_idx][primary_index] && tag_array[way_idx][primary_index] == primary_tag) begin
                cache_hit_comb = 1;
                hit_way_comb = way_idx[1:0];
            end
        end
    end

    // Combinational Replay and Valid Logic
    assign resp_replay = req_valid && (state != ST_IDLE || !cache_hit_comb);
    reg resp_valid_reg;
    assign resp_valid = resp_valid_reg || resp_replay;
    assign resp_pc = resp_replay ? req_pc : saved_pc;
    assign resp_warp_id = resp_replay ? req_warp_id : saved_warp_id;
    assign resp_rd = resp_replay ? req_rd : saved_rd;

    reg [1:0] replace_way_comb;
    always @(*) begin
        replace_way_comb = lru_array[primary_index];
        for (integer w = 0; w < NUM_WAYS; w = w + 1) begin
            if (!valid_array[w][primary_index]) replace_way_comb = w[1:0];
        end
    end

    integer i, j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            resp_valid_reg <= 0;
            resp_hit <= 0;
            saved_pc <= 0;
            saved_index <= 0;
            saved_tag <= 0;
            saved_way <= 0;
            mem_req <= 0;
            mem_write <= 0;
            stat_hits <= 0;
            stat_misses <= 0;
            latency_counter <= 0;
            for (i=0; i<NUM_WAYS; i=i+1)
                for (j=0; j<NUM_SETS; j=j+1)
                    valid_array[i][j] <= 0;
        end else begin
            resp_valid_reg <= 0;
            mem_req <= 0;

            case (state)
                ST_IDLE: begin
                    if (req_valid) begin
                        saved_pc <= req_pc;
                        saved_warp_id <= req_warp_id;
                        saved_rd <= req_rd;
                        saved_mask <= req_mask;
                        saved_index <= primary_index;
                        saved_tag <= primary_tag;
                        saved_way <= replace_way_comb;
                        for (i=0; i<THREADS; i=i+1) begin
                            saved_addr[i] <= req_addr[i*32 +: 32];
                        end

                        if (cache_hit_comb) begin
                            state <= ST_HIT;
                            latency_counter <= HIT_LATENCY - 1;
                        end else begin
                            state <= ST_MISS;
                            stat_misses <= stat_misses + 1;
                        end
                    end
                end

                ST_HIT: begin
                    if (latency_counter == 0) begin
                        resp_valid_reg <= 1;
                        resp_hit <= 1;
                        stat_hits <= stat_hits + 1;
                        for (i=0; i<THREADS; i=i+1) begin
                            if (saved_mask[i])
                                resp_rdata[i*32 +: 32] <= data_array[hit_way_comb][saved_index][saved_addr[i][OFFSET_BITS-1:2]];
                        end
                        state <= ST_IDLE;
                    end else begin
                        latency_counter <= latency_counter - 1;
                    end
                end

                ST_MISS: begin
                    if (dirty_array[saved_way][saved_index] && valid_array[saved_way][saved_index])
                        state <= ST_WRITEBACK;
                    else
                        state <= ST_FILL;
                end

                ST_WRITEBACK: begin
                    mem_req <= 1;
                    mem_write <= 1;
                    mem_addr <= {tag_array[saved_way][saved_index], saved_index, {OFFSET_BITS{1'b0}}};
                    for (i=0; i<WORDS_PER_LINE; i=i+1)
                        mem_wdata[i*32 +: 32] <= data_array[saved_way][saved_index][i];
                    if (mem_ready) state <= ST_FILL;
                end

                ST_FILL: begin
                    mem_req <= 1;
                    mem_write <= 0;
                    mem_addr <= {saved_tag, saved_index, {OFFSET_BITS{1'b0}}};
                    if (mem_ready) state <= ST_FILL_WAIT;
                end

                ST_FILL_WAIT: begin
                    if (mem_valid) begin
                        tag_array[saved_way][saved_index] <= saved_tag;
                        valid_array[saved_way][saved_index] <= 1;
                        dirty_array[saved_way][saved_index] <= 0;
                        for (i=0; i<WORDS_PER_LINE; i=i+1)
                            data_array[saved_way][saved_index][i] <= mem_rdata[i*32 +: 32];
                        state <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
