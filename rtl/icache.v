//============================================================================
// RalphGPU - Instruction Cache (I-Cache)
// Direct-mapped or N-way set-associative instruction cache
// Supports prefetch and fetch buffer
//============================================================================

`include "gpu_defines.vh"
`include "memory_config.vh"

module icache #(
    parameter SIZE_KB       = 4,                // Cache size in KB
    parameter LINE_SIZE     = 64,               // Cache line size in bytes (16 instructions)
    parameter NUM_WAYS      = 2,                // Associativity
    parameter PREFETCH_DEPTH = 4,               // Prefetch buffer depth
    parameter ADDR_WIDTH    = 32,
    parameter DATA_WIDTH    = 32                // Instruction width
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Fetch Interface (from SM)
    //------------------------------------------------------------------------
    input  wire                     fetch_req,
    input  wire [ADDR_WIDTH-1:0]    fetch_addr,
    output wire                     fetch_ready,
    output wire [DATA_WIDTH-1:0]    fetch_data,
    output wire                     fetch_valid,

    //------------------------------------------------------------------------
    // Invalidation Interface
    //------------------------------------------------------------------------
    input  wire                     invalidate_req,
    input  wire [ADDR_WIDTH-1:0]    invalidate_addr,
    input  wire                     invalidate_all,
    output wire                     invalidate_done,

    //------------------------------------------------------------------------
    // Memory Interface (to L2/Memory)
    //------------------------------------------------------------------------
    output wire                     mem_req_valid,
    output wire [ADDR_WIDTH-1:0]    mem_req_addr,
    input  wire                     mem_req_ready,
    input  wire [LINE_SIZE*8-1:0]   mem_resp_data,
    input  wire                     mem_resp_valid,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]              stat_hits,
    output wire [31:0]              stat_misses,
    output wire [31:0]              stat_prefetch_hits
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam SIZE_BYTES       = SIZE_KB * 1024;
    localparam NUM_SETS         = SIZE_BYTES / (NUM_WAYS * LINE_SIZE);
    localparam OFFSET_BITS      = $clog2(LINE_SIZE);
    localparam INDEX_BITS       = $clog2(NUM_SETS);
    localparam TAG_BITS         = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
    localparam LINE_BITS        = LINE_SIZE * 8;
    localparam WORDS_PER_LINE   = LINE_SIZE / (DATA_WIDTH / 8);
    localparam WORD_BITS        = $clog2(WORDS_PER_LINE);
    localparam WAY_BITS         = (NUM_WAYS > 1) ? $clog2(NUM_WAYS) : 1;

    //------------------------------------------------------------------------
    // Cache Storage
    //------------------------------------------------------------------------
    // Tag array: valid + tag
    reg [TAG_BITS-1:0]      tag_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [NUM_WAYS-1:0]      valid_array [0:NUM_SETS-1];
    reg [LINE_BITS-1:0]     data_array  [0:NUM_SETS-1][0:NUM_WAYS-1];

    // LRU tracking (pseudo-LRU for >2 ways)
    reg [WAY_BITS-1:0]      lru_array   [0:NUM_SETS-1];

    //------------------------------------------------------------------------
    // Address Decoding
    //------------------------------------------------------------------------
    wire [OFFSET_BITS-1:0]  req_offset  = fetch_addr[OFFSET_BITS-1:0];
    wire [INDEX_BITS-1:0]   req_index   = fetch_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     req_tag     = fetch_addr[ADDR_WIDTH-1 -: TAG_BITS];
    wire [WORD_BITS-1:0]    req_word    = fetch_addr[2 +: WORD_BITS];

    //------------------------------------------------------------------------
    // Tag Comparison
    //------------------------------------------------------------------------
    wire [NUM_WAYS-1:0] way_hit;
    wire                cache_hit;
    reg  [WAY_BITS-1:0] hit_way;

    genvar w;
    generate
        for (w = 0; w < NUM_WAYS; w = w + 1) begin : gen_way_hit
            assign way_hit[w] = valid_array[req_index][w] &&
                               (tag_array[req_index][w] == req_tag);
        end
    endgenerate

    assign cache_hit = |way_hit;

    // Priority encoder for hit way
    integer hit_i;
    always @(*) begin
        hit_way = 0;
        for (hit_i = 0; hit_i < NUM_WAYS; hit_i = hit_i + 1) begin
            if (way_hit[hit_i]) hit_way = hit_i[WAY_BITS-1:0];
        end
    end

    //------------------------------------------------------------------------
    // Cache Line Data Selection
    //------------------------------------------------------------------------
    wire [LINE_BITS-1:0]    hit_line    = data_array[req_index][hit_way];
    wire [DATA_WIDTH-1:0]   hit_data    = hit_line[req_word * DATA_WIDTH +: DATA_WIDTH];

    //------------------------------------------------------------------------
    // Prefetch Buffer
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0]    prefetch_addr [0:PREFETCH_DEPTH-1];
    reg [LINE_BITS-1:0]     prefetch_data [0:PREFETCH_DEPTH-1];
    reg [PREFETCH_DEPTH-1:0] prefetch_valid;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_head;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_tail;

    // Check if address is in prefetch buffer
    wire [PREFETCH_DEPTH-1:0] prefetch_hit_vec;
    wire prefetch_buffer_hit;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_hit_idx;

    generate
        for (w = 0; w < PREFETCH_DEPTH; w = w + 1) begin : gen_prefetch_hit
            wire [ADDR_WIDTH-1:0] prefetch_line_addr = {prefetch_addr[w][ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            wire [ADDR_WIDTH-1:0] fetch_line_addr = {fetch_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            assign prefetch_hit_vec[w] = prefetch_valid[w] && (prefetch_line_addr == fetch_line_addr);
        end
    endgenerate

    assign prefetch_buffer_hit = |prefetch_hit_vec;

    integer pf_i;
    always @(*) begin
        prefetch_hit_idx = 0;
        for (pf_i = 0; pf_i < PREFETCH_DEPTH; pf_i = pf_i + 1) begin
            if (prefetch_hit_vec[pf_i]) prefetch_hit_idx = pf_i[$clog2(PREFETCH_DEPTH)-1:0];
        end
    end

    wire [DATA_WIDTH-1:0] prefetch_hit_data = prefetch_data[prefetch_hit_idx][req_word * DATA_WIDTH +: DATA_WIDTH];

    //------------------------------------------------------------------------
    // FSM States
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 3'd0;
    localparam ST_TAG_CHECK     = 3'd1;
    localparam ST_MISS_REQ      = 3'd2;
    localparam ST_MISS_WAIT     = 3'd3;
    localparam ST_FILL          = 3'd4;
    localparam ST_INVALIDATE    = 3'd5;
    localparam ST_PREFETCH      = 3'd6;

    reg [2:0] state;
    reg [ADDR_WIDTH-1:0] miss_addr;
    reg [WORD_BITS-1:0] miss_word;  // Word offset within line for miss handling
    reg [WAY_BITS-1:0] replace_way;
    reg is_prefetch_miss;

    //------------------------------------------------------------------------
    // Replacement Policy (Pseudo-LRU)
    //------------------------------------------------------------------------
    wire [WAY_BITS-1:0] victim_way = lru_array[req_index];

    // Update LRU on hit
    task update_lru;
        input [INDEX_BITS-1:0] index;
        input [WAY_BITS-1:0] accessed_way;
        begin
            if (NUM_WAYS == 2) begin
                lru_array[index] <= ~accessed_way;
            end else begin
                // Simple round-robin for >2 ways
                lru_array[index] <= (accessed_way + 1) % NUM_WAYS;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Statistics Counters
    //------------------------------------------------------------------------
    reg [31:0] hit_count;
    reg [31:0] miss_count;
    reg [31:0] prefetch_hit_count;

    assign stat_hits          = hit_count;
    assign stat_misses        = miss_count;
    assign stat_prefetch_hits = prefetch_hit_count;

    //------------------------------------------------------------------------
    // Main FSM
    //------------------------------------------------------------------------
    reg fetch_valid_r;
    reg [DATA_WIDTH-1:0] fetch_data_r;
    reg mem_req_valid_r;
    reg [ADDR_WIDTH-1:0] mem_req_addr_r;
    reg invalidate_done_r;
    reg fetch_ready_r;

    // Prefetch next line detection
    wire [ADDR_WIDTH-1:0] next_line_addr = {fetch_addr[ADDR_WIDTH-1:OFFSET_BITS] + 1'b1, {OFFSET_BITS{1'b0}}};
    wire should_prefetch = fetch_req && cache_hit && (state == ST_IDLE);

    // Check if next line is already cached or in prefetch buffer
    wire [INDEX_BITS-1:0]   next_index   = next_line_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     next_tag     = next_line_addr[ADDR_WIDTH-1 -: TAG_BITS];
    wire [NUM_WAYS-1:0] next_way_hit;

    generate
        for (w = 0; w < NUM_WAYS; w = w + 1) begin : gen_next_way_hit
            assign next_way_hit[w] = valid_array[next_index][w] &&
                                    (tag_array[next_index][w] == next_tag);
        end
    endgenerate

    wire next_line_cached = |next_way_hit;

    // Check prefetch buffer for next line
    wire [PREFETCH_DEPTH-1:0] next_prefetch_hit_vec;
    generate
        for (w = 0; w < PREFETCH_DEPTH; w = w + 1) begin : gen_next_prefetch
            wire [ADDR_WIDTH-1:0] pf_line_addr = {prefetch_addr[w][ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            assign next_prefetch_hit_vec[w] = prefetch_valid[w] && (pf_line_addr == next_line_addr);
        end
    endgenerate

    wire next_line_prefetched = |next_prefetch_hit_vec;
    // Disable prefetch for now - it blocks subsequent hits from same cache line
    // TODO: Implement non-blocking prefetch that allows concurrent hits
    wire need_prefetch = 1'b0;  // was: should_prefetch && !next_line_cached && !next_line_prefetched;

    integer rst_i, rst_j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            fetch_valid_r <= 1'b0;
            fetch_data_r <= 0;
            fetch_ready_r <= 1'b1;
            mem_req_valid_r <= 1'b0;
            mem_req_addr_r <= 0;
            invalidate_done_r <= 1'b0;
            hit_count <= 0;
            miss_count <= 0;
            prefetch_hit_count <= 0;
            miss_addr <= 0;
            miss_word <= 0;
            replace_way <= 0;
            is_prefetch_miss <= 1'b0;
            prefetch_valid <= 0;
            prefetch_head <= 0;
            prefetch_tail <= 0;

            // Reset valid bits
            for (rst_i = 0; rst_i < NUM_SETS; rst_i = rst_i + 1) begin
                valid_array[rst_i] <= 0;
                lru_array[rst_i] <= 0;
            end
        end else begin
            fetch_valid_r <= 1'b0;
            invalidate_done_r <= 1'b0;

            case (state)
                ST_IDLE: begin
                    fetch_ready_r <= 1'b1;

                    if (invalidate_req) begin
                        state <= ST_INVALIDATE;
                        fetch_ready_r <= 1'b0;
                    end else if (fetch_req) begin
                        state <= ST_TAG_CHECK;
                        fetch_ready_r <= 1'b0;
                    end
                end

                ST_TAG_CHECK: begin
                    if (cache_hit) begin
                        // Cache hit
                        fetch_data_r <= hit_data;
                        fetch_valid_r <= 1'b1;
                        hit_count <= hit_count + 1;
                        update_lru(req_index, hit_way);

                        // Check for prefetch opportunity
                        if (need_prefetch) begin
                            state <= ST_PREFETCH;
                            miss_addr <= next_line_addr;
                            is_prefetch_miss <= 1'b1;
                        end else begin
                            state <= ST_IDLE;
                            fetch_ready_r <= 1'b1;  // Ready for next request immediately
                        end
                    end else if (prefetch_buffer_hit) begin
                        // Prefetch buffer hit - promote to cache
                        fetch_data_r <= prefetch_hit_data;
                        fetch_valid_r <= 1'b1;
                        prefetch_hit_count <= prefetch_hit_count + 1;

                        // Write prefetch data to cache
                        replace_way <= victim_way;
                        tag_array[req_index][victim_way] <= req_tag;
                        data_array[req_index][victim_way] <= prefetch_data[prefetch_hit_idx];
                        valid_array[req_index][victim_way] <= 1'b1;
                        update_lru(req_index, victim_way);

                        // Invalidate prefetch entry
                        prefetch_valid[prefetch_hit_idx] <= 1'b0;

                        state <= ST_IDLE;
                        fetch_ready_r <= 1'b1;  // Ready for next request immediately
                    end else begin
                        // Cache miss
                        miss_count <= miss_count + 1;
                        miss_addr <= {fetch_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
                        miss_word <= req_word;  // Latch word offset for fill
                        replace_way <= victim_way;
                        is_prefetch_miss <= 1'b0;
                        state <= ST_MISS_REQ;
                    end
                end

                ST_MISS_REQ: begin
                    mem_req_valid_r <= 1'b1;
                    mem_req_addr_r <= miss_addr;

                    if (mem_req_valid_r && mem_req_ready) begin
                        mem_req_valid_r <= 1'b0;
                        state <= ST_MISS_WAIT;
                    end
                end

                ST_MISS_WAIT: begin
                    if (mem_resp_valid) begin
                        state <= ST_FILL;
                    end
                end

                ST_FILL: begin
                    if (!is_prefetch_miss) begin
                        // Fill cache line - use latched miss_addr for index/tag
                        // miss_addr is line-aligned, extract index from it
                        tag_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= miss_addr[ADDR_WIDTH-1 -: TAG_BITS];
                        data_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= mem_resp_data;
                        valid_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= 1'b1;
                        update_lru(miss_addr[OFFSET_BITS +: INDEX_BITS], replace_way);

                        // Return data - use latched miss_word for word selection
                        fetch_data_r <= mem_resp_data[miss_word * DATA_WIDTH +: DATA_WIDTH];
                        fetch_valid_r <= 1'b1;

                        // Trigger prefetch if possible
                        if (need_prefetch) begin
                            state <= ST_PREFETCH;
                            miss_addr <= next_line_addr;
                            is_prefetch_miss <= 1'b1;
                        end else begin
                            state <= ST_IDLE;
                            fetch_ready_r <= 1'b1;  // Ready for next request immediately
                        end
                    end else begin
                        // Fill prefetch buffer
                        prefetch_addr[prefetch_head] <= miss_addr;
                        prefetch_data[prefetch_head] <= mem_resp_data;
                        prefetch_valid[prefetch_head] <= 1'b1;
                        prefetch_head <= (prefetch_head + 1) % PREFETCH_DEPTH;

                        state <= ST_IDLE;
                        fetch_ready_r <= 1'b1;  // Ready for next request immediately
                    end
                end

                ST_PREFETCH: begin
                    // Issue prefetch request
                    mem_req_valid_r <= 1'b1;
                    mem_req_addr_r <= miss_addr;

                    if (mem_req_ready) begin
                        mem_req_valid_r <= 1'b0;
                        state <= ST_MISS_WAIT;
                    end
                end

                ST_INVALIDATE: begin
                    if (invalidate_all) begin
                        // Invalidate entire cache
                        for (rst_i = 0; rst_i < NUM_SETS; rst_i = rst_i + 1) begin
                            valid_array[rst_i] <= 0;
                        end
                        prefetch_valid <= 0;
                    end else begin
                        // Invalidate specific line
                        for (rst_j = 0; rst_j < NUM_WAYS; rst_j = rst_j + 1) begin
                            if (valid_array[req_index][rst_j] &&
                                tag_array[req_index][rst_j] == req_tag) begin
                                valid_array[req_index][rst_j] <= 1'b0;
                            end
                        end
                    end
                    invalidate_done_r <= 1'b1;
                    state <= ST_IDLE;
                    fetch_ready_r <= 1'b1;  // Ready for next request immediately
                end

                default: begin
                    state <= ST_IDLE;
                    fetch_ready_r <= 1'b1;  // Ready for next request immediately
                end
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign fetch_ready     = fetch_ready_r;
    assign fetch_data      = fetch_data_r;
    assign fetch_valid     = fetch_valid_r;
    assign mem_req_valid   = mem_req_valid_r;
    assign mem_req_addr    = mem_req_addr_r;
    assign invalidate_done = invalidate_done_r;

endmodule


//============================================================================
// Instruction Prefetch Unit
// Maintains a stream of prefetched instructions
//============================================================================
module instruction_prefetch_unit #(
    parameter PREFETCH_DEPTH = 8,           // Number of instructions to prefetch
    parameter ADDR_WIDTH     = 32,
    parameter DATA_WIDTH     = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Control Interface
    //------------------------------------------------------------------------
    input  wire                     enable,
    input  wire [ADDR_WIDTH-1:0]    base_addr,      // Starting address
    input  wire                     flush,           // Flush prefetch buffer
    input  wire                     branch_taken,    // Branch misprediction
    input  wire [ADDR_WIDTH-1:0]    branch_target,

    //------------------------------------------------------------------------
    // Consumer Interface (to Decode)
    //------------------------------------------------------------------------
    output wire [DATA_WIDTH-1:0]    prefetch_data,
    output wire [ADDR_WIDTH-1:0]    prefetch_addr,
    output wire                     prefetch_valid,
    input  wire                     prefetch_consume,

    //------------------------------------------------------------------------
    // I-Cache Interface
    //------------------------------------------------------------------------
    output wire                     icache_req,
    output wire [ADDR_WIDTH-1:0]    icache_addr,
    input  wire                     icache_ready,
    input  wire [DATA_WIDTH-1:0]    icache_data,
    input  wire                     icache_valid
);

    localparam PTR_W = $clog2(PREFETCH_DEPTH);
    localparam CNT_W = $clog2(PREFETCH_DEPTH + 1);

    //------------------------------------------------------------------------
    // Prefetch Buffer
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] buffer_data [0:PREFETCH_DEPTH-1];
    reg [ADDR_WIDTH-1:0] buffer_addr [0:PREFETCH_DEPTH-1];
    reg [PREFETCH_DEPTH-1:0] buffer_valid;
    reg [PTR_W-1:0] head_ptr;  // Next to consume
    reg [PTR_W-1:0] tail_ptr;  // Next to fill
    reg [CNT_W-1:0] count;

    wire buffer_empty = (count == 0);
    wire buffer_full  = (count == PREFETCH_DEPTH);

    //------------------------------------------------------------------------
    // Fetch Address Tracking
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] next_fetch_addr;
    reg fetch_pending;

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE     = 2'd0;
    localparam ST_FETCH    = 2'd1;
    localparam ST_WAIT     = 2'd2;

    reg [1:0] state;

    //------------------------------------------------------------------------
    // Main Logic
    //------------------------------------------------------------------------
    integer rst_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            head_ptr <= 0;
            tail_ptr <= 0;
            count <= 0;
            buffer_valid <= 0;
            next_fetch_addr <= 0;
            fetch_pending <= 1'b0;
        end else begin
            // Handle flush or branch
            if (flush || branch_taken) begin
                head_ptr <= 0;
                tail_ptr <= 0;
                count <= 0;
                buffer_valid <= 0;
                next_fetch_addr <= branch_taken ? branch_target : base_addr;
                fetch_pending <= 1'b0;
                state <= ST_IDLE;
            end else begin
                // Consumer takes an entry
                if (prefetch_consume && !buffer_empty) begin
                    buffer_valid[head_ptr] <= 1'b0;
                    head_ptr <= (head_ptr + 1) % PREFETCH_DEPTH;
                    count <= count - 1;
                end

                // State machine for prefetching
                case (state)
                    ST_IDLE: begin
                        if (enable && !buffer_full && !fetch_pending) begin
                            state <= ST_FETCH;
                        end
                    end

                    ST_FETCH: begin
                        if (icache_ready) begin
                            fetch_pending <= 1'b1;
                            state <= ST_WAIT;
                        end
                    end

                    ST_WAIT: begin
                        if (icache_valid) begin
                            // Store in buffer
                            buffer_data[tail_ptr] <= icache_data;
                            buffer_addr[tail_ptr] <= next_fetch_addr;
                            buffer_valid[tail_ptr] <= 1'b1;
                            tail_ptr <= (tail_ptr + 1) % PREFETCH_DEPTH;
                            count <= count + 1;

                            // Advance to next address
                            next_fetch_addr <= next_fetch_addr + 4;
                            fetch_pending <= 1'b0;

                            // Continue prefetching if not full
                            if (count + 1 < PREFETCH_DEPTH) begin
                                state <= ST_FETCH;
                            end else begin
                                state <= ST_IDLE;
                            end
                        end
                    end

                    default: state <= ST_IDLE;
                endcase
            end
        end
    end

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign prefetch_data  = buffer_data[head_ptr];
    assign prefetch_addr  = buffer_addr[head_ptr];
    assign prefetch_valid = buffer_valid[head_ptr] && !buffer_empty;
    assign icache_req     = (state == ST_FETCH);
    assign icache_addr    = next_fetch_addr;

endmodule
