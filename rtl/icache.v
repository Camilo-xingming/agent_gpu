//============================================================================
// RalphGPU - Instruction Cache (I-Cache)
// Direct-mapped or N-way set-associative instruction cache
// Supports prefetch, fetch buffer, and DUAL-PORT fetch (Port A + Port B)
//
// Port B: Second read port for dual-issue fetch pipeline.
// - Combinational hit on Port B when no bank conflict with Port A
// - Bank conflict = same set index on both ports -> Port B stalls
// - Port B misses are queued and serviced after Port A miss completes
//============================================================================

`timescale 1ns / 1ps
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
    // Fetch Interface — Port A (from SM, slot 0 / even warps)
    //------------------------------------------------------------------------
    input  wire                     fetch_req,
    input  wire [ADDR_WIDTH-1:0]    fetch_addr,
    output wire                     fetch_ready,
    output wire [DATA_WIDTH-1:0]    fetch_data,
    output wire [LINE_SIZE*8-1:0]   fetch_line_data,
    output wire                     fetch_valid,

    // RALPH-8 P2: Hit-bypass during miss (non-blocking)
    output wire                     fetch_hit_bypass,
    output wire [DATA_WIDTH-1:0]    fetch_hit_bypass_data,
    output wire [LINE_SIZE*8-1:0]   fetch_hit_bypass_line_data,

    //------------------------------------------------------------------------
    // Fetch Interface — Port B (slot 1 / odd warps) — #148 dual-port
    //------------------------------------------------------------------------
    input  wire                     fetch_req_b,
    input  wire [ADDR_WIDTH-1:0]    fetch_addr_b,
    output wire                     fetch_ready_b,
    output wire [DATA_WIDTH-1:0]    fetch_data_b,
    output wire [LINE_SIZE*8-1:0]   fetch_line_data_b,
    output wire                     fetch_valid_b,

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
    reg [TAG_BITS-1:0]      tag_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [NUM_WAYS-1:0]      valid_array [0:NUM_SETS-1];
    reg [LINE_BITS-1:0]     data_array  [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [WAY_BITS-1:0]      lru_array   [0:NUM_SETS-1];

    //------------------------------------------------------------------------
    // Port A Address Decoding
    //------------------------------------------------------------------------
    wire [OFFSET_BITS-1:0]  req_offset  = fetch_addr[OFFSET_BITS-1:0];
    wire [INDEX_BITS-1:0]   req_index   = fetch_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     req_tag     = fetch_addr[ADDR_WIDTH-1 -: TAG_BITS];
    wire [WORD_BITS-1:0]    req_word    = fetch_addr[2 +: WORD_BITS];

    //------------------------------------------------------------------------
    // Port B Address Decoding (#148)
    //------------------------------------------------------------------------
    wire [OFFSET_BITS-1:0]  req_offset_b  = fetch_addr_b[OFFSET_BITS-1:0];
    wire [INDEX_BITS-1:0]   req_index_b   = fetch_addr_b[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     req_tag_b     = fetch_addr_b[ADDR_WIDTH-1 -: TAG_BITS];
    wire [WORD_BITS-1:0]    req_word_b    = fetch_addr_b[2 +: WORD_BITS];

    //------------------------------------------------------------------------
    // Port A Tag Comparison
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

    integer hit_i;
    always @(*) begin
        hit_way = 0;
        for (hit_i = 0; hit_i < NUM_WAYS; hit_i = hit_i + 1) begin
            if (way_hit[hit_i]) hit_way = hit_i[WAY_BITS-1:0];
        end
    end

    //------------------------------------------------------------------------
    // Port B Tag Comparison (#148)
    //------------------------------------------------------------------------
    wire [NUM_WAYS-1:0] way_hit_b;
    wire                cache_hit_b;
    reg  [WAY_BITS-1:0] hit_way_b;

    generate
        for (w = 0; w < NUM_WAYS; w = w + 1) begin : gen_way_hit_b
            assign way_hit_b[w] = valid_array[req_index_b][w] &&
                                 (tag_array[req_index_b][w] == req_tag_b);
        end
    endgenerate

    assign cache_hit_b = |way_hit_b;

    integer hit_i_b;
    always @(*) begin
        hit_way_b = 0;
        for (hit_i_b = 0; hit_i_b < NUM_WAYS; hit_i_b = hit_i_b + 1) begin
            if (way_hit_b[hit_i_b]) hit_way_b = hit_i_b[WAY_BITS-1:0];
        end
    end

    //------------------------------------------------------------------------
    // Bank Conflict Detection (#148)
    // Both ports accessing the same set index -> conflict
    // Port A always wins; Port B stalls
    //------------------------------------------------------------------------
    wire bank_conflict = fetch_req && fetch_req_b && (req_index == req_index_b);

    //------------------------------------------------------------------------
    // Cache Line Data Selection — Port A
    //------------------------------------------------------------------------
    wire [LINE_BITS-1:0]    hit_line    = data_array[req_index][hit_way];
    wire [DATA_WIDTH-1:0]   hit_data    = hit_line[req_word * DATA_WIDTH +: DATA_WIDTH];

    //------------------------------------------------------------------------
    // Cache Line Data Selection — Port B (#148)
    //------------------------------------------------------------------------
    wire [LINE_BITS-1:0]    hit_line_b    = data_array[req_index_b][hit_way_b];
    wire [DATA_WIDTH-1:0]   hit_data_b    = hit_line_b[req_word_b * DATA_WIDTH +: DATA_WIDTH];

    //------------------------------------------------------------------------
    // Prefetch Buffer (shared, Port A only for simplicity)
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0]    prefetch_addr [0:PREFETCH_DEPTH-1];
    reg [LINE_BITS-1:0]     prefetch_data [0:PREFETCH_DEPTH-1];
    reg [PREFETCH_DEPTH-1:0] prefetch_valid;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_head;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_tail;

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

    // Port B prefetch check (#148)
    wire [PREFETCH_DEPTH-1:0] prefetch_hit_vec_b;
    wire prefetch_buffer_hit_b;
    reg [$clog2(PREFETCH_DEPTH)-1:0] prefetch_hit_idx_b;

    generate
        for (w = 0; w < PREFETCH_DEPTH; w = w + 1) begin : gen_prefetch_hit_b
            wire [ADDR_WIDTH-1:0] pf_line_addr_b = {prefetch_addr[w][ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            wire [ADDR_WIDTH-1:0] fetch_line_addr_b = {fetch_addr_b[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            assign prefetch_hit_vec_b[w] = prefetch_valid[w] && (pf_line_addr_b == fetch_line_addr_b);
        end
    endgenerate

    assign prefetch_buffer_hit_b = |prefetch_hit_vec_b;

    integer pf_i_b;
    always @(*) begin
        prefetch_hit_idx_b = 0;
        for (pf_i_b = 0; pf_i_b < PREFETCH_DEPTH; pf_i_b = pf_i_b + 1) begin
            if (prefetch_hit_vec_b[pf_i_b]) prefetch_hit_idx_b = pf_i_b[$clog2(PREFETCH_DEPTH)-1:0];
        end
    end

    wire [DATA_WIDTH-1:0] prefetch_hit_data_b = prefetch_data[prefetch_hit_idx_b][req_word_b * DATA_WIDTH +: DATA_WIDTH];

    //------------------------------------------------------------------------
    // FSM States (Port A miss handling — Port B misses queued)
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
    reg [WORD_BITS-1:0] miss_word;
    reg [WAY_BITS-1:0] replace_way;
    reg is_prefetch_miss;
    // #148: Track whether current miss is for Port B
    reg miss_is_port_b;

    reg prefetch_pending;
    reg [ADDR_WIDTH-1:0] prefetch_pending_addr;

    // #148: Port B miss queue (1-deep: queue Port B miss while Port A miss in flight)
    reg portb_miss_pending;
    reg [ADDR_WIDTH-1:0] portb_miss_addr;
    reg [WORD_BITS-1:0] portb_miss_word;

    //------------------------------------------------------------------------
    // Replacement Policy
    //------------------------------------------------------------------------
    wire [WAY_BITS-1:0] victim_way = lru_array[req_index];
    wire [WAY_BITS-1:0] victim_way_b = lru_array[req_index_b];

    task update_lru;
        input [INDEX_BITS-1:0] index;
        input [WAY_BITS-1:0] accessed_way;
        begin
            if (NUM_WAYS == 2) begin
                lru_array[index] <= ~accessed_way;
            end else begin
                lru_array[index] <= accessed_way + 1'b1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] hit_count;
    reg [31:0] miss_count;
    reg [31:0] prefetch_hit_count;

    assign stat_hits          = hit_count;
    assign stat_misses        = miss_count;
    assign stat_prefetch_hits = prefetch_hit_count;

    //------------------------------------------------------------------------
    // Registered outputs
    //------------------------------------------------------------------------
    reg fetch_valid_r;
    reg [DATA_WIDTH-1:0] fetch_data_r;
    reg [LINE_BITS-1:0] fetch_line_data_r;
    reg mem_req_valid_r;
    reg [ADDR_WIDTH-1:0] mem_req_addr_r;
    reg [ADDR_WIDTH-1:0] fetch_addr_latched;
    reg invalidate_done_r;
    reg fetch_ready_r;

    // #148: Port B registered outputs
    reg fetch_valid_b_r;
    reg [DATA_WIDTH-1:0] fetch_data_b_r;
    reg [LINE_BITS-1:0] fetch_line_data_b_r;
    reg fetch_ready_b_r;

    // Prefetch next-line
    wire [ADDR_WIDTH-1:0] miss_next_line_addr = {miss_addr[ADDR_WIDTH-1:OFFSET_BITS] + 1'b1, {OFFSET_BITS{1'b0}}};

    wire [INDEX_BITS-1:0]   next_index   = miss_next_line_addr[OFFSET_BITS +: INDEX_BITS];
    wire [TAG_BITS-1:0]     next_tag     = miss_next_line_addr[ADDR_WIDTH-1 -: TAG_BITS];
    wire [NUM_WAYS-1:0] next_way_hit;

    generate
        for (w = 0; w < NUM_WAYS; w = w + 1) begin : gen_next_way_hit
            assign next_way_hit[w] = valid_array[next_index][w] &&
                                    (tag_array[next_index][w] == next_tag);
        end
    endgenerate

    wire next_line_cached = |next_way_hit;

    wire [PREFETCH_DEPTH-1:0] next_prefetch_hit_vec;
    generate
        for (w = 0; w < PREFETCH_DEPTH; w = w + 1) begin : gen_next_prefetch
            wire [ADDR_WIDTH-1:0] pf_line_addr = {prefetch_addr[w][ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            assign next_prefetch_hit_vec[w] = prefetch_valid[w] && (pf_line_addr == miss_next_line_addr);
        end
    endgenerate

    wire next_line_prefetched = |next_prefetch_hit_vec;
    wire need_prefetch = !next_line_cached && !next_line_prefetched;

    //------------------------------------------------------------------------
    // Main FSM
    //------------------------------------------------------------------------
    integer rst_i, rst_j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            fetch_valid_r <= 1'b0;
            fetch_data_r <= 0;
            fetch_line_data_r <= 0;
            fetch_ready_r <= 1'b1;
            fetch_valid_b_r <= 1'b0;
            fetch_data_b_r <= 0;
            fetch_line_data_b_r <= 0;
            fetch_ready_b_r <= 1'b1;
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
            miss_is_port_b <= 1'b0;
            prefetch_pending <= 1'b0;
            prefetch_pending_addr <= 0;
            prefetch_valid <= 0;
            prefetch_head <= 0;
            prefetch_tail <= 0;
            portb_miss_pending <= 1'b0;
            portb_miss_addr <= 0;
            portb_miss_word <= 0;

            for (rst_i = 0; rst_i < NUM_SETS; rst_i = rst_i + 1) begin
                valid_array[rst_i] <= 0;
                lru_array[rst_i] <= 0;
            end
        end else begin
            fetch_valid_r <= 1'b0;
            fetch_valid_b_r <= 1'b0;
            invalidate_done_r <= 1'b0;

            case (state)
                ST_IDLE: begin
                    fetch_ready_r <= 1'b1;
                    fetch_ready_b_r <= 1'b1;

                    if (invalidate_req) begin
                        state <= ST_INVALIDATE;
                        fetch_ready_r <= 1'b0;
                        fetch_ready_b_r <= 1'b0;
                        prefetch_pending <= 1'b0;
                    end else begin
                        // === Port A handling (priority) ===
                        if (fetch_req) begin
                            prefetch_pending <= 1'b0;

                            if (cache_hit) begin
                                hit_count <= hit_count + 1;
                                update_lru(req_index, hit_way);
                            end else if (prefetch_buffer_hit) begin
                                prefetch_hit_count <= prefetch_hit_count + 1;
                                tag_array[req_index][victim_way] <= req_tag;
                                data_array[req_index][victim_way] <= prefetch_data[prefetch_hit_idx];
                                valid_array[req_index][victim_way] <= 1'b1;
                                update_lru(req_index, victim_way);
                                prefetch_valid[prefetch_hit_idx] <= 1'b0;
                            end else begin
                                // Port A miss
                                fetch_addr_latched <= fetch_addr;
                                state <= ST_TAG_CHECK;
                                fetch_ready_r <= 1'b0;
                                fetch_ready_b_r <= 1'b0;
                                miss_is_port_b <= 1'b0;
                            end
                        end

                        // === Port B handling (concurrent with Port A hits) ===
                        // Port B can be served in same cycle if:
                        // 1. No bank conflict with Port A
                        // 2. Port A didn't miss (we're still in IDLE)
                        // 3. Port B hits in cache or prefetch buffer
                        if (fetch_req_b && !bank_conflict) begin
                            if (cache_hit_b) begin
                                hit_count <= hit_count + 1;
                                // Only update LRU if not conflicting with Port A's LRU update
                                if (!fetch_req || req_index != req_index_b)
                                    update_lru(req_index_b, hit_way_b);
                            end else if (prefetch_buffer_hit_b) begin
                                prefetch_hit_count <= prefetch_hit_count + 1;
                                tag_array[req_index_b][victim_way_b] <= req_tag_b;
                                data_array[req_index_b][victim_way_b] <= prefetch_data[prefetch_hit_idx_b];
                                valid_array[req_index_b][victim_way_b] <= 1'b1;
                                update_lru(req_index_b, victim_way_b);
                                // Don't invalidate same prefetch entry if Port A also hit it
                                if (prefetch_hit_idx_b != prefetch_hit_idx || !prefetch_buffer_hit)
                                    prefetch_valid[prefetch_hit_idx_b] <= 1'b0;
                            end else if (state == ST_IDLE) begin
                                // Port B miss — if Port A didn't also miss, handle Port B miss
                                if (!fetch_req || cache_hit || prefetch_buffer_hit) begin
                                    fetch_addr_latched <= fetch_addr_b;
                                    state <= ST_TAG_CHECK;
                                    fetch_ready_r <= 1'b0;
                                    fetch_ready_b_r <= 1'b0;
                                    miss_is_port_b <= 1'b1;
                                end else begin
                                    // Both ports missed — queue Port B
                                    portb_miss_pending <= 1'b1;
                                    portb_miss_addr <= {fetch_addr_b[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
                                    portb_miss_word <= fetch_addr_b[2 +: WORD_BITS];
                                end
                            end
                        end else if (fetch_req_b && bank_conflict) begin
                            // Bank conflict: Port B miss on cache check, queue if not a hit
                            // Actually re-check: bank_conflict means same index, so tag arrays
                            // read the same set. Port B can still detect a hit in the same set.
                            // But the LRU/fill conflict makes it unsafe to modify same set.
                            // For safety, if bank conflict AND Port A missed: queue Port B.
                            if (!fetch_req || cache_hit || prefetch_buffer_hit) begin
                                // Port A hit or not requesting — Port B can still check tags
                                // (same set, different tag is fine for read)
                                if (cache_hit_b) begin
                                    hit_count <= hit_count + 1;
                                    // Update LRU even under bank conflict so slot1 hit
                                    // traffic is reflected in replacement policy.
                                    // If Port A also updates the same set this cycle,
                                    // the later Port B update wins deterministically.
                                    update_lru(req_index_b, hit_way_b);
                                end
                                // Port B miss with bank conflict: queue it
                                else if (!cache_hit_b && !prefetch_buffer_hit_b) begin
                                    if (state == ST_IDLE) begin
                                        fetch_addr_latched <= fetch_addr_b;
                                        state <= ST_TAG_CHECK;
                                        fetch_ready_r <= 1'b0;
                                        fetch_ready_b_r <= 1'b0;
                                        miss_is_port_b <= 1'b1;
                                    end
                                end
                            end else begin
                                // Port A also missed — queue Port B
                                portb_miss_pending <= 1'b1;
                                portb_miss_addr <= {fetch_addr_b[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
                                portb_miss_word <= fetch_addr_b[2 +: WORD_BITS];
                            end
                        end

                        // Service queued Port B miss (when no new requests)
                        if (!fetch_req && !fetch_req_b && portb_miss_pending && state == ST_IDLE) begin
                            miss_addr <= portb_miss_addr;
                            miss_word <= portb_miss_word;
                            miss_is_port_b <= 1'b1;
                            portb_miss_pending <= 1'b0;
                            is_prefetch_miss <= 1'b0;
                            miss_count <= miss_count + 1;
                            replace_way <= lru_array[portb_miss_addr[OFFSET_BITS +: INDEX_BITS]];
                            state <= ST_MISS_REQ;
                            fetch_ready_r <= 1'b0;
                            fetch_ready_b_r <= 1'b0;
                        end else if (!fetch_req && !fetch_req_b && !portb_miss_pending &&
                                     prefetch_pending && state == ST_IDLE) begin
                            prefetch_pending <= 1'b0;
                            miss_addr <= prefetch_pending_addr;
                            is_prefetch_miss <= 1'b1;
                            state <= ST_PREFETCH;
                            fetch_ready_r <= 1'b0;
                            fetch_ready_b_r <= 1'b0;
                        end
                    end
                end

                ST_TAG_CHECK: begin
                    miss_count <= miss_count + 1;
                    miss_addr <= {fetch_addr_latched[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
                    miss_word <= fetch_addr_latched[2 +: WORD_BITS];
                    replace_way <= lru_array[fetch_addr_latched[OFFSET_BITS +: INDEX_BITS]];
                    is_prefetch_miss <= 1'b0;
                    state <= ST_MISS_REQ;
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
                        if (!is_prefetch_miss) begin
                            // Fill cache
                            tag_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= miss_addr[ADDR_WIDTH-1 -: TAG_BITS];
                            data_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= mem_resp_data;
                            valid_array[miss_addr[OFFSET_BITS +: INDEX_BITS]][replace_way] <= 1'b1;
                            update_lru(miss_addr[OFFSET_BITS +: INDEX_BITS], replace_way);

                            // Return data to correct port
                            if (miss_is_port_b) begin
                                fetch_data_b_r <= mem_resp_data[miss_word * DATA_WIDTH +: DATA_WIDTH];
                                fetch_line_data_b_r <= mem_resp_data;
                                fetch_valid_b_r <= 1'b1;
                            end else begin
                                fetch_data_r <= mem_resp_data[miss_word * DATA_WIDTH +: DATA_WIDTH];
                                fetch_line_data_r <= mem_resp_data;
                                fetch_valid_r <= 1'b1;
                            end

                            if (need_prefetch && !portb_miss_pending) begin
                                prefetch_pending <= 1'b1;
                                prefetch_pending_addr <= miss_next_line_addr;
                            end

                            // Check if queued Port B miss is now satisfied by this fill
                            if (portb_miss_pending &&
                                portb_miss_addr[ADDR_WIDTH-1:OFFSET_BITS] == miss_addr[ADDR_WIDTH-1:OFFSET_BITS]) begin
                                // Same line — serve from fill data
                                fetch_data_b_r <= mem_resp_data[portb_miss_word * DATA_WIDTH +: DATA_WIDTH];
                                fetch_line_data_b_r <= mem_resp_data;
                                fetch_valid_b_r <= 1'b1;
                                portb_miss_pending <= 1'b0;
                            end

                            state <= ST_IDLE;
                            fetch_ready_r <= 1'b1;
                            fetch_ready_b_r <= 1'b1;
                        end else begin
                            // Prefetch fill
                            prefetch_addr[prefetch_head] <= miss_addr;
                            prefetch_data[prefetch_head] <= mem_resp_data;
                            prefetch_valid[prefetch_head] <= 1'b1;
                            prefetch_head <= prefetch_head + 1'b1;

                            state <= ST_IDLE;
                            fetch_ready_r <= 1'b1;
                            fetch_ready_b_r <= 1'b1;
                        end
                    end
                end

                ST_FILL: begin
                    state <= ST_IDLE;
                    fetch_ready_r <= 1'b1;
                    fetch_ready_b_r <= 1'b1;
                end

                ST_PREFETCH: begin
                    mem_req_valid_r <= 1'b1;
                    mem_req_addr_r <= miss_addr;

                    if (mem_req_valid_r && mem_req_ready) begin
                        mem_req_valid_r <= 1'b0;
                        state <= ST_MISS_WAIT;
                    end
                end

                ST_INVALIDATE: begin
                    if (invalidate_all) begin
                        for (rst_i = 0; rst_i < NUM_SETS; rst_i = rst_i + 1) begin
                            valid_array[rst_i] <= 0;
                        end
                        prefetch_valid <= 0;
                    end else begin
                        for (rst_j = 0; rst_j < NUM_WAYS; rst_j = rst_j + 1) begin
                            if (valid_array[req_index][rst_j] &&
                                tag_array[req_index][rst_j] == req_tag) begin
                                valid_array[req_index][rst_j] <= 1'b0;
                            end
                        end
                    end
                    invalidate_done_r <= 1'b1;
                    state <= ST_IDLE;
                    fetch_ready_r <= 1'b1;
                    fetch_ready_b_r <= 1'b1;
                end

                default: begin
                    state <= ST_IDLE;
                    fetch_ready_r <= 1'b1;
                    fetch_ready_b_r <= 1'b1;
                end
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Port A Output — combinational hit bypass
    //------------------------------------------------------------------------
    wire combo_hit = (state == ST_IDLE) && fetch_req && cache_hit;
    wire combo_prefetch_hit = (state == ST_IDLE) && fetch_req && !cache_hit && prefetch_buffer_hit;

    assign fetch_ready     = fetch_ready_r;
    assign fetch_data      = combo_hit ? hit_data :
                            combo_prefetch_hit ? prefetch_hit_data :
                            fetch_data_r;
    assign fetch_line_data = combo_hit ? hit_line :
                            combo_prefetch_hit ? prefetch_data[prefetch_hit_idx] :
                            fetch_line_data_r;
    assign fetch_valid     = combo_hit || combo_prefetch_hit || fetch_valid_r;
    assign mem_req_valid   = mem_req_valid_r;
    assign mem_req_addr    = mem_req_addr_r;
    assign invalidate_done = invalidate_done_r;

    // Port A hit-bypass during miss
    wire miss_pending = (state == ST_TAG_CHECK) || (state == ST_MISS_REQ) || (state == ST_MISS_WAIT) || (state == ST_PREFETCH);
    assign fetch_hit_bypass           = miss_pending && fetch_req && cache_hit;
    assign fetch_hit_bypass_data      = hit_data;
    assign fetch_hit_bypass_line_data = hit_line;

    //------------------------------------------------------------------------
    // Port B Output — combinational hit bypass (#148)
    //------------------------------------------------------------------------
    wire combo_hit_b = (state == ST_IDLE) && fetch_req_b && cache_hit_b && !bank_conflict;
    wire combo_prefetch_hit_b = (state == ST_IDLE) && fetch_req_b && !cache_hit_b && prefetch_buffer_hit_b && !bank_conflict;
    // Bank conflict but still a cache hit (same set, different tag match is possible)
    wire combo_hit_b_banked = (state == ST_IDLE) && fetch_req_b && bank_conflict && cache_hit_b;

    assign fetch_ready_b     = fetch_ready_b_r;
    assign fetch_data_b      = (combo_hit_b || combo_hit_b_banked) ? hit_data_b :
                              combo_prefetch_hit_b ? prefetch_hit_data_b :
                              fetch_data_b_r;
    assign fetch_line_data_b = (combo_hit_b || combo_hit_b_banked) ? hit_line_b :
                              combo_prefetch_hit_b ? prefetch_data[prefetch_hit_idx_b] :
                              fetch_line_data_b_r;
    assign fetch_valid_b     = combo_hit_b || combo_hit_b_banked || combo_prefetch_hit_b || fetch_valid_b_r;

endmodule


//============================================================================
// Instruction Prefetch Unit (unchanged)
//============================================================================
module instruction_prefetch_unit #(
    parameter PREFETCH_DEPTH = 8,
    parameter ADDR_WIDTH     = 32,
    parameter DATA_WIDTH     = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire [ADDR_WIDTH-1:0]    base_addr,
    input  wire                     flush,
    input  wire                     branch_taken,
    input  wire [ADDR_WIDTH-1:0]    branch_target,
    output wire [DATA_WIDTH-1:0]    prefetch_data,
    output wire [ADDR_WIDTH-1:0]    prefetch_addr,
    output wire                     prefetch_valid,
    input  wire                     prefetch_consume,
    output wire                     icache_req,
    output wire [ADDR_WIDTH-1:0]    icache_addr,
    input  wire                     icache_ready,
    input  wire [DATA_WIDTH-1:0]    icache_data,
    input  wire                     icache_valid
);

    localparam PTR_W = $clog2(PREFETCH_DEPTH);
    localparam CNT_W = $clog2(PREFETCH_DEPTH + 1);

    reg [DATA_WIDTH-1:0] buffer_data [0:PREFETCH_DEPTH-1];
    reg [ADDR_WIDTH-1:0] buffer_addr [0:PREFETCH_DEPTH-1];
    reg [PREFETCH_DEPTH-1:0] buffer_valid;
    reg [PTR_W-1:0] head_ptr;
    reg [PTR_W-1:0] tail_ptr;
    reg [CNT_W-1:0] count;

    wire buffer_empty = (count == 0);
    wire buffer_full  = (count == PREFETCH_DEPTH);

    reg [ADDR_WIDTH-1:0] next_fetch_addr;
    reg fetch_pending;

    localparam ST_IDLE     = 2'd0;
    localparam ST_FETCH    = 2'd1;
    localparam ST_WAIT     = 2'd2;

    reg [1:0] state;

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
            if (flush || branch_taken) begin
                head_ptr <= 0;
                tail_ptr <= 0;
                count <= 0;
                buffer_valid <= 0;
                next_fetch_addr <= branch_taken ? branch_target : base_addr;
                fetch_pending <= 1'b0;
                state <= ST_IDLE;
            end else begin
                if (prefetch_consume && !buffer_empty) begin
                    buffer_valid[head_ptr] <= 1'b0;
                    head_ptr <= (head_ptr + 1) % PREFETCH_DEPTH;
                    count <= count - 1;
                end

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
                            buffer_data[tail_ptr] <= icache_data;
                            buffer_addr[tail_ptr] <= next_fetch_addr;
                            buffer_valid[tail_ptr] <= 1'b1;
                            tail_ptr <= (tail_ptr + 1) % PREFETCH_DEPTH;
                            count <= count + 1;
                            next_fetch_addr <= next_fetch_addr + 4;
                            fetch_pending <= 1'b0;
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

    assign prefetch_data  = buffer_data[head_ptr];
    assign prefetch_addr  = buffer_addr[head_ptr];
    assign prefetch_valid = buffer_valid[head_ptr] && !buffer_empty;
    assign icache_req     = (state == ST_FETCH);
    assign icache_addr    = next_fetch_addr;

endmodule
