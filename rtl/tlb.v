//============================================================================
// RalphGPU - Translation Lookaside Buffer (TLB)
// Two-level TLB for virtual to physical address translation
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

//============================================================================
// L1 TLB - Per SM, small and fast
//============================================================================
module l1_tlb #(
    parameter NUM_ENTRIES   = `L1_TLB_ENTRIES,      // 32 entries
    parameter NUM_WAYS      = `L1_TLB_WAYS,         // 4-way
    parameter VA_WIDTH      = `VA_BITS,             // 48-bit VA
    parameter PA_WIDTH      = `PA_BITS,             // 40-bit PA
    parameter PAGE_SIZE     = `L1_TLB_PAGE_SIZE     // 4KB
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Translation request
    input  wire                     req_valid,
    input  wire [VA_WIDTH-1:0]      req_va,
    input  wire                     req_write,      // For permission check

    // Translation response
    output reg                      resp_valid,
    output reg                      resp_hit,
    output reg  [PA_WIDTH-1:0]      resp_pa,
    output reg                      resp_fault,     // Permission fault

    // L2 TLB interface (on miss)
    output reg                      l2_req_valid,
    output reg  [VA_WIDTH-1:0]      l2_req_va,
    input  wire                     l2_resp_valid,
    input  wire                     l2_resp_hit,
    input  wire [PA_WIDTH-1:0]      l2_resp_pa,
    input  wire [3:0]               l2_resp_perm,   // R/W/X/U permissions

    // Invalidation
    input  wire                     inv_valid,
    input  wire [VA_WIDTH-1:0]      inv_va,
    input  wire                     inv_all         // Flush entire TLB
);

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam PAGE_OFFSET_BITS = $clog2(PAGE_SIZE);            // 12 bits
    localparam VPN_BITS         = VA_WIDTH - PAGE_OFFSET_BITS;  // 36 bits
    localparam PPN_BITS         = PA_WIDTH - PAGE_OFFSET_BITS;  // 28 bits
    localparam NUM_SETS         = NUM_ENTRIES / NUM_WAYS;
    localparam INDEX_BITS       = $clog2(NUM_SETS);
    localparam TAG_BITS         = VPN_BITS - INDEX_BITS;

    //------------------------------------------------------------------------
    // TLB Entry: [valid][tag][ppn][permissions]
    //------------------------------------------------------------------------
    reg [TAG_BITS-1:0]      tag_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [PPN_BITS-1:0]      ppn_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [3:0]               perm_array  [0:NUM_SETS-1][0:NUM_WAYS-1];  // R/W/X/U
    reg [NUM_WAYS-1:0]      valid_array [0:NUM_SETS-1];

    // LRU for replacement
    reg [NUM_WAYS-2:0]      lru_state   [0:NUM_SETS-1];

    //------------------------------------------------------------------------
    // Address Parsing
    //------------------------------------------------------------------------
    wire [PAGE_OFFSET_BITS-1:0] va_offset = req_va[PAGE_OFFSET_BITS-1:0];
    wire [VPN_BITS-1:0]         va_vpn    = req_va[VA_WIDTH-1:PAGE_OFFSET_BITS];
    wire [INDEX_BITS-1:0]       va_index  = va_vpn[INDEX_BITS-1:0];
    wire [TAG_BITS-1:0]         va_tag    = va_vpn[VPN_BITS-1:INDEX_BITS];

    //------------------------------------------------------------------------
    // Tag Comparison (Combinational)
    //------------------------------------------------------------------------
    reg [NUM_WAYS-1:0] way_hit;
    reg [$clog2(NUM_WAYS)-1:0] hit_way;
    reg tlb_hit;

    integer w;
    always @(*) begin
        way_hit = 0;
        hit_way = 0;
        tlb_hit = 0;

        for (w = 0; w < NUM_WAYS; w = w + 1) begin
            if (valid_array[va_index][w] &&
                tag_array[va_index][w] == va_tag) begin
                way_hit[w] = 1'b1;
                hit_way = w[$clog2(NUM_WAYS)-1:0];
                tlb_hit = 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Permission Check
    //------------------------------------------------------------------------
    wire [3:0] hit_perm = perm_array[va_index][hit_way];
    wire perm_ok = req_write ? hit_perm[1] : hit_perm[0];  // W or R

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam S_IDLE       = 2'd0;
    localparam S_LOOKUP     = 2'd1;
    localparam S_L2_WAIT    = 2'd2;
    localparam S_FILL       = 2'd3;

    reg [1:0] state;
    reg [VA_WIDTH-1:0] va_reg;
    reg write_reg;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            resp_valid <= 0;
            resp_hit <= 0;
            resp_pa <= 0;
            resp_fault <= 0;
            l2_req_valid <= 0;
            l2_req_va <= 0;
            va_reg <= 0;
            write_reg <= 0;

            // Initialize arrays
            for (i = 0; i < NUM_SETS; i = i + 1) begin
                valid_array[i] <= 0;
                lru_state[i] <= 0;
            end
        end else begin
            // Default outputs
            resp_valid <= 0;
            l2_req_valid <= 0;

            // Invalidation handling
            if (inv_valid) begin
                if (inv_all) begin
                    for (i = 0; i < NUM_SETS; i = i + 1) begin
                        valid_array[i] <= 0;
                    end
                end else begin
                    // Selective invalidation
                    // (simplified: invalidate entire set)
                    valid_array[inv_va[PAGE_OFFSET_BITS +: INDEX_BITS]] <= 0;
                end
            end

            case (state)
                S_IDLE: begin
                    if (req_valid) begin
                        va_reg <= req_va;
                        write_reg <= req_write;
                        state <= S_LOOKUP;
                    end
                end

                S_LOOKUP: begin
                    if (tlb_hit) begin
                        // TLB Hit
                        resp_valid <= 1'b1;
                        resp_hit <= 1'b1;
                        resp_pa <= {ppn_array[va_index][hit_way], va_offset};
                        resp_fault <= !perm_ok;

                        // Update LRU
                        lru_state[va_index] <= lru_state[va_index] ^ (1 << hit_way);
                        state <= S_IDLE;
                    end else begin
                        // TLB Miss - query L2
                        l2_req_valid <= 1'b1;
                        l2_req_va <= va_reg;
                        state <= S_L2_WAIT;
                    end
                end

                S_L2_WAIT: begin
                    if (l2_resp_valid) begin
                        if (l2_resp_hit) begin
                            // Fill from L2
                            state <= S_FILL;
                        end else begin
                            // Page fault (L2 miss)
                            resp_valid <= 1'b1;
                            resp_hit <= 1'b0;
                            resp_fault <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
                end

                S_FILL: begin
                    // Find victim way (LRU or invalid)
                    reg [$clog2(NUM_WAYS)-1:0] victim;
                    victim = 0;

                    for (w = 0; w < NUM_WAYS; w = w + 1) begin
                        if (!valid_array[va_index][w]) victim = w[$clog2(NUM_WAYS)-1:0];
                    end
                    if (valid_array[va_index] == {NUM_WAYS{1'b1}}) begin
                        victim = lru_state[va_index][$clog2(NUM_WAYS)-1:0];
                    end

                    // Install new entry
                    tag_array[va_index][victim] <= va_tag;
                    ppn_array[va_index][victim] <= l2_resp_pa[PA_WIDTH-1:PAGE_OFFSET_BITS];
                    perm_array[va_index][victim] <= l2_resp_perm;
                    valid_array[va_index][victim] <= 1'b1;
                    lru_state[va_index] <= lru_state[va_index] ^ (1 << victim);

                    // Return translation
                    resp_valid <= 1'b1;
                    resp_hit <= 1'b1;
                    resp_pa <= l2_resp_pa;
                    resp_fault <= write_reg ? !l2_resp_perm[1] : !l2_resp_perm[0];

                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule


//============================================================================
// L2 TLB - Shared across SMs, larger capacity
//============================================================================
module l2_tlb #(
    parameter NUM_ENTRIES   = `L2_TLB_ENTRIES,      // 512 entries
    parameter NUM_WAYS      = `L2_TLB_WAYS,         // 8-way
    parameter VA_WIDTH      = `VA_BITS,
    parameter PA_WIDTH      = `PA_BITS,
    parameter PAGE_SIZE     = `L1_TLB_PAGE_SIZE,
    parameter NUM_PORTS     = `NUM_SM               // One port per SM
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Requests from L1 TLBs (multi-port)
    input  wire [NUM_PORTS-1:0]     req_valid,
    input  wire [NUM_PORTS*VA_WIDTH-1:0] req_va,

    // Responses to L1 TLBs
    output reg  [NUM_PORTS-1:0]     resp_valid,
    output reg  [NUM_PORTS-1:0]     resp_hit,
    output reg  [NUM_PORTS*PA_WIDTH-1:0] resp_pa,
    output reg  [NUM_PORTS*4-1:0]   resp_perm,

    // Page Table Walker interface (on L2 miss)
    output reg                      ptw_req_valid,
    output reg  [VA_WIDTH-1:0]      ptw_req_va,
    input  wire                     ptw_resp_valid,
    input  wire                     ptw_resp_hit,
    input  wire [PA_WIDTH-1:0]      ptw_resp_pa,
    input  wire [3:0]               ptw_resp_perm,

    // Global invalidation
    input  wire                     inv_valid,
    input  wire                     inv_all
);

    // Similar structure to L1 TLB but larger and multi-ported
    // Implementation follows same pattern with arbitration

    localparam PAGE_OFFSET_BITS = $clog2(PAGE_SIZE);
    localparam VPN_BITS         = VA_WIDTH - PAGE_OFFSET_BITS;
    localparam PPN_BITS         = PA_WIDTH - PAGE_OFFSET_BITS;
    localparam NUM_SETS         = NUM_ENTRIES / NUM_WAYS;
    localparam INDEX_BITS       = $clog2(NUM_SETS);
    localparam TAG_BITS         = VPN_BITS - INDEX_BITS;

    // Storage arrays
    reg [TAG_BITS-1:0]      tag_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [PPN_BITS-1:0]      ppn_array   [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [3:0]               perm_array  [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [NUM_WAYS-1:0]      valid_array [0:NUM_SETS-1];
    reg [NUM_WAYS-2:0]      lru_state   [0:NUM_SETS-1];

    // Request arbitration (round-robin)
    reg [$clog2(NUM_PORTS)-1:0] arb_ptr;
    reg [$clog2(NUM_PORTS)-1:0] active_port;
    reg req_pending;

    // State machine
    localparam S_IDLE   = 2'd0;
    localparam S_LOOKUP = 2'd1;
    localparam S_PTW    = 2'd2;
    localparam S_FILL   = 2'd3;

    reg [1:0] state;
    reg [VA_WIDTH-1:0] va_reg;

    integer i, p, w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            resp_valid <= 0;
            resp_hit <= 0;
            resp_pa <= 0;
            resp_perm <= 0;
            ptw_req_valid <= 0;
            ptw_req_va <= 0;
            arb_ptr <= 0;
            req_pending <= 0;

            for (i = 0; i < NUM_SETS; i = i + 1) begin
                valid_array[i] <= 0;
                lru_state[i] <= 0;
            end
        end else begin
            resp_valid <= 0;
            ptw_req_valid <= 0;

            // Invalidation
            if (inv_valid && inv_all) begin
                for (i = 0; i < NUM_SETS; i = i + 1) begin
                    valid_array[i] <= 0;
                end
            end

            case (state)
                S_IDLE: begin
                    // Round-robin arbitration
                    for (p = 0; p < NUM_PORTS; p = p + 1) begin
                        if (req_valid[(arb_ptr + p) % NUM_PORTS] && !req_pending) begin
                            active_port <= (arb_ptr + p) % NUM_PORTS;
                            va_reg <= req_va[((arb_ptr + p) % NUM_PORTS) * VA_WIDTH +: VA_WIDTH];
                            req_pending <= 1'b1;
                            arb_ptr <= (arb_ptr + p + 1) % NUM_PORTS;
                            state <= S_LOOKUP;
                        end
                    end
                end

                S_LOOKUP: begin
                    // Tag lookup (simplified - single cycle)
                    reg [INDEX_BITS-1:0] idx;
                    reg [TAG_BITS-1:0] tag;
                    reg hit_found;
                    reg [$clog2(NUM_WAYS)-1:0] hit_w;

                    idx = va_reg[PAGE_OFFSET_BITS +: INDEX_BITS];
                    tag = va_reg[VA_WIDTH-1 -: TAG_BITS];
                    hit_found = 0;
                    hit_w = 0;

                    for (w = 0; w < NUM_WAYS; w = w + 1) begin
                        if (valid_array[idx][w] && tag_array[idx][w] == tag) begin
                            hit_found = 1'b1;
                            hit_w = w[$clog2(NUM_WAYS)-1:0];
                        end
                    end

                    if (hit_found) begin
                        resp_valid[active_port] <= 1'b1;
                        resp_hit[active_port] <= 1'b1;
                        resp_pa[active_port*PA_WIDTH +: PA_WIDTH] <=
                            {ppn_array[idx][hit_w], va_reg[PAGE_OFFSET_BITS-1:0]};
                        resp_perm[active_port*4 +: 4] <= perm_array[idx][hit_w];
                        lru_state[idx] <= lru_state[idx] ^ (1 << hit_w);
                        req_pending <= 0;
                        state <= S_IDLE;
                    end else begin
                        // Miss - initiate page table walk
                        ptw_req_valid <= 1'b1;
                        ptw_req_va <= va_reg;
                        state <= S_PTW;
                    end
                end

                S_PTW: begin
                    if (ptw_resp_valid) begin
                        if (ptw_resp_hit) begin
                            state <= S_FILL;
                        end else begin
                            // Page fault
                            resp_valid[active_port] <= 1'b1;
                            resp_hit[active_port] <= 1'b0;
                            req_pending <= 0;
                            state <= S_IDLE;
                        end
                    end
                end

                S_FILL: begin
                    reg [INDEX_BITS-1:0] idx;
                    reg [TAG_BITS-1:0] tag;
                    reg [$clog2(NUM_WAYS)-1:0] victim;

                    idx = va_reg[PAGE_OFFSET_BITS +: INDEX_BITS];
                    tag = va_reg[VA_WIDTH-1 -: TAG_BITS];

                    // Find victim
                    victim = 0;
                    for (w = 0; w < NUM_WAYS; w = w + 1) begin
                        if (!valid_array[idx][w]) victim = w[$clog2(NUM_WAYS)-1:0];
                    end
                    if (valid_array[idx] == {NUM_WAYS{1'b1}}) begin
                        victim = lru_state[idx][$clog2(NUM_WAYS)-1:0];
                    end

                    // Install entry
                    tag_array[idx][victim] <= tag;
                    ppn_array[idx][victim] <= ptw_resp_pa[PA_WIDTH-1:PAGE_OFFSET_BITS];
                    perm_array[idx][victim] <= ptw_resp_perm;
                    valid_array[idx][victim] <= 1'b1;
                    lru_state[idx] <= lru_state[idx] ^ (1 << victim);

                    // Return result
                    resp_valid[active_port] <= 1'b1;
                    resp_hit[active_port] <= 1'b1;
                    resp_pa[active_port*PA_WIDTH +: PA_WIDTH] <= ptw_resp_pa;
                    resp_perm[active_port*4 +: 4] <= ptw_resp_perm;

                    req_pending <= 0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
