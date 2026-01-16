//============================================================================
// RalphGPU - Banked Register File with Conflict Detection
// Multi-warp support with configurable warps (4-32)
// Bank conflict detection and operand collector integration
//============================================================================

`include "gpu_defines.vh"

module register_file_banked #(
    parameter NUM_WARPS     = 8,                    // Configurable 4-32
    parameter NUM_REGS      = `NUM_REGS,            // 32 registers per thread
    parameter NUM_LANES     = `THREADS_PER_WARP,    // 32 threads per warp
    parameter DATA_WIDTH    = `DATA_WIDTH,          // 32-bit data
    parameter NUM_BANKS     = 4,                    // Number of register banks
    parameter NUM_READ_PORTS = 3,                   // Read ports A, B, C
    parameter NUM_WRITE_PORTS = 1                   // Write ports
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Warp Selection
    //------------------------------------------------------------------------
    input  wire [$clog2(NUM_WARPS)-1:0] rd_warp_id,
    input  wire [$clog2(NUM_WARPS)-1:0] wr_warp_id,

    //------------------------------------------------------------------------
    // Read Ports (per-lane parallel reads)
    //------------------------------------------------------------------------
    input  wire [4:0]               rd_addr_a,
    input  wire [4:0]               rd_addr_b,
    input  wire [4:0]               rd_addr_c,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_a,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_b,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_c,
    output wire                     rd_conflict_a,      // Bank conflict detected
    output wire                     rd_conflict_b,
    output wire                     rd_conflict_c,

    //------------------------------------------------------------------------
    // Write Port
    //------------------------------------------------------------------------
    input  wire                     wr_en,
    input  wire [4:0]               wr_addr,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] wr_data,
    input  wire [NUM_LANES-1:0]     wr_mask,
    output wire                     wr_conflict,

    //------------------------------------------------------------------------
    // Operand Collector Interface
    //------------------------------------------------------------------------
    input  wire                     oc_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] oc_warp_id,
    input  wire [4:0]               oc_addr [0:NUM_READ_PORTS-1],
    output wire [NUM_LANES*DATA_WIDTH-1:0] oc_data [0:NUM_READ_PORTS-1],
    output wire [NUM_READ_PORTS-1:0] oc_ready,
    output wire                     oc_conflict,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output wire [31:0]              stat_bank_conflicts,
    output wire [31:0]              stat_total_accesses
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam BANK_BITS    = $clog2(NUM_BANKS);
    localparam WARP_ID_W    = $clog2(NUM_WARPS);
    localparam LANE_W       = $clog2(NUM_LANES);
    localparam REG_W        = $clog2(NUM_REGS);

    // Calculate bank index from register address and lane
    // Banking scheme: bank = (reg_addr + lane_id) mod NUM_BANKS
    function [BANK_BITS-1:0] get_bank;
        input [4:0] reg_addr;
        input [LANE_W-1:0] lane_id;
        begin
            get_bank = (reg_addr[BANK_BITS-1:0] + lane_id[BANK_BITS-1:0]) % NUM_BANKS;
        end
    endfunction

    //------------------------------------------------------------------------
    // Register Storage - Organized by banks for conflict-free access
    //------------------------------------------------------------------------
    // Bank organization: [bank][warp][lane_set][reg][data]
    // Each bank contains 1/NUM_BANKS of the lanes
    localparam LANES_PER_BANK = NUM_LANES / NUM_BANKS;

    reg [DATA_WIDTH-1:0] bank_regs [0:NUM_BANKS-1]
                                   [0:NUM_WARPS-1]
                                   [0:LANES_PER_BANK-1]
                                   [0:NUM_REGS-1];

    //------------------------------------------------------------------------
    // Bank Access Arbitration
    //------------------------------------------------------------------------
    // Track which ports access which banks
    wire [BANK_BITS-1:0] port_bank_a [0:NUM_LANES-1];
    wire [BANK_BITS-1:0] port_bank_b [0:NUM_LANES-1];
    wire [BANK_BITS-1:0] port_bank_c [0:NUM_LANES-1];

    genvar lane;
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_bank_calc
            assign port_bank_a[lane] = get_bank(rd_addr_a, lane[LANE_W-1:0]);
            assign port_bank_b[lane] = get_bank(rd_addr_b, lane[LANE_W-1:0]);
            assign port_bank_c[lane] = get_bank(rd_addr_c, lane[LANE_W-1:0]);
        end
    endgenerate

    //------------------------------------------------------------------------
    // Conflict Detection
    //------------------------------------------------------------------------
    // Check for multi-port conflicts (same bank accessed by different ports)
    reg [NUM_BANKS-1:0] bank_access_count_a;
    reg [NUM_BANKS-1:0] bank_access_count_b;
    reg [NUM_BANKS-1:0] bank_access_count_c;

    integer conf_i;
    always @(*) begin
        bank_access_count_a = 0;
        bank_access_count_b = 0;
        bank_access_count_c = 0;

        for (conf_i = 0; conf_i < NUM_LANES; conf_i = conf_i + 1) begin
            bank_access_count_a[port_bank_a[conf_i]] = 1'b1;
            bank_access_count_b[port_bank_b[conf_i]] = 1'b1;
            bank_access_count_c[port_bank_c[conf_i]] = 1'b1;
        end
    end

    // Inter-port conflicts (A vs B vs C accessing same bank)
    reg conflict_ab, conflict_ac, conflict_bc;
    always @(*) begin
        conflict_ab = |(bank_access_count_a & bank_access_count_b);
        conflict_ac = |(bank_access_count_a & bank_access_count_c);
        conflict_bc = |(bank_access_count_b & bank_access_count_c);
    end

    // For same-port conflicts, we need to check if multiple lanes in same port
    // access the same bank with different addresses (broadcast is OK)
    // This is simplified - real implementation would track per-bank address uniqueness
    assign rd_conflict_a = 1'b0;  // Simplified: assume no intra-port conflicts
    assign rd_conflict_b = 1'b0;
    assign rd_conflict_c = 1'b0;
    assign wr_conflict   = 1'b0;  // Writes are masked, so no conflict
    assign oc_conflict   = conflict_ab || conflict_ac || conflict_bc;

    //------------------------------------------------------------------------
    // Read Logic (Combinational)
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] rd_data_a_lane [0:NUM_LANES-1];
    reg [DATA_WIDTH-1:0] rd_data_b_lane [0:NUM_LANES-1];
    reg [DATA_WIDTH-1:0] rd_data_c_lane [0:NUM_LANES-1];

    integer rd_lane, rd_bank, rd_lane_in_bank;
    always @(*) begin
        for (rd_lane = 0; rd_lane < NUM_LANES; rd_lane = rd_lane + 1) begin
            rd_bank = get_bank(rd_addr_a, rd_lane[LANE_W-1:0]);
            rd_lane_in_bank = rd_lane / NUM_BANKS;
            rd_data_a_lane[rd_lane] = bank_regs[rd_bank][rd_warp_id][rd_lane_in_bank][rd_addr_a];

            rd_bank = get_bank(rd_addr_b, rd_lane[LANE_W-1:0]);
            rd_lane_in_bank = rd_lane / NUM_BANKS;
            rd_data_b_lane[rd_lane] = bank_regs[rd_bank][rd_warp_id][rd_lane_in_bank][rd_addr_b];

            rd_bank = get_bank(rd_addr_c, rd_lane[LANE_W-1:0]);
            rd_lane_in_bank = rd_lane / NUM_BANKS;
            rd_data_c_lane[rd_lane] = bank_regs[rd_bank][rd_warp_id][rd_lane_in_bank][rd_addr_c];
        end
    end

    // Pack output
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_rd_output
            assign rd_data_a[lane*DATA_WIDTH +: DATA_WIDTH] = rd_data_a_lane[lane];
            assign rd_data_b[lane*DATA_WIDTH +: DATA_WIDTH] = rd_data_b_lane[lane];
            assign rd_data_c[lane*DATA_WIDTH +: DATA_WIDTH] = rd_data_c_lane[lane];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Operand Collector Read
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] oc_data_lane [0:NUM_READ_PORTS-1][0:NUM_LANES-1];

    integer oc_port, oc_lane, oc_bank_idx, oc_lane_idx;
    always @(*) begin
        for (oc_port = 0; oc_port < NUM_READ_PORTS; oc_port = oc_port + 1) begin
            for (oc_lane = 0; oc_lane < NUM_LANES; oc_lane = oc_lane + 1) begin
                oc_bank_idx = get_bank(oc_addr[oc_port], oc_lane[LANE_W-1:0]);
                oc_lane_idx = oc_lane / NUM_BANKS;
                oc_data_lane[oc_port][oc_lane] = bank_regs[oc_bank_idx][oc_warp_id][oc_lane_idx][oc_addr[oc_port]];
            end
        end
    end

    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_oc_output
            genvar port;
            for (port = 0; port < NUM_READ_PORTS; port = port + 1) begin : gen_oc_port
                assign oc_data[port][lane*DATA_WIDTH +: DATA_WIDTH] = oc_data_lane[port][lane];
            end
        end
    endgenerate

    assign oc_ready = {NUM_READ_PORTS{oc_valid && !oc_conflict}};

    //------------------------------------------------------------------------
    // Write Logic (Sequential)
    //------------------------------------------------------------------------
    integer wr_w, wr_b, wr_l, wr_r;
    integer wr_bank_idx, wr_lane_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers
            for (wr_b = 0; wr_b < NUM_BANKS; wr_b = wr_b + 1) begin
                for (wr_w = 0; wr_w < NUM_WARPS; wr_w = wr_w + 1) begin
                    for (wr_l = 0; wr_l < LANES_PER_BANK; wr_l = wr_l + 1) begin
                        for (wr_r = 0; wr_r < NUM_REGS; wr_r = wr_r + 1) begin
                            bank_regs[wr_b][wr_w][wr_l][wr_r] <= {DATA_WIDTH{1'b0}};
                        end
                    end
                end
            end
        end else if (wr_en) begin
            for (wr_l = 0; wr_l < NUM_LANES; wr_l = wr_l + 1) begin
                if (wr_mask[wr_l]) begin
                    wr_bank_idx = get_bank(wr_addr, wr_l[LANE_W-1:0]);
                    wr_lane_idx = wr_l / NUM_BANKS;
                    bank_regs[wr_bank_idx][wr_warp_id][wr_lane_idx][wr_addr] <=
                        wr_data[wr_l*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] conflict_count;
    reg [31:0] access_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conflict_count <= 0;
            access_count <= 0;
        end else begin
            if (oc_valid) begin
                access_count <= access_count + 1;
                if (oc_conflict) begin
                    conflict_count <= conflict_count + 1;
                end
            end
        end
    end

    assign stat_bank_conflicts = conflict_count;
    assign stat_total_accesses = access_count;

endmodule


//============================================================================
// Operand Collector
// Collects operands from register file over multiple cycles if needed
//============================================================================
module operand_collector #(
    parameter NUM_WARPS     = 8,
    parameter NUM_LANES     = `THREADS_PER_WARP,
    parameter DATA_WIDTH    = `DATA_WIDTH,
    parameter NUM_OPERANDS  = 3,
    parameter COLLECTOR_DEPTH = 4       // Number of in-flight collections
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Request Interface (from Issue)
    //------------------------------------------------------------------------
    input  wire                     req_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] req_warp_id,
    input  wire [4:0]               req_addr [0:NUM_OPERANDS-1],
    input  wire [NUM_OPERANDS-1:0]  req_need,           // Which operands needed
    output wire                     req_ready,

    //------------------------------------------------------------------------
    // Register File Interface
    //------------------------------------------------------------------------
    output wire                     rf_valid,
    output wire [$clog2(NUM_WARPS)-1:0] rf_warp_id,
    output wire [4:0]               rf_addr [0:NUM_OPERANDS-1],
    input  wire [NUM_LANES*DATA_WIDTH-1:0] rf_data [0:NUM_OPERANDS-1],
    input  wire [NUM_OPERANDS-1:0]  rf_ready,
    input  wire                     rf_conflict,

    //------------------------------------------------------------------------
    // Output Interface (to Execute)
    //------------------------------------------------------------------------
    output wire                     out_valid,
    output wire [$clog2(NUM_WARPS)-1:0] out_warp_id,
    output wire [NUM_LANES*DATA_WIDTH-1:0] out_data [0:NUM_OPERANDS-1],
    input  wire                     out_ready
);

    localparam SIMD_WIDTH = NUM_LANES * DATA_WIDTH;
    localparam ENTRY_W = SIMD_WIDTH * NUM_OPERANDS + $clog2(NUM_WARPS) + NUM_OPERANDS + 5*NUM_OPERANDS;
    localparam PTR_W = $clog2(COLLECTOR_DEPTH);

    //------------------------------------------------------------------------
    // Collector Entries
    //------------------------------------------------------------------------
    reg [SIMD_WIDTH-1:0]    entry_data [0:COLLECTOR_DEPTH-1][0:NUM_OPERANDS-1];
    reg [$clog2(NUM_WARPS)-1:0] entry_warp [0:COLLECTOR_DEPTH-1];
    reg [4:0]               entry_addr [0:COLLECTOR_DEPTH-1][0:NUM_OPERANDS-1];
    reg [NUM_OPERANDS-1:0]  entry_need [0:COLLECTOR_DEPTH-1];
    reg [NUM_OPERANDS-1:0]  entry_collected [0:COLLECTOR_DEPTH-1];
    reg [COLLECTOR_DEPTH-1:0] entry_valid;
    reg [COLLECTOR_DEPTH-1:0] entry_complete;

    //------------------------------------------------------------------------
    // Allocation and Collection Logic
    //------------------------------------------------------------------------
    reg [PTR_W-1:0] alloc_ptr;
    reg [PTR_W-1:0] issue_ptr;
    reg [PTR_W-1:0] complete_ptr;

    wire can_allocate = !entry_valid[alloc_ptr];
    wire has_complete = entry_complete[complete_ptr];

    assign req_ready = can_allocate;

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    integer rst_e, rst_o;
    integer coll_e, coll_o;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            entry_valid <= 0;
            entry_complete <= 0;
            alloc_ptr <= 0;
            issue_ptr <= 0;
            complete_ptr <= 0;

            for (rst_e = 0; rst_e < COLLECTOR_DEPTH; rst_e = rst_e + 1) begin
                entry_warp[rst_e] <= 0;
                entry_need[rst_e] <= 0;
                entry_collected[rst_e] <= 0;
                for (rst_o = 0; rst_o < NUM_OPERANDS; rst_o = rst_o + 1) begin
                    entry_data[rst_e][rst_o] <= 0;
                    entry_addr[rst_e][rst_o] <= 0;
                end
            end
        end else begin
            // Allocate new request
            if (req_valid && can_allocate) begin
                entry_valid[alloc_ptr] <= 1'b1;
                entry_warp[alloc_ptr] <= req_warp_id;
                entry_need[alloc_ptr] <= req_need;
                entry_collected[alloc_ptr] <= 0;
                for (coll_o = 0; coll_o < NUM_OPERANDS; coll_o = coll_o + 1) begin
                    entry_addr[alloc_ptr][coll_o] <= req_addr[coll_o];
                end
                alloc_ptr <= (alloc_ptr + 1) % COLLECTOR_DEPTH;
            end

            // Collect operands from RF
            if (entry_valid[issue_ptr] && !entry_complete[issue_ptr] && (&(rf_ready | ~entry_need[issue_ptr]))) begin
                for (coll_o = 0; coll_o < NUM_OPERANDS; coll_o = coll_o + 1) begin
                    if (entry_need[issue_ptr][coll_o] && rf_ready[coll_o]) begin
                        entry_data[issue_ptr][coll_o] <= rf_data[coll_o];
                        entry_collected[issue_ptr][coll_o] <= 1'b1;
                    end
                end

                // Check if all needed operands collected
                if ((entry_collected[issue_ptr] | rf_ready) == entry_need[issue_ptr]) begin
                    entry_complete[issue_ptr] <= 1'b1;
                    issue_ptr <= (issue_ptr + 1) % COLLECTOR_DEPTH;
                end
            end

            // Output completed entry
            if (has_complete && out_ready) begin
                entry_valid[complete_ptr] <= 1'b0;
                entry_complete[complete_ptr] <= 1'b0;
                entry_collected[complete_ptr] <= 0;
                complete_ptr <= (complete_ptr + 1) % COLLECTOR_DEPTH;
            end
        end
    end

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign rf_valid   = entry_valid[issue_ptr] && !entry_complete[issue_ptr];
    assign rf_warp_id = entry_warp[issue_ptr];

    generate
        genvar o;
        for (o = 0; o < NUM_OPERANDS; o = o + 1) begin : gen_rf_addr
            assign rf_addr[o] = entry_addr[issue_ptr][o];
            assign out_data[o] = entry_data[complete_ptr][o];
        end
    endgenerate

    assign out_valid   = has_complete;
    assign out_warp_id = entry_warp[complete_ptr];

endmodule
