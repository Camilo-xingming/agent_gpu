/* verilator lint_off BLKSEQ */
//============================================================================
// RalphGPU - Banked Register File with Conflict Detection and ECC
// Multi-warp support with configurable warps (4-32)
// Bank conflict detection and operand collector integration
// RAS Features: SEC-DED ECC for single-bit error correction and double-bit detection
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module register_file_banked #(
    parameter NUM_WARPS     = 8,                    // Configurable 4-32
    parameter NUM_REGS      = `NUM_REGS,            // 32 registers per thread
    parameter NUM_LANES     = `THREADS_PER_WARP,    // 32 threads per warp
    parameter DATA_WIDTH    = `DATA_WIDTH,          // 32-bit data
    parameter NUM_BANKS     = 4,                    // Number of register banks
    parameter NUM_READ_PORTS = 3,                   // Read ports A, B, C
    parameter NUM_WRITE_PORTS = 1,                  // Write ports
    parameter ECC_ENABLE    = 1,                    // Enable ECC (RAS feature)
    parameter ECC_BITS      = 7                     // SEC-DED for 32-bit data needs 7 bits
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
    output wire [31:0]              stat_total_accesses,

    //------------------------------------------------------------------------
    // RAS/ECC Signals
    //------------------------------------------------------------------------
    output wire                     ecc_error_corrected,    // Single-bit error corrected
    output wire                     ecc_error_detected,     // Double-bit error detected (uncorrectable)
    output wire [31:0]              stat_ecc_corrections,   // Count of corrected errors
    output wire [31:0]              stat_ecc_uncorrectable, // Count of uncorrectable errors
    output wire [$clog2(NUM_WARPS)-1:0] ecc_error_warp,     // Warp ID with error
    output wire [4:0]               ecc_error_reg,          // Register with error
    output wire [$clog2(NUM_LANES)-1:0] ecc_error_lane      // Lane with error
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam BANK_BITS    = $clog2(NUM_BANKS);
    localparam WARP_ID_W    = $clog2(NUM_WARPS);
    localparam LANE_W       = $clog2(NUM_LANES);
    localparam REG_W        = $clog2(NUM_REGS);

    // ECC parameters
    localparam PROTECTED_WIDTH = ECC_ENABLE ? (DATA_WIDTH + ECC_BITS) : DATA_WIDTH;

    // Calculate bank index from register address and lane
    // Banking scheme: bank = (reg_addr + lane_id) mod NUM_BANKS
    function [BANK_BITS-1:0] get_bank;
        input [4:0] reg_addr;
        input [LANE_W-1:0] lane_id;
        reg [BANK_BITS:0] bank_sum;
        begin
            bank_sum = {1'b0, reg_addr[BANK_BITS-1:0]} + {1'b0, lane_id[BANK_BITS-1:0]};
            get_bank = bank_sum[BANK_BITS-1:0] % NUM_BANKS[BANK_BITS-1:0];
        end
    endfunction

    //------------------------------------------------------------------------
    // SEC-DED ECC Functions (Hamming Code with overall parity)
    // For 32-bit data: uses 7 check bits (SEC-DED)
    //------------------------------------------------------------------------

    // Calculate ECC syndrome/check bits for 32-bit data
    function [ECC_BITS-1:0] calc_ecc;
        input [DATA_WIDTH-1:0] data;
        reg [ECC_BITS-2:0] parity;      // 6 parity bits
        reg overall_parity;
        begin
            // Hamming (38,32) with overall parity
            // p0 covers bits 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
            parity[0] = data[0] ^ data[2] ^ data[4] ^ data[6] ^ data[8] ^ data[10] ^
                        data[12] ^ data[14] ^ data[16] ^ data[18] ^ data[20] ^ data[22] ^
                        data[24] ^ data[26] ^ data[28] ^ data[30];
            // p1 covers bits 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
            parity[1] = data[1] ^ data[2] ^ data[5] ^ data[6] ^ data[9] ^ data[10] ^
                        data[13] ^ data[14] ^ data[17] ^ data[18] ^ data[21] ^ data[22] ^
                        data[25] ^ data[26] ^ data[29] ^ data[30];
            // p2 covers bits 4-7, 12-15, 20-23, 28-31
            parity[2] = data[3] ^ data[4] ^ data[5] ^ data[6] ^ data[11] ^ data[12] ^
                        data[13] ^ data[14] ^ data[19] ^ data[20] ^ data[21] ^ data[22] ^
                        data[27] ^ data[28] ^ data[29] ^ data[30];
            // p3 covers bits 8-15, 24-31
            parity[3] = data[7] ^ data[8] ^ data[9] ^ data[10] ^ data[11] ^ data[12] ^
                        data[13] ^ data[14] ^ data[23] ^ data[24] ^ data[25] ^ data[26] ^
                        data[27] ^ data[28] ^ data[29] ^ data[30];
            // p4 covers bits 16-31
            parity[4] = data[15] ^ data[16] ^ data[17] ^ data[18] ^ data[19] ^ data[20] ^
                        data[21] ^ data[22] ^ data[23] ^ data[24] ^ data[25] ^ data[26] ^
                        data[27] ^ data[28] ^ data[29] ^ data[30];
            // p5 extends coverage
            parity[5] = data[31];

            // Overall parity for double-error detection
            overall_parity = ^data ^ ^parity;

            calc_ecc = {overall_parity, parity};
        end
    endfunction

    // Decode ECC - returns {corrected_data, single_error, double_error}
    function [DATA_WIDTH+1:0] decode_ecc;
        input [DATA_WIDTH-1:0] data;
        input [ECC_BITS-1:0] stored_ecc;
        reg [ECC_BITS-1:0] computed_ecc;
        reg [ECC_BITS-1:0] syndrome;
        reg single_error, double_error;
        reg [DATA_WIDTH-1:0] corrected_data;
        reg [5:0] error_position;
        begin
            computed_ecc = calc_ecc(data);
            syndrome = stored_ecc ^ computed_ecc;

            single_error = 1'b0;
            double_error = 1'b0;
            corrected_data = data;
            error_position = syndrome[5:0];

            if (syndrome == 0) begin
                // No error
                single_error = 1'b0;
                double_error = 1'b0;
            end else if (syndrome[6] == 1'b1) begin
                // Odd parity - single-bit error (correctable)
                single_error = 1'b1;
                if (error_position != 0 && error_position <= DATA_WIDTH) begin
                    // Error in data bit - correct it
                    corrected_data[error_position-1] = ~data[error_position-1];
                end
                // else error in check bit - data is already correct
            end else begin
                // Even parity with non-zero syndrome - double-bit error (uncorrectable)
                double_error = 1'b1;
            end

            decode_ecc = {double_error, single_error, corrected_data};
        end
    endfunction

    //------------------------------------------------------------------------
    // Register Storage - Organized by banks for conflict-free access
    //------------------------------------------------------------------------
    // Bank organization: [bank][warp][lane_set][reg][data+ecc]
    // Each bank contains 1/NUM_BANKS of the lanes
    localparam LANES_PER_BANK = NUM_LANES / NUM_BANKS;

    // Storage: simple 3D array [warp][lane][reg] for iverilog compatibility.
    // Banking is a physical optimization; functionally equivalent to flat layout.
    // iverilog cannot handle always @(*) or assign with 4D bank-indexed arrays.
    reg [PROTECTED_WIDTH-1:0] sim_regs [0:NUM_WARPS-1][0:NUM_LANES-1][0:NUM_REGS-1];

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
    // Read Logic — generate + assign with 3D array (iverilog compatible)
    //------------------------------------------------------------------------
    // iverilog cannot track multi-dim array changes in always @(*) sensitivity.
    // Using generate + assign with simple 3D indexing (like register_file.v).
    wire [NUM_LANES-1:0] rd_single_error_a, rd_single_error_b, rd_single_error_c;
    wire [NUM_LANES-1:0] rd_double_error_a, rd_double_error_b, rd_double_error_c;

    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_rd_logic
            // Raw data from sim_regs (3D: warp, lane, reg)
            wire [PROTECTED_WIDTH-1:0] raw_a = sim_regs[rd_warp_id][lane][rd_addr_a];
            wire [PROTECTED_WIDTH-1:0] raw_b = sim_regs[rd_warp_id][lane][rd_addr_b];
            wire [PROTECTED_WIDTH-1:0] raw_c = sim_regs[rd_warp_id][lane][rd_addr_c];

            // ECC decode
            wire [DATA_WIDTH+1:0] dec_a = decode_ecc(raw_a[DATA_WIDTH-1:0], raw_a[PROTECTED_WIDTH-1:DATA_WIDTH]);
            wire [DATA_WIDTH+1:0] dec_b = decode_ecc(raw_b[DATA_WIDTH-1:0], raw_b[PROTECTED_WIDTH-1:DATA_WIDTH]);
            wire [DATA_WIDTH+1:0] dec_c = decode_ecc(raw_c[DATA_WIDTH-1:0], raw_c[PROTECTED_WIDTH-1:DATA_WIDTH]);

            // Output data
            assign rd_data_a[lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? dec_a[DATA_WIDTH-1:0] : raw_a[DATA_WIDTH-1:0];
            assign rd_data_b[lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? dec_b[DATA_WIDTH-1:0] : raw_b[DATA_WIDTH-1:0];
            assign rd_data_c[lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? dec_c[DATA_WIDTH-1:0] : raw_c[DATA_WIDTH-1:0];

            // ECC error flags
            assign rd_single_error_a[lane] = ECC_ENABLE ? dec_a[DATA_WIDTH]   : 1'b0;
            assign rd_single_error_b[lane] = ECC_ENABLE ? dec_b[DATA_WIDTH]   : 1'b0;
            assign rd_single_error_c[lane] = ECC_ENABLE ? dec_c[DATA_WIDTH]   : 1'b0;
            assign rd_double_error_a[lane] = ECC_ENABLE ? dec_a[DATA_WIDTH+1] : 1'b0;
            assign rd_double_error_b[lane] = ECC_ENABLE ? dec_b[DATA_WIDTH+1] : 1'b0;
            assign rd_double_error_c[lane] = ECC_ENABLE ? dec_c[DATA_WIDTH+1] : 1'b0;
        end
    endgenerate

    // Aggregate ECC errors
    wire any_single_error = ECC_ENABLE ? (|rd_single_error_a | |rd_single_error_b | |rd_single_error_c) : 1'b0;
    wire any_double_error = ECC_ENABLE ? (|rd_double_error_a | |rd_double_error_b | |rd_double_error_c) : 1'b0;

    //------------------------------------------------------------------------
    // Operand Collector Read — generate + assign (iverilog compatible)
    //------------------------------------------------------------------------
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_oc_logic
            wire [PROTECTED_WIDTH-1:0] oc_raw_0 = sim_regs[oc_warp_id][lane][oc_addr[0]];
            wire [DATA_WIDTH+1:0] oc_dec_0 = decode_ecc(oc_raw_0[DATA_WIDTH-1:0], oc_raw_0[PROTECTED_WIDTH-1:DATA_WIDTH]);
            assign oc_data[0][lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? oc_dec_0[DATA_WIDTH-1:0] : oc_raw_0[DATA_WIDTH-1:0];

            wire [PROTECTED_WIDTH-1:0] oc_raw_1 = sim_regs[oc_warp_id][lane][oc_addr[1]];
            wire [DATA_WIDTH+1:0] oc_dec_1 = decode_ecc(oc_raw_1[DATA_WIDTH-1:0], oc_raw_1[PROTECTED_WIDTH-1:DATA_WIDTH]);
            assign oc_data[1][lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? oc_dec_1[DATA_WIDTH-1:0] : oc_raw_1[DATA_WIDTH-1:0];

            wire [PROTECTED_WIDTH-1:0] oc_raw_2 = sim_regs[oc_warp_id][lane][oc_addr[2]];
            wire [DATA_WIDTH+1:0] oc_dec_2 = decode_ecc(oc_raw_2[DATA_WIDTH-1:0], oc_raw_2[PROTECTED_WIDTH-1:DATA_WIDTH]);
            assign oc_data[2][lane*DATA_WIDTH +: DATA_WIDTH] = ECC_ENABLE ? oc_dec_2[DATA_WIDTH-1:0] : oc_raw_2[DATA_WIDTH-1:0];
        end
    endgenerate

    assign oc_ready = {NUM_READ_PORTS{oc_valid && !oc_conflict}};

    //------------------------------------------------------------------------
    // Write Logic (Sequential) with ECC Encoding
    //------------------------------------------------------------------------
    integer wr_w, wr_b, wr_l, wr_r;
    reg [BANK_BITS-1:0] wr_bank_idx;
    integer wr_lane_idx;
    reg [DATA_WIDTH-1:0] wr_data_lane;
    reg [ECC_BITS-1:0] wr_ecc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers (including ECC bits)
            for (wr_b = 0; wr_b < NUM_BANKS; wr_b = wr_b + 1) begin
                for (wr_w = 0; wr_w < NUM_WARPS; wr_w = wr_w + 1) begin
                    for (wr_l = 0; wr_l < LANES_PER_BANK; wr_l = wr_l + 1) begin
                        for (wr_r = 0; wr_r < NUM_REGS; wr_r = wr_r + 1) begin
                            sim_regs[wr_w][wr_b*LANES_PER_BANK+wr_l][wr_r] <= {PROTECTED_WIDTH{1'b0}};
                        end
                    end
                end
            end
        end else if (wr_en) begin
            for (wr_l = 0; wr_l < NUM_LANES; wr_l = wr_l + 1) begin
                if (wr_mask[wr_l]) begin
                    wr_bank_idx = get_bank(wr_addr, wr_l[LANE_W-1:0]);
                    wr_lane_idx = wr_l / NUM_BANKS;
                    wr_data_lane = wr_data[wr_l*DATA_WIDTH +: DATA_WIDTH];

                    if (ECC_ENABLE) begin
                        // Compute and store data with ECC
                        wr_ecc = calc_ecc(wr_data_lane);
                        sim_regs[wr_warp_id][wr_l][wr_addr] <=
                            {wr_ecc, wr_data_lane};
                    end else begin
                        sim_regs[wr_warp_id][wr_l][wr_addr] <=
                            {{(PROTECTED_WIDTH-DATA_WIDTH){1'b0}}, wr_data_lane};
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics and ECC Error Tracking
    //------------------------------------------------------------------------
    reg [31:0] conflict_count;
    reg [31:0] access_count;
    reg [31:0] ecc_correction_count;
    reg [31:0] ecc_uncorrectable_count;

    // Error location tracking
    reg [$clog2(NUM_WARPS)-1:0] error_warp_r;
    reg [4:0] error_reg_r;
    reg [$clog2(NUM_LANES)-1:0] error_lane_r;
    reg error_corrected_r;
    reg error_detected_r;

    // Find first lane with error for reporting
    integer err_lane;
    reg found_error;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conflict_count <= 0;
            access_count <= 0;
            ecc_correction_count <= 0;
            ecc_uncorrectable_count <= 0;
            error_warp_r <= 0;
            error_reg_r <= 0;
            error_lane_r <= 0;
            error_corrected_r <= 0;
            error_detected_r <= 0;
        end else begin
            error_corrected_r <= 1'b0;
            error_detected_r <= 1'b0;

            if (oc_valid) begin
                access_count <= access_count + 1;
                if (oc_conflict) begin
                    conflict_count <= conflict_count + 1;
                end

                // ECC error tracking
                if (ECC_ENABLE) begin
                    if (any_single_error) begin
                        ecc_correction_count <= ecc_correction_count + 1;
                        error_corrected_r <= 1'b1;

                        // Find which lane had the error
                        found_error = 0;
                        for (err_lane = 0; err_lane < NUM_LANES && !found_error; err_lane = err_lane + 1) begin
                            if (rd_single_error_a[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_a;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end else if (rd_single_error_b[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_b;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end else if (rd_single_error_c[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_c;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end
                        end
                    end

                    if (any_double_error) begin
                        ecc_uncorrectable_count <= ecc_uncorrectable_count + 1;
                        error_detected_r <= 1'b1;

                        // Find which lane had the error
                        found_error = 0;
                        for (err_lane = 0; err_lane < NUM_LANES && !found_error; err_lane = err_lane + 1) begin
                            if (rd_double_error_a[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_a;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end else if (rd_double_error_b[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_b;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end else if (rd_double_error_c[err_lane]) begin
                                error_warp_r <= rd_warp_id;
                                error_reg_r <= rd_addr_c;
                                error_lane_r <= err_lane[$clog2(NUM_LANES)-1:0];
                                found_error = 1;
                            end
                        end
                    end
                end
            end
        end
    end

    assign stat_bank_conflicts = conflict_count;
    assign stat_total_accesses = access_count;

    // ECC status outputs
    assign ecc_error_corrected = error_corrected_r;
    assign ecc_error_detected = error_detected_r;
    assign stat_ecc_corrections = ecc_correction_count;
    assign stat_ecc_uncorrectable = ecc_uncorrectable_count;
    assign ecc_error_warp = error_warp_r;
    assign ecc_error_reg = error_reg_r;
    assign ecc_error_lane = error_lane_r;

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
                alloc_ptr <= alloc_ptr + 1'b1;
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
                    issue_ptr <= issue_ptr + 1'b1;
                end
            end

            // Output completed entry
            if (has_complete && out_ready) begin
                entry_valid[complete_ptr] <= 1'b0;
                entry_complete[complete_ptr] <= 1'b0;
                entry_collected[complete_ptr] <= 0;
                complete_ptr <= complete_ptr + 1'b1;
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
