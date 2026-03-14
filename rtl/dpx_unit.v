//============================================================================
// RalphGPU - DPX Unit (Dynamic Programming Extensions)
// Implements Blackwell DPX instructions for dynamic programming acceleration
// Used in: Viterbi decoding, DTW, sequence alignment, Smith-Waterman
//
// Reference: NVIDIA Blackwell Architecture, PTX ISA 8.5+
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module dpx_unit #(
    parameter DATA_WIDTH = 32,
    parameter NUM_LANES = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   func,
    input  wire [DATA_WIDTH-1:0]        src_a,          // First source operand
    input  wire [DATA_WIDTH-1:0]        src_b,          // Second source operand
    input  wire [DATA_WIDTH-1:0]        src_c,          // Third source operand (for 3-input ops)

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg  [DATA_WIDTH-1:0]        result,
    output reg  [DATA_WIDTH-1:0]        result2         // Secondary result (for viaddminmax)
);

    //------------------------------------------------------------------------
    // Internal signals
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] add_result;
    reg [DATA_WIDTH-1:0] min_result;
    reg [DATA_WIDTH-1:0] max_result;
    reg [DATA_WIDTH-1:0] abs_a, abs_b;

    // Signed comparisons
    wire signed [DATA_WIDTH-1:0] signed_a = $signed(src_a);
    wire signed [DATA_WIDTH-1:0] signed_b = $signed(src_b);
    wire signed [DATA_WIDTH-1:0] signed_c = $signed(src_c);
    wire signed [DATA_WIDTH-1:0] signed_add = signed_a + signed_b;

    //------------------------------------------------------------------------
    // Combinational logic for operations
    //------------------------------------------------------------------------
    always @(*) begin
        // Addition
        add_result = src_a + src_b;

        // Absolute values
        abs_a = (signed_a < 0) ? -signed_a : signed_a;
        abs_b = (signed_b < 0) ? -signed_b : signed_b;

        // Min/max of sum vs third operand
        min_result = (signed_add < signed_c) ? signed_add : src_c;
        max_result = (signed_add > signed_c) ? signed_add : src_c;
    end

    //------------------------------------------------------------------------
    // Main operation logic
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            result <= 0;
            result2 <= 0;
        end else begin
            done <= 1'b0;

            if (valid_in) begin
                done <= 1'b1;

                case (func)
                    `DPX_VIADDMIN: begin
                        // viaddmin: result = min(a + b, c)
                        // Used in Viterbi algorithm for path metrics
                        result <= min_result;
                        `ifdef SIMULATION
                        $display("[DPX] VIADDMIN: min(%0d + %0d, %0d) = %0d",
                                 signed_a, signed_b, signed_c, $signed(min_result));
                        `endif
                    end

                    `DPX_VIADDMAX: begin
                        // viaddmax: result = max(a + b, c)
                        // Used in sequence alignment for scoring
                        result <= max_result;
                        `ifdef SIMULATION
                        $display("[DPX] VIADDMAX: max(%0d + %0d, %0d) = %0d",
                                 signed_a, signed_b, signed_c, $signed(max_result));
                        `endif
                    end

                    `DPX_VIMINABS: begin
                        // viminabs: result = min(|a|, |b|)
                        result <= (abs_a < abs_b) ? abs_a : abs_b;
                        `ifdef SIMULATION
                        $display("[DPX] VIMINABS: min(|%0d|, |%0d|) = %0d",
                                 signed_a, signed_b, (abs_a < abs_b) ? abs_a : abs_b);
                        `endif
                    end

                    `DPX_VIMAXABS: begin
                        // vimaxabs: result = max(|a|, |b|)
                        result <= (abs_a > abs_b) ? abs_a : abs_b;
                        `ifdef SIMULATION
                        $display("[DPX] VIMAXABS: max(|%0d|, |%0d|) = %0d",
                                 signed_a, signed_b, (abs_a > abs_b) ? abs_a : abs_b);
                        `endif
                    end

                    `DPX_VIADDMINMAX: begin
                        // viaddminmax: result = min(a+b, c), result2 = max(a+b, c)
                        // Both min and max in single operation for bidirectional DP
                        result <= min_result;
                        result2 <= max_result;
                        `ifdef SIMULATION
                        $display("[DPX] VIADDMINMAX: min=%0d, max=%0d",
                                 $signed(min_result), $signed(max_result));
                        `endif
                    end

                    `DPX_VIBMATCH: begin
                        // vibmatch: bit pattern matching
                        // Returns mask of matching bits
                        result <= ~(src_a ^ src_b);  // XNOR for matching bits
                        `ifdef SIMULATION
                        $display("[DPX] VIBMATCH: 0x%08x XNOR 0x%08x = 0x%08x",
                                 src_a, src_b, ~(src_a ^ src_b));
                        `endif
                    end

                    `DPX_VIBSET: begin
                        // vibset: bit set operations
                        // result = (a & ~c) | (b & c)  -- select bits from a or b based on c
                        result <= (src_a & ~src_c) | (src_b & src_c);
                        `ifdef SIMULATION
                        $display("[DPX] VIBSET: select from 0x%08x/0x%08x by 0x%08x = 0x%08x",
                                 src_a, src_b, src_c, (src_a & ~src_c) | (src_b & src_c));
                        `endif
                    end

                    `DPX_RELU: begin
                        // ReLU: max(0, x)
                        result <= (signed_a < 0) ? 32'b0 : src_a;
                        `ifdef SIMULATION
                        $display("[DPX] RELU: max(0, %0d) = %0d",
                                 signed_a, (signed_a < 0) ? 0 : signed_a);
                        `endif
                    end

                    `DPX_TANH: begin
                        // Fast tanh approximation using piecewise linear
                        // tanh(x) ≈ x for |x| < 1, ±1 for |x| > 3, linear in between
                        if (signed_a >= 32'sd3) begin
                            result <= 32'd1;  // Saturate to 1
                        end else if (signed_a <= -32'sd3) begin
                            result <= 32'hFFFFFFFF;  // Saturate to -1
                        end else if (abs_a <= 32'd1) begin
                            result <= src_a;  // Linear region
                        end else begin
                            // Piecewise linear approximation
                            result <= (signed_a > 0) ?
                                      (32'd1 + ((signed_a - 32'sd1) >>> 1)) :
                                      (32'hFFFFFFFF + ((signed_a + 32'sd1) >>> 1));
                        end
                        `ifdef SIMULATION
                        $display("[DPX] TANH: tanh(%0d) approximation", signed_a);
                        `endif
                    end

                    `DPX_EXP2: begin
                        // Fast exp2 approximation for small integer inputs
                        // exp2(x) = 1 << x for positive integers
                        if (signed_a >= 32'sd31) begin
                            result <= 32'h80000000;  // Max power
                        end else if (signed_a < 0) begin
                            result <= 32'd0;  // Fractional result rounds to 0
                        end else begin
                            result <= (32'd1 << src_a[4:0]);
                        end
                        `ifdef SIMULATION
                        $display("[DPX] EXP2: 2^%0d = %0d", signed_a, (32'd1 << src_a[4:0]));
                        `endif
                    end

                    default: begin
                        result <= src_a;  // Pass through
                        done <= 1'b1;
                    end
                endcase
            end
        end
    end

    /* verilator lint_on BLKSEQ */
endmodule

//============================================================================
// Sparse MMA Unit - 2:4 Structured Sparsity Support
// Implements sparse matrix multiplication with 2:4 pattern
// (2 non-zero values per 4 elements)
//============================================================================
module sparse_mma_unit #(
    parameter DATA_WIDTH = 16,
    parameter TILE_M = 16,
    parameter TILE_N = 8,
    parameter TILE_K = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   func,

    //------------------------------------------------------------------------
    // Sparse Matrix A input (compressed 2:4 format)
    // For each 4-value group:
    // - sparse_a_data stores 2 values (2 * DATA_WIDTH bits)
    // - sparse_a_indices stores two 2-bit positions (4 bits total)
    //------------------------------------------------------------------------
    input  wire [TILE_M*TILE_K*DATA_WIDTH/2-1:0]    sparse_a_data,
    input  wire [TILE_M*TILE_K-1:0]                  sparse_a_indices,

    //------------------------------------------------------------------------
    // Dense Matrix B input
    //------------------------------------------------------------------------
    input  wire [TILE_K*TILE_N*DATA_WIDTH-1:0]      dense_b,

    //------------------------------------------------------------------------
    // Accumulator input/output
    //------------------------------------------------------------------------
    input  wire [TILE_M*TILE_N*32-1:0]              accum_in,
    output reg  [TILE_M*TILE_N*32-1:0]              accum_out,

    //------------------------------------------------------------------------
    // Control
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          busy
);

    localparam integer NUM_A_ELEMS   = TILE_M * TILE_K;
    localparam integer NUM_C_ELEMS   = TILE_M * TILE_N;
    localparam integer NUM_GROUPS    = NUM_A_ELEMS / 4;
    localparam integer SPARSE_DATA_W = NUM_A_ELEMS * DATA_WIDTH / 2;
    localparam integer DENSE_A_W     = NUM_A_ELEMS * DATA_WIDTH;
    localparam integer ACCUM_W       = NUM_C_ELEMS * 32;

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE      = 2'd0;
    localparam ST_DECOMP    = 2'd1;
    localparam ST_COMPUTE   = 2'd2;
    localparam ST_DONE      = 2'd3;

    reg [1:0] state;
    reg [5:0] saved_func;

    // Decompressed dense matrix A
    reg [DENSE_A_W-1:0] dense_a;

    // Temporary buffers used inside sequential states
    reg [DENSE_A_W-1:0] dense_a_next;
    reg [ACCUM_W-1:0] mma_result_next;

    // Loop / temp variables
    integer grp_i;
    integer elem_i;
    integer row_i;
    integer col_i;
    integer k_i;

    reg [1:0] idx0;
    reg [1:0] idx1;
    reg signed [DATA_WIDTH-1:0] val0;
    reg signed [DATA_WIDTH-1:0] val1;

    reg signed [DATA_WIDTH-1:0] dense_elem;
    reg [DATA_WIDTH:0] dense_abs;
    reg [DATA_WIDTH:0] top0_abs;
    reg [DATA_WIDTH:0] top1_abs;
    reg signed [DATA_WIDTH-1:0] top0_val;
    reg signed [DATA_WIDTH-1:0] top1_val;
    reg [1:0] top0_idx;
    reg [1:0] top1_idx;
    reg have_top0;
    reg have_top1;

    reg signed [DATA_WIDTH-1:0] a_elem;
    reg signed [DATA_WIDTH-1:0] b_elem;
    reg signed [63:0] mac_sum;

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            accum_out <= {ACCUM_W{1'b0}};
            dense_a <= {DENSE_A_W{1'b0}};
            saved_func <= 6'b0;
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (valid_in) begin
                        saved_func <= func;
                        busy <= 1'b1;

                        case (func)
                            `SPARSE_COMPRESS: begin
                                // Input dense matrix A is provided in accum_in[0 +: DENSE_A_W]
                                // Output packing in accum_out:
                                //   [0 +: SPARSE_DATA_W]              = compressed values
                                //   [SPARSE_DATA_W +: NUM_A_ELEMS]    = 4-bit index per group
                                accum_out <= {ACCUM_W{1'b0}};

                                for (grp_i = 0; grp_i < NUM_GROUPS; grp_i = grp_i + 1) begin
                                    top0_abs = {DATA_WIDTH+1{1'b0}};
                                    top1_abs = {DATA_WIDTH+1{1'b0}};
                                    top0_val = {DATA_WIDTH{1'b0}};
                                    top1_val = {DATA_WIDTH{1'b0}};
                                    top0_idx = 2'b00;
                                    top1_idx = 2'b10;
                                    have_top0 = 1'b0;
                                    have_top1 = 1'b0;

                                    for (elem_i = 0; elem_i < 4; elem_i = elem_i + 1) begin
                                        dense_elem = $signed(accum_in[((grp_i * 4 + elem_i) * DATA_WIDTH) +: DATA_WIDTH]);
                                        dense_abs = dense_elem[DATA_WIDTH-1] ? ({1'b0, ~dense_elem} + 1'b1) : {1'b0, dense_elem};

                                        if (dense_elem != {DATA_WIDTH{1'b0}}) begin
                                            if (!have_top0 || (dense_abs > top0_abs)) begin
                                                top1_abs = top0_abs;
                                                top1_val = top0_val;
                                                top1_idx = top0_idx;
                                                have_top1 = have_top0;

                                                top0_abs = dense_abs;
                                                top0_val = dense_elem;
                                                top0_idx = elem_i[1:0];
                                                have_top0 = 1'b1;
                                            end else if (!have_top1 || (dense_abs > top1_abs)) begin
                                                top1_abs = dense_abs;
                                                top1_val = dense_elem;
                                                top1_idx = elem_i[1:0];
                                                have_top1 = 1'b1;
                                            end
                                        end
                                    end

                                    accum_out[(grp_i * 2 * DATA_WIDTH) +: DATA_WIDTH] <= top0_val;
                                    accum_out[(grp_i * 2 * DATA_WIDTH) + DATA_WIDTH +: DATA_WIDTH] <= top1_val;
                                    accum_out[SPARSE_DATA_W + (grp_i * 4) +: 2] <= top0_idx;
                                    accum_out[SPARSE_DATA_W + (grp_i * 4) + 2 +: 2] <= top1_idx;
                                end

                                done <= 1'b1;
                                `ifdef SIMULATION
                                $display("[SPARSE] Compress operation complete");
                                `endif
                            end

                            `SPARSE_DECOMPRESS: begin
                                state <= ST_DECOMP;
                                `ifdef SIMULATION
                                $display("[SPARSE] Decompress operation");
                                `endif
                            end

                            default: begin
                                // Sparse MMA operations (FP16/BF16/TF32/INT8/FP8)
                                state <= ST_DECOMP;
                                `ifdef SIMULATION
                                $display("[SPARSE] MMA operation func=%0d", func);
                                `endif
                            end
                        endcase
                    end
                end

                ST_DECOMP: begin
                    dense_a_next = {DENSE_A_W{1'b0}};

                    for (grp_i = 0; grp_i < NUM_GROUPS; grp_i = grp_i + 1) begin
                        idx0 = sparse_a_indices[grp_i * 4 +: 2];
                        idx1 = sparse_a_indices[grp_i * 4 + 2 +: 2];
                        val0 = $signed(sparse_a_data[grp_i * 2 * DATA_WIDTH +: DATA_WIDTH]);
                        val1 = $signed(sparse_a_data[grp_i * 2 * DATA_WIDTH + DATA_WIDTH +: DATA_WIDTH]);

                        dense_a_next[((grp_i * 4 + idx0) * DATA_WIDTH) +: DATA_WIDTH] = val0;
                        if (idx1 != idx0) begin
                            dense_a_next[((grp_i * 4 + idx1) * DATA_WIDTH) +: DATA_WIDTH] = val1;
                        end
                    end

                    dense_a <= dense_a_next;

                    if (saved_func == `SPARSE_DECOMPRESS) begin
                        accum_out <= {ACCUM_W{1'b0}};
                        for (elem_i = 0; elem_i < NUM_A_ELEMS; elem_i = elem_i + 1) begin
                            accum_out[(elem_i * DATA_WIDTH) +: DATA_WIDTH] <= dense_a_next[(elem_i * DATA_WIDTH) +: DATA_WIDTH];
                        end
                        state <= ST_DONE;
                    end else begin
                        state <= ST_COMPUTE;
                    end
                end

                ST_COMPUTE: begin
                    // Sparse MMA: C = accum_in + A(decompressed) * B
                    mma_result_next = accum_in;

                    for (row_i = 0; row_i < TILE_M; row_i = row_i + 1) begin
                        for (col_i = 0; col_i < TILE_N; col_i = col_i + 1) begin
                            mac_sum = $signed(accum_in[((row_i * TILE_N + col_i) * 32) +: 32]);

                            for (k_i = 0; k_i < TILE_K; k_i = k_i + 1) begin
                                a_elem = $signed(dense_a[((row_i * TILE_K + k_i) * DATA_WIDTH) +: DATA_WIDTH]);
                                b_elem = $signed(dense_b[((k_i * TILE_N + col_i) * DATA_WIDTH) +: DATA_WIDTH]);
                                mac_sum = mac_sum + (a_elem * b_elem);
                            end

                            mma_result_next[((row_i * TILE_N + col_i) * 32) +: 32] = mac_sum[31:0];
                        end
                    end

                    accum_out <= mma_result_next;
                    state <= ST_DONE;
                    `ifdef SIMULATION
                    $display("[SPARSE] MMA compute complete");
                    `endif
                end

                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
    /* verilator lint_on BLKSEQ */

endmodule
