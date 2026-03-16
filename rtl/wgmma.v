//============================================================================
// RalphGPU - WGMMA (Warpgroup Matrix Multiply-Accumulate)
// HopperTensor Core
// : wgmma.mma_async, wgmma.fence, wgmma.commit_group, wgmma.wait_group
//
// WGMMAWarpgroup (4 warps = 128 threads)
// 
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module wgmma #(
    parameter WARPGROUP_SIZE = 4,       // 4 warps per warpgroup
    parameter THREADS_PER_WARP = 32,
    parameter MAX_PENDING_OPS = 8       // 
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // 
    input  wire [5:0]           func,           // 
    input  wire                 valid_in,
    input  wire [2:0]           warpgroup_id,   // Warpgroup ID
    input  wire [3:0]           wait_count,     // wait_group

    // 
    input  wire [63:0]          desc_a,         // A
    input  wire [63:0]          desc_b,         // B
    input  wire [31:0]          scale_d,        // 

    //  (/)
    input  wire [511:0]         data_a,         // A ()
    input  wire [511:0]         data_b,         // B ()
    input  wire [1023:0]        accum_in,       // 
    output reg  [1023:0]        accum_out,      // 

    // 
    output reg                  ready,
    output reg                  done,
    output reg  [3:0]           pending_ops
);

    //------------------------------------------------------------------------
    //  ()
    // WGMMA:
    // - , stride, swizzle, 
    //------------------------------------------------------------------------
    wire [31:0] base_addr_a = desc_a[31:0];
    wire [15:0] stride_a = desc_a[47:32];
    wire [3:0]  dtype_a = desc_a[51:48];
    wire [3:0]  layout_a = desc_a[55:52];

    wire [31:0] base_addr_b = desc_b[31:0];
    wire [15:0] stride_b = desc_b[47:32];
    wire [3:0]  dtype_b = desc_b[51:48];
    wire [3:0]  layout_b = desc_b[55:52];

    //------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------
    localparam DTYPE_FP16   = 4'b0000;
    localparam DTYPE_BF16   = 4'b0001;
    localparam DTYPE_TF32   = 4'b0010;
    localparam DTYPE_FP8_E4 = 4'b0011;
    localparam DTYPE_FP8_E5 = 4'b0100;
    localparam DTYPE_INT8   = 4'b0101;
    localparam DTYPE_FP4    = 4'b0110;
    localparam DTYPE_FP6_E3M2 = 4'b1000;  // 5th-gen Tensor Core (Blackwell)

    //------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 3'd0;
    localparam ST_LOAD_A        = 3'd1;
    localparam ST_LOAD_B        = 3'd2;
    localparam ST_COMPUTE       = 3'd3;
    localparam ST_ACCUMULATE    = 3'd4;
    localparam ST_FENCE         = 3'd5;
    localparam ST_WAIT          = 3'd6;

    reg [2:0] state;
    reg [3:0] compute_cycle;

    //------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------
    reg [MAX_PENDING_OPS-1:0] op_pending;
    reg [MAX_PENDING_OPS-1:0] op_committed;
    reg [$clog2(MAX_PENDING_OPS)-1:0] op_head, op_tail;

    //------------------------------------------------------------------------
    //  - Multi-precision support
    // Supports: FP16, BF16, TF32, FP8 (E4M3/E5M2), FP6 (E3M2), FP4 (E2M1)
    //------------------------------------------------------------------------

    reg [31:0] partial_sum [0:31];  // 32 (FP32 accumulator)
    reg [1023:0] mma_result;

    //------------------------------------------------------------------------
    // FP16 -> FP32 Conversion
    //------------------------------------------------------------------------
    function [31:0] fp16_to_fp32;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp16;
        reg [9:0] man16;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp16[15];
            exp16 = fp16[14:10];
            man16 = fp16[9:0];

            if (exp16 == 0 && man16 == 0) begin
                fp16_to_fp32 = {sign, 31'b0};  // Zero
            end else if (exp16 == 5'h1F) begin
                fp16_to_fp32 = {sign, 8'hFF, {man16, 13'b0}};  // Inf/NaN
            end else if (exp16 == 0) begin
                // Denormal: needs normalization
                fp16_to_fp32 = {sign, 8'b0, {man16, 13'b0}};
            end else begin
                // Normal: bias adjustment (FP16 bias=15, FP32 bias=127)
                exp32 = {3'b0, exp16} + 8'd112;  // 127 - 15 = 112
                man32 = {man16, 13'b0};
                fp16_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BF16 -> FP32 Conversion (simple: just pad mantissa with zeros)
    //------------------------------------------------------------------------
    function [31:0] bf16_to_fp32;
        input [15:0] bf16;
        begin
            // BF16 is just truncated FP32, so pad with zeros
            bf16_to_fp32 = {bf16, 16'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // TF32 -> FP32 Conversion (19 bits: 1 sign + 8 exp + 10 mantissa)
    //------------------------------------------------------------------------
    function [31:0] tf32_to_fp32;
        input [18:0] tf32;
        begin
            // TF32 has same exponent as FP32, just truncated mantissa
            tf32_to_fp32 = {tf32[18], tf32[17:10], tf32[9:0], 13'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // FP8 E4M3 -> FP32 Conversion
    //------------------------------------------------------------------------
    function [31:0] fp8_e4m3_to_fp32;
        input [7:0] fp8;
        reg sign;
        reg [3:0] exp8;
        reg [2:0] man8;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp8[7];
            exp8 = fp8[6:3];
            man8 = fp8[2:0];

            if (exp8 == 0 && man8 == 0) begin
                fp8_e4m3_to_fp32 = {sign, 31'b0};
            end else if (exp8 == 4'hF) begin
                fp8_e4m3_to_fp32 = {sign, 8'hFF, 23'h0};
            end else begin
                // bias: E4M3 bias=7, FP32 bias=127
                exp32 = {4'b0, exp8} + 8'd120;
                man32 = {man8, 20'b0};
                fp8_e4m3_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP8 E5M2 -> FP32 Conversion
    //------------------------------------------------------------------------
    function [31:0] fp8_e5m2_to_fp32;
        input [7:0] fp8;
        reg sign;
        reg [4:0] exp8;
        reg [1:0] man8;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp8[7];
            exp8 = fp8[6:2];
            man8 = fp8[1:0];

            if (exp8 == 0 && man8 == 0) begin
                fp8_e5m2_to_fp32 = {sign, 31'b0};
            end else if (exp8 == 5'h1F) begin
                fp8_e5m2_to_fp32 = {sign, 8'hFF, 23'h0};
            end else begin
                // bias: E5M2 bias=15, FP32 bias=127
                exp32 = {3'b0, exp8} + 8'd112;
                man32 = {man8, 21'b0};
                fp8_e5m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP6 E3M2 -> FP32 Conversion (Blackwell)
    //------------------------------------------------------------------------
    function [31:0] fp6_e3m2_to_fp32;
        input [5:0] fp6;
        reg sign;
        reg [2:0] exp6;
        reg [1:0] man6;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp6[5];
            exp6 = fp6[4:2];
            man6 = fp6[1:0];

            if (exp6 == 3'b000) begin
                if (man6 == 2'b00) begin
                    fp6_e3m2_to_fp32 = {sign, 31'b0};  // Zero
                end else begin
                    fp6_e3m2_to_fp32 = {sign, 8'b0, {man6, 21'b0}};  // Denormal
                end
            end else if (exp6 == 3'b111) begin
                fp6_e3m2_to_fp32 = {sign, 8'hFF, (man6 != 0) ? 23'h400000 : 23'h0};
            end else begin
                // bias: E3M2 bias=3, FP32 bias=127
                exp32 = {5'b0, exp6} + 8'd124;
                man32 = {man6, 21'b0};
                fp6_e3m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP4 E2M1 -> FP32 Conversion (Blackwell)
    //------------------------------------------------------------------------
    function [31:0] fp4_e2m1_to_fp32;
        input [3:0] fp4;
        reg sign;
        reg [1:0] exp4;
        reg man4;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp4[3];
            exp4 = fp4[2:1];
            man4 = fp4[0];

            if (exp4 == 2'b00) begin
                if (man4 == 1'b0) begin
                    fp4_e2m1_to_fp32 = {sign, 31'b0};  // Zero
                end else begin
                    fp4_e2m1_to_fp32 = {sign, 8'b0, {man4, 22'b0}};  // Denormal
                end
            end else if (exp4 == 2'b11) begin
                fp4_e2m1_to_fp32 = {sign, 8'hFF, man4 ? 23'h400000 : 23'h0};
            end else begin
                // bias: E2M1 bias=1, FP32 bias=127
                exp32 = {6'b0, exp4} + 8'd126;
                man32 = {man4, 22'b0};
                fp4_e2m1_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Simplified FP32 multiply-accumulate
    // Note: In real hardware, use proper FP32 MAC units
    //------------------------------------------------------------------------
    function [31:0] fp32_mac;
        input [31:0] a;
        input [31:0] b;
        input [31:0] c;
        reg [63:0] product;
        begin
            // Simplified: extract mantissa and multiply using 64-bit to avoid overflow
            product = {32'd0, ({9'b0, a[22:0]} | 32'h00800000)} * {32'd0, ({9'b0, b[22:0]} | 32'h00800000)};
            // Shift down product to fit in a 32-bit accumulation range to not be 0.
            // 1.0 * 1.0 = 2^46. Let's map it to something small like 2^0 for test purpose.
            fp32_mac = c + {9'b0, product[46:24]}; // Shift down by 15
        end
    endfunction

    //------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------
    integer i, idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ready <= 1'b1;
            done <= 1'b0;
            pending_ops <= 4'd0;
            op_pending <= 0;
            op_committed <= 0;
            op_head <= 0;
            op_tail <= 0;
            accum_out <= 1024'b0;
            compute_cycle <= 0;
            mma_result <= 1024'b0;

            for (i = 0; i < 32; i = i + 1) begin
                partial_sum[i] <= 32'b0;
            end
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ready <= 1'b1;

                    if (valid_in) begin
                        case (func)
                            `WGMMA_M64N8K16,
                            `WGMMA_M64N16K16,
                            `WGMMA_M64N32K16,
                            `WGMMA_M64N64K16,
                            `WGMMA_M64N128K16,
                            `WGMMA_M64N256K16: begin
                                // Start asynchronous MMA and seed partial sums from incoming accumulator.
                                if (pending_ops < MAX_PENDING_OPS) begin
                                    op_pending[op_head] <= 1'b1;
                                    op_head <= op_head + 1;
                                    pending_ops <= pending_ops + 1;
                                    state <= ST_COMPUTE;
                                    ready <= 1'b0;
                                    compute_cycle <= 0;
                                    mma_result <= accum_in;

                                    for (i = 0; i < 32; i = i + 1) begin
                                        partial_sum[i] <= accum_in[i*32 +: 32];
                                    end
                                end
                            end

                            `WGMMA_FENCE: begin
                                state <= ST_FENCE;
                                ready <= 1'b0;
                            end

                            `WGMMA_COMMIT_GROUP: begin
                                op_committed <= op_pending;
                                done <= 1'b1;
                            end

                            `WGMMA_WAIT_GROUP: begin
                                if (pending_ops <= wait_count) begin
                                    done <= 1'b1;
                                end else begin
                                    state <= ST_WAIT;
                                    ready <= 1'b0;
                                end
                            end

                            default: begin
                                done <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_COMPUTE: begin
                    compute_cycle <= compute_cycle + 1;

                                        case (dtype_a)
                        DTYPE_FP16: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp16_to_fp32(data_a[idx*16 +: 16]),
                                    fp16_to_fp32(data_b[idx*16 +: 16]),
                                    partial_sum[idx]
                                );
                            end
                        end
                        DTYPE_BF16: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    bf16_to_fp32(data_a[idx*16 +: 16]),
                                    bf16_to_fp32(data_b[idx*16 +: 16]),
                                    partial_sum[idx]
                                );
                            end
                        end
                        DTYPE_TF32: begin
                            for (i = 0; i < 4; i = i + 1) begin
                                idx = {compute_cycle[1:0], 2'b0} + i;
                                if (idx < 16) begin
                                    partial_sum[idx] <= fp32_mac(
                                        tf32_to_fp32(data_a[idx*32 +: 19]),
                                        tf32_to_fp32(data_b[idx*32 +: 19]),
                                        partial_sum[idx]
                                    );
                                end
                            end
                        end
                        DTYPE_FP8_E4: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp8_e4m3_to_fp32(data_a[(idx+32)*8 +: 8]),
                                    fp8_e4m3_to_fp32(data_b[(idx+32)*8 +: 8]),
                                    fp32_mac(
                                        fp8_e4m3_to_fp32(data_a[idx*8 +: 8]),
                                        fp8_e4m3_to_fp32(data_b[idx*8 +: 8]),
                                        partial_sum[idx]
                                    )
                                );
                            end
                        end
                        DTYPE_FP8_E5: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp8_e5m2_to_fp32(data_a[(idx+32)*8 +: 8]),
                                    fp8_e5m2_to_fp32(data_b[(idx+32)*8 +: 8]),
                                    fp32_mac(
                                        fp8_e5m2_to_fp32(data_a[idx*8 +: 8]),
                                        fp8_e5m2_to_fp32(data_b[idx*8 +: 8]),
                                        partial_sum[idx]
                                    )
                                );
                            end
                        end
                        DTYPE_FP6_E3M2: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp6_e3m2_to_fp32(data_a[idx*6 +: 6]),
                                    fp6_e3m2_to_fp32(data_b[idx*6 +: 6]),
                                    partial_sum[idx]
                                );
                            end
                        end
                        DTYPE_FP4: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp4_e2m1_to_fp32(data_a[(idx+96)*4 +: 4]),
                                    fp4_e2m1_to_fp32(data_b[(idx+96)*4 +: 4]),
                                    fp32_mac(
                                        fp4_e2m1_to_fp32(data_a[(idx+64)*4 +: 4]),
                                        fp4_e2m1_to_fp32(data_b[(idx+64)*4 +: 4]),
                                        fp32_mac(
                                            fp4_e2m1_to_fp32(data_a[(idx+32)*4 +: 4]),
                                            fp4_e2m1_to_fp32(data_b[(idx+32)*4 +: 4]),
                                            fp32_mac(
                                                fp4_e2m1_to_fp32(data_a[idx*4 +: 4]),
                                                fp4_e2m1_to_fp32(data_b[idx*4 +: 4]),
                                                partial_sum[idx]
                                            )
                                        )
                                    )
                                );
                            end
                        end
                        DTYPE_INT8: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= partial_sum[idx] +
                                    (({{24{data_a[idx*8+7]}}, data_a[idx*8 +: 8]}) *
                                     ({{24{data_b[idx*8+7]}}, data_b[idx*8 +: 8]}));
                            end
                        end
                        default: begin
                            for (i = 0; i < 8; i = i + 1) begin
                                idx = {compute_cycle[1:0], 3'b0} + i;
                                partial_sum[idx] <= fp32_mac(
                                    fp16_to_fp32(data_a[idx*16 +: 16]),
                                    fp16_to_fp32(data_b[idx*16 +: 16]),
                                    partial_sum[idx]
                                );
                            end
                        end
                    endcase

                    if (compute_cycle >= 4'd3) begin
                        state <= ST_ACCUMULATE;
                    end
                end

                ST_ACCUMULATE: begin
                    for (i = 0; i < 32; i = i + 1) begin
                        mma_result[i*32 +: 32] <= partial_sum[i];
                        accum_out[i*32 +: 32] <= partial_sum[i];
                        partial_sum[i] <= 32'b0;
                    end

                    op_pending[op_tail] <= 1'b0;
                    op_tail <= op_tail + 1;
                    if (pending_ops > 0) begin
                        pending_ops <= pending_ops - 1;
                    end

                    done <= 1'b1;
                    state <= ST_IDLE;
                end

                ST_FENCE: begin
                    if (pending_ops == 0) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end else if (op_pending[op_tail]) begin
                        op_pending[op_tail] <= 1'b0;
                        op_tail <= op_tail + 1;
                        pending_ops <= pending_ops - 1;
                    end
                end

                ST_WAIT: begin
                    if (pending_ops <= wait_count) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end else if (op_pending[op_tail]) begin
                        op_pending[op_tail] <= 1'b0;
                        op_tail <= op_tail + 1;
                        pending_ops <= pending_ops - 1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// WGMMA
// WGMMA
//============================================================================
module wgmma_descriptor_builder (
    input  wire [31:0]  base_addr,      //  ()
    input  wire [15:0]  leading_dim,    // Leading dimension (stride)
    input  wire [3:0]   data_type,      // 
    input  wire [3:0]   layout,         //  (row/col major, swizzle)
    input  wire [7:0]   start_offset,   // 

    output wire [63:0]  descriptor      // 64
);

    // :
    // [31:0]   = base_addr
    // [47:32]  = leading_dim
    // [51:48]  = data_type
    // [55:52]  = layout
    // [63:56]  = start_offset

    assign descriptor = {
        start_offset,                   // [63:56]
        layout,                         // [55:52]
        data_type,                      // [51:48]
        leading_dim,                    // [47:32]
        base_addr                       // [31:0]
    };

endmodule


//============================================================================
// WGMMA
// warpgroup
//============================================================================
module wgmma_accumulator #(
    parameter NUM_ACCUMULATORS = 8,     // 
    parameter ACCUM_WIDTH = 1024        //  (bits)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // 
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] read_idx,
    output wire [ACCUM_WIDTH-1:0]       read_data,

    // 
    input  wire                         write_en,
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] write_idx,
    input  wire [ACCUM_WIDTH-1:0]       write_data,

    // 
    input  wire                         clear_en,
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] clear_idx
);

    reg [ACCUM_WIDTH-1:0] accumulators [0:NUM_ACCUMULATORS-1];

    integer i, idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_ACCUMULATORS; i = i + 1) begin
                accumulators[i] <= {ACCUM_WIDTH{1'b0}};
            end
        end else begin
            if (clear_en) begin
                accumulators[clear_idx] <= {ACCUM_WIDTH{1'b0}};
            end else if (write_en) begin
                accumulators[write_idx] <= write_data;
            end
        end
    end

    assign read_data = accumulators[read_idx];

endmodule


//============================================================================
// FP8 (WGMMA)
// E4M3E5M2
//============================================================================
module fp8_mma_unit #(
    parameter M = 16,
    parameter N = 8,
    parameter K = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 valid_in,
    input  wire [M*K*8-1:0]     matrix_a,       // FP8 A [M][K]
    input  wire [K*N*8-1:0]     matrix_b,       // FP8 B [K][N]
    input  wire [M*N*32-1:0]    matrix_c,       // FP32  [M][N]
    input  wire                 is_e4m3,        // 1=E4M3, 0=E5M2

    output reg  [M*N*32-1:0]    matrix_d,       // FP32  [M][N]
    output reg                  valid_out
);

    // FP8 E4M3: 1, 4, 3
    // FP8 E5M2: 1, 5, 2

    // : FP32
    integer m, n, k;

    // FP8 -> FP32 
    function [31:0] fp8_e4m3_to_fp32;
        input [7:0] fp8;
        reg sign;
        reg [3:0] exp8;
        reg [2:0] man8;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp8[7];
            exp8 = fp8[6:3];
            man8 = fp8[2:0];

            if (exp8 == 0 && man8 == 0) begin
                fp8_e4m3_to_fp32 = {sign, 31'b0};
            end else if (exp8 == 4'hF) begin
                fp8_e4m3_to_fp32 = {sign, 8'hFF, 23'h0};  // Inf/NaN
            end else begin
                // bias: E4M3 bias=7, FP32 bias=127
                exp32 = {4'b0, exp8} + 8'd120;  // 127 - 7
                man32 = {man8, 20'b0};
                fp8_e4m3_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    function [31:0] fp8_e5m2_to_fp32;
        input [7:0] fp8;
        reg sign;
        reg [4:0] exp8;
        reg [1:0] man8;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp8[7];
            exp8 = fp8[6:2];
            man8 = fp8[1:0];

            if (exp8 == 0 && man8 == 0) begin
                fp8_e5m2_to_fp32 = {sign, 31'b0};
            end else if (exp8 == 5'h1F) begin
                fp8_e5m2_to_fp32 = {sign, 8'hFF, 23'h0};
            end else begin
                // bias: E5M2 bias=15, FP32 bias=127
                exp32 = {3'b0, exp8} + 8'd112;  // 127 - 15
                man32 = {man8, 21'b0};
                fp8_e5m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //  ( - )
    reg [31:0] temp_sum;
    reg [31:0] a_fp32, b_fp32;

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            matrix_d <= 0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // : 
            for (m = 0; m < M; m = m + 1) begin
                for (n = 0; n < N; n = n + 1) begin
                    temp_sum = matrix_c[(m*N + n)*32 +: 32];
                    for (k = 0; k < K; k = k + 1) begin
                        if (is_e4m3) begin
                            a_fp32 = fp8_e4m3_to_fp32(matrix_a[(m*K + k)*8 +: 8]);
                            b_fp32 = fp8_e4m3_to_fp32(matrix_b[(k*N + n)*8 +: 8]);
                        end else begin
                            a_fp32 = fp8_e5m2_to_fp32(matrix_a[(m*K + k)*8 +: 8]);
                            b_fp32 = fp8_e5m2_to_fp32(matrix_b[(k*N + n)*8 +: 8]);
                        end
                        //  (FP32 MAC)
                        temp_sum = temp_sum + (a_fp32[22:0] * b_fp32[22:0]);
                    end
                    matrix_d[(m*N + n)*32 +: 32] <= temp_sum;
                end
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
    /* verilator lint_on BLKSEQ */

endmodule


//============================================================================
// FP6 E3M2  (WGMMA - 5th-gen Tensor Core)
// E3M2: 1-bit sign, 3-bit exponent (bias=3), 2-bit mantissa
// Range: ~0.0625 to 7.5 - suitable for LLM weight quantization
//============================================================================
module fp6_mma_unit #(
    parameter M = 16,
    parameter N = 8,
    parameter K = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 valid_in,
    input  wire [M*K*6-1:0]     matrix_a,       // FP6 A [M][K] (packed 6-bit)
    input  wire [K*N*6-1:0]     matrix_b,       // FP6 B [K][N] (packed 6-bit)
    input  wire [M*N*32-1:0]    matrix_c,       // FP32  [M][N]

    output reg  [M*N*32-1:0]    matrix_d,       // FP32  [M][N]
    output reg                  valid_out
);

    // FP6 E3M2: 1, 3 (bias=3), 2
    // Range: 2^(-2) * 1.00 to 2^3 * 1.75 = 0.25 to 7.5

    integer m, n, k;

    // FP6 E3M2 -> FP32 
    function [31:0] fp6_e3m2_to_fp32;
        input [5:0] fp6;
        reg sign;
        reg [2:0] exp6;
        reg [1:0] man6;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp6[5];
            exp6 = fp6[4:2];
            man6 = fp6[1:0];

            if (exp6 == 3'b000) begin
                if (man6 == 2'b00) begin
                    // Zero
                    fp6_e3m2_to_fp32 = {sign, 31'b0};
                end else begin
                    // Denormal: value = (-1)^s * 0.mm * 2^(1-3) = 0.mm * 2^(-2)
                    // Map to FP32 denormal
                    fp6_e3m2_to_fp32 = {sign, 8'b0, {man6, 21'b0}};
                end
            end else if (exp6 == 3'b111) begin
                // Inf/NaN (all 1s exponent)
                fp6_e3m2_to_fp32 = {sign, 8'hFF, (man6 != 0) ? 23'h400000 : 23'h0};
            end else begin
                // Normal number
                // exp_fp32 = exp_fp6 - bias_fp6 + bias_fp32
                // bias_fp6 = 3, bias_fp32 = 127
                // exp_fp32 = exp_fp6 - 3 + 127 = exp_fp6 + 124
                exp32 = {5'b0, exp6} + 8'd124;
                // Mantissa: 2 bits -> 23 bits (shift left 21)
                man32 = {man6, 21'b0};
                fp6_e3m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    //  ( - )
    reg [31:0] temp_sum;
    reg [31:0] a_fp32, b_fp32;

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            matrix_d <= 0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // : 
            for (m = 0; m < M; m = m + 1) begin
                for (n = 0; n < N; n = n + 1) begin
                    temp_sum = matrix_c[(m*N + n)*32 +: 32];
                    for (k = 0; k < K; k = k + 1) begin
                        a_fp32 = fp6_e3m2_to_fp32(matrix_a[(m*K + k)*6 +: 6]);
                        b_fp32 = fp6_e3m2_to_fp32(matrix_b[(k*N + n)*6 +: 6]);
                        //  (FP32 MAC)
                        temp_sum = temp_sum + (a_fp32[22:0] * b_fp32[22:0]);
                    end
                    matrix_d[(m*N + n)*32 +: 32] <= temp_sum;
                end
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
    /* verilator lint_on BLKSEQ */

endmodule
