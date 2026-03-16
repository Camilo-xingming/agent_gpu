//============================================================================
// RalphGPU - WGMMA (Warpgroup Matrix Multiply-Accumulate)
// Hopper Architecture Tensor Core Extension
// Supports: wgmma.mma_async, wgmma.fence, wgmma.commit_group, wgmma.wait_group
//
// WGMMA operates at Warpgroup (4 warps = 128 threads) level
// Supports larger matrix sizes and asynchronous execution
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module wgmma #(
    parameter WARPGROUP_SIZE = 4,       // 4 warps per warpgroup
    parameter THREADS_PER_WARP = 32,
    parameter MAX_PENDING_OPS = 8       // Max pending operations
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Control Interface
    input  wire [5:0]           func,           // Function code
    input  wire                 valid_in,
    input  wire [2:0]           warpgroup_id,   // Warpgroup ID
    input  wire [3:0]           wait_count,     // wait_group count

    // Matrix Descriptors
    input  wire [63:0]          desc_a,         // Matrix A descriptor
    input  wire [63:0]          desc_b,         // Matrix B descriptor
    input  wire [31:0]          scale_d,        // Scale factor
    
    // Data Interface
    input  wire [511:0]         data_a,         // Matrix A data
    input  wire [511:0]         data_b,         // Matrix B data
    input  wire [1023:0]        accum_in,       // Input accumulator
    output reg  [1023:0]        accum_out,      // Output accumulator
    
    output reg                  ready,
    output reg                  done,
    output reg  [3:0]           pending_ops
);

    //------------------------------------------------------------------------
    // Descriptor Decoding
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
    // Data Type Definitions
    //------------------------------------------------------------------------
    localparam DTYPE_FP16   = 4'b0000;
    localparam DTYPE_BF16   = 4'b0001;
    localparam DTYPE_TF32   = 4'b0010;
    localparam DTYPE_FP8_E4 = 4'b0011;
    localparam DTYPE_FP8_E5 = 4'b0100;
    localparam DTYPE_INT8   = 4'b0101;
    localparam DTYPE_FP4    = 4'b0110;
    localparam DTYPE_FP6_E3M2 = 4'b1000;

    //------------------------------------------------------------------------
    // FSM
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
    // Pending Ops Tracking
    //------------------------------------------------------------------------
    reg [MAX_PENDING_OPS-1:0] op_pending;
    reg [MAX_PENDING_OPS-1:0] op_committed;
    reg [$clog2(MAX_PENDING_OPS)-1:0] op_head, op_tail;

    //------------------------------------------------------------------------
    // Compute Core
    //------------------------------------------------------------------------
    reg [31:0] partial_sum [0:31];
    reg [1023:0] mma_result;

    // FP16 -> FP32 Conversion
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
                fp16_to_fp32 = {sign, 31'b0};
            end else if (exp16 == 5'h1F) begin
                fp16_to_fp32 = {sign, 8'hFF, {man16, 13'b0}};
            end else if (exp16 == 0) begin
                fp16_to_fp32 = {sign, 8'b0, {man16, 13'b0}};
            end else begin
                exp32 = {3'b0, exp16} + 8'd112;
                man32 = {man16, 13'b0};
                fp16_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    // BF16 -> FP32 Conversion
    function [31:0] bf16_to_fp32;
        input [15:0] bf16;
        begin
            bf16_to_fp32 = {bf16, 16'b0};
        end
    endfunction

    // TF32 -> FP32 Conversion
    function [31:0] tf32_to_fp32;
        input [18:0] tf32;
        begin
            tf32_to_fp32 = {tf32[18], tf32[17:10], tf32[9:0], 13'b0};
        end
    endfunction

    // FP8 E4M3 -> FP32 Conversion
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
                exp32 = {4'b0, exp8} + 8'd120;
                man32 = {man8, 20'b0};
                fp8_e4m3_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    // FP8 E5M2 -> FP32 Conversion
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
                exp32 = {3'b0, exp8} + 8'd112;
                man32 = {man8, 21'b0};
                fp8_e5m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    // FP6 E3M2 -> FP32 Conversion
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
                    fp6_e3m2_to_fp32 = {sign, 31'b0};
                end else begin
                    fp6_e3m2_to_fp32 = {sign, 8'b0, {man6, 21'b0}};
                end
            end else if (exp6 == 3'b111) begin
                fp6_e3m2_to_fp32 = {sign, 8'hFF, (man6 != 0) ? 23'h400000 : 23'h0};
            end else begin
                exp32 = {5'b0, exp6} + 8'd124;
                man32 = {man6, 21'b0};
                fp6_e3m2_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    // FP4 E2M1 -> FP32 Conversion
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
                    fp4_e2m1_to_fp32 = {sign, 31'b0};
                end else begin
                    fp4_e2m1_to_fp32 = {sign, 8'b0, {man4, 22'b0}};
                end
            end else if (exp4 == 2'b11) begin
                fp4_e2m1_to_fp32 = {sign, 8'hFF, man4 ? 23'h400000 : 23'h0};
            end else begin
                exp32 = {6'b0, exp4} + 8'd126;
                man32 = {man4, 22'b0};
                fp4_e2m1_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    // Simplified FP32 MAC
    function [31:0] fp32_mac;
        input [31:0] a;
        input [31:0] b;
        input [31:0] c;
        reg [63:0] product;
        begin
            product = {32'd0, ({9'b0, a[22:0]} | 32'h00800000)} * {32'd0, ({9'b0, b[22:0]} | 32'h00800000)};
            fp32_mac = c + {9'b0, product[46:24]};
        end
    endfunction

    integer i;

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
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp16_to_fp32(data_a[i*16 +: 16]),
                                    fp16_to_fp32(data_b[i*16 +: 16]),
                                    partial_sum[i]
                                );
                            end
                        end

                        DTYPE_BF16: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    bf16_to_fp32(data_a[i*16 +: 16]),
                                    bf16_to_fp32(data_b[i*16 +: 16]),
                                    partial_sum[i]
                                );
                            end
                        end

                        DTYPE_TF32: begin
                            for (i = 0; i < 16; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    tf32_to_fp32(data_a[i*32 +: 19]),
                                    tf32_to_fp32(data_b[i*32 +: 19]),
                                    partial_sum[i]
                                );
                            end
                        end

                        DTYPE_FP8_E4: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp8_e4m3_to_fp32(data_a[(i+32)*8 +: 8]),
                                    fp8_e4m3_to_fp32(data_b[(i+32)*8 +: 8]),
                                    fp32_mac(
                                        fp8_e4m3_to_fp32(data_a[i*8 +: 8]),
                                        fp8_e4m3_to_fp32(data_b[i*8 +: 8]),
                                        partial_sum[i]
                                    )
                                );
                            end
                        end

                        DTYPE_FP8_E5: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp8_e5m2_to_fp32(data_a[(i+32)*8 +: 8]),
                                    fp8_e5m2_to_fp32(data_b[(i+32)*8 +: 8]),
                                    fp32_mac(
                                        fp8_e5m2_to_fp32(data_a[i*8 +: 8]),
                                        fp8_e5m2_to_fp32(data_b[i*8 +: 8]),
                                        partial_sum[i]
                                    )
                                );
                            end
                        end

                        DTYPE_FP6_E3M2: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp6_e3m2_to_fp32(data_a[i*6 +: 6]),
                                    fp6_e3m2_to_fp32(data_b[i*6 +: 6]),
                                    partial_sum[i]
                                );
                            end
                        end

                        DTYPE_FP4: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp4_e2m1_to_fp32(data_a[(i+96)*4 +: 4]),
                                    fp4_e2m1_to_fp32(data_b[(i+96)*4 +: 4]),
                                    fp32_mac(
                                        fp4_e2m1_to_fp32(data_a[(i+64)*4 +: 4]),
                                        fp4_e2m1_to_fp32(data_b[(i+64)*4 +: 4]),
                                        fp32_mac(
                                            fp4_e2m1_to_fp32(data_a[(i+32)*4 +: 4]),
                                            fp4_e2m1_to_fp32(data_b[(i+32)*4 +: 4]),
                                            fp32_mac(
                                                fp4_e2m1_to_fp32(data_a[i*4 +: 4]),
                                                fp4_e2m1_to_fp32(data_b[i*4 +: 4]),
                                                partial_sum[i]
                                            )
                                        )
                                    )
                                );
                            end
                        end

                        DTYPE_INT8: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= partial_sum[i] +
                                    ($signed({{24{data_a[i*8+7]}}, data_a[i*8 +: 8]}) *
                                     $signed({{24{data_b[i*8+7]}}, data_b[i*8 +: 8]}));
                            end
                        end

                        default: begin
                            for (i = 0; i < 32; i = i + 1) begin
                                partial_sum[i] <= fp32_mac(
                                    fp16_to_fp32(data_a[i*16 +: 16]),
                                    fp16_to_fp32(data_b[i*16 +: 16]),
                                    partial_sum[i]
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

module wgmma_descriptor_builder (
    input  wire [31:0]  base_addr,
    input  wire [3:0]   layout,
    input  wire [3:0]   data_type,
    input  wire [15:0]  leading_dim,
    input  wire [7:0]   start_offset,

    output wire [63:0]  descriptor
);
    assign descriptor = {
        start_offset,
        layout,
        data_type,
        leading_dim,
        base_addr
    };
endmodule

module wgmma_accumulator #(
    parameter NUM_ACCUMULATORS = 8,
    parameter ACCUM_WIDTH = 1024
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] read_idx,
    output wire [ACCUM_WIDTH-1:0]       read_data,
    input  wire                         write_en,
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] write_idx,
    input  wire [ACCUM_WIDTH-1:0]       write_data,
    input  wire                         clear_en,
    input  wire [$clog2(NUM_ACCUMULATORS)-1:0] clear_idx
);
    reg [ACCUM_WIDTH-1:0] accumulators [0:NUM_ACCUMULATORS-1];
    integer i;
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
