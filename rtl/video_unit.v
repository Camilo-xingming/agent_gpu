//============================================================================
// RalphGPU - Video Processing Unit
// PTX video instructions for image/video processing and ML workloads
// Supports SIMD byte/half-word operations and dot products
//============================================================================

`include "gpu_defines.vh"

module video_unit (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire [5:0]  func,
    input  wire        valid_in,
    input  wire        is_signed,     // Signed vs unsigned operation

    // Operands
    input  wire [31:0] operand_a,
    input  wire [31:0] operand_b,
    input  wire [31:0] operand_c,     // Accumulator for MAD/DP4A

    // Result
    output reg  [31:0] result,
    output reg         valid_out,

    // Flags
    output reg         overflow,
    output reg         saturate_flag  // Result was saturated
);

    //------------------------------------------------------------------------
    // Pipeline registers
    //------------------------------------------------------------------------
    reg [5:0]  func_r;
    reg        is_signed_r;
    reg [31:0] op_a_r, op_b_r, op_c_r;
    reg        valid_r;

    //------------------------------------------------------------------------
    // Byte extraction helpers
    //------------------------------------------------------------------------
    wire [7:0] a_byte0 = operand_a[7:0];
    wire [7:0] a_byte1 = operand_a[15:8];
    wire [7:0] a_byte2 = operand_a[23:16];
    wire [7:0] a_byte3 = operand_a[31:24];

    wire [7:0] b_byte0 = operand_b[7:0];
    wire [7:0] b_byte1 = operand_b[15:8];
    wire [7:0] b_byte2 = operand_b[23:16];
    wire [7:0] b_byte3 = operand_b[31:24];

    // Half-word extraction
    wire [15:0] a_half0 = operand_a[15:0];
    wire [15:0] a_half1 = operand_a[31:16];

    wire [15:0] b_half0 = operand_b[15:0];
    wire [15:0] b_half1 = operand_b[31:16];

    //------------------------------------------------------------------------
    // Signed extension helpers
    //------------------------------------------------------------------------
    wire signed [31:0] a_signed = $signed(operand_a);
    wire signed [31:0] b_signed = $signed(operand_b);
    wire signed [31:0] c_signed = $signed(operand_c);

    wire signed [8:0] a_b0_s = $signed({a_byte0[7], a_byte0});
    wire signed [8:0] a_b1_s = $signed({a_byte1[7], a_byte1});
    wire signed [8:0] a_b2_s = $signed({a_byte2[7], a_byte2});
    wire signed [8:0] a_b3_s = $signed({a_byte3[7], a_byte3});

    wire signed [8:0] b_b0_s = $signed({b_byte0[7], b_byte0});
    wire signed [8:0] b_b1_s = $signed({b_byte1[7], b_byte1});
    wire signed [8:0] b_b2_s = $signed({b_byte2[7], b_byte2});
    wire signed [8:0] b_b3_s = $signed({b_byte3[7], b_byte3});

    //------------------------------------------------------------------------
    // Absolute difference helper
    //------------------------------------------------------------------------
    function [31:0] absdiff_u;
        input [31:0] a, b;
        begin
            if (a >= b)
                absdiff_u = a - b;
            else
                absdiff_u = b - a;
        end
    endfunction

    function [31:0] absdiff_s;
        input signed [31:0] a, b;
        begin
            if (a >= b)
                absdiff_s = a - b;
            else
                absdiff_s = b - a;
        end
    endfunction

    //------------------------------------------------------------------------
    // Byte absolute difference
    //------------------------------------------------------------------------
    function [7:0] absdiff_byte_u;
        input [7:0] a, b;
        begin
            if (a >= b)
                absdiff_byte_u = a - b;
            else
                absdiff_byte_u = b - a;
        end
    endfunction

    function [7:0] absdiff_byte_s;
        input signed [7:0] a, b;
        begin
            if (a >= b)
                absdiff_byte_s = a - b;
            else
                absdiff_byte_s = b - a;
        end
    endfunction

    //------------------------------------------------------------------------
    // Saturation helpers
    //------------------------------------------------------------------------
    function [7:0] saturate_s8;
        input signed [15:0] val;
        begin
            if (val > 127)
                saturate_s8 = 8'd127;
            else if (val < -128)
                saturate_s8 = -8'd128;
            else
                saturate_s8 = val[7:0];
        end
    endfunction

    function [7:0] saturate_u8;
        input [15:0] val;
        begin
            if (val > 255)
                saturate_u8 = 8'd255;
            else
                saturate_u8 = val[7:0];
        end
    endfunction

    //------------------------------------------------------------------------
    // DP4A - 4-element dot product with accumulate (key for INT8 ML)
    // result = c + sum(a[i] * b[i]) for i in 0..3 (4 bytes each)
    //------------------------------------------------------------------------
    wire signed [31:0] dp4a_signed_result;
    wire [31:0] dp4a_unsigned_result;

    // Signed DP4A
    wire signed [17:0] prod0_s = $signed({{8{a_byte0[7]}}, a_byte0}) * $signed({{8{b_byte0[7]}}, b_byte0});
    wire signed [17:0] prod1_s = $signed({{8{a_byte1[7]}}, a_byte1}) * $signed({{8{b_byte1[7]}}, b_byte1});
    wire signed [17:0] prod2_s = $signed({{8{a_byte2[7]}}, a_byte2}) * $signed({{8{b_byte2[7]}}, b_byte2});
    wire signed [17:0] prod3_s = $signed({{8{a_byte3[7]}}, a_byte3}) * $signed({{8{b_byte3[7]}}, b_byte3});

    assign dp4a_signed_result = c_signed + prod0_s + prod1_s + prod2_s + prod3_s;

    // Unsigned DP4A
    wire [17:0] prod0_u = {10'b0, a_byte0} * {10'b0, b_byte0};
    wire [17:0] prod1_u = {10'b0, a_byte1} * {10'b0, b_byte1};
    wire [17:0] prod2_u = {10'b0, a_byte2} * {10'b0, b_byte2};
    wire [17:0] prod3_u = {10'b0, a_byte3} * {10'b0, b_byte3};

    assign dp4a_unsigned_result = operand_c + prod0_u + prod1_u + prod2_u + prod3_u;

    //------------------------------------------------------------------------
    // DP2A - 2-element dot product with accumulate (16-bit elements)
    //------------------------------------------------------------------------
    wire signed [31:0] dp2a_signed_result;
    wire [31:0] dp2a_unsigned_result;

    wire signed [33:0] prod_h0_s = $signed({{16{a_half0[15]}}, a_half0}) * $signed({{16{b_half0[15]}}, b_half0});
    wire signed [33:0] prod_h1_s = $signed({{16{a_half1[15]}}, a_half1}) * $signed({{16{b_half1[15]}}, b_half1});

    assign dp2a_signed_result = c_signed + prod_h0_s[31:0] + prod_h1_s[31:0];

    wire [33:0] prod_h0_u = {16'b0, a_half0} * {16'b0, b_half0};
    wire [33:0] prod_h1_u = {16'b0, a_half1} * {16'b0, b_half1};

    assign dp2a_unsigned_result = operand_c + prod_h0_u[31:0] + prod_h1_u[31:0];

    //------------------------------------------------------------------------
    // SIMD 4x8-bit operations
    //------------------------------------------------------------------------
    wire [7:0] vadd4_r0 = a_byte0 + b_byte0;
    wire [7:0] vadd4_r1 = a_byte1 + b_byte1;
    wire [7:0] vadd4_r2 = a_byte2 + b_byte2;
    wire [7:0] vadd4_r3 = a_byte3 + b_byte3;

    wire [7:0] vsub4_r0 = a_byte0 - b_byte0;
    wire [7:0] vsub4_r1 = a_byte1 - b_byte1;
    wire [7:0] vsub4_r2 = a_byte2 - b_byte2;
    wire [7:0] vsub4_r3 = a_byte3 - b_byte3;

    wire [7:0] vabsdiff4_u_r0 = absdiff_byte_u(a_byte0, b_byte0);
    wire [7:0] vabsdiff4_u_r1 = absdiff_byte_u(a_byte1, b_byte1);
    wire [7:0] vabsdiff4_u_r2 = absdiff_byte_u(a_byte2, b_byte2);
    wire [7:0] vabsdiff4_u_r3 = absdiff_byte_u(a_byte3, b_byte3);

    //------------------------------------------------------------------------
    // SIMD 2x16-bit operations
    //------------------------------------------------------------------------
    wire [15:0] vadd2_r0 = a_half0 + b_half0;
    wire [15:0] vadd2_r1 = a_half1 + b_half1;

    wire [15:0] vsub2_r0 = a_half0 - b_half0;
    wire [15:0] vsub2_r1 = a_half1 - b_half1;

    wire [15:0] vmul2_r0 = a_half0 * b_half0;
    wire [15:0] vmul2_r1 = a_half1 * b_half1;

    //------------------------------------------------------------------------
    // Main computation pipeline
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r <= 1'b0;
            valid_out <= 1'b0;
            result <= 32'b0;
            overflow <= 1'b0;
            saturate_flag <= 1'b0;
        end else begin
            // Pipeline stage 1
            valid_r <= valid_in;
            func_r <= func;
            is_signed_r <= is_signed;
            op_a_r <= operand_a;
            op_b_r <= operand_b;
            op_c_r <= operand_c;

            // Pipeline stage 2: Output
            valid_out <= valid_r;
            overflow <= 1'b0;
            saturate_flag <= 1'b0;

            if (valid_r) begin
                case (func_r)
                    `VIDEO_VADD: begin
                        if (is_signed_r)
                            result <= $signed(op_a_r) + $signed(op_b_r);
                        else
                            result <= op_a_r + op_b_r;
                    end

                    `VIDEO_VSUB: begin
                        if (is_signed_r)
                            result <= $signed(op_a_r) - $signed(op_b_r);
                        else
                            result <= op_a_r - op_b_r;
                    end

                    `VIDEO_VABSDIFF: begin
                        if (is_signed_r)
                            result <= absdiff_s($signed(op_a_r), $signed(op_b_r));
                        else
                            result <= absdiff_u(op_a_r, op_b_r);
                    end

                    `VIDEO_VMIN: begin
                        if (is_signed_r)
                            result <= ($signed(op_a_r) < $signed(op_b_r)) ? op_a_r : op_b_r;
                        else
                            result <= (op_a_r < op_b_r) ? op_a_r : op_b_r;
                    end

                    `VIDEO_VMAX: begin
                        if (is_signed_r)
                            result <= ($signed(op_a_r) > $signed(op_b_r)) ? op_a_r : op_b_r;
                        else
                            result <= (op_a_r > op_b_r) ? op_a_r : op_b_r;
                    end

                    `VIDEO_VSHL: begin
                        result <= op_a_r << op_b_r[4:0];
                    end

                    `VIDEO_VSHR: begin
                        if (is_signed_r)
                            result <= $signed(op_a_r) >>> op_b_r[4:0];
                        else
                            result <= op_a_r >> op_b_r[4:0];
                    end

                    `VIDEO_VMAD: begin
                        if (is_signed_r)
                            result <= $signed(op_a_r) * $signed(op_b_r) + $signed(op_c_r);
                        else
                            result <= op_a_r * op_b_r + op_c_r;
                    end

                    //----------------------------------------------------
                    // SIMD 4x8-bit operations
                    //----------------------------------------------------
                    `VIDEO_VADD4: begin
                        result <= {vadd4_r3, vadd4_r2, vadd4_r1, vadd4_r0};
                    end

                    `VIDEO_VSUB4: begin
                        result <= {vsub4_r3, vsub4_r2, vsub4_r1, vsub4_r0};
                    end

                    `VIDEO_VABSDIFF4: begin
                        result <= {vabsdiff4_u_r3, vabsdiff4_u_r2,
                                  vabsdiff4_u_r1, vabsdiff4_u_r0};
                    end

                    //----------------------------------------------------
                    // SIMD 2x16-bit operations
                    //----------------------------------------------------
                    `VIDEO_VADD2: begin
                        result <= {vadd2_r1, vadd2_r0};
                    end

                    `VIDEO_VSUB2: begin
                        result <= {vsub2_r1, vsub2_r0};
                    end

                    `VIDEO_VMUL2: begin
                        result <= {vmul2_r1, vmul2_r0};
                    end

                    //----------------------------------------------------
                    // Dot Product instructions (critical for INT8 ML)
                    //----------------------------------------------------
                    `VIDEO_DP4A: begin
                        if (is_signed_r)
                            result <= dp4a_signed_result;
                        else
                            result <= dp4a_unsigned_result;
                    end

                    `VIDEO_DP2A: begin
                        if (is_signed_r)
                            result <= dp2a_signed_result;
                        else
                            result <= dp2a_unsigned_result;
                    end

                    default: begin
                        result <= 32'b0;
                    end
                endcase
            end
        end
    end

endmodule

//============================================================================
// SIMD Video Unit - 32 lanes for warp-wide operations
//============================================================================
module video_simd_unit #(
    parameter NUM_LANES = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [5:0]              func,
    input  wire                    valid_in,
    input  wire                    is_signed,
    input  wire [NUM_LANES*32-1:0] operand_a,
    input  wire [NUM_LANES*32-1:0] operand_b,
    input  wire [NUM_LANES*32-1:0] operand_c,
    input  wire [NUM_LANES-1:0]    lane_mask,
    output wire [NUM_LANES*32-1:0] result,
    output wire                    valid_out
);

    wire [NUM_LANES-1:0] lane_valid_out;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : video_lanes
            video_unit u_video (
                .clk          (clk),
                .rst_n        (rst_n),
                .func         (func),
                .valid_in     (valid_in & lane_mask[i]),
                .is_signed    (is_signed),
                .operand_a    (operand_a[i*32 +: 32]),
                .operand_b    (operand_b[i*32 +: 32]),
                .operand_c    (operand_c[i*32 +: 32]),
                .result       (result[i*32 +: 32]),
                .valid_out    (lane_valid_out[i]),
                .overflow     (),
                .saturate_flag()
            );
        end
    endgenerate

    assign valid_out = |lane_valid_out;

endmodule
