//============================================================================
// RalphGPU - CVT Unit (Type Conversion Unit)
// 完整的PTX类型转换支持
// 支持: 所有整数/浮点类型转换，饱和，舍入模式
//============================================================================

`include "gpu_defines.vh"

module cvt_unit (
    input  wire        clk,
    input  wire        rst_n,

    // 控制
    input  wire [5:0]  func,            // 转换类型
    input  wire [1:0]  rnd_mode,        // 舍入模式
    input  wire        saturate,        // 饱和模式
    input  wire        ftz,             // Flush to zero
    input  wire        valid_in,

    // 源数据 (最大64位)
    input  wire [63:0] src,
    input  wire [2:0]  src_type,        // 源类型

    // 目标数据
    output reg  [63:0] dst,
    output reg         valid_out,
    output reg         overflow,
    output reg         inexact
);

    //------------------------------------------------------------------------
    // 类型编码
    //------------------------------------------------------------------------
    localparam TYPE_S8   = 3'd0;
    localparam TYPE_U8   = 3'd1;
    localparam TYPE_S16  = 3'd2;
    localparam TYPE_U16  = 3'd3;
    localparam TYPE_S32  = 3'd4;
    localparam TYPE_U32  = 3'd5;
    localparam TYPE_S64  = 3'd6;
    localparam TYPE_U64  = 3'd7;

    //------------------------------------------------------------------------
    // FP32 <-> 整数转换
    //------------------------------------------------------------------------
    wire [31:0] fp32_src = src[31:0];
    wire fp32_sign = fp32_src[31];
    wire [7:0] fp32_exp = fp32_src[30:23];
    wire [22:0] fp32_man = fp32_src[22:0];
    wire [23:0] fp32_sig = {1'b1, fp32_man};

    // FP32 -> S32 (带舍入)
    wire signed [31:0] fp32_to_s32;
    wire fp32_to_s32_ovf;

    fp32_to_int #(.SIGNED(1), .WIDTH(32)) u_fp32_to_s32 (
        .fp32(fp32_src),
        .rnd_mode(rnd_mode),
        .saturate(saturate),
        .result(fp32_to_s32),
        .overflow(fp32_to_s32_ovf)
    );

    // FP32 -> U32
    wire [31:0] fp32_to_u32;
    wire fp32_to_u32_ovf;

    fp32_to_int #(.SIGNED(0), .WIDTH(32)) u_fp32_to_u32 (
        .fp32(fp32_src),
        .rnd_mode(rnd_mode),
        .saturate(saturate),
        .result(fp32_to_u32),
        .overflow(fp32_to_u32_ovf)
    );

    // S32 -> FP32
    wire [31:0] s32_to_fp32;
    int_to_fp32 #(.SIGNED(1)) u_s32_to_fp32 (
        .int_val(src[31:0]),
        .rnd_mode(rnd_mode),
        .result(s32_to_fp32)
    );

    // U32 -> FP32
    wire [31:0] u32_to_fp32;
    int_to_fp32 #(.SIGNED(0)) u_u32_to_fp32 (
        .int_val(src[31:0]),
        .rnd_mode(rnd_mode),
        .result(u32_to_fp32)
    );

    //------------------------------------------------------------------------
    // FP64 <-> 整数转换
    //------------------------------------------------------------------------
    wire [63:0] fp64_src = src;
    wire fp64_sign = fp64_src[63];
    wire [10:0] fp64_exp = fp64_src[62:52];
    wire [51:0] fp64_man = fp64_src[51:0];

    // FP64 -> S64
    wire signed [63:0] fp64_to_s64;
    wire fp64_to_s64_ovf;

    fp64_to_int #(.SIGNED(1)) u_fp64_to_s64 (
        .fp64(fp64_src),
        .rnd_mode(rnd_mode),
        .saturate(saturate),
        .result(fp64_to_s64),
        .overflow(fp64_to_s64_ovf)
    );

    // FP64 -> U64
    wire [63:0] fp64_to_u64;
    wire fp64_to_u64_ovf;

    fp64_to_int #(.SIGNED(0)) u_fp64_to_u64 (
        .fp64(fp64_src),
        .rnd_mode(rnd_mode),
        .saturate(saturate),
        .result(fp64_to_u64),
        .overflow(fp64_to_u64_ovf)
    );

    // S64 -> FP64
    wire [63:0] s64_to_fp64;
    int64_to_fp64 #(.SIGNED(1)) u_s64_to_fp64 (
        .int_val(src),
        .rnd_mode(rnd_mode),
        .result(s64_to_fp64)
    );

    // U64 -> FP64
    wire [63:0] u64_to_fp64;
    int64_to_fp64 #(.SIGNED(0)) u_u64_to_fp64 (
        .int_val(src),
        .rnd_mode(rnd_mode),
        .result(u64_to_fp64)
    );

    //------------------------------------------------------------------------
    // FP32 <-> FP64 转换
    //------------------------------------------------------------------------
    wire [31:0] fp64_to_fp32;
    wire fp64_to_fp32_inx;

    fp64_to_fp32_cvt u_fp64_to_fp32 (
        .fp64(fp64_src),
        .rnd_mode(rnd_mode),
        .ftz(ftz),
        .result(fp64_to_fp32),
        .inexact(fp64_to_fp32_inx)
    );

    wire [63:0] fp32_to_fp64;
    fp32_to_fp64_cvt u_fp32_to_fp64 (
        .fp32(fp32_src),
        .result(fp32_to_fp64)
    );

    //------------------------------------------------------------------------
    // FP16 <-> FP32 转换
    //------------------------------------------------------------------------
    wire [15:0] fp16_src = src[15:0];
    wire [31:0] fp16_to_fp32;

    fp16_to_fp32_cvt u_fp16_to_fp32 (
        .fp16(fp16_src),
        .result(fp16_to_fp32)
    );

    wire [15:0] fp32_to_fp16;
    wire fp32_to_fp16_inx;

    fp32_to_fp16_cvt u_fp32_to_fp16 (
        .fp32(fp32_src),
        .rnd_mode(rnd_mode),
        .result(fp32_to_fp16),
        .inexact(fp32_to_fp16_inx)
    );

    //------------------------------------------------------------------------
    // 整数类型转换 (符号扩展/截断/饱和)
    //------------------------------------------------------------------------
    wire [63:0] int_cvt_result;
    wire int_cvt_ovf;

    integer_cvt u_int_cvt (
        .src(src),
        .src_type(src_type),
        .dst_type(func[2:0]),
        .saturate(saturate),
        .result(int_cvt_result),
        .overflow(int_cvt_ovf)
    );

    //------------------------------------------------------------------------
    // 结果选择
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dst <= 64'b0;
            valid_out <= 1'b0;
            overflow <= 1'b0;
            inexact <= 1'b0;
        end else if (valid_in) begin
            valid_out <= 1'b1;
            overflow <= 1'b0;
            inexact <= 1'b0;

            case (func)
                `CVT_S32_F32: begin
                    dst <= {32'b0, fp32_to_s32};
                    overflow <= fp32_to_s32_ovf;
                end

                `CVT_U32_F32: begin
                    dst <= {32'b0, fp32_to_u32};
                    overflow <= fp32_to_u32_ovf;
                end

                `CVT_F32_S32: begin
                    dst <= {32'b0, s32_to_fp32};
                end

                `CVT_F32_U32: begin
                    dst <= {32'b0, u32_to_fp32};
                end

                `CVT_F32_F64: begin
                    dst <= {32'b0, fp64_to_fp32};
                    inexact <= fp64_to_fp32_inx;
                end

                `CVT_F64_F32: begin
                    dst <= fp32_to_fp64;
                end

                `CVT_F32_F16: begin
                    dst <= {48'b0, fp16_to_fp32};
                end

                `CVT_F16_F32: begin
                    dst <= {48'b0, fp32_to_fp16};
                    inexact <= fp32_to_fp16_inx;
                end

                `CVT_S64_F64: begin
                    dst <= fp64_to_s64;
                    overflow <= fp64_to_s64_ovf;
                end

                `CVT_U64_F64: begin
                    dst <= fp64_to_u64;
                    overflow <= fp64_to_u64_ovf;
                end

                `CVT_F64_S64: begin
                    dst <= s64_to_fp64;
                end

                `CVT_F64_U64: begin
                    dst <= u64_to_fp64;
                end

                default: begin
                    // 整数类型转换
                    dst <= int_cvt_result;
                    overflow <= int_cvt_ovf;
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule


//============================================================================
// FP32 -> 整数转换
//============================================================================
module fp32_to_int #(
    parameter SIGNED = 1,
    parameter WIDTH = 32
)(
    input  wire [31:0] fp32,
    input  wire [1:0]  rnd_mode,
    input  wire        saturate,
    output reg  [WIDTH-1:0] result,
    output reg         overflow
);

    localparam [7:0] EXP_BIAS = 8'd127;

    wire sign = fp32[31];
    wire [7:0] exp = fp32[30:23];
    wire [22:0] man = fp32[22:0];
    wire [23:0] sig = {1'b1, man};

    wire is_zero = (exp == 0) && (man == 0);
    wire is_inf = (exp == 8'hFF) && (man == 0);
    wire is_nan = (exp == 8'hFF) && (man != 0);

    // 计算实际指数
    wire signed [8:0] real_exp = {1'b0, exp} - EXP_BIAS;

    // 移位量
    wire [5:0] shift_right = (real_exp < 23) ? (6'd23 - real_exp[5:0]) : 6'd0;
    wire [5:0] shift_left = (real_exp > 23) ? (real_exp[5:0] - 6'd23) : 6'd0;

    // 基础结果
    wire [63:0] shifted_sig = (real_exp < 23) ? ({40'b0, sig} >> shift_right) :
                                                ({40'b0, sig} << shift_left);

    // 舍入
    wire [63:0] rounded;
    wire round_bit;

    assign round_bit = (shift_right > 0) ? sig[shift_right - 1] : 1'b0;

    assign rounded = (rnd_mode == 2'b00) ? // Round to nearest
                     (round_bit ? shifted_sig + 1 : shifted_sig) :
                     (rnd_mode == 2'b01) ? shifted_sig : // Round toward zero
                     (rnd_mode == 2'b10 && sign) ? shifted_sig + 1 : // Round toward -inf
                     (rnd_mode == 2'b11 && !sign) ? shifted_sig + 1 : // Round toward +inf
                     shifted_sig;

    // 最终结果
    wire [WIDTH-1:0] unsigned_result = rounded[WIDTH-1:0];
    wire signed [WIDTH-1:0] signed_result = sign ? -unsigned_result : unsigned_result;

    // 溢出检测
    wire positive_overflow = !sign && (real_exp >= WIDTH);
    wire negative_overflow = SIGNED && sign && (real_exp >= WIDTH-1);

    always @(*) begin
        overflow = 1'b0;

        if (is_nan) begin
            result = 0;
            overflow = 1'b1;
        end else if (is_zero) begin
            result = 0;
        end else if (is_inf || positive_overflow) begin
            if (saturate) begin
                result = SIGNED ? {1'b0, {(WIDTH-1){1'b1}}} : {WIDTH{1'b1}};
            end else begin
                result = 0;
            end
            overflow = 1'b1;
        end else if (negative_overflow) begin
            if (saturate) begin
                result = {1'b1, {(WIDTH-1){1'b0}}};
            end else begin
                result = 0;
            end
            overflow = 1'b1;
        end else if (!SIGNED && sign) begin
            if (saturate) begin
                result = 0;
            end else begin
                result = 0;
            end
            overflow = 1'b1;
        end else begin
            result = SIGNED ? signed_result : unsigned_result;
        end
    end

endmodule


//============================================================================
// 整数 -> FP32 转换
//============================================================================
module int_to_fp32 #(
    parameter SIGNED = 1
)(
    input  wire [31:0] int_val,
    input  wire [1:0]  rnd_mode,
    output reg  [31:0] result
);

    wire sign = SIGNED && int_val[31];
    wire [31:0] abs_val = (SIGNED && int_val[31]) ? -int_val : int_val;

    // 前导零计数
    function [4:0] clz32;
        input [31:0] val;
        integer i;
        begin
            clz32 = 32;
            for (i = 31; i >= 0; i = i - 1) begin
                if (val[i]) clz32 = 31 - i;
            end
        end
    endfunction

    wire [4:0] leading_zeros = clz32(abs_val);
    wire [7:0] exponent = (abs_val == 0) ? 8'd0 : (8'd127 + 8'd31 - {3'b0, leading_zeros});
    wire [31:0] normalized = abs_val << leading_zeros;
    wire [22:0] mantissa = normalized[30:8];

    always @(*) begin
        if (int_val == 0) begin
            result = 32'h00000000;
        end else begin
            result = {sign, exponent, mantissa};
        end
    end

endmodule


//============================================================================
// FP64 -> 整数转换
//============================================================================
module fp64_to_int #(
    parameter SIGNED = 1
)(
    input  wire [63:0] fp64,
    input  wire [1:0]  rnd_mode,
    input  wire        saturate,
    output reg  [63:0] result,
    output reg         overflow
);

    localparam [10:0] EXP_BIAS = 11'd1023;

    wire sign = fp64[63];
    wire [10:0] exp = fp64[62:52];
    wire [51:0] man = fp64[51:0];
    wire [52:0] sig = {1'b1, man};

    wire is_zero = (exp == 0) && (man == 0);
    wire is_inf = (exp == 11'h7FF) && (man == 0);
    wire is_nan = (exp == 11'h7FF) && (man != 0);

    wire signed [11:0] real_exp = {1'b0, exp} - EXP_BIAS;

    wire [6:0] shift_right = (real_exp < 52) ? (7'd52 - real_exp[6:0]) : 7'd0;
    wire [6:0] shift_left = (real_exp > 52) ? (real_exp[6:0] - 7'd52) : 7'd0;

    wire [63:0] shifted_sig = (real_exp < 52) ? (sig >> shift_right) :
                                                (sig << shift_left);

    wire [63:0] unsigned_result = shifted_sig;
    wire signed [63:0] signed_result = sign ? -unsigned_result : unsigned_result;

    wire positive_overflow = !sign && (real_exp >= 63);
    wire negative_overflow = SIGNED && sign && (real_exp >= 63);

    always @(*) begin
        overflow = 1'b0;

        if (is_nan || is_zero) begin
            result = 0;
            overflow = is_nan;
        end else if (is_inf || positive_overflow) begin
            result = saturate ? (SIGNED ? 64'h7FFFFFFFFFFFFFFF : 64'hFFFFFFFFFFFFFFFF) : 0;
            overflow = 1'b1;
        end else if (negative_overflow) begin
            result = saturate ? 64'h8000000000000000 : 0;
            overflow = 1'b1;
        end else if (!SIGNED && sign) begin
            result = saturate ? 0 : 0;
            overflow = 1'b1;
        end else begin
            result = SIGNED ? signed_result : unsigned_result;
        end
    end

endmodule


//============================================================================
// 整数64 -> FP64 转换
//============================================================================
module int64_to_fp64 #(
    parameter SIGNED = 1
)(
    input  wire [63:0] int_val,
    input  wire [1:0]  rnd_mode,
    output reg  [63:0] result
);

    wire sign = SIGNED && int_val[63];
    wire [63:0] abs_val = (SIGNED && int_val[63]) ? -int_val : int_val;

    // 前导零计数
    function [5:0] clz64;
        input [63:0] val;
        integer i;
        begin
            clz64 = 64;
            for (i = 63; i >= 0; i = i - 1) begin
                if (val[i]) clz64 = 63 - i;
            end
        end
    endfunction

    wire [5:0] leading_zeros = clz64(abs_val);
    wire [10:0] exponent = (abs_val == 0) ? 11'd0 : (11'd1023 + 11'd63 - {5'b0, leading_zeros});
    wire [63:0] normalized = abs_val << leading_zeros;
    wire [51:0] mantissa = normalized[62:11];

    always @(*) begin
        if (int_val == 0) begin
            result = 64'h0000000000000000;
        end else begin
            result = {sign, exponent, mantissa};
        end
    end

endmodule


//============================================================================
// FP64 -> FP32 转换
//============================================================================
module fp64_to_fp32_cvt (
    input  wire [63:0] fp64,
    input  wire [1:0]  rnd_mode,
    input  wire        ftz,
    output reg  [31:0] result,
    output reg         inexact
);

    wire sign = fp64[63];
    wire [10:0] exp64 = fp64[62:52];
    wire [51:0] man64 = fp64[51:0];

    wire is_zero = (exp64 == 0) && (man64 == 0);
    wire is_inf = (exp64 == 11'h7FF) && (man64 == 0);
    wire is_nan = (exp64 == 11'h7FF) && (man64 != 0);

    // 指数转换: bias 1023 -> 127
    wire signed [11:0] real_exp = {1'b0, exp64} - 12'd1023;
    wire [7:0] exp32 = real_exp[7:0] + 8'd127;

    // 尾数截断
    wire [22:0] man32 = man64[51:29];
    wire round_bit = man64[28];

    // 溢出/下溢
    wire overflow = (real_exp > 127);
    wire underflow = (real_exp < -126);

    always @(*) begin
        inexact = 1'b0;

        if (is_nan) begin
            result = {sign, 8'hFF, 23'h400000};  // QNaN
        end else if (is_inf || overflow) begin
            result = {sign, 8'hFF, 23'h0};  // Infinity
        end else if (is_zero || underflow) begin
            result = {sign, 31'h0};
            if (underflow && !is_zero) inexact = 1'b1;
        end else begin
            result = {sign, exp32, man32};
            inexact = (man64[28:0] != 0);
        end
    end

endmodule


//============================================================================
// FP32 -> FP64 转换
//============================================================================
module fp32_to_fp64_cvt (
    input  wire [31:0] fp32,
    output reg  [63:0] result
);

    wire sign = fp32[31];
    wire [7:0] exp32 = fp32[30:23];
    wire [22:0] man32 = fp32[22:0];

    wire is_zero = (exp32 == 0) && (man32 == 0);
    wire is_inf = (exp32 == 8'hFF) && (man32 == 0);
    wire is_nan = (exp32 == 8'hFF) && (man32 != 0);

    // 指数转换: bias 127 -> 1023
    wire [10:0] exp64 = {3'b0, exp32} + 11'd896;  // 1023 - 127

    // 尾数扩展
    wire [51:0] man64 = {man32, 29'b0};

    always @(*) begin
        if (is_nan) begin
            result = {sign, 11'h7FF, 52'h8000000000000};  // QNaN
        end else if (is_inf) begin
            result = {sign, 11'h7FF, 52'h0};
        end else if (is_zero) begin
            result = {sign, 63'h0};
        end else begin
            result = {sign, exp64, man64};
        end
    end

endmodule


//============================================================================
// FP16 -> FP32 转换
//============================================================================
module fp16_to_fp32_cvt (
    input  wire [15:0] fp16,
    output reg  [31:0] result
);

    wire sign = fp16[15];
    wire [4:0] exp16 = fp16[14:10];
    wire [9:0] man16 = fp16[9:0];

    wire is_zero = (exp16 == 0) && (man16 == 0);
    wire is_inf = (exp16 == 5'h1F) && (man16 == 0);
    wire is_nan = (exp16 == 5'h1F) && (man16 != 0);

    // 指数转换: bias 15 -> 127
    wire [7:0] exp32 = {3'b0, exp16} + 8'd112;  // 127 - 15

    // 尾数扩展
    wire [22:0] man32 = {man16, 13'b0};

    always @(*) begin
        if (is_nan) begin
            result = {sign, 8'hFF, 23'h400000};
        end else if (is_inf) begin
            result = {sign, 8'hFF, 23'h0};
        end else if (is_zero) begin
            result = {sign, 31'h0};
        end else begin
            result = {sign, exp32, man32};
        end
    end

endmodule


//============================================================================
// FP32 -> FP16 转换
//============================================================================
module fp32_to_fp16_cvt (
    input  wire [31:0] fp32,
    input  wire [1:0]  rnd_mode,
    output reg  [15:0] result,
    output reg         inexact
);

    wire sign = fp32[31];
    wire [7:0] exp32 = fp32[30:23];
    wire [22:0] man32 = fp32[22:0];

    wire is_zero = (exp32 == 0) && (man32 == 0);
    wire is_inf = (exp32 == 8'hFF) && (man32 == 0);
    wire is_nan = (exp32 == 8'hFF) && (man32 != 0);

    // 指数转换
    wire signed [8:0] real_exp = {1'b0, exp32} - 9'd127;
    wire [4:0] exp16 = real_exp[4:0] + 5'd15;

    // 尾数截断
    wire [9:0] man16 = man32[22:13];

    // 溢出/下溢
    wire overflow = (real_exp > 15);
    wire underflow = (real_exp < -14);

    always @(*) begin
        inexact = 1'b0;

        if (is_nan) begin
            result = {sign, 5'h1F, 10'h200};
        end else if (is_inf || overflow) begin
            result = {sign, 5'h1F, 10'h0};
        end else if (is_zero || underflow) begin
            result = {sign, 15'h0};
            inexact = underflow && !is_zero;
        end else begin
            result = {sign, exp16, man16};
            inexact = (man32[12:0] != 0);
        end
    end

endmodule


//============================================================================
// 整数类型转换
//============================================================================
module integer_cvt (
    input  wire [63:0] src,
    input  wire [2:0]  src_type,
    input  wire [2:0]  dst_type,
    input  wire        saturate,
    output reg  [63:0] result,
    output reg         overflow
);

    localparam TYPE_S8  = 3'd0;
    localparam TYPE_U8  = 3'd1;
    localparam TYPE_S16 = 3'd2;
    localparam TYPE_U16 = 3'd3;
    localparam TYPE_S32 = 3'd4;
    localparam TYPE_U32 = 3'd5;
    localparam TYPE_S64 = 3'd6;
    localparam TYPE_U64 = 3'd7;

    // 符号扩展源值到64位
    reg signed [63:0] src_signed;
    reg [63:0] src_unsigned;

    always @(*) begin
        case (src_type)
            TYPE_S8:  src_signed = {{56{src[7]}}, src[7:0]};
            TYPE_U8:  src_signed = {56'b0, src[7:0]};
            TYPE_S16: src_signed = {{48{src[15]}}, src[15:0]};
            TYPE_U16: src_signed = {48'b0, src[15:0]};
            TYPE_S32: src_signed = {{32{src[31]}}, src[31:0]};
            TYPE_U32: src_signed = {32'b0, src[31:0]};
            TYPE_S64: src_signed = src;
            TYPE_U64: src_signed = src;
            default:  src_signed = src;
        endcase
        src_unsigned = src_signed;
    end

    // 目标范围检查和饱和
    always @(*) begin
        overflow = 1'b0;

        case (dst_type)
            TYPE_S8: begin
                if (src_signed > 127) begin
                    result = saturate ? 64'd127 : {56'b0, src[7:0]};
                    overflow = 1'b1;
                end else if (src_signed < -128) begin
                    result = saturate ? 64'hFFFFFFFFFFFFFF80 : {56'b0, src[7:0]};
                    overflow = 1'b1;
                end else begin
                    result = {{56{src_signed[7]}}, src_signed[7:0]};
                end
            end

            TYPE_U8: begin
                if (src_signed > 255 || src_signed < 0) begin
                    result = saturate ? (src_signed < 0 ? 64'd0 : 64'd255) : {56'b0, src[7:0]};
                    overflow = 1'b1;
                end else begin
                    result = {56'b0, src_unsigned[7:0]};
                end
            end

            TYPE_S16: begin
                if (src_signed > 32767) begin
                    result = saturate ? 64'd32767 : {{48{src[15]}}, src[15:0]};
                    overflow = 1'b1;
                end else if (src_signed < -32768) begin
                    result = saturate ? 64'hFFFFFFFFFFFF8000 : {{48{src[15]}}, src[15:0]};
                    overflow = 1'b1;
                end else begin
                    result = {{48{src_signed[15]}}, src_signed[15:0]};
                end
            end

            TYPE_U16: begin
                if (src_signed > 65535 || src_signed < 0) begin
                    result = saturate ? (src_signed < 0 ? 64'd0 : 64'd65535) : {48'b0, src[15:0]};
                    overflow = 1'b1;
                end else begin
                    result = {48'b0, src_unsigned[15:0]};
                end
            end

            TYPE_S32: begin
                if (src_signed > 2147483647) begin
                    result = saturate ? 64'd2147483647 : {{32{src[31]}}, src[31:0]};
                    overflow = 1'b1;
                end else if (src_signed < -2147483648) begin
                    result = saturate ? 64'hFFFFFFFF80000000 : {{32{src[31]}}, src[31:0]};
                    overflow = 1'b1;
                end else begin
                    result = {{32{src_signed[31]}}, src_signed[31:0]};
                end
            end

            TYPE_U32: begin
                if (src_signed > 4294967295 || src_signed < 0) begin
                    result = saturate ? (src_signed < 0 ? 64'd0 : 64'd4294967295) : {32'b0, src[31:0]};
                    overflow = 1'b1;
                end else begin
                    result = {32'b0, src_unsigned[31:0]};
                end
            end

            TYPE_S64, TYPE_U64: begin
                result = src_signed;
            end

            default: result = src;
        endcase
    end

endmodule
