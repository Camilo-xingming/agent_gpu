//============================================================================
// RalphGPU - Tensor Core
// 矩阵乘累加单元，支持WMMA和MMA指令
// D = A × B + C
// 支持配置: m16n16k16 (FP16), m8n8k4 (FP16), m8n8k32 (INT8)
//============================================================================

`include "gpu_defines.vh"

//============================================================================
// WMMA Matrix Fragment
// 矩阵分片存储，每个Warp协作存储一个矩阵分片
//============================================================================
module wmma_fragment #(
    parameter ROWS = 16,
    parameter COLS = 16,
    parameter ELEM_WIDTH = 16  // FP16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // 加载接口
    input  wire                          load_en,
    input  wire [$clog2(ROWS*COLS)-1:0]  load_idx,
    input  wire [ELEM_WIDTH-1:0]         load_data,

    // 存储接口
    input  wire                          store_en,
    input  wire [$clog2(ROWS*COLS)-1:0]  store_idx,
    output wire [ELEM_WIDTH-1:0]         store_data,

    // 整体读取 (用于计算)
    output wire [ROWS*COLS*ELEM_WIDTH-1:0] fragment_data
);

    // 矩阵存储
    reg [ELEM_WIDTH-1:0] matrix [0:ROWS*COLS-1];

    integer i;

    // 加载
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < ROWS*COLS; i = i + 1) begin
                matrix[i] <= {ELEM_WIDTH{1'b0}};
            end
        end else if (load_en) begin
            matrix[load_idx] <= load_data;
        end
    end

    // 存储输出
    assign store_data = matrix[store_idx];

    // 整体输出
    genvar j;
    generate
        for (j = 0; j < ROWS*COLS; j = j + 1) begin : frag_out
            assign fragment_data[j*ELEM_WIDTH +: ELEM_WIDTH] = matrix[j];
        end
    endgenerate

endmodule


//============================================================================
// FP16 Dot Product Unit
// 4元素FP16点积单元 (基础构建块)
// result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
//============================================================================
module fp16_dot4 (
    input  wire [63:0] a,      // 4x FP16
    input  wire [63:0] b,      // 4x FP16
    input  wire [31:0] c,      // FP32 累加器
    output wire [31:0] result  // FP32 结果
);

    // FP16 乘法结果 (转换为FP32)
    wire [31:0] prod0, prod1, prod2, prod3;

    // FP16 乘法器
    fp16_mul u_mul0 (.a(a[15:0]),  .b(b[15:0]),  .result(prod0));
    fp16_mul u_mul1 (.a(a[31:16]), .b(b[31:16]), .result(prod1));
    fp16_mul u_mul2 (.a(a[47:32]), .b(b[47:32]), .result(prod2));
    fp16_mul u_mul3 (.a(a[63:48]), .b(b[63:48]), .result(prod3));

    // FP32 加法树
    wire [31:0] sum01, sum23, sum0123;

    fp32_add u_add01   (.a(prod0), .b(prod1), .result(sum01));
    fp32_add u_add23   (.a(prod2), .b(prod3), .result(sum23));
    fp32_add u_add0123 (.a(sum01), .b(sum23), .result(sum0123));
    fp32_add u_add_c   (.a(sum0123), .b(c), .result(result));

endmodule


//============================================================================
// FP16 Multiplier (简化版，输出FP32)
//============================================================================
module fp16_mul (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [31:0] result  // FP32
);

    // FP16: sign(1) + exp(5) + man(10)
    wire        sign_a = a[15];
    wire [4:0]  exp_a  = a[14:10];
    wire [9:0]  man_a  = a[9:0];

    wire        sign_b = b[15];
    wire [4:0]  exp_b  = b[14:10];
    wire [9:0]  man_b  = b[9:0];

    // 结果符号
    wire result_sign = sign_a ^ sign_b;

    // 特殊值检测
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == 31) && (man_a == 0);
    wire b_inf  = (exp_b == 31) && (man_b == 0);
    wire a_nan  = (exp_a == 31) && (man_a != 0);
    wire b_nan  = (exp_b == 31) && (man_b != 0);

    // 有效尾数
    wire [10:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [10:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 乘法
    wire [21:0] product = sig_a * sig_b;

    // 指数 (bias 15 -> bias 127)
    wire signed [7:0] exp_sum = (exp_a - 15) + (exp_b - 15) + 127;

    // 规范化
    wire norm_shift = product[21];
    wire [21:0] norm_product = norm_shift ? product : (product << 1);
    wire signed [7:0] norm_exp = norm_shift ? exp_sum + 1 : exp_sum;

    // 截断到FP32尾数
    wire [22:0] result_man = {norm_product[20:0], 2'b0};

    // 输出
    wire [31:0] normal_result = {result_sign, norm_exp[7:0], result_man};

    assign result = (a_nan || b_nan) ? 32'h7FC00000 :
                    ((a_inf && b_zero) || (b_inf && a_zero)) ? 32'h7FC00000 :
                    (a_inf || b_inf) ? {result_sign, 8'hFF, 23'h0} :
                    (a_zero || b_zero) ? {result_sign, 31'h0} :
                    normal_result;

endmodule


//============================================================================
// FP32 Adder (简化版 - 自包含实现)
//============================================================================
module fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);

    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_man  = a[22:0];

    wire        b_sign = b[31];
    wire [7:0]  b_exp  = b[30:23];
    wire [22:0] b_man  = b[22:0];

    // Special cases
    wire a_zero = (a_exp == 0) && (a_man == 0);
    wire b_zero = (b_exp == 0) && (b_man == 0);
    wire a_inf  = (a_exp == 8'hFF) && (a_man == 0);
    wire b_inf  = (b_exp == 8'hFF) && (b_man == 0);
    wire a_nan  = (a_exp == 8'hFF) && (a_man != 0);
    wire b_nan  = (b_exp == 8'hFF) && (b_man != 0);

    // Significands with implicit bit
    wire [23:0] a_sig = (a_exp == 0) ? {1'b0, a_man} : {1'b1, a_man};
    wire [23:0] b_sig = (b_exp == 0) ? {1'b0, b_man} : {1'b1, b_man};

    // Align exponents
    wire a_larger = (a_exp > b_exp) || ((a_exp == b_exp) && (a_sig >= b_sig));
    wire [7:0] exp_diff = a_larger ? (a_exp - b_exp) : (b_exp - a_exp);
    wire [7:0] common_exp = a_larger ? a_exp : b_exp;

    // Shift smaller significand
    wire [23:0] a_aligned = a_larger ? a_sig : (a_sig >> exp_diff);
    wire [23:0] b_aligned = a_larger ? (b_sig >> exp_diff) : b_sig;

    // Add or subtract
    wire same_sign = (a_sign == b_sign);
    wire [24:0] sum = same_sign ? ({1'b0, a_aligned} + {1'b0, b_aligned}) :
                                  (a_larger ? ({1'b0, a_aligned} - {1'b0, b_aligned}) :
                                             ({1'b0, b_aligned} - {1'b0, a_aligned}));

    wire result_sign = same_sign ? a_sign : (a_larger ? a_sign : b_sign);

    // Normalize
    wire [7:0] result_exp = sum[24] ? (common_exp + 1) :
                           (sum[23] ? common_exp : (common_exp - 1));
    wire [22:0] result_man = sum[24] ? sum[23:1] :
                            (sum[23] ? sum[22:0] : {sum[21:0], 1'b0});

    // Handle special cases
    assign result = (a_nan || b_nan) ? 32'h7FC00000 :
                   (a_inf && b_inf && (a_sign != b_sign)) ? 32'h7FC00000 :
                   (a_inf) ? a :
                   (b_inf) ? b :
                   (a_zero) ? b :
                   (b_zero) ? a :
                   (sum == 0) ? 32'h0 :
                   {result_sign, result_exp, result_man};

endmodule

//============================================================================
// FP32 Multiplier (简化版 - 自包含实现)
//============================================================================
module fp32_mul_simple (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);

    function [31:0] fp32_mul_func;
        input [31:0] a_in;
        input [31:0] b_in;
        reg         r_sign;
        reg [8:0]   r_exp;
        reg [23:0]  a_mant;
        reg [23:0]  b_mant;
        reg [47:0]  r_mant_full;
        reg [22:0]  r_mant;
        begin
            r_sign = a_in[31] ^ b_in[31];

            if (a_in[30:23] == 8'b0 || b_in[30:23] == 8'b0) begin
                fp32_mul_func = {r_sign, 31'b0};
            end else begin
                a_mant = {1'b1, a_in[22:0]};
                b_mant = {1'b1, b_in[22:0]};

                r_mant_full = a_mant * b_mant;
                r_exp = a_in[30:23] + b_in[30:23] - 8'd127;

                if (r_mant_full[47]) begin
                    r_mant = r_mant_full[46:24];
                    r_exp = r_exp + 1;
                end else begin
                    r_mant = r_mant_full[45:23];
                end

                if (r_exp >= 9'd255) begin
                    fp32_mul_func = {r_sign, 8'hFF, 23'b0};
                end else if (r_exp[8]) begin
                    fp32_mul_func = {r_sign, 31'b0};
                end else begin
                    fp32_mul_func = {r_sign, r_exp[7:0], r_mant};
                end
            end
        end
    endfunction

    assign result = fp32_mul_func(a, b);

endmodule


//============================================================================
// WMMA MMA Core - 16x16x16 FP16 配置
// 每个时钟周期计算部分结果，多周期完成
//============================================================================
module wmma_mma_16x16x16 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [5:0]  config_mode,    // WMMA配置

    // 矩阵A (16x16 FP16 = 512 bytes, 分段加载)
    input  wire [255:0] a_fragment,    // 当前行 (16 x FP16)
    input  wire [3:0]   a_row,

    // 矩阵B (16x16 FP16 = 512 bytes, 分段加载)
    input  wire [255:0] b_fragment,    // 当前列 (16 x FP16)
    input  wire [3:0]   b_col,

    // 累加器C (16x16 FP32 = 1024 bytes)
    input  wire [511:0] c_fragment,    // 当前行 (16 x FP32)

    // 结果D
    output reg  [511:0] d_fragment,    // 当前行结果 (16 x FP32)
    output reg          done,
    output reg          busy
);

    // 状态机
    localparam IDLE    = 3'd0;
    localparam LOAD_A  = 3'd1;
    localparam LOAD_B  = 3'd2;
    localparam COMPUTE = 3'd3;
    localparam STORE   = 3'd4;

    reg [2:0] state;
    reg [3:0] row_cnt, col_cnt, k_cnt;

    // 矩阵缓存
    reg [255:0] a_cache [0:15];  // 16行
    reg [255:0] b_cache [0:15];  // 16列
    reg [511:0] c_cache [0:15];  // 16行累加器

    // 点积单元输出
    wire [31:0] dot_result [0:15];

    // 16个并行点积单元 (每个处理一个输出元素)
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : dot_units
            // 对于D[row][i]，计算sum(A[row][k] * B[k][i])
            wire [63:0] a_slice = a_cache[row_cnt][k_cnt*64 +: 64];  // 4个FP16
            wire [63:0] b_slice = {b_cache[k_cnt*4+3][i*16 +: 16],
                                   b_cache[k_cnt*4+2][i*16 +: 16],
                                   b_cache[k_cnt*4+1][i*16 +: 16],
                                   b_cache[k_cnt*4+0][i*16 +: 16]};
            wire [31:0] c_val = c_cache[row_cnt][i*32 +: 32];

            fp16_dot4 u_dot (
                .a(a_slice),
                .b(b_slice),
                .c(c_val),
                .result(dot_result[i])
            );
        end
    endgenerate

    // 状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            done       <= 1'b0;
            busy       <= 1'b0;
            row_cnt    <= 4'b0;
            col_cnt    <= 4'b0;
            k_cnt      <= 4'b0;
            d_fragment <= 512'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        row_cnt <= 4'b0;
                        k_cnt   <= 4'b0;
                        state   <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // 每周期计算一行的部分结果
                    if (k_cnt < 4) begin  // 4次迭代完成16次乘累加
                        k_cnt <= k_cnt + 1;
                    end else begin
                        k_cnt <= 4'b0;
                        // 存储当前行结果
                        d_fragment <= {dot_result[15], dot_result[14], dot_result[13], dot_result[12],
                                      dot_result[11], dot_result[10], dot_result[9],  dot_result[8],
                                      dot_result[7],  dot_result[6],  dot_result[5],  dot_result[4],
                                      dot_result[3],  dot_result[2],  dot_result[1],  dot_result[0]};

                        if (row_cnt < 15) begin
                            row_cnt <= row_cnt + 1;
                        end else begin
                            state <= STORE;
                        end
                    end
                end

                STORE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// INT8 Tensor Core - 用于整数矩阵运算
// 支持m8n8k32配置
//============================================================================
module tensor_core_int8 #(
    parameter M = 8,
    parameter N = 8,
    parameter K = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [M*K*8-1:0]  a_matrix,   // M x K INT8
    input  wire [K*N*8-1:0]  b_matrix,   // K x N INT8
    input  wire [M*N*32-1:0] c_matrix,   // M x N INT32 累加器

    output reg  [M*N*32-1:0] d_matrix,   // M x N INT32 结果
    output reg         done,
    output reg         busy
);

    // INT8 点积计算
    // D[i][j] = sum(A[i][k] * B[k][j]) + C[i][j]

    genvar i, j;
    generate
        for (i = 0; i < M; i = i + 1) begin : row_loop
            for (j = 0; j < N; j = j + 1) begin : col_loop
                wire signed [31:0] dp_result;

                // 32元素INT8点积
                int8_dot32 u_dot (
                    .a(a_matrix[i*K*8 +: K*8]),
                    .b(b_matrix[j*8 +: K*8]),  // 需要转置B
                    .c(c_matrix[(i*N+j)*32 +: 32]),
                    .result(dp_result)
                );

                always @(posedge clk) begin
                    if (start) begin
                        d_matrix[(i*N+j)*32 +: 32] <= dp_result;
                    end
                end
            end
        end
    endgenerate

    // 简化状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            if (start) begin
                busy <= 1'b1;
                done <= 1'b0;
            end else if (busy) begin
                done <= 1'b1;
                busy <= 1'b0;
            end else begin
                done <= 1'b0;
            end
        end
    end

endmodule


//============================================================================
// INT8 32-element Dot Product
//============================================================================
module int8_dot32 (
    input  wire [255:0] a,     // 32 x INT8
    input  wire [255:0] b,     // 32 x INT8
    input  wire [31:0]  c,     // INT32 累加器
    output wire [31:0]  result
);

    // 分解为8组4元素点积
    wire signed [31:0] partial [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : dp4
            wire signed [7:0] a0 = a[i*32+7:i*32];
            wire signed [7:0] a1 = a[i*32+15:i*32+8];
            wire signed [7:0] a2 = a[i*32+23:i*32+16];
            wire signed [7:0] a3 = a[i*32+31:i*32+24];

            wire signed [7:0] b0 = b[i*32+7:i*32];
            wire signed [7:0] b1 = b[i*32+15:i*32+8];
            wire signed [7:0] b2 = b[i*32+23:i*32+16];
            wire signed [7:0] b3 = b[i*32+31:i*32+24];

            assign partial[i] = a0*b0 + a1*b1 + a2*b2 + a3*b3;
        end
    endgenerate

    // 加法树
    wire signed [31:0] sum01 = partial[0] + partial[1];
    wire signed [31:0] sum23 = partial[2] + partial[3];
    wire signed [31:0] sum45 = partial[4] + partial[5];
    wire signed [31:0] sum67 = partial[6] + partial[7];
    wire signed [31:0] sum0123 = sum01 + sum23;
    wire signed [31:0] sum4567 = sum45 + sum67;
    wire signed [31:0] sum_all = sum0123 + sum4567;

    assign result = sum_all + c;

endmodule


//============================================================================
// Tensor Core Top Module
// 统一Tensor Core接口
//============================================================================
module tensor_core_top (
    input  wire        clk,
    input  wire        rst_n,

    // 控制
    input  wire        start,
    input  wire [5:0]  mode_cfg,       // WMMA配置
    input  wire [1:0]  data_type,      // 00=FP16, 01=BF16, 10=INT8, 11=INT4

    // 矩阵加载接口
    input  wire        load_a_valid,
    input  wire [255:0] load_a_data,
    input  wire [3:0]  load_a_row,

    input  wire        load_b_valid,
    input  wire [255:0] load_b_data,
    input  wire [3:0]  load_b_col,

    input  wire        load_c_valid,
    input  wire [511:0] load_c_data,
    input  wire [3:0]  load_c_row,

    // 结果存储接口
    output wire [511:0] store_d_data,
    output wire [3:0]  store_d_row,
    output wire        store_d_valid,

    // 状态
    output wire        done,
    output wire        busy
);

    // 实例化WMMA核心
    wmma_mma_16x16x16 u_wmma (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .config_mode(mode_cfg),
        .a_fragment (load_a_data),
        .a_row      (load_a_row),
        .b_fragment (load_b_data),
        .b_col      (load_b_col),
        .c_fragment (load_c_data),
        .d_fragment (store_d_data),
        .done       (done),
        .busy       (busy)
    );

    assign store_d_row = load_c_row;  // 简化
    assign store_d_valid = done;

endmodule


//============================================================================
// Tensor Core Wrapper for SM V2 Integration
// Provides simplified interface for pipeline integration
//============================================================================
module tensor_core #(
    parameter NUM_LANES = 32,
    parameter DATA_WIDTH = 32,
    parameter TC_NUM_CORES = 4,
    parameter TC_LATENCY = 8,
    parameter [3:0] TC_DATA_DEFAULT = `TC_DATA_FP16,
    parameter TC_USE_OP_TYPE = 1,
    parameter [1:0] TC_FP4_FORMAT = `TC_FP4_E2M1,
    parameter [1:0] TC_FP6_FORMAT = `TC_FP6_E3M2,
    parameter [1:0] TC_FP8_FORMAT = `TC_FP8_E4M3
)(
    input  wire        clk,
    input  wire        rst_n,

    // Pipeline interface
    input  wire        op_valid,
    output wire        op_ready,
    input  wire [3:0]  op_type,     // Operation type (extended to 4-bit for FP6)

    // Fragment inputs (from register file)
    input  wire [NUM_LANES*DATA_WIDTH-1:0] frag_a,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] frag_b,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] frag_c,

    // Result output
    output reg         result_valid,
    input  wire        result_ready,
    output reg  [NUM_LANES*DATA_WIDTH-1:0] result_data
);

    localparam integer TC_LATENCY_P = (TC_LATENCY < 1) ? 1 : TC_LATENCY;
    localparam integer TC_COUNT_W = (TC_LATENCY_P > 1) ? $clog2(TC_LATENCY_P + 1) : 1;
    localparam integer TC_CORE_W = (TC_NUM_CORES > 1) ? $clog2(TC_NUM_CORES) : 1;

    function [15:0] fp4_to_fp16;
        input [3:0] fp4;
        input [1:0] format;
        reg        sign;
        reg [2:0]  exp3;
        reg [1:0]  exp2;
        reg        man;
        reg [4:0]  exp16;
        reg [9:0]  man16;
        begin
            sign = fp4[3];

            if (format == `TC_FP4_E3M0) begin
                exp3 = fp4[2:0];
                man16 = 10'b0;
                if (exp3 == 3'b000) begin
                    fp4_to_fp16 = {sign, 15'b0};
                end else if (exp3 == 3'b111) begin
                    fp4_to_fp16 = {sign, 5'h1F, 10'h000};
                end else begin
                    exp16 = (exp3 - 3'd3) + 5'd15;
                    fp4_to_fp16 = {sign, exp16, man16};
                end
            end else begin
                exp2 = fp4[2:1];
                man = fp4[0];

                // Default: FP4 E2M1 with bias=1
                if (exp2 == 2'b00) begin
                    if (man == 1'b0) begin
                        fp4_to_fp16 = {sign, 15'b0};
                    end else begin
                        fp4_to_fp16 = {sign, 5'b00000, {man, 9'b0}};
                    end
                end else if (exp2 == 2'b11) begin
                    fp4_to_fp16 = {sign, 5'h1F, man ? 10'h200 : 10'h000};
                end else begin
                    exp16 = (exp2 - 2'd1) + 5'd15;
                    man16 = {man, 9'b0};
                    fp4_to_fp16 = {sign, exp16, man16};
                end
            end
        end
    endfunction

    function signed [7:0] int4_to_s8;
        input [3:0] val;
        begin
            int4_to_s8 = { {4{val[3]}}, val };
        end
    endfunction

    function [31:0] bf16_to_fp32;
        input [15:0] bf16;
        begin
            bf16_to_fp32 = {bf16, 16'b0};
        end
    endfunction

    function [15:0] fp8_to_fp16;
        input [7:0] fp8;
        input [1:0] format;
        reg        sign;
        reg [4:0]  exp5;
        reg [3:0]  exp4;
        reg [2:0]  man3;
        reg [1:0]  man2;
        reg [4:0]  exp16;
        reg [9:0]  man16;
        begin
            sign = fp8[7];

            if (format == `TC_FP8_E5M2) begin
                exp5 = fp8[6:2];
                man2 = fp8[1:0];
                if (exp5 == 5'b00000) begin
                    if (man2 == 2'b00) begin
                        fp8_to_fp16 = {sign, 15'b0};
                    end else begin
                        fp8_to_fp16 = {sign, 5'b00000, {man2, 8'b0}};
                    end
                end else if (exp5 == 5'b11111) begin
                    fp8_to_fp16 = {sign, 5'h1F, (man2 != 0) ? 10'h200 : 10'h000};
                end else begin
                    exp16 = (exp5 - 5'd15) + 5'd15;
                    man16 = {man2, 8'b0};
                    fp8_to_fp16 = {sign, exp16, man16};
                end
            end else begin
                exp4 = fp8[6:3];
                man3 = fp8[2:0];
                if (exp4 == 4'b0000) begin
                    if (man3 == 3'b000) begin
                        fp8_to_fp16 = {sign, 15'b0};
                    end else begin
                        fp8_to_fp16 = {sign, 5'b00000, {man3, 7'b0}};
                    end
                end else if (exp4 == 4'b1111) begin
                    fp8_to_fp16 = {sign, 5'h1F, (man3 != 0) ? 10'h200 : 10'h000};
                end else begin
                    exp16 = (exp4 - 4'd7) + 5'd15;
                    man16 = {man3, 7'b0};
                    fp8_to_fp16 = {sign, exp16, man16};
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP6 E3M2 to FP16 conversion (5th-gen Tensor Core - Blackwell)
    // Format: S EEE MM (1+3+2 = 6 bits)
    // Exponent bias: 3 (2^(3-1) - 1)
    // Range: ~0.0625 to 7.5
    // Used for efficient weight storage in LLM inference
    //------------------------------------------------------------------------
    function [15:0] fp6_to_fp16;
        input [5:0] fp6;
        input [1:0] format;  // Reserved for future FP6 variants
        reg        sign;
        reg [2:0]  exp6;
        reg [1:0]  man6;
        reg [4:0]  exp16;
        reg [9:0]  man16;
        begin
            sign = fp6[5];
            exp6 = fp6[4:2];
            man6 = fp6[1:0];

            if (exp6 == 3'b000) begin
                if (man6 == 2'b00) begin
                    // Zero
                    fp6_to_fp16 = {sign, 15'b0};
                end else begin
                    // Denormal: treat as small value
                    // For denormals: value = (-1)^s * 0.mm * 2^(1-bias) = 0.mm * 2^(-2)
                    // Map to FP16 denormal or very small normal
                    fp6_to_fp16 = {sign, 5'b00000, {man6, 8'b0}};
                end
            end else if (exp6 == 3'b111) begin
                // Inf/NaN (all 1s exponent)
                fp6_to_fp16 = {sign, 5'h1F, (man6 != 0) ? 10'h200 : 10'h000};
            end else begin
                // Normal number
                // exp_fp16 = exp_fp6 - bias_fp6 + bias_fp16
                // bias_fp6 = 3, bias_fp16 = 15
                // exp_fp16 = exp_fp6 - 3 + 15 = exp_fp6 + 12
                exp16 = {2'b0, exp6} + 5'd12;
                // Mantissa: 2 bits -> 10 bits (shift left 8)
                man16 = {man6, 8'b0};
                fp6_to_fp16 = {sign, exp16, man16};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP6 E3M2 to FP32 conversion (for direct FP32 accumulation)
    // Format: S EEE MM (1+3+2 = 6 bits)
    //------------------------------------------------------------------------
    function [31:0] fp6_to_fp32;
        input [5:0] fp6;
        reg        sign;
        reg [2:0]  exp6;
        reg [1:0]  man6;
        reg [7:0]  exp32;
        reg [22:0] man32;
        begin
            sign = fp6[5];
            exp6 = fp6[4:2];
            man6 = fp6[1:0];

            if (exp6 == 3'b000) begin
                if (man6 == 2'b00) begin
                    // Zero
                    fp6_to_fp32 = {sign, 31'b0};
                end else begin
                    // Denormal: treat as very small value
                    fp6_to_fp32 = {sign, 8'b0, {man6, 21'b0}};
                end
            end else if (exp6 == 3'b111) begin
                // Inf/NaN
                fp6_to_fp32 = {sign, 8'hFF, (man6 != 0) ? 23'h400000 : 23'h0};
            end else begin
                // Normal number
                // exp_fp32 = exp_fp6 - bias_fp6 + bias_fp32
                // bias_fp6 = 3, bias_fp32 = 127
                // exp_fp32 = exp_fp6 - 3 + 127 = exp_fp6 + 124
                exp32 = {5'b0, exp6} + 8'd124;
                // Mantissa: 2 bits -> 23 bits (shift left 21)
                man32 = {man6, 21'b0};
                fp6_to_fp32 = {sign, exp32, man32};
            end
        end
    endfunction

    reg [NUM_LANES*DATA_WIDTH-1:0] slot_a [0:TC_NUM_CORES-1];
    reg [NUM_LANES*DATA_WIDTH-1:0] slot_b [0:TC_NUM_CORES-1];
    reg [NUM_LANES*DATA_WIDTH-1:0] slot_c [0:TC_NUM_CORES-1];
    reg [3:0]                      slot_type [0:TC_NUM_CORES-1];  // Extended to 4-bit for FP6
    reg [TC_COUNT_W-1:0]           slot_count [0:TC_NUM_CORES-1];
    reg                            slot_valid [0:TC_NUM_CORES-1];

    reg                            slot_free;
    reg  [TC_CORE_W-1:0]           slot_free_idx;
    reg                            done_sel_valid;
    reg  [TC_CORE_W-1:0]           done_sel_idx;

    wire [3:0] op_data_type = (TC_USE_OP_TYPE != 0) ? op_type : TC_DATA_DEFAULT;

    wire [TC_NUM_CORES-1:0] slot_done;

    genvar sd;
    generate
        for (sd = 0; sd < TC_NUM_CORES; sd = sd + 1) begin : done_flags
            assign slot_done[sd] = slot_valid[sd] && (slot_count[sd] == 1);
        end
    endgenerate

    integer ds;
    always @(*) begin
        done_sel_valid = 1'b0;
        done_sel_idx = {TC_CORE_W{1'b0}};
        for (ds = 0; ds < TC_NUM_CORES; ds = ds + 1) begin
            if (!done_sel_valid && slot_done[ds]) begin
                done_sel_valid = 1'b1;
                done_sel_idx = ds[TC_CORE_W-1:0];
            end
        end
    end

    integer sf;
    always @(*) begin
        slot_free = 1'b0;
        slot_free_idx = {TC_CORE_W{1'b0}};
        for (sf = 0; sf < TC_NUM_CORES; sf = sf + 1) begin
            if (!slot_free && !slot_valid[sf]) begin
                slot_free = 1'b1;
                slot_free_idx = sf[TC_CORE_W-1:0];
            end
        end
        if (!slot_free && done_sel_valid) begin
            slot_free = 1'b1;
            slot_free_idx = done_sel_idx;
        end
    end

    assign op_ready = slot_free;

    reg [NUM_LANES*DATA_WIDTH-1:0] frag_a_sel;
    reg [NUM_LANES*DATA_WIDTH-1:0] frag_b_sel;
    reg [NUM_LANES*DATA_WIDTH-1:0] frag_c_sel;
    reg [3:0]                      sel_type;  // Extended to 4-bit for FP6

    always @(*) begin
        frag_a_sel = 0;
        frag_b_sel = 0;
        frag_c_sel = 0;
        sel_type = TC_DATA_DEFAULT;
        if (done_sel_valid) begin
            frag_a_sel = slot_a[done_sel_idx];
            frag_b_sel = slot_b[done_sel_idx];
            frag_c_sel = slot_c[done_sel_idx];
            sel_type = slot_type[done_sel_idx];
        end
    end

    wire [1:0] fp4_format_sel = (sel_type == `TC_DATA_FP4_E3M0) ? `TC_FP4_E3M0 :
                                (sel_type == `TC_DATA_FP4_E2M1) ? `TC_FP4_E2M1 :
                                TC_FP4_FORMAT;
    wire [1:0] fp6_format_sel = (sel_type == `TC_DATA_FP6_E3M2) ? `TC_FP6_E3M2 :
                                TC_FP6_FORMAT;
    wire [1:0] fp8_format_sel = (sel_type == `TC_DATA_FP8_E5M2) ? `TC_FP8_E5M2 :
                                (sel_type == `TC_DATA_FP8_E4M3) ? `TC_FP8_E4M3 :
                                TC_FP8_FORMAT;

    // Default integer MAC path (placeholder for FP16/BF16/FP8/FP6)
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_int;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_int8;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_int4;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_fp16;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_bf16;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_fp8;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_fp6;
    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_fp4;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : mma_lanes
            wire [31:0] a32 = frag_a_sel[i*32 +: 32];
            wire [31:0] b32 = frag_b_sel[i*32 +: 32];
            wire [31:0] c32 = frag_c_sel[i*32 +: 32];
            wire [63:0] product = a32 * b32;
            assign mma_result_int[i*32 +: 32] = product[31:0] + c32;

            // INT8 dot4
            wire signed [7:0] a0 = a32[7:0];
            wire signed [7:0] a1 = a32[15:8];
            wire signed [7:0] a2 = a32[23:16];
            wire signed [7:0] a3 = a32[31:24];
            wire signed [7:0] b0 = b32[7:0];
            wire signed [7:0] b1 = b32[15:8];
            wire signed [7:0] b2 = b32[23:16];
            wire signed [7:0] b3 = b32[31:24];
            wire signed [31:0] int8_sum = a0*b0 + a1*b1 + a2*b2 + a3*b3;
            assign mma_result_int8[i*32 +: 32] = int8_sum + $signed(c32);

            // INT4 dot8
            wire signed [7:0] a4_0 = int4_to_s8(a32[3:0]);
            wire signed [7:0] a4_1 = int4_to_s8(a32[7:4]);
            wire signed [7:0] a4_2 = int4_to_s8(a32[11:8]);
            wire signed [7:0] a4_3 = int4_to_s8(a32[15:12]);
            wire signed [7:0] a4_4 = int4_to_s8(a32[19:16]);
            wire signed [7:0] a4_5 = int4_to_s8(a32[23:20]);
            wire signed [7:0] a4_6 = int4_to_s8(a32[27:24]);
            wire signed [7:0] a4_7 = int4_to_s8(a32[31:28]);
            wire signed [7:0] b4_0 = int4_to_s8(b32[3:0]);
            wire signed [7:0] b4_1 = int4_to_s8(b32[7:4]);
            wire signed [7:0] b4_2 = int4_to_s8(b32[11:8]);
            wire signed [7:0] b4_3 = int4_to_s8(b32[15:12]);
            wire signed [7:0] b4_4 = int4_to_s8(b32[19:16]);
            wire signed [7:0] b4_5 = int4_to_s8(b32[23:20]);
            wire signed [7:0] b4_6 = int4_to_s8(b32[27:24]);
            wire signed [7:0] b4_7 = int4_to_s8(b32[31:28]);
            wire signed [31:0] int4_sum = a4_0*b4_0 + a4_1*b4_1 + a4_2*b4_2 +
                                           a4_3*b4_3 + a4_4*b4_4 + a4_5*b4_5 +
                                           a4_6*b4_6 + a4_7*b4_7;
            assign mma_result_int4[i*32 +: 32] = int4_sum + $signed(c32);

            // FP16 dot2
            wire [15:0] a16_0 = a32[15:0];
            wire [15:0] a16_1 = a32[31:16];
            wire [15:0] b16_0 = b32[15:0];
            wire [15:0] b16_1 = b32[31:16];
            wire [31:0] fp16_prod0;
            wire [31:0] fp16_prod1;
            wire [31:0] fp16_sum01;
            wire [31:0] fp16_out;

            fp16_mul u_fp16_mul0 (.a(a16_0), .b(b16_0), .result(fp16_prod0));
            fp16_mul u_fp16_mul1 (.a(a16_1), .b(b16_1), .result(fp16_prod1));
            fp32_add u_fp16_add01 (.a(fp16_prod0), .b(fp16_prod1), .result(fp16_sum01));
            fp32_add u_fp16_add_c (.a(fp16_sum01), .b(c32), .result(fp16_out));

            assign mma_result_fp16[i*32 +: 32] = fp16_out;

            // BF16 dot2 (FP32 accumulate)
            wire [31:0] bf32_a0 = bf16_to_fp32(a32[15:0]);
            wire [31:0] bf32_a1 = bf16_to_fp32(a32[31:16]);
            wire [31:0] bf32_b0 = bf16_to_fp32(b32[15:0]);
            wire [31:0] bf32_b1 = bf16_to_fp32(b32[31:16]);
            wire [31:0] bf_prod0;
            wire [31:0] bf_prod1;
            wire [31:0] bf_sum01;
            wire [31:0] bf_out;

            fp32_mul_simple u_bf_mul0 (.a(bf32_a0), .b(bf32_b0), .result(bf_prod0));
            fp32_mul_simple u_bf_mul1 (.a(bf32_a1), .b(bf32_b1), .result(bf_prod1));
            fp32_add u_bf_add01 (.a(bf_prod0), .b(bf_prod1), .result(bf_sum01));
            fp32_add u_bf_add_c (.a(bf_sum01), .b(c32), .result(bf_out));

            assign mma_result_bf16[i*32 +: 32] = bf_out;

            // FP8 dot4 (FP16 multiply, FP32 accumulate)
            wire [7:0] fp8_a0 = a32[7:0];
            wire [7:0] fp8_a1 = a32[15:8];
            wire [7:0] fp8_a2 = a32[23:16];
            wire [7:0] fp8_a3 = a32[31:24];
            wire [7:0] fp8_b0 = b32[7:0];
            wire [7:0] fp8_b1 = b32[15:8];
            wire [7:0] fp8_b2 = b32[23:16];
            wire [7:0] fp8_b3 = b32[31:24];

            wire [15:0] fp8_a16_0 = fp8_to_fp16(fp8_a0, fp8_format_sel);
            wire [15:0] fp8_a16_1 = fp8_to_fp16(fp8_a1, fp8_format_sel);
            wire [15:0] fp8_a16_2 = fp8_to_fp16(fp8_a2, fp8_format_sel);
            wire [15:0] fp8_a16_3 = fp8_to_fp16(fp8_a3, fp8_format_sel);
            wire [15:0] fp8_b16_0 = fp8_to_fp16(fp8_b0, fp8_format_sel);
            wire [15:0] fp8_b16_1 = fp8_to_fp16(fp8_b1, fp8_format_sel);
            wire [15:0] fp8_b16_2 = fp8_to_fp16(fp8_b2, fp8_format_sel);
            wire [15:0] fp8_b16_3 = fp8_to_fp16(fp8_b3, fp8_format_sel);

            wire [31:0] fp8_prod0;
            wire [31:0] fp8_prod1;
            wire [31:0] fp8_prod2;
            wire [31:0] fp8_prod3;
            wire [31:0] fp8_sum01;
            wire [31:0] fp8_sum23;
            wire [31:0] fp8_sum0123;
            wire [31:0] fp8_out;

            fp16_mul u_fp8_mul0 (.a(fp8_a16_0), .b(fp8_b16_0), .result(fp8_prod0));
            fp16_mul u_fp8_mul1 (.a(fp8_a16_1), .b(fp8_b16_1), .result(fp8_prod1));
            fp16_mul u_fp8_mul2 (.a(fp8_a16_2), .b(fp8_b16_2), .result(fp8_prod2));
            fp16_mul u_fp8_mul3 (.a(fp8_a16_3), .b(fp8_b16_3), .result(fp8_prod3));
            fp32_add u_fp8_add01 (.a(fp8_prod0), .b(fp8_prod1), .result(fp8_sum01));
            fp32_add u_fp8_add23 (.a(fp8_prod2), .b(fp8_prod3), .result(fp8_sum23));
            fp32_add u_fp8_add0123 (.a(fp8_sum01), .b(fp8_sum23), .result(fp8_sum0123));
            fp32_add u_fp8_add_c (.a(fp8_sum0123), .b(c32), .result(fp8_out));

            assign mma_result_fp8[i*32 +: 32] = fp8_out;
        end
    endgenerate

    genvar j;
    generate
        for (j = 0; j < NUM_LANES; j = j + 1) begin : fp4_lanes
            wire [31:0] a32 = frag_a_sel[j*32 +: 32];
            wire [31:0] b32 = frag_b_sel[j*32 +: 32];
            wire [31:0] c32 = frag_c_sel[j*32 +: 32];

            wire [15:0] a_fp4_0 = fp4_to_fp16(a32[3:0], fp4_format_sel);
            wire [15:0] a_fp4_1 = fp4_to_fp16(a32[7:4], fp4_format_sel);
            wire [15:0] a_fp4_2 = fp4_to_fp16(a32[11:8], fp4_format_sel);
            wire [15:0] a_fp4_3 = fp4_to_fp16(a32[15:12], fp4_format_sel);
            wire [15:0] a_fp4_4 = fp4_to_fp16(a32[19:16], fp4_format_sel);
            wire [15:0] a_fp4_5 = fp4_to_fp16(a32[23:20], fp4_format_sel);
            wire [15:0] a_fp4_6 = fp4_to_fp16(a32[27:24], fp4_format_sel);
            wire [15:0] a_fp4_7 = fp4_to_fp16(a32[31:28], fp4_format_sel);

            wire [15:0] b_fp4_0 = fp4_to_fp16(b32[3:0], fp4_format_sel);
            wire [15:0] b_fp4_1 = fp4_to_fp16(b32[7:4], fp4_format_sel);
            wire [15:0] b_fp4_2 = fp4_to_fp16(b32[11:8], fp4_format_sel);
            wire [15:0] b_fp4_3 = fp4_to_fp16(b32[15:12], fp4_format_sel);
            wire [15:0] b_fp4_4 = fp4_to_fp16(b32[19:16], fp4_format_sel);
            wire [15:0] b_fp4_5 = fp4_to_fp16(b32[23:20], fp4_format_sel);
            wire [15:0] b_fp4_6 = fp4_to_fp16(b32[27:24], fp4_format_sel);
            wire [15:0] b_fp4_7 = fp4_to_fp16(b32[31:28], fp4_format_sel);

            wire [31:0] prod0;
            wire [31:0] prod1;
            wire [31:0] prod2;
            wire [31:0] prod3;
            wire [31:0] prod4;
            wire [31:0] prod5;
            wire [31:0] prod6;
            wire [31:0] prod7;

            fp16_mul u_fp4_mul0 (.a(a_fp4_0), .b(b_fp4_0), .result(prod0));
            fp16_mul u_fp4_mul1 (.a(a_fp4_1), .b(b_fp4_1), .result(prod1));
            fp16_mul u_fp4_mul2 (.a(a_fp4_2), .b(b_fp4_2), .result(prod2));
            fp16_mul u_fp4_mul3 (.a(a_fp4_3), .b(b_fp4_3), .result(prod3));
            fp16_mul u_fp4_mul4 (.a(a_fp4_4), .b(b_fp4_4), .result(prod4));
            fp16_mul u_fp4_mul5 (.a(a_fp4_5), .b(b_fp4_5), .result(prod5));
            fp16_mul u_fp4_mul6 (.a(a_fp4_6), .b(b_fp4_6), .result(prod6));
            fp16_mul u_fp4_mul7 (.a(a_fp4_7), .b(b_fp4_7), .result(prod7));

            wire [31:0] sum01;
            wire [31:0] sum23;
            wire [31:0] sum45;
            wire [31:0] sum67;
            wire [31:0] sum0123;
            wire [31:0] sum4567;
            wire [31:0] sum_all;
            wire [31:0] fp4_result;

            fp32_add u_fp4_add01   (.a(prod0),   .b(prod1),   .result(sum01));
            fp32_add u_fp4_add23   (.a(prod2),   .b(prod3),   .result(sum23));
            fp32_add u_fp4_add45   (.a(prod4),   .b(prod5),   .result(sum45));
            fp32_add u_fp4_add67   (.a(prod6),   .b(prod7),   .result(sum67));
            fp32_add u_fp4_add0123 (.a(sum01),   .b(sum23),   .result(sum0123));
            fp32_add u_fp4_add4567 (.a(sum45),   .b(sum67),   .result(sum4567));
            fp32_add u_fp4_add_all (.a(sum0123), .b(sum4567), .result(sum_all));
            fp32_add u_fp4_add_c   (.a(sum_all), .b(c32),     .result(fp4_result));

            assign mma_result_fp4[j*32 +: 32] = fp4_result;
        end
    endgenerate

    //------------------------------------------------------------------------
    // FP6 E3M2 dot5 computation (5th-gen Tensor Core - Blackwell)
    // 32 bits = 5 x 6-bit FP6 values + 2 padding bits
    // Converts FP6 to FP16, multiplies, accumulates in FP32
    //------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < NUM_LANES; k = k + 1) begin : fp6_lanes
            wire [31:0] a32 = frag_a_sel[k*32 +: 32];
            wire [31:0] b32 = frag_b_sel[k*32 +: 32];
            wire [31:0] c32 = frag_c_sel[k*32 +: 32];

            // Extract 5 x 6-bit FP6 values from 32-bit word (30 bits used, 2 bits padding)
            // Layout: [31:30]=padding, [29:24]=fp6_4, [23:18]=fp6_3, [17:12]=fp6_2, [11:6]=fp6_1, [5:0]=fp6_0
            wire [15:0] a_fp6_0 = fp6_to_fp16(a32[5:0], fp6_format_sel);
            wire [15:0] a_fp6_1 = fp6_to_fp16(a32[11:6], fp6_format_sel);
            wire [15:0] a_fp6_2 = fp6_to_fp16(a32[17:12], fp6_format_sel);
            wire [15:0] a_fp6_3 = fp6_to_fp16(a32[23:18], fp6_format_sel);
            wire [15:0] a_fp6_4 = fp6_to_fp16(a32[29:24], fp6_format_sel);

            wire [15:0] b_fp6_0 = fp6_to_fp16(b32[5:0], fp6_format_sel);
            wire [15:0] b_fp6_1 = fp6_to_fp16(b32[11:6], fp6_format_sel);
            wire [15:0] b_fp6_2 = fp6_to_fp16(b32[17:12], fp6_format_sel);
            wire [15:0] b_fp6_3 = fp6_to_fp16(b32[23:18], fp6_format_sel);
            wire [15:0] b_fp6_4 = fp6_to_fp16(b32[29:24], fp6_format_sel);

            // FP16 multiplications (output FP32)
            wire [31:0] fp6_prod0;
            wire [31:0] fp6_prod1;
            wire [31:0] fp6_prod2;
            wire [31:0] fp6_prod3;
            wire [31:0] fp6_prod4;

            fp16_mul u_fp6_mul0 (.a(a_fp6_0), .b(b_fp6_0), .result(fp6_prod0));
            fp16_mul u_fp6_mul1 (.a(a_fp6_1), .b(b_fp6_1), .result(fp6_prod1));
            fp16_mul u_fp6_mul2 (.a(a_fp6_2), .b(b_fp6_2), .result(fp6_prod2));
            fp16_mul u_fp6_mul3 (.a(a_fp6_3), .b(b_fp6_3), .result(fp6_prod3));
            fp16_mul u_fp6_mul4 (.a(a_fp6_4), .b(b_fp6_4), .result(fp6_prod4));

            // FP32 addition tree
            wire [31:0] fp6_sum01;
            wire [31:0] fp6_sum23;
            wire [31:0] fp6_sum0123;
            wire [31:0] fp6_sum01234;
            wire [31:0] fp6_result;

            fp32_add u_fp6_add01   (.a(fp6_prod0), .b(fp6_prod1), .result(fp6_sum01));
            fp32_add u_fp6_add23   (.a(fp6_prod2), .b(fp6_prod3), .result(fp6_sum23));
            fp32_add u_fp6_add0123 (.a(fp6_sum01), .b(fp6_sum23), .result(fp6_sum0123));
            fp32_add u_fp6_add_4   (.a(fp6_sum0123), .b(fp6_prod4), .result(fp6_sum01234));
            fp32_add u_fp6_add_c   (.a(fp6_sum01234), .b(c32), .result(fp6_result));

            assign mma_result_fp6[k*32 +: 32] = fp6_result;
        end
    endgenerate

    wire [NUM_LANES*DATA_WIDTH-1:0] mma_result_sel =
        (sel_type == `TC_DATA_FP6_E3M2) ? mma_result_fp6 :
        (sel_type == `TC_DATA_FP4_E2M1 || sel_type == `TC_DATA_FP4_E3M0) ? mma_result_fp4 :
        (sel_type == `TC_DATA_FP8_E4M3 || sel_type == `TC_DATA_FP8_E5M2) ? mma_result_fp8 :
        (sel_type == `TC_DATA_BF16) ? mma_result_bf16 :
        (sel_type == `TC_DATA_FP16) ? mma_result_fp16 :
        (sel_type == `TC_DATA_INT4) ? mma_result_int4 :
        (sel_type == `TC_DATA_INT8) ? mma_result_int8 :
                                      mma_result_int;

    integer s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_valid <= 1'b0;
            result_data <= 0;
            for (s = 0; s < TC_NUM_CORES; s = s + 1) begin
                slot_valid[s] <= 1'b0;
                slot_count[s] <= {TC_COUNT_W{1'b0}};
                slot_a[s] <= {NUM_LANES*DATA_WIDTH{1'b0}};
                slot_b[s] <= {NUM_LANES*DATA_WIDTH{1'b0}};
                slot_c[s] <= {NUM_LANES*DATA_WIDTH{1'b0}};
                slot_type[s] <= 4'b0;
            end
        end else begin
            // Clear result_valid only when consumer accepts (valid/ready handshake)
            if (result_valid && result_ready) begin
                result_valid <= 1'b0;
            end

            for (s = 0; s < TC_NUM_CORES; s = s + 1) begin
                if (slot_valid[s]) begin
                    if (slot_count[s] > 1) begin
                        slot_count[s] <= slot_count[s] - 1'b1;
                    end else if (slot_count[s] == 1) begin
                        // Only retire slot when result port is free or being consumed
                        if (done_sel_valid && (done_sel_idx == s[TC_CORE_W-1:0]) &&
                            (!result_valid || result_ready)) begin
                            slot_valid[s] <= 1'b0;
                            slot_count[s] <= {TC_COUNT_W{1'b0}};
                        end
                    end
                end
            end

            if (op_valid && slot_free) begin
                slot_valid[slot_free_idx] <= 1'b1;
                slot_count[slot_free_idx] <= TC_LATENCY_P[TC_COUNT_W-1:0];
                slot_a[slot_free_idx] <= frag_a;
                slot_b[slot_free_idx] <= frag_b;
                slot_c[slot_free_idx] <= frag_c;
                slot_type[slot_free_idx] <= op_data_type;
            end

            // Only produce new result when output port is free or being consumed
            if (done_sel_valid && (!result_valid || result_ready)) begin
                result_valid <= 1'b1;
                result_data <= mma_result_sel;
            end
        end
    end

endmodule
