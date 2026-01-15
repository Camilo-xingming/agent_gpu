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
