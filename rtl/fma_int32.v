//============================================================================
// RalphGPU - Integer FMA Unit (Fused Multiply-Add)
// 32位整数融合乘加单元: result = a * b + c
// 4级流水线，与单独MUL延迟相同，但省一条ADD指令
//============================================================================

`timescale 1ns / 1ps

module fma_int32 #(
    parameter NUM_UNITS = 8,           // 8个并行FMA单元
    parameter PIPELINE_STAGES = 4      // 4级流水线
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // 输入操作数 (32线程，但由NUM_UNITS个单元处理)
    input  wire                 valid_in,
    input wire [NUM_UNITS*32-1:0] a,  // 乘数A
    input wire [NUM_UNITS*32-1:0] b,  // 乘数B
    input wire [NUM_UNITS*32-1:0] c,  // 加数C
    input  wire                 is_signed,          // 有符号/无符号

    // 输出结果
    output reg                  valid_out,
    output reg [NUM_UNITS*32-1:0] result
);

    //------------------------------------------------------------------------
    // 流水线寄存器
    //------------------------------------------------------------------------
    // Stage 1: 输入寄存
    reg                 s1_valid;
    reg [31:0]          s1_a [0:NUM_UNITS-1];
    reg [31:0]          s1_b [0:NUM_UNITS-1];
    reg [31:0]          s1_c [0:NUM_UNITS-1];
    reg                 s1_signed;

    // Stage 2: 部分积计算
    reg                 s2_valid;
    reg [63:0]          s2_product [0:NUM_UNITS-1];  // 64位乘积
    reg [31:0]          s2_c [0:NUM_UNITS-1];
    reg                 s2_signed;

    // Stage 3: 加法
    reg                 s3_valid;
    reg [63:0]          s3_sum [0:NUM_UNITS-1];
    reg                 s3_signed;

    // Stage 4: 输出
    reg                 s4_valid;
    reg [31:0]          s4_result [0:NUM_UNITS-1];

    //------------------------------------------------------------------------
    // Stage 1: 输入寄存
    //------------------------------------------------------------------------
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid  <= 0;
            s1_signed <= 0;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s1_a[i] <= 0;
                s1_b[i] <= 0;
                s1_c[i] <= 0;
            end
        end else begin
            s1_valid  <= valid_in;
            s1_signed <= is_signed;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s1_a[i] <= a[i*32 +: 32];
                s1_b[i] <= b[i*32 +: 32];
                s1_c[i] <= c[i*32 +: 32];
            end
        end
    end

    //------------------------------------------------------------------------
    // Stage 2: 乘法 (a * b)
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid  <= 0;
            s2_signed <= 0;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s2_product[i] <= 0;
                s2_c[i] <= 0;
            end
        end else begin
            s2_valid  <= s1_valid;
            s2_signed <= s1_signed;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s2_c[i] <= s1_c[i];
                if (s1_signed) begin
                    // 有符号乘法
                    s2_product[i] <= $signed(s1_a[i]) * $signed(s1_b[i]);
                end else begin
                    // 无符号乘法
                    s2_product[i] <= s1_a[i] * s1_b[i];
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Stage 3: 加法 (product + c)
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid  <= 0;
            s3_signed <= 0;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s3_sum[i] <= 0;
            end
        end else begin
            s3_valid  <= s2_valid;
            s3_signed <= s2_signed;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                if (s2_signed) begin
                    // 有符号加法 (符号扩展c到64位)
                    s3_sum[i] <= $signed(s2_product[i]) + $signed({{32{s2_c[i][31]}}, s2_c[i]});
                end else begin
                    // 无符号加法
                    s3_sum[i] <= s2_product[i] + {32'b0, s2_c[i]};
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Stage 4: 截断到32位
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_valid <= 0;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                s4_result[i] <= 0;
            end
        end else begin
            s4_valid <= s3_valid;
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                // 取低32位作为结果 (与CUDA行为一致)
                s4_result[i] <= s3_sum[i][31:0];
            end
        end
    end

    //------------------------------------------------------------------------
    // 输出
    //------------------------------------------------------------------------
    always @(*) begin
        valid_out = s4_valid;
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            result[i*32 +: 32] = s4_result[i];
        end
    end

endmodule


//============================================================================
// FMA Array - 支持整个Warp (32线程)
//============================================================================
module fma_array #(
    parameter THREADS = 32,
    parameter FMA_UNITS = 8   // 8个FMA单元服务32个线程
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Warp级接口
    input  wire                 valid_in,
    input wire [THREADS*32-1:0] a,
    input wire [THREADS*32-1:0] b,
    input wire [THREADS*32-1:0] c,
    input  wire [THREADS-1:0]   mask,             // 活跃线程掩码
    input  wire                 is_signed,

    output wire                 valid_out,
    output wire [THREADS*32-1:0] result,
    output wire                 ready              // 准备接收新请求
);

    // 时分复用: 32线程分4批处理 (每批8线程)
    localparam BATCHES = THREADS / FMA_UNITS;  // 4批

    reg [1:0]   batch_counter;
    reg         processing;

    // FMA单元输入/输出
    reg         fma_valid;
    reg [FMA_UNITS*32-1:0] fma_a;
    reg [FMA_UNITS*32-1:0] fma_b;
    reg [FMA_UNITS*32-1:0] fma_c;
    wire        fma_out_valid;
    wire [FMA_UNITS*32-1:0] fma_result;

    // 结果缓存
    reg [31:0]  result_cache [0:THREADS-1];
    reg         result_ready;

    // FMA单元实例
    fma_int32 #(
        .NUM_UNITS(FMA_UNITS)
    ) fma_unit (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fma_valid),
        .a(fma_a),
        .b(fma_b),
        .c(fma_c),
        .is_signed(is_signed),
        .valid_out(fma_out_valid),
        .result(fma_result)
    );

    // 状态机
    localparam ST_IDLE = 2'd0;
    localparam ST_PROCESS = 2'd1;
    localparam ST_WAIT = 2'd2;
    localparam ST_DONE = 2'd3;

    reg [1:0] state;
    reg [1:0] wait_counter;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            batch_counter <= 0;
            processing <= 0;
            fma_valid <= 0;
            result_ready <= 0;
            wait_counter <= 0;
            for (i = 0; i < THREADS; i = i + 1) begin
                result_cache[i] <= 0;
            end
            for (i = 0; i < FMA_UNITS; i = i + 1) begin
                fma_a[i*32 +: 32] <= 0;
                fma_b[i*32 +: 32] <= 0;
                fma_c[i*32 +: 32] <= 0;
            end
        end else begin
            fma_valid <= 0;

            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        state <= ST_PROCESS;
                        batch_counter <= 0;
                        result_ready <= 0;
                    end
                end

                ST_PROCESS: begin
                    // 发送一批到FMA
                    fma_valid <= 1;
                    for (i = 0; i < FMA_UNITS; i = i + 1) begin
                        fma_a[i*32 +: 32] <= a[batch_counter * FMA_UNITS + i*32 +: 32];
                        fma_b[i*32 +: 32] <= b[batch_counter * FMA_UNITS + i*32 +: 32];
                        fma_c[i*32 +: 32] <= c[batch_counter * FMA_UNITS + i*32 +: 32];
                    end
                    state <= ST_WAIT;
                    wait_counter <= 0;
                end

                ST_WAIT: begin
                    // 等待FMA结果 (4 cycles)
                    if (fma_out_valid) begin
                        // 保存结果
                        for (i = 0; i < FMA_UNITS; i = i + 1) begin
                            result_cache[batch_counter * FMA_UNITS + i] <= fma_result[i*32 +: 32];
                        end

                        if (batch_counter == BATCHES - 1) begin
                            state <= ST_DONE;
                        end else begin
                            batch_counter <= batch_counter + 1;
                            state <= ST_PROCESS;
                        end
                    end
                end

                ST_DONE: begin
                    result_ready <= 1;
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    // 输出
    assign valid_out = result_ready;
    assign ready = (state == ST_IDLE);

    generate
        genvar g;
        for (g = 0; g < THREADS; g = g + 1) begin : gen_result
            assign result[g*32 +: 32] = result_cache[g];
        end
    endgenerate

endmodule
