//============================================================================
// RalphGPU - Shared Memory
// 每个SM的高速共享内存，支持32个bank并行访问
// 默认大小: 16KB
//============================================================================

`include "gpu_defines.vh"

module shared_memory #(
    parameter SIZE_KB    = `SHARED_MEM_KB,      // 16KB
    parameter NUM_BANKS  = 32,                   // 32 banks (match warp size)
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = `SHARED_MEM_ADDR_W   // 14 bits for 16KB
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // 并行访问接口 (32个线程同时访问)
    input  wire                     req_valid,
    input  wire                     req_write,      // 0=读, 1=写
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] req_addr,  // 32个地址
    input  wire [NUM_BANKS*DATA_WIDTH-1:0] req_wdata, // 32个写数据
    input  wire [NUM_BANKS-1:0]     req_mask,       // 活跃线程掩码
    output wire                     resp_valid,
    output wire [NUM_BANKS*DATA_WIDTH-1:0] resp_rdata,
    output wire                     bank_conflict   // Bank冲突标志
);

    //------------------------------------------------------------------------
    // 参数计算
    //------------------------------------------------------------------------
    localparam WORDS_PER_BANK = (SIZE_KB * 1024) / NUM_BANKS / (DATA_WIDTH/8);
    localparam BANK_ADDR_W = $clog2(WORDS_PER_BANK);  // 每bank的地址宽度
    localparam BANK_SEL_W = $clog2(NUM_BANKS);        // bank选择宽度 = 5

    //------------------------------------------------------------------------
    // Bank存储
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] bank_mem [0:NUM_BANKS-1][0:WORDS_PER_BANK-1];

    //------------------------------------------------------------------------
    // 地址解析
    // 地址格式: [高位:bank内偏移] [低5位:bank选择]
    // 这样连续地址访问不同bank，避免冲突
    //------------------------------------------------------------------------
    wire [BANK_SEL_W-1:0]  bank_sel   [0:NUM_BANKS-1];
    wire [BANK_ADDR_W-1:0] bank_addr  [0:NUM_BANKS-1];

    genvar lane;
    generate
        for (lane = 0; lane < NUM_BANKS; lane = lane + 1) begin : addr_decode
            wire [ADDR_WIDTH-1:0] lane_addr = req_addr[lane*ADDR_WIDTH +: ADDR_WIDTH];
            // 低5位选bank，高位是bank内地址
            assign bank_sel[lane]  = lane_addr[BANK_SEL_W-1:0];
            assign bank_addr[lane] = lane_addr[ADDR_WIDTH-1:BANK_SEL_W];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Bank冲突检测
    // 当多个线程访问同一个bank时产生冲突
    //------------------------------------------------------------------------
    reg [NUM_BANKS-1:0] bank_access_count [0:NUM_BANKS-1];
    reg has_conflict;

    integer i, j;
    always @(*) begin
        has_conflict = 1'b0;
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            bank_access_count[i] = {NUM_BANKS{1'b0}};
        end

        // 统计每个bank被多少线程访问
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            if (req_mask[i]) begin
                for (j = 0; j < NUM_BANKS; j = j + 1) begin
                    if (bank_sel[i] == j) begin
                        bank_access_count[j] = bank_access_count[j] + 1;
                    end
                end
            end
        end

        // 检查是否有冲突
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            if (bank_access_count[i] > 1) begin
                has_conflict = 1'b1;
            end
        end
    end

    assign bank_conflict = has_conflict & req_valid;

    //------------------------------------------------------------------------
    // 内存访问 (简化：假设无冲突或硬件处理冲突)
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] read_data [0:NUM_BANKS-1];
    reg resp_valid_reg;

    // 读操作
    always @(posedge clk) begin
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            if (req_valid && req_mask[i] && !req_write) begin
                read_data[i] <= bank_mem[bank_sel[i]][bank_addr[i]];
            end
        end
    end

    // 写操作
    always @(posedge clk) begin
        if (req_valid && req_write) begin
            for (i = 0; i < NUM_BANKS; i = i + 1) begin
                if (req_mask[i]) begin
                    bank_mem[bank_sel[i]][bank_addr[i]] <=
                        req_wdata[i*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

    // 响应有效信号
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid_reg <= 1'b0;
        end else begin
            resp_valid_reg <= req_valid & ~req_write;
        end
    end

    assign resp_valid = resp_valid_reg;

    // 输出读数据
    generate
        for (lane = 0; lane < NUM_BANKS; lane = lane + 1) begin : rdata_out
            assign resp_rdata[lane*DATA_WIDTH +: DATA_WIDTH] = read_data[lane];
        end
    endgenerate

    //------------------------------------------------------------------------
    // 初始化 (仿真用)
    //------------------------------------------------------------------------
    integer m, n;
    initial begin
        for (m = 0; m < NUM_BANKS; m = m + 1) begin
            for (n = 0; n < WORDS_PER_BANK; n = n + 1) begin
                bank_mem[m][n] = {DATA_WIDTH{1'b0}};
            end
        end
    end

endmodule
