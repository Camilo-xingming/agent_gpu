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
    output wire                     bank_conflict,  // Bank冲突标志

    // Async copy write port (for cp.async from async_copy_engine)
    input  wire                     async_wr_en,
    input  wire [ADDR_WIDTH-1:0]    async_wr_addr,
    input  wire [127:0]             async_wr_data,  // Up to 16 bytes
    input  wire [4:0]               async_wr_size,  // Size in bytes: 4, 8, or 16 (5 bits to hold 16)

    // WGMMA wide read ports (512-bit = 16 words each)
    // These provide high-bandwidth reads for tensor operations
    input  wire                     wgmma_rd_en,        // WGMMA read enable
    input  wire [ADDR_WIDTH-1:0]    wgmma_rd_addr_a,    // Base address for matrix A tile
    input  wire [ADDR_WIDTH-1:0]    wgmma_rd_addr_b,    // Base address for matrix B tile
    output reg  [511:0]             wgmma_rd_data_a,    // 512-bit data for matrix A (16 x 32-bit)
    output reg  [511:0]             wgmma_rd_data_b,    // 512-bit data for matrix B (16 x 32-bit)
    output reg                      wgmma_rd_valid      // Read data valid (1 cycle latency)
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

    // 写操作 (normal lane-parallel writes)
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

    //------------------------------------------------------------------------
    // Async copy write port (for cp.async)
    // Writes 4/8/16 bytes sequentially to consecutive words
    // Address format: byte address, converted to word address internally
    //------------------------------------------------------------------------
    wire [BANK_SEL_W-1:0]  async_bank_sel  = async_wr_addr[BANK_SEL_W-1:0];
    wire [BANK_ADDR_W-1:0] async_bank_addr = async_wr_addr[ADDR_WIDTH-1:BANK_SEL_W];

    // Calculate word addresses for multi-word writes (8B = 2 words, 16B = 4 words)
    wire [ADDR_WIDTH-1:0] async_addr_w0 = async_wr_addr;
    wire [ADDR_WIDTH-1:0] async_addr_w1 = async_wr_addr + 14'd1;
    wire [ADDR_WIDTH-1:0] async_addr_w2 = async_wr_addr + 14'd2;
    wire [ADDR_WIDTH-1:0] async_addr_w3 = async_wr_addr + 14'd3;

    // Bank/addr for each potential word
    wire [BANK_SEL_W-1:0]  async_bank0 = async_addr_w0[BANK_SEL_W-1:0];
    wire [BANK_ADDR_W-1:0] async_baddr0 = async_addr_w0[ADDR_WIDTH-1:BANK_SEL_W];
    wire [BANK_SEL_W-1:0]  async_bank1 = async_addr_w1[BANK_SEL_W-1:0];
    wire [BANK_ADDR_W-1:0] async_baddr1 = async_addr_w1[ADDR_WIDTH-1:BANK_SEL_W];
    wire [BANK_SEL_W-1:0]  async_bank2 = async_addr_w2[BANK_SEL_W-1:0];
    wire [BANK_ADDR_W-1:0] async_baddr2 = async_addr_w2[ADDR_WIDTH-1:BANK_SEL_W];
    wire [BANK_SEL_W-1:0]  async_bank3 = async_addr_w3[BANK_SEL_W-1:0];
    wire [BANK_ADDR_W-1:0] async_baddr3 = async_addr_w3[ADDR_WIDTH-1:BANK_SEL_W];

    always @(posedge clk) begin
        if (async_wr_en) begin
            // Write word 0 (always for size >= 4)
            bank_mem[async_bank0][async_baddr0] <= async_wr_data[31:0];

            // Write word 1 (for size >= 8)
            if (async_wr_size >= 4'd8) begin
                bank_mem[async_bank1][async_baddr1] <= async_wr_data[63:32];
            end

            // Write words 2-3 (for size == 16)
            if (async_wr_size >= 5'd16) begin
                bank_mem[async_bank2][async_baddr2] <= async_wr_data[95:64];
                bank_mem[async_bank3][async_baddr3] <= async_wr_data[127:96];
            end
        end
    end

    // 响应有效信号 (读和写都会产生完成信号)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid_reg <= 1'b0;
        end else begin
            // Both read and write operations produce a valid response
            // (for writes, the read data is undefined but the valid signal matters)
            resp_valid_reg <= req_valid;
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
    // WGMMA Wide Read Port (512-bit = 16 words)
    // Reads 16 consecutive words from shared memory for tensor operations
    // Uses word-aligned addressing (addr is word offset, not byte offset)
    // Data is read from 16 consecutive words starting at base address
    //------------------------------------------------------------------------

    // Calculate word addresses for 16-word reads (matrix A)
    wire [ADDR_WIDTH-1:0] wgmma_addr_a [0:15];
    wire [BANK_SEL_W-1:0] wgmma_bank_a [0:15];
    wire [BANK_ADDR_W-1:0] wgmma_baddr_a [0:15];

    // Calculate word addresses for 16-word reads (matrix B)
    wire [ADDR_WIDTH-1:0] wgmma_addr_b [0:15];
    wire [BANK_SEL_W-1:0] wgmma_bank_b [0:15];
    wire [BANK_ADDR_W-1:0] wgmma_baddr_b [0:15];

    genvar w;
    generate
        for (w = 0; w < 16; w = w + 1) begin : wgmma_addr_gen
            // Matrix A addresses (16 consecutive words)
            assign wgmma_addr_a[w] = wgmma_rd_addr_a + w[ADDR_WIDTH-1:0];
            assign wgmma_bank_a[w] = wgmma_addr_a[w][BANK_SEL_W-1:0];
            assign wgmma_baddr_a[w] = wgmma_addr_a[w][ADDR_WIDTH-1:BANK_SEL_W];

            // Matrix B addresses (16 consecutive words)
            assign wgmma_addr_b[w] = wgmma_rd_addr_b + w[ADDR_WIDTH-1:0];
            assign wgmma_bank_b[w] = wgmma_addr_b[w][BANK_SEL_W-1:0];
            assign wgmma_baddr_b[w] = wgmma_addr_b[w][ADDR_WIDTH-1:BANK_SEL_W];
        end
    endgenerate

    // WGMMA read operation (combinational with registered output)
    integer wgmma_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wgmma_rd_data_a <= 512'b0;
            wgmma_rd_data_b <= 512'b0;
            wgmma_rd_valid <= 1'b0;
        end else begin
            wgmma_rd_valid <= wgmma_rd_en;
            if (wgmma_rd_en) begin
                // Read 16 words for matrix A
                for (wgmma_i = 0; wgmma_i < 16; wgmma_i = wgmma_i + 1) begin
                    wgmma_rd_data_a[wgmma_i*32 +: 32] <= bank_mem[wgmma_bank_a[wgmma_i]][wgmma_baddr_a[wgmma_i]];
                end
                // Read 16 words for matrix B
                for (wgmma_i = 0; wgmma_i < 16; wgmma_i = wgmma_i + 1) begin
                    wgmma_rd_data_b[wgmma_i*32 +: 32] <= bank_mem[wgmma_bank_b[wgmma_i]][wgmma_baddr_b[wgmma_i]];
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Write Collision Detection (Simulation Only)
    // Detect simultaneous normal and async writes to shared memory
    // Async writes have lower priority in hardware, but this warns of conflicts
    //------------------------------------------------------------------------
`ifdef SIMULATION
    always @(posedge clk) begin
        if (req_valid && req_write && async_wr_en) begin
            `ifdef SIMULATION
            $display("WARNING: [%0t] Simultaneous normal and async writes to shared memory", $time);
            `endif
        end
    end
`endif

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
