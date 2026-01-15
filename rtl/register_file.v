//============================================================================
// RalphGPU - Register File
// 每个Warp拥有独立的寄存器文件
// 32个线程 × 32个寄存器 × 32位 = 4KB per Warp
//============================================================================

`include "gpu_defines.vh"

module register_file #(
    parameter NUM_REGS   = `NUM_REGS,        // 32 registers per thread
    parameter NUM_LANES  = `THREADS_PER_WARP, // 32 threads per warp
    parameter DATA_WIDTH = `DATA_WIDTH        // 32-bit data
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // 读端口 A (所有线程并行读)
    input  wire [4:0]                   rd_addr_a,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_a,

    // 读端口 B
    input  wire [4:0]                   rd_addr_b,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_b,

    // 读端口 C (用于MAD)
    input  wire [4:0]                   rd_addr_c,
    output wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_c,

    // 写端口
    input  wire                         wr_en,
    input  wire [4:0]                   wr_addr,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] wr_data,
    input  wire [NUM_LANES-1:0]         wr_mask  // 写掩码，只写活跃线程
);

    //------------------------------------------------------------------------
    // 寄存器存储
    // 使用二维数组: [线程][寄存器]
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] regs [0:NUM_LANES-1][0:NUM_REGS-1];

    //------------------------------------------------------------------------
    // 读操作 (组合逻辑)
    //------------------------------------------------------------------------
    genvar lane;
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : read_lanes
            assign rd_data_a[lane*DATA_WIDTH +: DATA_WIDTH] = regs[lane][rd_addr_a];
            assign rd_data_b[lane*DATA_WIDTH +: DATA_WIDTH] = regs[lane][rd_addr_b];
            assign rd_data_c[lane*DATA_WIDTH +: DATA_WIDTH] = regs[lane][rd_addr_c];
        end
    endgenerate

    //------------------------------------------------------------------------
    // 写操作 (时序逻辑)
    //------------------------------------------------------------------------
    integer i, j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位时清零所有寄存器
            for (i = 0; i < NUM_LANES; i = i + 1) begin
                for (j = 0; j < NUM_REGS; j = j + 1) begin
                    regs[i][j] <= {DATA_WIDTH{1'b0}};
                end
            end
        end else if (wr_en) begin
            // 只写被掩码选中的线程
            for (i = 0; i < NUM_LANES; i = i + 1) begin
                if (wr_mask[i]) begin
                    regs[i][wr_addr] <= wr_data[i*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

endmodule


//============================================================================
// 谓词寄存器文件 (Predicate Registers)
// 每个线程8个1-bit谓词寄存器，用于条件执行
//============================================================================
module predicate_regs #(
    parameter NUM_PREDS = 8,
    parameter NUM_LANES = `THREADS_PER_WARP
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // 读端口
    input  wire [2:0]              rd_addr,
    output wire [NUM_LANES-1:0]    rd_data,  // 每线程1-bit

    // 写端口
    input  wire                    wr_en,
    input  wire [2:0]              wr_addr,
    input  wire [NUM_LANES-1:0]    wr_data,
    input  wire [NUM_LANES-1:0]    wr_mask
);

    reg [NUM_PREDS-1:0] preds [0:NUM_LANES-1];

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : pred_lanes
            assign rd_data[i] = preds[i][rd_addr];
        end
    endgenerate

    integer lane;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                preds[lane] <= {NUM_PREDS{1'b0}};
            end
        end else if (wr_en) begin
            for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                if (wr_mask[lane]) begin
                    preds[lane][wr_addr] <= wr_data[lane];
                end
            end
        end
    end

endmodule
