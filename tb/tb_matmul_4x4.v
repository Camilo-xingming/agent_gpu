//============================================================================
// RalphGPU 4x4 Matrix Multiplication Testbench
// 测试4x4矩阵乘法并统计Cycle数
// C[4x4] = A[4x4] * B[4x4]
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_matmul_4x4;

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
    end

    initial begin
        rst_n = 0;
        #100 rst_n = 1;
    end

    //------------------------------------------------------------------------
    // Cycle计数器
    //------------------------------------------------------------------------
    reg [31:0] cycle_count;
    reg counting;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else if (counting) begin
            cycle_count <= cycle_count + 1;
        end
    end

    //------------------------------------------------------------------------
    // DUT接口
    //------------------------------------------------------------------------
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [31:0] imem_data;
    reg         imem_valid;

    // AXI4 接口
    wire [3:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready;
    reg  [3:0]  m_axi_bid;
    reg  [1:0]  m_axi_bresp;
    reg         m_axi_bvalid;
    wire        m_axi_bready;
    wire [3:0]  m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready;
    reg  [3:0]  m_axi_rid;
    reg  [31:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rlast;
    reg         m_axi_rvalid;
    wire        m_axi_rready;

    //------------------------------------------------------------------------
    // DUT实例化
    //------------------------------------------------------------------------
    ralph_gpu_top dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .csr_wr_en       (csr_wr_en),
        .csr_addr        (csr_addr),
        .csr_wr_data     (csr_wr_data),
        .csr_rd_data     (csr_rd_data),
        .irq_kernel_done (irq_kernel_done),
        .imem_req        (imem_req),
        .imem_addr       (imem_addr),
        .imem_data       (imem_data),
        .imem_valid      (imem_valid),
        .m_axi_awid      (m_axi_awid),
        .m_axi_awaddr    (m_axi_awaddr),
        .m_axi_awlen     (m_axi_awlen),
        .m_axi_awsize    (m_axi_awsize),
        .m_axi_awburst   (m_axi_awburst),
        .m_axi_awvalid   (m_axi_awvalid),
        .m_axi_awready   (m_axi_awready),
        .m_axi_wdata     (m_axi_wdata),
        .m_axi_wstrb     (m_axi_wstrb),
        .m_axi_wlast     (m_axi_wlast),
        .m_axi_wvalid    (m_axi_wvalid),
        .m_axi_wready    (m_axi_wready),
        .m_axi_bid       (m_axi_bid),
        .m_axi_bresp     (m_axi_bresp),
        .m_axi_bvalid    (m_axi_bvalid),
        .m_axi_bready    (m_axi_bready),
        .m_axi_arid      (m_axi_arid),
        .m_axi_araddr    (m_axi_araddr),
        .m_axi_arlen     (m_axi_arlen),
        .m_axi_arsize    (m_axi_arsize),
        .m_axi_arburst   (m_axi_arburst),
        .m_axi_arvalid   (m_axi_arvalid),
        .m_axi_arready   (m_axi_arready),
        .m_axi_rid       (m_axi_rid),
        .m_axi_rdata     (m_axi_rdata),
        .m_axi_rresp     (m_axi_rresp),
        .m_axi_rlast     (m_axi_rlast),
        .m_axi_rvalid    (m_axi_rvalid),
        .m_axi_rready    (m_axi_rready)
    );

    //------------------------------------------------------------------------
    // 指令内存模型
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:1023];

    always @(posedge clk) begin
        imem_valid <= imem_req;
        if (imem_req) begin
            imem_data <= instruction_mem[imem_addr[11:2]];
        end
    end

    //------------------------------------------------------------------------
    // 数据内存模型 (更大的内存用于矩阵)
    //------------------------------------------------------------------------
    reg [31:0] data_memory [0:8191];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1;
            m_axi_wready  <= 1;
            m_axi_bresp   <= 0;
            m_axi_bvalid  <= 0;
            m_axi_arready <= 1;
            m_axi_rvalid  <= 0;
            m_axi_rresp   <= 0;
            m_axi_rlast   <= 0;
        end else begin
            // 写响应
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1;
                m_axi_bid    <= m_axi_awid;
                data_memory[m_axi_awaddr[14:2]] <= m_axi_wdata;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 0;
            end

            // 读响应
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rvalid <= 1;
                m_axi_rid    <= m_axi_arid;
                m_axi_rdata  <= data_memory[m_axi_araddr[14:2]];
                m_axi_rlast  <= 1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
            end
        end
    end

    //------------------------------------------------------------------------
    // CSR写入任务
    //------------------------------------------------------------------------
    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr    <= addr;
            csr_wr_data <= data;
            csr_wr_en   <= 1;
            @(posedge clk);
            csr_wr_en   <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // 矩阵数据
    //------------------------------------------------------------------------
    // Matrix A (4x4) at address 0x0000
    // Matrix B (4x4) at address 0x0100 (256 bytes offset)
    // Matrix C (4x4) at address 0x0200 (512 bytes offset)

    integer i, j, k;
    integer expected_c [0:15];
    integer row, col;

    //------------------------------------------------------------------------
    // 测试程序
    //------------------------------------------------------------------------
    initial begin
        // 初始化
        csr_wr_en   = 0;
        csr_addr    = 0;
        csr_wr_data = 0;
        counting    = 0;

        // 初始化矩阵A (4x4) - 简单测试数据
        // A = [1  2  3  4 ]
        //     [5  6  7  8 ]
        //     [9  10 11 12]
        //     [13 14 15 16]
        data_memory[0]  = 1;  data_memory[1]  = 2;  data_memory[2]  = 3;  data_memory[3]  = 4;
        data_memory[4]  = 5;  data_memory[5]  = 6;  data_memory[6]  = 7;  data_memory[7]  = 8;
        data_memory[8]  = 9;  data_memory[9]  = 10; data_memory[10] = 11; data_memory[11] = 12;
        data_memory[12] = 13; data_memory[13] = 14; data_memory[14] = 15; data_memory[15] = 16;

        // 初始化矩阵B (4x4) at offset 64 (16 words)
        // B = [1  0  0  0]
        //     [0  1  0  0]
        //     [0  0  1  0]
        //     [0  0  0  1]
        // (Identity matrix for easy verification)
        data_memory[64] = 1;  data_memory[65] = 0;  data_memory[66] = 0;  data_memory[67] = 0;
        data_memory[68] = 0;  data_memory[69] = 1;  data_memory[70] = 0;  data_memory[71] = 0;
        data_memory[72] = 0;  data_memory[73] = 0;  data_memory[74] = 1;  data_memory[75] = 0;
        data_memory[76] = 0;  data_memory[77] = 0;  data_memory[78] = 0;  data_memory[79] = 1;

        // 计算期望结果 C = A * B (A * I = A)
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                expected_c[i*4 + j] = 0;
                for (k = 0; k < 4; k = k + 1) begin
                    expected_c[i*4 + j] = expected_c[i*4 + j] +
                        data_memory[i*4 + k] * data_memory[64 + k*4 + j];
                end
            end
        end

        // 4x4矩阵乘法Kernel - 16个线程并行计算
        // 每个线程计算C的一个元素: C[row][col] = sum(A[row][k] * B[k][col])
        //
        // 编码的PTX程序:
        // tid = get_tid()           // 0-15
        // row = tid / 4
        // col = tid % 4
        // sum = 0
        // for k in 0..3:
        //     a_val = A[row*4 + k]  = ld.global [row*16 + k*4]
        //     b_val = B[k*4 + col]  = ld.global [256 + k*16 + col*4]
        //     sum += a_val * b_val
        // C[row*4 + col] = sum      = st.global [512 + tid*4]

        // 简化版: 使用单线程循环计算每个元素 (便于调试)
        // 实际指令需要根据RTL的具体编码格式

        // 指令0: mov.u32 r0, %tid.x    (获取线程ID)
        instruction_mem[0] = {`OP_MOV_SPECIAL, 5'd0, 5'd0, 5'd0, 5'd0, 6'd0};

        // 指令1: mov.u32 r1, 4         (除数)
        instruction_mem[1] = {`OP_ALU, 5'd1, 5'd0, 5'd0, 5'd0, `FUNC_ADD}; // r1 = 0 + 4

        // 指令2-3: 计算row和col
        instruction_mem[2] = {`OP_DIV, 5'd2, 5'd0, 5'd1, 5'd0, 6'd0};  // r2 = tid / 4 = row
        instruction_mem[3] = {`OP_ALU, 5'd3, 5'd0, 5'd1, 5'd0, 6'd0};  // r3 = tid % 4 = col (需要rem)

        // 指令4: 初始化sum=0
        instruction_mem[4] = {`OP_ALU, 5'd10, 5'd0, 5'd0, 5'd0, `FUNC_ADD}; // r10 = 0 (sum)

        // 展开循环 k=0,1,2,3
        // k=0: load A[row][0], load B[0][col], mul, add to sum
        instruction_mem[5]  = {`OP_LD_GLOBAL, 5'd4, 5'd2, 5'd0, 5'd0, 6'd0}; // r4 = A[row*4+0]
        instruction_mem[6]  = {`OP_LD_GLOBAL, 5'd5, 5'd3, 5'd0, 5'd0, 6'd0}; // r5 = B[0*4+col]
        instruction_mem[7]  = {`OP_MUL, 5'd6, 5'd4, 5'd5, 5'd0, 6'd0};       // r6 = r4 * r5
        instruction_mem[8]  = {`OP_ALU, 5'd10, 5'd10, 5'd6, 5'd0, `FUNC_ADD}; // sum += r6

        // k=1
        instruction_mem[9]  = {`OP_LD_GLOBAL, 5'd4, 5'd2, 5'd0, 5'd0, 6'd1}; // A[row][1]
        instruction_mem[10] = {`OP_LD_GLOBAL, 5'd5, 5'd3, 5'd0, 5'd0, 6'd4}; // B[1][col]
        instruction_mem[11] = {`OP_MUL, 5'd6, 5'd4, 5'd5, 5'd0, 6'd0};
        instruction_mem[12] = {`OP_ALU, 5'd10, 5'd10, 5'd6, 5'd0, `FUNC_ADD};

        // k=2
        instruction_mem[13] = {`OP_LD_GLOBAL, 5'd4, 5'd2, 5'd0, 5'd0, 6'd2}; // A[row][2]
        instruction_mem[14] = {`OP_LD_GLOBAL, 5'd5, 5'd3, 5'd0, 5'd0, 6'd8}; // B[2][col]
        instruction_mem[15] = {`OP_MUL, 5'd6, 5'd4, 5'd5, 5'd0, 6'd0};
        instruction_mem[16] = {`OP_ALU, 5'd10, 5'd10, 5'd6, 5'd0, `FUNC_ADD};

        // k=3
        instruction_mem[17] = {`OP_LD_GLOBAL, 5'd4, 5'd2, 5'd0, 5'd0, 6'd3}; // A[row][3]
        instruction_mem[18] = {`OP_LD_GLOBAL, 5'd5, 5'd3, 5'd0, 5'd0, 6'd12}; // B[3][col]
        instruction_mem[19] = {`OP_MUL, 5'd6, 5'd4, 5'd5, 5'd0, 6'd0};
        instruction_mem[20] = {`OP_ALU, 5'd10, 5'd10, 5'd6, 5'd0, `FUNC_ADD};

        // 存储结果 C[tid]
        instruction_mem[21] = {`OP_ST_GLOBAL, 5'd0, 5'd0, 5'd10, 5'd0, 6'd0}; // C[tid] = sum

        // EXIT
        instruction_mem[22] = {`OP_EXIT, 26'd0};

        // 等待复位
        wait(rst_n);
        #200;

        $display("================================================================================");
        $display("RalphGPU 4x4 Matrix Multiplication Test");
        $display("================================================================================");
        $display("");
        $display("Matrix A (4x4):");
        for (row = 0; row < 4; row = row + 1) begin
            $display("  [%3d %3d %3d %3d]",
                data_memory[row*4+0], data_memory[row*4+1],
                data_memory[row*4+2], data_memory[row*4+3]);
        end
        $display("");
        $display("Matrix B (4x4) - Identity:");
        for (row = 0; row < 4; row = row + 1) begin
            $display("  [%3d %3d %3d %3d]",
                data_memory[64+row*4+0], data_memory[64+row*4+1],
                data_memory[64+row*4+2], data_memory[64+row*4+3]);
        end
        $display("");

        // 配置Kernel
        $display("Configuring Kernel...");
        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1
        csr_write(12'h018, 32'h0000_0010);   // BLOCK_DIM_X = 16 (4x4 threads)

        // 启动计数器
        counting = 1;
        cycle_count = 0;

        // 启动Kernel
        $display("Starting Kernel at cycle 0...");
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        // 等待完成或超时
        fork
            begin
                wait(irq_kernel_done);
            end
            begin
                #50000;  // 50us timeout
                $display("WARNING: Timeout waiting for kernel completion");
            end
        join_any
        disable fork;

        // 停止计数
        counting = 0;

        $display("");
        $display("================================================================================");
        $display("RESULTS");
        $display("================================================================================");
        $display("");
        $display("Execution Cycles: %0d", cycle_count);
        $display("");

        // 显示计算结果
        $display("Matrix C = A * B (Computed):");
        for (row = 0; row < 4; row = row + 1) begin
            $display("  [%3d %3d %3d %3d]",
                data_memory[128+row*4+0], data_memory[128+row*4+1],
                data_memory[128+row*4+2], data_memory[128+row*4+3]);
        end
        $display("");

        $display("Expected Result (A * I = A):");
        for (row = 0; row < 4; row = row + 1) begin
            $display("  [%3d %3d %3d %3d]",
                expected_c[row*4+0], expected_c[row*4+1],
                expected_c[row*4+2], expected_c[row*4+3]);
        end
        $display("");

        // 验证结果
        $display("Verification:");
        for (i = 0; i < 16; i = i + 1) begin
            if (data_memory[128+i] === expected_c[i])
                $display("  C[%0d][%0d] = %0d [PASS]", i/4, i%4, data_memory[128+i]);
            else
                $display("  C[%0d][%0d] = %0d (expected %0d) [FAIL]",
                    i/4, i%4, data_memory[128+i], expected_c[i]);
        end

        $display("");
        $display("================================================================================");
        $display("Test Completed - Total Cycles: %0d", cycle_count);
        $display("================================================================================");

        #1000;
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_matmul_4x4.vcd");
        $dumpvars(0, tb_matmul_4x4);
    end

    // 超时保护
    initial begin
        #200000;
        $display("ERROR: Global Timeout!");
        $finish;
    end

endmodule
