//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Testbench (Parallel Version)
// 16 threads compute 16 output elements in parallel
//============================================================================

`timescale 1ns / 1ps

module tb_matmul_4x4_fp16_parallel;

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    // Memory addresses
    parameter MATRIX_A = 32'h0000_1000;
    parameter MATRIX_B = 32'h0000_1020;
    parameter MATRIX_C = 32'h0000_1040;

    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT signals
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    wire [AXI_ID_WIDTH-1:0]   m_axi_awid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_awaddr;
    wire [7:0]                m_axi_awlen;
    wire [2:0]                m_axi_awsize;
    wire [1:0]                m_axi_awburst;
    wire                      m_axi_awvalid;
    reg                       m_axi_awready;

    wire [AXI_DATA_WIDTH-1:0]   m_axi_wdata;
    wire [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb;
    wire                        m_axi_wlast;
    wire                        m_axi_wvalid;
    reg                         m_axi_wready;

    reg  [AXI_ID_WIDTH-1:0] m_axi_bid;
    reg  [1:0]              m_axi_bresp;
    reg                     m_axi_bvalid;
    wire                    m_axi_bready;

    wire [AXI_ID_WIDTH-1:0]   m_axi_arid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_araddr;
    wire [7:0]                m_axi_arlen;
    wire [2:0]                m_axi_arsize;
    wire [1:0]                m_axi_arburst;
    wire                      m_axi_arvalid;
    reg                       m_axi_arready;

    reg  [AXI_ID_WIDTH-1:0]   m_axi_rid;
    reg  [AXI_DATA_WIDTH-1:0] m_axi_rdata;
    reg  [1:0]                m_axi_rresp;
    reg                       m_axi_rlast;
    reg                       m_axi_rvalid;
    wire                      m_axi_rready;

    ralph_gpu_top #(
        .NUM_SM(1),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_ID_WIDTH(AXI_ID_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .csr_wr_en(csr_wr_en),
        .csr_addr(csr_addr),
        .csr_wr_data(csr_wr_data),
        .csr_rd_data(csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req(imem_req),
        .imem_addr(imem_addr),
        .imem_data(imem_data),
        .imem_valid(imem_valid),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    // Instruction Memory
    reg [31:0] imem [0:1023];
    integer instr_count;

    initial begin
        for (integer i = 0; i < 1024; i = i + 1) begin
            imem[i] = 32'hFC000000;  // NOP
        end
        $readmemh("matmul_4x4_fp16_parallel.hex", imem);
        instr_count = 0;
        for (integer i = 0; i < 1024; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from hex file", instr_count);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                imem_data <= {imem[imem_addr[11:2] + 1], imem[imem_addr[11:2]]};
                imem_valid <= 1'b1;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // Global Memory
    reg [31:0] gmem [0:16383];
    reg [15:0] matrix_a [0:15];
    reg [15:0] matrix_b [0:15];
    real       matrix_a_real [0:15];
    real       matrix_b_real [0:15];
    real       expected_c [0:15];

    function real fp16_to_real;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real result;
        integer exp_unbiased, i;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            mant = fp16[9:0];
            if (exp == 0 && mant == 0) begin
                fp16_to_real = 0.0;
            end else if (exp == 5'h1F) begin
                fp16_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp - 15;
                result = 1.0 + (mant * 1.0 / 1024.0);
                if (exp_unbiased >= 0) begin
                    for (i = 0; i < exp_unbiased; i = i + 1)
                        result = result * 2.0;
                end else begin
                    for (i = 0; i < -exp_unbiased; i = i + 1)
                        result = result / 2.0;
                end
                fp16_to_real = sign ? -result : result;
            end
        end
    endfunction

    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] mant;
        real result;
        integer exp_unbiased, i;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            mant = fp32[22:0];
            if (exp == 0 && mant == 0) begin
                fp32_to_real = 0.0;
            end else if (exp == 8'hFF) begin
                fp32_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp - 127;
                result = 1.0 + (mant * 1.0 / 8388608.0);
                if (exp_unbiased >= 0) begin
                    for (i = 0; i < exp_unbiased; i = i + 1)
                        result = result * 2.0;
                end else begin
                    for (i = 0; i < -exp_unbiased; i = i + 1)
                        result = result / 2.0;
                end
                fp32_to_real = sign ? -result : result;
            end
        end
    endfunction

    integer i, j, k;
    initial begin
        for (i = 0; i < 16384; i = i + 1) gmem[i] = 32'h0;

        // Initialize A = all -0.5 (0xB800)
        for (i = 0; i < 16; i = i + 1) begin
            matrix_a[i] = 16'hB800;
            matrix_a_real[i] = -0.5;
        end

        // Initialize B = all 3.0 (0x4200)
        for (i = 0; i < 16; i = i + 1) begin
            matrix_b[i] = 16'h4200;
            matrix_b_real[i] = 3.0;
        end

        // Store matrices in memory
        for (i = 0; i < 8; i = i + 1) begin
            gmem[(MATRIX_A >> 2) + i] = {matrix_a[i*2+1], matrix_a[i*2]};
            gmem[(MATRIX_B >> 2) + i] = {matrix_b[i*2+1], matrix_b[i*2]};
        end

        // Compute expected C = A * B
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                expected_c[i*4+j] = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    expected_c[i*4+j] = expected_c[i*4+j] + 
                        matrix_a_real[i*4+k] * matrix_b_real[k*4+j];
                end
            end
        end

        $display("\nMatrix A (FP16, all -0.5):");
        for (i = 0; i < 4; i = i + 1)
            $display("  [%6.2f, %6.2f, %6.2f, %6.2f]",
                matrix_a_real[i*4+0], matrix_a_real[i*4+1],
                matrix_a_real[i*4+2], matrix_a_real[i*4+3]);

        $display("\nMatrix B (FP16, all 3.0):");
        for (i = 0; i < 4; i = i + 1)
            $display("  [%6.2f, %6.2f, %6.2f, %6.2f]",
                matrix_b_real[i*4+0], matrix_b_real[i*4+1],
                matrix_b_real[i*4+2], matrix_b_real[i*4+3]);

        $display("\nExpected C = A * B (FP32):");
        for (i = 0; i < 4; i = i + 1)
            $display("  [%8.2f, %8.2f, %8.2f, %8.2f]",
                expected_c[i*4+0], expected_c[i*4+1],
                expected_c[i*4+2], expected_c[i*4+3]);
    end

    // AXI Memory Model
    reg [31:0] axi_read_addr;
    reg [7:0]  axi_read_len;
    reg [7:0]  axi_read_cnt;
    reg        axi_read_active;

    reg [31:0] axi_write_addr;
    reg        axi_write_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rid <= 0;
            m_axi_rdata <= 0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            m_axi_rvalid <= 1'b0;
            axi_read_active <= 1'b0;
            axi_read_addr <= 0;
            axi_read_len <= 0;
            axi_read_cnt <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready && !axi_read_active) begin
                axi_read_addr <= m_axi_araddr;
                axi_read_len <= m_axi_arlen;
                axi_read_cnt <= 0;
                axi_read_active <= 1'b1;
                m_axi_arready <= 1'b0;
                m_axi_rid <= m_axi_arid;
            end else if (axi_read_active) begin
                m_axi_rdata <= gmem[axi_read_addr[15:2] + axi_read_cnt];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= (axi_read_cnt == axi_read_len);

                if (m_axi_rvalid && m_axi_rready) begin
                    if (axi_read_cnt == axi_read_len) begin
                        axi_read_active <= 1'b0;
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast <= 1'b0;
                        m_axi_arready <= 1'b1;
                    end else begin
                        axi_read_cnt <= axi_read_cnt + 1;
                    end
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bid <= 0;
            m_axi_bresp <= 2'b00;
            m_axi_bvalid <= 1'b0;
            axi_write_active <= 1'b0;
            axi_write_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                axi_write_addr <= m_axi_awaddr;
                axi_write_active <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end

            if (axi_write_active && m_axi_wvalid && m_axi_wready) begin
                gmem[axi_write_addr[15:2]] <= m_axi_wdata;
                if (m_axi_wlast) begin
                    axi_write_active <= 1'b0;
                    m_axi_bvalid <= 1'b1;
                end else begin
                    axi_write_addr <= axi_write_addr + 4;
                end
            end

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // CSR write task
    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr <= addr;
            csr_wr_data <= data;
            csr_wr_en <= 1'b1;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    // Main test
    integer pass_count, fail_count;
    real rtl_val, exp_val, error;
    reg [31:0] start_time, end_time;

    initial begin
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        $display("\n============================================================");
        $display("RalphGPU 4x4 FP16 Matrix Multiplication - PARALLEL (16 threads)");
        $display("============================================================\n");

        // Configure GPU with 16 threads
        $display("Configuring GPU with 16 threads...");
        csr_write(12'h008, 32'd0);        // Kernel PC
        csr_write(12'h00C, 32'd1);        // Grid X = 1
        csr_write(12'h010, 32'd1);        // Grid Y = 1
        csr_write(12'h014, 32'd1);        // Grid Z = 1
        csr_write(12'h018, 32'd16);       // Block X = 16 threads
        csr_write(12'h01C, 32'd1);        // Block Y = 1
        csr_write(12'h020, 32'd1);        // Block Z = 1

        $display("Starting kernel execution...");
        start_time = $time;
        csr_write(12'h004, 32'd1);

        // Wait a bit for kernel to start
        repeat(10) @(posedge clk);

        // Wait for completion (with timeout)
        fork
            begin
                wait(irq_kernel_done);
            end
            begin
                #5_000_000;  // 5ms timeout
                $display("ERROR: Timeout waiting for kernel!");
            end
        join_any
        disable fork;
        end_time = $time;
        $display("Kernel completed!");
        $display("Execution time: %0d ns", (end_time - start_time) / 1000);

        repeat(100) @(posedge clk);

        // Verify results
        $display("\n============================================================");
        $display("Verification Results");
        $display("============================================================\n");

        $display("Computed C matrix (from GPU):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%8.4f, %8.4f, %8.4f, %8.4f]",
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 0]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 1]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 2]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 3]));
        end

        $display("\nElement-by-element comparison:");
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                rtl_val = fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + j]);
                exp_val = expected_c[i*4 + j];
                error = (rtl_val > exp_val) ? (rtl_val - exp_val) : (exp_val - rtl_val);
                if (error < 0.01) begin
                    $display("  C[%0d][%0d]: PASS (RTL=%8.4f, Expected=%8.4f)", i, j, rtl_val, exp_val);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  C[%0d][%0d]: FAIL (RTL=%8.4f, Expected=%8.4f, Error=%8.4f)", i, j, rtl_val, exp_val, error);
                    fail_count = fail_count + 1;
                end
            end
        end

        $display("\n============================================================");
        if (fail_count == 0) begin
            $display("TEST PASSED: All 16 elements match!");
            $display("IPC = %0d instructions / %0d cycles = %.2f",
                instr_count, (end_time - start_time) / 10000,
                instr_count * 1.0 / ((end_time - start_time) / 10000));
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed", pass_count, fail_count);
        end
        $display("============================================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #10_000_000;
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule
