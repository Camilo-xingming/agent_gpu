//============================================================================
// RalphGPU - Simple Parallel Test
// 16 threads each write their thread ID to memory
//============================================================================

`timescale 1ns / 1ps

module tb_parallel_test;

    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

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
    reg [31:0] imem [0:255];

    initial begin
        for (integer i = 0; i < 256; i = i + 1)
            imem[i] = 32'hFC000000;
        $readmemh("parallel_test.hex", imem);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
                imem_valid <= 1'b1;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // Global Memory
    reg [31:0] gmem [0:16383];
    integer i;
    initial begin
        for (i = 0; i < 16384; i = i + 1)
            gmem[i] = 32'hDEADBEEF;
    end

    // AXI Read
    reg [31:0] axi_read_addr;
    reg [7:0]  axi_read_len;
    reg [7:0]  axi_read_cnt;
    reg        axi_read_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            axi_read_active <= 1'b0;
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

    // AXI Write
    reg [31:0] axi_write_addr;
    reg        axi_write_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            axi_write_active <= 1'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                axi_write_addr <= m_axi_awaddr;
                axi_write_active <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end
            if (axi_write_active && m_axi_wvalid && m_axi_wready) begin
                gmem[axi_write_addr[15:2]] <= m_axi_wdata;
                $display("[AXI_WRITE] addr=0x%08x data=0x%08x", axi_write_addr, m_axi_wdata);
                if (m_axi_wlast) begin
                    axi_write_active <= 1'b0;
                    m_axi_bvalid <= 1'b1;
                end else begin
                    axi_write_addr <= axi_write_addr + 4;
                end
            end
            if (m_axi_bvalid && m_axi_bready)
                m_axi_bvalid <= 1'b0;
        end
    end

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

    reg [31:0] start_time, end_time;
    integer pass_count;

    initial begin
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        $display("\n============================================================");
        $display("RalphGPU Simple Parallel Test - 16 threads");
        $display("Each thread writes its ID to memory");
        $display("============================================================\n");

        csr_write(12'h008, 32'd0);        // Kernel PC = 0
        csr_write(12'h00C, 32'd1);        // Grid X = 1
        csr_write(12'h010, 32'd1);        // Grid Y = 1
        csr_write(12'h014, 32'd1);        // Grid Z = 1
        csr_write(12'h018, 32'd16);       // Block X = 16
        csr_write(12'h01C, 32'd1);        // Block Y = 1
        csr_write(12'h020, 32'd1);        // Block Z = 1

        $display("Starting kernel with 16 threads...");
        start_time = $time;
        csr_write(12'h004, 32'd1);

        repeat(10) @(posedge clk);
        fork
            wait(irq_kernel_done);
            begin
                #1_000_000;
                $display("ERROR: Timeout!");
            end
        join_any
        disable fork;
        end_time = $time;

        $display("Kernel completed!");
        $display("Execution time: %0d cycles", (end_time - start_time) / CLK_PERIOD);

        repeat(100) @(posedge clk);

        $display("\nMemory contents at 0x1000:");
        pass_count = 0;
        for (i = 0; i < 16; i = i + 1) begin
            if (gmem[(32'h1000 >> 2) + i] == i) begin
                $display("  [0x%04x] = %0d (PASS)", 32'h1000 + i*4, gmem[(32'h1000 >> 2) + i]);
                pass_count = pass_count + 1;
            end else begin
                $display("  [0x%04x] = 0x%08x (FAIL, expected %0d)", 32'h1000 + i*4, gmem[(32'h1000 >> 2) + i], i);
            end
        end

        $display("\n============================================================");
        if (pass_count == 16) begin
            $display("TEST PASSED: All 16 threads wrote correctly!");
            $display("IPC = 5 instructions * 16 threads / %0d cycles = %.2f",
                (end_time - start_time) / CLK_PERIOD,
                80.0 / ((end_time - start_time) / CLK_PERIOD));
        end else begin
            $display("TEST FAILED: Only %0d/16 correct", pass_count);
        end
        $display("============================================================\n");

        $finish;
    end

endmodule
