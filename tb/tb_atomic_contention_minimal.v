//============================================================================
// Minimal Atomic Contention Testbench
// - Launches 4 warps (128 threads)
// - All threads atomically increment same address
// - Uses existing PTX: test_23_mem_consistency_atomicity
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_atomic_contention_minimal;
    localparam CLK_PERIOD = 10;
    localparam TIMEOUT_CYCLES = 300000;

    localparam GMEM_BASE = 32'h0000_1000;
    localparam RESULT_ADDR = 32'h0000_2000;
    localparam COUNTER_ADDR = 32'h0000_1000;
    localparam PASS_MARKER = 32'h0000CAFE;

    // Clock/reset
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // CSR interface
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    // IMEM
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    // AXI
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

    // DUT
    ralph_gpu_top #(
        .NUM_SM(1),
        .L1D_BYPASS(1)
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

    // Instruction memory
    reg [31:0] imem [0:4095];
    integer imem_i;

    initial begin
        for (imem_i = 0; imem_i < 4096; imem_i = imem_i + 1) begin
            imem[imem_i] = 32'h00000000;
        end
        $readmemh("../hex/ptx_comprehensive_tests/test_23_mem_consistency_atomicity.hex", imem);
    end

    // IMEM response (1-cycle latency)
    reg [31:0] imem_req_addr_d;
    reg        imem_req_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_req_pending <= 1'b0;
            imem_req_addr_d <= 32'b0;
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                imem_req_addr_d <= imem_addr;
                imem_req_pending <= 1'b1;
            end
            if (imem_req_pending) begin
                imem_data <= {imem[(imem_req_addr_d >> 2) + 1], imem[imem_req_addr_d >> 2]};
                imem_valid <= 1'b1;
                imem_req_pending <= 1'b0;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // Global memory model
    reg [31:0] global_mem [0:16383];
    reg [31:0] pending_axi_addr;
    reg        pending_axi_read;
    reg        pending_axi_write;

    integer gmem_i;
    initial begin
        for (gmem_i = 0; gmem_i < 16384; gmem_i = gmem_i + 1) begin
            global_mem[gmem_i] = 32'h0;
        end
    end

    // AXI read channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 32'b0;
            m_axi_rlast <= 1'b0;
            m_axi_rresp <= 2'b00;
            m_axi_rid <= 4'b0;
            pending_axi_read <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_addr <= m_axi_araddr;
                pending_axi_read <= 1'b1;
                m_axi_arready <= 1'b0;
            end else if (pending_axi_read) begin
                m_axi_rdata <= global_mem[(pending_axi_addr - GMEM_BASE) >> 2];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
                pending_axi_read <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end
        end
    end

    // AXI write channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
            pending_axi_write <= 1'b0;
            pending_axi_addr <= 32'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_axi_addr <= m_axi_awaddr;
                pending_axi_write <= 1'b1;
            end

            if (m_axi_wvalid && m_axi_wready && pending_axi_write) begin
                if (pending_axi_addr >= GMEM_BASE) begin
                    global_mem[(pending_axi_addr - GMEM_BASE) >> 2] <= m_axi_wdata;
                end
                pending_axi_write <= 1'b0;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // Helper tasks
    task write_csr;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr = addr;
            csr_wr_data = data;
            csr_wr_en = 1'b1;
            @(posedge clk);
            csr_wr_en = 1'b0;
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            csr_wr_en = 1'b0;
            csr_addr = 12'b0;
            csr_wr_data = 32'b0;
            #(CLK_PERIOD * 10);
            rst_n = 1'b1;
            #(CLK_PERIOD * 5);
        end
    endtask

    integer cycles;
    reg timeout;
    reg [31:0] result;
    reg [31:0] counter;

    initial begin
        $display("=== Atomic Contention Minimal Test ===");
        reset_dut();
        #50;
        $display("Time: %0t - Reset complete", $time);

        // Launch: grid 1x1x1, block 128x1x1 (4 warps)
        @(posedge clk);
        write_csr(12'h00C, 32'd1);    // GRID_DIM_X
        write_csr(12'h010, 32'd1);    // GRID_DIM_Y
        write_csr(12'h014, 32'd1);    // GRID_DIM_Z
        write_csr(12'h018, 32'd128);  // BLOCK_DIM_X
        write_csr(12'h01C, 32'd1);    // BLOCK_DIM_Y
        write_csr(12'h020, 32'd1);    // BLOCK_DIM_Z
        write_csr(12'h008, 32'd0);    // KERNEL_PC
        write_csr(12'h004, 32'd1);    // GPU_CONTROL: start
        $display("Time: %0t - Kernel launched", $time);

        cycles = 0;
        timeout = 0;
        while (!irq_kernel_done && cycles < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (cycles >= TIMEOUT_CYCLES) begin
            timeout = 1'b1;
        end

        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        counter = global_mem[(COUNTER_ADDR - GMEM_BASE) >> 2];

        if (!timeout && result == PASS_MARKER) begin
            $display("PASS: Atomic contention result=0x%08x counter=%0d cycles=%0d", result, counter, cycles);
        end else begin
            $display("FAIL: timeout=%b result=0x%08x counter=%0d cycles=%0d", timeout, result, counter, cycles);
        end

        #100;
        $finish;
    end
endmodule
