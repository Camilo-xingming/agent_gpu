//============================================================================
// RalphGPU - Shared Memory Debug Testbench
// 专门用于调试 shared memory 操作
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_smem_debug;
    parameter CLK_PERIOD = 10;
    parameter TIMEOUT_CYCLES = 10000;

    reg clk;
    reg rst_n;

    // Instruction memory
    reg [31:0] instruction_mem [0:1023];

    // Global memory
    localparam GMEM_BASE = 32'h0000_0000;
    reg [31:0] global_mem [0:16383];

    // Instruction fetch interface
    wire        imem_req;
    wire [31:0] imem_addr;
    reg         imem_ready;
    reg  [63:0] imem_data;
    reg         imem_valid;
    reg  [31:0] imem_req_addr_d;

    // CSR interface
    reg         csr_wr_en;
    reg [31:0]  csr_addr;
    reg [31:0]  csr_wdata;

    // Memory interface
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
    wire        kernel_done;

    // CSR addresses
    localparam CSR_GPU_CONTROL = 32'h0000_0000;
    localparam CSR_KERNEL_PC   = 32'h0000_0004;
    localparam CSR_GRID_DIM_X  = 32'h0000_0010;
    localparam CSR_GRID_DIM_Y  = 32'h0000_0014;
    localparam CSR_GRID_DIM_Z  = 32'h0000_0018;
    localparam CSR_BLOCK_DIM_X = 32'h0000_001C;
    localparam CSR_BLOCK_DIM_Y = 32'h0000_0020;
    localparam CSR_BLOCK_DIM_Z = 32'h0000_0024;

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // DUT
    ralph_gpu_top #(
        .NUM_SM(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .csr_wr_en(csr_wr_en),
        .csr_addr(csr_addr),
        .csr_wdata(csr_wdata),
        .csr_rdata(),
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
        .m_axi_rready(m_axi_rready),
        .kernel_done(kernel_done)
    );

    // Instruction memory response
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_ready <= 1'b1;
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
            imem_req_addr_d <= 32'b0;
        end else begin
            imem_req_addr_d <= dut.sm_imem_addr[0];
            if (dut.sm_imem_req[0]) begin
                imem_data <= {instruction_mem[(imem_req_addr_d >> 2) + 1],
                              instruction_mem[imem_req_addr_d >> 2]};
                imem_valid <= 1'b1;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // Connect imem to SM
    initial begin
        force dut.sm_imem_ready[0] = imem_ready;
        force dut.sm_imem_data[0] = imem_data;
        force dut.sm_imem_valid[0] = imem_valid;
    end

    // AXI memory model
    reg [31:0] pending_rd_addr;
    reg pending_rd;
    reg [7:0] pending_rd_len;
    reg [2:0] pending_rd_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            pending_rd <= 1'b0;
        end else begin
            // Write handling
            if (m_axi_awvalid && m_axi_awready) begin
                m_axi_awready <= 1'b0;
            end
            if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
                m_axi_bresp <= 2'b00;
                global_mem[(m_axi_awaddr - GMEM_BASE) >> 2] <= m_axi_wdata;
            end
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
                m_axi_awready <= 1'b1;
            end

            // Read handling
            if (m_axi_arvalid && m_axi_arready && !pending_rd) begin
                pending_rd <= 1'b1;
                pending_rd_addr <= m_axi_araddr;
                pending_rd_len <= m_axi_arlen;
                pending_rd_cnt <= 0;
                m_axi_arready <= 1'b0;
            end
            if (pending_rd) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rid <= m_axi_arid;
                m_axi_rresp <= 2'b00;
                m_axi_rdata <= global_mem[(pending_rd_addr - GMEM_BASE) >> 2];
                if (pending_rd_cnt >= pending_rd_len) begin
                    m_axi_rlast <= 1'b1;
                end else begin
                    m_axi_rlast <= 1'b0;
                end
                if (m_axi_rvalid && m_axi_rready) begin
                    if (m_axi_rlast) begin
                        pending_rd <= 1'b0;
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast <= 1'b0;
                        m_axi_arready <= 1'b1;
                    end else begin
                        pending_rd_addr <= pending_rd_addr + 4;
                        pending_rd_cnt <= pending_rd_cnt + 1;
                    end
                end
            end
        end
    end

    // Test
    integer i;
    integer cycle_count;
    reg [31:0] result;

    initial begin
        $dumpfile("tb_smem_debug.vcd");
        $dumpvars(0, tb_smem_debug);

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wdata = 0;

        // Clear memories
        for (i = 0; i < 1024; i = i + 1) instruction_mem[i] = 32'h00000000;
        for (i = 0; i < 16384; i = i + 1) global_mem[i] = 32'h00000000;

        // Load hex file
        $readmemh("sim/test_08_memory_shared.hex", instruction_mem);

        // Debug: print first few instructions
        $display("Loaded instructions:");
        for (i = 0; i < 10; i = i + 1) begin
            $display("  [%0d] 0x%08x", i, instruction_mem[i]);
        end

        // Reset
        #(CLK_PERIOD*10);
        rst_n = 1;
        #(CLK_PERIOD*5);

        // Configure and launch kernel
        @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_GRID_DIM_X;  csr_wdata = 1; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_GRID_DIM_Y;  csr_wdata = 1; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_GRID_DIM_Z;  csr_wdata = 1; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_BLOCK_DIM_X; csr_wdata = 32; @(posedge clk);  // 1 warp only
        csr_wr_en = 1; csr_addr = CSR_BLOCK_DIM_Y; csr_wdata = 1; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_BLOCK_DIM_Z; csr_wdata = 1; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_KERNEL_PC;   csr_wdata = 0; @(posedge clk);
        csr_wr_en = 1; csr_addr = CSR_GPU_CONTROL; csr_wdata = 1; @(posedge clk);
        csr_wr_en = 0;

        $display("Kernel launched, waiting...");

        // Wait for completion
        cycle_count = 0;
        while (!kernel_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        if (kernel_done) begin
            result = global_mem[32'h2000 >> 2];  // Check result at 0x2000
            $display("Kernel completed in %0d cycles", cycle_count);
            $display("Result at 0x2000: 0x%08x", result);
            if (result == 32'hCAFE) begin
                $display("[PASS] Shared memory test passed!");
            end else begin
                $fatal(1, "[FAIL] Shared memory test failed! Expected 0xCAFE, got 0x%08x", result);
            end
        end else begin
            $display("[TIMEOUT] Kernel did not complete in %0d cycles", TIMEOUT_CYCLES);
        end

        #(CLK_PERIOD*10);
        $finish;
    end

endmodule
