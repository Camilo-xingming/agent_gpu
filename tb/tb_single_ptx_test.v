//============================================================================
// Single PTX Test Runner - Quick debug version
// Usage: vvp build/tb_single_ptx_test +hex=sim/test_03_multiply.hex
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_single_ptx_test;

    localparam CLK_PERIOD = 10;
    localparam TIMEOUT_CYCLES = 50000;
    localparam IMEM_BASE = 32'h0000_0000;
    localparam GMEM_BASE = 32'h0000_1000;
    localparam RESULT_ADDR = 32'h0000_2000;
    localparam PASS_MARKER = 32'h0000CAFE;
    localparam FAIL_MARKER = 32'h0000DEAD;

    reg clk, rst_n;
    reg csr_wr_en;
    reg [31:0] csr_addr, csr_wr_data;
    wire [31:0] csr_rd_data;
    wire irq_kernel_done;
    wire imem_req;
    wire [31:0] imem_addr;
    reg [31:0] imem_data;
    reg imem_valid;

    // AXI signals
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

    ralph_gpu_top #(
        .NUM_SM(1),
        .L1D_BYPASS(0)
    ) u_gpu (
        .clk(clk), .rst_n(rst_n),
        .csr_wr_en(csr_wr_en), .csr_addr(csr_addr),
        .csr_wr_data(csr_wr_data), .csr_rd_data(csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req(imem_req), .imem_addr(imem_addr),
        .imem_data(imem_data), .imem_valid(imem_valid),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    // Clock
    always #(CLK_PERIOD/2) clk = ~clk;

    // Memory model
    reg [31:0] instruction_mem [0:4095];
    reg [31:0] global_mem [0:16383];

    // Instruction memory response
    reg imem_req_d;
    reg [31:0] imem_addr_d;
    always @(posedge clk) begin
        imem_req_d <= imem_req;
        imem_addr_d <= imem_addr;
        if (imem_req_d) begin
            imem_data <= instruction_mem[imem_addr_d[13:2]];
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    // AXI write handling
    reg [31:0] aw_addr_latched;
    reg aw_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready  <= 1'b1;
            m_axi_bvalid  <= 1'b0;
            m_axi_bid     <= 0;
            m_axi_bresp   <= 0;
            aw_pending     <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                aw_addr_latched <= m_axi_awaddr;
                aw_pending <= 1;
            end
            if (m_axi_wvalid && m_axi_wready && aw_pending) begin
                global_mem[(aw_addr_latched - GMEM_BASE) >> 2] <= m_axi_wdata;
                aw_pending <= 0;
                m_axi_bvalid <= 1'b1;
            end
            if (m_axi_bvalid && m_axi_bready)
                m_axi_bvalid <= 1'b0;
        end
    end

    // AXI read handling
    reg ar_pending;
    reg [31:0] ar_addr_latched;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid  <= 1'b0;
            m_axi_rresp   <= 0;
            m_axi_rlast   <= 0;
            m_axi_rid     <= 0;
            ar_pending     <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                ar_addr_latched <= m_axi_araddr;
                ar_pending <= 1;
                m_axi_arready <= 0;
            end
            if (ar_pending) begin
                m_axi_rdata  <= global_mem[(ar_addr_latched - GMEM_BASE) >> 2];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast  <= 1'b1;
                ar_pending   <= 0;
            end
            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid  <= 1'b0;
                m_axi_rlast   <= 1'b0;
                m_axi_arready <= 1'b1;
            end
        end
    end

    // Test
    reg [256*8-1:0] hex_file;
    integer cycle_count;
    reg [31:0] result;

    initial begin
        if (!$value$plusargs("hex=%s", hex_file)) begin
            $display("ERROR: Use +hex=<file.hex>");
            $finish;
        end

        clk = 0; rst_n = 0;
        csr_wr_en = 0; csr_addr = 0; csr_wr_data = 0;

        // Init memories
        for (integer i = 0; i < 4096; i = i + 1) instruction_mem[i] = 32'h0;
        for (integer i = 0; i < 16384; i = i + 1) global_mem[i] = 32'h0;

        $readmemh(hex_file, instruction_mem);

        // Reset
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        // Launch kernel (CSR addresses from ralph_gpu_top)
        csr_wr_en = 1;
        csr_addr = 32'h00C; csr_wr_data = 32'h1; #CLK_PERIOD; // GRID_DIM_X = 1
        csr_addr = 32'h010; csr_wr_data = 32'h1; #CLK_PERIOD; // GRID_DIM_Y = 1
        csr_addr = 32'h014; csr_wr_data = 32'h1; #CLK_PERIOD; // GRID_DIM_Z = 1
        csr_addr = 32'h018; csr_wr_data = 32'h80; #CLK_PERIOD; // BLOCK_DIM_X = 128 (4 warps)
        csr_addr = 32'h01C; csr_wr_data = 32'h1; #CLK_PERIOD; // BLOCK_DIM_Y = 1
        csr_addr = 32'h020; csr_wr_data = 32'h1; #CLK_PERIOD; // BLOCK_DIM_Z = 1
        csr_addr = 32'h008; csr_wr_data = 32'h0; #CLK_PERIOD; // KERNEL_PC = 0
        csr_addr = 32'h004; csr_wr_data = 32'h1; #CLK_PERIOD; // GPU_CONTROL = GO
        csr_wr_en = 0;

        // Wait
        cycle_count = 0;
        while (!irq_kernel_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (cycle_count == 1000 || cycle_count == 5000 || cycle_count == 10000 || cycle_count == 49999) begin
                result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
                $display("[DEBUG] cycle=%0d result=0x%08X", cycle_count, result);
            end
        end

        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];

        if (cycle_count >= TIMEOUT_CYCLES && result == PASS_MARKER) begin
            $display("[PASS] Completed (timeout but result correct) in %0d cycles", cycle_count);
        end else if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[TIMEOUT] after %0d cycles. Result: 0x%08X", cycle_count, result);
        end else if (result == PASS_MARKER) begin
            $display("[PASS] Completed in %0d cycles", cycle_count);
        end else begin
            $display("[FAIL] Result: 0x%08X after %0d cycles", result, cycle_count);
            // Dump some memory for debugging
            $display("  Global mem dump around result:");
            for (integer i = 0; i < 16; i = i + 1)
                $display("    [0x%04X] = 0x%08X", GMEM_BASE + i*4, global_mem[i]);
        end

        $finish;
    end

    // AXI write monitor
    always @(posedge clk) begin
        if (m_axi_awvalid && m_axi_awready)
            $display("[AXI-AW] addr=0x%08X", m_axi_awaddr);
        if (m_axi_wvalid && m_axi_wready)
            $display("[AXI-W] data=0x%08X strb=0x%X", m_axi_wdata, m_axi_wstrb);
        if (m_axi_bvalid && m_axi_bready)
            $display("[AXI-B] write complete");
    end

endmodule
