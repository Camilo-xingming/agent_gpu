//============================================================================
// RalphGPU Unified Top-Level Testbench
// Self-checking testbench for ralph_gpu_top verification
//
// Features:
// - Loads program from HEX_FILE (passed via -DHEX_FILE="...")
// - Configurable timeout via -DTIMEOUT_CYCLES=N
// - AXI memory model with read/write support
// - Automatic PASS/FAIL reporting
//============================================================================

`timescale 1ns / 1ps

`include "gpu_defines.vh"

`ifndef HEX_FILE
`define HEX_FILE "test.hex"
`endif

`ifndef TIMEOUT_CYCLES
`define TIMEOUT_CYCLES 1000
`endif

module tb_top_level_unified;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = `TIMEOUT_CYCLES;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    // AXI signals
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

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
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

    //------------------------------------------------------------------------
    // Instruction Memory
    //------------------------------------------------------------------------
    reg [31:0] imem [0:1023];
    integer instr_count;

    initial begin
        // Initialize with NOPs
        for (integer i = 0; i < 1024; i = i + 1) begin
            imem[i] = {`OP_NOP, 26'd0};
        end

        // Load program from hex file
        $readmemh(`HEX_FILE, imem);

        // Count non-NOP instructions
        instr_count = 0;
        for (integer i = 0; i < 1024; i = i + 1) begin
            if (imem[i][31:26] != `OP_NOP) instr_count = instr_count + 1;
        end

        $display("============================================================");
        $display("RalphGPU Top-Level Unified Test");
        $display("============================================================");
        $display("Program: %s", `HEX_FILE);
        $display("Instructions loaded: %0d", instr_count);
        $display("Timeout: %0d cycles", TIMEOUT_CYCLES);
    end

    // Instruction memory response - 1 cycle latency, 64-bit (2 instructions)
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[(imem_addr >> 2) + 1], imem[imem_addr >> 2]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Data Memory (AXI Slave Model)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:16383];  // 64KB
    reg [31:0] pending_write_addr;
    reg        pending_read;
    reg [31:0] pending_read_addr;
    reg [AXI_ID_WIDTH-1:0] pending_read_id;

    // Initialize memory
    initial begin
        for (integer i = 0; i < 16384; i = i + 1) begin
            gmem[i] = 32'h0;
        end
    end

    // AXI Write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            pending_write_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                gmem[pending_write_addr[15:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
                $display("  MEM WRITE: addr=0x%08h data=0x%08h", pending_write_addr, m_axi_wdata);
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            m_axi_rresp <= 2'b00;
            pending_read <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                pending_read <= 1'b1;
                pending_read_addr <= m_axi_araddr;
                pending_read_id <= m_axi_arid;
                m_axi_arready <= 1'b0;
            end else if (pending_read) begin
                m_axi_rdata <= gmem[pending_read_addr[15:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rid <= pending_read_id;
                m_axi_rlast <= 1'b1;
                pending_read <= 1'b0;
                $display("  MEM READ: addr=0x%08h data=0x%08h", pending_read_addr, gmem[pending_read_addr[15:2]]);
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // CSR Write Task
    //------------------------------------------------------------------------
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

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer cycle_count;
    reg test_passed;

    initial begin
        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        cycle_count = 0;
        test_passed = 0;

        // Reset sequence
        #100;
        rst_n = 1;
        #50;

        $display("\n--- Configuring Kernel ---");

        // Configure kernel parameters
        csr_write(12'h008, 32'h0000_0000);  // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);  // GRID_DIM_X = 1
        csr_write(12'h010, 32'h0000_0001);  // GRID_DIM_Y = 1
        csr_write(12'h014, 32'h0000_0001);  // GRID_DIM_Z = 1
        csr_write(12'h018, 32'h0000_0001);  // BLOCK_DIM_X = 1 (single thread)
        csr_write(12'h01C, 32'h0000_0001);  // BLOCK_DIM_Y = 1
        csr_write(12'h020, 32'h0000_0001);  // BLOCK_DIM_Z = 1

        $display("--- Starting Kernel ---");
        csr_write(12'h004, 32'h0000_0001);  // GPU_CONTROL.start = 1

        // Wait for completion with timeout
        while (!irq_kernel_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        #100;

        $display("\n============================================================");
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("TIMEOUT after %0d cycles", cycle_count);
            $display("TEST FAILED");
            test_passed = 0;
        end else begin
            $display("Kernel completed in %0d cycles", cycle_count);

            // Dump memory contents at key locations
            $display("\n--- Memory Contents ---");
            $display("  [0x1000] = 0x%08h", gmem[32'h1000 >> 2]);
            $display("  [0x1004] = 0x%08h", gmem[32'h1004 >> 2]);
            $display("  [0x1008] = 0x%08h", gmem[32'h1008 >> 2]);
            $display("  [0x100C] = 0x%08h", gmem[32'h100C >> 2]);
            $display("  [0x2000] = 0x%08h", gmem[32'h2000 >> 2]);

            // Basic pass criteria: kernel completed without timeout
            $display("\nTEST PASSED");
            test_passed = 1;
        end
        $display("============================================================");

        #100;
        $finish;
    end

    //------------------------------------------------------------------------
    // Watchdog Timer
    //------------------------------------------------------------------------
    initial begin
        #(TIMEOUT_CYCLES * CLK_PERIOD * 2);
        $display("WATCHDOG TIMEOUT");
        $finish;
    end

endmodule
