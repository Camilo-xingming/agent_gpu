//============================================================================
// RalphGPU - Divergence Test
// Tests SIMT divergence and reconvergence
// Program: if (tid & 1) result = 200; else result = 100;
// Expected: even threads -> 100, odd threads -> 200
// Loads program from divergence_test.hex
//============================================================================

`timescale 1ns / 1ps

module tb_divergence_test;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter NUM_THREADS = 32;
    parameter BASE_ADDR = 32'h0000_1000;

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
    // Instruction Memory - Load from hex file
    //------------------------------------------------------------------------
    reg [31:0] imem [0:255];
    integer instr_count;

    initial begin
        // Initialize with NOPs
        for (integer i = 0; i < 256; i = i + 1) begin
            imem[i] = 32'hFC000000;  // NOP/EXIT opcode
        end
        // Load program from hex file
        $readmemh("divergence_test.hex", imem);
        // Count instructions
        instr_count = 0;
        for (integer i = 0; i < 256; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from divergence_test.hex", instr_count);
        $display("Testing divergent branch: even threads -> 100, odd threads -> 200");
    end

    //------------------------------------------------------------------------
    // Instruction Memory Response
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Data Memory (AXI)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_write_addr;
    reg [31:0] write_count;

    // Initialize memory
    initial begin
        for (integer i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'hDEADBEEF;  // Pattern to detect unwritten locations
        end
        write_count = 0;
    end

    // AXI Write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            pending_write_addr <= 0;
        end else begin
            // Address phase
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            // Data phase
            if (m_axi_wvalid && m_axi_wready) begin
                gmem[pending_write_addr[13:2]] <= m_axi_wdata;
                write_count <= write_count + 1;
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bready && m_axi_bvalid) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
            end else if (m_axi_rready && m_axi_rvalid) begin
                m_axi_rvalid <= 1'b0;
            end
        end
    end

    assign m_axi_bresp = 2'b00;
    assign m_axi_rresp = 2'b00;

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    integer cycle_count;
    integer i;
    integer pass_count;
    integer fail_count;
    reg [31:0] expected_value;
    reg [31:0] actual_value;

    initial begin
        $display("\n============================================================");
        $display("RalphGPU Divergence Test");
        $display("Testing: if (tid & 1) result=200; else result=100;");
        $display("Expected: even threads=100, odd threads=200");
        $display("============================================================\n");

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        cycle_count = 0;

        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        // Configure kernel: 32 threads (1 warp)
        csr_addr = 12'h018;  // Block dim X
        csr_wr_data = NUM_THREADS;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h01C;  // Block dim Y
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h020;  // Block dim Z
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Set kernel PC = 0
        csr_addr = 12'h008;
        csr_wr_data = 0;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Start kernel
        $display("Starting kernel with %0d threads...\n", NUM_THREADS);
        csr_addr = 12'h004;
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;

        // Wait for completion
        while (!irq_kernel_done && cycle_count < 20000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        // Wait for memory operations to complete
        repeat(200) @(posedge clk);

        // Check results
        $display("\n============================================================");
        $display("Test Results");
        $display("============================================================");
        $display("Kernel completed in %0d cycles", cycle_count);
        $display("Total writes: %0d", write_count);
        $display("");

        pass_count = 0;
        fail_count = 0;

        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            actual_value = gmem[(BASE_ADDR >> 2) + i];
            expected_value = (i & 1) ? 200 : 100;  // Odd -> 200, Even -> 100

            if (actual_value == expected_value) begin
                pass_count = pass_count + 1;
                $display("Thread %2d: got %3d, expected %3d - PASS",
                         i, actual_value, expected_value);
            end else begin
                fail_count = fail_count + 1;
                $display("Thread %2d: got %3d (0x%08x), expected %3d - FAIL",
                         i, actual_value, actual_value, expected_value);
            end
        end

        $display("");
        $display("============================================================");
        if (fail_count == 0) begin
            $display("TEST PASSED: All %0d threads produced correct results!", pass_count);
            $display("Divergence and reconvergence worked correctly.");
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed", pass_count, fail_count);
        end
        $display("============================================================\n");

        $finish;
    end

endmodule
