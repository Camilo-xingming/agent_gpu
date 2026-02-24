//============================================================================
// RalphGPU - DP4A/DP2A Top-Level Integration Test
// Tests INT8/INT16 dot product instructions through full GPU pipeline
// Loads program from dp4a_top_test.hex
//============================================================================

`timescale 1ns / 1ps

module tb_dp4a_top;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = 500;

    // Expected results
    // Test 1: DP4A: 1*3 + 2*3 + 0 + 0 + 10 = 19
    parameter EXPECTED_DP4A_1 = 19;
    // Test 2: DP4A: 5*2 + 5*2 + 0 + 0 + 100 = 120
    parameter EXPECTED_DP4A_2 = 120;
    // Test 3: DP2A: 3*4 + 2*5 + 50 = 72
    parameter EXPECTED_DP2A = 72;

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
    // DUT Instantiation - Top Level GPU
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
        $readmemh("../programs/dp4a_top_test.hex", imem);
        // Count instructions
        instr_count = 0;
        for (integer i = 0; i < 256; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("============================================================");
        $display("RalphGPU DP4A/DP2A Top-Level Integration Test");
        $display("============================================================");
        $display("Loaded %0d instructions from programs/dp4a_top_test.hex", instr_count);
        $display("Expected results:");
        $display("  Test 1 (DP4A): %0d", EXPECTED_DP4A_1);
        $display("  Test 2 (DP4A): %0d", EXPECTED_DP4A_2);
        $display("  Test 3 (DP2A): %0d", EXPECTED_DP2A);
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

    // Initialize memory
    initial begin
        for (integer i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'hDEADBEEF;
        end
    end

    // AXI Write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            pending_write_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                gmem[pending_write_addr[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
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
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rid <= m_axi_arid;
                m_axi_rlast <= 1'b1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
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
    // Main Test
    //------------------------------------------------------------------------
    integer cycle_count;
    integer pass_count;
    integer fail_count;
    reg [31:0] result1, result2, result3;

    initial begin
        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        m_axi_bresp = 0;
        m_axi_rresp = 0;
        cycle_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Reset
        #100;
        rst_n = 1;
        #50;

        // Configure kernel
        $display("\nConfiguring kernel...");
        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1
        csr_write(12'h018, 32'h0000_0001);   // BLOCK_DIM_X = 1 (single thread)

        // Start kernel
        $display("Starting kernel...");
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        // Wait for completion
        while (!irq_kernel_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        #100;

        // Check results
        $display("\n============================================================");
        $display("Test Results");
        $display("============================================================");

        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("TIMEOUT after %0d cycles", cycle_count);
            fail_count = fail_count + 3;
        end else begin
            $display("Kernel completed in %0d cycles", cycle_count);

            // Read results from memory (0x1000, 0x1004, 0x1008)
            result1 = gmem[32'h1000 >> 2];
            result2 = gmem[32'h1004 >> 2];
            result3 = gmem[32'h1008 >> 2];

            // Test 1: DP4A
            $display("\nTest 1 (DP4A simple):");
            $display("  Result: %0d, Expected: %0d", result1, EXPECTED_DP4A_1);
            if (result1 == EXPECTED_DP4A_1) begin
                $display("  Status: PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  Status: FAIL");
                fail_count = fail_count + 1;
            end

            // Test 2: DP4A with larger accumulator
            $display("\nTest 2 (DP4A with accumulator):");
            $display("  Result: %0d, Expected: %0d", result2, EXPECTED_DP4A_2);
            if (result2 == EXPECTED_DP4A_2) begin
                $display("  Status: PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  Status: FAIL");
                fail_count = fail_count + 1;
            end

            // Test 3: DP2A
            $display("\nTest 3 (DP2A):");
            $display("  Result: %0d, Expected: %0d", result3, EXPECTED_DP2A);
            if (result3 == EXPECTED_DP2A) begin
                $display("  Status: PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  Status: FAIL");
                fail_count = fail_count + 1;
            end
        end

        // Summary
        $display("\n============================================================");
        $display("SUMMARY: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("TEST PASSED: All DP4A/DP2A operations verified!");
        end else begin
            $display("TEST FAILED: Some operations did not produce expected results");
        end

        $finish;
    end

endmodule
