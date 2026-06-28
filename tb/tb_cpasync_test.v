//============================================================================
// RalphGPU - cp.async Test
// Tests asynchronous copy and shared memory operations
// Program: Load from global, store to shared, barrier, load from shared, store to global
// Expected: out[tid] = A[tid] = tid * 100
//============================================================================

`timescale 1ns / 1ps

module tb_cpasync_test;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter NUM_THREADS = 32;
    parameter BASE_ADDR = 32'h0000_2000;

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
    reg [31:0] imem [0:255];

    initial begin
        for (integer i = 0; i < 256; i = i + 1) begin
            imem[i] = 32'hFC000000;  // NOP
        end
        $readmemh("test_cpasync.hex", imem);
        $display("Loaded test_cpasync.hex");
    end

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

    // Initialize memory: A[i] = i * 100
    initial begin
        for (integer i = 0; i < 4096; i = i + 1) begin
            if (i < 32)
                gmem[i] = i * 100;  // A[0..31] = 0, 100, 200, ...
            else
                gmem[i] = 32'hDEADBEEF;
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
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

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
    integer i;
    integer pass_count;
    integer fail_count;
    reg [31:0] expected_value;
    reg [31:0] actual_value;

    initial begin
        $display("============================================================");
        $display("RalphGPU cp.async + Shared Memory Test");
        $display("Program: Load global -> Store shared -> Barrier -> Load shared -> Store global");
        $display("Expected: out[tid] = A[tid] = tid * 100");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        pass_count = 0;
        fail_count = 0;

        // Release reset
        #100;
        rst_n = 1;
        #100;

        // Configure kernel
        $display("\nConfiguring kernel...");

        // Set kernel PC = 0
        @(posedge clk);
        csr_addr <= 12'h008;
        csr_wr_data <= 32'h0000_0000;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        // Set grid dims (1,1,1)
        @(posedge clk);
        csr_addr <= 12'h00C;
        csr_wr_data <= 32'h0000_0001;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        @(posedge clk);
        csr_addr <= 12'h010;
        csr_wr_data <= 32'h0000_0001;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        @(posedge clk);
        csr_addr <= 12'h014;
        csr_wr_data <= 32'h0000_0001;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        // Set block dims (32,1,1)
        @(posedge clk);
        csr_addr <= 12'h018;
        csr_wr_data <= 32'h0000_0020;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        // Start kernel
        $display("Starting kernel with 32 threads...");
        @(posedge clk);
        csr_addr <= 12'h004;
        csr_wr_data <= 32'h0000_0001;
        csr_wr_en <= 1;
        @(posedge clk);
        csr_wr_en <= 0;

        // Wait for completion
        fork: wait_done
            begin
                wait(irq_kernel_done);
                $display("Kernel completed!");
                disable wait_done;
            end
            begin
                #500000;
                $display("TIMEOUT waiting for kernel!");
                disable wait_done;
            end
        join

        // Wait for stores to complete
        #5000;

        // Verify results
        $display("\n============================================================");
        $display("Test Results");
        $display("============================================================");
        $display("Total writes: %0d", write_count);

        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            expected_value = i * 100;
            actual_value = gmem[(BASE_ADDR >> 2) + i];
            if (actual_value == expected_value) begin
                pass_count = pass_count + 1;
                if (i < 8) $display("Thread %2d: got %0d, expected %0d - PASS", i, actual_value, expected_value);
            end else begin
                fail_count = fail_count + 1;
                $display("Thread %2d: got %0d (0x%08x), expected %0d - FAIL", i, actual_value, actual_value, expected_value);
            end
        end

        $display("\n============================================================");
        if (fail_count == 0) begin
            $display("TEST PASSED: All %0d threads produced correct results!", pass_count);
            $display("Shared memory load/store and barrier work correctly.");
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed", pass_count, fail_count);
        end
        $display("============================================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout protection
    initial begin
        #1000000;
        $display("ERROR: Global Timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_cpasync_test.vcd");
        $dumpvars(0, tb_cpasync_test);
    end

endmodule
