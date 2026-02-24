//============================================================================
// Multi-Operation Top-Level Test
// Tests ALU, MUL, VIDEO (dp4a), and Memory operations
//============================================================================

`timescale 1ns / 1ps

module tb_multi_op_test;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = 500;

    // Expected results
    parameter EXPECTED_ADD  = 15;   // 10 + 5
    parameter EXPECTED_SUB  = 13;   // 20 - 7
    parameter EXPECTED_MUL  = 42;   // 6 * 7
    parameter EXPECTED_DP4A = 19;   // 1*3 + 2*3 + 10
    parameter EXPECTED_AND  = 15;   // 0xFF & 0x0F
    parameter EXPECTED_OR   = 255;  // 0xF0 | 0x0F

    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT Signals
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

    // DUT
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
        for (integer i = 0; i < 256; i = i + 1) begin
            imem[i] = 32'hFC000000;
        end
        $readmemh("../programs/multi_op_test.hex", imem);
        $display("============================================================");
        $display("Multi-Operation Top-Level Test");
        $display("Testing: ALU ADD/SUB/AND/OR, MUL, VIDEO DP4A");
        $display("============================================================");
    end

    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    // Data Memory
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_write_addr;

    initial begin
        for (integer i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'hDEADBEEF;
        end
    end

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

    integer cycle_count;
    integer pass_count;
    reg [31:0] result_add, result_sub, result_mul, result_dp4a, result_and, result_or;

    initial begin
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        m_axi_bresp = 0;
        m_axi_rresp = 0;
        cycle_count = 0;
        pass_count = 0;

        #100;
        rst_n = 1;
        #50;

        $display("Configuring and starting kernel...");
        csr_write(12'h008, 32'h0000_0000);  // Kernel PC
        csr_write(12'h00C, 32'h0000_0001);  // Block dim X
        csr_write(12'h018, 32'h0000_0001);  // Grid dim X
        csr_write(12'h004, 32'h0000_0001);  // Start

        while (!irq_kernel_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        #200;  // Extra time for memory writes

        $display("\n============================================================");
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("TIMEOUT after %0d cycles", cycle_count);
            $display("TEST FAILED");
        end else begin
            $display("Kernel completed in %0d cycles", cycle_count);
            $display("");

            // Check results
            result_add  = gmem[32'h1000 >> 2];
            result_sub  = gmem[32'h1004 >> 2];
            result_mul  = gmem[32'h1008 >> 2];
            result_dp4a = gmem[32'h100C >> 2];
            result_and  = gmem[32'h1010 >> 2];
            result_or   = gmem[32'h1014 >> 2];

            $display("Test Results:");
            $display("-----------------------------------------------------------");

            // Test 1: ADD
            if (result_add == EXPECTED_ADD) begin
                $display("[PASS] ALU ADD:  10 + 5 = %0d (expected %0d)", result_add, EXPECTED_ADD);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] ALU ADD:  10 + 5 = %0d (expected %0d)", result_add, EXPECTED_ADD);
            end

            // Test 2: SUB
            if (result_sub == EXPECTED_SUB) begin
                $display("[PASS] ALU SUB:  20 - 7 = %0d (expected %0d)", result_sub, EXPECTED_SUB);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] ALU SUB:  20 - 7 = %0d (expected %0d)", result_sub, EXPECTED_SUB);
            end

            // Test 3: MUL
            if (result_mul == EXPECTED_MUL) begin
                $display("[PASS] MUL:      6 * 7 = %0d (expected %0d)", result_mul, EXPECTED_MUL);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] MUL:      6 * 7 = %0d (expected %0d)", result_mul, EXPECTED_MUL);
            end

            // Test 4: DP4A
            if (result_dp4a == EXPECTED_DP4A) begin
                $display("[PASS] DP4A:     [1,2]*[3,3]+10 = %0d (expected %0d)", result_dp4a, EXPECTED_DP4A);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] DP4A:     [1,2]*[3,3]+10 = %0d (expected %0d)", result_dp4a, EXPECTED_DP4A);
            end

            // Test 5: AND
            if (result_and == EXPECTED_AND) begin
                $display("[PASS] ALU AND:  0xFF & 0x0F = %0d (expected %0d)", result_and, EXPECTED_AND);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] ALU AND:  0xFF & 0x0F = %0d (expected %0d)", result_and, EXPECTED_AND);
            end

            // Test 6: OR
            if (result_or == EXPECTED_OR) begin
                $display("[PASS] ALU OR:   0xF0 | 0x0F = %0d (expected %0d)", result_or, EXPECTED_OR);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] ALU OR:   0xF0 | 0x0F = %0d (expected %0d)", result_or, EXPECTED_OR);
            end

            $display("-----------------------------------------------------------");
            $display("");
            if (pass_count == 6) begin
                $display("ALL TESTS PASSED (%0d/6)", pass_count);
            end else begin
                $display("SOME TESTS FAILED: %0d/6 passed", pass_count);
            end
        end
        $display("============================================================");
        $finish;
    end

endmodule
