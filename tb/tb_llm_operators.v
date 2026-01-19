//============================================================================
// LLM Operators Top-Level Verification Testbench
// Tests common LLM operators using ralph_gpu_top as DUT
//============================================================================

`timescale 1ns / 1ps

module tb_llm_operators;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = 1000;

    // Test parameters - loaded dynamically
    reg [255:0] test_name;
    reg [31:0] expected_result;
    reg [31:0] expected_results [0:15];
    integer num_results;

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

    task init_memory;
        integer i;
        begin
            for (i = 0; i < 256; i = i + 1) begin
                imem[i] = 32'hFC000000;  // NOP/invalid
            end
            for (i = 0; i < 4096; i = i + 1) begin
                gmem[i] = 32'hDEADBEEF;
            end
        end
    endtask

    task run_kernel;
        input integer timeout;
        output integer cycles;
        begin
            cycles = 0;
            csr_write(12'h008, 32'h0000_0000);  // Kernel PC
            csr_write(12'h00C, 32'h0000_0001);  // Block dim X
            csr_write(12'h018, 32'h0000_0001);  // Grid dim X
            csr_write(12'h004, 32'h0000_0001);  // Start

            while (!irq_kernel_done && cycles < timeout) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            #100;  // Extra time for memory writes
        end
    endtask

    integer cycle_count;
    integer total_pass, total_fail;
    integer i;
    reg [31:0] result;

    initial begin
        total_pass = 0;
        total_fail = 0;

        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        m_axi_bresp = 0;
        m_axi_rresp = 0;

        $display("============================================================");
        $display("LLM Operators Top-Level Verification");
        $display("Using ralph_gpu_top as DUT");
        $display("============================================================");

        #100;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 1: Dot Product using DP4A
        //--------------------------------------------------------------------
        $display("\n--- Test 1: LLM Dot Product (DP4A) ---");
        init_memory();
        $readmemh("llm_dot_product.hex", imem);
        expected_result = 23;  // 1*3+2*3+10 + 1*2+1*2 = 19 + 4 = 23

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] Dot Product: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (result == expected_result) begin
            $display("[PASS] Dot Product: %0d cycles, result=%0d (expected %0d)",
                     cycle_count, result, expected_result);
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] Dot Product: result=%0d (expected %0d)",
                     result, expected_result);
            total_fail = total_fail + 1;
        end

        // Reset for next test
        rst_n = 0;
        #50;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 2: GEMM 2x2
        //--------------------------------------------------------------------
        $display("\n--- Test 2: LLM GEMM 2x2 ---");
        init_memory();
        $readmemh("llm_gemm_2x2.hex", imem);
        // Expected: C = [[19, 22], [43, 50]]

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] GEMM 2x2: TIMEOUT");
            total_fail = total_fail + 1;
        end else begin
            result = gmem[32'h1000 >> 2];
            if (result == 19 &&
                gmem[32'h1004 >> 2] == 22 &&
                gmem[32'h1008 >> 2] == 43 &&
                gmem[32'h100C >> 2] == 50) begin
                $display("[PASS] GEMM 2x2: %0d cycles", cycle_count);
                $display("       C[0][0]=%0d C[0][1]=%0d C[1][0]=%0d C[1][1]=%0d",
                         gmem[32'h1000 >> 2], gmem[32'h1004 >> 2],
                         gmem[32'h1008 >> 2], gmem[32'h100C >> 2]);
                total_pass = total_pass + 1;
            end else begin
                $display("[FAIL] GEMM 2x2: Wrong result");
                $display("       C[0][0]=%0d (exp 19) C[0][1]=%0d (exp 22)",
                         gmem[32'h1000 >> 2], gmem[32'h1004 >> 2]);
                $display("       C[1][0]=%0d (exp 43) C[1][1]=%0d (exp 50)",
                         gmem[32'h1008 >> 2], gmem[32'h100C >> 2]);
                total_fail = total_fail + 1;
            end
        end

        // Reset for next test
        rst_n = 0;
        #50;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 3: ReLU Activation
        //--------------------------------------------------------------------
        $display("\n--- Test 3: LLM ReLU Activation ---");
        init_memory();
        $readmemh("llm_relu.hex", imem);
        expected_result = 14;  // 3 + 7 + 0 + 4 = 14

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] ReLU: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (result == expected_result) begin
            $display("[PASS] ReLU: %0d cycles, sum=%0d (expected %0d)",
                     cycle_count, result, expected_result);
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] ReLU: sum=%0d (expected %0d)",
                     result, expected_result);
            total_fail = total_fail + 1;
        end

        // Reset for next test
        rst_n = 0;
        #50;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 4: Attention Score
        //--------------------------------------------------------------------
        $display("\n--- Test 4: LLM Attention Score ---");
        init_memory();
        $readmemh("llm_attention_score.hex", imem);
        expected_result = 15;  // Q.K = 8 + 7 = 15

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] Attention Score: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (result == expected_result) begin
            $display("[PASS] Attention Score: %0d cycles, score=%0d (expected %0d)",
                     cycle_count, result, expected_result);
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] Attention Score: score=%0d (expected %0d)",
                     result, expected_result);
            total_fail = total_fail + 1;
        end

        // Reset for next test
        rst_n = 0;
        #50;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 5: Residual Addition
        //--------------------------------------------------------------------
        $display("\n--- Test 5: LLM Residual Add ---");
        init_memory();
        $readmemh("llm_residual_add.hex", imem);
        expected_result = 110;  // 11 + 22 + 33 + 44 = 110

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] Residual Add: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (result == expected_result &&
                     gmem[32'h1004 >> 2] == 11 &&
                     gmem[32'h1008 >> 2] == 22 &&
                     gmem[32'h100C >> 2] == 33 &&
                     gmem[32'h1010 >> 2] == 44) begin
            $display("[PASS] Residual Add: %0d cycles, checksum=%0d",
                     cycle_count, result);
            $display("       out[0]=%0d out[1]=%0d out[2]=%0d out[3]=%0d",
                     gmem[32'h1004 >> 2], gmem[32'h1008 >> 2],
                     gmem[32'h100C >> 2], gmem[32'h1010 >> 2]);
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] Residual Add: checksum=%0d (expected %0d)",
                     result, expected_result);
            total_fail = total_fail + 1;
        end

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("\n============================================================");
        $display("LLM OPERATORS VERIFICATION SUMMARY");
        $display("============================================================");
        $display("Total Tests: %0d", total_pass + total_fail);
        $display("Passed:      %0d", total_pass);
        $display("Failed:      %0d", total_fail);
        $display("");
        if (total_fail == 0) begin
            $display("ALL LLM OPERATOR TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED - Debug required");
        end
        $display("============================================================");
        $finish;
    end

endmodule
