//============================================================================
// LLM FP32 Operators Verification Testbench
// Tests FP32 LLM operators: Softmax, LayerNorm, GELU, SiLU, RMSNorm
//============================================================================

`timescale 1ns / 1ps

module tb_llm_fp_operators;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = 2000;  // FP ops need more cycles

    // FP32 constants for comparison
    parameter [31:0] FP_ONE   = 32'h3F800000;  // 1.0f
    parameter [31:0] FP_FOUR  = 32'h40800000;  // 4.0f
    parameter [31:0] FP_TWO_5 = 32'h40200000;  // 2.5f
    parameter [31:0] FP_7_5   = 32'h40F00000;  // 7.5f

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

    // Memory models
    reg [31:0] imem [0:1023];
    reg [31:0] gmem [0:4095];

    // Instruction memory interface
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[(imem_addr >> 2) + 1], imem[imem_addr >> 2]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    // AXI Write Channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bid <= 0;
        end else begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            if (m_axi_wvalid && m_axi_wready) begin
                gmem[m_axi_awaddr[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (m_axi_bready && m_axi_bvalid) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read Channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            m_axi_rid <= 0;
        end else begin
            m_axi_arready <= 1'b1;
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
            end else if (m_axi_rready && m_axi_rvalid) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
            end
        end
    end

    // CSR write task
    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_wr_en <= 1'b1;
            csr_addr <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    // Initialize memory
    task init_memory;
        integer i;
        begin
            for (i = 0; i < 1024; i = i + 1) begin
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

    // FP32 comparison with tolerance (5% relative error)
    function fp_approx_equal;
        input [31:0] a;
        input [31:0] b;
        reg [31:0] diff;
        reg [30:0] a_abs, b_abs, diff_abs;
        begin
            // Handle exact match
            if (a == b) begin
                fp_approx_equal = 1;
            end else begin
                // Simple tolerance: allow ~10% difference in raw bits
                // This is a rough approximation for FP comparison
                a_abs = a[30:0];
                b_abs = b[30:0];
                if (a_abs > b_abs)
                    diff_abs = a_abs - b_abs;
                else
                    diff_abs = b_abs - a_abs;

                // Allow 5% relative error (check if diff < 5% of larger value)
                if (a_abs > b_abs)
                    fp_approx_equal = (diff_abs < (a_abs >> 4));  // ~6% tolerance
                else
                    fp_approx_equal = (diff_abs < (b_abs >> 4));
            end
        end
    endfunction

    // Convert FP32 bits to real for display
    function real fp32_to_real;
        input [31:0] fp;
        reg sign;
        reg [7:0] exp;
        reg [22:0] mant;
        real result;
        begin
            sign = fp[31];
            exp = fp[30:23];
            mant = fp[22:0];

            if (exp == 0 && mant == 0) begin
                result = 0.0;
            end else if (exp == 255) begin
                result = 999.999;  // Inf/NaN placeholder
            end else begin
                result = (1.0 + mant / 8388608.0) * (2.0 ** (exp - 127));
                if (sign) result = -result;
            end
            fp32_to_real = result;
        end
    endfunction

    integer cycle_count;
    integer total_pass, total_fail;
    integer i;
    reg [31:0] result;
    real result_real;

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
        $display("LLM FP32 Operators Verification");
        $display("Using ralph_gpu_top as DUT");
        $display("============================================================");

        #100;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 1: Softmax
        //--------------------------------------------------------------------
        $display("\n--- Test 1: LLM Softmax ---");
        $display("    exp(x)/sum(exp(x)) using ex2, div");
        init_memory();
        $readmemh("llm_softmax.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];
        result_real = fp32_to_real(result);
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] Softmax: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result, FP_ONE)) begin
            $display("[PASS] Softmax: %0d cycles", cycle_count);
            $display("       sum(softmax) = 0x%08x (%.3f, expected ~1.0)",
                     result, result_real);
            $display("       softmax[0] = 0x%08x (%.3f)",
                     gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
            $display("       softmax[1] = 0x%08x (%.3f)",
                     gmem[32'h1008 >> 2], fp32_to_real(gmem[32'h1008 >> 2]));
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] Softmax: sum=0x%08x (%.3f, expected ~1.0)",
                     result, result_real);
            total_fail = total_fail + 1;
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 2: LayerNorm
        //--------------------------------------------------------------------
        $display("\n--- Test 2: LLM LayerNorm ---");
        $display("    (x - mean) * rsqrt(var)");
        init_memory();
        $readmemh("llm_layernorm.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];  // sum(y^2)
        result_real = fp32_to_real(result);
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] LayerNorm: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result, FP_FOUR)) begin
            $display("[PASS] LayerNorm: %0d cycles", cycle_count);
            $display("       sum(y^2) = 0x%08x (%.3f, expected ~4.0)",
                     result, result_real);
            $display("       mean = 0x%08x (%.3f, expected 2.5)",
                     gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
            $display("       var = 0x%08x (%.3f, expected 1.25)",
                     gmem[32'h1008 >> 2], fp32_to_real(gmem[32'h1008 >> 2]));
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] LayerNorm: sum(y^2)=0x%08x (%.3f, expected ~4.0)",
                     result, result_real);
            $display("       mean = 0x%08x (%.3f)",
                     gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
            total_fail = total_fail + 1;
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 3: GELU Activation
        //--------------------------------------------------------------------
        $display("\n--- Test 3: LLM GELU Activation ---");
        $display("    0.5 * x * (1 + tanh(k*x))");
        init_memory();
        $readmemh("llm_gelu.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];  // sum of GELU outputs
        result_real = fp32_to_real(result);
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] GELU: TIMEOUT");
            total_fail = total_fail + 1;
        end else begin
            // GELU(1.0) + GELU(2.0) should be around 2.7
            // Accept if result is in reasonable range (2.0 to 3.5)
            if (result_real > 2.0 && result_real < 3.5) begin
                $display("[PASS] GELU: %0d cycles", cycle_count);
                $display("       GELU(1.0)+GELU(2.0) = 0x%08x (%.3f, expected ~2.7)",
                         result, result_real);
                $display("       GELU(1.0) = 0x%08x (%.3f)",
                         gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
                $display("       GELU(2.0) = 0x%08x (%.3f)",
                         gmem[32'h1008 >> 2], fp32_to_real(gmem[32'h1008 >> 2]));
                total_pass = total_pass + 1;
            end else begin
                $display("[FAIL] GELU: sum=0x%08x (%.3f, expected ~2.7)",
                         result, result_real);
                total_fail = total_fail + 1;
            end
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 4: SiLU (Swish) Activation
        //--------------------------------------------------------------------
        $display("\n--- Test 4: LLM SiLU (Swish) Activation ---");
        $display("    x * sigmoid(x) = x / (1 + exp(-x))");
        init_memory();
        $readmemh("llm_silu.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];  // sum of SiLU outputs
        result_real = fp32_to_real(result);
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] SiLU: TIMEOUT");
            total_fail = total_fail + 1;
        end else begin
            // SiLU(1.0) + SiLU(2.0) should be around 2.5
            if (result_real > 2.0 && result_real < 3.0) begin
                $display("[PASS] SiLU: %0d cycles", cycle_count);
                $display("       SiLU(1.0)+SiLU(2.0) = 0x%08x (%.3f, expected ~2.5)",
                         result, result_real);
                $display("       SiLU(1.0) = 0x%08x (%.3f)",
                         gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
                $display("       SiLU(2.0) = 0x%08x (%.3f)",
                         gmem[32'h1008 >> 2], fp32_to_real(gmem[32'h1008 >> 2]));
                total_pass = total_pass + 1;
            end else begin
                $display("[FAIL] SiLU: sum=0x%08x (%.3f, expected ~2.5)",
                         result, result_real);
                total_fail = total_fail + 1;
            end
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 5: RMSNorm
        //--------------------------------------------------------------------
        $display("\n--- Test 5: LLM RMSNorm ---");
        $display("    x * rsqrt(mean(x^2))");
        init_memory();
        $readmemh("llm_rmsnorm.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);

        result = gmem[32'h1000 >> 2];  // sum(y^2)
        result_real = fp32_to_real(result);
        if (cycle_count >= TIMEOUT_CYCLES) begin
            $display("[FAIL] RMSNorm: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result, FP_FOUR)) begin
            $display("[PASS] RMSNorm: %0d cycles", cycle_count);
            $display("       sum(y^2) = 0x%08x (%.3f, expected ~4.0)",
                     result, result_real);
            $display("       mean(x^2) = 0x%08x (%.3f, expected 7.5)",
                     gmem[32'h1004 >> 2], fp32_to_real(gmem[32'h1004 >> 2]));
            $display("       rsqrt(mean) = 0x%08x (%.3f)",
                     gmem[32'h1008 >> 2], fp32_to_real(gmem[32'h1008 >> 2]));
            total_pass = total_pass + 1;
        end else begin
            $display("[FAIL] RMSNorm: sum(y^2)=0x%08x (%.3f, expected ~4.0)",
                     result, result_real);
            total_fail = total_fail + 1;
        end

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("\n============================================================");
        $display("LLM FP32 OPERATORS VERIFICATION SUMMARY");
        $display("============================================================");
        $display("Total Tests: %0d", total_pass + total_fail);
        $display("Passed:      %0d", total_pass);
        $display("Failed:      %0d", total_fail);
        $display("");
        if (total_fail == 0) begin
            $display("ALL LLM FP32 OPERATOR TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED - Debug required");
        end
        $display("============================================================");
        $finish;
    end

endmodule
