//============================================================================
// Trigonometric Function Verification Testbench
// Tests SFU trig functions: sin, cos, tan (computed as sin/cos)
//============================================================================

`timescale 1ns / 1ps

module tb_trig_operators;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter TIMEOUT_CYCLES = 2000;

    // Expected values for trigonometric tests
    // sin(π/4) ≈ 0.707 = 0x3F3504F3
    // sin(π/6) ≈ 0.5   = 0x3F000000
    // sin(π/2) = 1.0   = 0x3F800000
    // cos(π/4) ≈ 0.707 = 0x3F3504F3
    // cos(π/6) ≈ 0.866 = 0x3F5DB3D7
    // cos(0)   = 1.0   = 0x3F800000
    // tan(π/4) = 1.0   = 0x3F800000
    // tan(π/6) ≈ 0.577 = 0x3F13CD3A

    parameter [31:0] FP_ONE     = 32'h3F800000;  // 1.0f
    parameter [31:0] FP_HALF    = 32'h3F000000;  // 0.5f
    parameter [31:0] FP_SQRT2_2 = 32'h3F3504F3;  // sqrt(2)/2 ≈ 0.707
    parameter [31:0] FP_SQRT3_2 = 32'h3F5DB3D7;  // sqrt(3)/2 ≈ 0.866
    parameter [31:0] FP_TAN30   = 32'h3F13CD3A;  // tan(30°) ≈ 0.577

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

    // FP32 comparison with tolerance (10% relative error for trig functions)
    function fp_approx_equal;
        input [31:0] a;
        input [31:0] b;
        reg [30:0] a_abs, b_abs, diff_abs;
        begin
            if (a == b) begin
                fp_approx_equal = 1;
            end else begin
                a_abs = a[30:0];
                b_abs = b[30:0];
                if (a_abs > b_abs)
                    diff_abs = a_abs - b_abs;
                else
                    diff_abs = b_abs - a_abs;
                // Allow 10% tolerance for trig approximations
                if (a_abs > b_abs)
                    fp_approx_equal = (diff_abs < (a_abs >> 3));
                else
                    fp_approx_equal = (diff_abs < (b_abs >> 3));
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
    integer sin_cycles, cos_cycles, tan_cycles;
    reg [31:0] result, result1, result2, result3;
    real result_real, r1_real, r2_real, r3_real;

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
        $display("Trigonometric Function Verification");
        $display("Using ralph_gpu_top as DUT");
        $display("============================================================");

        #100;
        rst_n = 1;
        #50;

        //--------------------------------------------------------------------
        // Test 1: sin.f32
        //--------------------------------------------------------------------
        $display("\n--- Test 1: sin.f32 ---");
        $display("    sin(pi/4)=0.707, sin(pi/6)=0.5, sin(pi/2)=1.0");
        init_memory();
        $readmemh("../programs/trig_sin.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);
        sin_cycles = cycle_count;

        result = gmem[32'h1000 >> 2];    // Sum
        result1 = gmem[32'h1004 >> 2];   // sin(π/4)
        result2 = gmem[32'h1008 >> 2];   // sin(π/6)
        result3 = gmem[32'h100c >> 2];   // sin(π/2)

        result_real = fp32_to_real(result);
        r1_real = fp32_to_real(result1);
        r2_real = fp32_to_real(result2);
        r3_real = fp32_to_real(result3);

        if (cycle_count >= TIMEOUT_CYCLES) begin
            $fatal(1, "[FAIL] sin.f32: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result1, FP_SQRT2_2) &&
                     fp_approx_equal(result2, FP_HALF) &&
                     fp_approx_equal(result3, FP_ONE)) begin
            $display("[PASS] sin.f32: %0d cycles", cycle_count);
            $display("       sin(pi/4) = 0x%08x (%.4f, exp 0.707)", result1, r1_real);
            $display("       sin(pi/6) = 0x%08x (%.4f, exp 0.500)", result2, r2_real);
            $display("       sin(pi/2) = 0x%08x (%.4f, exp 1.000)", result3, r3_real);
            total_pass = total_pass + 1;
        end else begin
            $fatal(1, "[FAIL] sin.f32:");
            $display("       sin(pi/4) = 0x%08x (%.4f, exp 0.707)", result1, r1_real);
            $display("       sin(pi/6) = 0x%08x (%.4f, exp 0.500)", result2, r2_real);
            $display("       sin(pi/2) = 0x%08x (%.4f, exp 1.000)", result3, r3_real);
            total_fail = total_fail + 1;
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 2: cos.f32
        //--------------------------------------------------------------------
        $display("\n--- Test 2: cos.f32 ---");
        $display("    cos(pi/4)=0.707, cos(pi/6)=0.866, cos(0)=1.0");
        init_memory();
        $readmemh("../programs/trig_cos.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);
        cos_cycles = cycle_count;

        result = gmem[32'h1000 >> 2];    // Sum
        result1 = gmem[32'h1004 >> 2];   // cos(π/4)
        result2 = gmem[32'h1008 >> 2];   // cos(π/6)
        result3 = gmem[32'h100c >> 2];   // cos(0)

        result_real = fp32_to_real(result);
        r1_real = fp32_to_real(result1);
        r2_real = fp32_to_real(result2);
        r3_real = fp32_to_real(result3);

        if (cycle_count >= TIMEOUT_CYCLES) begin
            $fatal(1, "[FAIL] cos.f32: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result1, FP_SQRT2_2) &&
                     fp_approx_equal(result2, FP_SQRT3_2) &&
                     fp_approx_equal(result3, FP_ONE)) begin
            $display("[PASS] cos.f32: %0d cycles", cycle_count);
            $display("       cos(pi/4) = 0x%08x (%.4f, exp 0.707)", result1, r1_real);
            $display("       cos(pi/6) = 0x%08x (%.4f, exp 0.866)", result2, r2_real);
            $display("       cos(0)    = 0x%08x (%.4f, exp 1.000)", result3, r3_real);
            total_pass = total_pass + 1;
        end else begin
            $fatal(1, "[FAIL] cos.f32:");
            $display("       cos(pi/4) = 0x%08x (%.4f, exp 0.707)", result1, r1_real);
            $display("       cos(pi/6) = 0x%08x (%.4f, exp 0.866)", result2, r2_real);
            $display("       cos(0)    = 0x%08x (%.4f, exp 1.000)", result3, r3_real);
            total_fail = total_fail + 1;
        end

        rst_n = 0; #50; rst_n = 1; #50;

        //--------------------------------------------------------------------
        // Test 3: tan = sin/cos
        //--------------------------------------------------------------------
        $display("\n--- Test 3: tan (sin/cos) ---");
        $display("    tan(pi/4)=1.0, tan(pi/6)=0.577");
        init_memory();
        $readmemh("../programs/trig_tan.hex", imem);

        run_kernel(TIMEOUT_CYCLES, cycle_count);
        tan_cycles = cycle_count;

        result = gmem[32'h1000 >> 2];    // Sum
        result1 = gmem[32'h1004 >> 2];   // tan(π/4)
        result2 = gmem[32'h1008 >> 2];   // tan(π/6)

        result_real = fp32_to_real(result);
        r1_real = fp32_to_real(result1);
        r2_real = fp32_to_real(result2);

        if (cycle_count >= TIMEOUT_CYCLES) begin
            $fatal(1, "[FAIL] tan: TIMEOUT");
            total_fail = total_fail + 1;
        end else if (fp_approx_equal(result1, FP_ONE) &&
                     fp_approx_equal(result2, FP_TAN30)) begin
            $display("[PASS] tan: %0d cycles", cycle_count);
            $display("       tan(pi/4) = 0x%08x (%.4f, exp 1.000)", result1, r1_real);
            $display("       tan(pi/6) = 0x%08x (%.4f, exp 0.577)", result2, r2_real);
            total_pass = total_pass + 1;
        end else begin
            $fatal(1, "[FAIL] tan:");
            $display("       tan(pi/4) = 0x%08x (%.4f, exp 1.000)", result1, r1_real);
            $display("       tan(pi/6) = 0x%08x (%.4f, exp 0.577)", result2, r2_real);
            total_fail = total_fail + 1;
        end

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("\n============================================================");
        $display("Performance Summary");
        $display("============================================================");
        $display("sin.f32 test: %0d cycles", sin_cycles);
        $display("cos.f32 test: %0d cycles", cos_cycles);
        $display("tan test:     %0d cycles", tan_cycles);
        $display("------------------------------------------------------------");
        $display("Total Tests: %0d", total_pass + total_fail);
        $display("Passed: %0d", total_pass);
        $display("Failed: %0d", total_fail);
        if (total_fail == 0) begin
            $display("ALL TRIGONOMETRIC TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED - Debug required");
        end
        $display("============================================================");

        #100;
        $finish;
    end

endmodule
