//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Testbench
// Uses ralph_gpu_top as DUT, loads PTX binary from file
// C = A * B where A, B are 4x4 FP16 random matrices, C is 4x4 FP32
//============================================================================

`timescale 1ns / 1ps

module tb_matmul_4x4_fp16_binary;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    // Memory addresses
    parameter KERNEL_PC   = 32'h0000_0000;  // Kernel starts at address 0
    parameter MATRIX_A    = 32'h0000_1000;  // A matrix base
    parameter MATRIX_B    = 32'h0000_1020;  // B matrix base
    parameter MATRIX_C    = 32'h0000_1040;  // C matrix base

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
    // CSR Interface
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    // Instruction Memory Interface
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    // AXI Write Address Channel
    wire [AXI_ID_WIDTH-1:0]   m_axi_awid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_awaddr;
    wire [7:0]                m_axi_awlen;
    wire [2:0]                m_axi_awsize;
    wire [1:0]                m_axi_awburst;
    wire                      m_axi_awvalid;
    reg                       m_axi_awready;

    // AXI Write Data Channel
    wire [AXI_DATA_WIDTH-1:0]   m_axi_wdata;
    wire [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb;
    wire                        m_axi_wlast;
    wire                        m_axi_wvalid;
    reg                         m_axi_wready;

    // AXI Write Response Channel
    reg  [AXI_ID_WIDTH-1:0] m_axi_bid;
    reg  [1:0]              m_axi_bresp;
    reg                     m_axi_bvalid;
    wire                    m_axi_bready;

    // AXI Read Address Channel
    wire [AXI_ID_WIDTH-1:0]   m_axi_arid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_araddr;
    wire [7:0]                m_axi_arlen;
    wire [2:0]                m_axi_arsize;
    wire [1:0]                m_axi_arburst;
    wire                      m_axi_arvalid;
    reg                       m_axi_arready;

    // AXI Read Data Channel
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
    // Instruction Memory (Load from hex file)
    //------------------------------------------------------------------------
    reg [31:0] imem [0:1023];  // 1K instructions
    integer instr_count;

    initial begin
        // Initialize to NOP
        for (integer i = 0; i < 1024; i = i + 1) begin
            imem[i] = 32'hFC000000;  // NOP opcode
        end
        // Load binary from hex file
        $readmemh("../programs/matmul_4x4_fp16.hex", imem);
        // Count instructions
        instr_count = 0;
        for (integer i = 0; i < 1024; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from hex file", instr_count);
    end

    // Instruction memory response (2 instructions per 64-bit fetch)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                // Return 2 instructions (64 bits)
                imem_data <= {imem[imem_addr[11:2] + 1], imem[imem_addr[11:2]]};
                imem_valid <= 1'b1;
                // Debug: show fetched instructions
                if (imem_addr < 32'h20) begin
                    $display("[%0t IMEM_FETCH] addr=0x%04x data={0x%08x, 0x%08x}",
                             $time, imem_addr, imem[imem_addr[11:2] + 1], imem[imem_addr[11:2]]);
                end
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Global Memory (Simple model with random A/B matrices)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:16383];  // 64KB global memory

    // FP16 matrices storage (for verification)
    reg [15:0] matrix_a [0:15];  // 4x4 FP16
    reg [15:0] matrix_b [0:15];  // 4x4 FP16
    real       matrix_a_real [0:15];
    real       matrix_b_real [0:15];
    real       expected_c [0:15];

    // Random FP16 generator (simple: small integers as FP16)
    function [15:0] random_fp16;
        input integer seed;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        integer r;
        begin
            r = $random(seed);
            sign = r[15];
            // Generate values in range [-4, 4] with some variety
            case (r[2:0])
                3'd0: random_fp16 = 16'h3C00;  //  1.0
                3'd1: random_fp16 = 16'hBC00;  // -1.0
                3'd2: random_fp16 = 16'h4000;  //  2.0
                3'd3: random_fp16 = 16'hC000;  // -2.0
                3'd4: random_fp16 = 16'h3800;  //  0.5
                3'd5: random_fp16 = 16'hB800;  // -0.5
                3'd6: random_fp16 = 16'h4200;  //  3.0
                3'd7: random_fp16 = 16'hC200;  // -3.0
            endcase
        end
    endfunction

    // FP16 to real conversion
    function real fp16_to_real;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real result;
        integer exp_unbiased;
        integer i;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            mant = fp16[9:0];
            if (exp == 0 && mant == 0) begin
                fp16_to_real = 0.0;
            end else if (exp == 5'h1F) begin
                fp16_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp - 15;
                result = 1.0 + (mant * 1.0 / 1024.0);
                if (exp_unbiased >= 0) begin
                    for (i = 0; i < exp_unbiased; i = i + 1)
                        result = result * 2.0;
                end else begin
                    for (i = 0; i < -exp_unbiased; i = i + 1)
                        result = result / 2.0;
                end
                fp16_to_real = sign ? -result : result;
            end
        end
    endfunction

    // FP32 to real conversion
    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] mant;
        real result;
        integer exp_unbiased;
        integer i;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            mant = fp32[22:0];
            if (exp == 0 && mant == 0) begin
                fp32_to_real = 0.0;
            end else if (exp == 8'hFF) begin
                fp32_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp - 127;
                result = 1.0 + (mant * 1.0 / 8388608.0);
                if (exp_unbiased >= 0) begin
                    for (i = 0; i < exp_unbiased; i = i + 1)
                        result = result * 2.0;
                end else begin
                    for (i = 0; i < -exp_unbiased; i = i + 1)
                        result = result / 2.0;
                end
                fp32_to_real = sign ? -result : result;
            end
        end
    endfunction

    // Initialize matrices with random FP16 values
    integer seed, i, j, k;
    initial begin
        seed = 12345;
        // Initialize memory to 0
        for (i = 0; i < 16384; i = i + 1) begin
            gmem[i] = 32'h0;
        end

        // For simple test: put a known value at 0x1000
        gmem[MATRIX_A >> 2] = 32'hDEADBEEF;

        // Generate random matrix A (16 FP16 values = 8 words)
        $display("");
        $display("Matrix A (FP16, random values):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                matrix_a[i*4+j] = random_fp16(seed + i*4 + j);
                matrix_a_real[i*4+j] = fp16_to_real(matrix_a[i*4+j]);
            end
            $display("  [%8.4f, %8.4f, %8.4f, %8.4f]",
                matrix_a_real[i*4+0], matrix_a_real[i*4+1],
                matrix_a_real[i*4+2], matrix_a_real[i*4+3]);
        end

        // Store A in memory (2 FP16 per word)
        // Address 0x1000 -> gmem[0x400]
        for (i = 0; i < 8; i = i + 1) begin
            gmem[(MATRIX_A >> 2) + i] = {matrix_a[i*2+1], matrix_a[i*2]};
        end

        // Generate random matrix B
        $display("");
        $display("Matrix B (FP16, random values):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                matrix_b[i*4+j] = random_fp16(seed + 100 + i*4 + j);
                matrix_b_real[i*4+j] = fp16_to_real(matrix_b[i*4+j]);
            end
            $display("  [%8.4f, %8.4f, %8.4f, %8.4f]",
                matrix_b_real[i*4+0], matrix_b_real[i*4+1],
                matrix_b_real[i*4+2], matrix_b_real[i*4+3]);
        end

        // Store B in memory
        for (i = 0; i < 8; i = i + 1) begin
            gmem[(MATRIX_B >> 2) + i] = {matrix_b[i*2+1], matrix_b[i*2]};
        end

        // Compute expected C = A * B
        $display("");
        $display("Expected C = A * B (FP32):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                expected_c[i*4+j] = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    expected_c[i*4+j] = expected_c[i*4+j] +
                        matrix_a_real[i*4+k] * matrix_b_real[k*4+j];
                end
            end
            $display("  [%10.4f, %10.4f, %10.4f, %10.4f]",
                expected_c[i*4+0], expected_c[i*4+1],
                expected_c[i*4+2], expected_c[i*4+3]);
        end
    end

    //------------------------------------------------------------------------
    // AXI Memory Model
    //------------------------------------------------------------------------
    reg [31:0] axi_read_addr;
    reg [7:0]  axi_read_len;
    reg [7:0]  axi_read_cnt;
    reg        axi_read_active;

    reg [31:0] axi_write_addr;
    reg [7:0]  axi_write_len;
    reg [7:0]  axi_write_cnt;
    reg        axi_write_active;

    // AXI Read FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rid <= 0;
            m_axi_rdata <= 0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            m_axi_rvalid <= 1'b0;
            axi_read_active <= 1'b0;
            axi_read_addr <= 0;
            axi_read_len <= 0;
            axi_read_cnt <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready && !axi_read_active) begin
                // Accept read request
                $display("[%0t AXI] Read request: addr=0x%08h len=%0d", $time, m_axi_araddr, m_axi_arlen);
                axi_read_addr <= m_axi_araddr;
                axi_read_len <= m_axi_arlen;
                axi_read_cnt <= 0;
                axi_read_active <= 1'b1;
                m_axi_arready <= 1'b0;
                m_axi_rid <= m_axi_arid;
            end else if (axi_read_active) begin
                // Return read data
                m_axi_rdata <= gmem[axi_read_addr[15:2] + axi_read_cnt];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= (axi_read_cnt == axi_read_len);

                if (m_axi_rvalid && m_axi_rready) begin
                    // Debug read data
                    $display("[%0t AXI] Read data: addr=0x%08h data=0x%08h cnt=%0d last=%0d",
                        $time, axi_read_addr + (axi_read_cnt << 2), gmem[axi_read_addr[15:2] + axi_read_cnt],
                        axi_read_cnt, (axi_read_cnt == axi_read_len));
                    if (axi_read_cnt == axi_read_len) begin
                        axi_read_active <= 1'b0;
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast <= 1'b0;
                        m_axi_arready <= 1'b1;
                    end else begin
                        axi_read_cnt <= axi_read_cnt + 1;
                    end
                end
            end
        end
    end

    // AXI Write FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b0;
            m_axi_bid <= 0;
            m_axi_bresp <= 2'b00;
            m_axi_bvalid <= 1'b0;
            axi_write_active <= 1'b0;
            axi_write_addr <= 0;
            axi_write_len <= 0;
            axi_write_cnt <= 0;
        end else begin
            // Clear bvalid when response accepted
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end

            if (m_axi_awvalid && m_axi_awready && !axi_write_active) begin
                // Accept write address
                axi_write_addr <= m_axi_awaddr;
                axi_write_len <= m_axi_awlen;
                axi_write_cnt <= 0;
                axi_write_active <= 1'b1;
                m_axi_awready <= 1'b0;
                m_axi_wready <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (axi_write_active && m_axi_wvalid && m_axi_wready) begin
                // Write data
                gmem[axi_write_addr[15:2] + axi_write_cnt] <= m_axi_wdata;

                if (m_axi_wlast || axi_write_cnt == axi_write_len) begin
                    axi_write_active <= 1'b0;
                    m_axi_wready <= 1'b0;
                    m_axi_awready <= 1'b1;
                    m_axi_bvalid <= 1'b1;
                end else begin
                    axi_write_cnt <= axi_write_cnt + 1;
                end
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
            csr_wr_en <= 1'b1;
            csr_addr <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test
    //------------------------------------------------------------------------
    integer pass_count, fail_count;
    reg [31:0] start_time, end_time;
    real rtl_val, error, abs_expected;

    initial begin
        $display("");
        $display("============================================================");
        $display("RalphGPU 4x4 FP16 Matrix Multiplication Test");
        $display("Using ralph_gpu_top as DUT, loading binary from file");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        // Reset
        #100;
        rst_n = 1;
        #100;

        $display("");
        $display("Configuring GPU...");

        // Configure kernel
        csr_write(12'h008, KERNEL_PC);    // Kernel PC
        csr_write(12'h00C, 32'd1);        // Grid X = 1
        csr_write(12'h010, 32'd1);        // Grid Y = 1
        csr_write(12'h014, 32'd1);        // Grid Z = 1
        csr_write(12'h018, 32'd1);        // Block X = 1 (only thread 0 active, no predicates needed)
        csr_write(12'h01C, 32'd1);        // Block Y = 1
        csr_write(12'h020, 32'd1);        // Block Z = 1

        $display("Starting kernel execution...");
        start_time = $time;
        csr_write(12'h004, 32'd1);        // Start kernel

        // Wait for completion
        fork
            begin
                wait(irq_kernel_done);
                end_time = $time;
                $display("Kernel completed (irq_kernel_done asserted)");
                $display("Execution time: %0d cycles", (end_time - start_time) / CLK_PERIOD);
            end
            begin
                #10_000_000;  // 10ms timeout
                $display("ERROR: Timeout waiting for kernel completion!");
            end
        join_any
        disable fork;

        // Wait a bit for memory writes to complete
        #1000;

        // Verify results
        $display("");
        $display("============================================================");
        $display("Verification Results");
        $display("============================================================");

        $display("");
        $display("Computed C matrix (from GPU):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%10.4f, %10.4f, %10.4f, %10.4f]",
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 0]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 1]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 2]),
                fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + 3]));
        end

        $display("");
        $display("Element-by-element comparison:");
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                rtl_val = fp32_to_real(gmem[(MATRIX_C >> 2) + i*4 + j]);
                error = rtl_val - expected_c[i*4+j];
                if (error < 0) error = -error;

                abs_expected = expected_c[i*4+j];
                if (abs_expected < 0) abs_expected = -abs_expected;
                if (error < 0.01 || (abs_expected > 0.001 && error/abs_expected < 0.01)) begin
                    $display("  C[%0d][%0d]: PASS (RTL=%10.4f, Expected=%10.4f)",
                        i, j, rtl_val, expected_c[i*4+j]);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  C[%0d][%0d]: FAIL (RTL=%10.4f, Expected=%10.4f, Error=%10.4f)",
                        i, j, rtl_val, expected_c[i*4+j], error);
                    fail_count = fail_count + 1;
                end
            end
        end

        $display("");
        $display("============================================================");
        if (fail_count == 0) begin
            $display("TEST PASSED: All %0d elements match!", pass_count);
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed", pass_count, fail_count);
        end
        $display("============================================================");

        #100;
        $finish;
    end

    // Progress indicator - print every 1ms simulation time
    reg [31:0] progress_cnt;
    initial progress_cnt = 0;
    always @(posedge clk) begin
        progress_cnt <= progress_cnt + 1;
        if (progress_cnt == 100_000) begin  // Every 1ms simulation time
            $display("[PROGRESS] time=%0t", $time);
            $fflush();  // Force output
            progress_cnt <= 0;
        end
    end

    // Timeout
    initial begin
        #100_000_000;  // Increased for full simulation
        $display("SIMULATION TIMEOUT");
        $finish;
    end

    //------------------------------------------------------------------------
    // Debug Probes - trace memory path
    //------------------------------------------------------------------------
    // Probe the SM's memory interface signals
    wire sm_gmem_req_valid;
    wire sm_gmem_req_ready;
    wire sm_issue_valid;
    wire sm_issue_mem_read;
    wire sm_issue_mem_write;
    wire sm_issue_mem_shared;
    wire [31:0] sm_issue_mask;
    wire [2:0] mem_if_state;

    assign sm_gmem_req_valid = u_dut.sm_gen[0].u_sm.gmem_req_valid;
    assign sm_gmem_req_ready = u_dut.sm_gen[0].u_sm.gmem_req_ready;
    assign sm_issue_valid = u_dut.sm_gen[0].u_sm.issue_valid;
    assign sm_issue_mem_read = u_dut.sm_gen[0].u_sm.issue_mem_read;
    assign sm_issue_mem_write = u_dut.sm_gen[0].u_sm.issue_mem_write;
    assign sm_issue_mem_shared = u_dut.sm_gen[0].u_sm.issue_mem_shared;
    assign sm_issue_mask = u_dut.sm_gen[0].u_sm.issue_mask;
    assign mem_if_state = u_dut.sm_gen[0].u_sm.u_mem_if.state;

    // Verbose debug disabled - uncomment if needed
    // always @(posedge clk) begin
    //     if (sm_gmem_req_valid) begin
    //         $display("[%0t MEM_DBG] gmem_req_valid=1 ready=%0d mask=0x%08h",
    //                  $time, sm_gmem_req_ready, sm_issue_mask);
    //     end
    //     if (mem_if_state != 0) begin
    //         $display("[%0t MEM_IF] state=%0d arvalid=%0d arready=%0d",
    //                  $time, mem_if_state, m_axi_arvalid, m_axi_arready);
    //     end
    // end

    // Report when issue_valid with mem_read - disabled for faster simulation
    // always @(posedge clk) begin
    //     if (sm_issue_valid && sm_issue_mem_read) begin
    //         $display("[%0t ISSUE] mem_read issued: shared=%0d mask=0x%08h",
    //                  $time, sm_issue_mem_shared, sm_issue_mask);
    //     end
    // end

    // Additional debug: trace decode and issue
    wire dec_valid = u_dut.sm_gen[0].u_sm.dec_valid;
    wire [5:0] dec_opcode = u_dut.sm_gen[0].u_sm.dec_opcode;
    wire dec_alu_op = u_dut.sm_gen[0].u_sm.dec_alu_op;
    wire dec_reg_write = u_dut.sm_gen[0].u_sm.dec_reg_write;
    wire issue_accept = u_dut.sm_gen[0].u_sm.issue_accept;
    wire issue_stall_mem = u_dut.sm_gen[0].u_sm.issue_stall_mem;

    // Additional probes for ALU/WB/RF debugging
    wire alu_issue = u_dut.sm_gen[0].u_sm.alu_issue;
    wire [4:0] alu_issue_rd = u_dut.sm_gen[0].u_sm.alu_issue_rd;
    wire [5:0] alu_issue_opcode = u_dut.sm_gen[0].u_sm.alu_issue_opcode;
    wire alu_is_mov_imm = u_dut.sm_gen[0].u_sm.alu_is_mov_imm;
    wire [15:0] alu_issue_imm16 = u_dut.sm_gen[0].u_sm.alu_issue_imm16;
    wire alu_valid_pipe = u_dut.sm_gen[0].u_sm.alu_valid_pipe;
    wire alu_valid_out = u_dut.sm_gen[0].u_sm.alu_valid_out;
    wire [4:0] alu_rd_pipe = u_dut.sm_gen[0].u_sm.alu_rd_pipe;
    wire [31:0] alu_result_pipe_lane0 = u_dut.sm_gen[0].u_sm.alu_result_pipe[31:0];

    wire wb_valid = u_dut.sm_gen[0].u_sm.wb_valid;
    wire [4:0] wb_rd = u_dut.sm_gen[0].u_sm.wb_rd;
    wire [31:0] wb_data_lane0 = u_dut.sm_gen[0].u_sm.wb_data[31:0];
    wire [3:0] wb_sel = u_dut.sm_gen[0].u_sm.wb_sel;

    wire rf_wr_en = u_dut.sm_gen[0].u_sm.rf_wr_en;
    wire [31:0] rf_rd_data_a_lane0 = u_dut.sm_gen[0].u_sm.rf_rd_data_a[31:0];
    wire [4:0] issue_ra = u_dut.sm_gen[0].u_sm.issue_ra;
    wire [4:0] issue_rd = u_dut.sm_gen[0].u_sm.issue_rd;
    wire [5:0] issue_opcode = u_dut.sm_gen[0].u_sm.issue_opcode;

    wire [31:0] scoreboard_busy_w0 = u_dut.sm_gen[0].u_sm.scoreboard_busy[0];

    wire issue0_fire = u_dut.sm_gen[0].u_sm.issue0_fire;
    wire lane0_stall_raw = u_dut.sm_gen[0].u_sm.lane0_stall_raw;
    wire [4:0] dec_ra = u_dut.sm_gen[0].u_sm.dec_ra;
    wire [4:0] dec_rd = u_dut.sm_gen[0].u_sm.dec_rd;

    // Debug - track instruction issue using decoded values (not latched ones)
    wire [5:0] dec_opcode_probe = u_dut.sm_gen[0].u_sm.dec_opcode;
    wire [4:0] dec_rd_probe = u_dut.sm_gen[0].u_sm.dec_rd;
    wire [4:0] dec_ra_probe = u_dut.sm_gen[0].u_sm.dec_ra;
    wire [31:0] dec_pc_probe = u_dut.sm_gen[0].u_sm.dec0_pc;
    always @(posedge clk) begin
        if (issue0_fire) begin
            $display("[%0t ISSUE] pc=0x%04h opcode=%0d rd=R%0d ra=R%0d",
                     $time, dec_pc_probe, dec_opcode_probe, dec_rd_probe, dec_ra_probe);
            $fflush();
        end
    end

    // Additional debug for ALU queue (disabled)
    wire alu_wbq_push_sig = u_dut.sm_gen[0].u_sm.alu_wbq_push;
    wire alu_wbq_pop_sig = u_dut.sm_gen[0].u_sm.alu_wbq_pop;
    wire alu_wbq_empty_sig = u_dut.sm_gen[0].u_sm.alu_wbq_empty;
    wire [31:0] alu_result_lane0 = u_dut.sm_gen[0].u_sm.alu_result[31:0];
    wire [15:0] alu_imm16_sig = u_dut.sm_gen[0].u_sm.alu_issue_imm16;
    wire alu_use_imm_sig = u_dut.sm_gen[0].u_sm.alu_issue_use_imm;
    wire [31:0] alu_operand_b_lane0 = u_dut.sm_gen[0].u_sm.u_simd_alu.operand_b[31:0];

    // FP32 writeback debug probes
    wire fpu32_issue_sig = u_dut.sm_gen[0].u_sm.fpu32_issue;
    wire fpu32_valid_out_sig = u_dut.sm_gen[0].u_sm.fpu32_valid_out;
    wire fpu32_wbq_push_sig = u_dut.sm_gen[0].u_sm.fpu32_wbq_push;
    wire fpu32_wbq_pop_sig = u_dut.sm_gen[0].u_sm.fpu32_wbq_pop;
    wire fpu32_wbq_empty_sig = u_dut.sm_gen[0].u_sm.fpu32_wbq_empty;
    wire [4:0] fpu32_rd_pipe_0 = u_dut.sm_gen[0].u_sm.fpu32_rd_pipe[0];
    wire [31:0] fpu32_result_lane0 = u_dut.sm_gen[0].u_sm.fpu32_result[31:0];

    // Track FP32 operations
    always @(posedge clk) begin
        if (fpu32_issue_sig) begin
            $display("[%0t FP32] ISSUE: rd=R%0d", $time, issue_rd_sig);
        end
        if (fpu32_valid_out_sig) begin
            $display("[%0t FP32] VALID_OUT: push=%b empty=%b rd_pipe[0]=R%0d result=0x%08x",
                     $time, fpu32_wbq_push_sig, fpu32_wbq_empty_sig, fpu32_rd_pipe_0, fpu32_result_lane0);
        end
        if (fpu32_wbq_pop_sig) begin
            $display("[%0t FP32] WBQ_POP: wb_sel=%0d", $time, wb_sel_sig);
        end
    end

    // FP16 writeback debug probes
    wire fp16_issue_sig = u_dut.sm_gen[0].u_sm.fp16_issue;
    wire fp16_valid_in_sig = u_dut.sm_gen[0].u_sm.fp16_valid_in;
    wire fp16_valid_out_sig = u_dut.sm_gen[0].u_sm.fp16_valid_out;
    wire fp16_wbq_push_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_push;
    wire fp16_wbq_pop_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_pop;
    wire fp16_wbq_empty_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_empty;
    wire fp16_wbq_full_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_full;
    wire [2:0] fp16_inflight_sig = u_dut.sm_gen[0].u_sm.fp16_inflight;
    wire [3:0] wb_sel_sig = u_dut.sm_gen[0].u_sm.wb_sel;
    wire wb_found_sig = u_dut.sm_gen[0].u_sm.wb_found;
    wire [10:0] fu_ready_sig = u_dut.sm_gen[0].u_sm.fu_ready;
    wire wb_valid_sig = u_dut.sm_gen[0].u_sm.wb_valid;
    wire [4:0] fp16_wbq_rd_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_rd;
    wire [1:0] fp16_wbq_warp_sig = u_dut.sm_gen[0].u_sm.fp16_wbq_warp;

    // Debug FP16 issue rd
    wire [4:0] fp16_issue_rd_sig = u_dut.sm_gen[0].u_sm.fp16_issue_rd;
    wire [4:0] issue_rd_sig = u_dut.sm_gen[0].u_sm.issue_rd;
    wire issue_fp16_op_sig = u_dut.sm_gen[0].u_sm.issue_fp16_op;
    wire fp16_issue0_sig = u_dut.sm_gen[0].u_sm.fp16_issue0;

    // Track FP16 operations
    always @(posedge clk) begin
        if (fp16_issue_sig) begin
            $display("[%0t FP16] ISSUE: valid_in=%b issue_rd=R%0d issue_fp16_op=%b fp16_issue0=%b",
                     $time, fp16_valid_in_sig, issue_rd_sig, issue_fp16_op_sig, fp16_issue0_sig);
            $fflush();
        end
        if (fp16_valid_out_sig) begin
            $display("[%0t FP16] VALID_OUT: push=%b empty=%b full=%b inflight=%0d rd_pipe[2]=R%0d",
                     $time, fp16_wbq_push_sig, fp16_wbq_empty_sig, fp16_wbq_full_sig, fp16_inflight_sig,
                     u_dut.sm_gen[0].u_sm.fp16_rd_pipe[2]);
            $fflush();
        end
        if (fp16_wbq_pop_sig) begin
            $display("[%0t FP16] WBQ_POP: wb_sel=%0d", $time, wb_sel_sig);
            $fflush();
        end
    end

    // Track writeback arbitration - only when FP16 queue is not empty
    always @(posedge clk) begin
        if (!fp16_wbq_empty_sig && wb_found_sig) begin
            $display("[%0t WB_ARB] fu_ready=%b wb_sel=%0d wb_valid=%b fp16_empty=%b sched_wb_valid=%b",
                     $time, fu_ready_sig, wb_sel_sig, wb_valid_sig, fp16_wbq_empty_sig, sched_wb_valid);
            $display("    fp16_wbq: rd=R%0d warp=%d", fp16_wbq_rd_sig, fp16_wbq_warp_sig);
            $fflush();
        end
    end

    // CVT/ALU debug - track ALU operations when CVT (opcode 17) is issued
    wire [5:0] issue_func_sig = u_dut.sm_gen[0].u_sm.issue_func;
    wire dec_cvt_op_sig = u_dut.sm_gen[0].u_sm.dec_cvt_op;
    wire [31:0] alu_op_a_lane0 = u_dut.sm_gen[0].u_sm.u_simd_alu.operand_a[31:0];
    wire [5:0] alu_func_wire = u_dut.sm_gen[0].u_sm.u_simd_alu.func;
    wire [5:0] dec_func_sig = u_dut.sm_gen[0].u_sm.dec_func;
    wire [5:0] alu_issue_func_sig = u_dut.sm_gen[0].u_sm.alu_issue_func;

    // Debug ALU operand_b
    wire [31:0] alu_op_b_lane0 = u_dut.sm_gen[0].u_sm.u_simd_alu.operand_b[31:0];
    wire [15:0] alu_imm16_probe = u_dut.sm_gen[0].u_sm.alu_issue_imm16;
    wire alu_use_imm_probe = u_dut.sm_gen[0].u_sm.alu_issue_use_imm;

    always @(posedge clk) begin
        // Track CVT decode
        if (issue0_fire && dec_opcode_probe == 17) begin
            $display("[%0t CVT] DECODE: rd=R%0d ra=R%0d dec_func=%0d dec_cvt_op=%b",
                     $time, dec_rd_probe, dec_ra_probe, dec_func_sig, dec_cvt_op_sig);
            $fflush();
        end
        // Track ALU issue (when the result is actually computed)
        if (alu_issue) begin
            $display("[%0t ALU] ISSUE: func=%0d op_a[0]=0x%08x op_b[0]=0x%08x result[0]=0x%08x use_imm=%b imm16=0x%04x",
                     $time, alu_func_wire, alu_op_a_lane0, alu_op_b_lane0, alu_result_lane0,
                     alu_use_imm_probe, alu_imm16_probe);
            $fflush();
        end
    end

    // Debug decode stall conditions
    wire dec0_valid_sig = u_dut.sm_gen[0].u_sm.dec0_valid;
    wire lane0_stall_raw_sig = u_dut.sm_gen[0].u_sm.lane0_stall_raw;
    wire lane0_stall_fu_sig = u_dut.sm_gen[0].u_sm.lane0_stall_fu;
    wire lane0_stall_mem_sig = u_dut.sm_gen[0].u_sm.lane0_stall_mem;
    wire lane0_stall_wbq_sig = u_dut.sm_gen[0].u_sm.lane0_stall_wbq;
    wire lane0_ready_sig = u_dut.sm_gen[0].u_sm.lane0_ready;
    wire [31:0] scoreboard_w0_sig = u_dut.sm_gen[0].u_sm.scoreboard_busy[0];
    wire lane0_ra_busy_sig = u_dut.sm_gen[0].u_sm.lane0_ra_busy;
    wire lane0_rb_busy_sig = u_dut.sm_gen[0].u_sm.lane0_rb_busy;
    wire lane0_rc_busy_sig = u_dut.sm_gen[0].u_sm.lane0_rc_busy;

    // Print stall status when decode is valid but stalled
    reg stall_printed;
    always @(posedge clk) begin
        if (dec0_valid_sig && !lane0_ready_sig && !stall_printed) begin
            $display("[%0t STALL] pc=0x%04h raw=%b fu=%b mem=%b wbq=%b scoreboard=0x%08x ra_busy=%b rb_busy=%b rc_busy=%b",
                     $time, dec_pc_probe, lane0_stall_raw_sig, lane0_stall_fu_sig,
                     lane0_stall_mem_sig, lane0_stall_wbq_sig, scoreboard_w0_sig,
                     lane0_ra_busy_sig, lane0_rb_busy_sig, lane0_rc_busy_sig);
            $fflush();
            stall_printed <= 1'b1;
        end
        if (issue0_fire) begin
            stall_printed <= 1'b0;
        end
    end

    // Debug fetch - track when we stop getting instructions
    wire [31:0] warp0_pc_sig = u_dut.sm_gen[0].u_sm.warp_pc[0];
    wire warp0_active_sig = u_dut.sm_gen[0].u_sm.warp_active[0];
    wire [31:0] warp0_fetch_pc_sig = u_dut.sm_gen[0].u_sm.warp_fetch_pc[0];
    wire fetch_req_sig = u_dut.sm_gen[0].u_sm.fetch_req;

    // Additional fetch debug signals
    wire warp0_valid_sig = u_dut.sm_gen[0].u_sm.warp_valid[0];
    wire warp0_needs_fetch_sig = u_dut.sm_gen[0].u_sm.warp_needs_fetch[0];
    wire warp0_exit_pending_sig = u_dut.sm_gen[0].u_sm.warp_exit_pending[0];
    wire warp0_fetch_pending_sig = u_dut.sm_gen[0].u_sm.warp_fetch_pending[0];
    wire warp0_inst_buf_valid_sig = u_dut.sm_gen[0].u_sm.warp_inst_buf_valid[0];
    wire fetch_valid_arb_sig = u_dut.sm_gen[0].u_sm.fetch_valid_arb;

    // Warp stall debug signals
    wire warp0_stalled_mem_sig = u_dut.sm_gen[0].u_sm.warp_stalled_mem[0];
    wire warp0_stalled_fu_sig = u_dut.sm_gen[0].u_sm.warp_stalled_fu[0];
    wire warp0_stalled_sync_sig = u_dut.sm_gen[0].u_sm.warp_stalled_sync[0];
    wire warp0_ready_sig = u_dut.sm_gen[0].u_sm.warp_ready[0];

    // Scheduler debug signals
    wire [31:0] warp0_inst_buf_sig = u_dut.sm_gen[0].u_sm.warp_inst_buf[0];
    wire pd0_is_compute_sig = u_dut.sm_gen[0].u_sm.pd_is_compute[0];
    wire pd0_is_memory_sig = u_dut.sm_gen[0].u_sm.pd_is_memory[0];
    wire pd0_is_branch_sig = u_dut.sm_gen[0].u_sm.pd_is_branch[0];
    wire pipe_compute0_ready_sig = u_dut.sm_gen[0].u_sm.pipe_compute0_ready;
    wire pipe_memory_ready_sig = u_dut.sm_gen[0].u_sm.pipe_memory_ready;
    wire [1:0] sched_issue_valid_sig = u_dut.sm_gen[0].u_sm.sched_issue_valid_mask;
    wire warp0_schedulable_sig = u_dut.sm_gen[0].u_sm.u_scheduler.warp_schedulable[0];
    wire warp0_eligible_sig = u_dut.sm_gen[0].u_sm.u_scheduler.warp_eligible[0];
    wire warp0_has_hazard_sig = u_dut.sm_gen[0].u_sm.u_scheduler.warp_has_hazard[0];
    wire compute0_eligible_sig = u_dut.sm_gen[0].u_sm.u_scheduler.compute_eligible[0];
    wire found_compute0_sig = u_dut.sm_gen[0].u_sm.u_scheduler.found_compute0;
    wire [31:0] scoreboard0_sig = u_dut.sm_gen[0].u_sm.u_scheduler.scoreboard[0];
    wire raw_hazard0_sig = u_dut.sm_gen[0].u_sm.u_scheduler.gen_hazard[0].raw_hazard;
    wire waw_hazard0_sig = u_dut.sm_gen[0].u_sm.u_scheduler.gen_hazard[0].waw_hazard;

    // Scheduler writeback signals
    wire sched_wb_valid = u_dut.sm_gen[0].u_sm.wb_valid;
    wire [1:0] sched_wb_warp = u_dut.sm_gen[0].u_sm.wb_warp_id;
    wire [4:0] sched_wb_rd = u_dut.sm_gen[0].u_sm.wb_rd;

    // Monitor writeback to scheduler
    always @(posedge clk) begin
        if (sched_wb_valid && sched_wb_rd == 5'd20) begin
            $display("[%0t WB2SCHED_R20] warp=%d rd=R%0d scoreboard=0x%08x",
                     $time, sched_wb_warp, sched_wb_rd, scoreboard0_sig);
        end
    end

    // R20 register file write tracking
    wire rf_wr_en_probe = u_dut.sm_gen[0].u_sm.rf_wr_en;
    wire [4:0] rf_wr_rd_probe = u_dut.sm_gen[0].u_sm.wb_rd;
    wire [31:0] rf_wr_data_lane0_probe = u_dut.sm_gen[0].u_sm.wb_data[31:0];

    always @(posedge clk) begin
        if (rf_wr_en_probe && rf_wr_rd_probe == 5'd20) begin
            $display("[%0t RF_WR_R20] data[0]=0x%08x (fp16=%f)",
                     $time, rf_wr_data_lane0_probe,
                     rf_wr_data_lane0_probe[15:0] == 16'hB800 ? -0.5 :
                     rf_wr_data_lane0_probe[15:0] == 16'h4200 ? 3.0 : 0.0);
            $fflush();
        end
    end

    // R20 register file read tracking - when CVT is issued
    wire [31:0] rf_rd_data_a_lane0_probe = u_dut.sm_gen[0].u_sm.rf_rd_data_a[31:0];

    always @(posedge clk) begin
        // Track when ra=R20 is being read for CVT (opcode 17)
        if (issue0_fire && dec_opcode_probe == 17 && dec_ra_probe == 5'd20) begin
            $display("[%0t CVT_RD_R20] ra=R%0d rf_rd_data[0]=0x%08x scoreboard=0x%08x",
                     $time, dec_ra_probe, rf_rd_data_a_lane0_probe, scoreboard0_sig);
            $fflush();
        end
    end

    // Check if wb_valid ever goes high after FP16 issue
    reg [31:0] fp16_issue_time;
    always @(posedge clk) begin
        if (fp16_issue_sig) fp16_issue_time <= $time;
        // Print at specific times around FP16 writeback
        if ($time == 23295000 || $time == 23305000 || $time == 23315000) begin
            $display("[%0t WB_DBG] wb_valid=%b wb_rd=R%0d wb_warp=%d scoreboard=0x%08x",
                     $time, sched_wb_valid, sched_wb_rd, sched_wb_warp, scoreboard0_sig);
        end
    end

    reg fetch_debug_printed;
    reg [31:0] last_issue_time;
    always @(posedge clk) begin
        if (issue0_fire) begin
            last_issue_time <= $time;
            fetch_debug_printed <= 1'b0;
        end
        // Print debug info if we haven't issued for 100ns after last issue
        if (!fetch_debug_printed && last_issue_time > 0 && $time > last_issue_time + 100000) begin
            $display("[%0t FETCH_DBG] last_issue=%0t warp0_pc=0x%08x dec0_valid=%b",
                     $time, last_issue_time, warp0_pc_sig, dec0_valid_sig);
            $display("    warp0: valid=%b needs_fetch=%b exit_pending=%b fetch_pending=%b inst_buf_valid=%b",
                     warp0_valid_sig, warp0_needs_fetch_sig, warp0_exit_pending_sig,
                     warp0_fetch_pending_sig, warp0_inst_buf_valid_sig);
            $display("    stalls: mem=%b fu=%b sync=%b warp_ready=%b",
                     warp0_stalled_mem_sig, warp0_stalled_fu_sig, warp0_stalled_sync_sig, warp0_ready_sig);
            $display("    inst_buf=0x%08x pd_is: compute=%b mem=%b branch=%b",
                     warp0_inst_buf_sig, pd0_is_compute_sig, pd0_is_memory_sig, pd0_is_branch_sig);
            $display("    sched: schedulable=%b eligible=%b hazard=%b compute_eligible=%b found_compute0=%b",
                     warp0_schedulable_sig, warp0_eligible_sig, warp0_has_hazard_sig, compute0_eligible_sig, found_compute0_sig);
            $display("    scoreboard[0]=0x%08x raw=%b waw=%b", scoreboard0_sig, raw_hazard0_sig, waw_hazard0_sig);
            $display("    pipes: compute0_ready=%b mem_ready=%b issue_valid=%b",
                     pipe_compute0_ready_sig, pipe_memory_ready_sig, sched_issue_valid_sig);
            $display("    fetch: valid_arb=%b req=%b imem_req=%b imem_valid=%b",
                     fetch_valid_arb_sig, fetch_req_sig, imem_req, imem_valid);
            $fflush();
            fetch_debug_printed <= 1'b1;
        end
    end

endmodule
