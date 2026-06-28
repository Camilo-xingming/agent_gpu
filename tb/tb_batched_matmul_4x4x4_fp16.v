//============================================================================
// RalphGPU - Batched 4x4x4 FP16 Tensor Matrix Multiplication Testbench
// Uses ralph_gpu_top as DUT, loads PTX binary from file
// C[b] = A[b] * B[b] for b=0,1,2,3
// Each A,B is 4x4 FP16, each C is 4x4 FP32
//============================================================================

`timescale 1ns / 1ps

module tb_batched_matmul_4x4x4_fp16;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter NUM_BATCHES = 4;

    // Memory addresses
    parameter KERNEL_PC   = 32'h0000_0000;
    parameter TENSOR_A    = 32'h0000_1000;  // 4 x 32 bytes = 128 bytes
    parameter TENSOR_B    = 32'h0000_1080;  // 4 x 32 bytes = 128 bytes
    parameter TENSOR_C    = 32'h0000_1100;  // 4 x 64 bytes = 256 bytes

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
    reg [31:0] imem [0:2047];  // 2K instructions for larger kernel
    integer instr_count;

    initial begin
        for (integer i = 0; i < 2048; i = i + 1) begin
            imem[i] = 32'hFC000000;  // NOP
        end
        $readmemh("../programs/batched_matmul_4x4x4_fp16.hex", imem);
        instr_count = 0;
        for (integer i = 0; i < 2048; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from hex file", instr_count);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                imem_data <= {imem[imem_addr[12:2] + 1], imem[imem_addr[12:2]]};
                imem_valid <= 1'b1;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Global Memory with 4x4x4 Tensor Data
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:16383];

    // Tensor storage for verification
    reg [15:0] tensor_a [0:NUM_BATCHES-1][0:15];  // 4 batches of 4x4 FP16
    reg [15:0] tensor_b [0:NUM_BATCHES-1][0:15];
    real       tensor_a_real [0:NUM_BATCHES-1][0:15];
    real       tensor_b_real [0:NUM_BATCHES-1][0:15];
    real       expected_c [0:NUM_BATCHES-1][0:15];

    // FP16 value generator - different values for variety
    function [15:0] get_fp16_value;
        input integer batch;
        input integer idx;
        integer sel;
        begin
            sel = (batch * 16 + idx) % 8;
            case (sel)
                0: get_fp16_value = 16'h3C00;  //  1.0
                1: get_fp16_value = 16'hBC00;  // -1.0
                2: get_fp16_value = 16'h4000;  //  2.0
                3: get_fp16_value = 16'hC000;  // -2.0
                4: get_fp16_value = 16'h3800;  //  0.5
                5: get_fp16_value = 16'hB800;  // -0.5
                6: get_fp16_value = 16'h4200;  //  3.0
                7: get_fp16_value = 16'hC200;  // -3.0
            endcase
        end
    endfunction

    function [15:0] get_fp16_value_b;
        input integer batch;
        input integer idx;
        integer sel;
        begin
            sel = (batch * 16 + idx + 3) % 8;  // Different pattern for B
            case (sel)
                0: get_fp16_value_b = 16'h3C00;
                1: get_fp16_value_b = 16'hBC00;
                2: get_fp16_value_b = 16'h4000;
                3: get_fp16_value_b = 16'hC000;
                4: get_fp16_value_b = 16'h3800;
                5: get_fp16_value_b = 16'hB800;
                6: get_fp16_value_b = 16'h4200;
                7: get_fp16_value_b = 16'hC200;
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

    // Initialize tensors
    integer b, i, j, k;
    integer a_base, b_base;
    initial begin
        // Initialize memory
        for (i = 0; i < 16384; i = i + 1) begin
            gmem[i] = 32'h0;
        end

        $display("");
        $display("============================================================");
        $display("Initializing 4x4x4 Tensor Data (4 batches of 4x4 matrices)");
        $display("============================================================");

        // Generate tensor A and B for each batch
        for (b = 0; b < NUM_BATCHES; b = b + 1) begin
            $display("");
            $display("--- Batch %0d ---", b);

            // Generate A[b]
            $display("Matrix A[%0d] (FP16):", b);
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tensor_a[b][i*4+j] = get_fp16_value(b, i*4+j);
                    tensor_a_real[b][i*4+j] = fp16_to_real(tensor_a[b][i*4+j]);
                end
                $display("  [%6.2f, %6.2f, %6.2f, %6.2f]",
                    tensor_a_real[b][i*4+0], tensor_a_real[b][i*4+1],
                    tensor_a_real[b][i*4+2], tensor_a_real[b][i*4+3]);
            end

            // Store A[b] in memory (2 FP16 per word, 8 words per matrix)
            a_base = (TENSOR_A >> 2) + b * 8;
            for (i = 0; i < 8; i = i + 1) begin
                gmem[a_base + i] = {tensor_a[b][i*2+1], tensor_a[b][i*2]};
            end

            // Generate B[b]
            $display("Matrix B[%0d] (FP16):", b);
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tensor_b[b][i*4+j] = get_fp16_value_b(b, i*4+j);
                    tensor_b_real[b][i*4+j] = fp16_to_real(tensor_b[b][i*4+j]);
                end
                $display("  [%6.2f, %6.2f, %6.2f, %6.2f]",
                    tensor_b_real[b][i*4+0], tensor_b_real[b][i*4+1],
                    tensor_b_real[b][i*4+2], tensor_b_real[b][i*4+3]);
            end

            // Store B[b] in memory
            b_base = (TENSOR_B >> 2) + b * 8;
            for (i = 0; i < 8; i = i + 1) begin
                gmem[b_base + i] = {tensor_b[b][i*2+1], tensor_b[b][i*2]};
            end

            // Compute expected C[b] = A[b] * B[b]
            $display("Expected C[%0d] (FP32):", b);
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    expected_c[b][i*4+j] = 0.0;
                    for (k = 0; k < 4; k = k + 1) begin
                        expected_c[b][i*4+j] = expected_c[b][i*4+j] +
                            tensor_a_real[b][i*4+k] * tensor_b_real[b][k*4+j];
                    end
                end
                $display("  [%8.2f, %8.2f, %8.2f, %8.2f]",
                    expected_c[b][i*4+0], expected_c[b][i*4+1],
                    expected_c[b][i*4+2], expected_c[b][i*4+3]);
            end
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
                axi_read_addr <= m_axi_araddr;
                axi_read_len <= m_axi_arlen;
                axi_read_cnt <= 0;
                axi_read_active <= 1'b1;
                m_axi_arready <= 1'b0;
                m_axi_rid <= m_axi_arid;
            end else if (axi_read_active) begin
                m_axi_rdata <= gmem[axi_read_addr[15:2] + axi_read_cnt];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= (axi_read_cnt == axi_read_len);

                if (m_axi_rvalid && m_axi_rready) begin
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
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end

            if (m_axi_awvalid && m_axi_awready && !axi_write_active) begin
                axi_write_addr <= m_axi_awaddr;
                axi_write_len <= m_axi_awlen;
                axi_write_cnt <= 0;
                axi_write_active <= 1'b1;
                m_axi_awready <= 1'b0;
                m_axi_wready <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (axi_write_active && m_axi_wvalid && m_axi_wready) begin
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
    integer pass_count, fail_count, total_pass, total_fail;
    reg [31:0] start_time, end_time;
    integer c_base;
    real rtl_val, error, abs_expected;

    initial begin
        $display("");
        $display("============================================================");
        $display("RalphGPU Batched 4x4x4 FP16 Tensor Matrix Multiplication Test");
        $display("4 batches of 4x4 matrix multiplications");
        $display("============================================================");

        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        #100;
        rst_n = 1;
        #100;

        $display("");
        $display("Configuring GPU...");

        csr_write(12'h008, KERNEL_PC);
        csr_write(12'h00C, 32'd1);        // Grid X = 1
        csr_write(12'h010, 32'd1);        // Grid Y = 1
        csr_write(12'h014, 32'd1);        // Grid Z = 1
        csr_write(12'h018, 32'd1);        // Block X = 1
        csr_write(12'h01C, 32'd1);        // Block Y = 1
        csr_write(12'h020, 32'd1);        // Block Z = 1

        $display("Starting kernel execution...");
        start_time = $time;
        csr_write(12'h004, 32'd1);

        // Wait for completion
        fork
            begin
                wait(irq_kernel_done);
                end_time = $time;
                $display("Kernel completed!");
                $display("Execution time: %0d cycles", (end_time - start_time) / CLK_PERIOD);
            end
            begin
                #50_000_000;  // 50ms timeout (longer for 4 batches)
                $display("ERROR: Timeout waiting for kernel completion!");
            end
        join_any
        disable fork;

        #1000;

        // Verify results
        $display("");
        $display("============================================================");
        $display("Verification Results");
        $display("============================================================");

        total_pass = 0;
        total_fail = 0;

        for (b = 0; b < NUM_BATCHES; b = b + 1) begin
            $display("");
            $display("--- Batch %0d Results ---", b);

            c_base = (TENSOR_C >> 2) + b * 16;

            $display("Computed C[%0d] (from GPU):", b);
            for (i = 0; i < 4; i = i + 1) begin
                $display("  [%8.2f, %8.2f, %8.2f, %8.2f]",
                    fp32_to_real(gmem[c_base + i*4 + 0]),
                    fp32_to_real(gmem[c_base + i*4 + 1]),
                    fp32_to_real(gmem[c_base + i*4 + 2]),
                    fp32_to_real(gmem[c_base + i*4 + 3]));
            end

            pass_count = 0;
            fail_count = 0;

            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    rtl_val = fp32_to_real(gmem[c_base + i*4 + j]);
                    error = rtl_val - expected_c[b][i*4+j];
                    if (error < 0) error = -error;

                    abs_expected = expected_c[b][i*4+j];
                    if (abs_expected < 0) abs_expected = -abs_expected;

                    if (error < 0.01 || (abs_expected > 0.001 && error/abs_expected < 0.01)) begin
                        pass_count = pass_count + 1;
                    end else begin
                        $display("  C[%0d][%0d][%0d]: FAIL (RTL=%8.2f, Expected=%8.2f, Error=%8.2f)",
                            b, i, j, rtl_val, expected_c[b][i*4+j], error);
                        fail_count = fail_count + 1;
                    end
                end
            end

            if (fail_count == 0) begin
                $display("  Batch %0d: PASSED (16/16 elements correct)", b);
            end else begin
                $display("  Batch %0d: FAILED (%0d passed, %0d failed)", b, pass_count, fail_count);
            end

            total_pass = total_pass + pass_count;
            total_fail = total_fail + fail_count;
        end

        $display("");
        $display("============================================================");
        if (total_fail == 0) begin
            $display("TEST PASSED: All %0d elements across %0d batches correct!",
                total_pass, NUM_BATCHES);
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed across %0d batches",
                total_pass, total_fail, NUM_BATCHES);
        end
        $display("============================================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Progress indicator
    reg [31:0] progress_cnt;
    initial progress_cnt = 0;
    always @(posedge clk) begin
        progress_cnt <= progress_cnt + 1;
        if (progress_cnt == 500_000) begin  // Every 5ms
            $display("[PROGRESS] time=%0t", $time);
            $fflush();
            progress_cnt <= 0;
        end
    end

    // Timeout
    initial begin
        #200_000_000;
        $display("SIMULATION TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // VCD waveform dump - disabled for faster simulation
    // initial begin
    //     $dumpfile("batched_matmul_4x4x4_fp16.vcd");
    //     $dumpvars(0, tb_batched_matmul_4x4x4_fp16);
    // end

endmodule
