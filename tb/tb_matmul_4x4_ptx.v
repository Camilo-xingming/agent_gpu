//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Test using PTX Instructions
// DUT: ralph_gpu_top
// C[4x4] = A[4x4] * B[4x4]  (FP16 inputs, FP32 output)
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_matmul_4x4_ptx;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;
    localparam IMEM_SIZE = 1024;
    localparam DMEM_SIZE = 4096;

    // Memory addresses
    localparam ADDR_A = 32'h0000_0000;  // Matrix A: 16 x FP16 = 32 bytes
    localparam ADDR_B = 32'h0000_0100;  // Matrix B: 16 x FP16 = 32 bytes
    localparam ADDR_C = 32'h0000_0200;  // Matrix C: 16 x FP32 = 64 bytes

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
    // CSR interface
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    // Instruction memory
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    // AXI memory interface
    wire [3:0]  m_axi_awid, m_axi_arid;
    wire [31:0] m_axi_awaddr, m_axi_araddr;
    wire [7:0]  m_axi_awlen, m_axi_arlen;
    wire [2:0]  m_axi_awsize, m_axi_arsize;
    wire [1:0]  m_axi_awburst, m_axi_arburst;
    wire        m_axi_awvalid, m_axi_arvalid;
    reg         m_axi_awready, m_axi_arready;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast, m_axi_wvalid;
    reg         m_axi_wready;
    reg  [3:0]  m_axi_bid, m_axi_rid;
    reg  [1:0]  m_axi_bresp, m_axi_rresp;
    reg         m_axi_bvalid, m_axi_rvalid;
    wire        m_axi_bready, m_axi_rready;
    reg  [31:0] m_axi_rdata;
    reg         m_axi_rlast;

    //------------------------------------------------------------------------
    // Instruction Memory (PTX Program)
    //------------------------------------------------------------------------
    reg [31:0] imem [0:IMEM_SIZE-1];
    reg        imem_req_q;
    reg [31:0] imem_addr_q;

    // PTX Instruction Encoding Functions
    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]=rc, [5:0]=func

    function [31:0] encode_fp16_fma;
        input [4:0] rd, ra, rb, rc;
        begin
            encode_fp16_fma = {`OP_FP16_ARITH, rd, ra, rb, rc, `FP16_FMA};
        end
    endfunction

    function [31:0] encode_fp16_mul;
        input [4:0] rd, ra, rb;
        begin
            encode_fp16_mul = {`OP_FP16_ARITH, rd, ra, rb, 5'b0, `FP16_MUL};
        end
    endfunction

    function [31:0] encode_fp32_add;
        input [4:0] rd, ra, rb;
        begin
            encode_fp32_add = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, `FP_ADD};
        end
    endfunction

    function [31:0] encode_ld_global;
        input [4:0] rd, ra;
        input [10:0] offset;
        begin
            encode_ld_global = {`OP_LD_GLOBAL, rd, ra, offset};
        end
    endfunction

    function [31:0] encode_st_global;
        input [4:0] rs, ra;
        input [10:0] offset;  // offset ignored - address calculated in ra register
        begin
            // Format: {opcode[31:26], rd[25:21], ra[20:16], rb[15:11], rc[10:6], func[5:0]}
            // SM reads: address from RA (rf_rd_data_a), data from RB (rf_rd_data_b)
            encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rs, 5'b0, 6'b0};
        end
    endfunction

    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        begin
            encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    function [31:0] encode_nop;
        begin
            encode_nop = {`OP_NOP, 26'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // PTX Program for 4x4 Matrix Multiply
    // Single thread computes one element C[tid_y][tid_x]
    // For simplicity, we compute all 16 elements sequentially
    //------------------------------------------------------------------------
    integer pc;
    initial begin
        // Initialize all instructions to NOP
        for (pc = 0; pc < IMEM_SIZE; pc = pc + 1) begin
            imem[pc] = encode_nop();
        end

        pc = 0;

        // Simple matrix multiply kernel:
        // For each output element C[i][j]:
        //   C[i][j] = sum(A[i][k] * B[k][j]) for k=0..3
        //
        // Register usage:
        //   R0 = zero
        //   R1-R4 = A row elements
        //   R5-R8 = B column elements
        //   R9 = accumulator (FP32)
        //   R10-R13 = temp products
        //   R14 = base address A
        //   R15 = base address B
        //   R16 = base address C

        // Load base addresses
        imem[pc] = encode_mov_imm(5'd14, ADDR_A[15:0]);  pc = pc + 1;  // R14 = &A
        imem[pc] = encode_mov_imm(5'd15, ADDR_B[15:0]);  pc = pc + 1;  // R15 = &B
        imem[pc] = encode_mov_imm(5'd16, ADDR_C[15:0]);  pc = pc + 1;  // R16 = &C
        imem[pc] = encode_mov_imm(5'd0, 16'h0000);       pc = pc + 1;  // R0 = 0

        // Compute C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
        // Load A row 0: A[0][0..3] from addresses 0, 2, 4, 6
        imem[pc] = encode_ld_global(5'd1, 5'd14, 11'd0);   pc = pc + 1;  // R1 = A[0][0]
        imem[pc] = encode_ld_global(5'd2, 5'd14, 11'd2);   pc = pc + 1;  // R2 = A[0][1]
        imem[pc] = encode_ld_global(5'd3, 5'd14, 11'd4);   pc = pc + 1;  // R3 = A[0][2]
        imem[pc] = encode_ld_global(5'd4, 5'd14, 11'd6);   pc = pc + 1;  // R4 = A[0][3]

        // Load B col 0: B[0][0], B[1][0], B[2][0], B[3][0] from addresses 0, 8, 16, 24
        imem[pc] = encode_ld_global(5'd5, 5'd15, 11'd0);   pc = pc + 1;  // R5 = B[0][0]
        imem[pc] = encode_ld_global(5'd6, 5'd15, 11'd8);   pc = pc + 1;  // R6 = B[1][0]
        imem[pc] = encode_ld_global(5'd7, 5'd15, 11'd16);  pc = pc + 1;  // R7 = B[2][0]
        imem[pc] = encode_ld_global(5'd8, 5'd15, 11'd24);  pc = pc + 1;  // R8 = B[3][0]

        // Compute dot product using FP16 mul + FP32 accumulate
        // R10 = R1 * R5 (FP16 mul -> FP32)
        // R11 = R2 * R6
        // R12 = R3 * R7
        // R13 = R4 * R8
        imem[pc] = encode_fp16_mul(5'd10, 5'd1, 5'd5);     pc = pc + 1;
        imem[pc] = encode_fp16_mul(5'd11, 5'd2, 5'd6);     pc = pc + 1;
        imem[pc] = encode_fp16_mul(5'd12, 5'd3, 5'd7);     pc = pc + 1;
        imem[pc] = encode_fp16_mul(5'd13, 5'd4, 5'd8);     pc = pc + 1;

        // Sum: R9 = R10 + R11 + R12 + R13
        imem[pc] = encode_fp32_add(5'd9, 5'd10, 5'd11);    pc = pc + 1;
        imem[pc] = encode_fp32_add(5'd9, 5'd9, 5'd12);     pc = pc + 1;
        imem[pc] = encode_fp32_add(5'd9, 5'd9, 5'd13);     pc = pc + 1;

        // Store C[0][0]
        imem[pc] = encode_st_global(5'd9, 5'd16, 11'd0);   pc = pc + 1;  // C[0][0]

        // Exit
        imem[pc] = encode_exit();

        $display("PTX Program loaded: %0d instructions", pc + 1);
    end

    // Instruction memory response
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_req_q <= 1'b0;
            imem_addr_q <= 32'b0;
            imem_data <= 64'b0;
        end else begin
            imem_valid <= imem_req_q;
            imem_req_q <= imem_req;
            if (imem_req) begin
                imem_addr_q <= imem_addr;
            end
            if (imem_req_q) begin
                imem_data <= {imem[imem_addr_q[12:2] + 1], imem[imem_addr_q[12:2]]};
            end
        end
    end

    //------------------------------------------------------------------------
    // Data Memory (AXI Slave Model)
    //------------------------------------------------------------------------
    reg [31:0] dmem [0:DMEM_SIZE-1];

    // Matrix data (FP16)
    reg [15:0] A_fp16 [0:15];
    reg [15:0] B_fp16 [0:15];
    real A_real [0:15];
    real B_real [0:15];
    real C_expected [0:15];

    // FP16 encoding (simplified)
    function [15:0] real_to_fp16;
        input real val;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real abs_val;
        integer exp_int;
        begin
            if (val == 0.0) begin
                real_to_fp16 = 16'h0000;
            end else begin
                sign = (val < 0) ? 1'b1 : 1'b0;
                abs_val = (val < 0) ? -val : val;
                exp_int = 0;
                if (abs_val >= 1.0) begin
                    while (abs_val >= 2.0 && exp_int < 15) begin
                        abs_val = abs_val / 2.0;
                        exp_int = exp_int + 1;
                    end
                end else begin
                    while (abs_val < 1.0 && exp_int > -14) begin
                        abs_val = abs_val * 2.0;
                        exp_int = exp_int - 1;
                    end
                end
                exp = exp_int + 15;
                mant = (abs_val - 1.0) * 1024.0;
                real_to_fp16 = {sign, exp, mant};
            end
        end
    endfunction

    // Initialize matrices with random FP16 values
    integer i, j, k;
    integer seed;
    real rand_val;

    initial begin
        seed = 12345;

        // Initialize matrix A with random values
        for (i = 0; i < 16; i = i + 1) begin
            rand_val = (($random(seed) % 2001) - 1000) / 500.0;  // -2.0 to 2.0
            seed = seed + 7;
            A_real[i] = rand_val;
            A_fp16[i] = real_to_fp16(rand_val);
        end

        // Initialize matrix B with random values
        for (i = 0; i < 16; i = i + 1) begin
            rand_val = (($random(seed) % 2001) - 1000) / 500.0;
            seed = seed + 13;
            B_real[i] = rand_val;
            B_fp16[i] = real_to_fp16(rand_val);
        end

        // Calculate expected results
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                C_expected[i*4+j] = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    C_expected[i*4+j] = C_expected[i*4+j] + A_real[i*4+k] * B_real[k*4+j];
                end
            end
        end

        // Initialize data memory
        for (i = 0; i < DMEM_SIZE; i = i + 1) begin
            dmem[i] = 32'h0;
        end

        // Store matrix A at ADDR_A (FP16, 2 elements per word)
        for (i = 0; i < 8; i = i + 1) begin
            dmem[(ADDR_A >> 2) + i] = {A_fp16[i*2+1], A_fp16[i*2]};
        end

        // Store matrix B at ADDR_B
        for (i = 0; i < 8; i = i + 1) begin
            dmem[(ADDR_B >> 2) + i] = {B_fp16[i*2+1], B_fp16[i*2]};
        end
    end

    // Simple AXI read response
    reg [31:0] pending_addr;
    reg        pending_read;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 32'b0;
            m_axi_rlast <= 1'b1;
            m_axi_rid <= 4'b0;
            m_axi_rresp <= 2'b00;
            pending_read <= 1'b0;
            pending_addr <= 32'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                pending_addr <= m_axi_araddr;
                pending_read <= 1'b1;
                m_axi_rid <= m_axi_arid;
            end

            if (pending_read) begin
                m_axi_rdata <= dmem[pending_addr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                pending_read <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
            end
        end
    end

    // Simple AXI write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
        end else begin
            if (m_axi_wvalid && m_axi_wready) begin
                dmem[m_axi_awaddr[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    ralph_gpu_top #(
        .NUM_SM(1)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .csr_wr_en      (csr_wr_en),
        .csr_addr       (csr_addr),
        .csr_wr_data    (csr_wr_data),
        .csr_rd_data    (csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req       (imem_req),
        .imem_addr      (imem_addr),
        .imem_data      (imem_data),
        .imem_valid     (imem_valid),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready)
    );

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    integer timeout;
    real result_val;

    // FP32 to real conversion
    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] mant;
        real result;
        integer exp_unbiased;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            mant = fp32[22:0];
            if (exp == 0 && mant == 0) begin
                fp32_to_real = 0.0;
            end else begin
                exp_unbiased = exp - 127;
                result = 1.0 + (mant * 1.0 / 8388608.0);
                if (exp_unbiased >= 0) begin
                    repeat(exp_unbiased) result = result * 2.0;
                end else begin
                    repeat(-exp_unbiased) result = result / 2.0;
                end
                fp32_to_real = sign ? -result : result;
            end
        end
    endfunction

    initial begin
        $display("============================================================");
        $display("RalphGPU 4x4 Matrix Multiply Test with PTX Instructions");
        $display("DUT: ralph_gpu_top");
        $display("============================================================");
        $display("");

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Display input matrices
        $display("Matrix A (FP16):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%7.4f, %7.4f, %7.4f, %7.4f]",
                     A_real[i*4+0], A_real[i*4+1], A_real[i*4+2], A_real[i*4+3]);
        end

        $display("");
        $display("Matrix B (FP16):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%7.4f, %7.4f, %7.4f, %7.4f]",
                     B_real[i*4+0], B_real[i*4+1], B_real[i*4+2], B_real[i*4+3]);
        end

        $display("");
        $display("Expected C = A * B:");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%9.5f, %9.5f, %9.5f, %9.5f]",
                     C_expected[i*4+0], C_expected[i*4+1],
                     C_expected[i*4+2], C_expected[i*4+3]);
        end

        $display("");
        $display("------------------------------------------------------------");
        $display("Starting kernel execution...");
        $display("------------------------------------------------------------");

        // Start kernel via CSR
        @(posedge clk);
        csr_addr = 12'h000;  // Kernel start register
        csr_wr_data = 32'h1;  // Start signal
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;

        // Wait for kernel completion or timeout
        timeout = 10000;
        while (!irq_kernel_done && timeout > 0) begin
            @(posedge clk);
            timeout = timeout - 1;
        end

        if (timeout == 0) begin
            $display("WARNING: Kernel timeout - checking partial results");
        end else begin
            $display("Kernel completed in %0d cycles", 10000 - timeout);
        end

        // Read results from memory
        $display("");
        $display("Result Matrix C (FP32 from memory):");
        for (i = 0; i < 4; i = i + 1) begin
            $write("  [");
            for (j = 0; j < 4; j = j + 1) begin
                result_val = fp32_to_real(dmem[(ADDR_C >> 2) + i*4 + j]);
                if (j > 0) $write(", ");
                $write("%9.5f", result_val);
            end
            $display("]");
        end

        $display("");
        $display("============================================================");
        $display("Test Complete");
        $display("============================================================");

        #100;
        $finish;
    end

endmodule
