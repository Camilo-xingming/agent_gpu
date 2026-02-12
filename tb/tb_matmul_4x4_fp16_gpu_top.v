//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Test using ralph_gpu_top
// Tests C = A * B where A, B are 4x4 FP16 matrices, C is FP32
// Uses PTX instruction encoding through the full GPU top-level module
//============================================================================

`timescale 1ns / 1ps

`include "gpu_defines.vh"

module tb_matmul_4x4_fp16_gpu_top;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;  // 100 MHz

    // Memory layout (addresses computed from constants in kernel)
    // Using smaller addresses for easier arithmetic computation
    // A at 0x0100 (256), B at 0x0200 (512), C at 0x0300 (768)
    // These can be computed from 1 using 8-9 doubling operations
    localparam ADDR_A_BASE = 32'h0000_0100;  // 256 = 2^8
    localparam ADDR_B_BASE = 32'h0000_0200;  // 512 = 2^9
    localparam ADDR_C_BASE = 32'h0000_0300;  // 768 = 256 + 512
    localparam ADDR_INCR   = 32'h0000_0004;

    // Instruction memory base
    localparam IMEM_BASE = 32'h0000_0000;

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

    // AXI4 Memory Interface
    wire [3:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready;
    reg  [3:0]  m_axi_bid;
    reg  [1:0]  m_axi_bresp;
    reg         m_axi_bvalid;
    wire        m_axi_bready;
    wire [3:0]  m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready;
    reg  [3:0]  m_axi_rid;
    reg  [31:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rlast;
    reg         m_axi_rvalid;
    wire        m_axi_rready;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    ralph_gpu_top #(
        .NUM_SM(1)  // Single SM for simplicity
    ) u_gpu (
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
    // Instruction Memory Model
    // Contains the PTX kernel for 4x4 matrix multiply
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:255];

    // Instruction encoding helpers
    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]=rc, [5:0]=func

    function [31:0] encode_mov_special;
        input [4:0] rd;
        input [4:0] sreg;  // Special register code (goes in ra field)
        begin
            // OP_MOV_SPECIAL = 6'b001001
            // Format: [31:26]=opcode, [25:21]=rd, [20:16]=sreg, [15:0]=unused
            encode_mov_special = {`OP_MOV_SPECIAL, rd, sreg, 16'b0};
        end
    endfunction

    function [31:0] encode_ld_global;
        input [4:0] rd;
        input [4:0] ra;  // Address register
        begin
            // OP_LD_GLOBAL = 6'b000101
            // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra (addr), [15:0]=offset
            encode_ld_global = {`OP_LD_GLOBAL, rd, ra, 16'b0};
        end
    endfunction

    function [31:0] encode_st_global;
        input [4:0] ra;  // Address register
        input [4:0] rb;  // Data register
        begin
            // OP_ST_GLOBAL = 6'b000110
            // Format: [31:26]=opcode, [25:21]=unused, [20:16]=ra (addr), [15:11]=rb (data)
            encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rb, 11'b0};
        end
    endfunction

    function [31:0] encode_fp16_mul;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            // OP_FP16_ARITH = 6'b010000, FP16_MUL = 6'b000010
            // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]=rc, [5:0]=func
            encode_fp16_mul = {`OP_FP16_ARITH, rd, ra, rb, 5'b0, `FP16_MUL_F32};
        end
    endfunction

    function [31:0] encode_fp32_add;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            // OP_FP32_ARITH = 6'b001101, FP_ADD = 6'b000000
            encode_fp32_add = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, `FP_ADD};
        end
    endfunction

    function [31:0] encode_alu_add;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            // OP_ALU = 6'b000000, FUNC_ADD = 6'b000000
            encode_alu_add = {`OP_ALU, rd, ra, rb, 5'b0, `FUNC_ADD};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            // OP_EXIT = 6'b001011
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    function [31:0] encode_nop;
        begin
            encode_nop = {`OP_NOP, 26'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // Global Memory Model (Data)
    //------------------------------------------------------------------------
    reg [31:0] global_mem [0:4095];

    // Pending write address (for split AW/W AXI4 handling)
    reg [31:0] pending_aw_addr;
    reg [3:0]  pending_aw_id;
    reg        pending_aw_valid;  // 16KB memory

    // FP16 encoding helper
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
                    while (abs_val >= 2.0) begin
                        abs_val = abs_val / 2.0;
                        exp_int = exp_int + 1;
                    end
                end else begin
                    while (abs_val < 1.0) begin
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

    // FP32 to real conversion for result checking
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
            end else if (exp == 8'hFF) begin
                fp32_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
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

    //------------------------------------------------------------------------
    // Instruction Memory Response
    //------------------------------------------------------------------------
    reg [31:0] pending_imem_addr;
    reg        pending_imem_req;
    integer fetch_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
            pending_imem_req <= 1'b0;
            pending_imem_addr <= 32'b0;
            fetch_count <= 0;
        end else begin
            if (imem_req) begin
                pending_imem_req <= 1'b1;
                pending_imem_addr <= imem_addr;
                imem_valid <= 1'b0;
                if (fetch_count < 20)
                    $display("  IMEM_REQ: addr=0x%08h", imem_addr);
                fetch_count <= fetch_count + 1;
            end else if (pending_imem_req) begin
                // Return 2 instructions (64 bits)
                imem_data <= {instruction_mem[(pending_imem_addr >> 2) + 1],
                              instruction_mem[pending_imem_addr >> 2]};
                imem_valid <= 1'b1;
                pending_imem_req <= 1'b0;
                if (fetch_count < 21)
                    $display("  IMEM_RESP: data=0x%016h (inst0=0x%08h, inst1=0x%08h)",
                             {instruction_mem[(pending_imem_addr >> 2) + 1],
                              instruction_mem[pending_imem_addr >> 2]},
                             instruction_mem[pending_imem_addr >> 2],
                             instruction_mem[(pending_imem_addr >> 2) + 1]);
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // AXI Memory Response (Simplified)
    //------------------------------------------------------------------------
    reg [31:0] pending_axi_addr;
    reg        pending_axi_read;
    reg [2:0]  axi_read_delay;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 32'b0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            m_axi_rid <= 4'b0;
            pending_axi_read <= 1'b0;
            pending_axi_addr <= 32'b0;
            axi_read_delay <= 3'b0;

            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            pending_aw_addr <= 0;
            pending_aw_id <= 0;
            pending_aw_valid <= 0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
        end else begin
            // Read channel
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_read <= 1'b1;
                pending_axi_addr <= m_axi_araddr;
                m_axi_arready <= 1'b0;
                axi_read_delay <= 3'd2;  // 2 cycle latency
                $display("  AXI_READ_REQ: addr=0x%08h", m_axi_araddr);
            end else if (pending_axi_read && axi_read_delay > 0) begin
                axi_read_delay <= axi_read_delay - 1'b1;
            end else if (pending_axi_read && axi_read_delay == 0) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= global_mem[pending_axi_addr[13:2]];
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
                pending_axi_read <= 1'b0;
                $display("  AXI_READ_RESP: addr=0x%08h data=0x%08h",
                         pending_axi_addr, global_mem[pending_axi_addr[13:2]]);
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end

            // Write channel
            if (m_axi_wvalid && m_axi_wready) begin
                global_mem[pending_aw_addr[13:2]] <= m_axi_wdata;
                $display("  AXI Write: addr=0x%08h data=0x%08h", pending_aw_addr, m_axi_wdata);
            end

            // Latch write address (AW and W channels can be split in AXI4)
            if (m_axi_awvalid && m_axi_awready) begin
                pending_aw_addr <= m_axi_awaddr;
                pending_aw_id <= m_axi_awid;
                pending_aw_valid <= 1'b1;
            end

            // Write response on W channel completion (using latched address)
            if (m_axi_wvalid && m_axi_wready && pending_aw_valid) begin
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= pending_aw_id;
                pending_aw_valid <= 1'b0;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Matrix Data and Expected Results
    //------------------------------------------------------------------------
    // Use simple, exact FP16 values for predictable results
    reg [15:0] A [0:15];  // 4x4 FP16 matrix A (row-major)
    reg [15:0] B [0:15];  // 4x4 FP16 matrix B (row-major)
    real A_real [0:15];
    real B_real [0:15];
    real C_expected [0:15];

    //------------------------------------------------------------------------
    // CSR Write Task
    //------------------------------------------------------------------------
    task write_csr;
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
    // Initialize Memory and Kernel
    //------------------------------------------------------------------------
    integer i, j, k, pc;

    initial begin
        $display("============================================================");
        $display("RalphGPU 4x4 FP16 Matrix Multiplication - GPU Top Test");
        $display("Using ralph_gpu_top with PTX instructions");
        $display("============================================================");
        $display("");

        // Initialize signals
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        // Initialize instruction memory to NOPs
        for (i = 0; i < 256; i = i + 1) begin
            instruction_mem[i] = encode_nop();
        end

        // Initialize global memory
        for (i = 0; i < 4096; i = i + 1) begin
            global_mem[i] = 32'b0;
        end

        //--------------------------------------------------------------------
        // Initialize Matrix A (simple values for exact FP16)
        //--------------------------------------------------------------------
        // Row 0: [1.0, 2.0, 0.5, -1.0]
        A[0] = 16'h3C00; A_real[0] =  1.0;
        A[1] = 16'h4000; A_real[1] =  2.0;
        A[2] = 16'h3800; A_real[2] =  0.5;
        A[3] = 16'hBC00; A_real[3] = -1.0;
        // Row 1: [2.0, 1.0, -0.5, 0.5]
        A[4] = 16'h4000; A_real[4] =  2.0;
        A[5] = 16'h3C00; A_real[5] =  1.0;
        A[6] = 16'hB800; A_real[6] = -0.5;
        A[7] = 16'h3800; A_real[7] =  0.5;
        // Row 2: [0.5, -0.5, 1.0, 2.0]
        A[8]  = 16'h3800; A_real[8]  =  0.5;
        A[9]  = 16'hB800; A_real[9]  = -0.5;
        A[10] = 16'h3C00; A_real[10] =  1.0;
        A[11] = 16'h4000; A_real[11] =  2.0;
        // Row 3: [-1.0, 0.5, 2.0, 1.0]
        A[12] = 16'hBC00; A_real[12] = -1.0;
        A[13] = 16'h3800; A_real[13] =  0.5;
        A[14] = 16'h4000; A_real[14] =  2.0;
        A[15] = 16'h3C00; A_real[15] =  1.0;

        //--------------------------------------------------------------------
        // Initialize Matrix B (simple values)
        //--------------------------------------------------------------------
        // Row 0: [1.0, 0.5, -1.0, 2.0]
        B[0] = 16'h3C00; B_real[0] =  1.0;
        B[1] = 16'h3800; B_real[1] =  0.5;
        B[2] = 16'hBC00; B_real[2] = -1.0;
        B[3] = 16'h4000; B_real[3] =  2.0;
        // Row 1: [0.5, 1.0, 2.0, -0.5]
        B[4] = 16'h3800; B_real[4] =  0.5;
        B[5] = 16'h3C00; B_real[5] =  1.0;
        B[6] = 16'h4000; B_real[6] =  2.0;
        B[7] = 16'hB800; B_real[7] = -0.5;
        // Row 2: [-0.5, 2.0, 1.0, 0.5]
        B[8]  = 16'hB800; B_real[8]  = -0.5;
        B[9]  = 16'h4000; B_real[9]  =  2.0;
        B[10] = 16'h3C00; B_real[10] =  1.0;
        B[11] = 16'h3800; B_real[11] =  0.5;
        // Row 3: [2.0, -1.0, 0.5, 1.0]
        B[12] = 16'h4000; B_real[12] =  2.0;
        B[13] = 16'hBC00; B_real[13] = -1.0;
        B[14] = 16'h3800; B_real[14] =  0.5;
        B[15] = 16'h3C00; B_real[15] =  1.0;

        //--------------------------------------------------------------------
        // Calculate Expected Results
        //--------------------------------------------------------------------
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                C_expected[i*4+j] = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    C_expected[i*4+j] = C_expected[i*4+j] + A_real[i*4+k] * B_real[k*4+j];
                end
            end
        end

        //--------------------------------------------------------------------
        // Store matrices in global memory
        // Each FP16 value stored in lower 16 bits of 32-bit word
        //--------------------------------------------------------------------
        for (i = 0; i < 16; i = i + 1) begin
            global_mem[(ADDR_A_BASE >> 2) + i] = {16'b0, A[i]};
            global_mem[(ADDR_B_BASE >> 2) + i] = {16'b0, B[i]};
        end

        //--------------------------------------------------------------------
        // Generate PTX Kernel
        // Simple kernel that computes C[0][0] = sum(A[0][k] * B[k][0])
        // Uses special registers for base addresses
        //--------------------------------------------------------------------
        pc = 0;

        // Compute base addresses from constant 1
        // R25 = 1 (from ntid.x = block_dim_x)
        // R20 = 256 (ADDR_A_BASE) = 1 << 8
        // R21 = 512 (ADDR_B_BASE) = 1 << 9
        // R22 = 768 (ADDR_C_BASE) = 256 + 512
        // R23 = 4 (ADDR_INCR) = 1 << 2

        // R25 = ntid.x = 1 (block_dim_x)
        instruction_mem[pc] = encode_mov_special(5'd25, `SREG_NTID_X); pc = pc + 1;

        // R23 = 4: double R25 twice
        instruction_mem[pc] = encode_alu_add(5'd23, 5'd25, 5'd25); pc = pc + 1;  // R23 = 2
        instruction_mem[pc] = encode_alu_add(5'd23, 5'd23, 5'd23); pc = pc + 1;  // R23 = 4

        // R20 = 256: continue doubling from R23 (4) 6 more times (4*64=256)
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd23, 5'd23); pc = pc + 1;  // R20 = 8
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd20); pc = pc + 1;  // R20 = 16
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd20); pc = pc + 1;  // R20 = 32
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd20); pc = pc + 1;  // R20 = 64
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd20); pc = pc + 1;  // R20 = 128
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd20); pc = pc + 1;  // R20 = 256 (ADDR_A_BASE)

        // R21 = 512: double R20 once
        instruction_mem[pc] = encode_alu_add(5'd21, 5'd20, 5'd20); pc = pc + 1;  // R21 = 512 (ADDR_B_BASE)

        // R22 = 768: R20 + R21
        instruction_mem[pc] = encode_alu_add(5'd22, 5'd20, 5'd21); pc = pc + 1;  // R22 = 768 (ADDR_C_BASE)

        // Initialize accumulator R10 = 0 (use ALU add R10 = R0 + R0 where R0 = 0)
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd0, 5'd0); pc = pc + 1;

        // Computing C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
        // NOTE: Using R3, R4, R5 instead of R0, R1, R2 to avoid R0 scoreboard hazard
        // (pre-decode interprets zero bits as R0 source register)

        // ---- k=0: A[0][0] * B[0][0] ----
        // Load A[0][0]: R3 = mem[R20]
        instruction_mem[pc] = encode_ld_global(5'd3, 5'd20); pc = pc + 1;
        // Load B[0][0]: R4 = mem[R21]
        instruction_mem[pc] = encode_ld_global(5'd4, 5'd21); pc = pc + 1;
        // FP16 multiply: R5 = R3 * R4 (FP32 result)
        instruction_mem[pc] = encode_fp16_mul(5'd5, 5'd3, 5'd4); pc = pc + 1;
        // Accumulate: R10 = R10 + R5
        instruction_mem[pc] = encode_fp32_add(5'd10, 5'd10, 5'd5); pc = pc + 1;

        // Update addresses for k=1
        // R20 = R20 + R23 (A address += 4)
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd23); pc = pc + 1;
        // R24 = R23 << 2 = R23 + R23 + R23 + R23 (B stride = 16)
        instruction_mem[pc] = encode_alu_add(5'd24, 5'd23, 5'd23); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd24, 5'd24, 5'd24); pc = pc + 1;
        // R21 = R21 + R24 (B address += 16 for next row)
        instruction_mem[pc] = encode_alu_add(5'd21, 5'd21, 5'd24); pc = pc + 1;

        // ---- k=1: A[0][1] * B[1][0] ----
        instruction_mem[pc] = encode_ld_global(5'd3, 5'd20); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd4, 5'd21); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd5, 5'd3, 5'd4); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd10, 5'd10, 5'd5); pc = pc + 1;

        // Update addresses for k=2
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd23); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd21, 5'd21, 5'd24); pc = pc + 1;

        // ---- k=2: A[0][2] * B[2][0] ----
        instruction_mem[pc] = encode_ld_global(5'd3, 5'd20); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd4, 5'd21); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd5, 5'd3, 5'd4); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd10, 5'd10, 5'd5); pc = pc + 1;

        // Update addresses for k=3
        instruction_mem[pc] = encode_alu_add(5'd20, 5'd20, 5'd23); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd21, 5'd21, 5'd24); pc = pc + 1;

        // ---- k=3: A[0][3] * B[3][0] ----
        instruction_mem[pc] = encode_ld_global(5'd3, 5'd20); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd4, 5'd21); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd5, 5'd3, 5'd4); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd10, 5'd10, 5'd5); pc = pc + 1;

        // Store result: mem[R22] = R10
        instruction_mem[pc] = encode_st_global(5'd22, 5'd10); pc = pc + 1;

        // Exit
        instruction_mem[pc] = encode_exit(); pc = pc + 1;

        $display("Kernel size: %0d instructions", pc);
        $display("");

        //--------------------------------------------------------------------
        // Display Matrices
        //--------------------------------------------------------------------
        $display("Matrix A (FP16):");
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", A_real[0], A_real[1], A_real[2], A_real[3]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", A_real[4], A_real[5], A_real[6], A_real[7]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", A_real[8], A_real[9], A_real[10], A_real[11]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", A_real[12], A_real[13], A_real[14], A_real[15]);

        $display("");
        $display("Matrix B (FP16):");
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", B_real[0], B_real[1], B_real[2], B_real[3]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", B_real[4], B_real[5], B_real[6], B_real[7]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", B_real[8], B_real[9], B_real[10], B_real[11]);
        $display("  [%8.4f, %8.4f, %8.4f, %8.4f]", B_real[12], B_real[13], B_real[14], B_real[15]);

        $display("");
        $display("Expected C[0][0] = %8.4f", C_expected[0]);
        $display("");

        //--------------------------------------------------------------------
        // Reset and Start
        //--------------------------------------------------------------------
        #100;
        rst_n = 1;
        #50;

        $display("------------------------------------------------------------");
        $display("Configuring GPU via CSR...");
        $display("------------------------------------------------------------");

        // Configure kernel parameters
        // block_dim = (1,1,1) for single thread execution
        write_csr(12'h018, 32'd1);        // BLOCK_DIM_X = 1 thread
        write_csr(12'h01C, 32'd1);        // BLOCK_DIM_Y = 1
        write_csr(12'h020, 32'd1);        // BLOCK_DIM_Z = 1

        // grid_dim = (1,1,1) for single block
        write_csr(12'h00C, 32'd1);        // GRID_DIM_X = 1
        write_csr(12'h010, 32'd1);        // GRID_DIM_Y = 1
        write_csr(12'h014, 32'd1);        // GRID_DIM_Z = 1

        // Set kernel PC (start at address 0)
        write_csr(12'h008, IMEM_BASE);

        $display("Starting kernel...");
        write_csr(12'h004, 32'h1);  // GPU_CONTROL = start

        //--------------------------------------------------------------------
        // Wait for Completion
        //--------------------------------------------------------------------
        $display("");
        $display("Waiting for kernel completion...");

        fork
            begin
                wait(irq_kernel_done);
                $display("Kernel completed (irq_kernel_done asserted)");
            end
            begin
                #500000;
                $display("ERROR: Timeout waiting for kernel completion");
            end
        join_any
        disable fork;

        #100;

        //--------------------------------------------------------------------
        // Check Results
        //--------------------------------------------------------------------
        $display("");
        $display("------------------------------------------------------------");
        $display("Results:");
        $display("------------------------------------------------------------");

        begin
            real rtl_result;
            rtl_result = fp32_to_real(global_mem[(ADDR_C_BASE >> 2)]);
            $display("C[0][0]: RTL = %8.4f, Expected = %8.4f, Match = %s",
                     rtl_result, C_expected[0],
                     ((rtl_result - C_expected[0]) < 0.01 &&
                      (rtl_result - C_expected[0]) > -0.01) ? "YES" : "NO");
            $display("Raw FP32 hex: 0x%08h", global_mem[(ADDR_C_BASE >> 2)]);
        end

        $display("");
        $display("============================================================");

        // ============================================================
        // Performance Counter Report (RALPH-7)
        // ============================================================
        $display("");
        $display("============================================================");
        $display("Performance Counters:");
        $display("============================================================");
        
        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        $display("  Cycles                   = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
        $display("  Instructions             = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h102; @(posedge clk); @(posedge clk);
        $display("  Dual Issued              = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h103; @(posedge clk); @(posedge clk);
        $display("  Stall: Scoreboard        = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h104; @(posedge clk); @(posedge clk);
        $display("  Stall: I-Fetch           = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h105; @(posedge clk); @(posedge clk);
        $display("  Stall: Memory            = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h108; @(posedge clk); @(posedge clk);
        $display("  FU: ALU Active           = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h109; @(posedge clk); @(posedge clk);
        $display("  FU: FPU Active           = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h10C; @(posedge clk); @(posedge clk);
        $display("  FU: LDST Active          = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h118; @(posedge clk); @(posedge clk);
        $display("  Branch Taken             = %0d", csr_rd_data);
        @(posedge clk); csr_addr <= 12'h119; @(posedge clk); @(posedge clk);
        $display("  Branch Divergent         = %0d", csr_rd_data);
        
        // IPC
        begin
            reg [31:0] perf_c, perf_i;
            @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
            perf_c = csr_rd_data;
            @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
            perf_i = csr_rd_data;
            if (perf_c > 0)
                $display("  IPC = %0d / %0d = %f", perf_i, perf_c,
                         $itor(perf_i) / $itor(perf_c));
        end
        $display("============================================================");

        $display("Test Complete");
        $display("============================================================");

        #100;
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #200000;
        $display("TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule
