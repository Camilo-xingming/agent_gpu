//============================================================================
// RalphGPU - 2-Layer MLP Forward Pass Test
// X[4] -> W1[4x4]+ReLU -> H[4] -> W2[1x4] -> Y[1]
// Uses ralph_gpu_top with PTX instructions
//============================================================================

`timescale 1ns / 1ps

`include "gpu_defines.vh"

module tb_tiny_mlp;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;  // 100 MHz

    // Memory layout
    localparam ADDR_X_BASE  = 32'h0000_0100;  // X[4] - FP16
    localparam ADDR_W1_BASE = 32'h0000_0200;  // W1[4x4] - FP16
    localparam ADDR_H_BASE  = 32'h0000_0300;  // H[4] - FP32
    localparam ADDR_W2_BASE = 32'h0000_0400;  // W2[4] - FP16
    localparam ADDR_Y_BASE  = 32'h0000_0500;  // Y[1] - FP32

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
        .NUM_SM(1)
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
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:511];

    // Instruction encoding helpers

    function [31:0] encode_mov_special;
        input [4:0] rd;
        input [4:0] sreg;
        begin
            encode_mov_special = {`OP_MOV_SPECIAL, rd, sreg, 16'b0};
        end
    endfunction

    function [31:0] encode_ld_global;
        input [4:0] rd;
        input [4:0] ra;
        begin
            encode_ld_global = {`OP_LD_GLOBAL, rd, ra, 16'b0};
        end
    endfunction

    function [31:0] encode_st_global;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rb, 11'b0};
        end
    endfunction

    function [31:0] encode_fp16_mul;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_fp16_mul = {`OP_FP16_ARITH, rd, ra, rb, 5'b0, `FP16_MUL_F32};
        end
    endfunction

    function [31:0] encode_fp32_add;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_fp32_add = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, `FP_ADD};
        end
    endfunction

    function [31:0] encode_fp32_max;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_fp32_max = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, `FP_MAX};
        end
    endfunction

    function [31:0] encode_fp32_mul;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_fp32_mul = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, `FP_MUL};
        end
    endfunction

    function [31:0] encode_cvt_f32_f16;
        input [4:0] rd;
        input [4:0] ra;
        begin
            encode_cvt_f32_f16 = {`OP_ALU, rd, ra, 5'b0, 5'b0, `CVT_F32_F16};
        end
    endfunction

    function [31:0] encode_alu_add;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_alu_add = {`OP_ALU, rd, ra, rb, 5'b0, `FUNC_ADD};
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
    // Global Memory Model (Data)
    //------------------------------------------------------------------------
    reg [31:0] global_mem [0:4095];

    reg [31:0] pending_aw_addr;
    reg [3:0]  pending_aw_id;
    reg        pending_aw_valid;

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
                axi_read_delay <= 3'd2;
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

            if (m_axi_awvalid && m_axi_awready) begin
                pending_aw_addr <= m_axi_awaddr;
                pending_aw_id <= m_axi_awid;
                pending_aw_valid <= 1'b1;
            end

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
    integer i, pc;
    reg test_pass;

    initial begin
        $display("============================================================");
        $display("RalphGPU - 2-Layer MLP Forward Pass Test");
        $display("X[4] -> W1[4x4]+ReLU -> H[4] -> W2[1x4] -> Y[1]");
        $display("============================================================");
        $display("");

        // Initialize signals
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        test_pass = 1;

        // Initialize instruction memory to NOPs
        for (i = 0; i < 512; i = i + 1)
            instruction_mem[i] = encode_nop();

        // Initialize global memory
        for (i = 0; i < 4096; i = i + 1)
            global_mem[i] = 32'b0;

        //--------------------------------------------------------------------
        // Initialize Data Memory
        //--------------------------------------------------------------------
        // X at 0x0100: [1.0, 0.5, -1.0, 2.0] (FP16 in lower 16 bits)
        global_mem[ADDR_X_BASE >> 2]       = {16'b0, 16'h3C00};  // 1.0
        global_mem[(ADDR_X_BASE >> 2) + 1] = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_X_BASE >> 2) + 2] = {16'b0, 16'hBC00};  // -1.0
        global_mem[(ADDR_X_BASE >> 2) + 3] = {16'b0, 16'h4000};  // 2.0

        // W1 at 0x0200: 4x4 matrix (row-major, FP16)
        // Row 0: [1.0, 0.5, -0.5, 0.25]
        global_mem[(ADDR_W1_BASE >> 2) + 0]  = {16'b0, 16'h3C00};  // 1.0
        global_mem[(ADDR_W1_BASE >> 2) + 1]  = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_W1_BASE >> 2) + 2]  = {16'b0, 16'hB800};  // -0.5
        global_mem[(ADDR_W1_BASE >> 2) + 3]  = {16'b0, 16'h3400};  // 0.25
        // Row 1: [-1.0, 2.0, 0.5, -0.5]
        global_mem[(ADDR_W1_BASE >> 2) + 4]  = {16'b0, 16'hBC00};  // -1.0
        global_mem[(ADDR_W1_BASE >> 2) + 5]  = {16'b0, 16'h4000};  // 2.0
        global_mem[(ADDR_W1_BASE >> 2) + 6]  = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_W1_BASE >> 2) + 7]  = {16'b0, 16'hB800};  // -0.5
        // Row 2: [0.5, -1.0, 1.0, 0.5]
        global_mem[(ADDR_W1_BASE >> 2) + 8]  = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_W1_BASE >> 2) + 9]  = {16'b0, 16'hBC00};  // -1.0
        global_mem[(ADDR_W1_BASE >> 2) + 10] = {16'b0, 16'h3C00};  // 1.0
        global_mem[(ADDR_W1_BASE >> 2) + 11] = {16'b0, 16'h3800};  // 0.5
        // Row 3: [2.0, 1.0, 0.5, -1.0]
        global_mem[(ADDR_W1_BASE >> 2) + 12] = {16'b0, 16'h4000};  // 2.0
        global_mem[(ADDR_W1_BASE >> 2) + 13] = {16'b0, 16'h3C00};  // 1.0
        global_mem[(ADDR_W1_BASE >> 2) + 14] = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_W1_BASE >> 2) + 15] = {16'b0, 16'hBC00};  // -1.0

        // W2 at 0x0400: [1.0, 0.5, -1.0, 2.0] (FP16)
        global_mem[ADDR_W2_BASE >> 2]       = {16'b0, 16'h3C00};  // 1.0
        global_mem[(ADDR_W2_BASE >> 2) + 1] = {16'b0, 16'h3800};  // 0.5
        global_mem[(ADDR_W2_BASE >> 2) + 2] = {16'b0, 16'hBC00};  // -1.0
        global_mem[(ADDR_W2_BASE >> 2) + 3] = {16'b0, 16'h4000};  // 2.0

        $display("Memory initialized:");
        $display("  X  at 0x%04h: [1.0, 0.5, -1.0, 2.0]", ADDR_X_BASE);
        $display("  W1 at 0x%04h: 4x4 matrix", ADDR_W1_BASE);
        $display("  H  at 0x%04h: output buffer", ADDR_H_BASE);
        $display("  W2 at 0x%04h: [1.0, 0.5, -1.0, 2.0]", ADDR_W2_BASE);
        $display("  Y  at 0x%04h: output scalar", ADDR_Y_BASE);
        $display("");

        //--------------------------------------------------------------------
        // Generate MLP Kernel
        //--------------------------------------------------------------------
        // Register allocation:
        // R0=zero, R1-R3=temps, R4=accumulator
        // R5-R8=H[0]-H[3] (kept in registers for layer 2)
        // R10=X_BASE, R11=W1_BASE, R12=H_BASE, R13=W2_BASE, R14=Y_BASE
        // R15=4(stride), R16=16(row stride)
        // R17-R18=address pointers
        // R20=1, R21=FP32 zero
        //--------------------------------------------------------------------
        pc = 0;

        // === Setup base addresses ===
        // R20 = ntid.x = 1
        instruction_mem[pc] = encode_mov_special(5'd20, `SREG_NTID_X); pc = pc + 1;

        // R15 = 4 (byte stride)
        instruction_mem[pc] = encode_alu_add(5'd15, 5'd20, 5'd20); pc = pc + 1;  // R15 = 2
        instruction_mem[pc] = encode_alu_add(5'd15, 5'd15, 5'd15); pc = pc + 1;  // R15 = 4

        // R16 = 16 (row stride = 4 elements * 4 bytes)
        instruction_mem[pc] = encode_alu_add(5'd16, 5'd15, 5'd15); pc = pc + 1;  // R16 = 8
        instruction_mem[pc] = encode_alu_add(5'd16, 5'd16, 5'd16); pc = pc + 1;  // R16 = 16

        // Build 256 = 2^8 from R15=4: need 6 more doublings
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd15, 5'd15); pc = pc + 1;  // R10 = 8
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd10, 5'd10); pc = pc + 1;  // R10 = 16
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd10, 5'd10); pc = pc + 1;  // R10 = 32
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd10, 5'd10); pc = pc + 1;  // R10 = 64
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd10, 5'd10); pc = pc + 1;  // R10 = 128
        instruction_mem[pc] = encode_alu_add(5'd10, 5'd10, 5'd10); pc = pc + 1;  // R10 = 256 = X_BASE

        // R11 = 512 = W1_BASE
        instruction_mem[pc] = encode_alu_add(5'd11, 5'd10, 5'd10); pc = pc + 1;  // R11 = 512

        // R12 = 768 = H_BASE = 256 + 512
        instruction_mem[pc] = encode_alu_add(5'd12, 5'd10, 5'd11); pc = pc + 1;  // R12 = 768

        // R13 = 1024 = W2_BASE = 512 + 512
        instruction_mem[pc] = encode_alu_add(5'd13, 5'd11, 5'd11); pc = pc + 1;  // R13 = 1024

        // R14 = 1280 = Y_BASE = 1024 + 256
        instruction_mem[pc] = encode_alu_add(5'd14, 5'd13, 5'd10); pc = pc + 1;  // R14 = 1280

        // R21 = 0 (FP32 zero for ReLU) — R0 is always 0
        instruction_mem[pc] = encode_alu_add(5'd21, 5'd0, 5'd0); pc = pc + 1;  // R21 = 0

        // ====================================================================
        // Layer 1: H = ReLU(W1 * X)
        // For each row i of W1, compute dot product with X, then ReLU
        // ====================================================================

        // --- H[0] = ReLU(W1[0] . X) ---
        // R4 = 0 (accumulator)
        instruction_mem[pc] = encode_alu_add(5'd4, 5'd0, 5'd0); pc = pc + 1;
        // R17 = X_BASE
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd10, 5'd0); pc = pc + 1;
        // R18 = W1_BASE + 0*16 = W1_BASE
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd11, 5'd0); pc = pc + 1;

        // j=0: X[0] * W1[0][0]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;   // R1 = X[0]
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;   // R2 = W1[0][0]
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;  // R3 = R1*R2 (FP32)
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;  // R4 += R3
        // Advance pointers
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1; // X ptr += 4
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1; // W1 ptr += 4

        // j=1: X[1] * W1[0][1]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;

        // j=2: X[2] * W1[0][2]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;

        // j=3: X[3] * W1[0][3]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;

        // ReLU: R5 = max(R4, R21=0)
        instruction_mem[pc] = encode_fp32_max(5'd5, 5'd4, 5'd21); pc = pc + 1;
        // Store H[0]
        instruction_mem[pc] = encode_st_global(5'd12, 5'd5); pc = pc + 1;

        // --- H[1] = ReLU(W1[1] . X) ---
        instruction_mem[pc] = encode_alu_add(5'd4, 5'd0, 5'd0); pc = pc + 1;  // acc = 0
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd10, 5'd0); pc = pc + 1; // R17 = X_BASE
        // R18 = W1_BASE + 16 (row 1)
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd11, 5'd16); pc = pc + 1;

        // j=0
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=1
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=2
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=3
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;

        // ReLU: R6 = max(R4, 0)
        instruction_mem[pc] = encode_fp32_max(5'd6, 5'd4, 5'd21); pc = pc + 1;
        // Store H[1]: addr = R12 + R15
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd12, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_st_global(5'd17, 5'd6); pc = pc + 1;

        // --- H[2] = ReLU(W1[2] . X) ---
        instruction_mem[pc] = encode_alu_add(5'd4, 5'd0, 5'd0); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd10, 5'd0); pc = pc + 1;
        // R18 = W1_BASE + 32 (row 2)
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd11, 5'd16); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd16); pc = pc + 1;

        // j=0
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=1
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=2
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=3
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;

        // ReLU: R7 = max(R4, 0)
        instruction_mem[pc] = encode_fp32_max(5'd7, 5'd4, 5'd21); pc = pc + 1;
        // Store H[2]: addr = R12 + 8
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd12, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_st_global(5'd17, 5'd7); pc = pc + 1;

        // --- H[3] = ReLU(W1[3] . X) ---
        instruction_mem[pc] = encode_alu_add(5'd4, 5'd0, 5'd0); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd10, 5'd0); pc = pc + 1;
        // R18 = W1_BASE + 48 (row 3)
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd11, 5'd16); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd16); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd16); pc = pc + 1;

        // j=0
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=1
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=2
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd18, 5'd18, 5'd15); pc = pc + 1;
        // j=3
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd18); pc = pc + 1;
        instruction_mem[pc] = encode_fp16_mul(5'd3, 5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;

        // ReLU: R8 = max(R4, 0)
        instruction_mem[pc] = encode_fp32_max(5'd8, 5'd4, 5'd21); pc = pc + 1;
        // Store H[3]: addr = R12 + 12
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd12, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;
        instruction_mem[pc] = encode_st_global(5'd17, 5'd8); pc = pc + 1;

        // ====================================================================
        // Layer 2: Y = W2 . H (no activation)
        // Y = W2[0]*H[0] + W2[1]*H[1] + W2[2]*H[2] + W2[3]*H[3]
        // H values are FP32 in R5-R8, W2 values are FP16 in memory
        // ====================================================================

        // R4 = 0 (accumulator)
        instruction_mem[pc] = encode_alu_add(5'd4, 5'd0, 5'd0); pc = pc + 1;
        // R17 = W2_BASE
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd13, 5'd0); pc = pc + 1;

        // j=0: W2[0] * H[0]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;       // R1 = W2[0] (FP16)
        instruction_mem[pc] = encode_cvt_f32_f16(5'd2, 5'd1); pc = pc + 1;      // R2 = cvt_f32(R1)
        instruction_mem[pc] = encode_fp32_mul(5'd3, 5'd2, 5'd5); pc = pc + 1;   // R3 = W2[0]*H[0]
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;   // acc += R3
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1; // W2 ptr += 4

        // j=1: W2[1] * H[1]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_cvt_f32_f16(5'd2, 5'd1); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_mul(5'd3, 5'd2, 5'd6); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;

        // j=2: W2[2] * H[2]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_cvt_f32_f16(5'd2, 5'd1); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_mul(5'd3, 5'd2, 5'd7); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;
        instruction_mem[pc] = encode_alu_add(5'd17, 5'd17, 5'd15); pc = pc + 1;

        // j=3: W2[3] * H[3]
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd17); pc = pc + 1;
        instruction_mem[pc] = encode_cvt_f32_f16(5'd2, 5'd1); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_mul(5'd3, 5'd2, 5'd8); pc = pc + 1;
        instruction_mem[pc] = encode_fp32_add(5'd4, 5'd4, 5'd3); pc = pc + 1;

        // Store Y: mem[R14] = R4
        instruction_mem[pc] = encode_st_global(5'd14, 5'd4); pc = pc + 1;

        // Exit
        instruction_mem[pc] = encode_exit(); pc = pc + 1;

        $display("Kernel size: %0d instructions", pc);
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

        write_csr(12'h018, 32'd1);        // BLOCK_DIM_X = 1
        write_csr(12'h01C, 32'd1);        // BLOCK_DIM_Y = 1
        write_csr(12'h020, 32'd1);        // BLOCK_DIM_Z = 1
        write_csr(12'h00C, 32'd1);        // GRID_DIM_X = 1
        write_csr(12'h010, 32'd1);        // GRID_DIM_Y = 1
        write_csr(12'h014, 32'd1);        // GRID_DIM_Z = 1
        write_csr(12'h008, IMEM_BASE);    // Kernel PC

        $display("Starting kernel...");
        write_csr(12'h004, 32'h1);        // GPU_CONTROL = start

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
                #1000000;
                $display("ERROR: Timeout waiting for kernel completion");
            end
        join_any
        disable fork;

        #100;

        //--------------------------------------------------------------------
        // Check Results
        //--------------------------------------------------------------------
        $display("");
        $display("============================================================");
        $display("Results:");
        $display("============================================================");

        // Check H values
        $display("");
        $display("Hidden layer H (FP32):");
        $display("  H[0] = 0x%08h (expected 0x40100000 = 2.25)", global_mem[ADDR_H_BASE >> 2]);
        $display("  H[1] = 0x%08h (expected 0x00000000 = 0.0)",  global_mem[(ADDR_H_BASE >> 2) + 1]);
        $display("  H[2] = 0x%08h (expected 0x00000000 = 0.0)",  global_mem[(ADDR_H_BASE >> 2) + 2]);
        $display("  H[3] = 0x%08h (expected 0x00000000 = 0.0)",  global_mem[(ADDR_H_BASE >> 2) + 3]);

        // Check Y value
        $display("");
        $display("Output Y (FP32):");
        $display("  Y    = 0x%08h (expected 0x40100000 = 2.25)", global_mem[ADDR_Y_BASE >> 2]);

        // Verify
        $display("");
        if (global_mem[ADDR_Y_BASE >> 2] !== 32'h40100000) begin
            $display("FAIL: Y != 2.25 (0x40100000)");
            test_pass = 0;
        end
        if (global_mem[ADDR_H_BASE >> 2] !== 32'h40100000) begin
            $display("FAIL: H[0] != 2.25 (0x40100000)");
            test_pass = 0;
        end
        if (global_mem[(ADDR_H_BASE >> 2) + 1] !== 32'h00000000) begin
            $display("FAIL: H[1] != 0.0");
            test_pass = 0;
        end

        $display("");
        if (test_pass)
            $display("*** PASS ***");
        else
            $display("*** FAIL ***");

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
        #2000000;
        $display("TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule
