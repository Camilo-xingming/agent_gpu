//============================================================================
// RalphGPU - 16x16 FP16 Matrix Multiplication using Tensor Core
// D = A * B where A, B are 16x16 FP16 random matrices, D is 16x16 FP32
// Uses ralph_gpu_top as DUT with full memory subsystem
//============================================================================

`timescale 1ns / 1ps

module tb_tensor_matmul_16x16_full;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    // Memory layout
    parameter MATRIX_A_BASE = 32'h0000_1000;  // A: 16x16 FP16 = 128 words (row-major, 2 FP16/word)
    parameter MATRIX_B_BASE = 32'h0000_1200;  // B: 16x16 FP16 = 128 words (transposed)
    parameter MATRIX_D_BASE = 32'h0000_1400;  // D: 16x16 FP32 = 256 words

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
    // FP16/FP32 Conversion Functions
    //------------------------------------------------------------------------
    function [15:0] real_to_fp16;
        input real val;
        reg sign;
        reg [4:0] exp;
        reg [9:0] man;
        real abs_val, frac;
        integer exp_int;
        begin
            if (val == 0.0) begin
                real_to_fp16 = 16'h0000;
            end else begin
                sign = (val < 0) ? 1 : 0;
                abs_val = (val < 0) ? -val : val;
                exp_int = 0;
                frac = abs_val;
                if (frac >= 2.0) begin
                    while (frac >= 2.0 && exp_int < 15) begin
                        frac = frac / 2.0;
                        exp_int = exp_int + 1;
                    end
                end else if (frac < 1.0 && frac > 0) begin
                    while (frac < 1.0 && exp_int > -14) begin
                        frac = frac * 2.0;
                        exp_int = exp_int - 1;
                    end
                end
                exp = exp_int + 15;
                man = (frac - 1.0) * 1024.0;
                real_to_fp16 = {sign, exp, man};
            end
        end
    endfunction

    function real fp16_to_real;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp;
        reg [9:0] man;
        real result;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            man = fp16[9:0];
            if (exp == 0 && man == 0)
                result = 0.0;
            else if (exp == 0)
                result = (man / 1024.0) * (2.0 ** (-14));
            else
                result = (1.0 + man / 1024.0) * (2.0 ** (exp - 15));
            if (sign) result = -result;
            fp16_to_real = result;
        end
    endfunction

    function [31:0] real_to_fp32;
        input real val;
        reg sign;
        reg [7:0] exp;
        reg [22:0] man;
        real abs_val, frac;
        integer exp_int;
        begin
            if (val == 0.0) begin
                real_to_fp32 = 32'h00000000;
            end else begin
                sign = (val < 0) ? 1 : 0;
                abs_val = (val < 0) ? -val : val;
                exp_int = 0;
                frac = abs_val;
                if (frac >= 2.0) begin
                    while (frac >= 2.0 && exp_int < 127) begin
                        frac = frac / 2.0;
                        exp_int = exp_int + 1;
                    end
                end else if (frac < 1.0 && frac > 0) begin
                    while (frac < 1.0 && exp_int > -126) begin
                        frac = frac * 2.0;
                        exp_int = exp_int - 1;
                    end
                end
                exp = exp_int + 127;
                man = (frac - 1.0) * 8388608.0;
                real_to_fp32 = {sign, exp, man};
            end
        end
    endfunction

    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] man;
        real result;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            man = fp32[22:0];
            if (exp == 0 && man == 0)
                result = 0.0;
            else if (exp == 255)
                result = 1.0/0.0;
            else
                result = (1.0 + man / 8388608.0) * (2.0 ** (exp - 127));
            if (sign) result = -result;
            fp32_to_real = result;
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Matrices
    //------------------------------------------------------------------------
    reg [15:0] matrix_A [0:255];
    reg [15:0] matrix_B [0:255];
    reg [31:0] expected_D [0:255];

    //------------------------------------------------------------------------
    // Global Memory
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:16383];

    //------------------------------------------------------------------------
    // Instruction Memory
    //------------------------------------------------------------------------
    reg [31:0] imem [0:2047];

    reg [31:0] total_imem_fetches;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
            total_imem_fetches <= 0;
        end else begin
            if (imem_req) begin
                imem_data <= {imem[imem_addr[12:2] + 1], imem[imem_addr[12:2]]};
                imem_valid <= 1'b1;
                total_imem_fetches <= total_imem_fetches + 1;
                // Show instruction opcode bits [31:26] for first instruction
                if (total_imem_fetches < 50 || imem[imem_addr[12:2]][31:26] == `OP_ST_GLOBAL) begin
                    $display("[FETCH] cnt=%0d PC=0x%04x op0=0x%02x op1=0x%02x",
                             total_imem_fetches, imem_addr,
                             imem[imem_addr[12:2]][31:26], imem[imem_addr[12:2]+1][31:26]);
                end
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // AXI Memory Model
    //------------------------------------------------------------------------
    reg [31:0] axi_read_addr;
    reg [7:0]  axi_read_len, axi_read_cnt;
    reg        axi_read_active;

    integer read_trace_cnt;
    initial read_trace_cnt = 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            axi_read_active <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready && !axi_read_active) begin
                axi_read_addr <= m_axi_araddr;
                axi_read_len <= m_axi_arlen;
                axi_read_cnt <= 0;
                axi_read_active <= 1'b1;
                m_axi_arready <= 1'b0;
                m_axi_rid <= m_axi_arid;
                if (read_trace_cnt < 20) begin
                    $display("[AXI-RD] addr=0x%08x len=%0d gmem_idx=%0d",
                        m_axi_araddr, m_axi_arlen, m_axi_araddr >> 2);
                end
            end else if (axi_read_active) begin
                m_axi_rdata <= gmem[(axi_read_addr >> 2) + axi_read_cnt];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= (axi_read_cnt == axi_read_len);
                if (m_axi_rvalid && m_axi_rready) begin
                    if (read_trace_cnt < 20) begin
                        $display("[AXI-RD] data=0x%08x from gmem[%0d]",
                            m_axi_rdata, (axi_read_addr >> 2) + axi_read_cnt);
                        read_trace_cnt = read_trace_cnt + 1;
                    end
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

    reg [31:0] axi_write_addr;
    reg        axi_write_active;
    integer    write_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            axi_write_active <= 1'b0;
            write_count <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                axi_write_addr <= m_axi_awaddr;
                axi_write_active <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end
            if (axi_write_active && m_axi_wvalid && m_axi_wready) begin
                gmem[axi_write_addr >> 2] <= m_axi_wdata;
                if (write_count < 20) begin
                    $display("[AXI-WR] addr=0x%08x data=0x%08x idx=%0d", axi_write_addr, m_axi_wdata, axi_write_addr >> 2);
                end
                write_count <= write_count + 1;
                if (m_axi_wlast) begin
                    axi_write_active <= 1'b0;
                    m_axi_bvalid <= 1'b1;
                end else begin
                    axi_write_addr <= axi_write_addr + 4;
                end
            end
            if (m_axi_bvalid && m_axi_bready)
                m_axi_bvalid <= 1'b0;
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
            csr_addr <= addr;
            csr_wr_data <= data;
            csr_wr_en <= 1'b1;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    //------------------------------------------------------------------------
    // Instruction Encoding
    //------------------------------------------------------------------------
    function [31:0] encode_mov_special;
        input [4:0] rd, sreg;
        encode_mov_special = {`OP_MOV_SPECIAL, rd, sreg, 16'b0};
    endfunction

    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm16;
        encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm16};
    endfunction

    function [31:0] encode_alu_imm;
        input [4:0] rd, ra;
        input [5:0] func;
        input [9:0] imm10;
        encode_alu_imm = {`OP_ALU_IMM, rd, ra, func, imm10};
    endfunction

    function [31:0] encode_alu_reg;
        input [4:0] rd, ra, rb;
        input [5:0] func;
        encode_alu_reg = {`OP_ALU, rd, ra, rb, 5'b0, func};
    endfunction

    function [31:0] encode_ld_global;
        input [4:0] rd, ra;
        encode_ld_global = {`OP_LD_GLOBAL, rd, ra, 16'b0};
    endfunction

    function [31:0] encode_st_global;
        input [4:0] rs, ra;
        // Format: {opcode[31:26], rd[25:21], ra[20:16], rb[15:11], rc[10:6], func[5:0]}
        // SM reads: address from rf_rd_data_a (ra), data from rf_rd_data_b (rb)
        // Total: 6 + 5 + 5 + 5 + 5 + 6 = 32 bits
        encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rs, 5'b0, 6'b0};
    endfunction

    function [31:0] encode_wmma_mma;
        input [4:0] rd, ra, rb, rc;
        input [2:0] dtype, shape;
        encode_wmma_mma = {`OP_WMMA_MMA, rd, ra, rb, rc, shape, dtype};
    endfunction

    function [31:0] encode_exit;
        encode_exit = {`OP_EXIT, 26'b0};
    endfunction

    //------------------------------------------------------------------------
    // Generate Program: Each of 16 threads computes one row of D
    //------------------------------------------------------------------------
    integer pc, col;
    initial begin
        for (pc = 0; pc < 2048; pc = pc + 1)
            imem[pc] = {`OP_NOP, 26'b0};

        pc = 0;

        // r0 = tid.x (row index, 0-15)
        imem[pc] = encode_mov_special(5'd0, 5'd0);
        pc = pc + 1;

        // Calculate A row base: r1 = A_BASE + row * 32
        imem[pc] = encode_alu_imm(5'd2, 5'd0, `FUNC_SHL, 10'd5);  // r2 = row * 32
        pc = pc + 1;
        imem[pc] = encode_mov_imm(5'd1, 16'h1000);  // r1 = 0x1000
        pc = pc + 1;
        imem[pc] = encode_alu_reg(5'd1, 5'd1, 5'd2, `FUNC_ADD);  // r1 = A_BASE + row*32
        pc = pc + 1;

        // Calculate D row base: r3 = D_BASE + row * 64
        imem[pc] = encode_alu_imm(5'd4, 5'd0, `FUNC_SHL, 10'd6);  // r4 = row * 64
        pc = pc + 1;
        imem[pc] = encode_mov_imm(5'd3, 16'h1400);  // r3 = 0x1400
        pc = pc + 1;
        imem[pc] = encode_alu_reg(5'd3, 5'd3, 5'd4, `FUNC_ADD);  // r3 = D_BASE + row*64
        pc = pc + 1;

        // B base: r5 = 0x1200
        imem[pc] = encode_mov_imm(5'd5, 16'h1200);
        pc = pc + 1;

        // Save A base for reload: r28 = r1
        imem[pc] = encode_alu_reg(5'd28, 5'd1, 5'd0, `FUNC_ADD);
        pc = pc + 1;

        // Load A row (8 words) into r10-r17
        imem[pc] = encode_ld_global(5'd10, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd11, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd12, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd13, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd14, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd15, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd16, 5'd1); pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_ADD, 10'd4); pc = pc + 1;
        imem[pc] = encode_ld_global(5'd17, 5'd1); pc = pc + 1;

        // Process all 16 columns (unrolled)
        for (col = 0; col < 16; col = col + 1) begin
            // r7 = B_BASE + col * 32
            imem[pc] = encode_mov_imm(5'd7, MATRIX_B_BASE[15:0] + col * 32);
            pc = pc + 1;

            // Load B column into r20-r27
            imem[pc] = encode_ld_global(5'd20, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd21, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd22, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd23, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd24, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd25, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd26, 5'd7); pc = pc + 1;
            imem[pc] = encode_alu_imm(5'd7, 5'd7, `FUNC_ADD, 10'd4); pc = pc + 1;
            imem[pc] = encode_ld_global(5'd27, 5'd7); pc = pc + 1;

            // Initialize accumulator r30 = 0
            imem[pc] = encode_mov_imm(5'd30, 16'h0000);
            pc = pc + 1;

            // 8 WMMA operations
            imem[pc] = encode_wmma_mma(5'd30, 5'd10, 5'd20, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd11, 5'd21, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd12, 5'd22, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd13, 5'd23, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd14, 5'd24, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd15, 5'd25, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd16, 5'd26, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;
            imem[pc] = encode_wmma_mma(5'd30, 5'd17, 5'd27, 5'd30, `TC_DATA_FP16, 3'b000); pc = pc + 1;

            // Store D[row][col]
            imem[pc] = encode_st_global(5'd30, 5'd3);
            pc = pc + 1;

            // Advance D pointer
            imem[pc] = encode_alu_imm(5'd3, 5'd3, `FUNC_ADD, 10'd4);
            pc = pc + 1;
        end

        // EXIT
        imem[pc] = encode_exit();
        $display("Program size: %0d instructions", pc + 1);
    end

    //------------------------------------------------------------------------
    // Initialize Matrices
    //------------------------------------------------------------------------
    integer i, j, k;
    integer seed;
    real sum, a_val, b_val;

    initial begin
        seed = 42;

        // Generate simple test matrices (small integer values)
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                // A: row i has value (i+1) in all columns
                matrix_A[i*16 + j] = real_to_fp16(1.0 + (i % 4) * 0.5);
                // B: column j has value (j+1) in all rows
                matrix_B[i*16 + j] = real_to_fp16(1.0 + (j % 4) * 0.5);
            end
        end

        // Compute expected D = A * B
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                sum = 0.0;
                for (k = 0; k < 16; k = k + 1) begin
                    a_val = fp16_to_real(matrix_A[i*16 + k]);
                    b_val = fp16_to_real(matrix_B[k*16 + j]);
                    sum = sum + a_val * b_val;
                end
                expected_D[i*16 + j] = real_to_fp32(sum);
            end
        end

        // Initialize global memory
        // A: row-major, 2 FP16 per word
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                gmem[(MATRIX_A_BASE >> 2) + i*8 + j] = {matrix_A[i*16 + j*2 + 1], matrix_A[i*16 + j*2]};
            end
        end

        // B: TRANSPOSED, 2 FP16 per word
        for (j = 0; j < 16; j = j + 1) begin
            for (k = 0; k < 8; k = k + 1) begin
                gmem[(MATRIX_B_BASE >> 2) + j*8 + k] = {matrix_B[(k*2+1)*16 + j], matrix_B[(k*2)*16 + j]};
            end
        end

        // Clear D area
        for (i = 0; i < 256; i = i + 1) begin
            gmem[(MATRIX_D_BASE >> 2) + i] = 32'hDEADBEEF;
        end

        // Debug: verify memory initialization
        $display("Memory verification:");
        $display("  A[0][0:1] at gmem[%0d] = 0x%08x (expect {A[0][1],A[0][0]}={%04x,%04x})",
            MATRIX_A_BASE >> 2, gmem[MATRIX_A_BASE >> 2],
            matrix_A[1], matrix_A[0]);
        $display("  B col0 at gmem[%0d] = 0x%08x (expect {B[1][0],B[0][0]}={%04x,%04x})",
            MATRIX_B_BASE >> 2, gmem[MATRIX_B_BASE >> 2],
            matrix_B[16], matrix_B[0]);
    end

    //------------------------------------------------------------------------
    // Main Test
    //------------------------------------------------------------------------
    reg [31:0] start_time, end_time;
    integer pass_count, fail_count;
    real rtl_val, exp_val, error;

    initial begin
        $display("");
        $display("============================================================");
        $display("RalphGPU 16x16 FP16 Matrix Multiplication - Tensor Core");
        $display("============================================================");
        $display("");

        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        m_axi_bresp = 2'b00;
        m_axi_rresp = 2'b00;

        repeat(20) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        // Display sample values
        $display("Sample A[0][0..3]: %f %f %f %f",
            fp16_to_real(matrix_A[0]), fp16_to_real(matrix_A[1]),
            fp16_to_real(matrix_A[2]), fp16_to_real(matrix_A[3]));
        $display("Sample B[0..3][0]: %f %f %f %f",
            fp16_to_real(matrix_B[0]), fp16_to_real(matrix_B[16]),
            fp16_to_real(matrix_B[32]), fp16_to_real(matrix_B[48]));
        $display("Expected D[0][0]: %f", fp32_to_real(expected_D[0]));
        $display("");

        csr_write(12'h008, 32'd0);
        csr_write(12'h00C, 32'd1);
        csr_write(12'h010, 32'd1);
        csr_write(12'h014, 32'd1);
        csr_write(12'h018, 32'd16);
        csr_write(12'h01C, 32'd1);
        csr_write(12'h020, 32'd1);

        $display("Starting kernel (16 threads)...");
        start_time = $time;
        csr_write(12'h004, 32'd1);

        fork
            wait(irq_kernel_done);
            begin
                #50_000_000;
                $display("ERROR: Timeout!");
            end
        join_any
        disable fork;

        end_time = $time;
        $display("Kernel completed in %0d cycles, total_imem_fetches=%0d", (end_time - start_time) / CLK_PERIOD, total_imem_fetches);

        repeat(100) @(posedge clk);

        // Verify results
        $display("");
        $display("Verification (first 4x4 block):");
        pass_count = 0;
        fail_count = 0;

        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                rtl_val = fp32_to_real(gmem[(MATRIX_D_BASE >> 2) + i*16 + j]);
                exp_val = fp32_to_real(expected_D[i*16 + j]);
                if (exp_val != 0)
                    error = ((rtl_val - exp_val) / exp_val) * 100.0;
                else
                    error = rtl_val * 100.0;

                if (error < 5.0 && error > -5.0) begin
                    pass_count = pass_count + 1;
                end else begin
                    $display("  D[%0d][%0d]: FAIL (RTL=%f, Exp=%f)", i, j, rtl_val, exp_val);
                    fail_count = fail_count + 1;
                end
            end
        end

        // Count total correct
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                rtl_val = fp32_to_real(gmem[(MATRIX_D_BASE >> 2) + i*16 + j]);
                exp_val = fp32_to_real(expected_D[i*16 + j]);
                if (exp_val != 0)
                    error = ((rtl_val - exp_val) / exp_val) * 100.0;
                else
                    error = rtl_val * 100.0;
                if (error < 5.0 && error > -5.0 && i*16+j >= 16)
                    pass_count = pass_count + 1;
                else if (i*16+j >= 16)
                    fail_count = fail_count + 1;
            end
        end

        $display("");
        $display("============================================================");
        $display("Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("Writes: %0d", write_count);
        if (fail_count == 0)
            $display("TEST PASSED");
        else
            $display("TEST FAILED");
        $display("============================================================");

        $finish;
    end

endmodule
