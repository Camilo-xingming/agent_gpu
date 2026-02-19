//============================================================================
// RalphGPU - 2-Layer MLP Forward Pass Test
// X[4] -> W1[4x4]+ReLU -> H[4] -> W2[1x4] -> Y[1]
// Uses ralph_gpu_top with PTX instructions
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_multiwarp_vecadd;

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


    //========================================================================
    // Encoding helpers for this test
    //========================================================================
    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        begin
            encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
        end
    endfunction

    function [31:0] encode_shl;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_shl = {`OP_ALU, rd, ra, rb, 5'b0, `FUNC_SHL};
        end
    endfunction

    //========================================================================
    // Kernel: C[tid] = A[tid] + B[tid], 128 elements, 4 warps
    //========================================================================
    localparam ADDR_A = 32'h0000;
    localparam ADDR_B = 32'h0200;
    localparam ADDR_C = 32'h0400;

    integer j, errors;
    initial begin
        for (j = 0; j < 512; j = j + 1)
            instruction_mem[j] = encode_nop();

        // Reordered kernel with NOP padding for WBQ latency (no per-reg scoreboard)
        // Phase 1: All immediates (no dependencies)
        instruction_mem[0]  = encode_mov_special(5'd0, 5'd0);     // r0 = %tid.x
        instruction_mem[1]  = encode_mov_imm(5'd11, 16'd2);       // r11 = 2
        instruction_mem[2]  = encode_mov_imm(5'd2, 16'h0000);     // r2 = base A
        instruction_mem[3]  = encode_mov_imm(5'd5, 16'h0200);     // r5 = base B
        instruction_mem[4]  = encode_mov_imm(5'd9, 16'h0400);     // r9 = base C
        instruction_mem[5]  = encode_nop();
        instruction_mem[6]  = encode_nop();
        instruction_mem[7]  = encode_nop();
        instruction_mem[8]  = encode_nop();
        instruction_mem[9]  = encode_nop();
        instruction_mem[10] = encode_nop();
        instruction_mem[11] = encode_nop();
        instruction_mem[12] = encode_nop();
        // Phase 2: Address computation (depends on r0, r11, r2, r5, r9)
        instruction_mem[13] = encode_shl(5'd1, 5'd0, 5'd11);     // r1 = tid * 4
        instruction_mem[14] = encode_nop();
        instruction_mem[15] = encode_nop();
        instruction_mem[16] = encode_nop();
        instruction_mem[17] = encode_nop();
        instruction_mem[18] = encode_nop();
        instruction_mem[19] = encode_nop();
        instruction_mem[20] = encode_nop();
        instruction_mem[21] = encode_nop();
        instruction_mem[22] = encode_alu_add(5'd3, 5'd2, 5'd1);  // r3 = &A[tid]
        instruction_mem[23] = encode_alu_add(5'd6, 5'd5, 5'd1);  // r6 = &B[tid]
        instruction_mem[24] = encode_alu_add(5'd10, 5'd9, 5'd1); // r10 = &C[tid]
        instruction_mem[25] = encode_nop();
        instruction_mem[26] = encode_nop();
        instruction_mem[27] = encode_nop();
        instruction_mem[28] = encode_nop();
        instruction_mem[29] = encode_nop();
        instruction_mem[30] = encode_nop();
        instruction_mem[31] = encode_nop();
        instruction_mem[32] = encode_nop();
        // Phase 3: Loads (depends on r3, r6)
        instruction_mem[33] = encode_ld_global(5'd4, 5'd3);      // r4 = A[tid]
        instruction_mem[34] = encode_ld_global(5'd7, 5'd6);      // r7 = B[tid]
        instruction_mem[35] = encode_nop();
        instruction_mem[36] = encode_nop();
        instruction_mem[37] = encode_nop();
        instruction_mem[38] = encode_nop();
        instruction_mem[39] = encode_nop();
        instruction_mem[40] = encode_nop();
        instruction_mem[41] = encode_nop();
        instruction_mem[42] = encode_nop();
        instruction_mem[43] = encode_nop();
        instruction_mem[44] = encode_nop();
        instruction_mem[45] = encode_nop();
        instruction_mem[46] = encode_nop();
        instruction_mem[47] = encode_nop();
        instruction_mem[48] = encode_nop();
        instruction_mem[49] = encode_nop();
        instruction_mem[50] = encode_nop();
        // Phase 4: Compute + Store (depends on r4, r7, r8, r10)
        instruction_mem[51] = encode_alu_add(5'd8, 5'd4, 5'd7);  // r8 = A[tid] + B[tid]
        instruction_mem[52] = encode_nop();
        instruction_mem[53] = encode_nop();
        instruction_mem[54] = encode_nop();
        instruction_mem[55] = encode_nop();
        instruction_mem[56] = encode_nop();
        instruction_mem[57] = encode_nop();
        instruction_mem[58] = encode_nop();
        instruction_mem[59] = encode_nop();
        instruction_mem[60] = encode_st_global(5'd10, 5'd8);     // C[tid] = r8
        instruction_mem[61] = encode_nop();
        instruction_mem[62] = encode_nop();
        instruction_mem[63] = encode_nop();
        instruction_mem[64] = encode_nop();
        instruction_mem[65] = encode_exit();                       // EXIT

        $display("Kernel: 14 instructions per warp, 4 warps x 32 threads");
    end

    //========================================================================
    // Test Sequence
    //========================================================================

    initial begin
        $display("============================================================");
        $display("RALPH-9 P1: Multi-Warp Vector Add (128 elements, 4 warps)");
        $display("============================================================");

        rst_n = 0; csr_wr_en = 0; csr_addr = 0; csr_wr_data = 0;

        for (i = 0; i < 4096; i = i + 1) global_mem[i] = 32'hDEADBEEF;
        for (i = 0; i < 128; i = i + 1) begin
            global_mem[(ADDR_A >> 2) + i] = i;
            global_mem[(ADDR_B >> 2) + i] = i * 10;
        end

        $display("A[0..127] = 0..127, B[0..127] = 0..1270");
        $display("Expected: C[i] = i*11");

        repeat(5) @(posedge clk); rst_n = 1; repeat(5) @(posedge clk);

        $display("Configuring kernel (128 threads = 4 warps)...");
        write_csr(12'h008, 32'd0);     // kernel_pc = 0
        write_csr(12'h018, 32'd128);   // block_dim_x = 128
        write_csr(12'h01C, 32'd1);
        write_csr(12'h020, 32'd1);
        write_csr(12'h00C, 32'd1);
        write_csr(12'h010, 32'd1);
        write_csr(12'h014, 32'd1);

        $display("Launching kernel...");
        write_csr(12'h004, 32'h1);

        // Wait with monitoring

        // Wait for completion
        begin : wait_block
            integer wait_cnt;
            wait_cnt = 0;
            while (!irq_kernel_done && wait_cnt < 60000) begin
                @(posedge clk);
                wait_cnt = wait_cnt + 1;
            end
            if (!irq_kernel_done) begin
                $display("*** TIMEOUT ***");
                $display("*** FAIL ***");
                $finish(1);
            end
        end

        repeat(100) @(posedge clk);

        $display("");
        $display("Kernel completed!");

        // Perf counters
        $display("Performance Counters:");
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
        @(posedge clk); csr_addr <= 12'h10C; @(posedge clk); @(posedge clk);
        $display("  FU: LDST Active          = %0d", csr_rd_data);

        // IPC
        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        begin : ipc_block
            integer cyc_val;
            cyc_val = csr_rd_data;
            @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
            if (cyc_val > 0)
                $display("  IPC = %0d / %0d = %f", csr_rd_data, cyc_val,
                         $itor(csr_rd_data) / $itor(cyc_val));
        end

        // Verify results
        $display("");
        $display("Verifying C[0..127]...");
        errors = 0;
        for (i = 0; i < 128; i = i + 1) begin
            if (global_mem[(ADDR_C >> 2) + i] !== i * 11) begin
                if (errors < 16)
                    $display("  C[%0d] = %0d (expected %0d) [FAIL]",
                             i, global_mem[(ADDR_C >> 2) + i], i * 11);
                errors = errors + 1;
            end
        end

        if (errors == 0) begin
            $display("  All 128 elements correct!");
            $display("  Warp 0: C[0]=%0d C[31]=%0d", global_mem[(ADDR_C>>2)+0], global_mem[(ADDR_C>>2)+31]);
            $display("  Warp 1: C[32]=%0d C[63]=%0d", global_mem[(ADDR_C>>2)+32], global_mem[(ADDR_C>>2)+63]);
            $display("  Warp 2: C[64]=%0d C[95]=%0d", global_mem[(ADDR_C>>2)+64], global_mem[(ADDR_C>>2)+95]);
            $display("  Warp 3: C[96]=%0d C[127]=%0d", global_mem[(ADDR_C>>2)+96], global_mem[(ADDR_C>>2)+127]);
            $display("");
            $display("*** PASS ***");
        end else begin
            $display("  %0d / 128 elements incorrect", errors);
            $display("*** FAIL ***");
                $finish(1);
        end

        $display("============================================================");
        $finish;
    end

endmodule
