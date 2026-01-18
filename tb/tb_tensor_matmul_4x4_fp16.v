//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication using Tensor Core
// Uses WMMA MMA instruction to compute D = A * B + C
// Each thread computes one output element
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_tensor_matmul_4x4_fp16;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 256;

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
    reg         kernel_start;
    reg  [31:0] kernel_pc;
    reg  [31:0] block_id_x, block_id_y, block_id_z;
    reg  [31:0] block_dim_x, block_dim_y, block_dim_z;
    reg  [31:0] grid_dim_x, grid_dim_y, grid_dim_z;
    wire        kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    wire        imem_ready;
    reg  [63:0] imem_data;
    reg         imem_valid;

    wire        l1d_req_valid;
    wire        l1d_req_write;
    wire [31:0] l1d_req_addr [0:NUM_LANES-1];
    wire [31:0] l1d_req_wdata [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [31:0] l1d_resp_rdata [0:NUM_LANES-1];
    reg         l1d_resp_valid;
    reg         l1d_resp_hit;

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
    // Test Matrices (4x4 FP16)
    // Using simple integer values that are exactly representable in FP16
    // A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    // B = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] (identity)
    // Result D = A (since B is identity)
    //------------------------------------------------------------------------
    // FP16 encoding: sign(1) + exp(5) + man(10), bias=15
    // 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400
    // 5.0 = 0x4500, 6.0 = 0x4600, 7.0 = 0x4700, 8.0 = 0x4800
    // 9.0 = 0x4880, 10.0 = 0x4900, 11.0 = 0x4980, 12.0 = 0x4A00
    // 13.0 = 0x4A80, 14.0 = 0x4B00, 15.0 = 0x4B80, 16.0 = 0x4C00

    // Matrix A (row-major FP16)
    reg [15:0] A [0:15];
    // Matrix B (column-major FP16 for efficient dot product)
    reg [15:0] B [0:15];
    // Expected result (FP32)
    reg [31:0] expected_D [0:15];

    // FP16 encoding function
    function [15:0] fp16_encode;
        input [31:0] value;  // Integer value 0-16
        reg [4:0] exp;
        reg [9:0] man;
        begin
            case (value)
                0:  fp16_encode = 16'h0000;
                1:  fp16_encode = 16'h3C00;
                2:  fp16_encode = 16'h4000;
                3:  fp16_encode = 16'h4200;
                4:  fp16_encode = 16'h4400;
                5:  fp16_encode = 16'h4500;
                6:  fp16_encode = 16'h4600;
                7:  fp16_encode = 16'h4700;
                8:  fp16_encode = 16'h4800;
                9:  fp16_encode = 16'h4880;
                10: fp16_encode = 16'h4900;
                11: fp16_encode = 16'h4980;
                12: fp16_encode = 16'h4A00;
                13: fp16_encode = 16'h4A80;
                14: fp16_encode = 16'h4B00;
                15: fp16_encode = 16'h4B80;
                16: fp16_encode = 16'h4C00;
                default: fp16_encode = 16'h0000;
            endcase
        end
    endfunction

    // FP32 encoding function for expected results
    function [31:0] fp32_encode;
        input [31:0] value;
        begin
            case (value)
                0:   fp32_encode = 32'h00000000;
                1:   fp32_encode = 32'h3F800000;
                2:   fp32_encode = 32'h40000000;
                3:   fp32_encode = 32'h40400000;
                4:   fp32_encode = 32'h40800000;
                5:   fp32_encode = 32'h40A00000;
                6:   fp32_encode = 32'h40C00000;
                7:   fp32_encode = 32'h40E00000;
                8:   fp32_encode = 32'h41000000;
                9:   fp32_encode = 32'h41100000;
                10:  fp32_encode = 32'h41200000;
                11:  fp32_encode = 32'h41300000;
                12:  fp32_encode = 32'h41400000;
                13:  fp32_encode = 32'h41500000;
                14:  fp32_encode = 32'h41600000;
                15:  fp32_encode = 32'h41700000;
                16:  fp32_encode = 32'h41800000;
                30:  fp32_encode = 32'h41F00000;
                70:  fp32_encode = 32'h428C0000;
                110: fp32_encode = 32'h42DC0000;
                150: fp32_encode = 32'h43160000;
                default: fp32_encode = 32'h00000000;
            endcase
        end
    endfunction

    initial begin
        // Matrix A: [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
        A[0]  = fp16_encode(1);  A[1]  = fp16_encode(2);  A[2]  = fp16_encode(3);  A[3]  = fp16_encode(4);
        A[4]  = fp16_encode(5);  A[5]  = fp16_encode(6);  A[6]  = fp16_encode(7);  A[7]  = fp16_encode(8);
        A[8]  = fp16_encode(9);  A[9]  = fp16_encode(10); A[10] = fp16_encode(11); A[11] = fp16_encode(12);
        A[12] = fp16_encode(13); A[13] = fp16_encode(14); A[14] = fp16_encode(15); A[15] = fp16_encode(16);

        // Matrix B: Identity [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        B[0]  = fp16_encode(1);  B[1]  = fp16_encode(0);  B[2]  = fp16_encode(0);  B[3]  = fp16_encode(0);
        B[4]  = fp16_encode(0);  B[5]  = fp16_encode(1);  B[6]  = fp16_encode(0);  B[7]  = fp16_encode(0);
        B[8]  = fp16_encode(0);  B[9]  = fp16_encode(0);  B[10] = fp16_encode(1);  B[11] = fp16_encode(0);
        B[12] = fp16_encode(0);  B[13] = fp16_encode(0);  B[14] = fp16_encode(0);  B[15] = fp16_encode(1);

        // Expected D = A * I = A (FP32 format)
        expected_D[0]  = fp32_encode(1);  expected_D[1]  = fp32_encode(2);  expected_D[2]  = fp32_encode(3);  expected_D[3]  = fp32_encode(4);
        expected_D[4]  = fp32_encode(5);  expected_D[5]  = fp32_encode(6);  expected_D[6]  = fp32_encode(7);  expected_D[7]  = fp32_encode(8);
        expected_D[8]  = fp32_encode(9);  expected_D[9]  = fp32_encode(10); expected_D[10] = fp32_encode(11); expected_D[11] = fp32_encode(12);
        expected_D[12] = fp32_encode(13); expected_D[13] = fp32_encode(14); expected_D[14] = fp32_encode(15); expected_D[15] = fp32_encode(16);
    end

    //------------------------------------------------------------------------
    // Instruction Memory
    // Program:
    //   1. Load matrix fragments into registers (using MOV_IMM for each lane)
    //   2. Execute WMMA MMA to compute D = A*B + 0
    //   3. Store result (not needed for this test - we verify via writeback)
    //------------------------------------------------------------------------
    reg [31:0] imem [0:IMEM_WORDS-1];
    reg        imem_req_q;
    reg [31:0] imem_addr_q;

    // Instruction encoding functions
    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm16;
        begin
            encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm16};
        end
    endfunction

    function [31:0] encode_wmma_mma;
        input [4:0] rd, ra, rb, rc;
        input [2:0] dtype;
        input [2:0] shape;
        reg [5:0] func;
        begin
            func = {shape, dtype};
            encode_wmma_mma = {`OP_WMMA_MMA, rd, ra, rb, rc, func};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    // ALU instruction encoding helper
    function [31:0] encode_alu_imm;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        input [9:0] imm10;
        begin
            // Format: [31:26]=OP_ALU_IMM, [25:21]=rd, [20:16]=ra, [15:10]=func, [9:0]=imm10
            encode_alu_imm = {`OP_ALU_IMM, rd, ra, func, imm10};
        end
    endfunction

    function [31:0] encode_alu_reg;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            // Format: [31:26]=OP_ALU, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]=unused, [5:0]=func
            encode_alu_reg = {`OP_ALU, rd, ra, rb, 5'b0, func};
        end
    endfunction

    // Program setup
    integer pc;
    integer lane_idx;
    initial begin
        // Initialize to NOP
        for (pc = 0; pc < IMEM_WORDS; pc = pc + 1) begin
            imem[pc] = {`OP_NOP, 26'b0};
        end

        pc = 0;

        // Simplified test with just FP16 1.0 values:
        // frag_a = 0x00003C00 (1.0 in low half, 0 in high half)
        // frag_b = 0x00003C00 (1.0 in low half, 0 in high half)
        // frag_c = 0
        // Result should be: 1.0 * 1.0 + 0 * 0 + 0 = 1.0 (FP32) = 0x3F800000

        // r1 = 0x00003C00 (1.0 FP16 in low half)
        imem[pc] = encode_mov_imm(5'd1, 16'h3C00);
        pc = pc + 1;

        // r2 = 0x00003C00 (1.0 FP16 in low half)
        imem[pc] = encode_mov_imm(5'd2, 16'h3C00);
        pc = pc + 1;

        // r3 = 0 (accumulator)
        imem[pc] = encode_mov_imm(5'd3, 16'h0000);
        pc = pc + 1;

        // First WMMA MMA: r4 = r1 * r2 + r3
        // Result: 1.0*1.0 + 0*0 + 0 = 1.0 (FP32) = 0x3F800000
        imem[pc] = encode_wmma_mma(5'd4, 5'd1, 5'd2, 5'd3, `TC_DATA_FP16, 3'b000);
        pc = pc + 1;

        // Now test with packed values: {2.0, 1.0}
        // r5 = low half = 0x3C00 (1.0)
        imem[pc] = encode_mov_imm(5'd5, 16'h3C00);
        pc = pc + 1;

        // r10 = high half = 0x4000 (2.0)
        imem[pc] = encode_mov_imm(5'd10, 16'h4000);
        pc = pc + 1;

        // Shift r10 left by 16 bits using ALU_IMM
        // shl r10, r10, 16
        imem[pc] = encode_alu_imm(5'd10, 5'd10, `FUNC_SHL, 10'd16);
        pc = pc + 1;

        // OR r5, r5, r10 -> r5 = 0x40003C00 = {2.0, 1.0}
        imem[pc] = encode_alu_reg(5'd5, 5'd5, 5'd10, `FUNC_OR);
        pc = pc + 1;

        // r6 = {0.5, 1.0} = 0x38003C00
        imem[pc] = encode_mov_imm(5'd6, 16'h3C00);  // low = 1.0
        pc = pc + 1;
        imem[pc] = encode_mov_imm(5'd11, 16'h3800); // high = 0.5
        pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd11, 5'd11, `FUNC_SHL, 10'd16);
        pc = pc + 1;
        imem[pc] = encode_alu_reg(5'd6, 5'd6, 5'd11, `FUNC_OR);
        pc = pc + 1;

        // r7 = 0 (accumulator)
        imem[pc] = encode_mov_imm(5'd7, 16'h0000);
        pc = pc + 1;

        // Second WMMA MMA: r8 = r5 * r6 + r7
        // r5 = {2.0, 1.0}, r6 = {0.5, 1.0}
        // Result: 1.0*1.0 + 2.0*0.5 + 0 = 1.0 + 1.0 = 2.0 (FP32) = 0x40000000
        imem[pc] = encode_wmma_mma(5'd8, 5'd5, 5'd6, 5'd7, `TC_DATA_FP16, 3'b000);
        pc = pc + 1;

        // Third WMMA MMA: r9 = r5 * r6 + r8
        // Result: 2.0 + 2.0 = 4.0 (FP32) = 0x40800000
        imem[pc] = encode_wmma_mma(5'd9, 5'd5, 5'd6, 5'd8, `TC_DATA_FP16, 3'b000);
        pc = pc + 1;

        // EXIT
        imem[pc] = encode_exit();
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_req_q <= 1'b0;
            imem_addr_q <= 0;
            imem_data <= 64'b0;
        end else begin
            imem_valid <= imem_req_q;
            if (imem_req_q) begin
                imem_data <= {imem[imem_addr_q[9:2] + 1], imem[imem_addr_q[9:2]]};
            end
            imem_req_q <= imem_req;
            if (imem_req) begin
                imem_addr_q <= imem_addr;
            end
        end
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .INIT_WARPS(1),
        .ICACHE_BYPASS(1)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .kernel_start  (kernel_start),
        .kernel_pc     (kernel_pc),
        .block_id_x    (block_id_x),
        .block_id_y    (block_id_y),
        .block_id_z    (block_id_z),
        .block_dim_x   (block_dim_x),
        .block_dim_y   (block_dim_y),
        .block_dim_z   (block_dim_z),
        .grid_dim_x    (grid_dim_x),
        .grid_dim_y    (grid_dim_y),
        .grid_dim_z    (grid_dim_z),
        .kernel_done   (kernel_done),
        .imem_req      (imem_req),
        .imem_addr     (imem_addr),
        .imem_ready    (imem_ready),
        .imem_data     (imem_data),
        .imem_valid    (imem_valid),
        .l1d_req_valid (l1d_req_valid),
        .l1d_req_write (l1d_req_write),
        .l1d_req_addr  (l1d_req_addr),
        .l1d_req_wdata (l1d_req_wdata),
        .l1d_req_mask  (l1d_req_mask),
        .l1d_resp_rdata(l1d_resp_rdata),
        .l1d_resp_valid(l1d_resp_valid),
        .l1d_resp_hit  (l1d_resp_hit),
        .m_axi_awid    (m_axi_awid),
        .m_axi_awaddr  (m_axi_awaddr),
        .m_axi_awlen   (m_axi_awlen),
        .m_axi_awsize  (m_axi_awsize),
        .m_axi_awburst (m_axi_awburst),
        .m_axi_awvalid (m_axi_awvalid),
        .m_axi_awready (m_axi_awready),
        .m_axi_wdata   (m_axi_wdata),
        .m_axi_wstrb   (m_axi_wstrb),
        .m_axi_wlast   (m_axi_wlast),
        .m_axi_wvalid  (m_axi_wvalid),
        .m_axi_wready  (m_axi_wready),
        .m_axi_bid     (m_axi_bid),
        .m_axi_bresp   (m_axi_bresp),
        .m_axi_bvalid  (m_axi_bvalid),
        .m_axi_bready  (m_axi_bready),
        .m_axi_arid    (m_axi_arid),
        .m_axi_araddr  (m_axi_araddr),
        .m_axi_arlen   (m_axi_arlen),
        .m_axi_arsize  (m_axi_arsize),
        .m_axi_arburst (m_axi_arburst),
        .m_axi_arvalid (m_axi_arvalid),
        .m_axi_arready (m_axi_arready),
        .m_axi_rid     (m_axi_rid),
        .m_axi_rdata   (m_axi_rdata),
        .m_axi_rresp   (m_axi_rresp),
        .m_axi_rlast   (m_axi_rlast),
        .m_axi_rvalid  (m_axi_rvalid),
        .m_axi_rready  (m_axi_rready)
    );

    assign imem_ready = 1'b1;

    //------------------------------------------------------------------------
    // Memory Interfaces (Idle)
    //------------------------------------------------------------------------
    integer ii;
    initial begin
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        for (ii = 0; ii < NUM_LANES; ii = ii + 1) begin
            l1d_resp_rdata[ii] = 32'b0;
        end
        m_axi_awready = 1'b1;
        m_axi_wready  = 1'b1;
        m_axi_bvalid  = 1'b0;
        m_axi_bresp   = 2'b00;
        m_axi_bid     = 4'b0;
        m_axi_arready = 1'b1;
        m_axi_rvalid  = 1'b0;
        m_axi_rresp   = 2'b00;
        m_axi_rid     = 4'b0;
        m_axi_rdata   = 32'b0;
        m_axi_rlast   = 1'b1;
    end

    //------------------------------------------------------------------------
    // Monitor Writeback for Verification
    //------------------------------------------------------------------------
    wire wb_fire = dut.wb_valid;
    wire [4:0] wb_rd = dut.wb_rd;
    wire [31:0] wb_data_lane0 = dut.wb_data[31:0];

    reg [31:0] captured_r4;
    reg [31:0] captured_r8;
    reg [31:0] captured_r9;
    reg r4_captured, r8_captured, r9_captured;

    always @(posedge clk) begin
        if (wb_fire) begin
            $display("[WB] r%0d = 0x%08x (lane0)", wb_rd, wb_data_lane0);
            if (wb_rd == 5'd4 && !r4_captured) begin
                captured_r4 <= wb_data_lane0;
                r4_captured <= 1'b1;
            end
            if (wb_rd == 5'd8 && !r8_captured) begin
                captured_r8 <= wb_data_lane0;
                r8_captured <= 1'b1;
            end
            if (wb_rd == 5'd9 && !r9_captured) begin
                captured_r9 <= wb_data_lane0;
                r9_captured <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    integer cycle_count;
    reg running;
    reg done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            cycle_count <= 0;
        end else begin
            if (kernel_start) begin
                running <= 1'b1;
                done <= 1'b0;
                cycle_count <= 0;
            end else if (running) begin
                cycle_count <= cycle_count + 1;
                if (kernel_done) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    reg test_passed;
    initial begin
        $display("============================================================");
        $display("RalphGPU Tensor Core 4x4 FP16 Matrix Multiplication Test");
        $display("Using WMMA MMA instruction");
        $display("============================================================");
        $display("");
        $display("Test cases:");
        $display("  1) r4 = {0,1.0} * {0,1.0} + 0 = 1.0 (FP32) = 0x3F800000");
        $display("  2) r8 = {2.0,1.0} * {0.5,1.0} + 0 = 1*1 + 2*0.5 = 2.0 = 0x40000000");
        $display("  3) r9 = {2.0,1.0} * {0.5,1.0} + r8 = 2.0 + 2.0 = 4.0 = 0x40800000");
        $display("");

        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 16; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        r4_captured = 0;
        r8_captured = 0;
        r9_captured = 0;
        captured_r4 = 0;
        captured_r8 = 0;
        captured_r9 = 0;
        test_passed = 1;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        @(posedge clk);
        kernel_start = 1;
        kernel_pc = 32'h0000_0000;
        @(posedge clk);
        kernel_start = 0;

        // Wait for completion or timeout
        fork
            wait(done);
            begin
                #100000;
                $display("ERROR: Timeout waiting for kernel completion");
                test_passed = 0;
            end
        join_any
        disable fork;

        repeat(20) @(posedge clk);

        $display("");
        $display("============================================================");
        $display("Results:");
        $display("  Execution time: %0d cycles", cycle_count);
        $display("");

        if (r4_captured) begin
            $display("  r4 (WMMA result 1) = 0x%08x", captured_r4);
            if (captured_r4 == 32'h3F800000) begin
                $display("    PASS: r4 = 1.0 (expected)");
            end else begin
                $display("    FAIL: r4 expected 0x3F800000 (1.0)");
                test_passed = 0;
            end
        end else begin
            $display("  r4 not captured - FAIL");
            test_passed = 0;
        end

        if (r8_captured) begin
            $display("  r8 (WMMA result 2) = 0x%08x", captured_r8);
            if (captured_r8 == 32'h40000000) begin
                $display("    PASS: r8 = 2.0 (expected)");
            end else begin
                $display("    FAIL: r8 expected 0x40000000 (2.0)");
                test_passed = 0;
            end
        end else begin
            $display("  r8 not captured - FAIL");
            test_passed = 0;
        end

        if (r9_captured) begin
            $display("  r9 (WMMA result 3) = 0x%08x", captured_r9);
            if (captured_r9 == 32'h40800000) begin
                $display("    PASS: r9 = 4.0 (expected)");
            end else begin
                $display("    FAIL: r9 expected 0x40800000 (4.0)");
                test_passed = 0;
            end
        end else begin
            $display("  r9 not captured - FAIL");
            test_passed = 0;
        end

        $display("");
        if (test_passed) begin
            $display("TEST PASSED: Tensor Core WMMA MMA works correctly!");
        end else begin
            $display("TEST FAILED");
        end
        $display("============================================================");

        $finish;
    end

endmodule
