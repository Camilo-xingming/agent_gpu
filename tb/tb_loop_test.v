//============================================================================
// RalphGPU - Loop Test
// Tests conditional branching for loop execution
// Simple test: sum = 0; for(i=N; i>0; i--) sum += 2; expect sum = 2*N
//============================================================================

`timescale 1ns / 1ps

module tb_loop_test;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter LOOP_COUNT = 5;          // Number of loop iterations
    parameter ADD_VALUE = 2;           // Value to add each iteration
    parameter EXPECTED_RESULT = LOOP_COUNT * ADD_VALUE;  // 10

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
    reg [31:0] imem [0:255];
    integer pc;

    // Instruction encoding functions

    // MOV_IMM: rd = imm16
    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
    endfunction

    // ALU_IMM: rd = ra op imm
    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:10]=func, [9:0]=imm10
    function [31:0] encode_alu_imm;
        input [4:0] rd, ra;
        input [5:0] func;
        input [9:0] imm;
        encode_alu_imm = {`OP_ALU_IMM, rd, ra, func, imm};
    endfunction

    // ST_GLOBAL: mem[ra] = rb
    function [31:0] encode_st_global;
        input [4:0] rs, ra;
        encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rs, 5'b0, 6'b0};
    endfunction

    // EXIT: terminate warp
    function [31:0] encode_exit;
        encode_exit = {`OP_EXIT, 26'b0};
    endfunction

    // BRANCH: conditional/unconditional branch
    // branch_type: 00=unconditional, 01=if_zero, 10=if_not_zero, 11=uniform
    // offset: signed 16-bit byte offset from current PC
    function [31:0] encode_branch;
        input [1:0] branch_type;
        input [4:0] ra;              // condition register
        input signed [15:0] offset;  // signed byte offset
        begin
            // Format: {opcode[31:26], type[25:24], unused[23:21], ra[20:16], offset[15:0]}
            encode_branch = {`OP_BRANCH, branch_type, 3'b0, ra, offset};
        end
    endfunction

    // Branch type constants
    localparam BR_UNCOND     = 2'b00;  // Always branch
    localparam BR_IF_ZERO    = 2'b01;  // Branch if ra == 0
    localparam BR_IF_NOTZERO = 2'b10;  // Branch if ra != 0
    localparam BR_UNIFORM    = 2'b11;  // Uniform branch

    //------------------------------------------------------------------------
    // Program: Loop to accumulate values
    //------------------------------------------------------------------------
    // R1 = loop counter (initialized to LOOP_COUNT)
    // R2 = accumulator (initialized to 0)
    // R3 = output address
    //
    // Program:
    //   0x00: MOV R1, LOOP_COUNT     ; counter = 5
    //   0x04: MOV R2, 0              ; sum = 0
    //   0x08: MOV R3, 0x1000         ; output address
    // loop_start (0x0C):
    //   0x0C: ADD R2, R2, ADD_VALUE  ; sum += 2
    //   0x10: SUB R1, R1, 1          ; counter--
    //   0x14: BRANCH loop_start if R1 != 0  ; offset = -12 (0xFFF4)
    //   0x18: ST [R3], R2            ; store result
    //   0x1C: EXIT
    //------------------------------------------------------------------------
    initial begin
        pc = 0;

        // Initialize counter and accumulator
        imem[pc] = encode_mov_imm(5'd1, LOOP_COUNT);  // R1 = 5 (counter)
        pc = pc + 1;
        imem[pc] = encode_mov_imm(5'd2, 16'd0);       // R2 = 0 (accumulator)
        pc = pc + 1;
        imem[pc] = encode_mov_imm(5'd3, 16'h1000);    // R3 = 0x1000 (output addr)
        pc = pc + 1;

        // Loop body (starts at PC = 0x0C = 12)
        // loop_start:
        imem[pc] = encode_alu_imm(5'd2, 5'd2, `FUNC_ADD, ADD_VALUE);  // R2 = R2 + 2
        pc = pc + 1;
        imem[pc] = encode_alu_imm(5'd1, 5'd1, `FUNC_SUB, 10'd1);      // R1 = R1 - 1
        pc = pc + 1;
        // Branch back to loop_start (PC=0x0C) if R1 != 0
        // Current PC = 0x14, target = 0x0C
        // Offset = target - current_pc = 0x0C - 0x14 = -8 (0xFFF8)
        imem[pc] = encode_branch(BR_IF_NOTZERO, 5'd1, -16'd8);
        pc = pc + 1;

        // After loop: store result
        imem[pc] = encode_st_global(5'd2, 5'd3);  // mem[R3] = R2
        pc = pc + 1;

        // Exit
        imem[pc] = encode_exit();
        pc = pc + 1;

        $display("Program loaded: %0d instructions", pc);
        $display("Expected result: %0d (loop_count=%0d * add_value=%0d)",
                 EXPECTED_RESULT, LOOP_COUNT, ADD_VALUE);
    end

    //------------------------------------------------------------------------
    // Instruction Memory Response
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
            $display("[IMEM] PC=0x%04x inst0=0x%08x inst1=0x%08x",
                     imem_addr, imem[imem_addr[9:2]], imem[imem_addr[9:2] + 1]);
        end else begin
            imem_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Data Memory (AXI)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_write_addr;
    reg [31:0] stored_result;
    reg        result_written;

    // Initialize memory
    initial begin
        integer i;
        for (i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'h0;
        end
        result_written = 0;
    end

    // AXI Write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            pending_write_addr <= 0;
        end else begin
            // Address phase
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            // Data phase
            if (m_axi_wvalid && m_axi_wready) begin
                gmem[pending_write_addr[13:2]] <= m_axi_wdata;
                stored_result <= m_axi_wdata;
                result_written <= 1'b1;
                $display("[AXI-WR] addr=0x%08x data=0x%08x (%0d)",
                         pending_write_addr, m_axi_wdata, m_axi_wdata);
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bready && m_axi_bvalid) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read handling (not used in this test)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
            end else if (m_axi_rready && m_axi_rvalid) begin
                m_axi_rvalid <= 1'b0;
            end
        end
    end

    assign m_axi_bresp = 2'b00;
    assign m_axi_rresp = 2'b00;

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    integer cycle_count;

    initial begin
        $display("\n============================================================");
        $display("RalphGPU Loop Test");
        $display("Testing: for(i=%0d; i>0; i--) sum += %0d", LOOP_COUNT, ADD_VALUE);
        $display("Expected result: %0d", EXPECTED_RESULT);
        $display("============================================================\n");

        // Initialize
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        cycle_count = 0;

        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        // Configure kernel: 1 thread (scalar execution)
        // CSR addresses: BLOCK_DIM_X=0x018, BLOCK_DIM_Y=0x01C, BLOCK_DIM_Z=0x020
        //                KERNEL_PC=0x008, GPU_CONTROL=0x004
        csr_addr = 12'h018;  // Block dim X
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h01C;  // Block dim Y
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h020;  // Block dim Z
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Set kernel PC = 0
        csr_addr = 12'h008;  // Kernel PC
        csr_wr_data = 0;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Start kernel
        $display("Starting kernel...\n");
        csr_addr = 12'h004;  // Kernel launch (GPU_CONTROL)
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;

        // Wait for completion
        while (!irq_kernel_done && cycle_count < 10000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        repeat(10) @(posedge clk);

        // Check result
        $display("\n============================================================");
        $display("Test Results");
        $display("============================================================");
        $display("Kernel completed in %0d cycles", cycle_count);

        if (result_written) begin
            $display("Stored result: %0d (expected: %0d)", stored_result, EXPECTED_RESULT);
            if (stored_result == EXPECTED_RESULT) begin
                $display("\nTEST PASSED: Loop executed correctly!");
            end else begin
                $display("\nTEST FAILED: Result mismatch!");
            end
        end else begin
            $display("\nTEST FAILED: No result written to memory!");
        end
        $display("============================================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #200000;
        $display("TIMEOUT: Test exceeded maximum time");
        $finish;
    end

endmodule
