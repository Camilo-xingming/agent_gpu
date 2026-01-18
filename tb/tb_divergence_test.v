//============================================================================
// RalphGPU - Divergence Test
// Tests SIMT branch divergence and reconvergence
// Program: if (tid & 1) { result = 200; } else { result = 100; }
// Expected: even threads store 100, odd threads store 200
//============================================================================

`timescale 1ns / 1ps

module tb_divergence_test;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;
    parameter NUM_THREADS = 32;          // Full warp
    parameter BASE_ADDR = 32'h1000;      // Output base address

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

    // MOV_SPECIAL: rd = special_register[ra]
    // Format: {opcode[31:26], rd[25:21], ra[20:16], unused[15:0]}
    function [31:0] encode_mov_special;
        input [4:0] rd;
        input [4:0] sreg;  // Special register code (0=tid.x, etc.)
        encode_mov_special = {`OP_MOV_SPECIAL, rd, sreg, 16'b0};
    endfunction

    // ALU_IMM: rd = ra op imm
    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:10]=func, [9:0]=imm10
    function [31:0] encode_alu_imm;
        input [4:0] rd, ra;
        input [5:0] func;
        input [9:0] imm;
        encode_alu_imm = {`OP_ALU_IMM, rd, ra, func, imm};
    endfunction

    // ALU_REG: rd = ra op rb
    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:5]=func, [4:0]=rc
    function [31:0] encode_alu_reg;
        input [4:0] rd, ra, rb;
        input [5:0] func;
        encode_alu_reg = {`OP_ALU, rd, ra, rb, func, 5'b0};
    endfunction

    // ST_GLOBAL: mem[ra + offset] = rs
    // For per-thread stores, we'll compute address in register
    function [31:0] encode_st_global;
        input [4:0] rs, ra;
        encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rs, 5'b0, 6'b0};
    endfunction

    // EXIT: terminate warp
    function [31:0] encode_exit;
        encode_exit = {`OP_EXIT, 26'b0};
    endfunction

    // BRANCH: conditional/unconditional branch
    function [31:0] encode_branch;
        input [1:0] branch_type;
        input [4:0] ra;
        input signed [15:0] offset;
        encode_branch = {`OP_BRANCH, branch_type, 3'b0, ra, offset};
    endfunction

    // Branch type constants
    localparam BR_UNCOND     = 2'b00;
    localparam BR_IF_ZERO    = 2'b01;
    localparam BR_IF_NOTZERO = 2'b10;
    localparam BR_UNIFORM    = 2'b11;

    // Special register codes
    localparam SREG_TID_X = 5'd0;

    //------------------------------------------------------------------------
    // Program: Divergent branch based on thread ID parity
    //------------------------------------------------------------------------
    // R0 = thread ID (0-31)
    // R1 = tid & 1 (0 for even, 1 for odd)
    // R2 = result value (100 for even, 200 for odd)
    // R3 = base address
    // R4 = store address (base + tid * 4)
    //
    // Program:
    //   0x00: MOV R0, %tid.x           ; R0 = thread_id
    //   0x04: AND R1, R0, 1            ; R1 = tid & 1
    //   0x08: MOV R2, 100              ; default: result = 100 (even path)
    //   0x0C: BRANCH skip if R1 == 0  ; if even, skip odd path
    //   0x10: MOV R2, 200              ; odd path: result = 200
    // skip (0x14):
    //   0x14: MOV R3, BASE_ADDR        ; base address
    //   0x18: SHL R4, R0, 2            ; R4 = tid * 4
    //   0x1C: ADD R4, R4, R3           ; R4 = base + tid * 4
    //   0x20: ST [R4], R2              ; store result
    //   0x24: EXIT
    //------------------------------------------------------------------------
    initial begin
        pc = 0;

        // Get thread ID
        imem[pc] = encode_mov_special(5'd0, SREG_TID_X);  // R0 = %tid.x
        pc = pc + 1;

        // R1 = tid & 1 (parity check)
        imem[pc] = encode_alu_imm(5'd1, 5'd0, `FUNC_AND, 10'd1);  // R1 = R0 & 1
        pc = pc + 1;

        // Default value for even threads
        imem[pc] = encode_mov_imm(5'd2, 16'd100);  // R2 = 100
        pc = pc + 1;

        // Branch to skip if even (R1 == 0)
        // Current PC = 0x0C, target = 0x14 (skip)
        // Offset = target - PC = 0x14 - 0x0C = 8
        imem[pc] = encode_branch(BR_IF_ZERO, 5'd1, 16'd8);
        pc = pc + 1;

        // Odd threads: set result = 200
        imem[pc] = encode_mov_imm(5'd2, 16'd200);  // R2 = 200
        pc = pc + 1;

        // skip: Common path after divergence reconverges
        // Load base address
        imem[pc] = encode_mov_imm(5'd3, BASE_ADDR[15:0]);  // R3 = base addr
        pc = pc + 1;

        // Compute store address: R4 = tid * 4
        imem[pc] = encode_alu_imm(5'd4, 5'd0, `FUNC_SHL, 10'd2);  // R4 = R0 << 2
        pc = pc + 1;

        // R4 = R4 + R3 (base + offset)
        imem[pc] = encode_alu_reg(5'd4, 5'd4, 5'd3, `FUNC_ADD);  // R4 = R4 + R3
        pc = pc + 1;

        // Store result to per-thread location
        imem[pc] = encode_st_global(5'd2, 5'd4);  // mem[R4] = R2
        pc = pc + 1;

        // Exit
        imem[pc] = encode_exit();
        pc = pc + 1;

        $display("Program loaded: %0d instructions", pc);
        $display("Testing divergent branch: even threads -> 100, odd threads -> 200");
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
    reg [31:0] write_count;

    // Initialize memory
    initial begin
        integer i;
        for (i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'hDEADBEEF;  // Pattern to detect unwritten locations
        end
        write_count = 0;
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
                write_count <= write_count + 1;
                $display("[AXI-WR] addr=0x%08x data=%0d (thread=%0d)",
                         pending_write_addr, m_axi_wdata, (pending_write_addr - BASE_ADDR) >> 2);
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bready && m_axi_bvalid) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read handling
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
    integer i;
    integer pass_count;
    integer fail_count;
    reg [31:0] expected_value;
    reg [31:0] actual_value;

    initial begin
        $display("\n============================================================");
        $display("RalphGPU Divergence Test");
        $display("Testing: if (tid & 1) result=200; else result=100;");
        $display("Expected: even threads=100, odd threads=200");
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

        // Configure kernel: 32 threads (1 warp)
        // CSR addresses from ralph_gpu_top.v:
        // CSR_BLOCK_DIM_X = 0x018, CSR_BLOCK_DIM_Y = 0x01C, CSR_BLOCK_DIM_Z = 0x020
        // CSR_KERNEL_PC = 0x008, CSR_GPU_CONTROL = 0x004

        csr_addr = 12'h018;  // Block dim X (CSR_BLOCK_DIM_X)
        csr_wr_data = NUM_THREADS;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h01C;  // Block dim Y (CSR_BLOCK_DIM_Y)
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        csr_addr = 12'h020;  // Block dim Z (CSR_BLOCK_DIM_Z)
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Set kernel PC = 0
        csr_addr = 12'h008;  // Kernel PC (CSR_KERNEL_PC)
        csr_wr_data = 0;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;
        @(posedge clk);

        // Start kernel
        $display("Starting kernel with %0d threads...\n", NUM_THREADS);
        csr_addr = 12'h004;  // Kernel launch (CSR_GPU_CONTROL)
        csr_wr_data = 1;
        csr_wr_en = 1;
        @(posedge clk);
        csr_wr_en = 0;

        // Wait for completion
        while (!irq_kernel_done && cycle_count < 20000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        // Wait for memory operations to complete
        // Each lane write takes several cycles through the AXI bus
        // With 32 threads, we need at least 32 * 5 = 160 cycles
        repeat(200) @(posedge clk);

        // Check results
        $display("\n============================================================");
        $display("Test Results");
        $display("============================================================");
        $display("Kernel completed in %0d cycles", cycle_count);
        $display("Total writes: %0d", write_count);
        $display("");

        pass_count = 0;
        fail_count = 0;

        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            actual_value = gmem[(BASE_ADDR >> 2) + i];
            expected_value = (i & 1) ? 200 : 100;  // Odd -> 200, Even -> 100

            if (actual_value == expected_value) begin
                pass_count = pass_count + 1;
                $display("Thread %2d: got %3d, expected %3d - PASS",
                         i, actual_value, expected_value);
            end else begin
                fail_count = fail_count + 1;
                $display("Thread %2d: got %3d (0x%08x), expected %3d - FAIL",
                         i, actual_value, actual_value, expected_value);
            end
        end

        $display("");
        $display("============================================================");
        if (fail_count == 0) begin
            $display("TEST PASSED: All %0d threads produced correct results!", pass_count);
            $display("Divergence and reconvergence worked correctly.");
        end else begin
            $display("TEST FAILED: %0d passed, %0d failed", pass_count, fail_count);
        end
        $display("============================================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #300000;
        $display("TIMEOUT: Test exceeded maximum time");
        $finish;
    end

endmodule
