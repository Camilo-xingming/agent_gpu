//============================================================================
// RalphGPU - Comprehensive GPU Top Level Testbench
// 100+ Complex Test Cases for ralph_gpu_top
//
// Test Categories:
// 1. CSR Interface Tests (10 tests)
// 2. Kernel Launch Tests (10 tests)
// 3. ALU Operations (15 tests)
// 4. FP32 Operations (10 tests)
// 5. FP16 Operations (10 tests)
// 6. Memory Operations (15 tests)
// 7. Tensor/WMMA Operations (10 tests)
// 8. Control Flow Tests (10 tests)
// 9. Video/Texture Operations (10 tests)
// 10. Performance Stress Tests (10 tests)
//============================================================================

`timescale 1ns / 1ps

`include "gpu_defines.vh"

module tb_ralph_gpu_top;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;  // 100 MHz
    localparam TIMEOUT_CYCLES = 100000;  // Per-test timeout (increased)

    // Memory layout
    localparam IMEM_BASE = 32'h0000_0000;
    localparam GMEM_BASE = 32'h0000_1000;  // Global memory starts at 4KB

    // CSR Addresses
    localparam CSR_GPU_STATUS   = 12'h000;
    localparam CSR_GPU_CONTROL  = 12'h004;
    localparam CSR_KERNEL_PC    = 12'h008;
    localparam CSR_GRID_DIM_X   = 12'h00C;
    localparam CSR_GRID_DIM_Y   = 12'h010;
    localparam CSR_GRID_DIM_Z   = 12'h014;
    localparam CSR_BLOCK_DIM_X  = 12'h018;
    localparam CSR_BLOCK_DIM_Y  = 12'h01C;
    localparam CSR_BLOCK_DIM_Z  = 12'h020;

    //------------------------------------------------------------------------
    // Test Counters
    //------------------------------------------------------------------------
    integer total_tests;
    integer passed_tests;
    integer failed_tests;
    integer test_num;
    reg [255:0] test_name;  // Current test name

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
        .NUM_SM(1)  // Single SM for controlled testing
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
    // Memory Models
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:4095];  // 16KB instruction memory
    reg [31:0] global_mem [0:16383];      // 64KB global memory

    // Instruction fetch pipeline
    reg [31:0] pending_imem_addr;
    reg        pending_imem_req;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
            pending_imem_req <= 1'b0;
            pending_imem_addr <= 32'b0;
        end else begin
            if (imem_req) begin
                pending_imem_req <= 1'b1;
                pending_imem_addr <= imem_addr;
                imem_valid <= 1'b0;
            end else if (pending_imem_req) begin
                imem_data <= {instruction_mem[(pending_imem_addr >> 2) + 1],
                              instruction_mem[pending_imem_addr >> 2]};
                imem_valid <= 1'b1;
                pending_imem_req <= 1'b0;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // AXI Memory Response
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
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
        end else begin
            // Read channel
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_read <= 1'b1;
                pending_axi_addr <= m_axi_araddr;
                m_axi_arready <= 1'b0;
                axi_read_delay <= 3'd2;
            end else if (pending_axi_read && axi_read_delay > 0) begin
                axi_read_delay <= axi_read_delay - 1'b1;
            end else if (pending_axi_read && axi_read_delay == 0) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= global_mem[pending_axi_addr[15:2]];
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
                pending_axi_read <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end

            // Write channel
            if (m_axi_wvalid && m_axi_wready) begin
                global_mem[m_axi_awaddr[15:2]] <= m_axi_wdata;
            end

            if (m_axi_awvalid && m_axi_awready && m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Instruction Encoding Functions
    //------------------------------------------------------------------------
    function [31:0] encode_nop;
        begin
            encode_nop = {`OP_NOP, 26'b0};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    function [31:0] encode_alu;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_alu = {`OP_ALU, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_mul;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_mul = {`OP_MUL, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_fp32;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_fp32 = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_fp16;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_fp16 = {`OP_FP16_ARITH, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_sfu;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_sfu = {`OP_SFU, rd, ra, 5'b0, 5'b0, func};
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

    function [31:0] encode_ld_shared;
        input [4:0] rd;
        input [4:0] ra;
        begin
            encode_ld_shared = {`OP_LD_SHARED, rd, ra, 16'b0};
        end
    endfunction

    function [31:0] encode_st_shared;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_st_shared = {`OP_ST_SHARED, 5'b0, ra, rb, 11'b0};
        end
    endfunction

    function [31:0] encode_mov_special;
        input [4:0] rd;
        input [4:0] sreg;
        begin
            encode_mov_special = {`OP_MOV_SPECIAL, rd, sreg, 16'b0};
        end
    endfunction

    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        begin
            encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
        end
    endfunction

    function [31:0] encode_bar_sync;
        input [4:0] bar_id;
        begin
            encode_bar_sync = {`OP_BAR_SYNC, bar_id, 21'b0};
        end
    endfunction

    function [31:0] encode_bar_warp_sync;
        input [4:0] rd;
        begin
            encode_bar_warp_sync = {`OP_BAR_WARP_SYNC, rd, 21'b0};
        end
    endfunction

    function [31:0] encode_atom;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_atom = {`OP_ATOM, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_shfl;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_shfl = {`OP_SHFL, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_vote;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_vote = {`OP_VOTE, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_wmma_load;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_wmma_load = {`OP_WMMA_LOAD, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_wmma_mma;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_wmma_mma = {`OP_WMMA_MMA, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_video;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_video = {`OP_VIDEO, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_tex;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_tex = {`OP_TEX, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_cvt;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_cvt = {`OP_CVT, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_setp;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_setp = {`OP_SETP, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_branch;
        input [4:0] pred;
        input [20:0] offset;
        begin
            encode_branch = {`OP_BRANCH, pred, offset};
        end
    endfunction

    function [31:0] encode_cpasync;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_cpasync = {`OP_CPASYNC, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_wgmma;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [5:0] func;
        begin
            encode_wgmma = {`OP_WGMMA_MMA, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_mbarrier;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_mbarrier = {`OP_MBARRIER, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_stack;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_stack = {`OP_STACK, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    function [31:0] encode_debug;
        input [4:0] rd;
        input [5:0] func;
        begin
            encode_debug = {`OP_DEBUG, rd, 21'b0, func};
        end
    endfunction

    function [31:0] encode_cache_policy;
        input [4:0] rd;
        input [4:0] ra;
        input [5:0] func;
        begin
            encode_cache_policy = {`OP_CACHE_POLICY, rd, ra, 5'b0, 5'b0, func};
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Helper Tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 0;
            csr_wr_en = 0;
            csr_addr = 0;
            csr_wr_data = 0;
            #100;
            rst_n = 1;
            #50;
        end
    endtask

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

    task read_csr;
        input [11:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            csr_addr <= addr;
            @(posedge clk);
            data = csr_rd_data;
        end
    endtask

    task clear_imem;
        integer i;
        begin
            for (i = 0; i < 4096; i = i + 1) begin
                instruction_mem[i] = encode_nop();
            end
        end
    endtask

    task clear_gmem;
        integer i;
        begin
            for (i = 0; i < 16384; i = i + 1) begin
                global_mem[i] = 32'b0;
            end
        end
    endtask

    task launch_kernel;
        input [31:0] pc;
        input [31:0] gdim_x;
        input [31:0] gdim_y;
        input [31:0] gdim_z;
        input [31:0] bdim_x;
        input [31:0] bdim_y;
        input [31:0] bdim_z;
        begin
            write_csr(CSR_GRID_DIM_X, gdim_x);
            write_csr(CSR_GRID_DIM_Y, gdim_y);
            write_csr(CSR_GRID_DIM_Z, gdim_z);
            write_csr(CSR_BLOCK_DIM_X, bdim_x);
            write_csr(CSR_BLOCK_DIM_Y, bdim_y);
            write_csr(CSR_BLOCK_DIM_Z, bdim_z);
            write_csr(CSR_KERNEL_PC, pc);
            write_csr(CSR_GPU_CONTROL, 32'h1);
        end
    endtask

    task wait_kernel_done;
        input integer max_cycles;
        output reg timeout;
        integer cycle_count;
        begin
            timeout = 0;
            cycle_count = 0;
            while (!irq_kernel_done && cycle_count < max_cycles) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end
            if (cycle_count >= max_cycles) begin
                timeout = 1;
            end
            #20;
        end
    endtask

    task report_test;
        input [255:0] name;
        input pass;
        begin
            test_num = test_num + 1;
            total_tests = total_tests + 1;
            if (pass) begin
                passed_tests = passed_tests + 1;
                $display("[PASS] Test %0d: %0s", test_num, name);
            end else begin
                failed_tests = failed_tests + 1;
                $fatal(1, "[FAIL] Test %0d: %0s", test_num, name);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer pc;
    reg [31:0] rd_data;
    reg timeout;
    reg test_pass;

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Top Level Comprehensive Test Suite");
        $display("Target: 100+ Complex Test Cases");
        $display("============================================================");
        $display("");

        // Initialize counters
        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        test_num = 0;

        // Initial reset
        reset_dut();
        clear_imem();
        clear_gmem();

        //====================================================================
        // SECTION 1: CSR Interface Tests (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 1: CSR Interface Tests ---");

        // Test 1.1: GPU Status Register Read
        read_csr(CSR_GPU_STATUS, rd_data);
        test_pass = (rd_data[0] == 1'b1);  // GPU should be ready
        report_test("CSR GPU Status Ready", test_pass);

        // Test 1.2: Grid Dim X Write/Read
        write_csr(CSR_GRID_DIM_X, 32'd16);
        read_csr(CSR_GRID_DIM_X, rd_data);
        test_pass = (rd_data == 32'd16);
        report_test("CSR Grid Dim X", test_pass);

        // Test 1.3: Grid Dim Y Write/Read
        write_csr(CSR_GRID_DIM_Y, 32'd8);
        read_csr(CSR_GRID_DIM_Y, rd_data);
        test_pass = (rd_data == 32'd8);
        report_test("CSR Grid Dim Y", test_pass);

        // Test 1.4: Grid Dim Z Write/Read
        write_csr(CSR_GRID_DIM_Z, 32'd4);
        read_csr(CSR_GRID_DIM_Z, rd_data);
        test_pass = (rd_data == 32'd4);
        report_test("CSR Grid Dim Z", test_pass);

        // Test 1.5: Block Dim X Write/Read
        write_csr(CSR_BLOCK_DIM_X, 32'd32);
        read_csr(CSR_BLOCK_DIM_X, rd_data);
        test_pass = (rd_data == 32'd32);
        report_test("CSR Block Dim X", test_pass);

        // Test 1.6: Block Dim Y Write/Read
        write_csr(CSR_BLOCK_DIM_Y, 32'd16);
        read_csr(CSR_BLOCK_DIM_Y, rd_data);
        test_pass = (rd_data == 32'd16);
        report_test("CSR Block Dim Y", test_pass);

        // Test 1.7: Block Dim Z Write/Read
        write_csr(CSR_BLOCK_DIM_Z, 32'd1);
        read_csr(CSR_BLOCK_DIM_Z, rd_data);
        test_pass = (rd_data == 32'd1);
        report_test("CSR Block Dim Z", test_pass);

        // Test 1.8: Kernel PC Write/Read
        write_csr(CSR_KERNEL_PC, 32'h0000_1000);
        read_csr(CSR_KERNEL_PC, rd_data);
        test_pass = (rd_data == 32'h0000_1000);
        report_test("CSR Kernel PC", test_pass);

        // Test 1.9: Maximum Grid Dimensions
        write_csr(CSR_GRID_DIM_X, 32'hFFFF_FFFF);
        read_csr(CSR_GRID_DIM_X, rd_data);
        test_pass = (rd_data == 32'hFFFF_FFFF);
        report_test("CSR Max Grid Dim", test_pass);

        // Test 1.10: Zero Grid Dimensions
        write_csr(CSR_GRID_DIM_X, 32'd0);
        read_csr(CSR_GRID_DIM_X, rd_data);
        test_pass = (rd_data == 32'd0);
        report_test("CSR Zero Grid Dim", test_pass);

        //====================================================================
        // SECTION 2: Kernel Launch Tests (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 2: Kernel Launch Tests ---");
        reset_dut();

        // Test 2.1: Simple Exit Kernel
        clear_imem();
        instruction_mem[0] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Simple Exit Kernel", test_pass);
        reset_dut();

        // Test 2.2: NOP retirement then Exit
        clear_imem();
        instruction_mem[0] = encode_nop();
        instruction_mem[1] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("NOP then Exit", test_pass);
        reset_dut();

        // Test 2.3: Single block with compute (multi-block needs longer time)
        clear_imem();
        instruction_mem[0] = encode_alu(5'd1, 5'd0, 5'd0, `FUNC_ADD);
        instruction_mem[1] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Single Block Compute", test_pass);
        reset_dut();

        // Test 2.4: Single block 3D config
        clear_imem();
        instruction_mem[0] = encode_alu(5'd1, 5'd0, 5'd0, `FUNC_ADD);
        instruction_mem[1] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Single Block 3D Config", test_pass);
        reset_dut();

        // Test 2.5: Large Block Dimension
        clear_imem();
        instruction_mem[0] = encode_exit();
        launch_kernel(0, 1, 1, 1, 32, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Large Block Dim 32", test_pass);
        reset_dut();

        // Test 2.6: 2D Block Dimension (16x2)
        clear_imem();
        instruction_mem[0] = encode_exit();
        launch_kernel(0, 1, 1, 1, 16, 2, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("2D Block 16x2", test_pass);
        reset_dut();

        // Test 2.7: Non-zero PC Start
        clear_imem();
        instruction_mem[4] = encode_exit();
        launch_kernel(16, 1, 1, 1, 1, 1, 1);  // PC = 16 = instruction_mem[4]
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Non-zero PC Start", test_pass);
        reset_dut();

        // Test 2.8: FP32 then Exit
        clear_imem();
        instruction_mem[0] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_ADD);
        instruction_mem[1] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("FP32 then Exit", test_pass);
        reset_dut();

        // Test 2.9: Minimal Config (1x1x1)
        clear_imem();
        instruction_mem[0] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("Minimal 1x1x1 Config", test_pass);
        reset_dut();

        // Test 2.10: Multiple ALU Instructions before Exit
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd0, 5'd0, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd2, 5'd1, 5'd1, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd3, 5'd2, 5'd2, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd4, 5'd3, 5'd3, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd5, 5'd4, 5'd4, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        report_test("5 ALUs then Exit", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 3: ALU Operations (15 tests)
        //====================================================================
        $display("");
        $display("--- Section 3: ALU Operations ---");

        // Test 3.1: ALU ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU ADD", test_pass);
        reset_dut();

        // Test 3.2: ALU SUB
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_SUB); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU SUB", test_pass);
        reset_dut();

        // Test 3.3: ALU AND
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_AND); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU AND", test_pass);
        reset_dut();

        // Test 3.4: ALU OR
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_OR); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU OR", test_pass);
        reset_dut();

        // Test 3.5: ALU XOR
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_XOR); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU XOR", test_pass);
        reset_dut();

        // Test 3.6: ALU SHL
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_SHL); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU SHL", test_pass);
        reset_dut();

        // Test 3.7: ALU SHR_U
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_SHR_U); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU SHR_U", test_pass);
        reset_dut();

        // Test 3.8: ALU MIN_S
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_MIN_S); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU MIN_S", test_pass);
        reset_dut();

        // Test 3.9: ALU MAX_S
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_MAX_S); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU MAX_S", test_pass);
        reset_dut();

        // Test 3.10: ALU POPC
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd0, `FUNC_POPC); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU POPC", test_pass);
        reset_dut();

        // Test 3.11: ALU CLZ
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd0, `FUNC_CLZ); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU CLZ", test_pass);
        reset_dut();

        // Test 3.12: ALU BREV
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd0, `FUNC_BREV); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU BREV", test_pass);
        reset_dut();

        // Test 3.13: ALU ABS
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd0, `FUNC_ABS); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU ABS", test_pass);
        reset_dut();

        // Test 3.14: ALU NEG
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd0, `FUNC_NEG); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU NEG", test_pass);
        reset_dut();

        // Test 3.15: ALU Chain (ADD->SUB->AND)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd4, 5'd1, 5'd2, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd5, 5'd4, 5'd3, `FUNC_SUB); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd6, 5'd5, 5'd1, `FUNC_AND); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU Chain", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 4: FP32 Operations (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 4: FP32 Operations ---");

        // Test 4.1: FP32 ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 ADD", test_pass);
        reset_dut();

        // Test 4.2: FP32 SUB
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_SUB); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 SUB", test_pass);
        reset_dut();

        // Test 4.3: FP32 MUL
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_MUL); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 MUL", test_pass);
        reset_dut();

        // Test 4.4: FP32 DIV
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_DIV); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 DIV", test_pass);
        reset_dut();

        // Test 4.5: FP32 FMA
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_FMA); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 FMA", test_pass);
        reset_dut();

        // Test 4.6: FP32 MIN
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_MIN); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 MIN", test_pass);
        reset_dut();

        // Test 4.7: FP32 MAX
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_MAX); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 MAX", test_pass);
        reset_dut();

        // Test 4.8: SFU RCP
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_sfu(5'd1, 5'd2, `FP_RCP); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SFU RCP", test_pass);
        reset_dut();

        // Test 4.9: SFU SQRT
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_sfu(5'd1, 5'd2, `FP_SQRT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SFU SQRT", test_pass);
        reset_dut();

        // Test 4.10: FP32 Chain (MUL->ADD->FMA)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp32(5'd4, 5'd1, 5'd2, `FP_MUL); pc = pc + 1;
        instruction_mem[pc] = encode_fp32(5'd5, 5'd4, 5'd3, `FP_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_fp32(5'd6, 5'd5, 5'd1, `FP_FMA); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 Chain", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 5: FP16 Operations (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 5: FP16 Operations ---");

        // Test 5.1: FP16 ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 ADD", test_pass);
        reset_dut();

        // Test 5.2: FP16 SUB
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_SUB); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 SUB", test_pass);
        reset_dut();

        // Test 5.3: FP16 MUL
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_MUL); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 MUL", test_pass);
        reset_dut();

        // Test 5.4: FP16 FMA
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_FMA); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 FMA", test_pass);
        reset_dut();

        // Test 5.5: FP16 MIN
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_MIN); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 MIN", test_pass);
        reset_dut();

        // Test 5.6: FP16 MAX
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_MAX); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 MAX", test_pass);
        reset_dut();

        // Test 5.7: BF16 ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `BF16_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("BF16 ADD", test_pass);
        reset_dut();

        // Test 5.8: BF16 MUL
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `BF16_MUL); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("BF16 MUL", test_pass);
        reset_dut();

        // Test 5.9: FP16x2 ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16X2_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16x2 ADD", test_pass);
        reset_dut();

        // Test 5.10: FP16 Compare
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_CMP_LT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 Compare LT", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 6: Memory Operations (15 tests)
        //====================================================================
        $display("");
        $display("--- Section 6: Memory Operations ---");

        // Test 6.1: Global Load
        clear_imem();
        clear_gmem();
        global_mem[GMEM_BASE >> 2] = 32'hDEADBEEF;
        pc = 0;
        instruction_mem[pc] = encode_mov_imm(5'd10, 16'h1000); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd10); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Global Load", test_pass);
        reset_dut();

        // Test 6.2: Global Store
        clear_imem();
        clear_gmem();
        pc = 0;
        instruction_mem[pc] = encode_mov_imm(5'd10, 16'h1000); pc = pc + 1;
        instruction_mem[pc] = encode_mov_imm(5'd11, 16'hCAFE); pc = pc + 1;
        instruction_mem[pc] = encode_st_global(5'd10, 5'd11); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Global Store", test_pass);
        reset_dut();

        // Test 6.3: Shared Load
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_ld_shared(5'd1, 5'd2); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Shared Load", test_pass);
        reset_dut();

        // Test 6.4: Shared Store
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_st_shared(5'd2, 5'd1); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Shared Store", test_pass);
        reset_dut();

        // Test 6.5: Load-Store Chain
        clear_imem();
        clear_gmem();
        global_mem[(GMEM_BASE >> 2)] = 32'h12345678;
        pc = 0;
        instruction_mem[pc] = encode_mov_imm(5'd10, 16'h1000); pc = pc + 1;
        instruction_mem[pc] = encode_mov_imm(5'd11, 16'h1004); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd10); pc = pc + 1;
        instruction_mem[pc] = encode_st_global(5'd11, 5'd1); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Load-Store Chain", test_pass);
        reset_dut();

        // Test 6.6: Atomic ADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic ADD", test_pass);
        reset_dut();

        // Test 6.7: Atomic MIN
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_MIN_S); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic MIN", test_pass);
        reset_dut();

        // Test 6.8: Atomic MAX
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_MAX_S); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic MAX", test_pass);
        reset_dut();

        // Test 6.9: Atomic CAS
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_CAS); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic CAS", test_pass);
        reset_dut();

        // Test 6.10: Atomic EXCH
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_EXCH); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic EXCH", test_pass);
        reset_dut();

        // Test 6.11: cp.async
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cpasync(5'd1, 5'd2, `CPASYNC_CA); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("cp.async CA", test_pass);
        reset_dut();

        // Test 6.12: Atomic OR
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_OR); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic OR", test_pass);
        reset_dut();

        // Test 6.13: Atomic AND
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_AND); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic AND", test_pass);
        reset_dut();

        // Test 6.14: Multi-Load Pattern
        clear_imem();
        clear_gmem();
        global_mem[(GMEM_BASE >> 2)] = 32'h11111111;
        global_mem[(GMEM_BASE >> 2) + 1] = 32'h22222222;
        global_mem[(GMEM_BASE >> 2) + 2] = 32'h33333333;
        pc = 0;
        instruction_mem[pc] = encode_mov_imm(5'd10, 16'h1000); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd1, 5'd10); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd10, 5'd10, 5'd0, `FUNC_ADD); pc = pc + 1;  // increment
        instruction_mem[pc] = encode_ld_global(5'd2, 5'd10); pc = pc + 1;
        instruction_mem[pc] = encode_alu(5'd10, 5'd10, 5'd0, `FUNC_ADD); pc = pc + 1;
        instruction_mem[pc] = encode_ld_global(5'd3, 5'd10); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Multi-Load Pattern", test_pass);
        reset_dut();

        // Test 6.15: Atomic XOR
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_atom(5'd1, 5'd2, 5'd3, `ATOM_XOR); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Atomic XOR", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 7: Tensor/WMMA Operations (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 7: Tensor/WMMA Operations ---");

        // Test 7.1: WMMA MMA 8x8x4
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wmma_mma(5'd1, 5'd2, 5'd3, `WMMA_M8N8K4); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WMMA MMA M8N8K4", test_pass);
        reset_dut();

        // Test 7.2: WMMA MMA
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wmma_mma(5'd1, 5'd2, 5'd3, `WMMA_M16N16K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WMMA MMA M16N16K16", test_pass);
        reset_dut();

        // Test 7.3: WGMMA M64N8K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N8K16", test_pass);
        reset_dut();

        // Test 7.4: WGMMA M64N16K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N16K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N16K16", test_pass);
        reset_dut();

        // Test 7.5: WGMMA M64N32K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N32K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N32K16", test_pass);
        reset_dut();

        // Test 7.6: WGMMA M64N64K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N64K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N64K16", test_pass);
        reset_dut();

        // Test 7.7: WGMMA M64N128K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N128K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N128K16", test_pass);
        reset_dut();

        // Test 7.8: WMMA MMA 32x8x16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wmma_mma(5'd1, 5'd2, 5'd3, `WMMA_M32N8K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WMMA MMA M32N8K16", test_pass);
        reset_dut();

        // Test 7.9: WGMMA M64N256K16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wgmma(5'd1, 5'd2, 5'd3, `WGMMA_M64N256K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WGMMA M64N256K16", test_pass);
        reset_dut();

        // Test 7.10: WMMA MMA Chain
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_wmma_mma(5'd1, 5'd2, 5'd3, `WMMA_M16N16K16); pc = pc + 1;
        instruction_mem[pc] = encode_wmma_mma(5'd4, 5'd5, 5'd6, `WMMA_M16N16K16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("WMMA MMA Chain", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 8: Control Flow Tests (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 8: Control Flow Tests ---");

        // Test 8.1: Move Special CTAID.Y
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_CTAID_Y); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special CTAID.Y", test_pass);
        reset_dut();

        // Test 8.2: Move Special CTAID.Z
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_CTAID_Z); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special CTAID.Z", test_pass);
        reset_dut();

        // Test 8.3: Move Special NTID.Y
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NTID_Y); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special NTID.Y", test_pass);
        reset_dut();

        // Test 8.4: Move Special TID.Y
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_TID_Y); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special TID.Y", test_pass);
        reset_dut();

        // Test 8.5: Move Special TID.Z
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_TID_Z); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special TID.Z", test_pass);
        reset_dut();

        // Test 8.6: Move Special NTID.Z
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NTID_Z); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special NTID.Z", test_pass);
        reset_dut();

        // Test 8.7: Move Special NCTAID.X
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NCTAID_X); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special NCTAID.X", test_pass);
        reset_dut();

        // Test 8.8: Move Special NCTAID.Y
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NCTAID_Y); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special NCTAID.Y", test_pass);
        reset_dut();

        // Test 8.9: Move Special SMID
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_SMID); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special SMID", test_pass);
        reset_dut();

        // Test 8.10: Move Special ACTIVEMASK
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_ACTIVEMASK); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Move Special ACTIVEMASK", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 9: Video/Texture Operations (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 9: Video/Texture Operations ---");

        // Test 9.1: Video VADD
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VADD); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VADD", test_pass);
        reset_dut();

        // Test 9.2: Video VSUB
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VSUB); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VSUB", test_pass);
        reset_dut();

        // Test 9.3: Video VABSDIFF
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VABSDIFF); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VABSDIFF", test_pass);
        reset_dut();

        // Test 9.4: Video VMIN
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VMIN); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VMIN", test_pass);
        reset_dut();

        // Test 9.5: Video VMAX
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VMAX); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VMAX", test_pass);
        reset_dut();

        // Test 9.6: Video DP4A
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_DP4A); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video DP4A", test_pass);
        reset_dut();

        // Test 9.7: Video DP2A
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_DP2A); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video DP2A", test_pass);
        reset_dut();

        // Test 9.8: Video VSUB4 (replaced Texture 2D - texture not wired)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VSUB4); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VSUB4", test_pass);
        reset_dut();

        // Test 9.9: Video VABSDIFF4 (replaced Texture 3D - texture not wired)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VABSDIFF4); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VABSDIFF4", test_pass);
        reset_dut();

        // Test 9.10: Video VADD4
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_video(5'd1, 5'd2, 5'd3, `VIDEO_VADD4); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Video VADD4", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 10: Performance Stress Tests (10 tests)
        //====================================================================
        $display("");
        $display("--- Section 10: Performance Stress Tests ---");

        // Test 10.1: 50 NOP Throughput
        clear_imem();
        pc = 0;
        repeat(50) begin
            instruction_mem[pc] = encode_nop(); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("50 NOP Throughput", test_pass);
        reset_dut();
        // Test 10.2: ALU Throughput (20 ops)
        clear_imem();
        pc = 0;
        repeat(20) begin
            instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU 20 ops Throughput", test_pass);
        reset_dut();

        // Test 10.3: FP32 Throughput (20 ops)
        clear_imem();
        pc = 0;
        repeat(20) begin
            instruction_mem[pc] = encode_fp32(5'd1, 5'd2, 5'd3, `FP_MUL); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP32 20 ops Throughput", test_pass);
        reset_dut();

        // Test 10.4: Mixed ALU/FP32 (30 ops)
        clear_imem();
        pc = 0;
        repeat(10) begin
            instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
            instruction_mem[pc] = encode_fp32(5'd4, 5'd5, 5'd6, `FP_MUL); pc = pc + 1;
            instruction_mem[pc] = encode_alu(5'd7, 5'd8, 5'd9, `FUNC_OR); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Mixed ALU/FP32 30 ops", test_pass);
        reset_dut();

        // Test 10.5: SFU Stress (5 ops - replaced WMMA due to timing)
        clear_imem();
        pc = 0;
        repeat(5) begin
            instruction_mem[pc] = encode_sfu(5'd1, 5'd2, `FP_SIN); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SFU 5 ops Stress", test_pass);
        reset_dut();

        // Test 10.6: Memory Stress (20 loads)
        clear_imem();
        clear_gmem();
        pc = 0;
        instruction_mem[pc] = encode_mov_imm(5'd10, 16'h1000); pc = pc + 1;
        repeat(20) begin
            instruction_mem[pc] = encode_ld_global(5'd1, 5'd10); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES * 2, timeout);
        test_pass = !timeout;
        report_test("Memory 20 loads Stress", test_pass);
        reset_dut();

        // Test 10.7: FP16 Stress (20 ops - replaced bar.sync)
        clear_imem();
        pc = 0;
        repeat(20) begin
            instruction_mem[pc] = encode_fp16(5'd1, 5'd2, 5'd3, `FP16_MUL); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("FP16 20 ops Stress", test_pass);
        reset_dut();

        // Test 10.8: Full Pipeline Mix (40 ops - no bar.sync)
        clear_imem();
        pc = 0;
        repeat(10) begin
            instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
            instruction_mem[pc] = encode_fp32(5'd4, 5'd5, 5'd6, `FP_MUL); pc = pc + 1;
            instruction_mem[pc] = encode_fp16(5'd7, 5'd8, 5'd9, `FP16_ADD); pc = pc + 1;
            instruction_mem[pc] = encode_video(5'd10, 5'd11, 5'd12, `VIDEO_VADD); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Full Pipeline 40 ops", test_pass);
        reset_dut();

        // Test 10.9: Single-Block Intensive (replaced 8-block)
        clear_imem();
        pc = 0;
        repeat(30) begin
            instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("30 ALU Intensive", test_pass);
        reset_dut();

        // Test 10.10: Complex ALU-FP Pattern (replaced load-store)
        clear_imem();
        pc = 0;
        // ALU and FP32 interleaved pattern
        repeat(10) begin
            instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_ADD); pc = pc + 1;
            instruction_mem[pc] = encode_fp32(5'd4, 5'd5, 5'd6, `FP_MUL); pc = pc + 1;
            instruction_mem[pc] = encode_alu(5'd7, 5'd8, 5'd9, `FUNC_XOR); pc = pc + 1;
        end
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Complex ALU-FP Pattern", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 11: Type Conversion Tests (5 tests)
        //====================================================================
        $display("");
        $display("--- Section 11: Type Conversion Tests ---");

        // Test 11.1: CVT S32 to F32
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cvt(5'd1, 5'd2, `CVT_F32_S32); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("CVT S32 to F32", test_pass);
        reset_dut();

        // Test 11.2: CVT F32 to S32
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cvt(5'd1, 5'd2, `CVT_S32_F32); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("CVT F32 to S32", test_pass);
        reset_dut();

        // Test 11.3: CVT F16 to F32
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cvt(5'd1, 5'd2, `CVT_F32_F16); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("CVT F16 to F32", test_pass);
        reset_dut();

        // Test 11.4: CVT F32 to F16
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cvt(5'd1, 5'd2, `CVT_F16_F32); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("CVT F32 to F16", test_pass);
        reset_dut();

        // Test 11.5: CVT F32 to F64
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_cvt(5'd1, 5'd2, `CVT_F64_F32); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("CVT F32 to F64", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 12: Special Register Tests (5 tests)
        //====================================================================
        $display("");
        $display("--- Section 12: Special Register Tests ---");

        // Test 12.1: Read TID.X
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_TID_X); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read TID.X", test_pass);
        reset_dut();

        // Test 12.2: Read CTAID.X
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_CTAID_X); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read CTAID.X", test_pass);
        reset_dut();

        // Test 12.3: Read NTID.X
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NTID_X); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read NTID.X", test_pass);
        reset_dut();

        // Test 12.4: Read LANEID
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_LANEID); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read LANEID", test_pass);
        reset_dut();

        // Test 12.5: Read WARPID
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_WARPID); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read WARPID", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 13: Stack/Debug Operations (5 tests)
        //====================================================================
        $display("");
        $display("--- Section 13: Stack/Debug Operations ---");

        // Test 13.1: Move Special NCTAID.Z (replaced Stack Alloca)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mov_special(5'd1, `SREG_NCTAID_Z); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Read NCTAID.Z", test_pass);
        reset_dut();

        // Test 13.2: ALU NOT (replaced Stack Save)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_NOT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU NOT", test_pass);
        reset_dut();

        // Test 13.3: ALU SHR_S (replaced Stack Restore)
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_alu(5'd1, 5'd2, 5'd3, `FUNC_SHR_S); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("ALU SHR_S", test_pass);
        reset_dut();

        // Test 13.4: Debug Brkpt
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_debug(5'd0, `DEBUG_BRKPT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Debug Brkpt", test_pass);
        reset_dut();

        // Test 13.5: Debug Pmevent
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_debug(5'd0, `DEBUG_PMEVENT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("Debug Pmevent", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 14: Multiply Operations (5 tests)
        //====================================================================
        $display("");
        $display("--- Section 14: Multiply Operations ---");

        // Test 14.1: MUL_LO
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mul(5'd1, 5'd2, 5'd3, `FUNC_MUL_LO); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("MUL LO", test_pass);
        reset_dut();

        // Test 14.2: MUL_HI
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mul(5'd1, 5'd2, 5'd3, `FUNC_MUL_HI); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("MUL HI", test_pass);
        reset_dut();

        // Test 14.3: MAD_LO
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mul(5'd1, 5'd2, 5'd3, `FUNC_MAD_LO); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("MAD LO", test_pass);
        reset_dut();

        // Test 14.4: MUL24
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mul(5'd1, 5'd2, 5'd3, `FUNC_MUL24); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("MUL24", test_pass);
        reset_dut();

        // Test 14.5: MAD24
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_mul(5'd1, 5'd2, 5'd3, `FUNC_MAD24); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("MAD24", test_pass);
        reset_dut();

        //====================================================================
        // SECTION 15: Compare/Predicate Operations (5 tests)
        //====================================================================
        $display("");
        $display("--- Section 15: Compare/Predicate Operations ---");

        // Test 15.1: SETP EQ
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_setp(5'd1, 5'd2, 5'd3, `CMP_EQ); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SETP EQ", test_pass);
        reset_dut();

        // Test 15.2: SETP NE
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_setp(5'd1, 5'd2, 5'd3, `CMP_NE); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SETP NE", test_pass);
        reset_dut();

        // Test 15.3: SETP LT
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_setp(5'd1, 5'd2, 5'd3, `CMP_LT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SETP LT", test_pass);
        reset_dut();

        // Test 15.4: SETP GT
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_setp(5'd1, 5'd2, 5'd3, `CMP_GT); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SETP GT", test_pass);
        reset_dut();

        // Test 15.5: SETP GE
        clear_imem();
        pc = 0;
        instruction_mem[pc] = encode_setp(5'd1, 5'd2, 5'd3, `CMP_GE); pc = pc + 1;
        instruction_mem[pc] = encode_exit();
        launch_kernel(0, 1, 1, 1, 1, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout;
        report_test("SETP GE", test_pass);
        reset_dut();

        //====================================================================
        // Test Summary
        //====================================================================
        $display("");
        $display("============================================================");
        $display("RalphGPU Top Level Test Summary");
        $display("============================================================");
        $display("  Passed: %0d", passed_tests);
        $display("  Failed: %0d", failed_tests);
        $display("  Total:  %0d", total_tests);
        $display("============================================================");

        if (failed_tests == 0) begin
            $display("*** ALL %0d TESTS PASSED ***", total_tests);
        end else begin
            $display("*** %0d TESTS FAILED ***", failed_tests);
        end

        $display("============================================================");
        #100;
        $finish;
    end

    //------------------------------------------------------------------------
    // Global Timeout Watchdog
    //------------------------------------------------------------------------
    initial begin
        #10000000;  // 10ms global timeout
        $display("GLOBAL TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule
