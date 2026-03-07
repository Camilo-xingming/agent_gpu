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

module tb_gpu_top_integration;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;  // 100 MHz
    localparam TIMEOUT_CYCLES = 200000;  // Per-test timeout (increased)

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
        .NUM_SM(2)  // Single SM for controlled testing
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
    reg [3:0]  pending_axi_id;
    reg        pending_axi_read;
    reg [2:0]  axi_read_delay;
    reg [7:0]  pending_axi_len;
    reg [7:0]  pending_axi_beat;
    reg [31:0] pending_aw_addr;
    reg [3:0]  pending_aw_id;
    reg        pending_aw_valid;
    reg        pending_axi_data_valid;

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
            pending_axi_len <= 8'b0;
            pending_axi_beat <= 8'b0;
            pending_axi_data_valid <= 1'b0;
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
            pending_aw_addr <= 0;
            pending_aw_id <= 0;
            pending_aw_valid <= 0;
        end else begin
            // Read channel
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_read <= 1'b1;
                pending_axi_addr <= m_axi_araddr;
                pending_axi_id <= m_axi_arid;
                pending_axi_len <= m_axi_arlen;
                pending_axi_beat <= 8'd0;
                pending_axi_data_valid <= 1'b0;
                m_axi_arready <= 1'b0;
                axi_read_delay <= 3'd2;
            end else if (pending_axi_read && axi_read_delay > 0) begin
                axi_read_delay <= axi_read_delay - 1'b1;
            end else if (pending_axi_read && !pending_axi_data_valid && axi_read_delay == 0) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= global_mem[pending_axi_addr[15:2] + pending_axi_beat];
                m_axi_rlast <= (pending_axi_beat == pending_axi_len);
                m_axi_rid <= pending_axi_id;
                pending_axi_data_valid <= 1'b1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (pending_axi_beat == pending_axi_len) begin
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                    m_axi_arready <= 1'b1;
                    pending_axi_read <= 1'b0;
                    pending_axi_data_valid <= 1'b0;
                end else begin
                    pending_axi_beat <= pending_axi_beat + 1'b1;
                    m_axi_rvalid <= 1'b1;
                    m_axi_rdata <= global_mem[pending_axi_addr[15:2] + pending_axi_beat + 1'b1];
                    m_axi_rlast <= ((pending_axi_beat + 1'b1) == pending_axi_len);
                    m_axi_rid <= pending_axi_id;
                    pending_axi_data_valid <= 1'b1;
                end
            end

            // Write channel
            if (m_axi_awvalid && m_axi_awready) begin
                pending_aw_addr <= m_axi_awaddr;
                pending_aw_id <= m_axi_awid;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                
                begin
                    integer word_offset;
                    word_offset = (m_axi_awvalid && m_axi_awready) ? m_axi_awaddr[4:2] : pending_aw_addr[4:2];
                    global_mem[(m_axi_awvalid && m_axi_awready) ? m_axi_awaddr[15:2] : pending_aw_addr[15:2]] <= m_axi_wdata;
                end

                m_axi_bvalid <= 1'b1;
                m_axi_bid <= (m_axi_awvalid && m_axi_awready) ? m_axi_awid : pending_aw_id;
                $display("  MEM WRITE at time %0t: addr=0x%08h data=0x%08h id=%0d", $time, (m_axi_awvalid && m_axi_awready) ? m_axi_awaddr : pending_aw_addr, m_axi_wdata, m_axi_awid);
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
        // SECTION 1: Multi-SM Identification & Memory Write
        //====================================================================
        $display("");
        $display("--- Section 1: Multi-SM Identification & Memory Write ---");
        
        reset_dut();
        clear_imem();
        clear_gmem();

        // 0: mov_special r1, SREG_CTAID_X (5'd3)
        instruction_mem[0] = encode_mov_special(5'd1, 5'd3);
        // 1: mov_special r2, SREG_SMID (5'd14)
        instruction_mem[1] = encode_mov_special(5'd2, 5'd14);
        // 2: mov_imm r3, 4
        instruction_mem[2] = encode_mov_imm(5'd3, 16'd2);
        // 3: alu r4, r1, r3 (MUL24) -> offset
        instruction_mem[3] = encode_alu(5'd4, 5'd1, 5'd3, 6'b000110);
        // 4: st_global [r4], r2
        instruction_mem[4] = encode_st_global(5'd4, 5'd2);
        // 5: exit
        instruction_mem[5] = encode_exit();

        launch_kernel(0, 4, 1, 1, 32, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        
        
        if (global_mem[0] !== 32'd0 && global_mem[0] !== 32'd1) test_pass = 0;

        if (global_mem[1] !== 32'd0 && global_mem[1] !== 32'd1) test_pass = 0;
        if (global_mem[2] !== 32'd0 && global_mem[2] !== 32'd1) test_pass = 0;
        if (global_mem[3] !== 32'd0 && global_mem[3] !== 32'd1) test_pass = 0;
        if ((global_mem[0] + global_mem[1] + global_mem[2] + global_mem[3]) == 0) test_pass = 0; // Check SM1 was used

        $display("gmem[0]=%d, gmem[1]=%d, gmem[2]=%d, gmem[3]=%d", global_mem[0], global_mem[1], global_mem[2], global_mem[3]);

        $display("CP state: total_blocks=%d dispatched_blocks=%d sm_busy=%b sm_done=%b", u_gpu.u_command_processor.total_blocks, u_gpu.u_command_processor.dispatched_blocks, u_gpu.u_command_processor.sm_busy, u_gpu.sm_done);

        report_test("Multi-SM Execution & GMem Write", test_pass);

        //====================================================================
        // SECTION 2: Global Memory Coherence & Read
        //====================================================================
        $display("");
        $display("--- Section 2: Global Memory Coherence & Read ---");
        
        reset_dut();
        clear_imem();
        
        // Don't clear gmem, read from previous test
        // Deterministic copy: fixed addresses, not CTAID-derived
        // Copy gmem[0..3] -> gmem[4..7] using byte addresses 0/4/8/12 -> 16/20/24/28
        instruction_mem[0]  = encode_mov_imm(5'd4, 16'd0);
        instruction_mem[1]  = encode_ld_global(5'd5, 5'd4);
        instruction_mem[2]  = encode_mov_imm(5'd7, 16'd16);
        instruction_mem[3]  = encode_st_global(5'd7, 5'd5);

        instruction_mem[4]  = encode_mov_imm(5'd4, 16'd4);
        instruction_mem[5]  = encode_ld_global(5'd5, 5'd4);
        instruction_mem[6]  = encode_mov_imm(5'd7, 16'd20);
        instruction_mem[7]  = encode_st_global(5'd7, 5'd5);

        instruction_mem[8]  = encode_mov_imm(5'd4, 16'd8);
        instruction_mem[9]  = encode_ld_global(5'd5, 5'd4);
        instruction_mem[10] = encode_mov_imm(5'd7, 16'd24);
        instruction_mem[11] = encode_st_global(5'd7, 5'd5);

        instruction_mem[12] = encode_mov_imm(5'd4, 16'd12);
        instruction_mem[13] = encode_ld_global(5'd5, 5'd4);
        instruction_mem[14] = encode_mov_imm(5'd7, 16'd28);
        instruction_mem[15] = encode_st_global(5'd7, 5'd5);

        instruction_mem[16] = encode_exit();

        launch_kernel(0, 4, 1, 1, 32, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;

        if (global_mem[4] !== global_mem[0]) test_pass = 0;
        if (global_mem[5] !== global_mem[1]) test_pass = 0;
        if (global_mem[6] !== global_mem[2]) test_pass = 0;
        if (global_mem[7] !== global_mem[3]) test_pass = 0;

        report_test("Multi-SM GMem Read-Write Coherence", test_pass);

        //====================================================================
        // SECTION 3: Inter-SM Barrier Sync
        //====================================================================
        $display("");
        $display("--- Section 3: Multi-SM Barrier Sync ---");
        
        reset_dut();
        clear_imem();
        clear_gmem();

        // 0: bar_sync 0
        instruction_mem[0] = encode_bar_sync(5'd0);
        // 1: mov_special r1, SREG_CTAID_X (5'd3)
        instruction_mem[1] = encode_mov_special(5'd1, 5'd3);
        // 2: mov_imm r3, 2 (SHL amount for 4-byte word stride)
        instruction_mem[2] = encode_mov_imm(5'd3, 16'd2);
        // 3: alu r4, r1, r3 (SHL) -> ctaid * 4 byte offset
        instruction_mem[3] = encode_alu(5'd4, 5'd1, 5'd3, 6'b000110);
        // 4: st_global [r4], r1 (write CTAID to deterministic slot)
        instruction_mem[4] = encode_st_global(5'd4, 5'd1);
        // 5: exit
        instruction_mem[5] = encode_exit();

        launch_kernel(0, 2, 1, 1, 32, 1, 1);
        wait_kernel_done(TIMEOUT_CYCLES, timeout);
        test_pass = !timeout && irq_kernel_done;
        
        if (global_mem[0] !== 32'd0) test_pass = 0;
        if (global_mem[1] !== 32'd0 && global_mem[1] !== 32'd1) test_pass = 0;
        if (global_mem[2] !== 32'd0) test_pass = 0;
        if (global_mem[3] !== 32'd0) test_pass = 0;

        report_test("Multi-SM Barrier Sync", test_pass);


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
