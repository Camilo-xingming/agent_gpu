//============================================================================
// RalphGPU - PTX Compiled Test Testbench
// Loads compiled PTX hex files and runs verification with performance metrics
//
// Tests compiled from asm/ptx_comprehensive_tests/*.ptx
// Success marker: 0xCAFE at 0x2000 indicates PASS
// Failure marker: 0xDEAD at 0x2000 indicates FAIL
//============================================================================

`timescale 1ns / 1ps

`include "gpu_defines.vh"

module tb_ptx_tests;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;  // 100 MHz
    localparam TIMEOUT_CYCLES = 200000;  // Per-test timeout

    // Memory layout
    localparam IMEM_BASE = 32'h0000_0000;
    localparam GMEM_BASE = 32'h0000_1000;  // Global memory starts at 4KB
    localparam RESULT_ADDR = 32'h0000_2000;  // Test result marker address

    // Expected values
    localparam PASS_MARKER = 32'h0000CAFE;
    localparam FAIL_MARKER = 32'h0000DEAD;

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
    // Test Counters and Performance Metrics
    //------------------------------------------------------------------------
    integer total_tests;
    integer passed_tests;
    integer failed_tests;
    integer test_num;

    // Performance counters
    integer cycle_count;
    integer total_cycles;
    integer min_cycles;
    integer max_cycles;
    integer instruction_count;

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

    // Instruction fetch - SAME-CYCLE response (combinatorial)
    // The SM's fetch pipeline expects same-cycle responses when ICACHE_BYPASS=1
    // This is the simplest model that works with the SM's same_cycle_hit logic
    always @(*) begin
        if (imem_req) begin
            imem_data = {instruction_mem[(imem_addr >> 2) + 1],
                         instruction_mem[imem_addr >> 2]};
            imem_valid = 1'b1;
        end else begin
            imem_data = 64'b0;
            imem_valid = 1'b0;
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
            m_axi_rlast <= 1'b0;
            m_axi_rresp <= 2'b00;
            m_axi_rid <= 4'b0;
            pending_axi_read <= 1'b0;
            pending_axi_addr <= 32'b0;
            axi_read_delay <= 3'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_addr <= m_axi_araddr;
                pending_axi_read <= 1'b1;
                axi_read_delay <= 3'd2;  // 2-cycle delay
                m_axi_arready <= 1'b0;
            end else if (pending_axi_read && axi_read_delay > 0) begin
                axi_read_delay <= axi_read_delay - 1;
            end else if (pending_axi_read && axi_read_delay == 0) begin
                m_axi_rdata <= global_mem[(pending_axi_addr - GMEM_BASE) >> 2];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
                pending_axi_read <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end
        end
    end

    // AXI Write Channel
    reg        pending_axi_write;
    reg [31:0] pending_axi_waddr;
    reg [2:0]  axi_write_delay;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
            pending_axi_write <= 1'b0;
            pending_axi_waddr <= 32'b0;
            axi_write_delay <= 3'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_axi_waddr <= m_axi_awaddr;
                pending_axi_write <= 1'b1;
                m_axi_awready <= 1'b0;
            end

            if (m_axi_wvalid && m_axi_wready && pending_axi_write) begin
                if (pending_axi_waddr >= GMEM_BASE) begin
                    global_mem[(pending_axi_waddr - GMEM_BASE) >> 2] <= m_axi_wdata;
                end
                pending_axi_write <= 1'b0;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
                m_axi_awready <= 1'b1;
            end

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 1'b0;
            csr_wr_en = 1'b0;
            csr_addr = 12'b0;
            csr_wr_data = 32'b0;
            #(CLK_PERIOD * 10);
            rst_n = 1'b1;
            #(CLK_PERIOD * 5);
        end
    endtask

    task write_csr;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr = addr;
            csr_wr_data = data;
            csr_wr_en = 1'b1;
            @(posedge clk);
            csr_wr_en = 1'b0;
        end
    endtask

    task clear_imem;
        integer i;
        begin
            for (i = 0; i < 4096; i = i + 1) begin
                instruction_mem[i] = 32'h00000000;  // NOP
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

    task wait_kernel_done_with_cycles;
        input integer max_cycles;
        output reg timeout;
        output integer cycles;
        begin
            timeout = 0;
            cycles = 0;
            while (!irq_kernel_done && cycles < max_cycles) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            if (cycles >= max_cycles) begin
                timeout = 1;
            end
            #20;
        end
    endtask

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    reg test_pass;
    integer test_cycles;
    reg timeout;
    reg [31:0] result;
    integer num_instr;
    integer num_warps;    // Number of warps in test
    real ipc;
    real throughput;

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU PTX Compiled Test Suite (Multi-Warp Mode)");
        $display("Testing with compiled hex files from PTX assembler");
        $display("Configuration: 4 warps (128 threads) per test");
        $display("============================================================");
        $display("");

        // Initialize counters
        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        test_num = 0;
        total_cycles = 0;
        min_cycles = 999999999;
        max_cycles = 0;
        num_warps = 4;  // 128 threads / 32 threads per warp

        // Initial reset
        reset_dut();

        //====================================================================
        // Test 1: ALU Basic
        //====================================================================
        $display("=== Functional Tests ===");
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_01_alu_basic.hex", instruction_mem);
        num_instr = 46;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: ALU Basic | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: ALU Basic | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 2: ALU Extended
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_02_alu_extended.hex", instruction_mem);
        num_instr = 43;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: ALU Extended | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: ALU Extended | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 3: Multiply
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_03_multiply.hex", instruction_mem);
        num_instr = 41;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Multiply | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Multiply | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 4: FP32 Arithmetic
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_04_fp32_arith.hex", instruction_mem);
        num_instr = 39;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: FP32 Arith | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: FP32 Arith | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 5: FP32 Special
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_05_fp32_special.hex", instruction_mem);
        num_instr = 40;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: FP32 Special | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: FP32 Special | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 6: FP16 Arithmetic
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_06_fp16_arith.hex", instruction_mem);
        num_instr = 39;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: FP16 Arith | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: FP16 Arith | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 7: Global Memory
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_07_memory_global.hex", instruction_mem);
        num_instr = 43;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Global Memory | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Global Memory | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 8: Shared Memory
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_08_memory_shared.hex", instruction_mem);
        num_instr = 47;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Shared Memory | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Shared Memory | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 9: Atomics
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_09_atomic.hex", instruction_mem);
        num_instr = 76;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Atomics | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Atomics | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 10: Type Conversions
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_10_cvt.hex", instruction_mem);
        num_instr = 53;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: CVT | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: CVT | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 11: Special Registers
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_11_special_regs.hex", instruction_mem);
        num_instr = 26;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Special Regs | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Special Regs | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 12: SETP Compare
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_12_setp_compare.hex", instruction_mem);
        num_instr = 40;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: SETP Compare | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: SETP Compare | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 13: Video SIMD
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_13_video_ops.hex", instruction_mem);
        num_instr = 92;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Video SIMD | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Video SIMD | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 14: WMMA
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_14_wmma.hex", instruction_mem);
        num_instr = 47;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: WMMA | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: WMMA | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 15: Control Flow
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_15_control_flow.hex", instruction_mem);
        num_instr = 39;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Control Flow | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Control Flow | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Memory Consistency Model Tests
        //====================================================================
        $display("");
        $display("=== Memory Consistency Model Tests (PTX 9.1) ===");

        //====================================================================
        // Test 16: Message Passing
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_20_mem_consistency_mp.hex", instruction_mem);
        num_instr = 32;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Message Passing | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Message Passing | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 17: Store Buffering
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_21_mem_consistency_sb.hex", instruction_mem);
        num_instr = 40;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Store Buffering | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Store Buffering | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 18: Coherence
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_22_mem_consistency_coherence.hex", instruction_mem);
        num_instr = 39;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Coherence | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Coherence | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 19: Atomicity
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_23_mem_consistency_atomicity.hex", instruction_mem);
        num_instr = 21;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Atomicity | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Atomicity | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Performance Tests
        //====================================================================
        $display("");
        $display("=== Performance Benchmark Tests ===");

        //====================================================================
        // Test 20: ALU Throughput
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_30_perf_alu_throughput.hex", instruction_mem);
        num_instr = 104;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            throughput = (100.0 * num_instr) / test_cycles;
            $display("[PASS] Test %0d: ALU Throughput | Cycles: %0d | IPC: %0.3f | MIPS: %0.2f", test_num, test_cycles, ipc, throughput);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: ALU Throughput | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 21: FP32 Throughput
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_31_perf_fp_throughput.hex", instruction_mem);
        num_instr = 75;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(32'h2004 - GMEM_BASE) >> 2];  // FP test uses 0x2004 for marker
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            throughput = (100.0 * num_instr) / test_cycles;
            $display("[PASS] Test %0d: FP32 Throughput | Cycles: %0d | IPC: %0.3f | MIPS: %0.2f", test_num, test_cycles, ipc, throughput);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: FP32 Throughput | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test 22: Memory Latency
        //====================================================================
        reset_dut();
        clear_gmem();
        clear_imem();
        $readmemh("sim/test_32_perf_memory_latency.hex", instruction_mem);
        num_instr = 50;
        launch_kernel(0, 1, 1, 1, 128, 1, 1);  // 4 warps for latency hiding
        wait_kernel_done_with_cycles(TIMEOUT_CYCLES, timeout, test_cycles);
        result = global_mem[(RESULT_ADDR - GMEM_BASE) >> 2];
        test_pass = !timeout && (result == PASS_MARKER);
        test_num = test_num + 1;
        total_tests = total_tests + 1;
        total_cycles = total_cycles + test_cycles;
        if (test_cycles < min_cycles && test_cycles > 0) min_cycles = test_cycles;
        if (test_cycles > max_cycles) max_cycles = test_cycles;
        if (test_pass) begin
            passed_tests = passed_tests + 1;
            ipc = (1.0 * num_warps * num_instr) / test_cycles;
            $display("[PASS] Test %0d: Memory Latency | Cycles: %0d | IPC: %0.3f", test_num, test_cycles, ipc);
        end else begin
            failed_tests = failed_tests + 1;
            $display("[FAIL] Test %0d: Memory Latency | Cycles: %0d | Result: 0x%08X", test_num, test_cycles, result);
        end

        //====================================================================
        // Test Summary
        //====================================================================
        $display("");
        $display("============================================================");
        $display("RalphGPU PTX Test Suite Summary");
        $display("============================================================");
        $display("  Passed: %0d", passed_tests);
        $display("  Failed: %0d", failed_tests);
        $display("  Total:  %0d", total_tests);
        $display("------------------------------------------------------------");
        $display("Performance Metrics:");
        $display("  Total Cycles:   %0d", total_cycles);
        $display("  Min Cycles:     %0d", min_cycles);
        $display("  Max Cycles:     %0d", max_cycles);
        if (total_tests > 0) begin
            $display("  Avg Cycles:     %0d", total_cycles / total_tests);
        end
        $display("============================================================");

        if (failed_tests == 0) begin
            $display("*** ALL %0d PTX TESTS PASSED ***", total_tests);
        end else begin
            $display("*** %0d PTX TESTS FAILED ***", failed_tests);
        end

        $display("============================================================");
        #100;
        $finish;
    end

    //------------------------------------------------------------------------
    // Global Timeout Watchdog
    //------------------------------------------------------------------------
    initial begin
        #100000000;  // 100ms global timeout
        $display("GLOBAL TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule
