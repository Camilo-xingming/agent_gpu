//============================================================================
// RalphGPU - SM V2 Integration Testbench
// Tests: Scoreboard, FU Latency Hiding, Round-Robin Writeback, CFU
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_integration;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;  // 100 MHz

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
    // Kernel interface
    reg         kernel_start;
    reg  [31:0] kernel_pc;
    reg  [31:0] block_id_x, block_id_y, block_id_z;
    reg  [31:0] block_dim_x, block_dim_y, block_dim_z;
    reg  [31:0] grid_dim_x, grid_dim_y, grid_dim_z;
    wire        kernel_done;

    // Instruction memory interface
    wire        imem_req;
    wire [31:0] imem_addr;
    wire        imem_ready;
    reg  [31:0] imem_data;
    reg         imem_valid;

    // AXI memory interface (simplified for test)
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

    // L1D cache interface
    wire        l1d_req_valid;
    wire        l1d_req_write;
    wire [31:0] l1d_req_addr [0:NUM_LANES-1];
    wire [31:0] l1d_req_wdata [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [31:0] l1d_resp_rdata [0:NUM_LANES-1];
    reg         l1d_resp_valid;
    reg         l1d_resp_hit;

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    integer cycle_count;
    integer instruction_count;
    integer stall_cycles_raw;
    integer stall_cycles_fu;
    integer stall_cycles_mem;
    integer warp_switch_count;
    real    ipc;

`ifdef DEBUG_SM_V2
    integer dbg_cycle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_cycle <= 0;
        end else begin
            dbg_cycle <= dbg_cycle + 1;
            if (dut.issue_valid) begin
                $display("[DBG] C%0d issue op=0x%02X func=0x%02X rd=%0d ra=%0d exit=%0b",
                         dbg_cycle, dut.issue_opcode, dut.issue_func,
                         dut.issue_rd, dut.issue_ra, dut.issue_exit_op);
            end
            if (dut.fpu32_valid_in) begin
                $display("[DBG] C%0d fpu32_in", dbg_cycle);
            end
            if (dut.wb_valid) begin
                $display("[DBG] C%0d wb warp=%0d rd=%0d", dbg_cycle, dut.wb_warp_id, dut.wb_rd);
            end
            if (dut.fpu32_valid_out) begin
                $display("[DBG] C%0d fpu32_out rd_pipe=%0d", dbg_cycle, dut.fpu32_rd_pipe[4]);
            end
            if (kernel_done) begin
                $display("[DBG] C%0d kernel_done", dbg_cycle);
            end
        end
    end
`endif

    //------------------------------------------------------------------------
    // Instruction Memory (ROM)
    //------------------------------------------------------------------------
    reg [31:0] imem [0:1023];
    reg [1:0] imem_valid_pipe;
    reg [31:0] imem_addr_pipe [0:1];

    // Simulated instruction memory with pipelined 2-cycle latency
    always @(posedge clk) begin
        if (!rst_n) begin
            imem_valid_pipe <= 2'b0;
            imem_addr_pipe[0] <= 0;
            imem_addr_pipe[1] <= 0;
        end else begin
            imem_valid_pipe[0] <= imem_req;
            imem_addr_pipe[0] <= imem_addr;
            imem_valid_pipe[1] <= imem_valid_pipe[0];
            imem_addr_pipe[1] <= imem_addr_pipe[0];
        end
    end

    always @(posedge clk) begin
        imem_valid <= imem_valid_pipe[1];
        if (imem_valid_pipe[1]) begin
            imem_data <= imem[imem_addr_pipe[1][11:2]];
        end
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH)
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
    // Memory Response Model (variable latency)
    //------------------------------------------------------------------------
    reg [7:0] mem_latency_counter;
    reg       mem_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            mem_pending <= 1'b0;
            mem_latency_counter <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                mem_pending <= 1'b1;
                mem_latency_counter <= 8'd20;  // 20-cycle global memory latency
                m_axi_arready <= 1'b0;
            end else if (mem_pending && mem_latency_counter > 0) begin
                mem_latency_counter <= mem_latency_counter - 1;
            end else if (mem_pending && mem_latency_counter == 0) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= 32'hDEADBEEF;
                m_axi_rlast <= 1'b1;
                mem_pending <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_arready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Test Instruction Encoding Helpers
    //------------------------------------------------------------------------
    // Simplified instruction format:
    // [31:26] opcode, [25:21] rd, [20:16] ra, [15:11] rb, [10:6] rc, [5:0] func

    function [31:0] encode_alu;
        input [4:0] rd, ra, rb;
        input [5:0] func;
        begin
            encode_alu = {`OP_ALU, rd, ra, rb, 5'b0, func};  // ALU opcode
        end
    endfunction

    function [31:0] encode_fpu;
        input [4:0] rd, ra, rb;
        input [5:0] func;
        begin
            encode_fpu = {`OP_FP32_ARITH, rd, ra, rb, 5'b0, func};  // FPU opcode
        end
    endfunction

    function [31:0] encode_load;
        input [4:0] rd, ra;
        input [15:0] offset;
        begin
            encode_load = {`OP_LD_GLOBAL, rd, ra, offset};  // LOAD opcode
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};  // EXIT opcode
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Programs
    //------------------------------------------------------------------------

    // Test 1: RAW Hazard Detection
    task load_test_raw_hazard;
        begin
            $display("\n[TEST 1] RAW Hazard Detection");
            // R1 = R0 + R0 (ALU, 1 cycle)
            imem[0] = encode_alu(5'd1, 5'd0, 5'd0, 6'h00);  // ADD R1, R0, R0
            // R2 = R1 + R0 (depends on R1 - should stall)
            imem[1] = encode_alu(5'd2, 5'd1, 5'd0, 6'h00);  // ADD R2, R1, R0
            // R3 = R2 + R0 (depends on R2 - should stall)
            imem[2] = encode_alu(5'd3, 5'd2, 5'd0, 6'h00);  // ADD R3, R2, R0
            imem[3] = encode_exit();
        end
    endtask

    // Test 2: FPU Multi-cycle Latency
    task load_test_fpu_latency;
        begin
            $display("\n[TEST 2] FPU Multi-cycle Latency (4 cycles)");
            // FP32 operations take 4 cycles
            imem[0] = encode_fpu(5'd1, 5'd0, 5'd0, 6'h00);  // FADD R1, R0, R0
            imem[1] = encode_fpu(5'd2, 5'd0, 5'd0, 6'h00);  // FADD R2, R0, R0 (independent)
            imem[2] = encode_fpu(5'd3, 5'd0, 5'd0, 6'h00);  // FADD R3, R0, R0 (independent)
            imem[3] = encode_fpu(5'd4, 5'd0, 5'd0, 6'h00);  // FADD R4, R0, R0 (independent)
            // Now use R1 - should have completed by now
            imem[4] = encode_alu(5'd5, 5'd1, 5'd0, 6'h00);  // ADD R5, R1, R0
            imem[5] = encode_exit();
        end
    endtask

    // Test 3: Mixed ALU/FPU Interleaving
    task load_test_interleave;
        begin
            $display("\n[TEST 3] ALU/FPU Interleaving");
            imem[0] = encode_fpu(5'd1, 5'd0, 5'd0, 6'h00);  // FADD (4 cycles)
            imem[1] = encode_alu(5'd2, 5'd0, 5'd0, 6'h00);  // ADD (1 cycle, independent)
            imem[2] = encode_alu(5'd3, 5'd2, 5'd0, 6'h00);  // ADD (uses R2)
            imem[3] = encode_alu(5'd4, 5'd3, 5'd0, 6'h00);  // ADD (uses R3)
            imem[4] = encode_alu(5'd5, 5'd4, 5'd0, 6'h00);  // ADD (uses R4)
            imem[5] = encode_alu(5'd6, 5'd1, 5'd0, 6'h00);  // ADD (uses R1 from FPU)
            imem[6] = encode_exit();
        end
    endtask

    // Test 4: Writeback Arbitration Stress
    task load_test_wb_arb;
        begin
            $display("\n[TEST 4] Writeback Arbitration Stress");
            // Issue multiple FPUs to create simultaneous completions
            imem[0] = encode_fpu(5'd1, 5'd0, 5'd0, 6'h00);
            imem[1] = encode_fpu(5'd2, 5'd0, 5'd0, 6'h00);
            imem[2] = encode_fpu(5'd3, 5'd0, 5'd0, 6'h00);
            imem[3] = encode_fpu(5'd4, 5'd0, 5'd0, 6'h00);
            // These should all complete around same time
            imem[4] = encode_alu(5'd5, 5'd0, 5'd0, 6'h00);
            imem[5] = encode_alu(5'd6, 5'd0, 5'd0, 6'h00);
            // Verify all results
            imem[6] = encode_alu(5'd7, 5'd1, 5'd2, 6'h00);
            imem[7] = encode_alu(5'd8, 5'd3, 5'd4, 6'h00);
            imem[8] = encode_exit();
        end
    endtask

    // Test 5: Memory Latency Hiding (needs warp switching)
    task load_test_mem_latency;
        begin
            $display("\n[TEST 5] Memory Latency Hiding");
            imem[0] = encode_load(5'd1, 5'd0, 16'h0000);  // LOAD R1 (20 cycles)
            imem[1] = encode_alu(5'd2, 5'd0, 5'd0, 6'h00);
            imem[2] = encode_alu(5'd3, 5'd0, 5'd0, 6'h00);
            imem[3] = encode_alu(5'd4, 5'd1, 5'd0, 6'h00);  // Use loaded value
            imem[4] = encode_exit();
        end
    endtask

    //------------------------------------------------------------------------
    // Performance Tracking
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && kernel_start && !kernel_done) begin
            cycle_count <= cycle_count + 1;
        end
    end

    // Track instruction completions via writeback
    always @(posedge clk) begin
        if (dut.wb_valid) begin
            instruction_count <= instruction_count + 1;
        end
    end

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer test_num;
    integer test_cycles;

    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 Integration Test");
        $display("Testing: Scoreboard, FU Tracking, WB Arbitration");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 32; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        imem_valid = 0;
        l1d_resp_valid = 0;
        l1d_resp_hit = 0;
        m_axi_awready = 1;
        m_axi_wready = 1;
        m_axi_bvalid = 0;

        // Clear instruction memory
        for (test_num = 0; test_num < 1024; test_num = test_num + 1) begin
            imem[test_num] = 32'b0;
        end

        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        //--------------------------------------------------------------------
        // Run Tests
        //--------------------------------------------------------------------

        // Test 1: RAW Hazard
        load_test_raw_hazard();
        run_kernel(50);
        report_test(1, "RAW Hazard Detection");

        // Test 2: FPU Latency
        load_test_fpu_latency();
        run_kernel(100);
        report_test(2, "FPU Multi-cycle Latency");

        // Test 3: Interleaving
        load_test_interleave();
        run_kernel(80);
        report_test(3, "ALU/FPU Interleaving");

        // Test 4: WB Arbitration
        load_test_wb_arb();
        run_kernel(100);
        report_test(4, "Writeback Arbitration");

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("\n============================================================");
        $display("TEST SUMMARY");
        $display("============================================================");
        $display("All integration tests completed.");
        $display("============================================================\n");

        $finish;
    end

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------

    task run_kernel;
        input integer max_cycles;
        begin
            cycle_count = 0;
            instruction_count = 0;

            @(posedge clk);
            kernel_start = 1;
            kernel_pc = 32'h0000_0000;
            @(posedge clk);
            kernel_start = 0;

            // Wait for completion or timeout
            fork
                begin
                    wait(kernel_done);
                end
                begin
                    repeat(max_cycles) @(posedge clk);
                    $display("  WARNING: Kernel timeout after %0d cycles", max_cycles);
                end
            join_any
            disable fork;

            repeat(5) @(posedge clk);
        end
    endtask

    task report_test;
        input integer num;
        input [255:0] name;
        begin
            ipc = (cycle_count > 0) ? (1.0 * instruction_count / cycle_count) : 0.0;
            $display("  Test %0d: %s", num, name);
            $display("    Cycles: %0d, Instructions: %0d, IPC: %0.3f",
                     cycle_count, instruction_count, ipc);
            if (kernel_done)
                $display("    Status: PASSED (kernel completed)");
            else
                $display("    Status: TIMEOUT");
        end
    endtask

endmodule
