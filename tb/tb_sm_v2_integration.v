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
    wire [63:0] imem_data;  // 64-bit for 8-byte cache line
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
    wire [NUM_LANES*32-1:0] l1d_req_addr;
    wire [NUM_LANES*32-1:0] l1d_req_wdata;
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg [NUM_LANES*32-1:0] l1d_resp_rdata;
    reg         l1d_resp_valid;
    reg         l1d_resp_hit;

    //------------------------------------------------------------------------
    // Performance Counters
    real ipc;

    //------------------------------------------------------------------------
    integer cycle_count;
    integer instruction_count;
    integer stall_cycles_raw;
    integer stall_cycles_fu;
    integer stall_cycles_mem;
    integer warp_switch_count;

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

    // Return 2 words (64 bits) for 8-byte cache line fills
    // Cache line address is 8-byte aligned, so bits [2:0] = 0
    // Words at offset 0 and 4 within the line
    reg [63:0] imem_data_wide;
    always @(posedge clk) begin
        imem_valid <= imem_valid_pipe[1];
        if (imem_valid_pipe[1]) begin
            // imem_addr is cache-line aligned (8-byte), return both words
            imem_data_wide <= {imem[imem_addr_pipe[1][11:2] + 1], imem[imem_addr_pipe[1][11:2]]};
        end
    end
    assign imem_data = imem_data_wide;  // Full 64-bit data for 8-byte cache line

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


    function [31:0] encode_video;
        input [4:0] rd, ra, rb, rc;
        input [5:0] func;
        begin
            encode_video = {`OP_VIDEO, rd, ra, rb, rc, func};
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Programs
    //------------------------------------------------------------------------
    task clear_program;
        integer i;
        begin
            for (i = 0; i < 1024; i = i + 1) begin
                imem[i] = 32'b0;
            end
        end
    endtask

    // Scenario 1: GEMM-style mixed compute chain (ALU + FP32)
    task load_scenario_gemm_like;
        begin
            clear_program();
            imem[0] = encode_alu(5'd1, 5'd0, 5'd0, `FUNC_ADD);
            imem[1] = encode_fpu(5'd2, 5'd1, 5'd0, `FP_ADD);
            imem[2] = encode_fpu(5'd3, 5'd2, 5'd1, `FP_MUL);
            imem[3] = encode_alu(5'd4, 5'd3, 5'd1, `FUNC_ADD);
            imem[4] = encode_alu(5'd5, 5'd4, 5'd2, `FUNC_ADD);
            imem[5] = encode_exit();
        end
    endtask

    // Scenario 2: Reduction-style path using video SIMD and dependent compute
    task load_scenario_video_reduce;
        begin
            clear_program();
            imem[0] = encode_video(5'd6, 5'd0, 5'd0, 5'd0, `VIDEO_VADD4);
            imem[1] = encode_video(5'd7, 5'd0, 5'd0, 5'd0, `VIDEO_DP4A);
            imem[2] = encode_alu(5'd8, 5'd6, 5'd7, `FUNC_ADD);
            imem[3] = encode_fpu(5'd9, 5'd8, 5'd0, `FP_ADD);
            imem[4] = encode_exit();
        end
    endtask

    // Scenario 3: Memory-coalescing style stream (multiple global loads)
    task load_scenario_memory_stream;
        begin
            clear_program();
            imem[0] = encode_load(5'd10, 5'd0, 16'h0000);
            imem[1] = encode_load(5'd11, 5'd0, 16'h0004);
            imem[2] = encode_load(5'd12, 5'd0, 16'h0008);
            imem[3] = encode_alu(5'd13, 5'd10, 5'd11, `FUNC_ADD);
            imem[4] = encode_alu(5'd14, 5'd13, 5'd12, `FUNC_ADD);
            imem[5] = encode_exit();
        end
    endtask

    //------------------------------------------------------------------------
    // Runtime Monitors: WB reference counting + X-propagation checks
    //------------------------------------------------------------------------
    reg monitor_active;
    integer x_warning_count;

    always @(posedge clk) begin
        if (monitor_active && (dut.wb_valid === 1'b1)) begin
            instruction_count <= instruction_count + 1;
        end
    end

    always @(posedge clk) begin
        if (monitor_active) begin
            if ((kernel_done === 1'bx) || (imem_req === 1'bx) || (m_axi_arvalid === 1'bx) ||
                (m_axi_rvalid === 1'bx) || (l1d_req_valid === 1'bx) || (dut.wb_valid === 1'bx)) begin
                x_warning_count = x_warning_count + 1;
                if (x_warning_count <= 8) begin
                    $display("[X-WARN][%0t] Unknown control detected: kernel_done=%b imem_req=%b arvalid=%b rvalid=%b l1d_req=%b wb_valid=%b",
                             $time, kernel_done, imem_req, m_axi_arvalid, m_axi_rvalid, l1d_req_valid, dut.wb_valid);
                end
            end

            if ((imem_req === 1'b1) && (^imem_addr === 1'bx)) begin
                x_warning_count = x_warning_count + 1;
                if (x_warning_count <= 8) $display("[X-WARN][%0t] imem_addr has X while imem_req=1", $time);
            end
            if ((m_axi_arvalid === 1'b1) && (^m_axi_araddr === 1'bx)) begin
                x_warning_count = x_warning_count + 1;
                if (x_warning_count <= 8) $display("[X-WARN][%0t] m_axi_araddr has X while arvalid=1", $time);
            end
        end
    end

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;
    integer x_before;

    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 Top-level Integration (Issue #451)");
        $display("Scenarios: GEMM-like, reduction-like, memory-coalescing");
        $display("Checks: WB reference count + X propagation");
        $display("============================================================");

        // Initialize testbench-side signals
        rst_n = 1'b0;
        kernel_start = 1'b0;
        kernel_pc = 32'b0;
        block_id_x = 32'b0; block_id_y = 32'b0; block_id_z = 32'b0;
        block_dim_x = 32'd128; block_dim_y = 32'd1; block_dim_z = 32'd1;
        grid_dim_x = 32'd1; grid_dim_y = 32'd1; grid_dim_z = 32'd1;
        imem_valid = 1'b0;
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        l1d_resp_rdata = {NUM_LANES*32{1'b0}};
        m_axi_awready = 1'b1;
        m_axi_wready = 1'b1;
        m_axi_bvalid = 1'b0;
        m_axi_bresp = 2'b00;
        m_axi_bid = 4'b0;
        m_axi_rid = 4'b0;
        m_axi_rresp = 2'b00;
        m_axi_rlast = 1'b0;
        m_axi_rdata = 32'h0;

        cycle_count = 0;
        instruction_count = 0;
        stall_cycles_raw = 0;
        stall_cycles_fu = 0;
        stall_cycles_mem = 0;
        warp_switch_count = 0;
        pass_count = 0;
        fail_count = 0;
        x_warning_count = 0;
        monitor_active = 1'b0;

        clear_program();

        //------------------------------------------------------------------------
        // Scenario 1
        //------------------------------------------------------------------------
        load_scenario_gemm_like();
        x_before = x_warning_count;
        run_kernel(500);
        report_and_check(1,
                         "GEMM-like mixed ALU/FP32",
                         5,
                         x_before);

        //------------------------------------------------------------------------
        // Scenario 2
        //------------------------------------------------------------------------
        load_scenario_video_reduce();
        x_before = x_warning_count;
        run_kernel(500);
        report_and_check(2,
                         "Reduction-like video SIMD",
                         4,
                         x_before);

        //------------------------------------------------------------------------
        // Scenario 3
        //------------------------------------------------------------------------
        load_scenario_memory_stream();
        x_before = x_warning_count;
        run_kernel(700);
        report_and_check(3,
                         "Memory-coalescing style stream",
                         5,
                         x_before);

        $display("\n============================================================");
        $display("SM V2 Integration Summary: PASS=%0d FAIL=%0d X_WARN_TOTAL=%0d", pass_count, fail_count, x_warning_count);
        $display("============================================================");

        if (fail_count != 0) begin
            $fatal(1, "SM V2 integration checks failed");
        end

        $finish;
    end

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task run_kernel;
        input integer timeout;
        begin
            // Clean reset per scenario for deterministic state
            rst_n = 1'b0;
            repeat(10) @(posedge clk);
            rst_n = 1'b1;
            repeat(5) @(posedge clk);

            cycle_count = 0;
            instruction_count = 0;
            monitor_active = 1'b1;

            kernel_start = 1'b1;
            @(posedge clk);
            kernel_start = 1'b0;

            while (!kernel_done && cycle_count < timeout) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            monitor_active = 1'b0;
        end
    endtask

    task report_and_check;
        input integer num;
        input [255:0] name;
        input integer expected_wb_count;
        input integer x_before_local;
        begin
            ipc = (cycle_count > 0) ? (1.0 * instruction_count / cycle_count) : 0.0;
            $display("  Scenario %0d: %s", num, name);
            $display("    Cycles=%0d WB(actual)=%0d WB(expected)=%0d IPC=%0.3f",
                     cycle_count, instruction_count, expected_wb_count, ipc);

            if (!kernel_done) begin
                fail_count = fail_count + 1;
                $display("    Status: FAIL (kernel timeout)");
            end else if (instruction_count != expected_wb_count) begin
                fail_count = fail_count + 1;
                $display("    Status: FAIL (WB count mismatch)");
            end else if (x_warning_count != x_before_local) begin
                fail_count = fail_count + 1;
                $display("    Status: FAIL (X-propagation warning detected)");
            end else begin
                pass_count = pass_count + 1;
                $display("    Status: PASS");
            end
        end
    endtask

endmodule
