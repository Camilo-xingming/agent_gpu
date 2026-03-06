//============================================================================
// RalphGPU - SM V2 Comprehensive Multiwarp Testbench
// Tests: warp lifecycle, fairness, stall recovery, bank conflict, barrier sync
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_comprehensive;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam WARP_ID_W  = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1;
    localparam TEST_INIT_WARPS = (NUM_WARPS >= 4) ? 4 : NUM_WARPS;
    localparam CLK_PERIOD = 10;  // 100 MHz

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
    end

    // Hard timeout to prevent stuck simulation
    initial begin
        #600000;
        $display("ABSOLUTE TIMEOUT");
        $finish;
    end

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
    wire [63:0] imem_data;
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
    // Monitors
    //------------------------------------------------------------------------
    integer cycle_count;
    integer instruction_count;
    integer warp_switch_count;
    integer max_active_warps;
    integer pass_count;
    integer fail_count;
    integer active_warp_now;

    reg monitor_enable;
    reg saw_warp_launch;
    reg saw_mem_stall;
    reg saw_sync_stall;
    reg saw_bank_conflict;
    reg saw_issue;
    reg [WARP_ID_W-1:0] last_issue_warp;
    reg [NUM_WARPS-1:0] issued_warp_mask;
    integer scenario_select;

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

    reg [63:0] imem_data_wide;
    always @(posedge clk) begin
        imem_valid <= imem_valid_pipe[1];
        if (imem_valid_pipe[1]) begin
            imem_data_wide <= {imem[imem_addr_pipe[1][11:2] + 1], imem[imem_addr_pipe[1][11:2]]};
        end
    end
    assign imem_data = imem_data_wide;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .INIT_WARPS(TEST_INIT_WARPS)
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
                mem_latency_counter <= 8'd20;
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
    // Monitor logic
    //------------------------------------------------------------------------
    function integer count_warps;
        input [NUM_WARPS-1:0] mask;
        integer i;
        begin
            count_warps = 0;
            for (i = 0; i < NUM_WARPS; i = i + 1) begin
                if (mask[i])
                    count_warps = count_warps + 1;
            end
        end
    endfunction

    always @(posedge clk) begin
        if (monitor_enable) begin
            if (dut.wb_valid)
                instruction_count <= instruction_count + 1;

            if (dut.issue_valid) begin
                issued_warp_mask[dut.issue_warp_id] <= 1'b1;
                if (saw_issue && (last_issue_warp != dut.issue_warp_id))
                    warp_switch_count <= warp_switch_count + 1;
                last_issue_warp <= dut.issue_warp_id;
                saw_issue <= 1'b1;
            end

            if (dut.warp_valid != {NUM_WARPS{1'b0}})
                saw_warp_launch <= 1'b1;
            if (|dut.warp_stalled_mem)
                saw_mem_stall <= 1'b1;
            if (|dut.warp_stalled_sync)
                saw_sync_stall <= 1'b1;
            if (dut.rf_conflict_a || dut.rf_conflict_b || dut.rf_conflict_c ||
                dut.rf1_conflict_a || dut.rf1_conflict_b || dut.rf1_conflict_c)
                saw_bank_conflict <= 1'b1;

            active_warp_now = count_warps(dut.warp_valid);
            if (active_warp_now > max_active_warps)
                max_active_warps <= active_warp_now;
        end
    end

    //------------------------------------------------------------------------
    // Test Instruction Encoding Helpers
    //------------------------------------------------------------------------
    function [31:0] encode_alu;
        input [4:0] rd, ra, rb;
        input [5:0] func;
        begin
            encode_alu = {`OP_ALU, rd, ra, rb, 5'b0, func};
        end
    endfunction

    function [31:0] encode_load;
        input [4:0] rd, ra;
        input [15:0] offset;
        begin
            encode_load = {`OP_LD_GLOBAL, rd, ra, offset};
        end
    endfunction

    function [31:0] encode_bar_sync;
        input [15:0] barrier_id;
        begin
            encode_bar_sync = {`OP_BAR_SYNC, 10'b0, barrier_id};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Programs
    //------------------------------------------------------------------------
    task clear_imem;
        integer i;
        begin
            for (i = 0; i < 1024; i = i + 1)
                imem[i] = 32'b0;
        end
    endtask

    // Multi-warp lifecycle + fairness + stall recovery
    task load_lifecycle_fairness_stall;
        begin
            clear_imem();
            imem[0] = encode_load(5'd1, 5'd0, 16'h0000);
            imem[1] = encode_alu(5'd2, 5'd1, 5'd0, 6'h00);
            imem[2] = encode_alu(5'd3, 5'd2, 5'd0, 6'h00);
            imem[3] = encode_alu(5'd4, 5'd3, 5'd1, 6'h00);
            imem[4] = encode_exit();
        end
    endtask

    // Bank conflict stress: source register pairs mapped to same modulo-4 bank
    task load_bank_conflict_program;
        begin
            clear_imem();
            imem[0] = encode_alu(5'd4, 5'd1, 5'd5, 6'h00);
            imem[1] = encode_alu(5'd8, 5'd2, 5'd6, 6'h00);
            imem[2] = encode_alu(5'd12, 5'd3, 5'd7, 6'h00);
            imem[3] = encode_alu(5'd16, 5'd4, 5'd8, 6'h00);
            imem[4] = encode_exit();
        end
    endtask

    // Cross-warp barrier sync
    task load_barrier_sync_program;
        begin
            clear_imem();
            imem[0] = encode_alu(5'd1, 5'd0, 5'd0, 6'h00);
            imem[1] = encode_bar_sync(16'h0000);
            imem[2] = encode_alu(5'd2, 5'd1, 5'd0, 6'h00);
            imem[3] = encode_exit();
        end
    endtask

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task reset_monitors;
        begin
            cycle_count = 0;
            instruction_count = 0;
            warp_switch_count = 0;
            max_active_warps = 0;
            saw_warp_launch = 1'b0;
            saw_mem_stall = 1'b0;
            saw_sync_stall = 1'b0;
            saw_bank_conflict = 1'b0;
            saw_issue = 1'b0;
            last_issue_warp = {WARP_ID_W{1'b0}};
            issued_warp_mask = {NUM_WARPS{1'b0}};
            active_warp_now = 0;
        end
    endtask

    task run_kernel;
        input integer timeout;
        begin
            rst_n = 0;
            repeat(10) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            reset_monitors();
            monitor_enable = 1'b1;

            kernel_start = 1;
            @(posedge clk);
            kernel_start = 0;

            while (!kernel_done && cycle_count < timeout) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            monitor_enable = 1'b0;
        end
    endtask

    task check_true;
        input [255:0] name;
        input cond;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
                $display("  [PASS] %s", name);
            end else begin
                fail_count = fail_count + 1;
                $display("  [FAIL] %s", name);
            end
        end
    endtask

    task report_metrics;
        input [255:0] name;
        begin
            $display("  [%s] cycles=%0d wb=%0d issued_warps=%0d switches=%0d max_active=%0d",
                     name,
                     cycle_count,
                     instruction_count,
                     count_warps(issued_warp_mask),
                     warp_switch_count,
                     max_active_warps);
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 Comprehensive Coverage Test (Issue #531)");
        $display("Lifecycle/Fairness/Stall/BankConflict/Barrier");
        $display("============================================================");

        // Initialize static TB inputs
        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = TEST_INIT_WARPS * NUM_LANES;
        block_dim_y = 1;
        block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        imem_valid = 0;
        l1d_resp_valid = 0;
        l1d_resp_hit = 0;
        l1d_resp_rdata = {NUM_LANES*32{1'b0}};
        m_axi_awready = 1;
        m_axi_wready = 1;
        m_axi_bvalid = 0;
        m_axi_bid = 0;
        m_axi_bresp = 0;
        m_axi_arready = 1;
        m_axi_rid = 0;
        m_axi_rresp = 0;
        m_axi_rdata = 0;
        m_axi_rlast = 0;
        monitor_enable = 1'b0;

        pass_count = 0;
        fail_count = 0;

        scenario_select = 0;
        if ($value$plusargs("SCENARIO=%d", scenario_select))
            $display("[INFO] Running single scenario SCENARIO=%0d", scenario_select);

        //--------------------------------------------------------------------
        // Scenario 1: Warp lifecycle + fairness + memory stall recovery
        //--------------------------------------------------------------------
        if (scenario_select == 0 || scenario_select == 1) begin
            $display("\n[SCENARIO 1] lifecycle/fairness/stall recovery");
            load_lifecycle_fairness_stall();
            run_kernel(80);
            report_metrics("S1");

            check_true("S1 warp launched", saw_warp_launch);
            check_true("S1 active warps reached target", max_active_warps >= TEST_INIT_WARPS);
            check_true("S1 fairness: multiple warps issued", count_warps(issued_warp_mask) >= 2);
            check_true("S1 fairness: warp switched", warp_switch_count >= 1);
            check_true("S1 memory stall observed", saw_mem_stall);
        end

        //--------------------------------------------------------------------
        // Scenario 2: Register bank conflict handling
        //--------------------------------------------------------------------
        if (scenario_select == 0 || scenario_select == 2) begin
            $display("\n[SCENARIO 2] register-bank conflict");
            load_bank_conflict_program();
            run_kernel(80);
            report_metrics("S2");

            check_true("S2 multiple warps issued", count_warps(issued_warp_mask) >= 2);
            check_true("S2 register-bank conflict observed", saw_bank_conflict);
        end

        //--------------------------------------------------------------------
        // Scenario 3: Barrier synchronization across warps
        //--------------------------------------------------------------------
        if (scenario_select == 0 || scenario_select == 3) begin
            $display("\n[SCENARIO 3] barrier synchronization");
            load_barrier_sync_program();
            run_kernel(100);
            report_metrics("S3");

            check_true("S3 sync stall observed", saw_sync_stall);
        end

        $display("\n============================================================");
        $display("Coverage summary: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("============================================================");

        if (fail_count != 0) begin
            $fatal(1, "tb_sm_v2_comprehensive: %0d checks failed", fail_count);
        end

        $finish;
    end

endmodule
