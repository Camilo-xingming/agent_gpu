//============================================================================
// RalphGPU - SM V2 Performance Test (GEMM 16x16x16 FMA Stream, PTX)
// Uses an external PTX-assembled hex file for instruction memory.
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_perf_gemm16_ptx;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 8192;
    localparam N_OPS      = 4096;  // 16*16*16 FMAs

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
    wire [NUM_LANES*32-1:0] l1d_req_addr;
    wire [NUM_LANES*32-1:0] l1d_req_wdata;
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg [NUM_LANES*32-1:0] l1d_resp_rdata;
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
    // Instruction Memory
    //------------------------------------------------------------------------
    reg [31:0] imem [0:IMEM_WORDS-1];
    reg        imem_req_q;
    reg [31:0] imem_addr_q;
    string     imem_file;
    integer    imem_fd;
    reg        imem_loaded;

    integer i;
    initial begin
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            imem[i] = {`OP_NOP, 26'b0};
        end
        if (!$value$plusargs("imem=%s", imem_file)) begin
            imem_file = "../programs/gemm16_fma.hex";
        end
        imem_loaded = 1'b0;

        // Try default cwd path first, then common build-relative fallback paths.
        imem_fd = $fopen(imem_file, "r");
        if (imem_fd != 0) begin
            $fclose(imem_fd);
            $readmemh(imem_file, imem, 0, N_OPS);
            imem_loaded = 1'b1;
        end else begin
            imem_fd = $fopen("../programs/gemm16_fma.hex", "r");
            if (imem_fd != 0) begin
                $fclose(imem_fd);
                imem_file = "../programs/gemm16_fma.hex";
                $readmemh(imem_file, imem, 0, N_OPS);
                imem_loaded = 1'b1;
            end else begin
                imem_fd = $fopen("programs/gemm16_fma.hex", "r");
                if (imem_fd != 0) begin
                    $fclose(imem_fd);
                    imem_file = "programs/gemm16_fma.hex";
                    $readmemh(imem_file, imem, 0, N_OPS);
                    imem_loaded = 1'b1;
                end
            end
        end

        if (!imem_loaded) begin
            $display("FATAL: unable to load instruction hex file (tried: %s, ../programs/gemm16_fma.hex, programs/gemm16_fma.hex)", imem_file);
            $finish;
        end else begin
            $display("INFO: loaded instruction hex: %s", imem_file);
        end
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
                imem_data <= {imem[imem_addr_q[14:2] + 1], imem[imem_addr_q[14:2]]};
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
        .INIT_WARPS(4),
        .ICACHE_BYPASS(1)  // Isolate core issue/wb performance from fetch cache misses
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
    initial begin
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            l1d_resp_rdata[i*32 +: 32] = 32'b0;
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
    // Performance Counters
    //------------------------------------------------------------------------
    integer cycle_count;
    integer wb_count;
    integer fetch_count;
    integer issue_count;
    integer stall_raw;
    integer stall_fu;
    integer stall_mem;
    integer stall_atomic;
    integer stall_tensor;
    integer stall_wbq;
    integer stall_ifetch;
    integer fu_alu_active;
    integer fu_fpu_active;
    integer fu_sfu_active;
    integer fu_ldst_active;
    integer fu_tensor_active;
    integer timeout_cycles;
    integer timeout_left;
    integer progress_interval;
    reg running;
    reg done;
    real ipc;
    wire wb_fire = dut.wb_valid && (dut.wb_rd != 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            cycle_count <= 0;
            wb_count <= 0;
            fetch_count <= 0;
            issue_count <= 0;
            stall_raw <= 0;
            stall_fu <= 0;
            stall_mem <= 0;
            stall_atomic <= 0;
            stall_tensor <= 0;
            stall_wbq <= 0;
            stall_ifetch <= 0;
            fu_alu_active <= 0;
            fu_fpu_active <= 0;
            fu_sfu_active <= 0;
            fu_ldst_active <= 0;
            fu_tensor_active <= 0;
        end else begin
            if (kernel_start) begin
                running <= 1'b1;
                done <= 1'b0;
                cycle_count <= 0;
                wb_count <= 0;
                fetch_count <= 0;
                issue_count <= 0;
                stall_raw <= 0;
                stall_fu <= 0;
                stall_mem <= 0;
                stall_atomic <= 0;
                stall_tensor <= 0;
                stall_wbq <= 0;
                stall_ifetch <= 0;
            end else if (running) begin
                cycle_count <= cycle_count + 1;
                if (wb_fire) begin
                    wb_count <= wb_count + 1;
                end
                if (imem_req) begin
                    fetch_count <= fetch_count + 1;
                end
                if (dut.issue_valid) begin
                    issue_count <= issue_count + 1;
                end
                if (dut.perf_stall_ifetch) begin
                    stall_ifetch <= stall_ifetch + 1;
                end
                if (dut.lane0_stall_raw) begin
                    stall_raw <= stall_raw + 1;
                end
                if (dut.lane0_stall_fu) begin
                    stall_fu <= stall_fu + 1;
                end
                if (dut.lane0_stall_mem) begin
                    stall_mem <= stall_mem + 1;
                end
                if (dut.lane0_stall_atomic) begin
                    stall_atomic <= stall_atomic + 1;
                end
                if (dut.lane0_stall_tensor) begin
                    stall_tensor <= stall_tensor + 1;
                end
                if (dut.lane0_stall_wbq) begin
                    stall_wbq <= stall_wbq + 1;
                end
                if (dut.perf_fu_alu_active) begin
                    fu_alu_active <= fu_alu_active + 1;
                end
                if (dut.perf_fu_fpu_active) begin
                    fu_fpu_active <= fu_fpu_active + 1;
                end
                if (dut.perf_fu_sfu_active) begin
                    fu_sfu_active <= fu_sfu_active + 1;
                end
                if (dut.perf_fu_ldst_active) begin
                    fu_ldst_active <= fu_ldst_active + 1;
                end
                if (dut.perf_fu_tensor_active) begin
                    fu_tensor_active <= fu_tensor_active + 1;
                end
                if (wb_count + (wb_fire ? 1 : 0) >= N_OPS) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 PTX Performance Test");
        $display("Kernel: GEMM 16x16x16 (FMA stream from PTX)");
        $display("Ops: %0d FMA", N_OPS);
        $display("============================================================");

        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 16; block_dim_y = 16; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        @(posedge clk);
        kernel_start = 1;
        kernel_pc = 32'h0000_0000;
        @(posedge clk);
        kernel_start = 0;

        timeout_cycles = (N_OPS * 8) + 2000;  // Default headroom for post-RALPH-10c timing
        progress_interval = 2000;
        if ($value$plusargs("timeout_cycles=%d", timeout_cycles)) begin
            $display("INFO: override timeout_cycles=%0d", timeout_cycles);
        end
        if ($value$plusargs("progress_interval=%d", progress_interval)) begin
            $display("INFO: override progress_interval=%0d", progress_interval);
        end
        if (progress_interval <= 0) begin
            progress_interval = 2000;
        end
        timeout_left = timeout_cycles;
        while (!done && (timeout_left > 0)) begin
            @(posedge clk);
            timeout_left = timeout_left - 1;
            if ((cycle_count > 0) && ((cycle_count % progress_interval) == 0)) begin
                $display("PROGRESS: cycle=%0d wb=%0d issue=%0d fetch=%0d stalls(raw=%0d fu=%0d mem=%0d atomic=%0d tensor=%0d wbq=%0d ifetch=%0d timeout_left=%0d",
                         cycle_count, wb_count, issue_count, fetch_count,
                         stall_raw, stall_fu, stall_mem, stall_atomic, stall_tensor, stall_wbq, stall_ifetch,
                         timeout_left);
            end
        end

        ipc = (cycle_count > 0) ? (1.0 * wb_count / cycle_count) : 0.0;
        $display("Cycles: %0d", cycle_count);
        $display("Writebacks: %0d", wb_count);
        $display("Fetches: %0d", fetch_count);
        $display("Issues: %0d", issue_count);
        $display("Stalls: raw=%0d fu=%0d mem=%0d atomic=%0d tensor=%0d wbq=%0d ifetch=%0d",
                 stall_raw, stall_fu, stall_mem, stall_atomic, stall_tensor, stall_wbq, stall_ifetch);
        $display("IPC: %0.3f", ipc);
        $display("FU: ALU Active = %0d", fu_alu_active);
        $display("FU: FPU Active = %0d", fu_fpu_active);
        $display("FU: SFU Active = %0d", fu_sfu_active);
        $display("FU: LDST Active = %0d", fu_ldst_active);
        $display("FU: Tensor Active = %0d", fu_tensor_active);

        if (!done) begin
            $display("FAIL: timeout before completing all FMAs");
        end else if (wb_count != N_OPS) begin
            $display("FAIL: expected %0d writebacks, got %0d", N_OPS, wb_count);
        end else begin
            $display("PASS: completed GEMM FMA stream");
        end

        $finish;
    end

endmodule
