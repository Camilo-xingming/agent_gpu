//============================================================================
// RalphGPU - SM V2 WGMMA GEMM 64x8x16 PTX Path Test
// Verifies WGMMA mma_async + commit/wait/fence execution plumbing.
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_perf_gemm64_wgmma_ptx;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 8192;

    localparam EXPECTED_WGMMA_MMAS    = 8;
    localparam EXPECTED_WGMMA_COMMITS = 1;
    localparam EXPECTED_WGMMA_WAITS   = 1;
    localparam EXPECTED_WGMMA_FENCES  = 1;

    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

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
    reg  [NUM_LANES*32-1:0] l1d_resp_rdata;
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
            imem_file = "gemm64_wgmma.hex";
        end
        imem_loaded = 1'b0;

        imem_fd = $fopen(imem_file, "r");
        if (imem_fd != 0) begin
            $fclose(imem_fd);
            $readmemh(imem_file, imem);
            imem_loaded = 1'b1;
        end else begin
            imem_fd = $fopen("../gemm64_wgmma.hex", "r");
            if (imem_fd != 0) begin
                $fclose(imem_fd);
                imem_file = "../gemm64_wgmma.hex";
                $readmemh(imem_file, imem);
                imem_loaded = 1'b1;
            end else begin
                imem_fd = $fopen("../../gemm64_wgmma.hex", "r");
                if (imem_fd != 0) begin
                    $fclose(imem_fd);
                    imem_file = "../../gemm64_wgmma.hex";
                    $readmemh(imem_file, imem);
                    imem_loaded = 1'b1;
                end
            end
        end

        if (!imem_loaded) begin
            $display("FATAL: unable to load instruction hex file (tried: %s, ../gemm64_wgmma.hex, ../../gemm64_wgmma.hex)", imem_file);
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

    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .INIT_WARPS(1)
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

    initial begin
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        l1d_resp_rdata = {NUM_LANES*32{1'b0}};
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

    integer cycle_count;
    integer fetch_count;
    integer wb_count;
    integer wgmma_mma_issues;
    integer wgmma_commit_issues;
    integer wgmma_wait_issues;
    integer wgmma_fence_issues;
    integer wgmma_smem_reads;
    integer wgmma_done_pulses;
    integer tensor_issue_count;
    integer tensor_wb_count;
    integer timeout_cycles;
    integer timeout_left;
    integer progress_interval;
    reg running;
    reg done;

    wire wb_fire = dut.wb_valid && (dut.wb_rd != 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            cycle_count <= 0;
            fetch_count <= 0;
            wb_count <= 0;
            wgmma_mma_issues <= 0;
            wgmma_commit_issues <= 0;
            wgmma_wait_issues <= 0;
            wgmma_fence_issues <= 0;
            wgmma_smem_reads <= 0;
            wgmma_done_pulses <= 0;
            tensor_issue_count <= 0;
            tensor_wb_count <= 0;
        end else begin
            if (kernel_start) begin
                running <= 1'b1;
                done <= 1'b0;
                cycle_count <= 0;
                fetch_count <= 0;
                wb_count <= 0;
                wgmma_mma_issues <= 0;
                wgmma_commit_issues <= 0;
                wgmma_wait_issues <= 0;
                wgmma_fence_issues <= 0;
                wgmma_smem_reads <= 0;
                wgmma_done_pulses <= 0;
                tensor_issue_count <= 0;
                tensor_wb_count <= 0;
            end else if (running) begin
                cycle_count <= cycle_count + 1;
                if (kernel_done) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end
                if (imem_req) begin
                    fetch_count <= fetch_count + 1;
                end
                if (wb_fire) begin
                    wb_count <= wb_count + 1;
                end

                if (dut.issue_valid && dut.issue_opcode == `OP_WGMMA_MMA) begin
                    case (dut.issue_func)
                        `WGMMA_COMMIT_GROUP: wgmma_commit_issues <= wgmma_commit_issues + 1;
                        `WGMMA_WAIT_GROUP:   wgmma_wait_issues <= wgmma_wait_issues + 1;
                        `WGMMA_FENCE:        wgmma_fence_issues <= wgmma_fence_issues + 1;
                        default:             wgmma_mma_issues <= wgmma_mma_issues + 1;
                    endcase
                end

                if (dut.wgmma_smem_rd_en) begin
                    wgmma_smem_reads <= wgmma_smem_reads + 1;
                end
                if (dut.wgmma_done) begin
                    wgmma_done_pulses <= wgmma_done_pulses + 1;
                end
                if (dut.tensor_issue_push_fire) begin
                    tensor_issue_count <= tensor_issue_count + 1;
                end
                if (dut.tensor_wbq_push_fire) begin
                    tensor_wb_count <= tensor_wb_count + 1;
                end
            end
        end
    end

    initial begin
        $display("============================================================");
        $display("RalphGPU SM V2 WGMMA PTX Path Test");
        $display("Kernel: GEMM 64x8x16 (WGMMA mma_async + commit/wait/fence)");
        $display("============================================================");

        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 32; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        @(posedge clk);
        kernel_start = 1;
        kernel_pc = 32'h0000_0000;
        @(posedge clk);
        kernel_start = 0;

        timeout_cycles = 30000;
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
                $display("PROGRESS: cycle=%0d fetch=%0d wb=%0d mma=%0d commit=%0d wait=%0d fence=%0d smem_rd=%0d done=%0d tensor_issue=%0d tensor_wb=%0d timeout_left=%0d",
                         cycle_count, fetch_count, wb_count,
                         wgmma_mma_issues, wgmma_commit_issues, wgmma_wait_issues, wgmma_fence_issues,
                         wgmma_smem_reads, wgmma_done_pulses,
                         tensor_issue_count, tensor_wb_count,
                         timeout_left);
            end
        end

        $display("Cycles: %0d", cycle_count);
        $display("Fetches: %0d", fetch_count);
        $display("Writebacks: %0d", wb_count);
        $display("WGMMA issues: mma=%0d commit=%0d wait=%0d fence=%0d", wgmma_mma_issues, wgmma_commit_issues, wgmma_wait_issues, wgmma_fence_issues);
        $display("WGMMA SMEM reads: %0d", wgmma_smem_reads);
        $display("WGMMA done pulses: %0d", wgmma_done_pulses);
        $display("Tensor path: issue=%0d wb=%0d", tensor_issue_count, tensor_wb_count);

        if (!done) begin
            $display("FAIL: timeout waiting for kernel_done");
        end else if (wgmma_mma_issues < EXPECTED_WGMMA_MMAS) begin
            $display("FAIL: expected at least %0d WGMMA MMA ops, got %0d", EXPECTED_WGMMA_MMAS, wgmma_mma_issues);
        end else if (wgmma_commit_issues < EXPECTED_WGMMA_COMMITS) begin
            $display("FAIL: expected commit_group >= %0d, got %0d", EXPECTED_WGMMA_COMMITS, wgmma_commit_issues);
        end else if (wgmma_wait_issues < EXPECTED_WGMMA_WAITS) begin
            $display("FAIL: expected wait_group >= %0d, got %0d", EXPECTED_WGMMA_WAITS, wgmma_wait_issues);
        end else if (wgmma_fence_issues < EXPECTED_WGMMA_FENCES) begin
            $display("FAIL: expected fence >= %0d, got %0d", EXPECTED_WGMMA_FENCES, wgmma_fence_issues);
        end else if (wgmma_smem_reads == 0) begin
            $display("FAIL: expected WGMMA SMEM read activity, got 0");
        end else if (wgmma_done_pulses == 0) begin
            $display("FAIL: expected WGMMA completion pulse, got 0");
        end else begin
            $display("PASS: WGMMA PTX path exercised (mma_async + commit/wait/fence)");
        end

        $finish;
    end

endmodule
