//============================================================================
// RalphGPU - SM V2 Tensor Core Performance Test (WMMA stream)
// Streams WMMA MMA ops to measure sustained throughput.
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_perf_tensor;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 8192;
    localparam N_OPS      = 2048;
    localparam REG_STRIDE = 16;

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
    reg  [31:0] imem_data;
    reg         imem_valid;

    wire        l1d_req_valid;
    wire        l1d_req_write;
    wire [31:0] l1d_req_addr [0:NUM_LANES-1];
    wire [31:0] l1d_req_wdata [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [31:0] l1d_resp_rdata [0:NUM_LANES-1];
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

    function [31:0] encode_wmma_mma;
        input [4:0] rd, ra, rb, rc;
        input [2:0] dtype;
        input [2:0] shape;
        reg [5:0] func;
        begin
            func = {shape, dtype};
            encode_wmma_mma = {`OP_WMMA_MMA, rd, ra, rb, rc, func};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    integer i;
    initial begin
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            imem[i] = {`OP_NOP, 26'b0};
        end
        for (i = 0; i < N_OPS; i = i + 1) begin
            imem[i] = encode_wmma_mma((i % REG_STRIDE) + 1, 5'd0, 5'd0, 5'd0,
                                     `TC_DATA_FP16, `WMMA_M16N16K16);
        end
        imem[N_OPS] = encode_exit();
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_req_q <= 1'b0;
            imem_addr_q <= 0;
            imem_data <= 32'b0;
        end else begin
            imem_valid <= imem_req_q;
            if (imem_req_q) begin
                imem_data <= imem[imem_addr_q[14:2]];
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
    // Memory Interfaces (Idle)
    //------------------------------------------------------------------------
    initial begin
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            l1d_resp_rdata[i] = 32'b0;
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
    integer timeout_cycles;
    integer timeout_left;
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
            end else if (running) begin
                cycle_count <= cycle_count + 1;
                if (wb_fire) begin
                    wb_count <= wb_count + 1;
                end
                if (imem_req) begin
                    fetch_count <= fetch_count + 1;
                end
                if (dut.issue_accept) begin
                    issue_count <= issue_count + 1;
                end
                if (dut.issue_stall_raw) begin
                    stall_raw <= stall_raw + 1;
                end
                if (dut.issue_stall_fu) begin
                    stall_fu <= stall_fu + 1;
                end
                if (dut.issue_stall_mem) begin
                    stall_mem <= stall_mem + 1;
                end
                if (dut.issue_stall_atomic) begin
                    stall_atomic <= stall_atomic + 1;
                end
                if (dut.issue_stall_tensor) begin
                    stall_tensor <= stall_tensor + 1;
                end
                if (dut.issue_stall_wbq) begin
                    stall_wbq <= stall_wbq + 1;
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
        $display("RalphGPU SM V2 Tensor Core Performance Test");
        $display("Kernel: WMMA MMA stream (m16n16k16, FP16)");
        $display("Ops: %0d MMA", N_OPS);
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

        timeout_cycles = (N_OPS * 8) + 2000;
        timeout_left = timeout_cycles;
        while (!done && (timeout_left > 0)) begin
            @(posedge clk);
            timeout_left = timeout_left - 1;
        end

        ipc = (cycle_count > 0) ? (1.0 * wb_count / cycle_count) : 0.0;
        $display("Cycles: %0d", cycle_count);
        $display("Writebacks: %0d", wb_count);
        $display("Fetches: %0d", fetch_count);
        $display("Issues: %0d", issue_count);
        $display("Stalls: raw=%0d fu=%0d mem=%0d atomic=%0d tensor=%0d wbq=%0d",
                 stall_raw, stall_fu, stall_mem, stall_atomic, stall_tensor, stall_wbq);
        $display("IPC: %0.3f", ipc);

        if (!done) begin
            $display("FAIL: timeout before completing all MMAs");
        end else if (wb_count != N_OPS) begin
            $display("FAIL: expected %0d writebacks, got %0d", N_OPS, wb_count);
        end else begin
            $display("PASS: completed WMMA stream");
        end

        $finish;
    end

endmodule
