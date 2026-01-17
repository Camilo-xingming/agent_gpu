//============================================================================
// Debug test to analyze SM V2 fetch stalls
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_sm_v2_debug;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 8192;
    localparam N_OPS      = 100;  // Small number to debug
    localparam REG_STRIDE = 8;

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

    reg [31:0] imem [0:IMEM_WORDS-1];
    reg        imem_req_q;
    reg [31:0] imem_addr_q;

    function [31:0] encode_fma;
        input [4:0] rd, ra, rb, rc;
        begin
            encode_fma = {`OP_FP32_ARITH, rd, ra, rb, rc, `FP_FMA};
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
            imem[i] = encode_fma((i % REG_STRIDE) + 1, 5'd0, 5'd0, 5'd0);
        end
        imem[N_OPS] = encode_exit();
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

    initial begin
        l1d_resp_valid = 1'b0;
        l1d_resp_hit = 1'b0;
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            l1d_resp_rdata[i] = 32'b0;
        end
        m_axi_awready = 1'b1;
        m_axi_wready  = 1'b1;
        m_axi_bvalid  = 1'b0;
        m_axi_arready = 1'b1;
        m_axi_rvalid  = 1'b0;
        m_axi_rlast   = 1'b1;
    end

    integer cycle_count;
    integer wb_count;
    integer fetch_count;
    integer issue_count;
    integer buf_valid_count;
    integer fetch_valid_arb_count;
    integer fetch_blocked_count;
    integer timeout_cycles;
    reg running;
    wire wb_fire = dut.wb_valid && (dut.wb_rd != 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            cycle_count <= 0;
            wb_count <= 0;
            fetch_count <= 0;
            issue_count <= 0;
            buf_valid_count <= 0;
            fetch_valid_arb_count <= 0;
            fetch_blocked_count <= 0;
        end else if (kernel_start) begin
            running <= 1'b1;
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
            // Debug counters
            if (dut.fetch_valid_arb) begin
                fetch_valid_arb_count <= fetch_valid_arb_count + 1;
            end
            if (dut.fetch_blocked_by_fill) begin
                fetch_blocked_count <= fetch_blocked_count + 1;
            end
            if (dut.warp_inst_buf_valid != 0) begin
                buf_valid_count <= buf_valid_count + 1;
            end
            
            // Print debug info every 50 cycles
            if (cycle_count % 50 == 0 && cycle_count > 0) begin
                $display("Cycle %0d: wb=%0d fetch=%0d issue=%0d buf_valid=%0d arb=%0d blocked=%0d",
                         cycle_count, wb_count, fetch_count, issue_count,
                         buf_valid_count, fetch_valid_arb_count, fetch_blocked_count);
                $display("  warp_valid=%b warp_inst_buf_valid=%b fetch_inflight_valid=%0d",
                         dut.warp_valid, dut.warp_inst_buf_valid, dut.fetch_inflight_valid);
            end
            
            if (wb_count >= N_OPS) begin
                running <= 1'b0;
            end
            if (cycle_count >= 5000) begin
                running <= 1'b0;
            end
        end
    end

    initial begin
        rst_n = 0;
        kernel_start = 0;
        kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 1; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        timeout_cycles = 5000;

        #100;
        rst_n = 1;
        #100;

        $display("============================================================");
        $display("SM V2 Debug Test");
        $display("============================================================");

        kernel_start = 1;
        #CLK_PERIOD;
        kernel_start = 0;

        wait(running == 0);
        #100;

        $display("============================================================");
        $display("Final: cycles=%0d wb=%0d fetch=%0d issue=%0d", 
                 cycle_count, wb_count, fetch_count, issue_count);
        $display("  buf_valid_cycles=%0d fetch_arb_cycles=%0d fetch_blocked_cycles=%0d",
                 buf_valid_count, fetch_valid_arb_count, fetch_blocked_count);
        $display("============================================================");

        $finish;
    end

endmodule
