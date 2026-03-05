`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_memory_interface;
    localparam ADDR_WIDTH = `GLOBAL_ADDR_WIDTH;
    localparam DATA_WIDTH = `GLOBAL_DATA_WIDTH;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam AXI_ID_W   = 4;

    reg clk;
    reg rst_n;

    reg                              req_valid;
    reg                              req_write;
    reg [NUM_LANES*ADDR_WIDTH-1:0]   req_addr;
    reg [NUM_LANES*DATA_WIDTH-1:0]   req_wdata;
    reg [NUM_LANES-1:0]              req_mask;
    wire                             req_ready;
    wire                             resp_valid;
    wire [NUM_LANES*DATA_WIDTH-1:0]  resp_rdata;

    wire [AXI_ID_W-1:0]              m_axi_awid;
    wire [ADDR_WIDTH-1:0]            m_axi_awaddr;
    wire [7:0]                       m_axi_awlen;
    wire [2:0]                       m_axi_awsize;
    wire [1:0]                       m_axi_awburst;
    wire                             m_axi_awvalid;
    reg                              m_axi_awready;

    wire [DATA_WIDTH-1:0]            m_axi_wdata;
    wire [DATA_WIDTH/8-1:0]          m_axi_wstrb;
    wire                             m_axi_wlast;
    wire                             m_axi_wvalid;
    reg                              m_axi_wready;

    reg  [AXI_ID_W-1:0]              m_axi_bid;
    reg  [1:0]                       m_axi_bresp;
    reg                              m_axi_bvalid;
    wire                             m_axi_bready;

    wire [AXI_ID_W-1:0]              m_axi_arid;
    wire [ADDR_WIDTH-1:0]            m_axi_araddr;
    wire [7:0]                       m_axi_arlen;
    wire [2:0]                       m_axi_arsize;
    wire [1:0]                       m_axi_arburst;
    wire                             m_axi_arvalid;
    reg                              m_axi_arready;

    reg  [AXI_ID_W-1:0]              m_axi_rid;
    reg  [DATA_WIDTH-1:0]            m_axi_rdata;
    reg  [1:0]                       m_axi_rresp;
    reg                              m_axi_rlast;
    reg                              m_axi_rvalid;
    wire                             m_axi_rready;

    integer pass_count;
    integer fail_count;
    integer test_num;
    integer timeout;
    integer i;

    integer ar_hs_count;
    integer r_hs_count;
    integer aw_hs_count;
    integer w_hs_count;
    integer b_hs_count;
    reg [7:0] last_arlen;

    reg [ADDR_WIDTH-1:0] wr_addr_log [0:127];
    reg [DATA_WIDTH-1:0] wr_data_log [0:127];
    integer wr_log_count;

    reg [ADDR_WIDTH-1:0] pending_awaddr;
    reg [AXI_ID_W-1:0] pending_awid;

    reg rd_active;
    reg [ADDR_WIDTH-1:0] rd_base;
    reg [7:0] rd_len;
    reg [7:0] rd_idx;
    reg [AXI_ID_W-1:0] rd_id;

    integer sim_cycles;
    integer ar0, r0, aw0, w0, b0, wr0;
    reg hit_found;

    memory_interface #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_LANES(NUM_LANES),
        .AXI_ID_W(AXI_ID_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_mask(req_mask),
        .req_ready(req_ready),
        .resp_valid(resp_valid),
        .resp_rdata(resp_rdata),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [DATA_WIDTH-1:0] make_data;
        input [ADDR_WIDTH-1:0] addr;
        begin
            make_data = 32'hA500_0000 | addr[23:0];
        end
    endfunction

    task check_result;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
                $display("PASS: %s", msg);
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: %s", msg);
            end
        end
    endtask

    task clear_req_vectors;
        begin
            req_addr = {NUM_LANES*ADDR_WIDTH{1'b0}};
            req_wdata = {NUM_LANES*DATA_WIDTH{1'b0}};
            req_mask = {NUM_LANES{1'b0}};
        end
    endtask

    task set_lane_addr;
        input integer lane;
        input [ADDR_WIDTH-1:0] addr;
        begin
            req_addr[lane*ADDR_WIDTH +: ADDR_WIDTH] = addr;
        end
    endtask

    task set_lane_wdata;
        input integer lane;
        input [DATA_WIDTH-1:0] data;
        begin
            req_wdata[lane*DATA_WIDTH +: DATA_WIDTH] = data;
        end
    endtask

    task submit_request;
        input is_write;
        begin
            while (!req_ready) @(posedge clk);
            req_write <= is_write;
            req_valid <= 1'b1;
            @(posedge clk);
            req_valid <= 1'b0;
            req_write <= 1'b0;
        end
    endtask

    task wait_resp_valid;
        input [255:0] name;
        begin
            timeout = 0;
            while (!resp_valid && timeout < 400) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            check_result(resp_valid, name);
        end
    endtask

    // Watchdog
    always @(posedge clk) begin
        if (!rst_n) sim_cycles <= 0;
        else begin
            sim_cycles <= sim_cycles + 1;
            if (sim_cycles > 20000) $fatal(1, "tb_memory_interface timeout");
        end
    end

    // AXI slave model
    always @(posedge clk) begin
        if (!rst_n) begin
            ar_hs_count <= 0; r_hs_count <= 0;
            aw_hs_count <= 0; w_hs_count <= 0; b_hs_count <= 0;
            last_arlen <= 0; wr_log_count <= 0;
            pending_awaddr <= 0; pending_awid <= 0;
            rd_active <= 1'b0; rd_base <= 0; rd_len <= 0; rd_idx <= 0; rd_id <= 0;
            m_axi_bvalid <= 1'b0; m_axi_bid <= 0; m_axi_bresp <= 2'b00;
            m_axi_rvalid <= 1'b0; m_axi_rid <= 0; m_axi_rdata <= 0; m_axi_rresp <= 2'b00; m_axi_rlast <= 1'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_awaddr <= m_axi_awaddr;
                pending_awid <= m_axi_awid;
                aw_hs_count <= aw_hs_count + 1;
            end
            if (m_axi_wvalid && m_axi_wready) begin
                if (wr_log_count < 128) begin
                    wr_addr_log[wr_log_count] <= pending_awaddr;
                    wr_data_log[wr_log_count] <= m_axi_wdata;
                    wr_log_count <= wr_log_count + 1;
                end
                w_hs_count <= w_hs_count + 1;
                m_axi_bid <= pending_awid;
                m_axi_bresp <= (pending_awaddr[31:12] == 20'h0000E) ? 2'b10 : 2'b00;
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
                b_hs_count <= b_hs_count + 1;
            end
            if (m_axi_arvalid && m_axi_arready) begin
                rd_active <= 1'b1; rd_base <= m_axi_araddr; rd_len <= m_axi_arlen; rd_idx <= 0; rd_id <= m_axi_arid;
                ar_hs_count <= ar_hs_count + 1; last_arlen <= m_axi_arlen;
            end
            if (rd_active) begin
                m_axi_rvalid <= 1'b1; m_axi_rid <= rd_id;
                m_axi_rdata <= make_data(rd_base + {22'd0, rd_idx, 2'b00});
                m_axi_rresp <= (rd_base[31:12] == 20'h0000E) ? 2'b10 : 2'b00;
                m_axi_rlast <= (rd_idx == rd_len);
                if (m_axi_rready) begin
                    r_hs_count <= r_hs_count + 1;
                    if (rd_idx == rd_len) begin
                        rd_active <= 1'b0; m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;
                    end else rd_idx <= rd_idx + 1'b1;
                end
            end else begin
                m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;
            end
        end
    end

    initial begin
        pass_count = 0; fail_count = 0; test_num = 0; sim_cycles = 0;
        req_valid = 1'b0; req_write = 1'b0; clear_req_vectors();
        m_axi_awready = 1'b1; m_axi_wready = 1'b1; m_axi_arready = 1'b1;
        rst_n = 1'b0; repeat (5) @(posedge clk); rst_n = 1'b1; repeat (3) @(posedge clk);

        // Test 1: sparse read
        test_num = test_num + 1; $display("=== TEST %0d: sparse read ===", test_num);
        clear_req_vectors(); set_lane_addr(0, 32'h0000_1000); set_lane_addr(3, 32'h0000_1018); set_lane_addr(7, 32'h0000_10A0);
        req_mask[0] = 1'b1; req_mask[3] = 1'b1; req_mask[7] = 1'b1;
        ar0 = ar_hs_count; r0 = r_hs_count; submit_request(1'b0); wait_resp_valid("sparse read resp_valid");
        check_result((ar_hs_count - ar0) >= 3, "sparse read AR count >= active lanes");
        check_result((r_hs_count - r0) == 3, "sparse read R count = active lanes");
        check_result(resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_1000), "lane0 read data");
        check_result(resp_rdata[1*DATA_WIDTH +: DATA_WIDTH] == 0, "inactive lane1 remains zero");

        // Test 2: contiguous burst read
        test_num = test_num + 1; $display("=== TEST %0d: fullwarp contiguous burst read ===", test_num);
        clear_req_vectors(); for (i = 0; i < NUM_LANES; i = i + 1) begin set_lane_addr(i, 32'h0000_2000 + (i * 4)); req_mask[i] = 1'b1; end
        ar0 = ar_hs_count; r0 = r_hs_count; submit_request(1'b0); wait_resp_valid("burst read resp_valid");
        check_result((ar_hs_count - ar0) >= 1, "burst read AR count >= 1");
        check_result(last_arlen == (NUM_LANES-1), "burst ARLEN = NUM_LANES-1");
        check_result((r_hs_count - r0) == NUM_LANES, "burst read R count = NUM_LANES");
        check_result(resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_2000), "burst lane0 data");

        // Test 3: sparse write
        test_num = test_num + 1; $display("=== TEST %0d: sparse write ===", test_num);
        clear_req_vectors(); set_lane_addr(2, 32'h0000_3008); set_lane_addr(5, 32'h0000_3014); set_lane_wdata(2, 32'hDEAD_BEEF); set_lane_wdata(5, 32'hCAFE_BABE);
        req_mask[2] = 1'b1; req_mask[5] = 1'b1; aw0 = aw_hs_count; w0 = w_hs_count; b0 = b_hs_count; wr0 = wr_log_count;
        submit_request(1'b1); wait_resp_valid("sparse write resp_valid");
        check_result((b_hs_count - b0) == 2, "sparse write B count = active lanes");
        hit_found = 1'b0; for (i = wr0; i < wr_log_count; i = i + 1) begin
            if (wr_addr_log[i] == 32'h0000_3014 && wr_data_log[i] == 32'hCAFE_BABE) hit_found = 1'b1;
        end
        check_result(hit_found, "lane5 write present in logs");

        // Test 4: strided read
        test_num = test_num + 1; $display("=== TEST %0d: strided read ===", test_num);
        clear_req_vectors(); set_lane_addr(0, 32'h0000_4000); set_lane_addr(2, 32'h0000_4008);
        req_mask[0] = 1'b1; req_mask[2] = 1'b1; ar0 = ar_hs_count; r0 = r_hs_count;
        submit_request(1'b0); wait_resp_valid("strided read resp_valid");
        check_result((r_hs_count - r0) == 2, "strided read R count = active lanes");
        check_result(resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_4000), "lane0 read data");

        // Test 5: scattered read
        test_num = test_num + 1; $display("=== TEST %0d: scattered read ===", test_num);
        clear_req_vectors(); set_lane_addr(1, 32'h0000_5004); set_lane_addr(31, 32'h0000_5F00);
        req_mask[1] = 1'b1; req_mask[31] = 1'b1; ar0 = ar_hs_count; r0 = r_hs_count;
        submit_request(1'b0); wait_resp_valid("scattered read resp_valid");
        check_result((r_hs_count - r0) == 2, "scattered read R count = active lanes");
        check_result(resp_rdata[31*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_5F00), "lane31 data");

        // Test 6: single-lane write
        test_num = test_num + 1; $display("=== TEST %0d: single-lane write ===", test_num);
        clear_req_vectors(); set_lane_addr(15, 32'h0000_6000); set_lane_wdata(15, 32'h1234_5678);
        req_mask[15] = 1'b1; aw0 = aw_hs_count; w0 = w_hs_count; b0 = b_hs_count; wr0 = wr_log_count;
        submit_request(1'b1); wait_resp_valid("single-lane write resp_valid");
        check_result((b_hs_count - b0) == 1, "single write B count = 1");
        check_result(wr_addr_log[wr_log_count-1] == 32'h0000_6000, "write addr lane15");

        // Test 7: fullwarp write
        test_num = test_num + 1; $display("=== TEST %0d: fullwarp write ===", test_num);
        clear_req_vectors(); for (i = 0; i < NUM_LANES; i = i + 1) begin
            set_lane_addr(i, 32'h0000_7000 + (i * 4)); set_lane_wdata(i, 32'h7000_0000 | i); req_mask[i] = 1'b1;
        end
        aw0 = aw_hs_count; w0 = w_hs_count; b0 = b_hs_count; wr0 = wr_log_count;
        submit_request(1'b1); wait_resp_valid("fullwarp write resp_valid");
        check_result((b_hs_count - b0) == NUM_LANES, "fullwarp write B count = NUM_LANES");
        hit_found = 1'b0; for (i = wr0; i < wr_log_count; i = i + 1) begin
            if (wr_addr_log[i] == 32'h0000_7000 && wr_data_log[i] == 32'h7000_0000) hit_found = 1'b1;
        end
        check_result(hit_found, "write lane0 present in logs");
        hit_found = 1'b0; for (i = wr0; i < wr_log_count; i = i + 1) begin
            if (wr_addr_log[i] == 32'h0000_7078) hit_found = 1'b1;
        end
        check_result(hit_found, "lane30 write present in logs");

        // Test 8: AXI backpressure stall
        test_num = test_num + 1; $display("=== TEST %0d: AXI backpressure stall ===", test_num);
        clear_req_vectors(); set_lane_addr(0, 32'h0000_8000); req_mask[0] = 1'b1;
        m_axi_arready = 1'b0; submit_request(1'b0); repeat (10) @(posedge clk);
        check_result(!resp_valid, "No response during stall");
        m_axi_arready = 1'b1; wait_resp_valid("Stall recovery resp_valid");
        check_result(resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_8000), "stall recovery data");

        // Test 9: Misaligned Base Address
        test_num = test_num + 1; $display("=== TEST %0d: misaligned base address ===", test_num);
        clear_req_vectors(); for (i = 0; i < 4; i = i + 1) begin
            set_lane_addr(i, 32'h0000_900E + (i * 4)); req_mask[i] = 1'b1;
        end
        submit_request(1'b0); wait_resp_valid("misaligned read resp_valid");
        check_result(resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] == make_data(32'h0000_900E), "lane0 misaligned data");

        // Test 10: AXI Error Response
        test_num = test_num + 1; $display("=== TEST %0d: AXI error response ===", test_num);
        clear_req_vectors(); set_lane_addr(0, 32'h0000_E000); req_mask[0] = 1'b1;
        submit_request(1'b0); wait_resp_valid("error response resp_valid");
        $display("INFO: Observed rresp for Test 10. Check if RTL propagates it.");

        $display("========================================");
        $display("tb_memory_interface RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $finish; else $fatal(1, "tb_memory_interface failed");
    end
endmodule
