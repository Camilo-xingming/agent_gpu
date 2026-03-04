`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_memory_qos;
    localparam NUM_SMS         = 4;
    localparam NUM_CHANNELS    = 2;
    localparam ADDR_WIDTH      = 32;
    localparam DATA_WIDTH      = 64;
    localparam REQ_QUEUE_DEPTH = 4;
    localparam BANDWIDTH_BITS  = 8;
    localparam SM_WIDTH        = 2;

    reg                              clk;
    reg                              rst_n;

    reg  [NUM_SMS-1:0]               sm_req_valid;
    reg  [NUM_SMS-1:0]               sm_req_write;
    reg  [ADDR_WIDTH*NUM_SMS-1:0]    sm_req_addr;
    reg  [DATA_WIDTH*NUM_SMS-1:0]    sm_req_wdata;
    reg  [2*NUM_SMS-1:0]             sm_req_priority;
    reg  [NUM_SMS-1:0]               sm_req_latency_sensitive;
    wire [NUM_SMS-1:0]               sm_req_ready;

    wire [NUM_SMS-1:0]               sm_resp_valid;
    wire [DATA_WIDTH*NUM_SMS-1:0]    sm_resp_rdata;

    wire [NUM_CHANNELS-1:0]          ch_req_valid;
    wire [NUM_CHANNELS-1:0]          ch_req_write;
    wire [ADDR_WIDTH*NUM_CHANNELS-1:0] ch_req_addr;
    wire [DATA_WIDTH*NUM_CHANNELS-1:0] ch_req_wdata;
    wire [SM_WIDTH*NUM_CHANNELS-1:0] ch_req_source;
    reg  [NUM_CHANNELS-1:0]          ch_req_ready;

    reg  [NUM_CHANNELS-1:0]          ch_resp_valid;
    reg  [DATA_WIDTH*NUM_CHANNELS-1:0] ch_resp_rdata;
    reg  [SM_WIDTH*NUM_CHANNELS-1:0] ch_resp_source;

    reg  [BANDWIDTH_BITS*NUM_SMS-1:0] cfg_bandwidth_limit;
    reg  [7:0]                       cfg_fairness_window;
    reg                              cfg_throttle_enable;
    reg  [7:0]                       cfg_throttle_level;

    wire [31:0]                      stat_total_requests;
    wire [31:0]                      stat_throttled_requests;
    wire [31:0]                      stat_priority_inversions;
    wire [BANDWIDTH_BITS*NUM_SMS-1:0] stat_sm_bandwidth;

    integer pass_count;
    integer fail_count;
    integer i;

    memory_qos #(
        .NUM_SMS(NUM_SMS),
        .NUM_CHANNELS(NUM_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .REQ_QUEUE_DEPTH(REQ_QUEUE_DEPTH),
        .BANDWIDTH_BITS(BANDWIDTH_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sm_req_valid(sm_req_valid),
        .sm_req_write(sm_req_write),
        .sm_req_addr(sm_req_addr),
        .sm_req_wdata(sm_req_wdata),
        .sm_req_priority(sm_req_priority),
        .sm_req_latency_sensitive(sm_req_latency_sensitive),
        .sm_req_ready(sm_req_ready),
        .sm_resp_valid(sm_resp_valid),
        .sm_resp_rdata(sm_resp_rdata),
        .ch_req_valid(ch_req_valid),
        .ch_req_write(ch_req_write),
        .ch_req_addr(ch_req_addr),
        .ch_req_wdata(ch_req_wdata),
        .ch_req_source(ch_req_source),
        .ch_req_ready(ch_req_ready),
        .ch_resp_valid(ch_resp_valid),
        .ch_resp_rdata(ch_resp_rdata),
        .ch_resp_source(ch_resp_source),
        .cfg_bandwidth_limit(cfg_bandwidth_limit),
        .cfg_fairness_window(cfg_fairness_window),
        .cfg_throttle_enable(cfg_throttle_enable),
        .cfg_throttle_level(cfg_throttle_level),
        .stat_total_requests(stat_total_requests),
        .stat_throttled_requests(stat_throttled_requests),
        .stat_priority_inversions(stat_priority_inversions),
        .stat_sm_bandwidth(stat_sm_bandwidth)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task expect_true;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %s", msg);
            end
        end
    endtask

    task clear_requests;
        begin
            sm_req_valid = {NUM_SMS{1'b0}};
            sm_req_write = {NUM_SMS{1'b0}};
            sm_req_addr = {ADDR_WIDTH*NUM_SMS{1'b0}};
            sm_req_wdata = {DATA_WIDTH*NUM_SMS{1'b0}};
            sm_req_priority = {2*NUM_SMS{1'b0}};
            sm_req_latency_sensitive = {NUM_SMS{1'b0}};
        end
    endtask

    task clear_responses;
        begin
            ch_resp_valid = {NUM_CHANNELS{1'b0}};
            ch_resp_rdata = {DATA_WIDTH*NUM_CHANNELS{1'b0}};
            ch_resp_source = {SM_WIDTH*NUM_CHANNELS{1'b0}};
        end
    endtask

    task set_default_cfg;
        integer smi;
        begin
            cfg_fairness_window = 8'd32;
            cfg_throttle_enable = 1'b0;
            cfg_throttle_level = 8'd0;
            for (smi = 0; smi < NUM_SMS; smi = smi + 1) begin
                cfg_bandwidth_limit[smi*BANDWIDTH_BITS +: BANDWIDTH_BITS] = 8'd16;
            end
        end
    endtask

    task apply_req;
        input integer sm_id;
        input wr;
        input [1:0] prio;
        input lat_sens;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            sm_req_valid[sm_id] = 1'b1;
            sm_req_write[sm_id] = wr;
            sm_req_addr[sm_id*ADDR_WIDTH +: ADDR_WIDTH] = addr;
            sm_req_wdata[sm_id*DATA_WIDTH +: DATA_WIDTH] = data;
            sm_req_priority[sm_id*2 +: 2] = prio;
            sm_req_latency_sensitive[sm_id] = lat_sens;
        end
    endtask

    task pulse_requests;
        begin
            @(posedge clk);
            #1;
            clear_requests();
        end
    endtask

    task wait_issue;
        output integer out_src;
        output integer out_ch;
        output [ADDR_WIDTH-1:0] out_addr;
        integer t;
        integer ch;
        begin
            out_src = -1;
            out_ch = -1;
            out_addr = {ADDR_WIDTH{1'b0}};

            for (t = 0; t < 40; t = t + 1) begin
                @(posedge clk);
                #1;
                for (ch = 0; ch < NUM_CHANNELS; ch = ch + 1) begin
                    if (ch_req_valid[ch] && out_src == -1) begin
                        out_src = ch_req_source[ch*SM_WIDTH +: SM_WIDTH];
                        out_ch = ch;
                        out_addr = ch_req_addr[ch*ADDR_WIDTH +: ADDR_WIDTH];
                    end
                end
                if (out_src != -1)
                    t = 40;
            end
        end
    endtask

    task do_reset;
        begin
            clear_requests();
            clear_responses();
            ch_req_ready = {NUM_CHANNELS{1'b1}};
            set_default_cfg();

            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #1;
        end
    endtask

    integer src0;
    integer src1;
    integer src2;
    integer src3;
    integer ch;
    reg [ADDR_WIDTH-1:0] seen_addr;
    reg [NUM_SMS-1:0] seen_sms;

    initial begin
        pass_count = 0;
        fail_count = 0;

        //============================================================
        // Test 1: Priority arbitration (higher priority first)
        //============================================================
        do_reset();

        apply_req(0, 1'b0, 2'b01, 1'b0, 32'h0000_0000, 64'h0000_0000_0000_00A0);
        apply_req(1, 1'b0, 2'b11, 1'b0, 32'h0000_0040, 64'h0000_0000_0000_00B1);
        pulse_requests();

        wait_issue(src0, ch, seen_addr);
        expect_true(src0 == 1, "priority: first issue should be SM1 (higher prio)");

        wait_issue(src1, ch, seen_addr);
        expect_true(src1 == 0, "priority: second issue should be SM0");
        expect_true(stat_total_requests >= 1, "priority: total request counter should increment");

        //============================================================
        // Test 2: Latency-sensitive override
        //============================================================
        do_reset();

        apply_req(2, 1'b0, 2'b00, 1'b1, 32'h0000_0080, 64'h0000_0000_0000_00C2);
        apply_req(3, 1'b0, 2'b11, 1'b0, 32'h0000_00C0, 64'h0000_0000_0000_00D3);
        pulse_requests();

        wait_issue(src0, ch, seen_addr);
        expect_true(src0 == 2, "latency-sensitive request should preempt higher-priority normal request");
        wait_issue(src1, ch, seen_addr);
        expect_true(src1 == 3, "remaining request should issue after latency-sensitive");

        //============================================================
        // Test 3: Bandwidth throttling and QoS transition (limit change)
        //============================================================
        do_reset();
        cfg_bandwidth_limit[0*BANDWIDTH_BITS +: BANDWIDTH_BITS] = 8'd1;
        #1;
        expect_true(sm_req_ready[0] == 1'b1, "bw limit=1 should allow first request");

        apply_req(0, 1'b0, 2'b10, 1'b0, 32'h0000_0100, 64'h1111_0000_0000_0001);
        pulse_requests();
        wait_issue(src0, ch, seen_addr);
        expect_true(src0 == 0, "bw test: first request issued");
        #1;
        expect_true(sm_req_ready[0] == 1'b0, "bw limit reached should block next request");

        cfg_bandwidth_limit[0*BANDWIDTH_BITS +: BANDWIDTH_BITS] = 8'd2;
        #1;
        expect_true(sm_req_ready[0] == 1'b1, "raising bw limit should unblock request");

        apply_req(0, 1'b0, 2'b10, 1'b0, 32'h0000_0140, 64'h2222_0000_0000_0002);
        pulse_requests();
        wait_issue(src1, ch, seen_addr);
        expect_true(src1 == 0, "bw test: second request issued after limit transition");
        expect_true(stat_sm_bandwidth[0*BANDWIDTH_BITS +: BANDWIDTH_BITS] == 8'd2,
                    "bw stat should track two issued requests for SM0");

        //============================================================
        // Test 4: Global throttle gate and release transition
        //============================================================
        do_reset();
        cfg_throttle_enable = 1'b1;
        cfg_throttle_level = 8'hFF;
        repeat (2) @(posedge clk);
        #1;
        expect_true(sm_req_ready == {NUM_SMS{1'b0}}, "throttle active should block all SM request acceptance");

        // Attempt blocked requests for 3 cycles
        repeat (3) begin
            apply_req(1, 1'b0, 2'b01, 1'b0, 32'h0000_0200, 64'hAAAA_BBBB_CCCC_DDDD);
            @(posedge clk);
            #1;
            clear_requests();
        end

        expect_true(stat_throttled_requests >= 3,
                    "throttle counter should increase while requests are blocked by throttle");

        cfg_throttle_enable = 1'b0;
        #1;
        expect_true(sm_req_ready[1] == 1'b1, "disabling throttle should restore request readiness");

        apply_req(1, 1'b0, 2'b01, 1'b0, 32'h0000_0240, 64'h1111_2222_3333_4444);
        pulse_requests();
        wait_issue(src0, ch, seen_addr);
        expect_true(src0 == 1, "request should issue after throttle is disabled");

        //============================================================
        // Test 5: Multi-requestor fairness / starvation prevention
        //============================================================
        do_reset();
        apply_req(0, 1'b0, 2'b10, 1'b0, 32'h0000_0300, 64'h10);
        apply_req(1, 1'b0, 2'b10, 1'b0, 32'h0000_0340, 64'h11);
        apply_req(2, 1'b0, 2'b10, 1'b0, 32'h0000_0380, 64'h12);
        apply_req(3, 1'b0, 2'b10, 1'b0, 32'h0000_03C0, 64'h13);
        pulse_requests();

        seen_sms = 0;
        wait_issue(src0, ch, seen_addr); if (src0 >= 0) seen_sms[src0] = 1'b1;
        wait_issue(src1, ch, seen_addr); if (src1 >= 0) seen_sms[src1] = 1'b1;
        wait_issue(src2, ch, seen_addr); if (src2 >= 0) seen_sms[src2] = 1'b1;
        wait_issue(src3, ch, seen_addr); if (src3 >= 0) seen_sms[src3] = 1'b1;

        expect_true(seen_sms == 4'b1111,
                    "fairness: all four SMs should be serviced (no starvation)");

        //============================================================
        // Test 6: Response routing to source SM
        //============================================================
        do_reset();
        ch_resp_source = {SM_WIDTH*NUM_CHANNELS{1'b0}};
        ch_resp_rdata = {DATA_WIDTH*NUM_CHANNELS{1'b0}};
        ch_resp_valid = {NUM_CHANNELS{1'b0}};

        ch_resp_valid[1] = 1'b1;
        ch_resp_source[1*SM_WIDTH +: SM_WIDTH] = 2'd3;
        ch_resp_rdata[1*DATA_WIDTH +: DATA_WIDTH] = 64'hDEAD_BEEF_0000_3791;

        @(posedge clk);
        #1;
        clear_responses();
        expect_true(sm_resp_valid[3] == 1'b1, "response routing: sm3 valid should pulse");
        expect_true(sm_resp_rdata[3*DATA_WIDTH +: DATA_WIDTH] == 64'hDEAD_BEEF_0000_3791,
                    "response routing: sm3 data should match channel payload");

        $display("============================================================");
        $display("tb_memory_qos Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_memory_qos failed");
        end
    end

    initial begin
        #800000;
        $fatal(1, "tb_memory_qos timeout");
    end

    initial begin
        $dumpfile("tb_memory_qos.vcd");
        $dumpvars(0, tb_memory_qos);
    end
endmodule
