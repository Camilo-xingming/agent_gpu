`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_l2_interconnect;
    localparam NUM_SM          = 4;
    localparam NUM_L2_SLICES   = 2;
    localparam ADDR_WIDTH      = 16;
    localparam DATA_WIDTH      = 64;
    localparam ID_WIDTH        = 4;
    localparam MAX_OUTSTANDING = 8;
    localparam SM_W            = 2;

    reg                              clk;
    reg                              rst_n;

    reg  [NUM_SM-1:0]                sm_req_valid;
    reg  [NUM_SM-1:0]                sm_req_write;
    reg  [NUM_SM*ADDR_WIDTH-1:0]     sm_req_addr;
    reg  [NUM_SM*DATA_WIDTH-1:0]     sm_req_wdata;
    reg  [NUM_SM*ID_WIDTH-1:0]       sm_req_id;
    wire [NUM_SM-1:0]                sm_req_ready;
    wire [NUM_SM-1:0]                sm_resp_valid;
    wire [NUM_SM*DATA_WIDTH-1:0]     sm_resp_rdata;
    wire [NUM_SM*ID_WIDTH-1:0]       sm_resp_id;

    wire [NUM_L2_SLICES-1:0]         l2_req_valid;
    wire [NUM_L2_SLICES-1:0]         l2_req_write;
    wire [NUM_L2_SLICES*ADDR_WIDTH-1:0] l2_req_addr;
    wire [NUM_L2_SLICES*DATA_WIDTH-1:0] l2_req_wdata;
    wire [NUM_L2_SLICES*ID_WIDTH-1:0]   l2_req_id;
    wire [NUM_L2_SLICES*SM_W-1:0]       l2_req_sm_id;
    reg  [NUM_L2_SLICES-1:0]         l2_req_ready;

    reg  [NUM_L2_SLICES-1:0]         l2_resp_valid;
    reg  [NUM_L2_SLICES*DATA_WIDTH-1:0] l2_resp_rdata;
    reg  [NUM_L2_SLICES*ID_WIDTH-1:0]   l2_resp_id;
    reg  [NUM_L2_SLICES*SM_W-1:0]       l2_resp_sm_id;

    wire [31:0]                      stat_total_requests;
    wire [31:0]                      stat_xbar_conflicts;
    wire [31:0]                      stat_avg_latency;

    integer pass_count;
    integer fail_count;
    integer cyc;
    integer grant_sm;
    integer smi;

    l2_interconnect #(
        .NUM_SM(NUM_SM),
        .NUM_L2_SLICES(NUM_L2_SLICES),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ID_WIDTH(ID_WIDTH),
        .MAX_OUTSTANDING(MAX_OUTSTANDING)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sm_req_valid(sm_req_valid),
        .sm_req_write(sm_req_write),
        .sm_req_addr(sm_req_addr),
        .sm_req_wdata(sm_req_wdata),
        .sm_req_id(sm_req_id),
        .sm_req_ready(sm_req_ready),
        .sm_resp_valid(sm_resp_valid),
        .sm_resp_rdata(sm_resp_rdata),
        .sm_resp_id(sm_resp_id),
        .l2_req_valid(l2_req_valid),
        .l2_req_write(l2_req_write),
        .l2_req_addr(l2_req_addr),
        .l2_req_wdata(l2_req_wdata),
        .l2_req_id(l2_req_id),
        .l2_req_sm_id(l2_req_sm_id),
        .l2_req_ready(l2_req_ready),
        .l2_resp_valid(l2_resp_valid),
        .l2_resp_rdata(l2_resp_rdata),
        .l2_resp_id(l2_resp_id),
        .l2_resp_sm_id(l2_resp_sm_id),
        .stat_total_requests(stat_total_requests),
        .stat_xbar_conflicts(stat_xbar_conflicts),
        .stat_avg_latency(stat_avg_latency)
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

    task clear_reqs;
        begin
            sm_req_valid = {NUM_SM{1'b0}};
            sm_req_write = {NUM_SM{1'b0}};
            sm_req_addr = {NUM_SM*ADDR_WIDTH{1'b0}};
            sm_req_wdata = {NUM_SM*DATA_WIDTH{1'b0}};
            sm_req_id = {NUM_SM*ID_WIDTH{1'b0}};
        end
    endtask

    task clear_resps;
        begin
            l2_resp_valid = {NUM_L2_SLICES{1'b0}};
            l2_resp_rdata = {NUM_L2_SLICES*DATA_WIDTH{1'b0}};
            l2_resp_id = {NUM_L2_SLICES*ID_WIDTH{1'b0}};
            l2_resp_sm_id = {NUM_L2_SLICES*SM_W{1'b0}};
        end
    endtask

    task set_req;
        input integer sm;
        input wr;
        input [ADDR_WIDTH-1:0] addr;
        input [ID_WIDTH-1:0] rid;
        input [DATA_WIDTH-1:0] wdata;
        begin
            sm_req_valid[sm] = 1'b1;
            sm_req_write[sm] = wr;
            sm_req_addr[sm*ADDR_WIDTH +: ADDR_WIDTH] = addr;
            sm_req_id[sm*ID_WIDTH +: ID_WIDTH] = rid;
            sm_req_wdata[sm*DATA_WIDTH +: DATA_WIDTH] = wdata;
        end
    endtask

    task do_reset;
        begin
            clear_reqs();
            clear_resps();
            l2_req_ready = {NUM_L2_SLICES{1'b1}};

            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #1;
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;

        //============================================================
        // Test 1: Multi-bank routing correctness
        // slice = addr[6] xor addr[12] for NUM_L2_SLICES=2
        //============================================================
        do_reset();

        // SM0 -> slice0 (0x0000: b6=0, b12=0)
        clear_reqs();
        set_req(0, 1'b0, 16'h0000, 4'h1, 64'h1111);
        #1;
        expect_true(l2_req_valid[0] == 1'b1, "routing: SM0 addr 0x0000 should go to slice0");
        expect_true(l2_req_sm_id[0*SM_W +: SM_W] == 2'd0, "routing: slice0 should tag SM0");
        expect_true(sm_req_ready[0] == 1'b1, "routing: SM0 ready should assert when granted");
        @(posedge clk);
        #1;

        // SM1 -> slice1 (0x0040: b6=1, b12=0)
        clear_reqs();
        set_req(1, 1'b0, 16'h0040, 4'h2, 64'h2222);
        #1;
        expect_true(l2_req_valid[1] == 1'b1, "routing: SM1 addr 0x0040 should go to slice1");
        expect_true(l2_req_sm_id[1*SM_W +: SM_W] == 2'd1, "routing: slice1 should tag SM1");
        @(posedge clk);
        #1;

        //============================================================
        // Test 2: 2-way arbitration fairness (same slice, same line)
        // Expect RR grants: SM0, SM1, SM0, SM1
        //============================================================
        do_reset();
        for (cyc = 0; cyc < 4; cyc = cyc + 1) begin
            clear_reqs();
            set_req(0, 1'b0, 16'h0000, 4'h3, 64'hA0 + cyc);
            set_req(1, 1'b0, 16'h0000, 4'h4, 64'hB0 + cyc);

            #1;
            expect_true(l2_req_valid[0] == 1'b1, "fairness-2way: slice0 should grant every contended cycle");
            grant_sm = l2_req_sm_id[0*SM_W +: SM_W];
            if (cyc[0] == 1'b0)
                expect_true(grant_sm == 0, "fairness-2way: even cycle should grant SM0");
            else
                expect_true(grant_sm == 1, "fairness-2way: odd cycle should grant SM1");

            @(posedge clk);
            #1;
        end
        clear_reqs();

        //============================================================
        // Test 3: 4-way same-line contention fairness
        // All SMs target the same line/slice. Expect rotating grants 0,1,2,3...
        //============================================================
        do_reset();
        for (cyc = 0; cyc < 8; cyc = cyc + 1) begin
            clear_reqs();
            for (smi = 0; smi < NUM_SM; smi = smi + 1) begin
                set_req(smi, 1'b0, 16'h0000, smi[ID_WIDTH-1:0], 64'h1000 + smi + cyc);
            end

            #1;
            expect_true(l2_req_valid[0] == 1'b1, "fairness-4way: slice0 must issue one grant");
            grant_sm = l2_req_sm_id[0*SM_W +: SM_W];
            expect_true(grant_sm == (cyc % NUM_SM), "fairness-4way: round-robin order mismatch");

            for (smi = 0; smi < NUM_SM; smi = smi + 1) begin
                if (smi == grant_sm)
                    expect_true(sm_req_ready[smi] == 1'b1, "fairness-4way: granted SM ready should assert");
                else
                    expect_true(sm_req_ready[smi] == 1'b0, "fairness-4way: non-granted SM ready should deassert");
            end

            @(posedge clk);
            #1;
        end
        clear_reqs();

        //============================================================
        // Test 4: Parallel bandwidth across slices under load
        // Two independent slices should issue in parallel each cycle.
        //============================================================
        do_reset();
        for (cyc = 0; cyc < 5; cyc = cyc + 1) begin
            clear_reqs();
            set_req(0, 1'b0, 16'h0000, 4'h5, 64'hAAAA + cyc); // slice0
            set_req(2, 1'b0, 16'h0040, 4'h6, 64'hBBBB + cyc); // slice1
            #1;
            expect_true(l2_req_valid == 2'b11, "bandwidth: two slices should issue concurrently");
            expect_true(sm_req_ready[0] && sm_req_ready[2], "bandwidth: both requesting SMs should be ready");
            @(posedge clk);
            #1;
        end
        clear_reqs();

        //============================================================
        // Test 5: Response routing back to source SM
        //============================================================
        do_reset();
        clear_resps();
        l2_resp_valid[0] = 1'b1;
        l2_resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] = 64'hDEAD_BEEF_0000_0003;
        l2_resp_id[0*ID_WIDTH +: ID_WIDTH] = 4'hA;
        l2_resp_sm_id[0*SM_W +: SM_W] = 2'd3;

        l2_resp_valid[1] = 1'b1;
        l2_resp_rdata[1*DATA_WIDTH +: DATA_WIDTH] = 64'hCAFE_BABE_0000_0001;
        l2_resp_id[1*ID_WIDTH +: ID_WIDTH] = 4'hB;
        l2_resp_sm_id[1*SM_W +: SM_W] = 2'd1;

        #1;
        expect_true(sm_resp_valid[3] == 1'b1, "resp route: SM3 valid should assert");
        expect_true(sm_resp_rdata[3*DATA_WIDTH +: DATA_WIDTH] == 64'hDEAD_BEEF_0000_0003,
                    "resp route: SM3 data mismatch");
        expect_true(sm_resp_id[3*ID_WIDTH +: ID_WIDTH] == 4'hA,
                    "resp route: SM3 id mismatch");

        expect_true(sm_resp_valid[1] == 1'b1, "resp route: SM1 valid should assert");
        expect_true(sm_resp_rdata[1*DATA_WIDTH +: DATA_WIDTH] == 64'hCAFE_BABE_0000_0001,
                    "resp route: SM1 data mismatch");
        expect_true(sm_resp_id[1*ID_WIDTH +: ID_WIDTH] == 4'hB,
                    "resp route: SM1 id mismatch");
        expect_true(sm_resp_valid[0] == 1'b0 && sm_resp_valid[2] == 1'b0,
                    "resp route: non-target SMs should stay invalid");

        //============================================================
        // Test 6: Counter accuracy under single-slice contention
        // 3 cycles of 2-way contention -> 3 accepted requests, 3 conflicts.
        //============================================================
        do_reset();
        for (cyc = 0; cyc < 3; cyc = cyc + 1) begin
            clear_reqs();
            set_req(0, 1'b0, 16'h0000, 4'h7, 64'h100 + cyc);
            set_req(1, 1'b0, 16'h0000, 4'h8, 64'h200 + cyc);
            #1;
            @(posedge clk);
            #1;
        end
        clear_reqs();
        @(posedge clk);
        #1;
        expect_true(stat_total_requests == 32'd3, "stats: expected exactly 3 total requests");
        expect_true(stat_xbar_conflicts == 32'd3, "stats: expected exactly 3 conflicts");

        //============================================================
        // Test 7: Counter accuracy under dual-slice parallel issue
        // 4 cycles, 2 accepted requests per cycle -> 8 total requests, 0 conflict.
        //============================================================
        do_reset();
        for (cyc = 0; cyc < 4; cyc = cyc + 1) begin
            clear_reqs();
            set_req(0, 1'b0, 16'h0000, 4'h9, 64'h300 + cyc); // slice0
            set_req(1, 1'b0, 16'h0040, 4'hA, 64'h400 + cyc); // slice1
            #1;
            expect_true(l2_req_valid == 2'b11, "stats-parallel: expected dual-slice issue");
            @(posedge clk);
            #1;
        end
        clear_reqs();
        @(posedge clk);
        #1;
        expect_true(stat_total_requests == 32'd8, "stats-parallel: expected exactly 8 total requests");
        expect_true(stat_xbar_conflicts == 32'd0, "stats-parallel: expected 0 conflicts");

        $display("============================================================");
        $display("tb_l2_interconnect Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_l2_interconnect failed");
        end
    end

    initial begin
        #500000;
        $fatal(1, "tb_l2_interconnect timeout");
    end

    initial begin
        $dumpfile("tb_l2_interconnect.vcd");
        $dumpvars(0, tb_l2_interconnect);
    end
endmodule
