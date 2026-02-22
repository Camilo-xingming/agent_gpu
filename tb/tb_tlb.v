`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_tlb;

    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    reg              req_valid;
    reg  [47:0]      req_va;
    reg              req_write;
    wire             resp_valid;
    wire             resp_hit;
    wire [39:0]      resp_pa;
    wire             resp_fault;

    wire             l2_req_valid;
    wire [47:0]      l2_req_va;
    reg              l2_resp_valid;
    reg              l2_resp_hit;
    reg  [39:0]      l2_resp_pa;
    reg  [3:0]       l2_resp_perm;

    reg              inv_valid;
    reg  [47:0]      inv_va;
    reg              inv_all;

    integer pass_count;
    integer fail_count;
    integer test_num;
    integer l2_req_count;

    l1_tlb dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_va(req_va),
        .req_write(req_write),
        .resp_valid(resp_valid),
        .resp_hit(resp_hit),
        .resp_pa(resp_pa),
        .resp_fault(resp_fault),
        .l2_req_valid(l2_req_valid),
        .l2_req_va(l2_req_va),
        .l2_resp_valid(l2_resp_valid),
        .l2_resp_hit(l2_resp_hit),
        .l2_resp_pa(l2_resp_pa),
        .l2_resp_perm(l2_resp_perm),
        .inv_valid(inv_valid),
        .inv_va(inv_va),
        .inv_all(inv_all)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always @(posedge clk) begin
        if (l2_req_valid) begin
            l2_req_count <= l2_req_count + 1;
        end
    end

    task pulse_req;
        input [47:0] va;
        input wr;
        begin
            @(negedge clk);
            req_valid <= 1'b1;
            req_va <= va;
            req_write <= wr;
            @(negedge clk);
            req_valid <= 1'b0;
        end
    endtask

    task respond_l2;
        input hit;
        input [39:0] pa;
        input [3:0] perm;
        begin
            wait (l2_req_valid === 1'b1);
            @(negedge clk);
            l2_resp_valid <= 1'b1;
            l2_resp_hit <= hit;
            l2_resp_pa <= pa;
            l2_resp_perm <= perm;
            @(negedge clk);
            l2_resp_valid <= 1'b0;
        end
    endtask

    task check_result;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                $display("  PASS: %s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        reg [47:0] va0;
        reg [47:0] va1;
        reg [47:0] va2;
        reg [39:0] pa0;
        reg [39:0] pa1;

        va0 = 48'h0000_0001_2340;
        va1 = 48'h0000_0002_3450;
        va2 = 48'h0000_0003_4560;
        pa0 = 40'h0000A_12340;
        pa1 = 40'h0000B_34560;

        pass_count = 0;
        fail_count = 0;
        test_num = 0;
        l2_req_count = 0;

        rst_n = 1'b0;
        req_valid = 1'b0;
        req_va = 48'd0;
        req_write = 1'b0;
        l2_resp_valid = 1'b0;
        l2_resp_hit = 1'b0;
        l2_resp_pa = 40'd0;
        l2_resp_perm = 4'h0;
        inv_valid = 1'b0;
        inv_va = 48'd0;
        inv_all = 1'b0;

        #(CLK_PERIOD * 4);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        $display("====================================================");
        $display("TLB Testbench");
        $display("====================================================");

        // Test 1: Miss -> L2 fill -> success
        test_num = test_num + 1;
        $display("\n[TEST %0d] L1 miss then L2 fill", test_num);
        fork
            pulse_req(va0, 1'b0);
            respond_l2(1'b1, pa0, 4'b0011);
        join
        wait (resp_valid === 1'b1);
        #1;
        check_result(resp_hit === 1'b1, "translation hit after L2 fill");
        check_result(resp_fault === 1'b0, "read permission accepted");
        check_result(resp_pa === pa0, "physical address returned from L2");
        check_result(l2_req_count == 1, "one L2 request observed");

        // Test 2: Same VA should now hit in L1 with no new L2 request
        test_num = test_num + 1;
        $display("\n[TEST %0d] L1 hit after fill", test_num);
        pulse_req(va0, 1'b0);
        wait (resp_valid === 1'b1);
        #1;
        check_result(resp_hit === 1'b1, "L1 hit on cached translation");
        check_result(resp_pa === pa0, "L1 returns same PA");
        check_result(l2_req_count == 1, "no additional L2 request on L1 hit");

        // Test 3: Write request with read-only permissions should fault
        test_num = test_num + 1;
        $display("\n[TEST %0d] Permission fault on write", test_num);
        fork
            pulse_req(va1, 1'b1);
            respond_l2(1'b1, pa1, 4'b0001);
        join
        wait (resp_valid === 1'b1);
        #1;
        check_result(resp_hit === 1'b1, "L2 translation exists");
        check_result(resp_fault === 1'b1, "write fault asserted for read-only page");

        // Test 4: L2 miss should return page fault
        test_num = test_num + 1;
        $display("\n[TEST %0d] L2 miss page fault", test_num);
        fork
            pulse_req(va2, 1'b0);
            respond_l2(1'b0, 40'd0, 4'b0000);
        join
        wait (resp_valid === 1'b1);
        #1;
        check_result(resp_hit === 1'b0, "response marks translation miss");
        check_result(resp_fault === 1'b1, "page fault asserted on L2 miss");

        // Test 5: inv_all flushes cache and forces refetch
        test_num = test_num + 1;
        $display("\n[TEST %0d] Invalidate all", test_num);
        @(negedge clk);
        inv_valid <= 1'b1;
        inv_all <= 1'b1;
        @(negedge clk);
        inv_valid <= 1'b0;
        inv_all <= 1'b0;

        fork
            pulse_req(va0, 1'b0);
            respond_l2(1'b1, pa0, 4'b0011);
        join
        wait (resp_valid === 1'b1);
        #1;
        check_result(l2_req_count == 4, "flush caused another L2 lookup for va0");

        $display("\n====================================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("====================================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_tlb failed with %0d checks", fail_count);
        end
    end

endmodule
