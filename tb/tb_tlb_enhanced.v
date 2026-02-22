`timescale 1ns / 1ps

module tb_tlb_enhanced;

    localparam CLK_PERIOD   = 10;
    localparam NUM_SMS      = 1;
    localparam VADDR_WIDTH  = 48;
    localparam PADDR_WIDTH  = 40;
    localparam ASID_WIDTH   = 16;
    localparam PPN_WIDTH    = PADDR_WIDTH - 12;

    localparam MODE_MAP_RW      = 2'd0;
    localparam MODE_MAP_RO      = 2'd1;
    localparam MODE_NOT_PRESENT = 2'd2;

    localparam [3:0] FAULT_NOT_PRESENT   = 4'h1;
    localparam [3:0] FAULT_WRITE_PROTECT = 4'h2;

    reg clk;
    reg rst_n;

    reg  [NUM_SMS-1:0] req_valid;
    reg  [VADDR_WIDTH*NUM_SMS-1:0] req_vaddr;
    reg  [NUM_SMS-1:0] req_write;
    reg  [ASID_WIDTH*NUM_SMS-1:0] req_asid;
    wire [NUM_SMS-1:0] req_ready;

    wire [NUM_SMS-1:0] resp_valid;
    wire [PADDR_WIDTH*NUM_SMS-1:0] resp_paddr;
    wire [NUM_SMS-1:0] resp_fault;
    wire [NUM_SMS*4-1:0] resp_fault_code;

    wire ptw_req_valid;
    wire [PADDR_WIDTH-1:0] ptw_req_addr;
    reg ptw_req_ready;
    reg ptw_resp_valid;
    reg [63:0] ptw_resp_data;

    reg [PADDR_WIDTH-1:0] page_table_base;
    reg [ASID_WIDTH-1:0] current_asid;

    reg invalidate_all;
    reg invalidate_asid;
    reg [ASID_WIDTH-1:0] invalidate_asid_val;
    reg invalidate_page;
    reg [VADDR_WIDTH-1:0] invalidate_vaddr;

    wire [31:0] stat_l1_hits;
    wire [31:0] stat_l1_misses;
    wire [31:0] stat_l2_hits;
    wire [31:0] stat_l2_misses;
    wire [31:0] stat_page_walks;
    wire [31:0] stat_page_faults;

    integer pass_count;
    integer fail_count;
    integer ptw_req_count;

    reg [1:0] ptw_mode;
    reg [PPN_WIDTH-1:0] final_ppn;
    reg [2:0] ptw_level;

    reg seen_resp;
    reg seen_fault;
    reg [3:0] seen_fault_code;
    reg [PADDR_WIDTH-1:0] seen_paddr;

    tlb_enhanced #(
        .NUM_SMS(NUM_SMS),
        .VADDR_WIDTH(VADDR_WIDTH),
        .PADDR_WIDTH(PADDR_WIDTH),
        .L1_ENTRIES(8),
        .L1_WAYS(2),
        .L2_ENTRIES(16),
        .L2_WAYS(2),
        .ASID_WIDTH(ASID_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_vaddr(req_vaddr),
        .req_write(req_write),
        .req_asid(req_asid),
        .req_ready(req_ready),
        .resp_valid(resp_valid),
        .resp_paddr(resp_paddr),
        .resp_fault(resp_fault),
        .resp_fault_code(resp_fault_code),
        .ptw_req_valid(ptw_req_valid),
        .ptw_req_addr(ptw_req_addr),
        .ptw_req_ready(ptw_req_ready),
        .ptw_resp_valid(ptw_resp_valid),
        .ptw_resp_data(ptw_resp_data),
        .page_table_base(page_table_base),
        .current_asid(current_asid),
        .invalidate_all(invalidate_all),
        .invalidate_asid(invalidate_asid),
        .invalidate_asid_val(invalidate_asid_val),
        .invalidate_page(invalidate_page),
        .invalidate_vaddr(invalidate_vaddr),
        .stat_l1_hits(stat_l1_hits),
        .stat_l1_misses(stat_l1_misses),
        .stat_l2_hits(stat_l2_hits),
        .stat_l2_misses(stat_l2_misses),
        .stat_page_walks(stat_page_walks),
        .stat_page_faults(stat_page_faults)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always @(posedge clk) begin
        reg [63:0] pte;

        ptw_resp_valid <= 1'b0;

        if (resp_valid[0]) begin
            seen_resp <= 1'b1;
            seen_fault <= resp_fault[0];
            seen_fault_code <= resp_fault_code[3:0];
            seen_paddr <= resp_paddr[PADDR_WIDTH-1:0];
        end

        if (ptw_req_valid && ptw_req_ready) begin
            pte = 64'd0;
            ptw_req_count <= ptw_req_count + 1;

            case (ptw_mode)
                MODE_MAP_RW: begin
                    pte[0] = 1'b1;
                    pte[1] = 1'b1;
                    pte[3:0] = 4'b0011;
                    if (ptw_level < 3)
                        pte[12 + PPN_WIDTH - 1 -: PPN_WIDTH] = {PPN_WIDTH{1'b1}} - ptw_level;
                    else
                        pte[12 + PPN_WIDTH - 1 -: PPN_WIDTH] = final_ppn;
                end

                MODE_MAP_RO: begin
                    pte[0] = 1'b1;
                    pte[1] = 1'b0;
                    pte[3:0] = 4'b0001;
                    pte[12 + PPN_WIDTH - 1 -: PPN_WIDTH] = final_ppn;
                end

                default: begin
                    pte[0] = 1'b0;
                end
            endcase

            ptw_resp_valid <= 1'b1;
            ptw_resp_data <= pte;
            ptw_level <= ptw_level + 1'b1;
        end
    end

    task automatic clear_resp_capture;
        begin
            seen_resp = 1'b0;
            seen_fault = 1'b0;
            seen_fault_code = 4'd0;
            seen_paddr = {PADDR_WIDTH{1'b0}};
        end
    endtask

    task automatic check;
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

    task automatic issue_req;
        input [VADDR_WIDTH-1:0] va;
        input wr;
        input [ASID_WIDTH-1:0] asid;
        begin
            ptw_level = 0;
            @(negedge clk);
            req_vaddr = va;
            req_write = wr;
            req_asid = asid;
            req_valid = 1'b1;
            @(negedge clk);
            req_valid = 1'b0;
        end
    endtask

    task automatic wait_resp;
        input integer timeout_cycles;
        output reg got;
        integer i;
        begin
            got = 1'b0;
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (seen_resp) begin
                    got = 1'b1;
                    i = timeout_cycles;
                end
            end
        end
    endtask

    initial begin
        reg got;
        reg [VADDR_WIDTH-1:0] va0;
        reg [VADDR_WIDTH-1:0] va1;
        reg [VADDR_WIDTH-1:0] va2;
        reg [PADDR_WIDTH-1:0] exp_pa;
        integer ptw_before;

        pass_count = 0;
        fail_count = 0;
        ptw_req_count = 0;

        rst_n = 1'b0;
        req_valid = 0;
        req_vaddr = 0;
        req_write = 0;
        req_asid = 0;

        ptw_req_ready = 1'b1;
        ptw_resp_valid = 1'b0;
        ptw_resp_data = 64'd0;
        ptw_mode = MODE_MAP_RW;
        final_ppn = 28'h0123AB;
        ptw_level = 0;

        page_table_base = 40'h00000_10000;
        current_asid = 16'h0001;

        invalidate_all = 1'b0;
        invalidate_asid = 1'b0;
        invalidate_asid_val = 0;
        invalidate_page = 1'b0;
        invalidate_vaddr = 0;

        clear_resp_capture();

        va0 = 48'h0000_1234_5678;
        va1 = 48'h0000_2234_5000;
        va2 = 48'h0000_3333_4000;

        #(CLK_PERIOD * 4);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        $display("============================================");
        $display("TLB Enhanced Testbench");
        $display("============================================");

        $display("\n[TEST 1] PTW walk and translation success");
        clear_resp_capture();
        ptw_mode = MODE_MAP_RW;
        final_ppn = 28'h0ABCDE1;
        issue_req(va0, 1'b0, 16'h0001);
        wait_resp(60, got);
        exp_pa = {final_ppn, va0[11:0]};
        check(got, "response observed");
        check(seen_fault == 1'b0, "no fault on readable mapping");
        check(seen_paddr == exp_pa, "PA matches PTW final mapping");
        check(ptw_req_count >= 4, "PTW issued multi-level requests");

        $display("\n[TEST 2] L1 hit after fill");
        clear_resp_capture();
        ptw_before = ptw_req_count;
        issue_req(va0, 1'b0, 16'h0001);
        wait_resp(20, got);
        check(got, "response observed");
        check(seen_fault == 1'b0, "L1 hit returns no fault");
        check(seen_paddr == exp_pa, "L1 returns same PA");
        check(ptw_req_count == ptw_before, "no extra PTW traffic on L1 hit");

        $display("\n[TEST 3] Invalidate all");
        @(negedge clk);
        invalidate_all = 1'b1;
        @(negedge clk);
        invalidate_all = 1'b0;

        clear_resp_capture();
        ptw_before = ptw_req_count;
        issue_req(va0, 1'b0, 16'h0001);
        wait_resp(60, got);
        check(got, "response observed after invalidate_all");
        check(ptw_req_count >= ptw_before + 4, "translation re-walked after flush");

        $display("\n[TEST 4] PTW write-protect fault");
        clear_resp_capture();
        ptw_mode = MODE_MAP_RO;
        issue_req(va1, 1'b1, 16'h0001);
        wait_resp(40, got);
        check(got, "fault response observed");
        check(seen_fault == 1'b1, "write fault asserted");
        check(seen_fault_code == FAULT_WRITE_PROTECT, "fault code is WRITE_PROTECT");

        $display("\n[TEST 5] Not-present fault");
        clear_resp_capture();
        ptw_mode = MODE_NOT_PRESENT;
        issue_req(va2, 1'b0, 16'h0001);
        wait_resp(40, got);
        check(got, "fault response observed");
        check(seen_fault == 1'b1, "not-present raises fault");
        check(seen_fault_code == FAULT_NOT_PRESENT, "fault code is NOT_PRESENT");

        $display("\n============================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("============================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_tlb_enhanced failed with %0d checks", fail_count);
        end
    end

endmodule
