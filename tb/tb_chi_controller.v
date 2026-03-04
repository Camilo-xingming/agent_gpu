`timescale 1ns / 1ps

module tb_chi_controller;
    localparam NUM_CHIPLETS    = 4;
    localparam CHIPLET_ID      = 1;
    localparam ADDR_WIDTH      = 32;
    localparam DATA_WIDTH      = 128;
    localparam TXN_ID_WIDTH    = 8;
    localparam NODE_ID_WIDTH   = 4;
    localparam NUM_SN_ENTRIES  = 16;
    localparam NUM_OUTSTANDING = 4;

    localparam OP_READ_SHARED      = 5'h01;
    localparam OP_WRITE_NO_SNP     = 5'h06;
    localparam SNP_READ            = 5'h10;
    localparam SNP_MAKE_INVALID    = 5'h13;
    localparam RSP_COMP            = 5'h18;
    localparam RSP_SNP_RESP        = 5'h1B;
    localparam RSP_SNP_RESP_DATA   = 5'h1C;

    reg                         clk;
    reg                         rst_n;

    reg                         local_req_valid;
    wire                        local_req_ready;
    reg  [4:0]                  local_req_opcode;
    reg  [ADDR_WIDTH-1:0]       local_req_addr;
    reg  [TXN_ID_WIDTH-1:0]     local_req_txn_id;
    reg  [DATA_WIDTH-1:0]       local_req_data;
    reg                         local_req_excl;

    wire                        local_resp_valid;
    reg                         local_resp_ready;
    wire [4:0]                  local_resp_opcode;
    wire [TXN_ID_WIDTH-1:0]     local_resp_txn_id;
    wire [DATA_WIDTH-1:0]       local_resp_data;
    wire [1:0]                  local_resp_result;

    wire                        chi_req_valid;
    reg                         chi_req_ready;
    wire [4:0]                  chi_req_opcode;
    wire [ADDR_WIDTH-1:0]       chi_req_addr;
    wire [TXN_ID_WIDTH-1:0]     chi_req_txn_id;
    wire [NODE_ID_WIDTH-1:0]    chi_req_src_id;
    wire [NODE_ID_WIDTH-1:0]    chi_req_tgt_id;
    wire                        chi_req_excl;

    reg                         chi_resp_valid;
    wire                        chi_resp_ready;
    reg  [4:0]                  chi_resp_opcode;
    reg  [TXN_ID_WIDTH-1:0]     chi_resp_txn_id;
    reg  [NODE_ID_WIDTH-1:0]    chi_resp_src_id;
    reg  [1:0]                  chi_resp_result;

    reg                         chi_snp_valid;
    wire                        chi_snp_ready;
    reg  [4:0]                  chi_snp_opcode;
    reg  [ADDR_WIDTH-1:0]       chi_snp_addr;
    reg  [TXN_ID_WIDTH-1:0]     chi_snp_txn_id;
    reg  [NODE_ID_WIDTH-1:0]    chi_snp_src_id;

    wire                        chi_snp_resp_valid;
    reg                         chi_snp_resp_ready;
    wire [4:0]                  chi_snp_resp_opcode;
    wire [TXN_ID_WIDTH-1:0]     chi_snp_resp_txn_id;
    wire [NODE_ID_WIDTH-1:0]    chi_snp_resp_tgt_id;
    wire [DATA_WIDTH-1:0]       chi_snp_resp_data;
    wire                        chi_snp_resp_has_data;

    wire                        chi_data_valid;
    reg                         chi_data_ready;
    wire [DATA_WIDTH-1:0]       chi_data_data;
    wire [TXN_ID_WIDTH-1:0]     chi_data_txn_id;
    wire [NODE_ID_WIDTH-1:0]    chi_data_tgt_id;

    reg                         chi_rxdata_valid;
    wire                        chi_rxdata_ready;
    reg  [DATA_WIDTH-1:0]       chi_rxdata_data;
    reg  [TXN_ID_WIDTH-1:0]     chi_rxdata_txn_id;

    wire                        snp_req_valid;
    reg                         snp_req_ready;
    wire [4:0]                  snp_req_opcode;
    wire [ADDR_WIDTH-1:0]       snp_req_addr;

    reg                         snp_resp_valid;
    wire                        snp_resp_ready;
    reg  [2:0]                  snp_resp_state;
    reg  [DATA_WIDTH-1:0]       snp_resp_data;
    reg                         snp_resp_has_data;

    wire [31:0]                 stat_req_sent;
    wire [31:0]                 stat_req_rcvd;
    wire [31:0]                 stat_snp_sent;
    wire [31:0]                 stat_snp_rcvd;
    wire [31:0]                 stat_data_transfers;

    integer pass_count;
    integer fail_count;
    integer tmo;

    chi_controller #(
        .NUM_CHIPLETS(NUM_CHIPLETS),
        .CHIPLET_ID(CHIPLET_ID),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TXN_ID_WIDTH(TXN_ID_WIDTH),
        .NODE_ID_WIDTH(NODE_ID_WIDTH),
        .NUM_SN_ENTRIES(NUM_SN_ENTRIES),
        .NUM_OUTSTANDING(NUM_OUTSTANDING)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .local_req_valid(local_req_valid),
        .local_req_ready(local_req_ready),
        .local_req_opcode(local_req_opcode),
        .local_req_addr(local_req_addr),
        .local_req_txn_id(local_req_txn_id),
        .local_req_data(local_req_data),
        .local_req_excl(local_req_excl),
        .local_resp_valid(local_resp_valid),
        .local_resp_ready(local_resp_ready),
        .local_resp_opcode(local_resp_opcode),
        .local_resp_txn_id(local_resp_txn_id),
        .local_resp_data(local_resp_data),
        .local_resp_result(local_resp_result),
        .chi_req_valid(chi_req_valid),
        .chi_req_ready(chi_req_ready),
        .chi_req_opcode(chi_req_opcode),
        .chi_req_addr(chi_req_addr),
        .chi_req_txn_id(chi_req_txn_id),
        .chi_req_src_id(chi_req_src_id),
        .chi_req_tgt_id(chi_req_tgt_id),
        .chi_req_excl(chi_req_excl),
        .chi_resp_valid(chi_resp_valid),
        .chi_resp_ready(chi_resp_ready),
        .chi_resp_opcode(chi_resp_opcode),
        .chi_resp_txn_id(chi_resp_txn_id),
        .chi_resp_src_id(chi_resp_src_id),
        .chi_resp_result(chi_resp_result),
        .chi_snp_valid(chi_snp_valid),
        .chi_snp_ready(chi_snp_ready),
        .chi_snp_opcode(chi_snp_opcode),
        .chi_snp_addr(chi_snp_addr),
        .chi_snp_txn_id(chi_snp_txn_id),
        .chi_snp_src_id(chi_snp_src_id),
        .chi_snp_resp_valid(chi_snp_resp_valid),
        .chi_snp_resp_ready(chi_snp_resp_ready),
        .chi_snp_resp_opcode(chi_snp_resp_opcode),
        .chi_snp_resp_txn_id(chi_snp_resp_txn_id),
        .chi_snp_resp_tgt_id(chi_snp_resp_tgt_id),
        .chi_snp_resp_data(chi_snp_resp_data),
        .chi_snp_resp_has_data(chi_snp_resp_has_data),
        .chi_data_valid(chi_data_valid),
        .chi_data_ready(chi_data_ready),
        .chi_data_data(chi_data_data),
        .chi_data_txn_id(chi_data_txn_id),
        .chi_data_tgt_id(chi_data_tgt_id),
        .chi_rxdata_valid(chi_rxdata_valid),
        .chi_rxdata_ready(chi_rxdata_ready),
        .chi_rxdata_data(chi_rxdata_data),
        .chi_rxdata_txn_id(chi_rxdata_txn_id),
        .snp_req_valid(snp_req_valid),
        .snp_req_ready(snp_req_ready),
        .snp_req_opcode(snp_req_opcode),
        .snp_req_addr(snp_req_addr),
        .snp_resp_valid(snp_resp_valid),
        .snp_resp_ready(snp_resp_ready),
        .snp_resp_state(snp_resp_state),
        .snp_resp_data(snp_resp_data),
        .snp_resp_has_data(snp_resp_has_data),
        .stat_req_sent(stat_req_sent),
        .stat_req_rcvd(stat_req_rcvd),
        .stat_snp_sent(stat_snp_sent),
        .stat_snp_rcvd(stat_snp_rcvd),
        .stat_data_transfers(stat_data_transfers)
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

    task clear_inputs;
        begin
            local_req_valid = 1'b0;
            local_req_opcode = 5'd0;
            local_req_addr = {ADDR_WIDTH{1'b0}};
            local_req_txn_id = {TXN_ID_WIDTH{1'b0}};
            local_req_data = {DATA_WIDTH{1'b0}};
            local_req_excl = 1'b0;

            chi_resp_valid = 1'b0;
            chi_resp_opcode = 5'd0;
            chi_resp_txn_id = {TXN_ID_WIDTH{1'b0}};
            chi_resp_src_id = {NODE_ID_WIDTH{1'b0}};
            chi_resp_result = 2'b00;

            chi_snp_valid = 1'b0;
            chi_snp_opcode = 5'd0;
            chi_snp_addr = {ADDR_WIDTH{1'b0}};
            chi_snp_txn_id = {TXN_ID_WIDTH{1'b0}};
            chi_snp_src_id = {NODE_ID_WIDTH{1'b0}};

            chi_rxdata_valid = 1'b0;
            chi_rxdata_data = {DATA_WIDTH{1'b0}};
            chi_rxdata_txn_id = {TXN_ID_WIDTH{1'b0}};

            snp_resp_valid = 1'b0;
            snp_resp_state = 3'd0;
            snp_resp_data = {DATA_WIDTH{1'b0}};
            snp_resp_has_data = 1'b0;
        end
    endtask

    task do_reset;
        begin
            clear_inputs();
            chi_req_ready = 1'b1;
            local_resp_ready = 1'b1;
            chi_snp_resp_ready = 1'b1;
            chi_data_ready = 1'b1;
            snp_req_ready = 1'b1;

            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #1;
        end
    endtask

    task send_local_req;
        input [4:0] op;
        input [ADDR_WIDTH-1:0] addr;
        input [TXN_ID_WIDTH-1:0] txn;
        input [DATA_WIDTH-1:0] data;
        input excl;
        begin
            while (!local_req_ready) @(posedge clk);
            @(negedge clk);
            local_req_valid = 1'b1;
            local_req_opcode = op;
            local_req_addr = addr;
            local_req_txn_id = txn;
            local_req_data = data;
            local_req_excl = excl;
            @(posedge clk);
            #1;
            local_req_valid = 1'b0;
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;

        //============================================================
        // Test 1: Local read request -> CHI req -> RX data -> local response
        //============================================================
        do_reset();
        local_resp_ready = 1'b0;
        send_local_req(OP_READ_SHARED, 32'h0000_1200, 8'h11, {DATA_WIDTH{1'b0}}, 1'b0);

        tmo = 0;
        while (!chi_req_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(chi_req_valid, "read flow: chi_req_valid should assert");
        expect_true(chi_req_opcode == OP_READ_SHARED, "read flow: chi_req opcode mismatch");
        expect_true(chi_req_txn_id == 8'h11, "read flow: chi_req txn_id mismatch");

        // Wait request handshake to complete
        tmo = 0;
        while (chi_req_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end

        // Provide data response (hold 2 cycles for robust capture)
        @(negedge clk);
        chi_rxdata_valid = 1'b1;
        chi_rxdata_txn_id = 8'h11;
        chi_rxdata_data = 128'hA1A2_A3A4_A5A6_A7A8_B1B2_B3B4_B5B6_B7B8;
        @(posedge clk);
        #1;
        @(posedge clk);
        #1;
        chi_rxdata_valid = 1'b0;

        tmo = 0;
        while (!local_resp_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(local_resp_valid, "read flow: local_resp_valid should assert");
        expect_true(local_resp_txn_id == 8'h11, "read flow: local_resp txn_id mismatch");
        expect_true(local_resp_data == 128'hA1A2_A3A4_A5A6_A7A8_B1B2_B3B4_B5B6_B7B8,
                    "read flow: local_resp data mismatch");
        expect_true(stat_req_sent >= 1, "read flow: stat_req_sent should increment");
        expect_true(stat_data_transfers >= 1, "read flow: data transfer stat should increment");
        local_resp_ready = 1'b1;
        @(posedge clk);
        #1;

        //============================================================
        // Test 2: Write request + error response propagation
        //============================================================
        do_reset();
        local_resp_ready = 1'b0;
        send_local_req(OP_WRITE_NO_SNP, 32'h0000_2200, 8'h22, 128'h1122_3344_5566_7788_99AA_BBCC_DDEE_FF00, 1'b0);

        tmo = 0;
        while (!chi_req_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(chi_req_valid, "write flow: chi_req_valid should assert");

        // Mismatched txn response should be ignored
        @(negedge clk);
        chi_resp_valid = 1'b1;
        chi_resp_opcode = RSP_COMP;
        chi_resp_txn_id = 8'h23;
        chi_resp_result = 2'b00;
        @(posedge clk);
        #1;
        chi_resp_valid = 1'b0;

        repeat (2) @(posedge clk);
        #1;
        expect_true(!local_resp_valid, "write flow: mismatched txn response should not complete local response");

        // Matching error response
        @(negedge clk);
        chi_resp_valid = 1'b1;
        chi_resp_opcode = RSP_COMP;
        chi_resp_txn_id = 8'h22;
        chi_resp_result = 2'b11;
        @(posedge clk);
        #1;
        chi_resp_valid = 1'b0;

        tmo = 0;
        while (!local_resp_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(local_resp_valid, "write flow: local_resp_valid should assert");
        expect_true(local_resp_result == 2'b11, "write flow: error result should propagate");
        local_resp_ready = 1'b1;
        @(posedge clk);
        #1;

        //============================================================
        // Test 3: Snoop no-data path
        //============================================================
        do_reset();

        @(negedge clk);
        chi_snp_valid = 1'b1;
        chi_snp_opcode = SNP_MAKE_INVALID;
        chi_snp_addr = 32'h0000_3300;
        chi_snp_txn_id = 8'h33;
        chi_snp_src_id = 4'd2;
        @(posedge clk);
        #1;
        chi_snp_valid = 1'b0;

        tmo = 0;
        while (!snp_req_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(snp_req_valid, "snoop no-data: snp_req_valid should assert");
        expect_true(snp_req_opcode == SNP_MAKE_INVALID, "snoop no-data: snp_req opcode mismatch");

        @(negedge clk);
        snp_resp_valid = 1'b1;
        snp_resp_has_data = 1'b0;
        snp_resp_data = {DATA_WIDTH{1'b0}};
        snp_resp_state = 3'd0;
        @(posedge clk);
        #1;
        snp_resp_valid = 1'b0;

        tmo = 0;
        while (!chi_snp_resp_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(chi_snp_resp_valid, "snoop no-data: chi_snp_resp_valid should assert");
        expect_true(chi_snp_resp_opcode == RSP_SNP_RESP, "snoop no-data: opcode should be RSP_SNP_RESP");
        expect_true(chi_snp_resp_tgt_id == 4'd2, "snoop no-data: target id mismatch");
        expect_true(chi_snp_resp_has_data == 1'b0, "snoop no-data: has_data should be 0");
        expect_true(stat_snp_rcvd == 1, "snoop no-data: stat_snp_rcvd should increment");
        expect_true(stat_snp_sent == 1, "snoop no-data: stat_snp_sent should increment");

        //============================================================
        // Test 4: Snoop data path (SNP_RESP_DATA + DATA channel)
        //============================================================
        do_reset();

        @(negedge clk);
        chi_snp_valid = 1'b1;
        chi_snp_opcode = SNP_READ;
        chi_snp_addr = 32'h0000_4400;
        chi_snp_txn_id = 8'h44;
        chi_snp_src_id = 4'd3;
        @(posedge clk);
        #1;
        chi_snp_valid = 1'b0;

        tmo = 0;
        while (!snp_req_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(snp_req_valid, "snoop data: snp_req_valid should assert");

        @(negedge clk);
        snp_resp_valid = 1'b1;
        snp_resp_has_data = 1'b1;
        snp_resp_data = 128'hFACE_CAFE_DEAD_BEEF_0123_4567_89AB_CDEF;
        snp_resp_state = 3'd3;
        @(posedge clk);
        #1;
        snp_resp_valid = 1'b0;

        tmo = 0;
        while (!chi_snp_resp_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(chi_snp_resp_valid, "snoop data: chi_snp_resp_valid should assert");
        expect_true(chi_snp_resp_opcode == RSP_SNP_RESP_DATA,
                    "snoop data: opcode should be RSP_SNP_RESP_DATA");
        expect_true(chi_snp_resp_has_data == 1'b1, "snoop data: has_data should be 1");

        tmo = 0;
        while (!chi_data_valid && tmo < 20) begin
            @(posedge clk);
            #1;
            tmo = tmo + 1;
        end
        expect_true(chi_data_valid, "snoop data: chi_data_valid should assert");
        expect_true(chi_data_txn_id == 8'h44, "snoop data: chi_data txn_id mismatch");
        expect_true(chi_data_tgt_id == 4'd3, "snoop data: chi_data tgt_id mismatch");
        expect_true(chi_data_data == 128'hFACE_CAFE_DEAD_BEEF_0123_4567_89AB_CDEF,
                    "snoop data: chi_data payload mismatch");
        expect_true(stat_data_transfers >= 1, "snoop data: stat_data_transfers should increment");

        //============================================================
        // Test 5: Credit management gate (txn_full -> local_req_ready low)
        //============================================================
        do_reset();
        dut.txn_count = NUM_OUTSTANDING;
        dut.pending_req_valid = 1'b0;
        @(posedge clk);
        #1;
        expect_true(local_req_ready == 1'b0, "credit gate: local_req_ready must deassert when txn table is full");

        dut.txn_count = 0;
        @(posedge clk);
        #1;
        expect_true(local_req_ready == 1'b1, "credit gate: local_req_ready should recover after credits return");

        $display("============================================================");
        $display("tb_chi_controller Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_chi_controller failed");
        end
    end

    initial begin
        #1000000;
        $fatal(1, "tb_chi_controller timeout");
    end

    initial begin
        $dumpfile("tb_chi_controller.vcd");
        $dumpvars(0, tb_chi_controller);
    end
endmodule
