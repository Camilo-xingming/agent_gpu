`timescale 1ns / 1ps

module tb_forwarding_unit;

    localparam THREADS = 4;
    localparam DATA_WIDTH = 32;
    localparam REG_ADDR_WIDTH = 5;

    reg clk;
    reg rst_n;

    reg [REG_ADDR_WIDTH-1:0] id_ra;
    reg [REG_ADDR_WIDTH-1:0] id_rb;
    reg [REG_ADDR_WIDTH-1:0] id_rc;
    reg                      id_use_ra;
    reg                      id_use_rb;
    reg                      id_use_rc;

    reg [REG_ADDR_WIDTH-1:0] ex_rd;
    reg                      ex_reg_write;
    reg [THREADS*DATA_WIDTH-1:0] ex_result;

    reg [REG_ADDR_WIDTH-1:0] mem_rd;
    reg                      mem_reg_write;
    reg [THREADS*DATA_WIDTH-1:0] mem_result;

    reg [REG_ADDR_WIDTH-1:0] wb_rd;
    reg                      wb_reg_write;
    reg [THREADS*DATA_WIDTH-1:0] wb_result;

    wire [1:0] forward_a;
    wire [1:0] forward_b;
    wire [1:0] forward_c;

    wire [THREADS*DATA_WIDTH-1:0] forwarded_a;
    wire [THREADS*DATA_WIDTH-1:0] forwarded_b;
    wire [THREADS*DATA_WIDTH-1:0] forwarded_c;

    wire load_use_hazard;
    reg  ex_is_load;

    integer i;
    integer pass_count;
    integer fail_count;
    integer test_num;

    forwarding_unit #(
        .THREADS(THREADS),
        .DATA_WIDTH(DATA_WIDTH),
        .REG_ADDR_WIDTH(REG_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .id_ra(id_ra),
        .id_rb(id_rb),
        .id_rc(id_rc),
        .id_use_ra(id_use_ra),
        .id_use_rb(id_use_rb),
        .id_use_rc(id_use_rc),
        .ex_rd(ex_rd),
        .ex_reg_write(ex_reg_write),
        .ex_result(ex_result),
        .mem_rd(mem_rd),
        .mem_reg_write(mem_reg_write),
        .mem_result(mem_result),
        .wb_rd(wb_rd),
        .wb_reg_write(wb_reg_write),
        .wb_result(wb_result),
        .forward_a(forward_a),
        .forward_b(forward_b),
        .forward_c(forward_c),
        .forwarded_a(forwarded_a),
        .forwarded_b(forwarded_b),
        .forwarded_c(forwarded_c),
        .load_use_hazard(load_use_hazard),
        .ex_is_load(ex_is_load)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

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

    task clear_inputs;
        begin
            id_ra = 0;
            id_rb = 0;
            id_rc = 0;
            id_use_ra = 0;
            id_use_rb = 0;
            id_use_rc = 0;
            ex_rd = 0;
            ex_reg_write = 0;
            mem_rd = 0;
            mem_reg_write = 0;
            wb_rd = 0;
            wb_reg_write = 0;
            ex_is_load = 0;
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        rst_n = 1'b0;
        ex_result = 0;
        mem_result = 0;
        wb_result = 0;

        for (i = 0; i < THREADS; i = i + 1) begin
            ex_result[i*DATA_WIDTH +: DATA_WIDTH] = 32'hE000_0000 + i;
            mem_result[i*DATA_WIDTH +: DATA_WIDTH] = 32'hD000_0000 + i;
            wb_result[i*DATA_WIDTH +: DATA_WIDTH] = 32'hB000_0000 + i;
        end

        clear_inputs();
        #20;
        rst_n = 1'b1;
        #10;

        $display("====================================================");
        $display("Forwarding Unit Testbench");
        $display("====================================================");

        // Test 1: No forwarding
        test_num = test_num + 1;
        $display("\n[TEST %0d] No forwarding when operands unused", test_num);
        clear_inputs();
        #1;
        check_result(forward_a == 2'b00 && forward_b == 2'b00 && forward_c == 2'b00,
               "all forwarding selects are 00");
        check_result(load_use_hazard == 1'b0, "no load-use hazard");

        // Test 2: EX has highest priority
        test_num = test_num + 1;
        $display("\n[TEST %0d] EX forwarding priority", test_num);
        clear_inputs();
        id_ra = 5'd7;
        id_use_ra = 1'b1;
        ex_rd = 5'd7;
        ex_reg_write = 1'b1;
        mem_rd = 5'd7;
        mem_reg_write = 1'b1;
        wb_rd = 5'd7;
        wb_reg_write = 1'b1;
        #1;
        check_result(forward_a == 2'b01, "A selects EX result over MEM/WB");
        check_result(forwarded_a[31:0] == ex_result[31:0], "A data comes from EX");

        // Test 3: MEM forwarding when EX does not match
        test_num = test_num + 1;
        $display("\n[TEST %0d] MEM forwarding", test_num);
        clear_inputs();
        id_rb = 5'd9;
        id_use_rb = 1'b1;
        ex_rd = 5'd8;
        ex_reg_write = 1'b1;
        mem_rd = 5'd9;
        mem_reg_write = 1'b1;
        wb_rd = 5'd9;
        wb_reg_write = 1'b1;
        #1;
        check_result(forward_b == 2'b10, "B selects MEM result");
        check_result(forwarded_b[31:0] == mem_result[31:0], "B data comes from MEM");

        // Test 4: WB forwarding when only WB matches
        test_num = test_num + 1;
        $display("\n[TEST %0d] WB forwarding", test_num);
        clear_inputs();
        id_rc = 5'd11;
        id_use_rc = 1'b1;
        ex_rd = 5'd10;
        ex_reg_write = 1'b1;
        mem_rd = 5'd12;
        mem_reg_write = 1'b1;
        wb_rd = 5'd11;
        wb_reg_write = 1'b1;
        #1;
        check_result(forward_c == 2'b11, "C selects WB result");
        check_result(forwarded_c[31:0] == wb_result[31:0], "C data comes from WB");

        // Test 5: r0 should never forward
        test_num = test_num + 1;
        $display("\n[TEST %0d] Zero register bypass disabled", test_num);
        clear_inputs();
        id_ra = 5'd0;
        id_use_ra = 1'b1;
        ex_rd = 5'd0;
        ex_reg_write = 1'b1;
        #1;
        check_result(forward_a == 2'b00, "r0 does not trigger forwarding");

        // Test 6: Load-use hazard detection
        test_num = test_num + 1;
        $display("\n[TEST %0d] Load-use hazard", test_num);
        clear_inputs();
        id_ra = 5'd13;
        id_use_ra = 1'b1;
        ex_rd = 5'd13;
        ex_reg_write = 1'b1;
        ex_is_load = 1'b1;
        #1;
        check_result(load_use_hazard == 1'b1, "hazard asserted for dependent load-use");

        // Test 7: No hazard if EX op is not load
        test_num = test_num + 1;
        $display("\n[TEST %0d] No hazard for non-load producer", test_num);
        ex_is_load = 1'b0;
        #1;
        check_result(load_use_hazard == 1'b0, "hazard cleared when producer is not load");

        $display("\n====================================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("====================================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_forwarding_unit failed with %0d checks", fail_count);
        end
    end

endmodule
