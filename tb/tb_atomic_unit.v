`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_atomic_unit;
    parameter NUM_LANES = 32;

    reg clk;
    reg rst_n;

    // DUT inputs
    reg req_valid;
    reg [5:0] func;
    reg [NUM_LANES*32-1:0] addr;
    reg [NUM_LANES*32-1:0] operand_a;
    reg [NUM_LANES*32-1:0] operand_b;
    reg [NUM_LANES-1:0] lane_mask;
    reg mem_shared;

    // Mem responses
    reg resp_read_valid;
    reg resp_write_valid;
    reg [NUM_LANES*32-1:0] mem_rdata;

    // DUT outputs
    wire req_ready;
    wire mem_req;
    wire mem_write;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [5:0] mem_lane;
    wire [NUM_LANES*32-1:0] result;
    wire [NUM_LANES-1:0] result_mask;
    wire result_valid;
    wire busy;

    atomic_unit #(
        .NUM_LANES(NUM_LANES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .func(func),
        .addr(addr),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .lane_mask(lane_mask),
        .mem_shared(mem_shared),
        .mem_req(mem_req),
        .mem_write(mem_write),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_lane(mem_lane),
        .resp_read_valid(resp_read_valid),
        .resp_write_valid(resp_write_valid),
        .mem_rdata(mem_rdata),
        .result(result),
        .result_mask(result_mask),
        .result_valid(result_valid),
        .busy(busy)
    );

    // Clock gen
    always #5 clk = ~clk;

    // Memory model (simple)
    reg [31:0] fake_mem [0:255];
    
    // Simulate memory delay
    reg mem_req_d;
    reg mem_write_d;
    reg [31:0] mem_addr_d;
    reg [31:0] mem_wdata_d;
    reg [5:0] mem_lane_d;

    always @(posedge clk) begin
        mem_req_d <= mem_req;
        mem_write_d <= mem_write;
        mem_addr_d <= mem_addr;
        mem_wdata_d <= mem_wdata;
        mem_lane_d <= mem_lane;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            resp_read_valid <= 1'b0;
            resp_write_valid <= 1'b0;
            mem_rdata <= 0;
        end else begin
            resp_read_valid <= 1'b0;
            resp_write_valid <= 1'b0;

            if (mem_req_d) begin
                if (!mem_write_d) begin
                    resp_read_valid <= 1'b1;
                    // Write the read value to the appropriate lane in mem_rdata
                    mem_rdata <= {NUM_LANES{fake_mem[mem_addr_d[7:0]]}};
                end else begin
                    fake_mem[mem_addr_d[7:0]] <= mem_wdata_d;
                    resp_write_valid <= 1'b1;
                end
            end
        end
    end

    integer failures;

    initial begin
        $dumpfile("tb_atomic_unit.vcd");
        $dumpvars(0, tb_atomic_unit);
        
        failures = 0;
        
        for (integer i=0; i<256; i=i+1) fake_mem[i] = 0;

        clk = 0;
        rst_n = 0;
        req_valid = 0;
        func = 0;
        addr = 0;
        operand_a = 0;
        operand_b = 0;
        lane_mask = 0;
        mem_shared = 0;

        #20 rst_n = 1;
        #10;

        // --- Test 1: ATOM.ADD ---
        $display("Test 1: ATOM.ADD (Lane 0)");
        fake_mem[8'h10] = 32'd100;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_ADD;
        addr[31:0] = 32'h10;
        operand_a[31:0] = 32'd50;
        lane_mask = 32'h00000001;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h10] !== 32'd150) begin
            $display("FAIL: ATOM.ADD. Expected 150, got %0d", fake_mem[8'h10]);
            failures = failures + 1;
        end else $display("PASS: ATOM.ADD");
        #20;

        // --- Test 2: ATOM.MAX_U ---
        $display("Test 2: ATOM.MAX_U (Lane 1)");
        fake_mem[8'h20] = 32'd100;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_MAX_U;
        addr[63:32] = 32'h20;
        operand_a[63:32] = 32'd200;
        lane_mask = 32'h00000002;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h20] !== 32'd200) begin
            $display("FAIL: ATOM.MAX. Expected 200, got %0d", fake_mem[8'h20]);
            failures = failures + 1;
        end else $display("PASS: ATOM.MAX");
        #20;

        // --- Test 3: ATOM.MIN_U ---
        $display("Test 3: ATOM.MIN_U (Lane 2)");
        fake_mem[8'h24] = 32'd100;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_MIN_U;
        addr[95:64] = 32'h24;
        operand_a[95:64] = 32'd50;
        lane_mask = 32'h00000004;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h24] !== 32'd50) begin
            $display("FAIL: ATOM.MIN. Expected 50, got %0d", fake_mem[8'h24]);
            failures = failures + 1;
        end else $display("PASS: ATOM.MIN");
        #20;

        // --- Test 4: ATOM.CAS (Success) ---
        $display("Test 4: ATOM.CAS Success (Lane 3)");
        fake_mem[8'h30] = 32'd100;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_CAS;
        addr[127:96] = 32'h30;
        operand_a[127:96] = 32'd100; // compare
        operand_b[127:96] = 32'd999; // swap
        lane_mask = 32'h00000008;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h30] !== 32'd999) begin
            $display("FAIL: ATOM.CAS (Success). Expected 999, got %0d", fake_mem[8'h30]);
            failures = failures + 1;
        end else $display("PASS: ATOM.CAS (Success)");
        #20;

        // --- Test 5: ATOM.CAS (Fail) ---
        $display("Test 5: ATOM.CAS Fail (Lane 3)");
        fake_mem[8'h30] = 32'd100;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_CAS;
        addr[127:96] = 32'h30;
        operand_a[127:96] = 32'd50;  // compare (mismatch)
        operand_b[127:96] = 32'd999; // swap
        lane_mask = 32'h00000008;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h30] !== 32'd100) begin
            $display("FAIL: ATOM.CAS (Fail). Expected 100, got %0d", fake_mem[8'h30]);
            failures = failures + 1;
        end else $display("PASS: ATOM.CAS (Fail)");
        #20;

        // --- Test 6: Concurrent ATOM.ADD (Lanes 0 and 1) ---
        $display("Test 6: Concurrent ATOM.ADD (Lanes 0,1)");
        fake_mem[8'h40] = 32'd0;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_ADD;
        addr[31:0] = 32'h40;
        operand_a[31:0] = 32'd10;
        addr[63:32] = 32'h40;
        operand_a[63:32] = 32'd20;
        lane_mask = 32'h00000003;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h40] !== 32'd30) begin
            $display("FAIL: Concurrent ATOM.ADD. Expected 30, got %0d", fake_mem[8'h40]);
            failures = failures + 1;
        end else $display("PASS: Concurrent ATOM.ADD");
        #20;

        // --- Test 7: Overflow ATOM.ADD ---
        $display("Test 7: Overflow ATOM.ADD (Lane 0)");
        fake_mem[8'h50] = 32'hFFFFFFFF;
        wait(req_ready);
        req_valid = 1;
        func = `ATOM_ADD;
        addr[31:0] = 32'h50;
        operand_a[31:0] = 32'd5;
        lane_mask = 32'h00000001;
        #10 req_valid = 0;
        wait(result_valid);
        if (fake_mem[8'h50] !== 32'd4) begin
            $display("FAIL: Overflow ATOM.ADD. Expected 4, got %0d", fake_mem[8'h50]);
            failures = failures + 1;
        end else $display("PASS: Overflow ATOM.ADD");
        #20;

        if (failures == 0) begin
            $display("========================================");
            $display("ALL TESTS PASSED");
            $display("========================================");
        end else begin
            $display("========================================");
            $display("%0d TESTS FAILED", failures);
            $display("========================================");
            $finish(1);
        end

        $finish;
    end
endmodule
