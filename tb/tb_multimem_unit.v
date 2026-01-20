//============================================================================
// RalphGPU - Multimem Unit Testbench
// Tests distributed shared memory operations
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_multimem_unit;

    parameter CLK_PERIOD = 10;
    parameter SM_ID = 0;
    parameter NUM_SMs = 2;
    parameter SHARED_MEM_ADDR_W = 14;
    parameter DATA_WIDTH = 32;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    // Issue interface
    reg                         valid_in;
    reg [5:0]                   func;
    reg [31:0]                  addr;
    reg [DATA_WIDTH-1:0]        wdata;
    reg [2:0]                   red_op;

    // Result interface
    wire                        done;
    wire                        result_valid;
    wire [DATA_WIDTH-1:0]       result_data;

    // Local SMEM interface
    wire                        local_smem_rd_en;
    wire [SHARED_MEM_ADDR_W-1:0] local_smem_rd_addr;
    reg  [DATA_WIDTH-1:0]       local_smem_rd_data;
    reg                         local_smem_rd_valid;

    wire                        local_smem_wr_en;
    wire [SHARED_MEM_ADDR_W-1:0] local_smem_wr_addr;
    wire [DATA_WIDTH-1:0]       local_smem_wr_data;
    reg                         local_smem_wr_done;

    // Cluster interface (stub for single-SM testing)
    wire                        cluster_req_valid;
    wire [NUM_SMs-1:0]          cluster_req_targets;
    wire [1:0]                  cluster_req_op;
    wire [SHARED_MEM_ADDR_W-1:0] cluster_req_addr;
    wire [DATA_WIDTH-1:0]       cluster_req_data;
    wire [2:0]                  cluster_req_red_op;
    reg                         cluster_req_ack;
    reg                         cluster_resp_valid;
    reg  [DATA_WIDTH-1:0]       cluster_resp_data;

    // Remote request interface (stub)
    reg                         remote_req_valid;
    reg  [1:0]                  remote_req_op;
    reg  [SHARED_MEM_ADDR_W-1:0] remote_req_addr;
    reg  [DATA_WIDTH-1:0]       remote_req_data;
    reg  [2:0]                  remote_req_red_op;
    wire                        remote_req_done;
    wire [DATA_WIDTH-1:0]       remote_resp_data;

    // Test tracking
    integer test_count;
    integer pass_count;
    integer fail_count;

    //------------------------------------------------------------------------
    // Shared Memory Simulation
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] smem [0:1023];

    always @(posedge clk) begin
        if (local_smem_rd_en && !local_smem_rd_valid) begin
            local_smem_rd_data <= smem[local_smem_rd_addr[9:0]];
            local_smem_rd_valid <= 1'b1;
            `ifdef SIMULATION
            $display("[SMEM] Read: addr=0x%04x data=0x%08x", local_smem_rd_addr, smem[local_smem_rd_addr[9:0]]);
            `endif
        end else begin
            local_smem_rd_valid <= 1'b0;
        end

        if (local_smem_wr_en && !local_smem_wr_done) begin
            smem[local_smem_wr_addr[9:0]] <= local_smem_wr_data;
            local_smem_wr_done <= 1'b1;
            `ifdef SIMULATION
            $display("[SMEM] Write: addr=0x%04x data=0x%08x", local_smem_wr_addr, local_smem_wr_data);
            `endif
        end else begin
            local_smem_wr_done <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    multimem_unit #(
        .SM_ID(SM_ID),
        .NUM_SMs(NUM_SMs),
        .SHARED_MEM_ADDR_W(SHARED_MEM_ADDR_W),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .func(func),
        .addr(addr),
        .wdata(wdata),
        .red_op(red_op),
        .done(done),
        .result_valid(result_valid),
        .result_data(result_data),
        .local_smem_rd_en(local_smem_rd_en),
        .local_smem_rd_addr(local_smem_rd_addr),
        .local_smem_rd_data(local_smem_rd_data),
        .local_smem_rd_valid(local_smem_rd_valid),
        .local_smem_wr_en(local_smem_wr_en),
        .local_smem_wr_addr(local_smem_wr_addr),
        .local_smem_wr_data(local_smem_wr_data),
        .local_smem_wr_done(local_smem_wr_done),
        .cluster_req_valid(cluster_req_valid),
        .cluster_req_targets(cluster_req_targets),
        .cluster_req_op(cluster_req_op),
        .cluster_req_addr(cluster_req_addr),
        .cluster_req_data(cluster_req_data),
        .cluster_req_red_op(cluster_req_red_op),
        .cluster_req_ack(cluster_req_ack),
        .cluster_resp_valid(cluster_resp_valid),
        .cluster_resp_data(cluster_resp_data),
        .remote_req_valid(remote_req_valid),
        .remote_req_op(remote_req_op),
        .remote_req_addr(remote_req_addr),
        .remote_req_data(remote_req_data),
        .remote_req_red_op(remote_req_red_op),
        .remote_req_done(remote_req_done),
        .remote_resp_data(remote_resp_data)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_inputs;
        begin
            valid_in <= 1'b0;
            func <= 6'b0;
            addr <= 32'b0;
            wdata <= 32'b0;
            red_op <= 3'b0;
            cluster_req_ack <= 1'b0;
            cluster_resp_valid <= 1'b0;
            cluster_resp_data <= 32'b0;
            remote_req_valid <= 1'b0;
            remote_req_op <= 2'b0;
            remote_req_addr <= 14'b0;
            remote_req_data <= 32'b0;
            remote_req_red_op <= 3'b0;
        end
    endtask

    task test_local_load;
        input [SHARED_MEM_ADDR_W-1:0] smem_addr;
        input [31:0] expected_data;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: multimem.ld local ---", test_count);
            $display("  addr=0x%04x expected=0x%08x", smem_addr, expected_data);

            @(posedge clk);
            valid_in <= 1'b1;
            func <= `MULTIMEM_LD;
            addr <= {8'h01, {(24-SHARED_MEM_ADDR_W){1'b0}}, smem_addr};  // Target SM 0
            @(posedge clk);
            valid_in <= 1'b0;

            wait (done);
            @(posedge clk);

            if (result_valid && result_data == expected_data) begin
                $display("[PASS] result_data=0x%08x", result_data);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] result_data=0x%08x, expected=0x%08x", result_data, expected_data);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_local_store;
        input [SHARED_MEM_ADDR_W-1:0] smem_addr;
        input [31:0] store_data;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: multimem.st local ---", test_count);
            $display("  addr=0x%04x data=0x%08x", smem_addr, store_data);

            @(posedge clk);
            valid_in <= 1'b1;
            func <= `MULTIMEM_ST;
            addr <= {8'h01, {(24-SHARED_MEM_ADDR_W){1'b0}}, smem_addr};  // Target SM 0
            wdata <= store_data;
            @(posedge clk);
            valid_in <= 1'b0;

            wait (done);
            @(posedge clk);
            @(posedge clk);

            if (smem[smem_addr[9:0]] == store_data) begin
                $display("[PASS] smem[0x%04x]=0x%08x", smem_addr, smem[smem_addr[9:0]]);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] smem[0x%04x]=0x%08x, expected=0x%08x",
                         smem_addr, smem[smem_addr[9:0]], store_data);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_local_reduce;
        input [SHARED_MEM_ADDR_W-1:0] smem_addr;
        input [31:0] reduce_data;
        input [2:0] op;
        input [31:0] expected_result;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: multimem.red local ---", test_count);
            $display("  addr=0x%04x data=0x%08x op=%0d", smem_addr, reduce_data, op);

            @(posedge clk);
            valid_in <= 1'b1;
            func <= `MULTIMEM_RED;
            addr <= {8'h01, {(24-SHARED_MEM_ADDR_W){1'b0}}, smem_addr};  // Target SM 0
            wdata <= reduce_data;
            red_op <= op;
            @(posedge clk);
            valid_in <= 1'b0;

            wait (done);
            @(posedge clk);
            @(posedge clk);

            if (smem[smem_addr[9:0]] == expected_result) begin
                $display("[PASS] smem[0x%04x]=0x%08x", smem_addr, smem[smem_addr[9:0]]);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] smem[0x%04x]=0x%08x, expected=0x%08x",
                         smem_addr, smem[smem_addr[9:0]], expected_result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_remote_request;
        input [1:0] op;
        input [SHARED_MEM_ADDR_W-1:0] smem_addr;
        input [31:0] req_data;
        input [2:0] r_op;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Remote request handling (op=%0d) ---", test_count, op);

            @(posedge clk);
            remote_req_valid <= 1'b1;
            remote_req_op <= op;
            remote_req_addr <= smem_addr;
            remote_req_data <= req_data;
            remote_req_red_op <= r_op;
            @(posedge clk);
            remote_req_valid <= 1'b0;

            wait (remote_req_done);
            @(posedge clk);

            $display("[PASS] Remote request handled");
            pass_count = pass_count + 1;
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer i;
    initial begin
        $display("============================================================");
        $display("RalphGPU Multimem Unit Testbench");
        $display("============================================================");

        $dumpfile("tb_multimem_unit.vcd");
        $dumpvars(0, tb_multimem_unit);

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Initialize
        rst_n = 0;
        reset_inputs();
        local_smem_rd_data = 32'b0;
        local_smem_rd_valid = 1'b0;
        local_smem_wr_done = 1'b0;

        // Initialize SMEM with test data
        for (i = 0; i < 1024; i = i + 1) begin
            smem[i] = 32'h1000_0000 + i;
        end

        // Reset
        #100;
        rst_n = 1;
        #20;

        //--------------------------------------------------------------------
        // Test 1: Local load
        //--------------------------------------------------------------------
        test_local_load(14'h0100, 32'h1000_0100);

        //--------------------------------------------------------------------
        // Test 2: Local load from different address
        //--------------------------------------------------------------------
        test_local_load(14'h0200, 32'h1000_0200);

        //--------------------------------------------------------------------
        // Test 3: Local store
        //--------------------------------------------------------------------
        test_local_store(14'h0300, 32'hDEAD_BEEF);

        //--------------------------------------------------------------------
        // Test 4: Verify stored data with load
        //--------------------------------------------------------------------
        test_local_load(14'h0300, 32'hDEAD_BEEF);

        //--------------------------------------------------------------------
        // Test 5: Local store to another address
        //--------------------------------------------------------------------
        test_local_store(14'h0400, 32'hCAFE_BABE);

        //--------------------------------------------------------------------
        // Test 6: Local reduce ADD (0x1000_0010 + 0x10 = 0x1000_0020)
        //--------------------------------------------------------------------
        smem[16] = 32'h0000_0010;
        @(posedge clk);
        test_local_reduce(14'h0010, 32'h0000_0010, 3'd0, 32'h0000_0020);

        //--------------------------------------------------------------------
        // Test 7: Local reduce MAX
        //--------------------------------------------------------------------
        smem[32] = 32'h0000_0050;
        @(posedge clk);
        test_local_reduce(14'h0020, 32'h0000_0100, 3'd2, 32'h0000_0100);

        //--------------------------------------------------------------------
        // Test 8: Local reduce MIN
        //--------------------------------------------------------------------
        smem[48] = 32'h0000_0100;
        @(posedge clk);
        test_local_reduce(14'h0030, 32'h0000_0050, 3'd1, 32'h0000_0050);

        //--------------------------------------------------------------------
        // Test 9: Local reduce OR
        //--------------------------------------------------------------------
        smem[64] = 32'h0F0F_0F0F;
        @(posedge clk);
        test_local_reduce(14'h0040, 32'hF0F0_F0F0, 3'd4, 32'hFFFF_FFFF);

        //--------------------------------------------------------------------
        // Test 10: Local reduce AND
        //--------------------------------------------------------------------
        smem[80] = 32'hFF00_FF00;
        @(posedge clk);
        test_local_reduce(14'h0050, 32'h00FF_00FF, 3'd3, 32'h0000_0000);

        //--------------------------------------------------------------------
        // Test 11: Handle remote load request
        //--------------------------------------------------------------------
        test_remote_request(2'd0, 14'h0100, 32'b0, 3'b0);

        //--------------------------------------------------------------------
        // Test 12: Handle remote store request
        //--------------------------------------------------------------------
        test_remote_request(2'd1, 14'h0500, 32'h1234_5678, 3'b0);

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        #100;
        $display("\n============================================================");
        $display("Test Summary: %0d/%0d tests passed", pass_count, test_count);
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("FAILURES: %0d", fail_count);
        end
        $display("============================================================");

        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
