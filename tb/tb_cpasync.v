//============================================================================
// RalphGPU - cp.async RTL Testbench
// Tests async_copy_engine: CA/CG/BULK transfers, commit+wait pipeline
// State path: ST_IDLE → ST_ISSUE → ST_WAIT_RESP → ST_WRITE_SM → ST_IDLE
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_cpasync;

    //------------------------------------------------------------------------
    // DUT signals
    //------------------------------------------------------------------------
    reg         clk, rst_n;
    reg  [5:0]  opcode, func;
    reg         valid_in;
    reg  [31:0] src_addr;
    reg  [13:0] dst_addr;
    reg  [3:0]  size;
    reg  [2:0]  cache_hint;
    reg  [3:0]  wait_count;

    // TMA interface (unused in basic tests)
    reg  [63:0] tensor_desc;
    reg  [31:0] tensor_coord_x, tensor_coord_y;

    // st.async interface (unused in basic tests)
    reg         is_store;
    reg  [31:0] store_gmem_addr;
    reg  [127:0] store_data;

    // DUT outputs
    wire        ready, done;
    wire [3:0]  pending_count;
    wire        tma_busy;

    // Global memory read interface
    wire        gmem_req_valid;
    wire [31:0] gmem_req_addr;
    wire [4:0]  gmem_req_size;
    wire [2:0]  gmem_req_cache;
    reg         gmem_resp_valid;
    reg  [127:0] gmem_resp_data;

    // Global memory write interface
    wire        gmem_wr_valid;
    wire [31:0] gmem_wr_addr;
    wire [127:0] gmem_wr_data;
    wire [4:0]  gmem_wr_size;
    reg         gmem_wr_done;

    // Shared memory write interface
    wire        smem_wr_en;
    wire [13:0] smem_wr_addr;
    wire [127:0] smem_wr_data;
    wire [4:0]  smem_wr_size;

    // Shared memory read interface (unused)
    wire        smem_rd_en;
    wire [13:0] smem_rd_addr;
    reg  [127:0] smem_rd_data;
    reg         smem_rd_valid;

    //------------------------------------------------------------------------
    // Mock global memory — 1-cycle latency response
    //------------------------------------------------------------------------
    reg [127:0] gmem [0:255];  // 256 x 128-bit entries (indexed by addr[11:4])
    reg         gmem_req_pending;
    reg [31:0]  gmem_req_addr_saved;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gmem_resp_valid <= 1'b0;
            gmem_req_pending <= 1'b0;
        end else begin
            gmem_resp_valid <= 1'b0;
            if (gmem_req_valid) begin
                // 1-cycle latency: register the request
                gmem_req_pending <= 1'b1;
                gmem_req_addr_saved <= gmem_req_addr;
            end
            if (gmem_req_pending) begin
                gmem_resp_valid <= 1'b1;
                gmem_resp_data <= gmem[gmem_req_addr_saved[11:4]];
                gmem_req_pending <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Mock shared memory — capture writes
    //------------------------------------------------------------------------
    reg [127:0] smem [0:255];  // 256 x 128-bit entries

    always @(posedge clk) begin
        if (smem_wr_en) begin
            smem[smem_wr_addr[11:4]] <= smem_wr_data;
        end
    end

    //------------------------------------------------------------------------
    // DUT instantiation
    //------------------------------------------------------------------------
    async_copy_engine #(
        .MAX_GROUPS(8),
        .MAX_PENDING(16),
        .SHARED_MEM_ADDR_W(14),
        .GLOBAL_ADDR_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .opcode(opcode),
        .func(func),
        .valid_in(valid_in),
        .src_addr(src_addr),
        .dst_addr(dst_addr),
        .size(size),
        .cache_hint(cache_hint),
        .wait_count(wait_count),
        .tensor_desc(tensor_desc),
        .tensor_coord_x(tensor_coord_x),
        .tensor_coord_y(tensor_coord_y),
        .is_store(is_store),
        .store_gmem_addr(store_gmem_addr),
        .store_data(store_data),
        .ready(ready),
        .done(done),
        .pending_count(pending_count),
        .tma_busy(tma_busy),
        .gmem_req_valid(gmem_req_valid),
        .gmem_req_addr(gmem_req_addr),
        .gmem_req_size(gmem_req_size),
        .gmem_req_cache(gmem_req_cache),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_data(gmem_resp_data),
        .gmem_wr_valid(gmem_wr_valid),
        .gmem_wr_addr(gmem_wr_addr),
        .gmem_wr_data(gmem_wr_data),
        .gmem_wr_size(gmem_wr_size),
        .gmem_wr_done(gmem_wr_done),
        .smem_wr_en(smem_wr_en),
        .smem_wr_addr(smem_wr_addr),
        .smem_wr_data(smem_wr_data),
        .smem_wr_size(smem_wr_size),
        .smem_rd_en(smem_rd_en),
        .smem_rd_addr(smem_rd_addr),
        .smem_rd_data(smem_rd_data),
        .smem_rd_valid(smem_rd_valid)
    );

    //------------------------------------------------------------------------
    // Clock generation
    //------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // Test infrastructure
    //------------------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task issue_cpasync;
        input [5:0] t_func;
        input [31:0] t_src;
        input [13:0] t_dst;
        input [3:0] t_size;
        begin
            @(negedge clk);
            opcode <= `OP_CPASYNC;
            func <= t_func;
            valid_in <= 1'b1;
            src_addr <= t_src;
            dst_addr <= t_dst;
            size <= t_size;
            cache_hint <= 3'b0;
            is_store <= 1'b0;
            @(negedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_commit;
        begin
            @(negedge clk);
            opcode <= `OP_CPASYNC;
            func <= `CPASYNC_COMMIT;
            valid_in <= 1'b1;
            @(negedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_wait_all;
        begin
            @(negedge clk);
            opcode <= `OP_CPASYNC;
            func <= `CPASYNC_WAIT_ALL;
            valid_in <= 1'b1;
            @(negedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_copy_done;
        input integer max_cycles;
        integer i;
        begin
            // First wait for ready to drop (FSM starts processing)
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (!ready) begin
                    i = max_cycles;
                end
            end
            // Then wait for ready to come back (FSM back to IDLE)
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (ready && pending_count == 0) begin
                    i = max_cycles;
                end
            end
            // Extra cycle for smem write to settle
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    task wait_idle;
        input integer max_cycles;
        integer i;
        begin
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (ready) begin
                    i = max_cycles; // break
                end
            end
        end
    endtask

    task check_smem;
        input [13:0] addr;
        input [127:0] expected;
        input [3:0] byte_count;
        reg [127:0] actual;
        reg match;
        begin
            actual = smem[addr[11:4]];
            // Compare only the relevant bytes
            case (byte_count)
                4'd4:  match = (actual[31:0] == expected[31:0]);
                4'd8:  match = (actual[63:0] == expected[63:0]);
                4'd16: match = (actual == expected);  // full 128-bit for bulk 16B
                default: match = (actual == expected);
            endcase
            if (match) begin
                $display("  PASS: smem[0x%03x] = 0x%032x (%0d bytes)", addr, actual, byte_count);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: smem[0x%03x] = 0x%032x, expected 0x%032x (%0d bytes)",
                         addr, actual, expected, byte_count);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_cpasync.vcd");
        $dumpvars(0, tb_cpasync);

        // Init signals
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        src_addr = 0;
        dst_addr = 0;
        size = 0;
        cache_hint = 0;
        wait_count = 0;
        tensor_desc = 0;
        tensor_coord_x = 0;
        tensor_coord_y = 0;
        is_store = 0;
        store_gmem_addr = 0;
        store_data = 0;
        gmem_wr_done = 0;
        smem_rd_data = 0;
        smem_rd_valid = 0;

        // Pre-load mock global memory
        gmem[8'h10] = {96'b0, 32'hDEADBEEF};                              // 0x100
        gmem[8'h20] = {64'b0, 32'h9ABCDEF0, 32'h12345678};                // 0x200
        gmem[8'h30] = {32'hDDDDDDDD, 32'hCCCCCCCC, 32'hBBBBBBBB, 32'hAAAAAAAA}; // 0x300
        gmem[8'h40] = {96'b0, 32'hCAFEBABE};                              // 0x400
        gmem[8'h50] = {96'b0, 32'hFEEDFACE};                              // 0x500

        // Reset
        #20;
        rst_n = 1;
        #20;

        $display("========================================");
        $display("cp.async RTL Testbench");
        $display("========================================");

        //--------------------------------------------------------------------
        // Test 1: cp.async.ca 4B (gmem[0x100] → smem[0x100])
        //--------------------------------------------------------------------
        test_num = 1;
        $display("\nTest %0d: cp.async.ca 4B", test_num);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0100, 14'h100, 4'd4);
        wait_copy_done(50);
        check_smem(14'h100, {96'b0, 32'hDEADBEEF}, 4'd4);

        //--------------------------------------------------------------------
        // Test 2: cp.async.cg 8B (gmem[0x200] → smem[0x200])
        //--------------------------------------------------------------------
        test_num = 2;
        $display("\nTest %0d: cp.async.cg 8B", test_num);
        issue_cpasync(`CPASYNC_CG, 32'h0000_0200, 14'h200, 4'd8);
        wait_copy_done(50);
        check_smem(14'h200, {64'b0, 32'h9ABCDEF0, 32'h12345678}, 4'd8);

        //--------------------------------------------------------------------
        // Test 3: cp.async.bulk 16B (gmem[0x300] → smem[0x300])
        //--------------------------------------------------------------------
        test_num = 3;
        $display("\nTest %0d: cp.async.bulk 16B", test_num);
        // RTL bulk path is same as ca/cg. Mock gmem returns full 128-bit line.
        issue_cpasync(`CPASYNC_BULK, 32'h0000_0300, 14'h300, 4'd0);
        wait_copy_done(50);
        check_smem(14'h300, {32'hDDDDDDDD, 32'hCCCCCCCC, 32'hBBBBBBBB, 32'hAAAAAAAA}, 4'd0);

        //--------------------------------------------------------------------
        // Test 4: cp.async.ca + commit + wait_all pipeline
        //--------------------------------------------------------------------
        test_num = 4;
        $display("\nTest %0d: cp.async.ca + commit + wait_all", test_num);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0400, 14'h400, 4'd4);
        wait_copy_done(50);
        issue_commit();
        @(posedge clk); @(posedge clk);
        issue_wait_all();
        wait_idle(50);
        check_smem(14'h400, {96'b0, 32'hCAFEBABE}, 4'd4);

        //--------------------------------------------------------------------
        // Test 5: Back-to-back cp.async
        //--------------------------------------------------------------------
        test_num = 5;
        $display("\nTest %0d: back-to-back cp.async", test_num);
        // Issue first copy
        issue_cpasync(`CPASYNC_CA, 32'h0000_0400, 14'h040, 4'd4);
        // Wait for first to complete, then issue second
        wait_copy_done(50);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0500, 14'h050, 4'd4);
        wait_copy_done(50);
        check_smem(14'h040, {96'b0, 32'hCAFEBABE}, 4'd4);
        check_smem(14'h050, {96'b0, 32'hFEEDFACE}, 4'd4);

        //--------------------------------------------------------------------
        // Results
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Results: %0d/%0d PASSED", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("%0d TESTS FAILED", fail_count);
        $display("========================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT: simulation exceeded 50000ns");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
