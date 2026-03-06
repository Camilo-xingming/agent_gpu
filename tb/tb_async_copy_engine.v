`timescale 1ns/1ps
`include "gpu_defines.vh"

module tb_async_copy_engine;

    reg clk;
    reg rst_n;

    // Command interface
    reg [5:0] opcode;
    reg [5:0] func;
    reg valid_in;
    reg [31:0] src_addr;
    reg [17:0] dst_addr;
    reg [3:0] size;
    reg [2:0] cache_hint;
    reg [3:0] wait_count;

    // TMA interface
    reg [63:0] tensor_desc;
    reg [31:0] tensor_coord_x;
    reg [31:0] tensor_coord_y;

    // st.async interface
    reg is_store;
    reg [31:0] store_gmem_addr;
    reg [127:0] store_data;

    wire ready;
    wire done;
    wire [3:0] pending_count;
    wire tma_busy;

    wire gmem_req_valid;
    wire [31:0] gmem_req_addr;
    wire [4:0] gmem_req_size;
    wire [2:0] gmem_req_cache;
    reg gmem_resp_valid;
    reg [127:0] gmem_resp_data;

    wire gmem_wr_valid;
    wire [31:0] gmem_wr_addr;
    wire [127:0] gmem_wr_data;
    wire [4:0] gmem_wr_size;
    reg gmem_wr_done;

    wire smem_wr_en;
    wire [17:0] smem_wr_addr;
    wire [127:0] smem_wr_data;
    wire [4:0] smem_wr_size;

    wire smem_rd_en;
    wire [17:0] smem_rd_addr;
    reg [127:0] smem_rd_data;
    reg smem_rd_valid;

    async_copy_engine #(
        .GLOBAL_ADDR_W(32),
        .SHARED_MEM_ADDR_W(18),
        .MAX_GROUPS(8)
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

    // Clock
    always #5 clk = ~clk;

    // Shadow memories + scoreboards
    reg [127:0] smem_shadow [0:1023];
    reg [127:0] gmem_shadow [0:1023];

    reg [31:0]  last_req_addr;
    reg [4:0]   last_req_size;
    reg [2:0]   last_req_cache;
    reg [31:0]  last_wr_addr;
    reg [4:0]   last_wr_size;
    reg [127:0] last_wr_data;
    reg [17:0]  last_smem_addr;
    reg [4:0]   last_smem_size;
    reg [127:0] last_smem_data;

    integer gmem_read_count;
    integer gmem_write_count;
    integer smem_write_count;

    integer total_checks;
    integer passed_checks;
    integer failed_checks;

    integer gmem_resp_latency;
    integer gmem_wr_latency;
    integer resp_timer;
    integer wr_timer;
    reg [31:0] resp_addr_pending;
    reg [31:0] wr_addr_pending;
    reg [127:0] wr_data_pending;
    reg [4:0] wr_size_pending;

    function [127:0] mk_resp_data(input [31:0] addr);
        begin
            mk_resp_data = {32'hDEADBEEF, 32'hCAFEBABE, 32'h8BADF00D, addr};
        end
    endfunction

    function [9:0] smem_idx(input [17:0] addr);
        begin
            smem_idx = addr[9:0];
        end
    endfunction

    function [9:0] gmem_idx(input [31:0] addr);
        begin
            gmem_idx = addr[13:4];
        end
    endfunction

    task expect32(input [255:0] name, input [31:0] actual, input [31:0] expected);
        begin
            total_checks = total_checks + 1;
            if (actual === expected) begin
                passed_checks = passed_checks + 1;
                $display("[PASS] %0s = 0x%08h", name, actual);
            end else begin
                failed_checks = failed_checks + 1;
                $fatal(1, "[FAIL] %0s expected=0x%08h actual=0x%08h", name, expected, actual);
            end
        end
    endtask

    task expect128(input [255:0] name, input [127:0] actual, input [127:0] expected);
        begin
            total_checks = total_checks + 1;
            if (actual === expected) begin
                passed_checks = passed_checks + 1;
                $display("[PASS] %0s = 0x%032h", name, actual);
            end else begin
                failed_checks = failed_checks + 1;
                $fatal(1, "[FAIL] %0s expected=0x%032h actual=0x%032h", name, expected, actual);
            end
        end
    endtask

    task expect_true(input [255:0] name, input condition);
        begin
            total_checks = total_checks + 1;
            if (condition) begin
                passed_checks = passed_checks + 1;
                $display("[PASS] %0s", name);
            end else begin
                failed_checks = failed_checks + 1;
                $fatal(1, "[FAIL] %0s condition=false", name);
            end
        end
    endtask

    task wait_done(input integer timeout_cycles);
        integer t;
        begin
            t = 0;
            while (!done && t < timeout_cycles) begin
                @(posedge clk);
                t = t + 1;
            end
            if (!done) begin
                $fatal(1, "[FAIL] wait_done timeout after %0d cycles", timeout_cycles);
            end
            @(posedge clk);
        end
    endtask

    task wait_pending_zero(input integer timeout_cycles);
        integer t;
        begin
            t = 0;
            while (pending_count != 0 && t < timeout_cycles) begin
                @(posedge clk);
                t = t + 1;
            end
            if (pending_count != 0) begin
                $fatal(1, "[FAIL] wait_pending_zero timeout after %0d cycles", timeout_cycles);
            end
        end
    endtask

    task issue_cpasync_copy(
        input [5:0] f,
        input [31:0] s_addr,
        input [17:0] d_addr,
        input [3:0] sz,
        input [2:0] hint
    );
        integer base_reads;
        integer t;
        begin
            base_reads = gmem_read_count;
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in   <= 1'b1;
            opcode     <= `OP_CPASYNC;
            func       <= f;
            src_addr   <= s_addr;
            dst_addr   <= d_addr;
            size       <= sz;
            cache_hint <= hint;
            @(posedge clk);
            valid_in <= 1'b0;

            t = 0;
            while (gmem_read_count == base_reads && t < 200) begin
                @(posedge clk);
                t = t + 1;
            end
            if (gmem_read_count == base_reads) begin
                $fatal(1, "[FAIL] issue_cpasync_copy not accepted for addr=0x%08h", s_addr);
            end
        end
    endtask

    task issue_cpasync_bulk(
        input [31:0] s_addr,
        input [17:0] d_addr,
        input [3:0] sz,
        input [2:0] hint
    );
        begin
            issue_cpasync_copy(`CPASYNC_BULK, s_addr, d_addr, sz, hint);
        end
    endtask

    task issue_cpasync_commit;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            opcode   <= `OP_CPASYNC;
            func     <= `CPASYNC_COMMIT;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_cpasync_wait(input [3:0] w_cnt);
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in    <= 1'b1;
            opcode      <= `OP_CPASYNC;
            func        <= `CPASYNC_WAIT;
            wait_count  <= w_cnt;
            @(posedge clk);
            valid_in <= 1'b0;
            wait_done(400);
        end
    endtask

    task issue_cpasync_wait_no_block(input [3:0] w_cnt);
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in    <= 1'b1;
            opcode      <= `OP_CPASYNC;
            func        <= `CPASYNC_WAIT;
            wait_count  <= w_cnt;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_cpasync_wait_all;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            opcode   <= `OP_CPASYNC;
            func     <= `CPASYNC_WAIT_ALL;
            @(posedge clk);
            valid_in <= 1'b0;
            wait_done(500);
        end
    endtask

    task issue_cpasync_tensor;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            opcode   <= `OP_CPASYNC;
            func     <= `CPASYNC_BULK_TENSOR;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_st_global(input [31:0] addr, input [127:0] data, input [3:0] sz);
        integer base_writes;
        integer t;
        begin
            base_writes = gmem_write_count;
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in         <= 1'b1;
            opcode           <= `OP_ST_ASYNC;
            func             <= `ST_ASYNC_GLOBAL;
            store_gmem_addr  <= addr;
            store_data       <= data;
            size             <= sz;
            @(posedge clk);
            valid_in <= 1'b0;

            t = 0;
            while (gmem_write_count == base_writes && t < 300) begin
                @(posedge clk);
                t = t + 1;
            end
            if (gmem_write_count == base_writes) begin
                $fatal(1, "[FAIL] issue_st_global not accepted for addr=0x%08h", addr);
            end
        end
    endtask

    task issue_st_shared(input [17:0] addr, input [127:0] data, input [3:0] sz);
        integer base_smem_writes;
        integer t;
        begin
            base_smem_writes = smem_write_count;
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in   <= 1'b1;
            opcode     <= `OP_ST_ASYNC;
            func       <= `ST_ASYNC_SHARED;
            dst_addr   <= addr;
            store_data <= data;
            size       <= sz;
            @(posedge clk);
            valid_in <= 1'b0;

            t = 0;
            while (smem_write_count == base_smem_writes && t < 100) begin
                @(posedge clk);
                t = t + 1;
            end
            if (smem_write_count == base_smem_writes) begin
                $fatal(1, "[FAIL] issue_st_shared not accepted for addr=0x%05h", addr);
            end
        end
    endtask

    task issue_st_commit;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            opcode   <= `OP_ST_ASYNC;
            func     <= `ST_ASYNC_COMMIT;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_st_wait;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            opcode   <= `OP_ST_ASYNC;
            func     <= `ST_ASYNC_WAIT;
            @(posedge clk);
            valid_in <= 1'b0;
            wait_done(500);
        end
    endtask

    // Protocol/memory model
    always @(posedge clk) begin
        integer idx;

        if (!rst_n) begin
            gmem_resp_valid  <= 1'b0;
            gmem_resp_data   <= 128'b0;
            gmem_wr_done     <= 1'b0;
            resp_timer       <= -1;
            wr_timer         <= -1;
            resp_addr_pending<= 32'b0;
            wr_addr_pending  <= 32'b0;
            wr_data_pending  <= 128'b0;
            wr_size_pending  <= 5'b0;

            for (idx = 0; idx < 1024; idx = idx + 1) begin
                smem_shadow[idx] <= 128'b0;
                gmem_shadow[idx] <= 128'b0;
            end

            last_req_addr   <= 32'b0;
            last_req_size   <= 5'b0;
            last_req_cache  <= 3'b0;
            last_wr_addr    <= 32'b0;
            last_wr_size    <= 5'b0;
            last_wr_data    <= 128'b0;
            last_smem_addr  <= 18'b0;
            last_smem_size  <= 5'b0;
            last_smem_data  <= 128'b0;

            gmem_read_count  <= 0;
            gmem_write_count <= 0;
            smem_write_count <= 0;
        end else begin
            gmem_resp_valid <= 1'b0;
            gmem_wr_done    <= 1'b0;

            if (gmem_req_valid) begin
                last_req_addr  <= gmem_req_addr;
                last_req_size  <= gmem_req_size;
                last_req_cache <= gmem_req_cache;
                gmem_read_count <= gmem_read_count + 1;

                resp_addr_pending <= gmem_req_addr;
                resp_timer <= gmem_resp_latency;
            end else if (resp_timer >= 0) begin
                if (resp_timer == 0) begin
                    gmem_resp_valid <= 1'b1;
                    gmem_resp_data  <= mk_resp_data(resp_addr_pending);
                    resp_timer <= -1;
                end else begin
                    resp_timer <= resp_timer - 1;
                end
            end

            if (gmem_wr_valid) begin
                last_wr_addr <= gmem_wr_addr;
                last_wr_size <= gmem_wr_size;
                last_wr_data <= gmem_wr_data;
                gmem_write_count <= gmem_write_count + 1;

                wr_addr_pending <= gmem_wr_addr;
                wr_data_pending <= gmem_wr_data;
                wr_size_pending <= gmem_wr_size;
                wr_timer <= gmem_wr_latency;
            end else if (wr_timer >= 0) begin
                if (wr_timer == 0) begin
                    gmem_wr_done <= 1'b1;
                    gmem_shadow[gmem_idx(wr_addr_pending)] <= wr_data_pending;
                    wr_timer <= -1;
                end else begin
                    wr_timer <= wr_timer - 1;
                end
            end

            if (smem_wr_en) begin
                last_smem_addr <= smem_wr_addr;
                last_smem_size <= smem_wr_size;
                last_smem_data <= smem_wr_data;
                smem_write_count <= smem_write_count + 1;
                smem_shadow[smem_idx(smem_wr_addr)] <= smem_wr_data;
            end
        end
    end

    initial begin
        integer start_reads;
        integer i;

        $dumpfile("tb_async_copy_engine.vcd");
        $dumpvars(0, tb_async_copy_engine);

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        src_addr = 0;
        dst_addr = 0;
        size = 0;
        cache_hint = 0;
        wait_count = 0;
        tensor_desc = 64'h0210_0010_0002_0000;
        tensor_coord_x = 0;
        tensor_coord_y = 0;
        is_store = 0;
        store_gmem_addr = 0;
        store_data = 0;
        smem_rd_data = 0;
        smem_rd_valid = 0;

        total_checks = 0;
        passed_checks = 0;
        failed_checks = 0;

        gmem_resp_latency = 1;
        gmem_wr_latency = 1;

        #30;
        rst_n = 1;
        #20;

        $display("=== Test 1: Reset defaults ===");
        expect32("T1 ready", {31'b0, ready}, 32'd1);
        expect32("T1 pending_count", {28'b0, pending_count}, 32'd0);
        expect32("T1 tma_busy", {31'b0, tma_busy}, 32'd0);

        $display("=== Test 2: cp.async.ca single copy (4B) ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_1000, 18'h0010, 4'd4, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T2 smem data", smem_shadow[smem_idx(18'h0010)], mk_resp_data(32'h0000_1000));
        expect32("T2 req cache=CA", {29'b0, last_req_cache}, {29'b0, `CACHE_CA});
        expect32("T2 smem size", {27'b0, last_smem_size}, 32'd4);

        $display("=== Test 3: cp.async.cg cache hint path ===");
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_2000, 18'h0020, 4'd8, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T3 smem data", smem_shadow[smem_idx(18'h0020)], mk_resp_data(32'h0000_2000));
        expect32("T3 req cache=CG", {29'b0, last_req_cache}, {29'b0, `CACHE_CG});

        $display("=== Test 4: cp.async.bulk uses cache_hint ===");
        issue_cpasync_bulk(32'h0000_2100, 18'h0021, 4'd15, `CACHE_CV);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T4 smem data", smem_shadow[smem_idx(18'h0021)], mk_resp_data(32'h0000_2100));
        expect32("T4 req cache=CV", {29'b0, last_req_cache}, {29'b0, `CACHE_CV});

        $display("=== Test 5: Unaligned src/dst copy ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_3001, 18'h0033, 4'd12, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T5 smem data", smem_shadow[smem_idx(18'h0033)], mk_resp_data(32'h0000_3001));

        $display("=== Test 6: Zero-length copy ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_4000, 18'h0040, 4'd0, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect32("T6 smem size=0", {27'b0, last_smem_size}, 32'd0);

        $display("=== Test 7: Max-size copy (15B encoding) ===");
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_4100, 18'h0041, 4'd15, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect32("T7 smem size=15", {27'b0, last_smem_size}, 32'd15);

        $display("=== Test 8: Back-to-back two copies in one group ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_4200, 18'h0042, 4'd4, 3'b000);
        wait_pending_zero(400);
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_4210, 18'h0043, 4'd4, 3'b000);
        wait_pending_zero(400);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T8 copy0", smem_shadow[smem_idx(18'h0042)], mk_resp_data(32'h0000_4200));
        expect128("T8 copy1", smem_shadow[smem_idx(18'h0043)], mk_resp_data(32'h0000_4210));

        $display("=== Test 9: Pending count non-zero before wait ===");
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_5000, 18'h0050, 4'd4, 3'b000);
        expect_true("T9 pending_count non-zero", pending_count != 0);
        wait_pending_zero(400);
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_5010, 18'h0051, 4'd4, 3'b000);
        wait_pending_zero(400);
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_5020, 18'h0052, 4'd4, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T9 copy0", smem_shadow[smem_idx(18'h0050)], mk_resp_data(32'h0000_5000));
        expect128("T9 copy1", smem_shadow[smem_idx(18'h0051)], mk_resp_data(32'h0000_5010));
        expect128("T9 copy2", smem_shadow[smem_idx(18'h0052)], mk_resp_data(32'h0000_5020));
        expect32("T9 pending_count zero", {28'b0, pending_count}, 32'd0);

        $display("=== Test 10: wait_all with no pending ===");
        issue_cpasync_wait_all();
        expect32("T10 pending_count", {28'b0, pending_count}, 32'd0);

        $display("=== Test 11: wait_all with pending requests ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_6000, 18'h0060, 4'd8, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect128("T11 smem data", smem_shadow[smem_idx(18'h0060)], mk_resp_data(32'h0000_6000));
        expect32("T11 pending_count", {28'b0, pending_count}, 32'd0);

        $display("=== Test 12: st.async.global basic ===");
        issue_st_global(32'h0000_A000, 128'h1111_2222_3333_4444_5555_6666_7777_8888, 4'd15);
        issue_st_commit();
        issue_st_wait();
        expect128("T12 gmem shadow", gmem_shadow[gmem_idx(32'h0000_A000)], 128'h1111_2222_3333_4444_5555_6666_7777_8888);
        expect32("T12 wr addr", last_wr_addr, 32'h0000_A000);

        $display("=== Test 13: st.async.shared basic ===");
        issue_st_shared(18'h0070, 128'h9999_AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0001, 4'd8);
        expect128("T13 smem shared", smem_shadow[smem_idx(18'h0070)], 128'h9999_AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0001);
        expect32("T13 smem size", {27'b0, last_smem_size}, 32'd8);

        $display("=== Test 14: st.async.global zero-size ===");
        issue_st_global(32'h0000_A010, 128'h0001_0002_0003_0004_0005_0006_0007_0008, 4'd0);
        issue_st_commit();
        issue_st_wait();
        expect32("T14 wr size=0", {27'b0, last_wr_size}, 32'd0);
        expect128("T14 gmem shadow", gmem_shadow[gmem_idx(32'h0000_A010)], 128'h0001_0002_0003_0004_0005_0006_0007_0008);

        $display("=== Test 15: st.async.global back-to-back ===");
        issue_st_global(32'h0000_A020, 128'hAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111, 4'd4);
        wait_pending_zero(500);
        issue_st_global(32'h0000_A030, 128'h2222_3333_4444_5555_6666_7777_8888_9999, 4'd12);
        issue_st_commit();
        issue_st_wait();
        expect128("T15 gmem shadow0", gmem_shadow[gmem_idx(32'h0000_A020)], 128'hAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111);
        expect128("T15 gmem shadow1", gmem_shadow[gmem_idx(32'h0000_A030)], 128'h2222_3333_4444_5555_6666_7777_8888_9999);

        $display("=== Test 16: ready deasserts during wait ===");
        issue_cpasync_copy(`CPASYNC_CA, 32'h0000_7000, 18'h0080, 4'd4, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait_no_block(4'd1);
        @(posedge clk);
        expect32("T16 ready low", {31'b0, ready}, 32'd0);
        wait_done(400);
        @(posedge clk);
        expect32("T16 ready high", {31'b0, ready}, 32'd1);
        expect128("T16 smem data", smem_shadow[smem_idx(18'h0080)], mk_resp_data(32'h0000_7000));

        $display("=== Test 17: wait_count barrier across two committed groups ===");
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_7100, 18'h0081, 4'd4, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_7200, 18'h0082, 4'd4, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait(4'd2);
        expect128("T17 group0", smem_shadow[smem_idx(18'h0081)], mk_resp_data(32'h0000_7100));
        expect128("T17 group1", smem_shadow[smem_idx(18'h0082)], mk_resp_data(32'h0000_7200));

        $display("=== Test 18: stress 20 sequential copies ===");
        start_reads = gmem_read_count;
        for (i = 0; i < 20; i = i + 1) begin
            issue_cpasync_copy(`CPASYNC_CA, (32'h0000_8000 + i*16), (18'h0100 + i[17:0]), 4'd4, 3'b000);
            wait_pending_zero(400);
        end
        issue_cpasync_commit();
        issue_cpasync_wait_all();
        expect_true("T18 read_count advanced", (gmem_read_count - start_reads) >= 20);
        expect128("T18 first data", smem_shadow[smem_idx(18'h0100)], mk_resp_data(32'h0000_8000));
        expect128("T18 last data", smem_shadow[smem_idx(18'h0113)], mk_resp_data(32'h0000_8130));
        expect32("T18 pending_count", {28'b0, pending_count}, 32'd0);

        $display("=== Test 19: mixed cp.async then st.async no interference ===");
        issue_cpasync_copy(`CPASYNC_CG, 32'h0000_9000, 18'h0120, 4'd8, 3'b000);
        issue_cpasync_commit();
        issue_cpasync_wait(4'd1);
        issue_st_global(32'h0000_B000, 128'hDEAD_BEEF_CAFE_BABE_0123_4567_89AB_CDEF, 4'd15);
        issue_st_commit();
        issue_st_wait();
        expect128("T19 cp data", smem_shadow[smem_idx(18'h0120)], mk_resp_data(32'h0000_9000));
        expect128("T19 st data", gmem_shadow[gmem_idx(32'h0000_B000)], 128'hDEAD_BEEF_CAFE_BABE_0123_4567_89AB_CDEF);

        $display("=== Test 20: cp.async.bulk max-size cache CA ===");
        issue_cpasync_bulk(32'h0000_A100, 18'h0130, 4'd15, `CACHE_CA);
        issue_cpasync_commit();
        issue_cpasync_wait(4'd1);
        expect32("T20 last req size", {27'b0, last_req_size}, 32'd15);
        expect32("T20 last req cache", {29'b0, last_req_cache}, {29'b0, `CACHE_CA});

        $display("=== Test 21: TMA tensor command smoke ===");
        start_reads = gmem_read_count;
        issue_cpasync_tensor();
        repeat (20) @(posedge clk);
        expect_true("T21 tma request observed", (gmem_read_count - start_reads) > 0 || tma_busy || pending_count != 0);
        issue_cpasync_wait_all();
        expect_true("T21 tma read_count advanced", (gmem_read_count - start_reads) > 0);
        expect32("T21 pending zero", {28'b0, pending_count}, 32'd0);

        $display("=== Test 22: coverage summary sanity ===");
        expect_true("T22 smem writes > 0", smem_write_count > 0);
        expect_true("T22 gmem reads > 0", gmem_read_count > 0);
        expect_true("T22 gmem writes > 0", gmem_write_count > 0);

        $display("============================================================");
        $display("tb_async_copy_engine Summary: %0d PASSED, %0d FAILED", passed_checks, failed_checks);
        $display("============================================================");
        if (failed_checks == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $fatal(1, "SOME TESTS FAILED");
        end

        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "[TIMEOUT] tb_async_copy_engine timed out");
    end

endmodule
