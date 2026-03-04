`timescale 1ns/1ps
`include "gpu_defines.vh"

module tb_async_copy_engine();

    reg clk;
    reg rst_n;

    // 指令和控制接口
    reg [5:0] opcode;
    reg [5:0] func;
    reg valid_in;
    reg [31:0] src_addr;
    reg [17:0] dst_addr;
    reg [3:0] size;
    reg [2:0] cache_hint;
    reg [3:0] wait_count;

    // TMA Interface
    reg [63:0] tensor_desc;
    reg [31:0] tensor_coord_x;
    reg [31:0] tensor_coord_y;

    // st.async Interface
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

    // Clock gen
    always #5 clk = ~clk;

    // Dummy Memory Response
    always @(posedge clk) begin
        if (!rst_n) begin
            gmem_resp_valid <= 0;
            gmem_resp_data <= 0;
            gmem_wr_done <= 0;
        end else begin
            gmem_resp_valid <= 0;
            gmem_wr_done <= 0;
            if (gmem_req_valid) begin
                gmem_resp_valid <= 1;
                gmem_resp_data <= {32'hDEADBEEF, 32'hCAFEBABE, 32'h8BADF00D, gmem_req_addr};
            end
            if (gmem_wr_valid) begin
                gmem_wr_done <= 1;
            end
        end
    end

    // Task to issue async copy
    task issue_copy(input [31:0] s_addr, input [17:0] d_addr, input [3:0] sz);
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in = 1;
            opcode = `OP_CPASYNC;
            func = `CPASYNC_CG; // CA/CG
            src_addr = s_addr;
            dst_addr = d_addr;
            size = sz;
            cache_hint = 0;
            @(posedge clk);
            valid_in = 0;
        end
    endtask

    task issue_commit();
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in = 1;
            opcode = `OP_CPASYNC;
            func = `CPASYNC_COMMIT;
            @(posedge clk);
            valid_in = 0;
        end
    endtask

    task issue_wait(input [3:0] w_count);
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in = 1;
            opcode = `OP_CPASYNC;
            func = `CPASYNC_WAIT;
            wait_count = w_count;
            @(posedge clk);
            valid_in = 0;
            while (!done) @(posedge clk);
        end
    endtask

    initial begin
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
        tensor_desc = 0;
        tensor_coord_x = 0;
        tensor_coord_y = 0;
        is_store = 0;
        store_gmem_addr = 0;
        store_data = 0;
        smem_rd_data = 0;
        smem_rd_valid = 0;

        #20;
        rst_n = 1;
        #20;

        $display("--- Test 1: Single element copy (4 bytes) ---");
        issue_copy(32'h1000, 18'h10, 4'd4);
        issue_commit();
        issue_wait(0);
        $display("Single element copy done.");

        $display("--- Test 2: Bulk copy (16 bytes) ---");
        issue_copy(32'h2000, 18'h20, 4'd15);
        issue_commit();
        issue_wait(0);
        $display("Bulk copy done.");

        $display("--- Test 3: Unaligned addresses ---");
        issue_copy(32'h3001, 18'h33, 4'd8);
        issue_commit();
        issue_wait(0);
        $display("Unaligned copy done.");

        $display("--- Test 4: Edge cases (Max size 31, Zero length) ---");
        issue_copy(32'h4000, 18'h40, 4'd15);
        issue_copy(32'h5000, 18'h50, 4'd0);
        issue_commit();
        issue_wait(0);
        $display("Edge cases copy done.");

        $display("--- Test 5: ST_ASYNC Edge Cases ---");
        @(posedge clk);
        valid_in = 1;
        opcode = `OP_ST_ASYNC;
        func = `ST_ASYNC_GLOBAL;
        store_gmem_addr = 32'hA000;
        store_data = 128'hFFFF;
        size = 4'd15;
        @(posedge clk);
        valid_in = 0;
        while (!gmem_wr_done) @(posedge clk);
        $display("ST_ASYNC Edge cases copy done.");

        #50;
        $display("All tests passed!");
        $finish;
    end

endmodule
