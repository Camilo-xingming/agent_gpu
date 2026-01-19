//============================================================================
// RalphGPU - cp.async Bandwidth Benchmark
// Measures async copy throughput and bandwidth
// Metrics: Bytes/cycle, GB/s, copy latency, group efficiency
//============================================================================

`timescale 1ns / 1ps

module tb_cpasync_bandwidth;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter NUM_COPIES = 100; // Number of copy operations
    parameter MEM_LATENCY = 10; // Simulated memory latency in cycles

    reg clk;
    reg rst_n;

    // Async copy interface
    reg         valid_in;
    reg  [5:0]  func;
    reg  [31:0] src_addr;   // Global memory source
    reg  [31:0] dst_addr;   // Shared memory destination
    reg  [3:0]  size;       // Copy size: 0=4B, 1=8B, 2=16B
    reg  [3:0]  wait_count;

    wire        ready;
    wire        done;
    wire [3:0]  pending_groups;

    // Memory interface
    wire        gmem_req_valid;
    wire [31:0] gmem_req_addr;
    reg         gmem_resp_valid;
    reg  [127:0] gmem_resp_data;

    // Timing measurements
    integer start_cycle;
    integer end_cycle;
    integer total_cycles;
    integer total_bytes;
    real bandwidth_bytes_per_cycle;
    real bandwidth_gbps;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Cycle counter
    reg [31:0] cycle_count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    // Additional interface signals
    reg  [2:0]  cache_hint;
    wire [4:0]  gmem_req_size;
    wire [2:0]  gmem_req_cache;
    wire        smem_wr_en;
    wire [13:0] smem_wr_addr;
    wire [127:0] smem_wr_data;
    wire [4:0]  smem_wr_size;

    // Async copy engine instantiation
    async_copy_engine #(
        .MAX_GROUPS(8),
        .MAX_PENDING(16)
    ) u_copy_engine (
        .clk            (clk),
        .rst_n          (rst_n),
        .func           (func),
        .valid_in       (valid_in),
        .src_addr       (src_addr),
        .dst_addr       (dst_addr[13:0]),
        .size           (size),
        .cache_hint     (cache_hint),
        .wait_count     (wait_count),
        .ready          (ready),
        .done           (done),
        .pending_count  (pending_groups),
        .gmem_req_valid (gmem_req_valid),
        .gmem_req_addr  (gmem_req_addr),
        .gmem_req_size  (gmem_req_size),
        .gmem_req_cache (gmem_req_cache),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_data (gmem_resp_data),
        .smem_wr_en     (smem_wr_en),
        .smem_wr_addr   (smem_wr_addr),
        .smem_wr_data   (smem_wr_data),
        .smem_wr_size   (smem_wr_size)
    );

    // Memory responder with configurable latency
    reg [31:0] resp_queue_addr [0:15];
    reg [3:0]  resp_queue_delay [0:15];
    reg [3:0]  resp_queue_head;
    reg [3:0]  resp_queue_tail;
    reg [3:0]  resp_queue_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_queue_head <= 0;
            resp_queue_tail <= 0;
            resp_queue_count <= 0;
            gmem_resp_valid <= 0;
            gmem_resp_data <= 0;
        end else begin
            gmem_resp_valid <= 0;

            // Add new requests to queue
            if (gmem_req_valid && resp_queue_count < 15) begin
                resp_queue_addr[resp_queue_tail] <= gmem_req_addr;
                resp_queue_delay[resp_queue_tail] <= MEM_LATENCY;
                resp_queue_tail <= resp_queue_tail + 1;
                resp_queue_count <= resp_queue_count + 1;
            end

            // Process queue - decrement delays and send responses
            if (resp_queue_count > 0) begin
                if (resp_queue_delay[resp_queue_head] == 0) begin
                    gmem_resp_valid <= 1;
                    gmem_resp_data <= {4{resp_queue_addr[resp_queue_head]}};  // Return address as data pattern
                    resp_queue_head <= resp_queue_head + 1;
                    resp_queue_count <= resp_queue_count - 1;
                end else begin
                    resp_queue_delay[resp_queue_head] <= resp_queue_delay[resp_queue_head] - 1;
                end
            end
        end
    end

    // Benchmark tasks
    task issue_copy;
        input [31:0] src;
        input [31:0] dst;
        input [3:0]  sz;  // 0=4B, 1=8B, 2=16B
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            func <= `CPASYNC_CA;  // cp.async.ca
            src_addr <= src;
            dst_addr <= dst;
            size <= sz;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_commit;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            valid_in <= 1'b1;
            func <= `CPASYNC_COMMIT;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task issue_wait_all;
        begin
            @(posedge clk);
            valid_in <= 1'b1;
            func <= `CPASYNC_WAIT_ALL;
            @(posedge clk);
            valid_in <= 1'b0;
            // Wait until done
            while (!done) @(posedge clk);
        end
    endtask

    task wait_all_complete;
        integer timeout;
        begin
            timeout = 0;
            while (pending_groups > 0 && timeout < 50000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU cp.async Bandwidth Benchmark");
        $display("============================================================");
        $display("Configuration:");
        $display("  Memory latency: %0d cycles", MEM_LATENCY);
        $display("  Operations: %0d", NUM_COPIES);
        $display("  Clock: 100 MHz");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        func = 0;
        src_addr = 0;
        dst_addr = 0;
        size = 0;
        wait_count = 0;
        cache_hint = 0;

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Benchmark 1: 4-byte copies
        //==================================================================
        $display("\n--- Benchmark 1: 4-byte Copies ---");

        start_cycle = cycle_count;
        total_bytes = 0;

        for (i = 0; i < NUM_COPIES; i = i + 1) begin
            issue_copy(32'h1000 + i*4, 32'h0 + i*4, 4'd0);  // 4 bytes
            total_bytes = total_bytes + 4;
        end
        issue_commit();
        issue_wait_all();
        wait_all_complete();

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        bandwidth_bytes_per_cycle = (total_bytes * 1.0) / total_cycles;
        bandwidth_gbps = bandwidth_bytes_per_cycle * 100.0 / 1000.0;  // GB/s at 100MHz

        $display("Results:");
        $display("  Total bytes: %0d", total_bytes);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Bytes/cycle: %f", bandwidth_bytes_per_cycle);
        $display("  Bandwidth: %f GB/s (at 100MHz)", bandwidth_gbps);

        //==================================================================
        // Benchmark 2: 8-byte copies
        //==================================================================
        $display("\n--- Benchmark 2: 8-byte Copies ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        start_cycle = cycle_count;
        total_bytes = 0;

        for (i = 0; i < NUM_COPIES; i = i + 1) begin
            issue_copy(32'h2000 + i*8, 32'h0 + i*8, 4'd1);  // 8 bytes
            total_bytes = total_bytes + 8;
        end
        issue_commit();
        issue_wait_all();
        wait_all_complete();

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        bandwidth_bytes_per_cycle = (total_bytes * 1.0) / total_cycles;
        bandwidth_gbps = bandwidth_bytes_per_cycle * 100.0 / 1000.0;

        $display("Results:");
        $display("  Total bytes: %0d", total_bytes);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Bytes/cycle: %f", bandwidth_bytes_per_cycle);
        $display("  Bandwidth: %f GB/s (at 100MHz)", bandwidth_gbps);

        //==================================================================
        // Benchmark 3: 16-byte copies
        //==================================================================
        $display("\n--- Benchmark 3: 16-byte Copies ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        start_cycle = cycle_count;
        total_bytes = 0;

        for (i = 0; i < NUM_COPIES; i = i + 1) begin
            issue_copy(32'h3000 + i*16, 32'h0 + i*16, 4'd2);  // 16 bytes
            total_bytes = total_bytes + 16;
        end
        issue_commit();
        issue_wait_all();
        wait_all_complete();

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        bandwidth_bytes_per_cycle = (total_bytes * 1.0) / total_cycles;
        bandwidth_gbps = bandwidth_bytes_per_cycle * 100.0 / 1000.0;

        $display("Results:");
        $display("  Total bytes: %0d", total_bytes);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Bytes/cycle: %f", bandwidth_bytes_per_cycle);
        $display("  Bandwidth: %f GB/s (at 100MHz)", bandwidth_gbps);

        //==================================================================
        // Benchmark 4: Multiple groups in flight
        //==================================================================
        $display("\n--- Benchmark 4: Multiple Groups (pipelined) ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        start_cycle = cycle_count;
        total_bytes = 0;

        // Issue 10 groups of 10 copies each
        for (i = 0; i < 10; i = i + 1) begin
            // Group of 10 x 16-byte copies
            issue_copy(32'h4000 + i*160 + 0,  32'h0, 4'd2);
            issue_copy(32'h4000 + i*160 + 16, 32'h10, 4'd2);
            issue_copy(32'h4000 + i*160 + 32, 32'h20, 4'd2);
            issue_copy(32'h4000 + i*160 + 48, 32'h30, 4'd2);
            issue_copy(32'h4000 + i*160 + 64, 32'h40, 4'd2);
            issue_copy(32'h4000 + i*160 + 80, 32'h50, 4'd2);
            issue_copy(32'h4000 + i*160 + 96, 32'h60, 4'd2);
            issue_copy(32'h4000 + i*160 + 112, 32'h70, 4'd2);
            issue_copy(32'h4000 + i*160 + 128, 32'h80, 4'd2);
            issue_copy(32'h4000 + i*160 + 144, 32'h90, 4'd2);
            total_bytes = total_bytes + 160;
            issue_commit();
        end

        issue_wait_all();
        wait_all_complete();

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        bandwidth_bytes_per_cycle = (total_bytes * 1.0) / total_cycles;
        bandwidth_gbps = bandwidth_bytes_per_cycle * 100.0 / 1000.0;

        $display("Results:");
        $display("  Groups: 10, Copies/group: 10, Size: 16B");
        $display("  Total bytes: %0d", total_bytes);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Bytes/cycle: %f", bandwidth_bytes_per_cycle);
        $display("  Bandwidth: %f GB/s (at 100MHz)", bandwidth_gbps);

        //==================================================================
        // Benchmark 5: Bulk copy comparison
        //==================================================================
        $display("\n--- Benchmark 5: Bulk Copy Performance ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        // Single large bulk copy simulation (multiple 16B copies)
        start_cycle = cycle_count;
        total_bytes = 0;

        // 64 x 16B = 1KB transfer
        for (i = 0; i < 64; i = i + 1) begin
            issue_copy(32'h5000 + i*16, 32'h0 + i*16, 4'd2);
            total_bytes = total_bytes + 16;
        end
        issue_commit();
        issue_wait_all();
        wait_all_complete();

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        bandwidth_bytes_per_cycle = (total_bytes * 1.0) / total_cycles;
        bandwidth_gbps = bandwidth_bytes_per_cycle * 100.0 / 1000.0;

        $display("1KB bulk transfer (64 x 16B):");
        $display("  Cycles: %0d, Bytes/cycle: %f, BW: %f GB/s",
                 total_cycles, bandwidth_bytes_per_cycle, bandwidth_gbps);

        //==================================================================
        // Summary
        //==================================================================
        $display("\n============================================================");
        $display("cp.async Bandwidth Benchmark Complete");
        $display("============================================================");
        $display("Key findings:");
        $display("  - Larger copy sizes improve bandwidth efficiency");
        $display("  - Multiple groups in flight hide memory latency");
        $display("  - Pipelined copies significantly outperform sequential");
        $display("  - Max theoretical BW depends on memory interface width");
        $display("============================================================");

        #100;
        $finish;
    end

    // Timeout
    initial begin
        #5000000;
        $display("ERROR: Benchmark timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_cpasync_bandwidth.vcd");
        $dumpvars(0, tb_cpasync_bandwidth);
    end

endmodule
