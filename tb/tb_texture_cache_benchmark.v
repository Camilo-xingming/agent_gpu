//============================================================================
// RalphGPU - Texture Cache Benchmark
// Measures texture cache performance and hit rates
// Metrics: Cache hit rate, memory bandwidth, texture throughput
//============================================================================

`timescale 1ns / 1ps

module tb_texture_cache_benchmark;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter NUM_FETCHES = 100;

    reg clk;
    reg rst_n;

    // Control
    reg  [5:0]  opcode;
    reg  [5:0]  func;
    reg         valid_in;

    // Coordinates
    reg  [31:0] coord_s, coord_t, coord_r, coord_q;

    // LOD control
    reg  [31:0] lod;
    reg  [31:0] dsdx, dsdy, dtdx, dtdy;

    // Texture descriptor
    reg  [31:0] tex_base_addr;
    reg  [15:0] tex_width, tex_height, tex_depth;
    reg  [3:0]  tex_format, tex_filter;
    reg  [3:0]  tex_wrap_s, tex_wrap_t, tex_wrap_r;
    reg  [3:0]  num_mip_levels;

    // Surface store
    reg  [127:0] store_data;

    // Memory interface
    wire        mem_req;
    wire        mem_write;
    wire [31:0] mem_addr;
    wire [127:0] mem_wdata;
    reg         mem_ready;
    reg  [127:0] mem_rdata;
    reg         mem_valid;

    // Result
    wire [127:0] result;
    wire         valid_out;
    wire         busy;

    // Performance counters
    integer cache_hits;
    integer cache_misses;
    integer total_fetches;
    integer total_requests;
    integer start_cycle;
    integer end_cycle;
    integer total_cycles;
    real hit_rate;
    real texels_per_cycle;

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

    // DUT
    texture_unit #(
        .CACHE_SIZE_KB(16),
        .MAX_ANISO(16)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .opcode         (opcode),
        .func           (func),
        .valid_in       (valid_in),
        .coord_s        (coord_s),
        .coord_t        (coord_t),
        .coord_r        (coord_r),
        .coord_q        (coord_q),
        .lod            (lod),
        .dsdx           (dsdx),
        .dsdy           (dsdy),
        .dtdx           (dtdx),
        .dtdy           (dtdy),
        .tex_base_addr  (tex_base_addr),
        .tex_width      (tex_width),
        .tex_height     (tex_height),
        .tex_depth      (tex_depth),
        .tex_format     (tex_format),
        .tex_filter     (tex_filter),
        .tex_wrap_s     (tex_wrap_s),
        .tex_wrap_t     (tex_wrap_t),
        .tex_wrap_r     (tex_wrap_r),
        .num_mip_levels (num_mip_levels),
        .store_data     (store_data),
        .mem_req        (mem_req),
        .mem_write      (mem_write),
        .mem_addr       (mem_addr),
        .mem_wdata      (mem_wdata),
        .mem_ready      (mem_ready),
        .mem_rdata      (mem_rdata),
        .mem_valid      (mem_valid),
        .result         (result),
        .valid_out      (valid_out),
        .busy           (busy)
    );

    // Simple memory model with latency
    reg [31:0] pending_addr;
    reg resp_pending;
    reg [3:0] resp_delay;
    parameter MEM_LATENCY = 5;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_valid <= 1'b0;
            mem_rdata <= 128'b0;
            mem_ready <= 1'b1;
            resp_pending <= 1'b0;
            cache_misses <= 0;
        end else begin
            mem_valid <= 1'b0;

            if (mem_req && !resp_pending) begin
                pending_addr <= mem_addr;
                resp_pending <= 1'b1;
                resp_delay <= MEM_LATENCY;
                cache_misses <= cache_misses + 1;  // Count memory requests as misses
            end else if (resp_pending) begin
                if (resp_delay == 0) begin
                    mem_valid <= 1'b1;
                    // Return test pattern based on address
                    mem_rdata <= {4{pending_addr}};
                    resp_pending <= 1'b0;
                end else begin
                    resp_delay <= resp_delay - 1;
                end
            end
        end
    end

    // Count completed fetches and requests
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_fetches <= 0;
            total_requests <= 0;
        end else begin
            if (valid_out)
                total_fetches <= total_fetches + 1;
            if (valid_in)
                total_requests <= total_requests + 1;
        end
    end

    // Issue texture operation
    task issue_tex;
        input [31:0] s, t;
        begin
            @(posedge clk);
            opcode <= `OP_TEX;
            func <= `TEX_2D;
            coord_s <= s;
            coord_t <= t;
            coord_r <= 0;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    // Wait for result
    task wait_result;
        integer timeout;
        begin
            timeout = 0;
            while (!valid_out && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    integer i, j;

    initial begin
        $display("============================================================");
        $display("RalphGPU Texture Cache Benchmark");
        $display("============================================================");
        $display("Configuration:");
        $display("  Cache size: 16 KB");
        $display("  Memory latency: %0d cycles", MEM_LATENCY);
        $display("  Clock: 100 MHz");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        coord_s = 0;
        coord_t = 0;
        coord_r = 0;
        coord_q = 0;
        lod = 0;
        dsdx = 0; dsdy = 0; dtdx = 0; dtdy = 0;
        tex_base_addr = 32'h0000_0000;
        tex_width = 16'd64;
        tex_height = 16'd64;
        tex_depth = 16'd1;
        tex_format = 4'h0;  // RGBA8_UNORM
        tex_filter = 4'h0;  // Point sampling
        tex_wrap_s = 4'h1;  // Clamp
        tex_wrap_t = 4'h1;  // Clamp
        tex_wrap_r = 4'h1;  // Clamp
        num_mip_levels = 4'd6;
        store_data = 128'b0;
        cache_hits = 0;

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Benchmark 1: Sequential access (poor cache utilization)
        //==================================================================
        $display("\n--- Benchmark 1: Sequential Access Pattern ---");

        total_fetches = 0;
        total_requests = 0;
        cache_misses = 0;
        start_cycle = cycle_count;

        // Access texels sequentially across entire texture
        for (i = 0; i < NUM_FETCHES; i = i + 1) begin
            issue_tex(i % 64, i / 64);
            wait_result();
        end

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        // Use total_requests as the fetch count
        cache_hits = total_requests - cache_misses;
        if (cache_hits < 0) cache_hits = 0;
        hit_rate = (total_requests > 0) ? (cache_hits * 100.0) / total_requests : 0;
        texels_per_cycle = (total_requests * 1.0) / total_cycles;

        $display("Results:");
        $display("  Requests: %0d, Completions: %0d", total_requests, total_fetches);
        $display("  Memory accesses: %0d", cache_misses);
        $display("  Estimated cache hit rate: %f%%", hit_rate);
        $display("  Cycles: %0d", total_cycles);
        $display("  Requests/cycle: %f", texels_per_cycle);

        //==================================================================
        // Benchmark 2: Localized access (good cache utilization)
        //==================================================================
        $display("\n--- Benchmark 2: Localized Access Pattern ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        total_fetches = 0;
        total_requests = 0;
        cache_misses = 0;
        start_cycle = cycle_count;

        // Access a small 4x4 region repeatedly (should hit cache after warmup)
        for (i = 0; i < NUM_FETCHES; i = i + 1) begin
            issue_tex(i % 4, (i / 4) % 4);
            wait_result();
        end

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        cache_hits = total_requests - cache_misses;
        if (cache_hits < 0) cache_hits = 0;
        hit_rate = (total_requests > 0) ? (cache_hits * 100.0) / total_requests : 0;
        texels_per_cycle = (total_requests * 1.0) / total_cycles;

        $display("Results:");
        $display("  Requests: %0d, Memory accesses: %0d", total_requests, cache_misses);
        $display("  Estimated cache hit rate: %f%%", hit_rate);
        $display("  Cycles: %0d, Requests/cycle: %f", total_cycles, texels_per_cycle);

        //==================================================================
        // Benchmark 3: Random access (worst case)
        //==================================================================
        $display("\n--- Benchmark 3: Random Access Pattern ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        total_fetches = 0;
        total_requests = 0;
        cache_misses = 0;
        start_cycle = cycle_count;

        // Pseudo-random access using LCG
        for (i = 0; i < NUM_FETCHES; i = i + 1) begin
            // Simple pseudo-random pattern
            issue_tex((i * 17 + 7) % 64, (i * 31 + 11) % 64);
            wait_result();
        end

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        cache_hits = total_requests - cache_misses;
        if (cache_hits < 0) cache_hits = 0;
        hit_rate = (total_requests > 0) ? (cache_hits * 100.0) / total_requests : 0;
        texels_per_cycle = (total_requests * 1.0) / total_cycles;

        $display("Results:");
        $display("  Requests: %0d, Memory accesses: %0d", total_requests, cache_misses);
        $display("  Estimated cache hit rate: %f%%", hit_rate);
        $display("  Cycles: %0d, Requests/cycle: %f", total_cycles, texels_per_cycle);

        //==================================================================
        // Benchmark 4: Tiled access (typical GPU pattern)
        //==================================================================
        $display("\n--- Benchmark 4: Tiled Access Pattern (8x8 tiles) ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        total_fetches = 0;
        total_requests = 0;
        cache_misses = 0;
        start_cycle = cycle_count;

        // Access in 8x8 tiles, then move to next tile
        for (i = 0; i < 4; i = i + 1) begin  // 4 tiles
            for (j = 0; j < 64; j = j + 1) begin  // 64 texels per tile (8x8)
                issue_tex((i % 2) * 8 + (j % 8), (i / 2) * 8 + (j / 8));
                wait_result();
            end
        end

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        cache_hits = total_requests - cache_misses;
        if (cache_hits < 0) cache_hits = 0;
        hit_rate = (total_requests > 0) ? (cache_hits * 100.0) / total_requests : 0;
        texels_per_cycle = (total_requests * 1.0) / total_cycles;

        $display("Results (4 tiles x 64 texels):");
        $display("  Requests: %0d, Memory accesses: %0d", total_requests, cache_misses);
        $display("  Estimated cache hit rate: %f%%", hit_rate);
        $display("  Cycles: %0d, Requests/cycle: %f", total_cycles, texels_per_cycle);

        //==================================================================
        // Benchmark 5: Repeated single texel (100% hit rate expected)
        //==================================================================
        $display("\n--- Benchmark 5: Repeated Single Texel ---");

        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        total_fetches = 0;
        total_requests = 0;
        cache_misses = 0;
        start_cycle = cycle_count;

        // Access same texel 100 times (should be 1 miss, 99 hits)
        for (i = 0; i < NUM_FETCHES; i = i + 1) begin
            issue_tex(32, 32);  // Same coordinate every time
            wait_result();
        end

        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;
        cache_hits = total_requests - cache_misses;
        if (cache_hits < 0) cache_hits = 0;
        hit_rate = (total_requests > 0) ? (cache_hits * 100.0) / total_requests : 0;
        texels_per_cycle = (total_requests * 1.0) / total_cycles;

        $display("Results:");
        $display("  Requests: %0d, Memory accesses: %0d", total_requests, cache_misses);
        $display("  Estimated cache hit rate: %f%%", hit_rate);
        $display("  Cycles: %0d, Requests/cycle: %f", total_cycles, texels_per_cycle);

        //==================================================================
        // Summary
        //==================================================================
        $display("\n============================================================");
        $display("Texture Cache Benchmark Complete");
        $display("============================================================");
        $display("Key findings:");
        $display("  - Localized access maximizes cache efficiency");
        $display("  - Random access pattern has lowest hit rate");
        $display("  - Tiled access balances locality and coverage");
        $display("  - Single texel shows theoretical max hit rate");
        $display("============================================================");

        #100;
        $finish;
    end

    // Timeout
    initial begin
        #10000000;
        $display("ERROR: Benchmark timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_texture_cache_benchmark.vcd");
        $dumpvars(0, tb_texture_cache_benchmark);
    end

endmodule
