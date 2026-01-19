//============================================================================
// RalphGPU - WGMMA Throughput Benchmark
// Measures WGMMA operations per second and computational throughput
// Metrics: Operations/cycle, TFLOPS estimate, latency hiding efficiency
//============================================================================

`timescale 1ns / 1ps

module tb_wgmma_throughput;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter NUM_OPS = 100;    // Number of WGMMA operations to benchmark
    parameter MAX_PENDING_OPS = 8;

    // Matrix dimensions for throughput calculation
    // M64N8K16: 64*8*16*2 = 16384 FLOPs per op (mul + add)
    parameter M = 64;
    parameter N = 8;
    parameter K = 16;
    parameter FLOPS_PER_OP = M * N * K * 2;  // Multiply-accumulate = 2 FLOPs

    reg clk;
    reg rst_n;

    // WGMMA interface
    reg  [5:0]    func;
    reg           valid_in;
    reg  [2:0]    warpgroup_id;
    reg  [3:0]    wait_count;
    reg  [63:0]   desc_a;
    reg  [63:0]   desc_b;
    reg  [31:0]   scale_d;
    reg  [511:0]  data_a;
    reg  [511:0]  data_b;
    reg  [1023:0] accum_in;
    wire [1023:0] accum_out;
    wire          ready;
    wire          done;
    wire [3:0]    pending_ops;

    // Timing measurements
    integer start_cycle;
    integer end_cycle;
    integer total_cycles;
    integer ops_issued;
    real throughput_ops_per_cycle;
    real throughput_tflops;
    real efficiency;

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

    // WGMMA Unit instantiation
    wgmma #(
        .WARPGROUP_SIZE(4),
        .THREADS_PER_WARP(32),
        .MAX_PENDING_OPS(MAX_PENDING_OPS)
    ) u_wgmma (
        .clk            (clk),
        .rst_n          (rst_n),
        .func           (func),
        .valid_in       (valid_in),
        .warpgroup_id   (warpgroup_id),
        .wait_count     (wait_count),
        .desc_a         (desc_a),
        .desc_b         (desc_b),
        .scale_d        (scale_d),
        .data_a         (data_a),
        .data_b         (data_b),
        .accum_in       (accum_in),
        .accum_out      (accum_out),
        .ready          (ready),
        .done           (done),
        .pending_ops    (pending_ops)
    );

    // Done counter (track completed ops)
    reg [31:0] done_count;
    reg done_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_count <= 0;
            done_prev <= 0;
        end else begin
            done_prev <= done;
            if (done && !done_prev)  // Rising edge of done
                done_count <= done_count + 1;
        end
    end

    // Benchmark tasks
    task issue_wgmma_op;
        input [2:0] wg_id;
        input [5:0] op_func;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            func <= op_func;
            warpgroup_id <= wg_id;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_all_complete;
        integer timeout;
        begin
            timeout = 0;
            while (pending_ops > 0 && timeout < 50000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 50000) begin
                $display("TIMEOUT waiting for operations to complete");
            end
        end
    endtask

    // Setup test data
    task setup_test_data;
        integer i;
        begin
            // Simple incrementing pattern for matrix A and B
            for (i = 0; i < 32; i = i + 1) begin
                data_a[i*16 +: 16] <= i + 1;
                data_b[i*16 +: 16] <= i + 1;
            end
            accum_in <= 1024'b0;
        end
    endtask

    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU WGMMA Throughput Benchmark");
        $display("============================================================");
        $display("Configuration:");
        $display("  Matrix size: M%0dN%0dK%0d", M, N, K);
        $display("  Operations: %0d", NUM_OPS);
        $display("  FLOPs per op: %0d", FLOPS_PER_OP);
        $display("  Clock: 100 MHz");
        $display("  Max pending ops: %0d", MAX_PENDING_OPS);
        $display("============================================================");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        func = 0;
        warpgroup_id = 0;
        wait_count = 0;
        desc_a = 64'h0000_0010_0000_1000;
        desc_b = 64'h0000_0010_0000_2000;
        scale_d = 32'h3F800000;
        data_a = 512'b0;
        data_b = 512'b0;
        accum_in = 1024'b0;

        #100;
        rst_n = 1;
        #50;

        setup_test_data();

        //==================================================================
        // Benchmark 1: Single Warpgroup Sequential
        //==================================================================
        $display("\n--- Benchmark 1: Single Warpgroup Sequential ---");

        done_count = 0;
        start_cycle = cycle_count;
        ops_issued = 0;

        for (i = 0; i < NUM_OPS; i = i + 1) begin
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
            ops_issued = ops_issued + 1;
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        throughput_ops_per_cycle = (ops_issued * 1.0) / total_cycles;
        throughput_tflops = (ops_issued * FLOPS_PER_OP * 100.0) / (total_cycles * 1000000.0);

        $display("Results:");
        $display("  Ops issued: %0d", ops_issued);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Ops/cycle: %f", throughput_ops_per_cycle);
        $display("  Throughput: %f TFLOPS (at 100MHz)", throughput_tflops);
        $display("  Avg latency: %f cycles/op", total_cycles * 1.0 / ops_issued);

        //==================================================================
        // Benchmark 2: Multi-Warpgroup Round-Robin
        //==================================================================
        $display("\n--- Benchmark 2: Multi-Warpgroup Round-Robin (4 WGs) ---");

        // Reset
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        done_count = 0;
        start_cycle = cycle_count;
        ops_issued = 0;

        // Issue to 4 warpgroups in round-robin
        for (i = 0; i < NUM_OPS; i = i + 1) begin
            issue_wgmma_op(i[1:0], `WGMMA_M64N8K16);
            ops_issued = ops_issued + 1;
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        throughput_ops_per_cycle = (ops_issued * 1.0) / total_cycles;
        throughput_tflops = (ops_issued * FLOPS_PER_OP * 100.0) / (total_cycles * 1000000.0);

        $display("Results:");
        $display("  Ops issued: %0d", ops_issued);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Ops/cycle: %f", throughput_ops_per_cycle);
        $display("  Throughput: %f TFLOPS (at 100MHz)", throughput_tflops);
        $display("  Avg latency: %f cycles/op", total_cycles * 1.0 / ops_issued);

        //==================================================================
        // Benchmark 3: Burst Issue (Pipeline Saturation)
        //==================================================================
        $display("\n--- Benchmark 3: Burst Issue (fill pending queue) ---");

        // Reset
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        done_count = 0;
        start_cycle = cycle_count;
        ops_issued = 0;

        // Issue as fast as possible
        for (i = 0; i < NUM_OPS; i = i + 1) begin
            @(posedge clk);
            if (ready && pending_ops < MAX_PENDING_OPS) begin
                func <= `WGMMA_M64N8K16;
                warpgroup_id <= i[1:0];
                valid_in <= 1'b1;
                ops_issued = ops_issued + 1;
            end else begin
                valid_in <= 1'b0;
                // Wait for space in queue
                while (!ready || pending_ops >= MAX_PENDING_OPS - 1) begin
                    @(posedge clk);
                    valid_in <= 1'b0;
                end
                // Retry
                func <= `WGMMA_M64N8K16;
                warpgroup_id <= i[1:0];
                valid_in <= 1'b1;
                ops_issued = ops_issued + 1;
            end
            @(posedge clk);
            valid_in <= 1'b0;
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        throughput_ops_per_cycle = (ops_issued * 1.0) / total_cycles;
        throughput_tflops = (ops_issued * FLOPS_PER_OP * 100.0) / (total_cycles * 1000000.0);
        efficiency = throughput_ops_per_cycle * 100.0;

        $display("Results:");
        $display("  Ops issued: %0d", ops_issued);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Ops/cycle: %f", throughput_ops_per_cycle);
        $display("  Throughput: %f TFLOPS (at 100MHz)", throughput_tflops);
        $display("  Pipeline efficiency: %f%%", efficiency);

        //==================================================================
        // Benchmark 4: Commit/Wait Group Pattern
        //==================================================================
        $display("\n--- Benchmark 4: Commit/Wait Group Pattern ---");

        // Reset
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        done_count = 0;
        start_cycle = cycle_count;
        ops_issued = 0;

        // Issue in groups of 4, commit, wait
        for (i = 0; i < 25; i = i + 1) begin  // 25 groups of 4 = 100 ops
            // Issue 4 MMA ops
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
            ops_issued = ops_issued + 4;

            // Commit group
            issue_wgmma_op(3'd0, `WGMMA_COMMIT_GROUP);

            // Wait for group to complete
            wait_count <= 4'd0;  // Wait for all
            issue_wgmma_op(3'd0, `WGMMA_WAIT_GROUP);
        end

        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        throughput_ops_per_cycle = (ops_issued * 1.0) / total_cycles;
        throughput_tflops = (ops_issued * FLOPS_PER_OP * 100.0) / (total_cycles * 1000000.0);

        $display("Results (with sync overhead):");
        $display("  MMA ops issued: %0d", ops_issued);
        $display("  Total cycles: %0d", total_cycles);
        $display("  Effective ops/cycle: %f", throughput_ops_per_cycle);
        $display("  Effective throughput: %f TFLOPS", throughput_tflops);

        //==================================================================
        // Benchmark 5: Different Matrix Sizes
        //==================================================================
        $display("\n--- Benchmark 5: Matrix Size Comparison ---");

        // M64N8K16 (small)
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        done_count = 0;
        start_cycle = cycle_count;

        for (i = 0; i < 50; i = i + 1) begin
            issue_wgmma_op(3'd0, `WGMMA_M64N8K16);
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        // M64N8K16: 64*8*16*2 = 16384 FLOPs
        throughput_tflops = (50.0 * 16384 * 100.0) / (total_cycles * 1000000.0);

        $display("M64N8K16 (50 ops):");
        $display("  Cycles: %0d, Throughput: %f TFLOPS", total_cycles, throughput_tflops);

        // M64N64K16 (medium)
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        start_cycle = cycle_count;

        for (i = 0; i < 50; i = i + 1) begin
            issue_wgmma_op(3'd0, `WGMMA_M64N64K16);
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        // M64N64K16: 64*64*16*2 = 131072 FLOPs
        throughput_tflops = (50.0 * 131072 * 100.0) / (total_cycles * 1000000.0);

        $display("M64N64K16 (50 ops):");
        $display("  Cycles: %0d, Throughput: %f TFLOPS", total_cycles, throughput_tflops);

        // M64N128K16 (large)
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;
        setup_test_data();

        start_cycle = cycle_count;

        for (i = 0; i < 50; i = i + 1) begin
            issue_wgmma_op(3'd0, `WGMMA_M64N128K16);
        end

        wait_all_complete();
        end_cycle = cycle_count;

        total_cycles = end_cycle - start_cycle;
        // M64N128K16: 64*128*16*2 = 262144 FLOPs
        throughput_tflops = (50.0 * 262144 * 100.0) / (total_cycles * 1000000.0);

        $display("M64N128K16 (50 ops):");
        $display("  Cycles: %0d, Throughput: %f TFLOPS", total_cycles, throughput_tflops);

        //==================================================================
        // Summary
        //==================================================================
        $display("\n============================================================");
        $display("WGMMA Throughput Benchmark Complete");
        $display("============================================================");
        $display("Key findings:");
        $display("  - Single WG throughput establishes baseline");
        $display("  - Multi-WG round-robin shows warpgroup parallelism");
        $display("  - Burst issue shows pipeline saturation limit");
        $display("  - Commit/wait pattern shows synchronization overhead");
        $display("  - Larger matrices increase compute density");
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
        $dumpfile("tb_wgmma_throughput.vcd");
        $dumpvars(0, tb_wgmma_throughput);
    end

endmodule
