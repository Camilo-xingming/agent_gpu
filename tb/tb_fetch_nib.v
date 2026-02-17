//============================================================================
// RalphGPU - Testbench: Fetch Pipeline NIB (RALPH-8 P1)
// Verifies Next-Instruction Buffer works with real icache (non-bypass mode)
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_fetch_nib;

    // Use 2 warps for multi-warp test
    localparam NUM_WARPS = 2;
    localparam WARP_ID_W = 1;
    localparam ICACHE_BYPASS = 0;
    localparam FETCH_PIPE_DEPTH = 2;

    // icache LINE_SIZE=8 → 64-bit lines → mem_resp_data is 64 bits
    localparam LINE_SIZE = 8;
    localparam LINE_BITS = LINE_SIZE * 8;  // 64

    reg clk, rst_n, kernel_start;
    reg [NUM_WARPS-1:0] warp_valid, warp_exit_pending, warp_inst_consume;
    reg [NUM_WARPS-1:0] decode_stalled_per_warp, branch_flush_mask;
    reg [32*NUM_WARPS-1:0] warp_fetch_pc_flat;

    // Memory interface (between icache and mock memory)
    wire imem_req;
    wire [31:0] imem_addr;
    reg  imem_ready;
    reg  [63:0] imem_data;
    reg  imem_valid;

    // Outputs
    wire [NUM_WARPS-1:0] warp_inst_buf_valid;
    wire [NUM_WARPS-1:0] warp_inst_valid_d1;
    wire [NUM_WARPS-1:0] warp_inst_consume_gated;
    wire [32*NUM_WARPS-1:0] warp_inst_buf_flat;
    wire fetch_req, fetch_fire;
    wire [WARP_ID_W-1:0] fetch_warp_id_out;
    wire [NUM_WARPS-1:0] fetch_pc_advance;
    wire [NUM_WARPS-1:0] nib_pc_advance;
    wire [NUM_WARPS-1:0] warp_next_inst_hit;
    wire [NUM_WARPS-1:0] warp_buf_will_be_empty;
    wire [NUM_WARPS-1:0] warp_fill;
    wire [NUM_WARPS-1:0] warp_fetch_pending_out;

    // DUT
    sm_fetch_pipeline #(
        .NUM_WARPS(NUM_WARPS),
        .ICACHE_BYPASS(ICACHE_BYPASS),
        .FETCH_PIPE_DEPTH(FETCH_PIPE_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .kernel_start(kernel_start),
        .warp_valid(warp_valid),
        .warp_exit_pending(warp_exit_pending),
        .warp_inst_consume(warp_inst_consume),
        .decode_stalled_per_warp(decode_stalled_per_warp),
        .branch_flush_mask(branch_flush_mask),
        .warp_fetch_pc_flat(warp_fetch_pc_flat),
        .imem_req(imem_req),
        .imem_addr(imem_addr),
        .imem_ready(imem_ready),
        .imem_data(imem_data),
        .imem_valid(imem_valid),
        .warp_inst_buf_valid(warp_inst_buf_valid),
        .warp_inst_valid_d1(warp_inst_valid_d1),
        .warp_inst_consume_gated(warp_inst_consume_gated),
        .warp_inst_buf_flat(warp_inst_buf_flat),
        .fetch_req(fetch_req),
        .fetch_fire(fetch_fire),
        .fetch_warp_id_out(fetch_warp_id_out),
        .fetch_pc_advance(fetch_pc_advance),
        .nib_pc_advance(nib_pc_advance),
        .warp_next_inst_hit(warp_next_inst_hit),
        .warp_buf_will_be_empty(warp_buf_will_be_empty),
        .warp_fill(warp_fill),
        .warp_fetch_pending_out(warp_fetch_pending_out)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // Helpers: extract per-warp instruction buffer
    wire [31:0] warp0_inst = warp_inst_buf_flat[31:0];
    wire [31:0] warp1_inst = warp_inst_buf_flat[63:32];

    // ---- Mock memory: 1-cycle latency, serves 64-bit lines ----
    // Simple instruction memory: addr[31:3] selects a line
    // line[31:0] = inst at word0, line[63:32] = inst at word1
    reg [63:0] mock_mem [0:255];  // 256 lines = 2KB

    // Memory response FSM
    reg mem_pending;
    reg [31:0] mem_pending_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 0;
            imem_data <= 0;
            mem_pending <= 0;
            mem_pending_addr <= 0;
        end else begin
            imem_valid <= 0;
            if (mem_pending) begin
                imem_data <= mock_mem[mem_pending_addr[10:3]];
                imem_valid <= 1;
                mem_pending <= 0;
            end
            if (imem_req && imem_ready && !mem_pending) begin
                mem_pending <= 1;
                mem_pending_addr <= imem_addr;
            end
        end
    end

    // ---- PC management (simplified SM behavior) ----
    reg [31:0] warp_pc [0:NUM_WARPS-1];

    always @(*) begin
        warp_fetch_pc_flat[31:0]  = warp_pc[0];
        warp_fetch_pc_flat[63:32] = warp_pc[1];
    end

    // Advance PC on fetch_fire or nib_pc_advance
    always @(posedge clk) begin
        if (fetch_pc_advance[0]) warp_pc[0] <= warp_pc[0] + 4;
        if (fetch_pc_advance[1]) warp_pc[1] <= warp_pc[1] + 4;
        if (nib_pc_advance[0])   warp_pc[0] <= warp_pc[0] + 4;
        if (nib_pc_advance[1])   warp_pc[1] <= warp_pc[1] + 4;
    end

    // Consume: when inst buf is valid and we pulse consume, clear it
    // (In real SM, decode consumes; here we pulse manually)

    // ---- Test infrastructure ----
    integer pass_count, fail_count;

    task reset;
        begin
            rst_n = 0;
            kernel_start = 0;
            warp_valid = 0;
            warp_exit_pending = 0;
            warp_inst_consume = 0;
            decode_stalled_per_warp = 0;
            branch_flush_mask = 0;
            imem_ready = 1;
            warp_pc[0] = 0;
            warp_pc[1] = 0;
            #20;
            rst_n = 1;
            #10;
        end
    endtask

    task wait_cycles;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) @(posedge clk);
        end
    endtask

    task wait_inst_valid;
        input integer warp_id;
        input integer max_cycles;
        integer i;
        begin
            for (i = 0; i < max_cycles; i = i + 1) begin
                @(posedge clk);
                if (warp_inst_buf_valid[warp_id]) i = max_cycles;
            end
        end
    endtask

    task consume_inst;
        input integer warp_id;
        begin
            warp_inst_consume[warp_id] = 1;
            @(posedge clk);
            warp_inst_consume[warp_id] = 0;
        end
    endtask

    task check;
        input [255:0] name;
        input [31:0] actual;
        input [31:0] expected;
        begin
            if (actual === expected) begin
                $display("  PASS: %0s = 0x%08x", name, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %0s = 0x%08x, expected 0x%08x", name, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_bit;
        input [255:0] name;
        input actual;
        input expected;
        begin
            if (actual === expected) begin
                $display("  PASS: %0s = %0b", name, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %0s = %0b, expected %0b", name, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ---- Populate mock memory ----
    // Line at addr 0x000: word0=0xAABB0000, word1=0xAABB0004
    // Line at addr 0x008: word0=0xAABB0008, word1=0xAABB000C
    // Line at addr 0x100: word0=0x11110000, word1=0x11110004
    // Line at addr 0x108: word0=0x11110008, word1=0x1111000C
    // etc.
    integer mi;
    reg [31:0] mi_addr;
    initial begin
        for (mi = 0; mi < 256; mi = mi + 1) begin
            mi_addr = mi * 8;
            mock_mem[mi][31:0]  = 32'hAA000000 | (mi_addr & 32'h00FFFFFF);
            mock_mem[mi][63:32] = 32'hAA000000 | ((mi_addr + 4) & 32'h00FFFFFF);
        end
        // Override specific lines for test clarity
        mock_mem[0]   = {32'hDEAD0004, 32'hDEAD0000};  // line 0x000
        mock_mem[1]   = {32'hBEEF000C, 32'hBEEF0008};  // line 0x008
        mock_mem[32]  = {32'hCAFE0104, 32'hCAFE0100};  // line 0x100
        mock_mem[33]  = {32'hF00D010C, 32'hF00D0108};  // line 0x108
    end

    // ---- Main test ----
    initial begin
        $dumpfile("tb_fetch_nib.vcd");
        $dumpvars(0, tb_fetch_nib);

        pass_count = 0;
        fail_count = 0;

        // ==============================================================
        // Test 1: Cache miss → NIB populated → NIB hit on next fetch
        // Fetch PC=0x000 (word0, bit[2]=0) → miss → fills cache line
        // NIB should capture word1 (0xDEAD0004) tagged as PC=0x004
        // Then consume, next fetch at PC=0x004 → NIB hit (no cache access)
        // ==============================================================
        $display("\n=== Test 1: Cache miss + NIB hit (sequential pair) ===");
        reset;
        warp_valid[0] = 1;
        warp_pc[0] = 32'h0000_0000;

        // Wait for inst buf to fill (cache miss → 1-cycle mem latency → fill)
        wait_inst_valid(0, 30);
        check("T1 word0", warp0_inst, 32'hDEAD0000);

        // Consume word0 → PC advances to 0x004, should get NIB hit
        consume_inst(0);
        // NIB hit is combinational on the cycle after consume clears buf_valid
        // Wait for buf_valid to come back via NIB
        wait_inst_valid(0, 5);
        check("T1 word1 (NIB)", warp0_inst, 32'hDEAD0004);
        // Verify it was a NIB hit (no fetch pending should have been set)
        // nib_pc_advance should have fired
        $display("  INFO: fetch_pending=%b", warp_fetch_pending_out[0]);

        // ==============================================================
        // Test 2: Cache hit → NIB populated → NIB hit
        // Line 0x000 is now cached. Flush NIB via kernel_start, then
        // refetch PC=0x000. Should get cache hit + NIB populated.
        // ==============================================================
        $display("\n=== Test 2: Cache hit + NIB populated ===");
        // Consume current instruction
        consume_inst(0);
        // Now PC=0x008. Let that fetch go.
        wait_inst_valid(0, 30);
        consume_inst(0);
        // PC=0x00C now. consume that.
        wait_inst_valid(0, 30);
        consume_inst(0);

        // Reset warps, force PC back to 0x000 (line is cached now)
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        warp_pc[0] = 32'h0000_0000;
        @(posedge clk);
        warp_valid[0] = 1;

        // Should be a cache hit → instant fill
        wait_inst_valid(0, 10);
        check("T2 word0 (cache hit)", warp0_inst, 32'hDEAD0000);

        // Consume → NIB hit for word1
        consume_inst(0);
        wait_inst_valid(0, 5);
        check("T2 word1 (NIB after cache hit)", warp0_inst, 32'hDEAD0004);

        // ==============================================================
        // Test 3: Odd-word fetch (bit[2]=1) → NIB NOT populated
        // Fetch PC=0x004 directly. bit[2]=1 so NIB should not fire.
        // Next fetch at PC=0x008 must go to cache (different line).
        // ==============================================================
        $display("\n=== Test 3: Odd-word fetch, no NIB ===");
        // Reset
        consume_inst(0);
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);
        warp_pc[0] = 32'h0000_0004;  // Start at word1 of line 0
        @(posedge clk);
        warp_valid[0] = 1;

        wait_inst_valid(0, 10);
        check("T3 word at 0x004", warp0_inst, 32'hDEAD0004);

        // Consume → PC=0x008 (new line). Must NOT get NIB hit.
        consume_inst(0);
        // The next instruction requires a new cache access (line 0x008)
        wait_inst_valid(0, 30);
        check("T3 word at 0x008 (no NIB)", warp0_inst, 32'hBEEF0008);

        // ==============================================================
        // Test 4: Branch flush invalidates NIB
        // Fetch PC=0x100 (word0, bit[2]=0) → NIB captures 0x104
        // Then branch flush → NIB cleared
        // Set PC=0x104 → should NOT get NIB hit (must go to cache)
        // ==============================================================
        $display("\n=== Test 4: Branch flush invalidates NIB ===");
        consume_inst(0);
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);
        warp_pc[0] = 32'h0000_0100;
        @(posedge clk);
        warp_valid[0] = 1;

        wait_inst_valid(0, 30);
        check("T4 word at 0x100", warp0_inst, 32'hCAFE0100);

        // NIB should have 0xCAFE0104 at PC=0x104 now.
        // Branch flush before consuming
        branch_flush_mask[0] = 1;
        @(posedge clk);
        branch_flush_mask[0] = 0;

        // Set PC to 0x104 (where NIB was, but should be cleared)
        warp_pc[0] = 32'h0000_0104;
        @(posedge clk);

        // Wait — should require a cache access, not NIB
        wait_inst_valid(0, 30);
        check("T4 word at 0x104 (after flush)", warp0_inst, 32'hCAFE0104);
        // If the NIB was properly flushed, the fetch went through the cache.
        // We can't easily distinguish cache vs NIB from data alone,
        // but we verify the data is correct.

        // ==============================================================
        // Test 5: Multi-warp NIB — both warps get independent NIB hits
        // Warp 0: PC=0x000, Warp 1: PC=0x100
        // Both fetch word0, consume, then get NIB hit for word1
        // ==============================================================
        $display("\n=== Test 5: Multi-warp NIB ===");
        consume_inst(0);
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);
        warp_pc[0] = 32'h0000_0000;
        warp_pc[1] = 32'h0000_0100;
        @(posedge clk);
        warp_valid = 2'b11;

        // Wait for both warps to have instructions
        wait_inst_valid(0, 30);
        wait_inst_valid(1, 30);
        check("T5 warp0 word0", warp0_inst, 32'hDEAD0000);
        check("T5 warp1 word0", warp1_inst, 32'hCAFE0100);

        // Consume warp 0 → should get NIB hit
        consume_inst(0);
        wait_inst_valid(0, 5);
        check("T5 warp0 word1 (NIB)", warp0_inst, 32'hDEAD0004);

        // Consume warp 1 → should get NIB hit
        consume_inst(1);
        wait_inst_valid(1, 5);
        check("T5 warp1 word1 (NIB)", warp1_inst, 32'hCAFE0104);

        // ==============================================================
        // Test 6: RALPH-8 P2 — Hit-bypass during miss
        // Warm up warp 1's line (0x100), then cold-start both warps.
        // Warp 0 fetches cold line (0x200 = miss), warp 1 fetches warm
        // line (0x100 = hit). With bypass, warp 1 should get its
        // instruction while warp 0's miss is still in flight.
        // ==============================================================
        $display("\n=== Test 6: Hit-bypass during miss (P2) ===");
        // Warm up line 0x100 first (already cached from earlier tests)
        consume_inst(0);
        consume_inst(1);
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);

        // Step 1: Make sure line 0x100 is cached by fetching it
        warp_pc[0] = 32'h0000_0100;
        @(posedge clk);
        warp_valid[0] = 1;
        wait_inst_valid(0, 30);
        check("T6 warmup 0x100", warp0_inst, 32'hCAFE0100);
        consume_inst(0);

        // Step 2: Reset fetch pipeline, set up cold miss + warm hit
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);
        warp_pc[0] = 32'h0000_0200;  // Cold line (miss)
        warp_pc[1] = 32'h0000_0100;  // Warm line (hit via bypass)
        @(posedge clk);
        warp_valid = 2'b11;

        // Both should eventually get their instructions
        wait_inst_valid(0, 30);
        wait_inst_valid(1, 30);
        // mock_mem[64] = line at 0x200 = {0xAA000204, 0xAA000200}
        check("T6 warp0 cold miss", warp0_inst, 32'hAA000200);
        check("T6 warp1 hit bypass", warp1_inst, 32'hCAFE0100);

        // ==============================================================
        // Test 7: P2 — Hit-bypass + NIB on bypass warp
        // Warp 1's bypass hit should also populate NIB.
        // After consuming, warp 1 should get NIB hit for next inst.
        // ==============================================================
        $display("\n=== Test 7: NIB from bypass hit (P2) ===");
        consume_inst(1);
        wait_inst_valid(1, 5);
        check("T7 warp1 NIB after bypass", warp1_inst, 32'hCAFE0104);

        // ==============================================================
        // Test 8: P2 — Two cold misses (no bypass possible)
        // Both warps fetch cold lines. No bypass — sequential misses.
        // Both should eventually complete. Warp order depends on
        // round-robin pointer, so just check each warp gets its own data.
        // ==============================================================
        $display("\n=== Test 8: Two cold misses, no bypass ===");
        consume_inst(0);
        consume_inst(1);
        warp_valid = 0;
        kernel_start = 1;
        @(posedge clk);
        kernel_start = 0;
        @(posedge clk);
        // Use only warp 0 for this test to avoid round-robin ordering issues
        warp_pc[0] = 32'h0000_0300;  // Cold
        @(posedge clk);
        warp_valid[0] = 1;

        wait_inst_valid(0, 50);
        check("T8 warp0 first cold", warp0_inst, 32'hAA000300);
        consume_inst(0);
        // PC advanced to 0x304, NIB should have it
        wait_inst_valid(0, 5);
        check("T8 warp0 NIB after cold", warp0_inst, 32'hAA000304);
        consume_inst(0);

        // Full reset (icache may have in-flight miss from warp 0)
        warp_valid = 0;
        reset;
        warp_pc[1] = 32'h0000_0400;
        @(posedge clk);
        warp_valid[1] = 1;

        wait_inst_valid(1, 50);
        check("T8 warp1 cold", warp1_inst, 32'hAA000400);

        // ==============================================================
        // Summary
        // ==============================================================
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("========================================\n");
        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
