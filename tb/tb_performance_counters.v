`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_performance_counters;

    // Parameters
    localparam NUM_SM        = 2;
    localparam NUM_WARPS     = 4;
    localparam NUM_COUNTERS  = 64;
    localparam COUNTER_WIDTH = 16; // Use 16 bits to test overflow easily
    localparam SAMPLE_INTERVAL = 10;

    // Clocks and Resets
    reg clk;
    reg rst_n;

    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // signals for performance_counters
    //------------------------------------------------------------------------
    reg                     enable;
    reg                     clear;
    reg  [5:0]              select;
    wire [COUNTER_WIDTH-1:0] counter_value;

    reg  [NUM_SM-1:0]        sm_active;
    reg  [NUM_SM-1:0]        sm_issue_valid;
    reg  [NUM_SM-1:0]        sm_dual_issue;
    reg  [NUM_SM-1:0]        sm_stall_scoreboard;
    reg  [NUM_SM-1:0]        sm_stall_ifetch;
    reg  [NUM_SM-1:0]        sm_stall_mem;
    reg  [NUM_SM-1:0]        sm_stall_sync;
    reg  [NUM_SM-1:0]        sm_stall_other;

    reg  [NUM_SM-1:0]        fu_alu_active;
    reg  [NUM_SM-1:0]        fu_fpu_active;
    reg  [NUM_SM-1:0]        fu_sfu_active;
    reg  [NUM_SM-1:0]        fu_tensor_active;
    reg  [NUM_SM-1:0]        fu_ldst_active;

    reg  [NUM_SM-1:0]        l1_hit;
    reg  [NUM_SM-1:0]        l1_miss;
    reg                      l2_hit;
    reg                      l2_miss;
    reg                      dram_access;

    reg  [NUM_SM*NUM_WARPS-1:0] warp_issued;
    reg  [NUM_SM*NUM_WARPS-1:0] warp_active;
    reg  [NUM_SM*NUM_WARPS-1:0] warp_stalled;
    reg  [NUM_SM*NUM_WARPS-1:0] warp_diverged;

    reg  [NUM_SM-1:0]        branch_taken;
    reg  [NUM_SM-1:0]        branch_divergent;
    reg  [NUM_SM-1:0]        branch_reconverge;

    reg                      tensor_mma_issued;
    reg                      tensor_mma_completed;
    reg  [15:0]              tensor_flops;

    wire [NUM_SM*8-1:0]      sm_occupancy;
    wire [31:0]              achieved_ipc;
    wire [31:0]              memory_throughput;
    wire [63:0]              total_instructions;
    wire [63:0]              total_cycles;
    wire [63:0]              total_memory_bytes;

    performance_counters #(
        .NUM_SM(NUM_SM),
        .NUM_WARPS(NUM_WARPS),
        .NUM_COUNTERS(NUM_COUNTERS),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) dut_global (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .clear(clear),
        .select(select),
        .counter_value(counter_value),
        .sm_active(sm_active),
        .sm_issue_valid(sm_issue_valid),
        .sm_dual_issue(sm_dual_issue),
        .sm_stall_scoreboard(sm_stall_scoreboard),
        .sm_stall_ifetch(sm_stall_ifetch),
        .sm_stall_mem(sm_stall_mem),
        .sm_stall_sync(sm_stall_sync),
        .sm_stall_other(sm_stall_other),
        .fu_alu_active(fu_alu_active),
        .fu_fpu_active(fu_fpu_active),
        .fu_sfu_active(fu_sfu_active),
        .fu_tensor_active(fu_tensor_active),
        .fu_ldst_active(fu_ldst_active),
        .l1_hit(l1_hit),
        .l1_miss(l1_miss),
        .l2_hit(l2_hit),
        .l2_miss(l2_miss),
        .dram_access(dram_access),
        .warp_issued(warp_issued),
        .warp_active(warp_active),
        .warp_stalled(warp_stalled),
        .warp_diverged(warp_diverged),
        .branch_taken(branch_taken),
        .branch_divergent(branch_divergent),
        .branch_reconverge(branch_reconverge),
        .tensor_mma_issued(tensor_mma_issued),
        .tensor_mma_completed(tensor_mma_completed),
        .tensor_flops(tensor_flops),
        .sm_occupancy(sm_occupancy),
        .achieved_ipc(achieved_ipc),
        .memory_throughput(memory_throughput),
        .total_instructions(total_instructions),
        .total_cycles(total_cycles),
        .total_memory_bytes(total_memory_bytes)
    );

    //------------------------------------------------------------------------
    // signals for sm_performance_monitor
    //------------------------------------------------------------------------
    reg                      sm_active_mon;
    reg  [NUM_WARPS-1:0]     warp_valid_mon;
    reg  [NUM_WARPS-1:0]     warp_ready_mon;
    reg  [NUM_WARPS-1:0]     warp_issued_mon;
    reg  [NUM_WARPS-1:0]     warp_stalled_mem_mon;
    reg  [NUM_WARPS-1:0]     warp_stalled_sync_mon;
    reg  [NUM_WARPS-1:0]     warp_stalled_fu_mon;

    reg                      issue_valid_mon;
    reg                      dual_issue_mon;
    reg                      stall_ifetch_mon;
    reg                      stall_decode_mon;
    reg                      stall_issue_mon;

    reg                      alu_busy_mon;
    reg                      fpu_busy_mon;
    reg                      sfu_busy_mon;
    reg                      tensor_busy_mon;
    reg                      ldst_busy_mon;

    reg                      l1_access_mon;
    reg                      l1_hit_mon;
    reg                      smem_access_mon;
    reg                      smem_bank_conflict_mon;

    wire [31:0]              ipc_sampled_mon;
    wire [31:0]              issue_efficiency_mon;
    wire [31:0]              warp_occupancy_mon;
    wire [31:0]              mem_stall_rate_mon;
    wire [31:0]              l1_hit_rate_mon;
    wire [31:0]              fu_utilization_mon;

    sm_performance_monitor #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .SAMPLE_INTERVAL(SAMPLE_INTERVAL)
    ) dut_sm (
        .clk(clk),
        .rst_n(rst_n),
        .sm_active(sm_active_mon),
        .warp_valid(warp_valid_mon),
        .warp_ready(warp_ready_mon),
        .warp_issued(warp_issued_mon),
        .warp_stalled_mem(warp_stalled_mem_mon),
        .warp_stalled_sync(warp_stalled_sync_mon),
        .warp_stalled_fu(warp_stalled_fu_mon),
        .issue_valid(issue_valid_mon),
        .dual_issue(dual_issue_mon),
        .stall_ifetch(stall_ifetch_mon),
        .stall_decode(stall_decode_mon),
        .stall_issue(stall_issue_mon),
        .alu_busy(alu_busy_mon),
        .fpu_busy(fpu_busy_mon),
        .sfu_busy(sfu_busy_mon),
        .tensor_busy(tensor_busy_mon),
        .ldst_busy(ldst_busy_mon),
        .l1_access(l1_access_mon),
        .l1_hit(l1_hit_mon),
        .smem_access(smem_access_mon),
        .smem_bank_conflict(smem_bank_conflict_mon),
        .ipc_sampled(ipc_sampled_mon),
        .issue_efficiency(issue_efficiency_mon),
        .warp_occupancy(warp_occupancy_mon),
        .mem_stall_rate(mem_stall_rate_mon),
        .l1_hit_rate(l1_hit_rate_mon),
        .fu_utilization(fu_utilization_mon)
    );

    integer errors = 0;

    initial begin
        $dumpfile("tb_performance_counters.vcd");
        $dumpvars(0, tb_performance_counters);

        // Initialize signals
        clk = 0;
        rst_n = 0;
        enable = 0;
        clear = 0;
        select = 0;

        sm_active = 0; sm_issue_valid = 0; sm_dual_issue = 0;
        sm_stall_scoreboard = 0; sm_stall_ifetch = 0; sm_stall_mem = 0;
        sm_stall_sync = 0; sm_stall_other = 0;
        fu_alu_active = 0; fu_fpu_active = 0; fu_sfu_active = 0;
        fu_tensor_active = 0; fu_ldst_active = 0;
        l1_hit = 0; l1_miss = 0; l2_hit = 0; l2_miss = 0; dram_access = 0;
        warp_issued = 0; warp_active = 0; warp_stalled = 0; warp_diverged = 0;
        branch_taken = 0; branch_divergent = 0; branch_reconverge = 0;
        tensor_mma_issued = 0; tensor_mma_completed = 0; tensor_flops = 0;

        sm_active_mon = 0; warp_valid_mon = 0; warp_ready_mon = 0;
        warp_issued_mon = 0; warp_stalled_mem_mon = 0; warp_stalled_sync_mon = 0;
        warp_stalled_fu_mon = 0; issue_valid_mon = 0; dual_issue_mon = 0;
        stall_ifetch_mon = 0; stall_decode_mon = 0; stall_issue_mon = 0;
        alu_busy_mon = 0; fpu_busy_mon = 0; sfu_busy_mon = 0;
        tensor_busy_mon = 0; ldst_busy_mon = 0; l1_access_mon = 0;
        l1_hit_mon = 0; smem_access_mon = 0; smem_bank_conflict_mon = 0;

        // Reset sequence
        #20 rst_n = 1;
        #10 enable = 1;

        // --------------------------------------------------------
        // Test 1: Counter Increment and Multiple Channels
        // --------------------------------------------------------
        $display("--- Test 1: Increment & Channels ---");
        sm_active = 2'b11;
        sm_issue_valid = 2'b11;  // 2 issues
        warp_active = 8'b00001111; // 4 warps active on SM 0
        l1_hit = 2'b01; // 1 L1 hit on SM 0
        l1_miss = 2'b10; // 1 L1 miss on SM 1
        tensor_flops = 16'd50;

        #10;
        sm_issue_valid = 2'b00;
        l1_hit = 2'b00;
        l1_miss = 2'b00;
        tensor_flops = 16'd0;
        #10;

        // Select instructions counter (CTR_INSTRUCTIONS = 1)
        select = 6'd1;
        #10;
        if (counter_value !== 2) begin $display("FAIL: Expected 2 instructions, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: Global instructions counter");

        // Select L1 hits (CTR_L1_HITS = 16)
        select = 6'd16;
        #10;
        if (counter_value !== 1) begin $display("FAIL: Expected 1 L1 hit, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: L1 hits counter");

        // Select Tensor FLOPs (CTR_TENSOR_FLOPS = 29)
        select = 6'd29;
        #10;
        if (counter_value !== 50) begin $display("FAIL: Expected 50 Tensor FLOPs, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: Tensor FLOPs counter");

        // --------------------------------------------------------
        // Test 2: Counter Reset (Clear)
        // --------------------------------------------------------
        $display("--- Test 2: Counter Clear ---");
        clear = 1;
        #10;
        clear = 0;
        #10;

        select = 6'd1; // CTR_INSTRUCTIONS
        #10;
        if (counter_value !== 0) begin $display("FAIL: Instructions counter not cleared, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: Instructions counter cleared");

        select = 6'd29; // CTR_TENSOR_FLOPS
        #10;
        if (counter_value !== 0) begin $display("FAIL: Tensor FLOPs counter not cleared, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: Tensor FLOPs counter cleared");

        // --------------------------------------------------------
        // Test 3: Overflow Behavior
        // --------------------------------------------------------
        $display("--- Test 3: Overflow Behavior ---");
        // Force overflow by injecting max value repeatedly
        enable = 1;
        tensor_flops = 16'hFFFF;
        // Inject for 3 cycles: 3 * 65535 = 196605
        // COUNTER_WIDTH is 16, max val is 65535.
        // 196605 % 65536 = 65533 (which is 16'hFFFD)
        #10;
        #10;
        #10;
        tensor_flops = 16'd0;
        #10;

        select = 6'd29; // CTR_TENSOR_FLOPS
        #10;
        if (counter_value !== 16'hFFFD) begin $display("FAIL: Expected overflow value 65533, got %0d", counter_value); errors = errors + 1; end
        else $display("PASS: Overflow behavior correct");

        // --------------------------------------------------------
        // Test 4: SM Monitor Sampling
        // --------------------------------------------------------
        $display("--- Test 4: SM Monitor Sampling ---");
        sm_active_mon = 1;
        warp_valid_mon = 4'b1111;
        warp_ready_mon = 4'b1111;
        issue_valid_mon = 1;

        // Wait for sample interval
        repeat (SAMPLE_INTERVAL) @(posedge clk);
        #10;

        if (issue_efficiency_mon !== 90) begin $display("FAIL: Expected 90%% issue efficiency, got %0d", issue_efficiency_mon); errors = errors + 1; end
        else $display("PASS: Issue efficiency sampling");

        if (warp_occupancy_mon !== 90) begin $display("FAIL: Expected 90%% warp occupancy, got %0d", warp_occupancy_mon); errors = errors + 1; end
        else $display("PASS: Warp occupancy sampling");

        if (errors == 0) begin
            $display("=================================================");
            $display("ALL TESTS PASSED (%0d errors)", errors);
            $display("=================================================");
        end else begin
            $display("=================================================");
            $display("TESTS FAILED with %0d errors", errors);
            $display("=================================================");
            $fatal(1);
        end
        $finish;
    end
endmodule
