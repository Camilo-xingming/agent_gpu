//============================================================================
// RalphGPU - Cluster Barrier Unit
// Implements barrier.cluster synchronization across multiple SMs in a cluster
// PTX Instructions: barrier.cluster.arrive, barrier.cluster.wait,
//                   barrier.cluster.sync, barrier.cluster.init
//
// This unit manages cross-SM synchronization for Thread Block Clusters.
// Each SM in a cluster communicates arrive/complete signals through this unit.
// Reference: NVIDIA PTX ISA 8.5+, Hopper/Blackwell Architecture
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module cluster_barrier_unit #(
    parameter NUM_SM = 4,               // SMs per cluster
    parameter NUM_BARRIERS = 16,        // Concurrent barriers per cluster
    parameter BARRIER_ID_W = 4,         // log2(NUM_BARRIERS)
    parameter THREAD_COUNT_W = 16       // Max threads per barrier
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Per-SM Arrive Interface
    // Each SM signals when threads arrive at a barrier
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]            sm_arrive_valid,     // SM signals arrival
    input wire [NUM_SM*(BARRIER_ID_W)-1:0] sm_arrive_barrier_id,  // Barrier ID
    input wire [NUM_SM*(THREAD_COUNT_W)-1:0] sm_arrive_count,       // Thread count arriving

    //------------------------------------------------------------------------
    // Per-SM Wait Interface
    // Each SM queries if barrier is complete
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]            sm_wait_valid,       // SM wants to wait
    input wire [NUM_SM*(BARRIER_ID_W)-1:0] sm_wait_barrier_id,  // Barrier to wait on
    output wire [NUM_SM-1:0]            sm_wait_complete,    // Barrier is complete

    //------------------------------------------------------------------------
    // Per-SM Init Interface
    // Initialize barrier with expected thread count
    //------------------------------------------------------------------------
    input  wire [NUM_SM-1:0]            sm_init_valid,       // SM initializes barrier
    input wire [NUM_SM*(BARRIER_ID_W)-1:0] sm_init_barrier_id,  // Barrier to init
    input wire [NUM_SM*(THREAD_COUNT_W)-1:0] sm_init_count,       // Expected thread count

    //------------------------------------------------------------------------
    // Status
    //------------------------------------------------------------------------
    output wire [NUM_BARRIERS-1:0]      barrier_active,      // Which barriers are in use
    output wire [NUM_BARRIERS-1:0]      barrier_complete     // Which barriers have completed
);

    //------------------------------------------------------------------------
    // Barrier State
    //------------------------------------------------------------------------
    reg [THREAD_COUNT_W-1:0] barrier_expected [0:NUM_BARRIERS-1];  // Expected thread count
    reg [THREAD_COUNT_W-1:0] barrier_arrived [0:NUM_BARRIERS-1];   // Current arrive count
    reg [NUM_BARRIERS-1:0]   barrier_valid;                        // Barrier is initialized
    reg [NUM_BARRIERS-1:0]   barrier_done;                         // Barrier has completed

    //------------------------------------------------------------------------
    // Status outputs
    //------------------------------------------------------------------------
    assign barrier_active = barrier_valid;
    assign barrier_complete = barrier_done;

    //------------------------------------------------------------------------
    // Per-SM wait completion
    //------------------------------------------------------------------------
    genvar sm;
    generate
        for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : gen_wait_complete
            assign sm_wait_complete[sm] = sm_wait_valid[sm] ?
                barrier_done[sm_wait_barrier_id[sm*BARRIER_ID_W +: BARRIER_ID_W]] : 1'b0;
        end
    endgenerate

    //------------------------------------------------------------------------
    // Barrier logic
    //------------------------------------------------------------------------
    integer i, s;
    reg [THREAD_COUNT_W-1:0] total_arrive;
    reg [BARRIER_ID_W-1:0] arrive_bid, init_bid;
    reg [THREAD_COUNT_W-1:0] arrive_cnt, init_cnt;
    reg arrive_found, init_found;

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            barrier_valid <= 0;
            barrier_done <= 0;
            for (i = 0; i < NUM_BARRIERS; i = i + 1) begin
                barrier_expected[i] <= 0;
                barrier_arrived[i] <= 0;
            end
        end else begin
            // Process init requests (priority over arrive)
            // Find first SM that wants to init
            init_found = 1'b0;
            init_bid = 0;
            init_cnt = 0;
            for (s = 0; s < NUM_SM; s = s + 1) begin
                if (sm_init_valid[s] && !init_found) begin
                    init_found = 1'b1;
                    init_bid = sm_init_barrier_id[s*BARRIER_ID_W +: BARRIER_ID_W];
                    init_cnt = sm_init_count[s*THREAD_COUNT_W +: THREAD_COUNT_W];
                end
            end

            if (init_found) begin
                barrier_expected[init_bid] <= init_cnt;
                barrier_arrived[init_bid] <= 0;
                barrier_valid[init_bid] <= 1'b1;
                barrier_done[init_bid] <= 1'b0;
                `ifdef SIMULATION
                $display("[CLUSTER_BARRIER] INIT: barrier=%0d expected=%0d", init_bid, init_cnt);
                `endif
            end

            // Process arrive requests from all SMs
            // Accumulate arrivals for each barrier
            for (i = 0; i < NUM_BARRIERS; i = i + 1) begin
                total_arrive = 0;
                for (s = 0; s < NUM_SM; s = s + 1) begin
                    if (sm_arrive_valid[s] && sm_arrive_barrier_id[s*BARRIER_ID_W +: BARRIER_ID_W] == i[BARRIER_ID_W-1:0]) begin
                        total_arrive = total_arrive + sm_arrive_count[s*THREAD_COUNT_W +: THREAD_COUNT_W];
                    end
                end

                if (total_arrive > 0 && barrier_valid[i]) begin
                    barrier_arrived[i] <= barrier_arrived[i] + total_arrive;
                    `ifdef SIMULATION
                    $display("[CLUSTER_BARRIER] ARRIVE: barrier=%0d +%0d (now %0d/%0d)",
                             i, total_arrive, barrier_arrived[i] + total_arrive, barrier_expected[i]);
                    `endif

                    // Check if barrier is now complete
                    if ((barrier_arrived[i] + total_arrive) >= barrier_expected[i]) begin
                        barrier_done[i] <= 1'b1;
                        `ifdef SIMULATION
                        $display("[CLUSTER_BARRIER] COMPLETE: barrier=%0d", i);
                        `endif
                    end
                end
    /* verilator lint_on BLKSEQ */
            end

            // Reset completed barriers that have no waiters
            // (This would normally be managed by software or a cleanup mechanism)
        end
    end

endmodule
