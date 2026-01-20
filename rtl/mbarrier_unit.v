//============================================================================
// RalphGPU - mbarrier Unit (Hopper+ Memory Barrier)
//
// Implements Hopper-class mbarrier (memory barrier) for async operation
// synchronization. Supports per-barrier objects with arrival counting,
// phase tracking, and transaction byte expectations.
//
// Key Features:
// - Per-barrier objects stored in shared memory (16 bytes each)
// - Arrival counting (tracks threads/transactions arrived)
// - Phase tracking (even/odd for ping-pong synchronization)
// - Transaction byte expectations for cp.async integration
// - Non-blocking test_wait/try_wait operations
//
// mbarrier Operations:
// - init: Initialize barrier with expected arrival count
// - arrive: Signal arrival at barrier
// - arrive_drop: Arrive and decrement expected count
// - arrive_and_expect_tx: Arrive with expected transaction bytes
// - test_wait: Non-blocking test if phase completed
// - try_wait: Non-blocking wait attempt
// - inval: Invalidate barrier
//
// Barrier Object Layout (16 bytes in shared memory):
//   [31:0]   - arrival_count (current arrivals)
//   [63:32]  - expected_count (expected arrivals)
//   [95:64]  - pending_tx_bytes (expected transaction bytes)
//   [127:96] - phase[0], valid[1], other flags
//
//============================================================================

`include "gpu_defines.vh"

module mbarrier_unit #(
    parameter NUM_BARRIERS = 8,          // Number of barrier objects
    parameter NUM_WARPS = `WARPS_PER_SM, // Number of warps in SM
    parameter SMEM_ADDR_W = 14,          // Shared memory address width
    parameter WARP_ID_W = $clog2(NUM_WARPS)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Control Interface (from issue stage)
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   func,           // MBAR_INIT, MBAR_ARRIVE, etc.
    input  wire [SMEM_ADDR_W-1:0]       barrier_addr,   // Barrier address in shared mem
    input  wire [31:0]                  count,          // Expected count for init, or tx bytes
    input  wire [WARP_ID_W-1:0]         warp_id,
    input  wire [31:0]                  thread_mask,    // Active threads

    output reg                          ready,
    output reg                          done,
    output reg  [31:0]                  result,         // For test_wait/try_wait result
    output reg                          result_valid,   // Result is valid

    //------------------------------------------------------------------------
    // Async Arrival Interface (from cp.async engine)
    //------------------------------------------------------------------------
    input  wire                         async_arrive_valid,
    input  wire [SMEM_ADDR_W-1:0]       async_barrier_addr,
    input  wire [31:0]                  async_tx_bytes,

    //------------------------------------------------------------------------
    // Warp Stall Interface (to scheduler)
    //------------------------------------------------------------------------
    output reg  [NUM_WARPS-1:0]         warp_blocked,   // Warps blocked on mbarrier

    //------------------------------------------------------------------------
    // Shared Memory Interface (for barrier object storage)
    //------------------------------------------------------------------------
    output reg                          smem_rd_en,
    output reg  [SMEM_ADDR_W-1:0]       smem_rd_addr,
    input  wire [127:0]                 smem_rd_data,
    input  wire                         smem_rd_valid,

    output reg                          smem_wr_en,
    output reg  [SMEM_ADDR_W-1:0]       smem_wr_addr,
    output reg  [127:0]                 smem_wr_data,
    output reg  [15:0]                  smem_wr_mask    // Byte write mask
);

    //------------------------------------------------------------------------
    // Internal Barrier State
    // Each barrier is 16 bytes, indexed by barrier_addr[SMEM_ADDR_W-1:4]
    //------------------------------------------------------------------------
    localparam BARRIER_IDX_W = (NUM_BARRIERS > 1) ? $clog2(NUM_BARRIERS) : 1;

    // Barrier object fields (cached from shared memory or local)
    reg [31:0] arrival_count   [0:NUM_BARRIERS-1];
    reg [31:0] expected_count  [0:NUM_BARRIERS-1];
    reg [31:0] pending_tx      [0:NUM_BARRIERS-1];
    reg        phase           [0:NUM_BARRIERS-1];
    reg        barrier_valid   [0:NUM_BARRIERS-1];

    // Per-warp wait state
    reg [BARRIER_IDX_W-1:0] warp_wait_barrier [0:NUM_WARPS-1];
    reg                     warp_wait_phase   [0:NUM_WARPS-1];

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 4'd0;
    localparam ST_READ_BARRIER  = 4'd1;
    localparam ST_WAIT_READ     = 4'd2;
    localparam ST_PROCESS       = 4'd3;
    localparam ST_WRITE_BARRIER = 4'd4;
    localparam ST_COMPLETE      = 4'd5;
    localparam ST_ASYNC_ARRIVE  = 4'd6;

    reg [3:0]  state;
    reg [5:0]  saved_func;
    reg [SMEM_ADDR_W-1:0] saved_addr;
    reg [31:0] saved_count;
    reg [WARP_ID_W-1:0] saved_warp_id;
    reg [31:0] saved_thread_mask;
    reg [BARRIER_IDX_W-1:0] current_barrier;

    // Compute barrier index from address (each barrier is 16 bytes)
    wire [BARRIER_IDX_W-1:0] barrier_idx = barrier_addr[4 +: BARRIER_IDX_W];
    wire [BARRIER_IDX_W-1:0] async_barrier_idx = async_barrier_addr[4 +: BARRIER_IDX_W];

    // Count active threads
    function [5:0] popcount32;
        input [31:0] mask;
        integer i;
        begin
            popcount32 = 0;
            for (i = 0; i < 32; i = i + 1) begin
                popcount32 = popcount32 + mask[i];
            end
        end
    endfunction

    wire [5:0] active_threads = popcount32(saved_thread_mask);

    //------------------------------------------------------------------------
    // Check if barrier phase is complete
    // A barrier completes when:
    // 1. It is valid
    // 2. arrival_count >= expected_count (enough arrivals)
    // 3. pending_tx == 0 (no pending transactions)
    //------------------------------------------------------------------------
    wire [NUM_BARRIERS-1:0] barrier_complete;
    genvar g;
    generate
        for (g = 0; g < NUM_BARRIERS; g = g + 1) begin : gen_complete
            assign barrier_complete[g] = barrier_valid[g] &&
                                         (arrival_count[g] >= expected_count[g]) &&
                                         (pending_tx[g] == 0);
        end
    endgenerate

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    integer w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ready <= 1'b1;
            done <= 1'b0;
            result <= 32'b0;
            result_valid <= 1'b0;
            warp_blocked <= {NUM_WARPS{1'b0}};
            smem_rd_en <= 1'b0;
            smem_wr_en <= 1'b0;

            for (w = 0; w < NUM_BARRIERS; w = w + 1) begin
                arrival_count[w] <= 32'b0;
                expected_count[w] <= 32'b0;
                pending_tx[w] <= 32'b0;
                phase[w] <= 1'b0;
                barrier_valid[w] <= 1'b0;
            end

            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                warp_wait_barrier[w] <= {BARRIER_IDX_W{1'b0}};
                warp_wait_phase[w] <= 1'b0;
            end
        end else begin
            // Default outputs
            done <= 1'b0;
            result_valid <= 1'b0;
            smem_rd_en <= 1'b0;
            smem_wr_en <= 1'b0;

            //----------------------------------------------------------------
            // Check for warp release (phase flipped)
            // A warp waiting on a barrier is released when the phase has
            // changed from what it was when it started waiting.
            // This works because barrier completion flips the phase.
            //----------------------------------------------------------------
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                if (warp_blocked[w]) begin
                    // Release if phase has flipped since we started waiting
                    if (phase[warp_wait_barrier[w]] != warp_wait_phase[w]) begin
                        warp_blocked[w] <= 1'b0;
                    end
                end
            end

            //----------------------------------------------------------------
            // Handle async arrivals from cp.async (always processed)
            // This can happen at any time when transactions complete.
            //----------------------------------------------------------------
            if (async_arrive_valid) begin
                `ifdef MBARRIER_DEBUG
                `ifdef SIMULATION
                $display("[MBARRIER ASYNC] addr=0x%h idx=%0d tx_bytes=%0d pending_before=%0d arrival=%0d expected=%0d valid=%b",
                         async_barrier_addr, async_barrier_idx, async_tx_bytes,
                         pending_tx[async_barrier_idx], arrival_count[async_barrier_idx],
                         expected_count[async_barrier_idx], barrier_valid[async_barrier_idx]);
                `endif
                `endif
                // Decrement pending transaction bytes
                if (pending_tx[async_barrier_idx] >= async_tx_bytes) begin
                    pending_tx[async_barrier_idx] <= pending_tx[async_barrier_idx] - async_tx_bytes;

                    // Check if this completion triggers barrier completion and phase flip
                    if (barrier_valid[async_barrier_idx] &&
                        (arrival_count[async_barrier_idx] >= expected_count[async_barrier_idx]) &&
                        (pending_tx[async_barrier_idx] - async_tx_bytes == 0)) begin
                        // Barrier just completed! Flip phase
                        `ifdef MBARRIER_DEBUG
                        `ifdef SIMULATION
                        $display("[MBARRIER ASYNC] Phase flip for barrier %0d!", async_barrier_idx);
                        `endif
                        `endif
                        phase[async_barrier_idx] <= ~phase[async_barrier_idx];
                        arrival_count[async_barrier_idx] <= 32'b0;
                    end
                end else begin
                    pending_tx[async_barrier_idx] <= 32'b0;
                    // Check for completion
                    if (barrier_valid[async_barrier_idx] &&
                        (arrival_count[async_barrier_idx] >= expected_count[async_barrier_idx])) begin
                        `ifdef MBARRIER_DEBUG
                        `ifdef SIMULATION
                        $display("[MBARRIER ASYNC] Phase flip for barrier %0d (pending underflow)!", async_barrier_idx);
                        `endif
                        `endif
                        phase[async_barrier_idx] <= ~phase[async_barrier_idx];
                        arrival_count[async_barrier_idx] <= 32'b0;
                    end
                end
            end

            //----------------------------------------------------------------
            // Main State Machine
            //----------------------------------------------------------------
            case (state)
                ST_IDLE: begin
                    ready <= 1'b1;

                    if (valid_in) begin
                        ready <= 1'b0;
                        saved_func <= func;
                        saved_addr <= barrier_addr;
                        saved_count <= count;
                        saved_warp_id <= warp_id;
                        saved_thread_mask <= thread_mask;
                        current_barrier <= barrier_idx;

                        // Process based on function
                        case (func)
                            `MBAR_INIT: begin
                                // Initialize barrier directly (no read needed)
                                state <= ST_PROCESS;
                            end

                            `MBAR_ARRIVE,
                            `MBAR_ARRIVE_DROP,
                            `MBAR_ARRIVE_TX,
                            `MBAR_ARRIVE_NOCOMP,
                            `MBAR_EXPECT_TX: begin
                                // These modify barrier state
                                state <= ST_PROCESS;
                            end

                            `MBAR_TEST_WAIT,
                            `MBAR_TRY_WAIT: begin
                                // Check barrier state
                                state <= ST_PROCESS;
                            end

                            `MBAR_INVALIDATE: begin
                                // Invalidate barrier
                                state <= ST_PROCESS;
                            end

                            default: begin
                                done <= 1'b1;
                                state <= ST_IDLE;
                            end
                        endcase
                    end
                end

                ST_PROCESS: begin
                    case (saved_func)
                        `MBAR_INIT: begin
                            // Initialize barrier with expected count
                            arrival_count[current_barrier] <= 32'b0;
                            expected_count[current_barrier] <= saved_count;
                            pending_tx[current_barrier] <= 32'b0;
                            phase[current_barrier] <= 1'b0;
                            barrier_valid[current_barrier] <= 1'b1;

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_ARRIVE: begin
                            // Increment arrival count by number of active threads
                            arrival_count[current_barrier] <=
                                arrival_count[current_barrier] + {26'b0, active_threads};

                            // Check if phase complete
                            if ((arrival_count[current_barrier] + {26'b0, active_threads}) >=
                                expected_count[current_barrier] &&
                                pending_tx[current_barrier] == 0) begin
                                // Flip phase and reset for next iteration
                                phase[current_barrier] <= ~phase[current_barrier];
                                arrival_count[current_barrier] <= 32'b0;
                            end

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_ARRIVE_DROP: begin
                            // Arrive and decrement expected count
                            // Compute new values
                            // new_arrival = arrival + active_threads
                            // new_expected = max(expected - 1, 0)
                            arrival_count[current_barrier] <=
                                arrival_count[current_barrier] + {26'b0, active_threads};

                            if (expected_count[current_barrier] > 1) begin
                                expected_count[current_barrier] <=
                                    expected_count[current_barrier] - 1;

                                // Check completion with new values
                                if ((arrival_count[current_barrier] + {26'b0, active_threads}) >=
                                    (expected_count[current_barrier] - 1) &&
                                    pending_tx[current_barrier] == 0) begin
                                    phase[current_barrier] <= ~phase[current_barrier];
                                    arrival_count[current_barrier] <= 32'b0;
                                end
                            end else begin
                                // expected becomes 0, so barrier immediately completes
                                expected_count[current_barrier] <= 32'b0;
                                if (pending_tx[current_barrier] == 0) begin
                                    phase[current_barrier] <= ~phase[current_barrier];
                                    arrival_count[current_barrier] <= 32'b0;
                                end
                            end

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_ARRIVE_TX: begin
                            // Arrive and set expected transaction bytes
                            arrival_count[current_barrier] <=
                                arrival_count[current_barrier] + {26'b0, active_threads};
                            pending_tx[current_barrier] <=
                                pending_tx[current_barrier] + saved_count;

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_EXPECT_TX: begin
                            // Add expected transaction bytes
                            pending_tx[current_barrier] <=
                                pending_tx[current_barrier] + saved_count;

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_ARRIVE_NOCOMP: begin
                            // Arrive without checking completion
                            arrival_count[current_barrier] <=
                                arrival_count[current_barrier] + {26'b0, active_threads};

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_TEST_WAIT: begin
                            // Non-blocking test: return 1 if barrier phase complete, 0 otherwise
                            // Phase complete if barrier is valid AND current phase != requested phase
                            if (barrier_valid[current_barrier] &&
                                phase[current_barrier] != saved_count[0]) begin
                                result <= 32'h1;
                            end else begin
                                result <= 32'h0;
                            end
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_TRY_WAIT: begin
                            // Non-blocking wait: if not complete, stall the warp
                            if (barrier_valid[current_barrier] &&
                                phase[current_barrier] != saved_count[0]) begin
                                // Already complete (phase flipped)
                                result <= 32'h1;
                                result_valid <= 1'b1;
                            end else begin
                                // Block the warp
                                warp_blocked[saved_warp_id] <= 1'b1;
                                warp_wait_barrier[saved_warp_id] <= current_barrier;
                                warp_wait_phase[saved_warp_id] <= saved_count[0];
                                result <= 32'h0;
                                result_valid <= 1'b1;
                            end
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        `MBAR_INVALIDATE: begin
                            // Invalidate barrier and release any waiting warps
                            barrier_valid[current_barrier] <= 1'b0;
                            arrival_count[current_barrier] <= 32'b0;
                            expected_count[current_barrier] <= 32'b0;
                            pending_tx[current_barrier] <= 32'b0;

                            // Release warps waiting on this barrier
                            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                                if (warp_blocked[w] &&
                                    warp_wait_barrier[w] == current_barrier) begin
                                    warp_blocked[w] <= 1'b0;
                                end
                            end

                            done <= 1'b1;
                            state <= ST_IDLE;
                        end

                        default: begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    endcase
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Debug: Barrier state visibility
    //------------------------------------------------------------------------
    `ifdef MBARRIER_DEBUG
    always @(posedge clk) begin
        if (done) begin
            `ifdef SIMULATION
            $display("[MBARRIER] func=%0d barrier=%0d arrival=%0d expected=%0d pending_tx=%0d phase=%b",
                     saved_func, current_barrier,
                     arrival_count[current_barrier],
                     expected_count[current_barrier],
                     pending_tx[current_barrier],
                     phase[current_barrier]);
            `endif
        end
    end
    `endif

endmodule
