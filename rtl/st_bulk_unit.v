//============================================================================
// RalphGPU - Bulk Store Unit
// Implements st.bulk instructions for asynchronous bulk store from SMEM to GMEM
// PTX Instructions: st.bulk.global, st.bulk.shared, st.bulk.commit, st.bulk.wait
//
// Reference: NVIDIA PTX ISA 8.5+, Hopper/Blackwell Architecture
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module st_bulk_unit #(
    parameter DATA_WIDTH = 32,
    parameter SMEM_ADDR_W = 14,          // Shared memory address width (16KB)
    parameter GMEM_ADDR_W = 32,          // Global memory address width
    parameter MAX_BULK_SIZE = 1024,      // Max bytes per bulk transfer
    parameter MAX_PENDING = 8            // Max pending bulk operations
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   opcode,         // OP_ST_BULK
    input  wire [5:0]                   func,           // ST_BULK_* function
    input  wire [SMEM_ADDR_W-1:0]       smem_addr,      // Source address in shared memory
    input  wire [GMEM_ADDR_W-1:0]       gmem_addr,      // Destination address in global memory
    input  wire [15:0]                  byte_count,     // Number of bytes to transfer
    input  wire [2:0]                   cache_hint,     // Cache policy hint

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          result_valid,
    output reg  [DATA_WIDTH-1:0]        result,
    output wire                         busy,

    //------------------------------------------------------------------------
    // Shared Memory Read Interface
    //------------------------------------------------------------------------
    output reg                          smem_rd_en,
    output reg  [SMEM_ADDR_W-1:0]       smem_rd_addr,
    input  wire [127:0]                 smem_rd_data,
    input  wire                         smem_rd_valid,

    //------------------------------------------------------------------------
    // Global Memory Write Interface
    //------------------------------------------------------------------------
    output reg                          gmem_wr_valid,
    output reg  [GMEM_ADDR_W-1:0]       gmem_wr_addr,
    output reg  [127:0]                 gmem_wr_data,
    output reg  [4:0]                   gmem_wr_size,   // Bytes per write (16)
    input  wire                         gmem_wr_done,

    //------------------------------------------------------------------------
    // Completion Signal (for mbarrier integration)
    //------------------------------------------------------------------------
    output reg                          bulk_complete,
    output reg  [15:0]                  bytes_transferred
);

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE       = 3'd0;
    localparam ST_ENQUEUE    = 3'd1;
    localparam ST_READ_SMEM  = 3'd2;
    localparam ST_WAIT_READ  = 3'd3;
    localparam ST_WRITE_GMEM = 3'd4;
    localparam ST_WAIT_WRITE = 3'd5;
    localparam ST_COMPLETE   = 3'd6;
    localparam ST_WAIT_ALL   = 3'd7;

    reg [2:0] state;

    //------------------------------------------------------------------------
    // Pending operation queue
    //------------------------------------------------------------------------
    reg [SMEM_ADDR_W-1:0]   op_smem_base   [0:MAX_PENDING-1];
    reg [GMEM_ADDR_W-1:0]   op_gmem_base   [0:MAX_PENDING-1];
    reg [15:0]              op_byte_count  [0:MAX_PENDING-1];
    reg [15:0]              op_bytes_done  [0:MAX_PENDING-1];
    reg [MAX_PENDING-1:0]   op_valid;
    reg [MAX_PENDING-1:0]   op_committed;

    reg [2:0]               op_head;        // Next enqueue position
    reg [2:0]               op_tail;        // Next dequeue position
    reg [3:0]               pending_count;

    //------------------------------------------------------------------------
    // Current operation state
    //------------------------------------------------------------------------
    reg [SMEM_ADDR_W-1:0]   cur_smem_addr;
    reg [GMEM_ADDR_W-1:0]   cur_gmem_addr;
    reg [15:0]              cur_remaining;
    reg [2:0]               cur_op_idx;
    reg                     commit_pending;
    reg [3:0]               wait_count;     // Number of ops to wait for

    //------------------------------------------------------------------------
    // Busy signal
    //------------------------------------------------------------------------
    assign busy = (state != ST_IDLE) || (pending_count > 0);

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            result_valid <= 1'b0;
            result <= 0;
            smem_rd_en <= 1'b0;
            smem_rd_addr <= 0;
            gmem_wr_valid <= 1'b0;
            gmem_wr_addr <= 0;
            gmem_wr_data <= 0;
            gmem_wr_size <= 0;
            bulk_complete <= 1'b0;
            bytes_transferred <= 0;

            op_head <= 0;
            op_tail <= 0;
            pending_count <= 0;
            op_valid <= 0;
            op_committed <= 0;

            cur_smem_addr <= 0;
            cur_gmem_addr <= 0;
            cur_remaining <= 0;
            cur_op_idx <= 0;
            commit_pending <= 1'b0;
            wait_count <= 0;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            smem_rd_en <= 1'b0;
            gmem_wr_valid <= 1'b0;
            bulk_complete <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (valid_in && opcode == `OP_ST_BULK) begin
                        case (func)
                            `ST_BULK_GLOBAL: begin
                                // Enqueue a bulk store operation
                                if (pending_count < MAX_PENDING) begin
                                    op_smem_base[op_head] <= smem_addr;
                                    op_gmem_base[op_head] <= gmem_addr;
                                    op_byte_count[op_head] <= (byte_count > MAX_BULK_SIZE) ? MAX_BULK_SIZE[15:0] : byte_count;
                                    op_bytes_done[op_head] <= 0;
                                    op_valid[op_head] <= 1'b1;
                                    op_committed[op_head] <= 1'b0;
                                    op_head <= op_head + 1;
                                    pending_count <= pending_count + 1;
                                    result <= {16'b0, byte_count};  // Return bytes to be transferred
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                    `ifdef SIMULATION
                                    $display("[ST_BULK] GLOBAL: smem=0x%04x gmem=0x%08x bytes=%0d",
                                             smem_addr, gmem_addr, byte_count);
                                    `endif
                                end else begin
                                    // Queue full, return error
                                    result <= 32'hFFFFFFFF;
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                end
                            end

                            `ST_BULK_SHARED: begin
                                // st.bulk.shared - similar but stays in SMEM (for cluster ops)
                                // For now, treat same as global but flag differently
                                if (pending_count < MAX_PENDING) begin
                                    op_smem_base[op_head] <= smem_addr;
                                    op_gmem_base[op_head] <= gmem_addr;
                                    op_byte_count[op_head] <= (byte_count > MAX_BULK_SIZE) ? MAX_BULK_SIZE[15:0] : byte_count;
                                    op_bytes_done[op_head] <= 0;
                                    op_valid[op_head] <= 1'b1;
                                    op_committed[op_head] <= 1'b0;
                                    op_head <= op_head + 1;
                                    pending_count <= pending_count + 1;
                                    result <= {16'b0, byte_count};
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                    `ifdef SIMULATION
                                    $display("[ST_BULK] SHARED: smem_src=0x%04x smem_dst=0x%04x bytes=%0d",
                                             smem_addr, gmem_addr[SMEM_ADDR_W-1:0], byte_count);
                                    `endif
                                end else begin
                                    result <= 32'hFFFFFFFF;
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                end
                            end

                            `ST_BULK_COMMIT: begin
                                // Commit all pending ops - start actual transfer
                                if (pending_count > 0) begin
                                    commit_pending <= 1'b1;
                                    // Mark all pending as committed
                                    op_committed <= op_valid;
                                    // Start processing from tail
                                    cur_op_idx <= op_tail;
                                    cur_smem_addr <= op_smem_base[op_tail];
                                    cur_gmem_addr <= op_gmem_base[op_tail];
                                    cur_remaining <= op_byte_count[op_tail];
                                    state <= ST_READ_SMEM;
                                    `ifdef SIMULATION
                                    $display("[ST_BULK] COMMIT: %0d operations pending", pending_count);
                                    `endif
                                end else begin
                                    done <= 1'b1;
                                end
                            end

                            `ST_BULK_WAIT: begin
                                // Wait for all committed ops to complete
                                wait_count <= byte_count[3:0];  // Use byte_count as wait count
                                if (pending_count == 0) begin
                                    done <= 1'b1;
                                end else begin
                                    state <= ST_WAIT_ALL;
                                end
                                `ifdef SIMULATION
                                $display("[ST_BULK] WAIT: waiting for %0d operations", pending_count);
                                `endif
                            end

                            default: begin
                                done <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_READ_SMEM: begin
                    // Issue read from shared memory (16 bytes at a time)
                    smem_rd_en <= 1'b1;
                    smem_rd_addr <= cur_smem_addr;
                    state <= ST_WAIT_READ;
                end

                ST_WAIT_READ: begin
                    if (smem_rd_valid) begin
                        // Got data from SMEM, now write to GMEM
                        gmem_wr_data <= smem_rd_data;
                        state <= ST_WRITE_GMEM;
                    end
                end

                ST_WRITE_GMEM: begin
                    // Issue write to global memory
                    gmem_wr_valid <= 1'b1;
                    gmem_wr_addr <= cur_gmem_addr;
                    gmem_wr_size <= (cur_remaining >= 16) ? 5'd16 : cur_remaining[4:0];
                    state <= ST_WAIT_WRITE;
                end

                ST_WAIT_WRITE: begin
                    if (gmem_wr_done) begin
                        // Update progress
                        if (cur_remaining >= 16) begin
                            cur_smem_addr <= cur_smem_addr + 16;
                            cur_gmem_addr <= cur_gmem_addr + 16;
                            cur_remaining <= cur_remaining - 16;
                            op_bytes_done[cur_op_idx] <= op_bytes_done[cur_op_idx] + 16;
                        end else begin
                            cur_remaining <= 0;
                            op_bytes_done[cur_op_idx] <= op_byte_count[cur_op_idx];
                        end

                        if (cur_remaining <= 16) begin
                            // Current operation complete
                            bytes_transferred <= op_byte_count[cur_op_idx];
                            bulk_complete <= 1'b1;
                            op_valid[cur_op_idx] <= 1'b0;
                            pending_count <= pending_count - 1;
                            op_tail <= op_tail + 1;

                            // Check if more operations to process
                            if (pending_count > 1) begin
                                // Move to next operation
                                cur_op_idx <= op_tail + 1;
                                cur_smem_addr <= op_smem_base[op_tail + 1];
                                cur_gmem_addr <= op_gmem_base[op_tail + 1];
                                cur_remaining <= op_byte_count[op_tail + 1];
                                state <= ST_READ_SMEM;
                            end else begin
                                // All done
                                commit_pending <= 1'b0;
                                done <= 1'b1;
                                state <= ST_IDLE;
                            end
                        end else begin
                            // More data to transfer in current op
                            state <= ST_READ_SMEM;
                        end
                    end
                end

                ST_WAIT_ALL: begin
                    // Wait for all pending operations
                    if (pending_count == 0) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                ST_COMPLETE: begin
                    done <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
