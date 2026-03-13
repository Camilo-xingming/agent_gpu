//============================================================================
// RalphGPU - Multimem Unit
// Implements distributed shared memory operations for Thread Block Clusters
// PTX Instructions: multimem.ld, multimem.st, multimem.red
//
// Address format: [31:24] = target_sm_mask, [23:0] = smem_addr
// For single-SM operation, target_mask bit for local SM accesses local SMEM
// For cluster operation, requests are routed through cluster interconnect
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module multimem_unit #(
    parameter SM_ID = 0,
    parameter NUM_SMs = 2,
    parameter SHARED_MEM_ADDR_W = 14,
    parameter DATA_WIDTH = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   func,           // MULTIMEM_LD/ST/RED
    input  wire [31:0]                  addr,           // [31:24]=target_mask, [23:0]=smem_addr
    input  wire [DATA_WIDTH-1:0]        wdata,          // Data to write/reduce
    input  wire [2:0]                   red_op,         // Reduction operation (for MULTIMEM_RED)

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          result_valid,
    output reg  [DATA_WIDTH-1:0]        result_data,

    //------------------------------------------------------------------------
    // Local Shared Memory Interface
    //------------------------------------------------------------------------
    output reg                          local_smem_rd_en,
    output reg  [SHARED_MEM_ADDR_W-1:0] local_smem_rd_addr,
    input  wire [DATA_WIDTH-1:0]        local_smem_rd_data,
    input  wire                         local_smem_rd_valid,

    output reg                          local_smem_wr_en,
    output reg  [SHARED_MEM_ADDR_W-1:0] local_smem_wr_addr,
    output reg  [DATA_WIDTH-1:0]        local_smem_wr_data,
    input  wire                         local_smem_wr_done,

    //------------------------------------------------------------------------
    // Cluster Interconnect Interface (for cross-SM operations)
    // Request output to other SMs
    //------------------------------------------------------------------------
    output reg                          cluster_req_valid,
    output reg  [NUM_SMs-1:0]           cluster_req_targets,  // Target SM bitmap
    output reg  [1:0]                   cluster_req_op,       // 0=LD, 1=ST, 2=RED
    output reg  [SHARED_MEM_ADDR_W-1:0] cluster_req_addr,
    output reg  [DATA_WIDTH-1:0]        cluster_req_data,
    output reg  [2:0]                   cluster_req_red_op,
    input  wire                         cluster_req_ack,

    //------------------------------------------------------------------------
    // Cluster Interconnect Interface
    // Response input from other SMs (for loads)
    //------------------------------------------------------------------------
    input  wire                         cluster_resp_valid,
    input  wire [DATA_WIDTH-1:0]        cluster_resp_data,

    //------------------------------------------------------------------------
    // Incoming request from other SMs (for our local SMEM)
    //------------------------------------------------------------------------
    input  wire                         remote_req_valid,
    input  wire [1:0]                   remote_req_op,
    input  wire [SHARED_MEM_ADDR_W-1:0] remote_req_addr,
    input  wire [DATA_WIDTH-1:0]        remote_req_data,
    input  wire [2:0]                   remote_req_red_op,
    output reg                          remote_req_done,
    output reg  [DATA_WIDTH-1:0]        remote_resp_data
);

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 3'd0;
    localparam ST_LOCAL_READ    = 3'd1;
    localparam ST_LOCAL_WRITE   = 3'd2;
    localparam ST_CLUSTER_REQ   = 3'd3;
    localparam ST_CLUSTER_RESP  = 3'd4;
    localparam ST_REDUCE_READ   = 3'd5;
    localparam ST_REDUCE_WRITE  = 3'd6;
    localparam ST_COMPLETE      = 3'd7;

    reg [2:0] state;
    reg [5:0] saved_func;
    reg [7:0] saved_target_mask;
    reg [SHARED_MEM_ADDR_W-1:0] saved_smem_addr;
    reg [DATA_WIDTH-1:0] saved_wdata;
    reg [2:0] saved_red_op;
    reg [DATA_WIDTH-1:0] reduce_acc;
    reg [NUM_SMs-1:0] pending_targets;
    reg is_remote_op;  // Track if current operation is for a remote request

    // Extract address fields
    wire [7:0] target_mask = addr[31:24];
    wire [SHARED_MEM_ADDR_W-1:0] smem_addr = addr[SHARED_MEM_ADDR_W-1:0];
    wire local_target = target_mask[SM_ID];

    //------------------------------------------------------------------------
    // Reduction operation
    //------------------------------------------------------------------------
    function [DATA_WIDTH-1:0] do_reduction;
        input [DATA_WIDTH-1:0] acc;
        input [DATA_WIDTH-1:0] val;
        input [2:0] op;
        begin
            case (op)
                3'd0: do_reduction = acc + val;                      // ADD
                3'd1: do_reduction = ($signed(val) < $signed(acc)) ? val : acc;  // MIN (signed)
                3'd2: do_reduction = ($signed(val) > $signed(acc)) ? val : acc;  // MAX (signed)
                3'd3: do_reduction = acc & val;                      // AND
                3'd4: do_reduction = acc | val;                      // OR
                3'd5: do_reduction = acc ^ val;                      // XOR
                default: do_reduction = acc + val;
            endcase
        end
    endfunction

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            result_valid <= 1'b0;
            result_data <= 0;
            local_smem_rd_en <= 1'b0;
            local_smem_rd_addr <= 0;
            local_smem_wr_en <= 1'b0;
            local_smem_wr_addr <= 0;
            local_smem_wr_data <= 0;
            cluster_req_valid <= 1'b0;
            cluster_req_targets <= 0;
            cluster_req_op <= 2'b0;
            cluster_req_addr <= 0;
            cluster_req_data <= 0;
            cluster_req_red_op <= 3'b0;
            remote_req_done <= 1'b0;
            remote_resp_data <= 0;
            saved_func <= 6'b0;
            saved_target_mask <= 8'b0;
            saved_smem_addr <= 0;
            saved_wdata <= 0;
            saved_red_op <= 3'b0;
            reduce_acc <= 0;
            pending_targets <= 0;
            is_remote_op <= 1'b0;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            local_smem_rd_en <= 1'b0;
            local_smem_wr_en <= 1'b0;
            cluster_req_valid <= 1'b0;
            remote_req_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    is_remote_op <= 1'b0;
                    // Handle remote requests from other SMs first (higher priority)
                    if (remote_req_valid) begin
                        is_remote_op <= 1'b1;
                        case (remote_req_op)
                            2'd0: begin // Remote load request
                                local_smem_rd_en <= 1'b1;
                                local_smem_rd_addr <= remote_req_addr;
                                state <= ST_LOCAL_READ;
                                saved_func <= `MULTIMEM_LD;
                            end
                            2'd1: begin // Remote store request
                                local_smem_wr_en <= 1'b1;
                                local_smem_wr_addr <= remote_req_addr;
                                local_smem_wr_data <= remote_req_data;
                                state <= ST_LOCAL_WRITE;
                                saved_func <= `MULTIMEM_ST;
                            end
                            2'd2: begin // Remote reduction request
                                // Read current value first
                                local_smem_rd_en <= 1'b1;
                                local_smem_rd_addr <= remote_req_addr;
                                saved_wdata <= remote_req_data;
                                saved_smem_addr <= remote_req_addr;
                                saved_red_op <= remote_req_red_op;
                                state <= ST_REDUCE_READ;
                            end
                            default: ; // lint: CASEINCOMPLETE
                        endcase
                    end
                    // Handle local requests
                    else if (valid_in) begin
                        saved_func <= func;
                        saved_target_mask <= target_mask;
                        saved_smem_addr <= smem_addr;
                        saved_wdata <= wdata;
                        saved_red_op <= red_op;

                        case (func)
                            `MULTIMEM_LD: begin
                                if (local_target) begin
                                    // Load from local shared memory
                                    local_smem_rd_en <= 1'b1;
                                    local_smem_rd_addr <= smem_addr;
                                    state <= ST_LOCAL_READ;
                                    `ifdef SIMULATION
                                    $display("[MULTIMEM%0d] LD local: addr=0x%04x", SM_ID, smem_addr); // keep
                                    `endif
                                end else begin
                                    // Load from remote SM
                                    cluster_req_valid <= 1'b1;
                                    cluster_req_targets <= target_mask[NUM_SMs-1:0];
                                    cluster_req_op <= 2'd0;  // LD
                                    cluster_req_addr <= smem_addr;
                                    pending_targets <= target_mask[NUM_SMs-1:0];
                                    state <= ST_CLUSTER_REQ;
                                    `ifdef SIMULATION
                                    $display("[MULTIMEM%0d] LD remote: targets=0x%02x addr=0x%04x", // keep
                                             SM_ID, target_mask, smem_addr);
                                    `endif
                                end
                            end

                            `MULTIMEM_ST: begin
                                if (local_target && (target_mask == (8'b1 << SM_ID))) begin
                                    // Store to local shared memory only
                                    local_smem_wr_en <= 1'b1;
                                    local_smem_wr_addr <= smem_addr;
                                    local_smem_wr_data <= wdata;
                                    state <= ST_LOCAL_WRITE;
                                    `ifdef SIMULATION
                                    $display("[MULTIMEM%0d] ST local: addr=0x%04x data=0x%08x", // keep
                                             SM_ID, smem_addr, wdata);
                                    `endif
                                end else begin
                                    // Multicast store to multiple SMs
                                    // First handle local if targeted
                                    if (local_target) begin
                                        local_smem_wr_en <= 1'b1;
                                        local_smem_wr_addr <= smem_addr;
                                        local_smem_wr_data <= wdata;
                                    end
                                    // Send to remote SMs
                                    if (target_mask != (8'b1 << SM_ID)) begin
                                        cluster_req_valid <= 1'b1;
                                        cluster_req_targets <= target_mask[NUM_SMs-1:0] & ~(1 << SM_ID);
                                        cluster_req_op <= 2'd1;  // ST
                                        cluster_req_addr <= smem_addr;
                                        cluster_req_data <= wdata;
                                        state <= ST_CLUSTER_REQ;
                                    end else begin
                                        state <= ST_LOCAL_WRITE;
                                    end
                                    `ifdef SIMULATION
                                    $display("[MULTIMEM%0d] ST multicast: targets=0x%02x addr=0x%04x data=0x%08x", // keep
                                             SM_ID, target_mask, smem_addr, wdata);
                                    `endif
                                end
                            end

                            `MULTIMEM_RED: begin
                                // Reduction: first read local value
                                if (local_target) begin
                                    local_smem_rd_en <= 1'b1;
                                    local_smem_rd_addr <= smem_addr;
                                    state <= ST_REDUCE_READ;
                                    `ifdef SIMULATION
                                    $display("[MULTIMEM%0d] RED: addr=0x%04x data=0x%08x op=%0d", // keep
                                             SM_ID, smem_addr, wdata, red_op);
                                    `endif
                                end else begin
                                    // Remote reduction
                                    cluster_req_valid <= 1'b1;
                                    cluster_req_targets <= target_mask[NUM_SMs-1:0];
                                    cluster_req_op <= 2'd2;  // RED
                                    cluster_req_addr <= smem_addr;
                                    cluster_req_data <= wdata;
                                    cluster_req_red_op <= red_op;
                                    state <= ST_CLUSTER_REQ;
                                end
                            end

                            default: begin
                                done <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_LOCAL_READ: begin
                    if (local_smem_rd_valid) begin
                        result_data <= local_smem_rd_data;
                        result_valid <= 1'b1;
                        done <= 1'b1;
                        remote_req_done <= is_remote_op;  // Signal remote completion
                        remote_resp_data <= local_smem_rd_data;
                        state <= ST_IDLE;
                        `ifdef SIMULATION
                        $display("[MULTIMEM%0d] LD done: data=0x%08x", SM_ID, local_smem_rd_data); // keep
                        `endif
                    end
                end

                ST_LOCAL_WRITE: begin
                    if (local_smem_wr_done) begin
                        done <= 1'b1;
                        remote_req_done <= is_remote_op;
                        state <= ST_IDLE;
                        `ifdef SIMULATION
                        $display("[MULTIMEM%0d] ST done", SM_ID); // keep
                        `endif
                    end
                end

                ST_CLUSTER_REQ: begin
                    if (cluster_req_ack) begin
                        cluster_req_valid <= 1'b0;
                        if (saved_func == `MULTIMEM_LD) begin
                            state <= ST_CLUSTER_RESP;  // Wait for response
                        end else begin
                            done <= 1'b1;
                            state <= ST_IDLE;  // ST/RED complete after ack
                        end
                    end
                end

                ST_CLUSTER_RESP: begin
                    if (cluster_resp_valid) begin
                        result_data <= cluster_resp_data;
                        result_valid <= 1'b1;
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                ST_REDUCE_READ: begin
                    if (local_smem_rd_valid) begin
                        // Compute reduction
                        reduce_acc <= do_reduction(local_smem_rd_data, saved_wdata, saved_red_op);
                        state <= ST_REDUCE_WRITE;
                    end
                end

                ST_REDUCE_WRITE: begin
                    local_smem_wr_en <= 1'b1;
                    local_smem_wr_addr <= saved_smem_addr;
                    local_smem_wr_data <= reduce_acc;
                    if (local_smem_wr_done) begin
                        local_smem_wr_en <= 1'b0;
                        result_data <= reduce_acc;
                        result_valid <= 1'b1;
                        done <= 1'b1;
                        remote_req_done <= is_remote_op;
                        state <= ST_IDLE;
                        `ifdef SIMULATION
                        $display("[MULTIMEM%0d] RED done: result=0x%08x", SM_ID, reduce_acc); // keep
                        `endif
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
