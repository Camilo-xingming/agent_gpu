`timescale 1ns/1ps
//============================================================================
// sm_special_reg.v — Special Register Execution Unit
//
// Reads PTX special registers: %tid.x, %ctaid.x, %ntid.x, %laneid, etc.
// 1-cycle latency (combinational decode + 1 pipeline register stage).
// Also captures block/grid dimension registers on kernel_start.
//
// Part of RALPH-6 God Module Refactor.
//============================================================================
`include "gpu_defines.vh"

module sm_special_reg #(
    parameter NUM_LANES  = `THREADS_PER_WARP,
    parameter SIMD_WIDTH = NUM_LANES * 32,
    parameter WARP_ID_W  = $clog2(`WARPS_PER_SM),
    parameter SM_ID      = 0
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Issue interface (muxed slot 0/1)
    input  wire                     issue_valid,      // special_reg_issue
    input  wire                     use_slot0,        // special_reg_issue0
    input  wire [WARP_ID_W-1:0]    slot0_warp_id,
    input  wire [WARP_ID_W-1:0]    slot1_warp_id,
    input  wire [4:0]              slot0_rd,
    input  wire [4:0]              slot1_rd,
    input  wire [4:0]              slot0_ra,          // special reg code
    input  wire [4:0]              slot1_ra,
    input  wire [NUM_LANES-1:0]    slot0_mask,
    input  wire [NUM_LANES-1:0]    slot1_mask,

    // Active mask for the selected warp
    input  wire [NUM_LANES-1:0]    warp_active_mask,

    // Kernel configuration (latched on kernel_start)
    input  wire                     kernel_start,
    input  wire [31:0]             block_id_x,
    input  wire [31:0]             block_id_y,
    input  wire [31:0]             block_id_z,
    input  wire [31:0]             block_dim_x,
    input  wire [31:0]             block_dim_y,
    input  wire [31:0]             block_dim_z,
    input  wire [31:0]             grid_dim_x,
    input  wire [31:0]             grid_dim_y,
    input  wire [31:0]             grid_dim_z,

    // Pipeline output (1-cycle latency)
    output wire                     valid_out,
    output reg  [WARP_ID_W-1:0]    warp_out,
    output reg  [4:0]              rd_out,
    output reg  [NUM_LANES-1:0]    mask_out,
    output reg  [SIMD_WIDTH-1:0]   result_out
);

    //------------------------------------------------------------------------
    // Kernel configuration register capture
    //------------------------------------------------------------------------
    reg [31:0] block_id_regs  [0:2];
    reg [31:0] block_dim_regs [0:2];
    reg [31:0] grid_dim_regs  [0:2];

    always @(posedge clk) begin
        if (kernel_start) begin
            block_id_regs[0]  <= block_id_x;
            block_id_regs[1]  <= block_id_y;
            block_id_regs[2]  <= block_id_z;
            block_dim_regs[0] <= block_dim_x;
            block_dim_regs[1] <= block_dim_y;
            block_dim_regs[2] <= block_dim_z;
            grid_dim_regs[0]  <= grid_dim_x;
            grid_dim_regs[1]  <= grid_dim_y;
            grid_dim_regs[2]  <= grid_dim_z;
        end
    end

    //------------------------------------------------------------------------
    // Slot mux (select between dual-issue slot 0 and slot 1)
    //------------------------------------------------------------------------
    wire [WARP_ID_W-1:0]  issue_warp = use_slot0 ? slot0_warp_id : slot1_warp_id;
    wire [4:0]            issue_rd   = use_slot0 ? slot0_rd       : slot1_rd;
    wire [4:0]            issue_ra   = use_slot0 ? slot0_ra       : slot1_ra;
    wire [NUM_LANES-1:0]  issue_mask = use_slot0 ? slot0_mask     : slot1_mask;

    //------------------------------------------------------------------------
    // Combinational special register decode
    //------------------------------------------------------------------------
    wire is_tid_x  = (issue_ra == `SREG_TID_X);
    wire is_laneid = (issue_ra == `SREG_LANEID);

    reg [31:0] reg_scalar;
    always @(*) begin
        case (issue_ra)
            `SREG_TID_X:      reg_scalar = 32'd0;  // Per-lane below
            `SREG_TID_Y:      reg_scalar = 32'd0;
            `SREG_TID_Z:      reg_scalar = 32'd0;
            `SREG_CTAID_X:    reg_scalar = block_id_regs[0];
            `SREG_CTAID_Y:    reg_scalar = block_id_regs[1];
            `SREG_CTAID_Z:    reg_scalar = block_id_regs[2];
            `SREG_NTID_X:     reg_scalar = block_dim_regs[0];
            `SREG_NTID_Y:     reg_scalar = block_dim_regs[1];
            `SREG_NTID_Z:     reg_scalar = block_dim_regs[2];
            `SREG_NCTAID_X:   reg_scalar = grid_dim_regs[0];
            `SREG_NCTAID_Y:   reg_scalar = grid_dim_regs[1];
            `SREG_NCTAID_Z:   reg_scalar = grid_dim_regs[2];
            `SREG_WARPID:     reg_scalar = {{(32-WARP_ID_W){1'b0}}, issue_warp};
            `SREG_SMID:       reg_scalar = SM_ID;
            `SREG_ACTIVEMASK: reg_scalar = {{(32-NUM_LANES){1'b0}}, warp_active_mask};
            default:          reg_scalar = 32'd0;
        endcase
    end

    //------------------------------------------------------------------------
    // Per-lane result generation
    // %tid.x: each lane gets warp_id * NUM_LANES + lane_index
    // %laneid: each lane gets its lane index
    // Others: scalar replicated across all lanes
    //------------------------------------------------------------------------
    wire [SIMD_WIDTH-1:0] result;
    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : gen_lane
            assign result[i*32 +: 32] =
                is_tid_x  ? (issue_warp * NUM_LANES + i) :
                is_laneid ? i[31:0] :
                reg_scalar;
        end
    endgenerate

    //------------------------------------------------------------------------
    // 1-stage pipeline register
    //------------------------------------------------------------------------
    reg valid_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 1'b0;
            warp_out   <= {WARP_ID_W{1'b0}};
            rd_out     <= 5'b0;
            mask_out   <= {NUM_LANES{1'b0}};
            result_out <= {SIMD_WIDTH{1'b0}};
        end else begin
            valid_pipe <= issue_valid;
            if (issue_valid) begin
                warp_out   <= issue_warp;
                rd_out     <= issue_rd;
                mask_out   <= issue_mask;
                result_out <= result;
            end
        end
    end

    assign valid_out = valid_pipe;

endmodule
