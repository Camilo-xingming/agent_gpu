#!/usr/bin/env python3
"""Fix tensor lockout v5: replay suppressed tensor instructions.

When tensor push is locked out but the instruction was already consumed,
replay the instruction by re-asserting warp_inst_buf_valid next cycle.

The replay signal is: tensor_push_lane{0,1}_raw && !tensor_push_lane{0,1}
This fires exactly when a tensor op reaches the issue stage but can't push."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Add tensor replay signals after the lockout definitions
old_suppress = "    wire tensor_lockout_suppress_lane1 = tensor_push_lane1_raw && tensor_push_locked[issue1_warp_id];\n    wire tensor_lockout_suppress_lane0 = tensor_push_lane0_raw && tensor_push_locked[issue_warp_id];"

new_suppress = old_suppress + """

    // Tensor lockout replay: when tensor push is suppressed but instruction was consumed,
    // re-inject into fetch pipeline's inst_buf_valid on the next cycle
    reg [NUM_WARPS-1:0] tensor_replay_mask;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tensor_replay_mask <= {NUM_WARPS{1'b0}};
        end else begin
            tensor_replay_mask <= {NUM_WARPS{1'b0}};
            // Lane 0: tensor selected but locked out
            if (tensor_lockout_suppress_lane0) begin
                tensor_replay_mask[issue_warp_id] <= 1'b1;
            end
            // Lane 1: tensor selected but locked out
            if (tensor_lockout_suppress_lane1) begin
                tensor_replay_mask[issue1_warp_id] <= 1'b1;
            end
        end
    end"""

sm = sm.replace(old_suppress, new_suppress)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("SM: tensor replay mask added")

# Now wire the replay mask to sm_fetch_pipeline
with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'r') as f:
    fp = f.read()

# Add tensor_replay input port
fp = fp.replace(
    '    input  wire [NUM_WARPS-1:0]     warp_inst_consume,',
    '    input  wire [NUM_WARPS-1:0]     warp_inst_consume,\n    input  wire [NUM_WARPS-1:0]     tensor_replay,'
)

# In the valid bit management, add replay: force valid back to 1
# Find the block where valid is cleared on consume
fp = fp.replace(
    """            if (warp_fill[w_buf])
                warp_inst_buf_valid[w_buf] <= 1'b1;
            else if (warp_inst_consume_gated[w_buf] && !branch_flush_mask[w_buf])
                warp_inst_buf_valid[w_buf] <= 1'b0;""",
    """            if (tensor_replay[w_buf])
                warp_inst_buf_valid[w_buf] <= 1'b1;  // Replay: tensor push was locked out
            else if (warp_fill[w_buf])
                warp_inst_buf_valid[w_buf] <= 1'b1;
            else if (warp_inst_consume_gated[w_buf] && !branch_flush_mask[w_buf])
                warp_inst_buf_valid[w_buf] <= 1'b0;"""
)

with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'w') as f:
    f.write(fp)
print("Fetch pipeline: tensor replay input added")

# Wire tensor_replay in SM
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Find the sm_fetch_pipeline instantiation and add the new port
sm = sm.replace(
    '        .warp_inst_consume(warp_inst_consume),\n        .decode_stalled_per_warp(decode_stalled_per_warp),',
    '        .warp_inst_consume(warp_inst_consume),\n        .tensor_replay(tensor_replay_mask),\n        .decode_stalled_per_warp(decode_stalled_per_warp),'
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("SM: tensor_replay_mask wired to fetch pipeline")
