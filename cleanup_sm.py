#!/usr/bin/env python3
"""Clean up unused tensor replay and suppress signals."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Remove tensor_lockout_suppress signals (no longer used directly)
sm = sm.replace(
    """
    // Signal to scheduler: lane1 tensor is suppressed by lockout (don't consume)
    wire tensor_lockout_suppress_lane1 = tensor_push_lane1_raw && tensor_push_locked[issue1_warp_id];
    wire tensor_lockout_suppress_lane0 = tensor_push_lane0_raw && tensor_push_locked[issue_warp_id];

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
    end""",
    ""
)

# Remove tensor_replay_mask from fetch pipeline instantiation
sm = sm.replace(
    '        .tensor_replay(tensor_replay_mask),\n        .decode_stalled_per_warp(decode_stalled_per_warp),',
    '        .tensor_replay({NUM_WARPS{1\'b0}}),\n        .decode_stalled_per_warp(decode_stalled_per_warp),'
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Cleanup done")
