#!/usr/bin/env python3
"""Remove unused tensor_replay port."""

# Remove from sm_fetch_pipeline.v
with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'r') as f:
    fp = f.read()
fp = fp.replace(
    '    input  wire [NUM_WARPS-1:0]     warp_inst_consume,\n    input  wire [NUM_WARPS-1:0]     tensor_replay,',
    '    input  wire [NUM_WARPS-1:0]     warp_inst_consume,'
)
with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'w') as f:
    f.write(fp)

# Remove from SM instantiation
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()
sm = sm.replace(
    "        .tensor_replay({NUM_WARPS{1'b0}}),\n        .decode_stalled_per_warp(decode_stalled_per_warp),",
    "        .decode_stalled_per_warp(decode_stalled_per_warp),"
)
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Removed tensor_replay port")
