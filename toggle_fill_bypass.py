#!/usr/bin/env python3
"""Toggle fill bypass on/off in sm_fetch_pipeline.v"""
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else 'off'

with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'r') as f:
    code = f.read()

if mode == 'off':
    # Disable fill bypass: remove fill_into_empty from fast valid
    code = code.replace(
        """            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            assign warp_inst_valid_fast[upi] = warp_inst_buf_valid[upi]
                                             | nib_will_serve[upi]
                                             | fill_into_empty;""",
        """            // wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi]; // DISABLED for baseline
            assign warp_inst_valid_fast[upi] = warp_inst_buf_valid[upi]
                                             | nib_will_serve[upi];"""
    )
    # Disable fill bypass in data mux
    code = code.replace(
        """            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            assign warp_inst_buf_fast_flat[32*upi +: 32] =
                fill_into_empty      ? fill_data :
                nib_will_serve[upi]  ? warp_next_inst[upi] :
                                       warp_inst_buf[upi];""",
        """            // wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi]; // DISABLED for baseline
            assign warp_inst_buf_fast_flat[32*upi +: 32] =
                nib_will_serve[upi]  ? warp_next_inst[upi] :
                                       warp_inst_buf[upi];"""
    )
    print("Fill bypass DISABLED (NIB-only baseline)")
elif mode == 'on':
    # Re-enable fill bypass
    code = code.replace(
        """            // wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi]; // DISABLED for baseline
            assign warp_inst_valid_fast[upi] = warp_inst_buf_valid[upi]
                                             | nib_will_serve[upi];""",
        """            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            assign warp_inst_valid_fast[upi] = warp_inst_buf_valid[upi]
                                             | nib_will_serve[upi]
                                             | fill_into_empty;"""
    )
    code = code.replace(
        """            // wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi]; // DISABLED for baseline
            assign warp_inst_buf_fast_flat[32*upi +: 32] =
                nib_will_serve[upi]  ? warp_next_inst[upi] :
                                       warp_inst_buf[upi];""",
        """            wire fill_into_empty = warp_fill[upi] & ~warp_inst_buf_valid[upi];
            assign warp_inst_buf_fast_flat[32*upi +: 32] =
                fill_into_empty      ? fill_data :
                nib_will_serve[upi]  ? warp_next_inst[upi] :
                                       warp_inst_buf[upi];"""
    )
    print("Fill bypass ENABLED")

with open('/Users/jerry/RalphGPU/rtl/sm_fetch_pipeline.v', 'w') as f:
    f.write(code)
