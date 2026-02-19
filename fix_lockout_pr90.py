#!/usr/bin/env python3
"""Fix PR #90: suppress only issue_consume_r for locked tensor warps.

When the scheduler selects a tensor warp that is locked out by the
2-cycle push lockout, suppress the consume so the instruction stays
in the buffer and PC doesn't advance. The instruction still enters
decode/issue pipeline but the tensor push is suppressed by lockout
at the SM level — this is harmless since tensor-specific side effects
(pending_fu_count, scoreboard set) are all gated by tensor_issue_push_fire."""

# === Fix blackwell_scheduler.v ===
with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'r') as f:
    sched = f.read()

# 1. Add tensor_push_locked input port
sched = sched.replace(
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,',
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,\n    input  wire [NUM_WARPS-1:0]     tensor_push_locked,  // 2-cycle lockout from SM'
)

# 2. Add consume suppression after FU conflict block
# Only suppress issue_consume_r, NOT issue_valid_r
old_block = """                sched_fu_conflict = 1'b1;
                issue_valid_r[1] = 1'b0;  // Suppress slot 1
                issue_consume_r[issue_warp_r[1]] = 1'b0;  // Don't consume slot 1's instruction
            end
        end
end"""

new_block = """                sched_fu_conflict = 1'b1;
                issue_valid_r[1] = 1'b0;  // Suppress slot 1
                issue_consume_r[issue_warp_r[1]] = 1'b0;  // Don't consume slot 1's instruction
            end
        end

        // Tensor lockout guard: when a tensor warp is selected but locked out,
        // suppress the consume so the instruction stays in the buffer and PC
        // doesn't advance. The instruction still flows through decode/issue
        // but the tensor push is harmlessly suppressed at the SM level.
        for (s = 0; s < NUM_SCHEDULERS; s = s + 1) begin
            if (issue_valid_r[s] && warp_is_tensor[issue_warp_r[s]] &&
                tensor_push_locked[issue_warp_r[s]]) begin
                issue_consume_r[issue_warp_r[s]] = 1'b0;
            end
        end
end"""

sched = sched.replace(old_block, new_block)

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'w') as f:
    f.write(sched)
print("Scheduler: tensor lockout consume guard added")

# === Wire tensor_push_locked to scheduler ===
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Wire to first (blackwell) scheduler instance only
sm = sm.replace(
    '        .warp_is_tensor(pd_is_tensor),\n        .warp_is_memory(pd_is_memory),',
    '        .warp_is_tensor(pd_is_tensor),\n        .tensor_push_locked(tensor_push_locked),\n        .warp_is_memory(pd_is_memory),',
    1  # only first occurrence
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("SM: tensor_push_locked wired to scheduler")
