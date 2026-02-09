#!/usr/bin/env python3
import sys

content = '''
    // Debug: detailed hazard breakdown for warp 0
    always @(posedge clk) begin
        if (warp_inst_valid[0] && warp_has_hazard[0]) begin
            $display("[%0t SCHED] HAZARD: rs1=R%0d(%b) rs2=R%0d(%b) rs3=R%0d(%b) rd=R%0d(%b) wr=%b",
                     $time, 
                     warp_rs1[0], scoreboard[0][warp_rs1[0]],
                     warp_rs2[0], scoreboard[0][warp_rs2[0]],
                     warp_rs3[0], scoreboard[0][warp_rs3[0]],
                     warp_rd[0], scoreboard[0][warp_rd[0]],
                     warp_writes_reg[0]);
        end
    end

'''

with open('rtl/blackwell_scheduler.v', 'r') as f:
    lines = f.readlines()

# Insert after line 212 (after the endgenerate)
output = lines[:213] + [content] + lines[213:]

with open('rtl/blackwell_scheduler.v', 'w') as f:
    f.writelines(output)

print("Added hazard detail debug")
