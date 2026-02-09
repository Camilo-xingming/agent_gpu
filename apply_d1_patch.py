#!/usr/bin/env python3
import re

# Read the file
with open("rtl/streaming_multiprocessor_v2.v", "r") as f:
    lines = f.readlines()

# Step 1: Add warp_inst_valid_d1 definition after warp_inst_buf_valid
new_lines = []
added_d1_def = False
for i, line in enumerate(lines):
    new_lines.append(line)
    if not added_d1_def and "reg [NUM_WARPS-1:0] warp_inst_buf_valid;" in line:
        new_lines.append("    reg [NUM_WARPS-1:0] warp_inst_valid_d1;\n")
        added_d1_def = True
        print(f"✓ Added d1 definition at line {i+2}")

# Step 2: Add always block after the warp_inst_buf_valid update logic
# Find the end of the always block that updates warp_inst_buf_valid
in_always_block = False
block_depth = 0
added_always = False
lines = new_lines
new_lines = []

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Detect start of the buffer update always block
    if "always @(posedge clk or negedge rst_n) begin" in line and not added_always:
        in_always_block = True
        block_depth = 1
        continue
    
    if in_always_block:
        # Count begin/end to find block termination
        if "begin" in line:
            block_depth += 1
        if "end" in line:
            block_depth -= 1
            
        # When we exit the always block (block_depth == 0)
        if block_depth == 0 and not added_always:
            # Add the new always block right after
            new_lines.append("\n")
            new_lines.append("    // Delay warp_inst_buf_valid by 1 cycle to avoid same-cycle set/consume race\n")
            new_lines.append("    always @(posedge clk or negedge rst_n) begin\n")
            new_lines.append("        if (!rst_n) begin\n")
            new_lines.append("            warp_inst_valid_d1 <= {NUM_WARPS{1'b0}};\n")
            new_lines.append("        end else if (kernel_start) begin\n")
            new_lines.append("            warp_inst_valid_d1 <= {NUM_WARPS{1'b0}};\n")
            new_lines.append("        end else begin\n")
            new_lines.append("            warp_inst_valid_d1 <= warp_inst_buf_valid;\n")
            new_lines.append("        end\n")
            new_lines.append("    end\n")
            added_always = True
            print(f"✓ Added d1 always block at line {i+1}")
            in_always_block = False

# Step 3: Change scheduler connection
lines = new_lines
new_lines = []
for i, line in enumerate(lines):
    if ".warp_inst_valid(warp_inst_buf_valid)" in line:
        new_lines.append(line.replace("warp_inst_buf_valid", "warp_inst_valid_d1"))
        print(f"✓ Updated scheduler connection at line {i+1}")
    else:
        new_lines.append(line)

# Write back
with open("rtl/streaming_multiprocessor_v2.v", "w") as f:
    f.writelines(new_lines)

print("\n✅ Patch applied successfully!")
