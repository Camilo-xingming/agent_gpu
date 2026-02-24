#!/usr/bin/env python3

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False

for i, line in enumerate(lines):
    if skip:
        if 'if (m_axi_bvalid && m_axi_bready) begin' in line:
            skip = False
            new_lines.append(line)
        continue

    if 'if (m_axi_awvalid && m_axi_awready) begin' in line:
        # Found the start of the write block we want to replace
        new_lines.append('            // Modified AXI write model to handle simultaneous AW/W\n')
        new_lines.append('            if (m_axi_awvalid && m_axi_awready) begin\n')
        new_lines.append('                pending_axi_addr <= m_axi_awaddr;\n')
        new_lines.append('                pending_axi_write <= 1\'b1;\n')
        new_lines.append('            end\n\n')
        
        new_lines.append('            if (m_axi_wvalid && m_axi_wready) begin\n')
        new_lines.append('                if (pending_axi_write) begin\n')
        new_lines.append('                    if (pending_axi_addr >= GMEM_BASE) begin\n')
        new_lines.append('                        global_mem[(pending_axi_addr - GMEM_BASE) >> 2] <= m_axi_wdata;\n')
        new_lines.append('                    end\n')
        new_lines.append('                    pending_axi_write <= 1\'b0;\n')
        new_lines.append('                    m_axi_bvalid <= 1\'b1;\n')
        new_lines.append('                    m_axi_bid <= m_axi_awid;\n') # Use stored ID if we tracked it, but here simplistic
        new_lines.append('                end else if (m_axi_awvalid && m_axi_awready) begin\n')
        new_lines.append('                    if (m_axi_awaddr >= GMEM_BASE) begin\n')
        new_lines.append('                        global_mem[(m_axi_awaddr - GMEM_BASE) >> 2] <= m_axi_wdata;\n')
        new_lines.append('                    end\n')
        new_lines.append('                    pending_axi_write <= 1\'b0;\n')
        new_lines.append('                    m_axi_bvalid <= 1\'b1;\n')
        new_lines.append('                    m_axi_bid <= m_axi_awid;\n')
        new_lines.append('                end\n')
        new_lines.append('            end\n')
        
        skip = True # Skip lines until we find the next block
    else:
        new_lines.append(line)

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.writelines(new_lines)

print("Fixed AXI model")
