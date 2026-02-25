# Basic timing constraint for DE10-Nano demo top.
create_clock -name sys_clk -period 20.000 [get_ports {CLOCK_50}]
