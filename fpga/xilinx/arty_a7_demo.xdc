## Basic timing constraint for Arty A7 demo top.
create_clock -name sys_clk -period 10.000 [get_ports CLK100MHZ]

## Optional pin constraints for Arty A7-100T (verify against board revision before implementation).
## set_property PACKAGE_PIN E3 [get_ports CLK100MHZ]
## set_property PACKAGE_PIN C2 [get_ports CPU_RESETN]
## set_property PACKAGE_PIN H5 [get_ports {LED[0]}]
## set_property PACKAGE_PIN J5 [get_ports {LED[1]}]
## set_property PACKAGE_PIN T9 [get_ports {LED[2]}]
## set_property PACKAGE_PIN T10 [get_ports {LED[3]}]
## set_property PACKAGE_PIN D10 [get_ports UART_TXD]
## set_property IOSTANDARD LVCMOS33 [get_ports {CLK100MHZ CPU_RESETN LED[*] UART_TXD}]
