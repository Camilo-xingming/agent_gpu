# Usage:
#   vivado -mode batch -source fpga/xilinx/run_vivado.tcl -tclargs <out_dir> [run_impl]
# Example:
#   vivado -mode batch -source fpga/xilinx/run_vivado.tcl -tclargs build/fpga/xilinx 0

set script_dir [file dirname [info script]]
set repo_root  [file normalize [file join $script_dir ../..]]

if {[llength $argv] >= 1} {
    set out_dir [file normalize [lindex $argv 0]]
} else {
    set out_dir [file normalize [file join $repo_root build fpga xilinx]]
}

if {[llength $argv] >= 2} {
    set run_impl [lindex $argv 1]
} else {
    set run_impl 0
}

set part_name  xc7a100tcsg324-1
set top_module arty_a7_demo_top

file mkdir $out_dir
puts "[clock format [clock seconds]] : RalphGPU Xilinx flow"
puts "repo_root=$repo_root"
puts "out_dir=$out_dir"
puts "run_impl=$run_impl"

read_verilog [file join $repo_root fpga/common/uart_tx.v]
read_verilog [file join $repo_root fpga/common/ralph_gpu_fpga_demo_top.v]
read_verilog [file join $repo_root fpga/xilinx/arty_a7_demo_top.v]
read_xdc     [file join $repo_root fpga/xilinx/arty_a7_demo.xdc]

synth_design -top $top_module -part $part_name
opt_design

report_utilization    -file [file join $out_dir ${top_module}_utilization_synth.rpt]
report_timing_summary -file [file join $out_dir ${top_module}_timing_synth.rpt]
write_checkpoint -force [file join $out_dir ${top_module}_synth.dcp]
write_edif       -force [file join $out_dir ${top_module}.edf]

if {$run_impl == 1 || $run_impl == "1"} {
    place_design
    route_design
    report_utilization    -file [file join $out_dir ${top_module}_utilization_impl.rpt]
    report_timing_summary -file [file join $out_dir ${top_module}_timing_impl.rpt]
    write_bitstream -force [file join $out_dir ${top_module}.bit]
}

exit
