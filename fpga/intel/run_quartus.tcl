# Usage:
#   quartus_sh -t fpga/intel/run_quartus.tcl <out_dir>
# Example:
#   quartus_sh -t fpga/intel/run_quartus.tcl build/fpga/intel

package require ::quartus::project
package require ::quartus::flow

set script_dir [file dirname [info script]]
set repo_root  [file normalize [file join $script_dir ../..]]

if {[llength $argv] >= 1} {
    set out_dir [file normalize [lindex $argv 0]]
} else {
    set out_dir [file normalize [file join $repo_root build fpga intel]]
}

set project_name ralphgpu_de10_nano_demo
set project_dir  [file join $out_dir project]

file mkdir $out_dir
file mkdir $project_dir

set old_pwd [pwd]
cd $project_dir

puts "[clock format [clock seconds]] : RalphGPU Intel flow"
puts "repo_root=$repo_root"
puts "out_dir=$out_dir"

project_new $project_name -overwrite

set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEBA6U23I7
set_global_assignment -name TOP_LEVEL_ENTITY de10_nano_demo_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY $out_dir
set_global_assignment -name NUM_PARALLEL_PROCESSORS 4

set_global_assignment -name VERILOG_FILE [file join $repo_root fpga/common/uart_tx.v]
set_global_assignment -name VERILOG_FILE [file join $repo_root fpga/common/ralph_gpu_fpga_demo_top.v]
set_global_assignment -name VERILOG_FILE [file join $repo_root fpga/intel/de10_nano_demo_top.v]
set_global_assignment -name SDC_FILE     [file join $repo_root fpga/intel/de10_nano_demo.sdc]

export_assignments
execute_flow -analysis_and_synthesis

project_close
cd $old_pwd
