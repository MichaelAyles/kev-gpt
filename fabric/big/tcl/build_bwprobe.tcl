# R0 bandwidth probe — PS + bwprobe_axi, all six PS slave ports at 128 bits,
# then synth + impl + bitstream in one go.
#   vivado -mode batch -source fabric/big/tcl/build_bwprobe.tcl -tclargs <FREQMHZ> <BDIR>
# Master -> port map: m0..m3 -> S_AXI_HP0..HP3_FPD, m4 -> HPC0, m5 -> HPC1.
set freq [lindex $argv 0]
set bdir [lindex $argv 1]
if {$freq eq ""} { set freq 300 }
if {$bdir eq ""} { set bdir "/tmp/kevbuild/big_r0_bit" }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]

file mkdir $bdir
create_project bwprobe "$bdir/bwprobe" -part $part -force
set_property board_part $board [current_project]
add_files -norecurse [list \
    "$root/fabric/big/rtl/bwprobe_axi.v" \
    "$root/fabric/big/rtl/axi_rd_probe.v"]
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq \
    CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    ] [get_bd_cells ps]

set g [create_bd_cell -type module -reference bwprobe_axi eng]
set_property -dict [list CONFIG.C_S_AXI_ADDR_WIDTH {10}] [get_bd_cells eng]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/eng/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /eng/S_AXI]

# direct master -> PS slave port connections, no interconnect in the data path
set map {m0_axi S_AXI_HP0_FPD m1_axi S_AXI_HP1_FPD m2_axi S_AXI_HP2_FPD \
         m3_axi S_AXI_HP3_FPD m4_axi S_AXI_HPC0_FPD m5_axi S_AXI_HPC1_FPD}
foreach {m s} $map {
    connect_bd_intf_net [get_bd_intf_pins eng/$m] [get_bd_intf_pins ps/$s]
}
foreach clkpin {saxihpc0_fpd_aclk saxihpc1_fpd_aclk saxihp0_fpd_aclk \
                saxihp1_fpd_aclk saxihp2_fpd_aclk saxihp3_fpd_aclk} {
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/$clkpin]
}

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */eng/*] }
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces eng/*]] { puts "ADDRSEG $seg" }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
puts "BD_BUILD_BWPROBE_OK freq=$freq"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH_FAILED"; error "synth" }
puts "SYNTH_OK"
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "IMPL_FAILED"; error "impl" }

open_run impl_1
report_utilization    -file "$bdir/util_impl.rpt"
report_timing_summary -file "$bdir/timing_impl.rpt"
set wns [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
set bit "$bdir/bwprobe/bwprobe.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/bwprobe.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/bwprobe" -quiet
    puts "BITSTREAM -> $bdir/bwprobe.bit (wns=$wns)"
} else { puts "NO_BITSTREAM" }
puts "BWPROBE_BUILD_DONE wns=$wns"
