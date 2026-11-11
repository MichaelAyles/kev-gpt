# R2a on silicon — PS + gemv_axi (AXI-Lite slave + NPORTS 128-bit read masters straight into
# the PS HP ports), then synth + impl + bitstream in one go.
#   vivado -mode batch -source fabric/big/tcl/build_gemv_axi.tcl -tclargs <NPORTS> <FREQMHZ> <BDIR> [GMAX]
# Master -> port map (R0's best sets): 3 ports = HP0, HP1, HP3 (aux on HP2); 4 ports = HP0..HP3
# (aux on HPC0).
set nports [lindex $argv 0]
set freq   [lindex $argv 1]
set bdir   [lindex $argv 2]
set gmax   [lindex $argv 3]
if {$nports eq ""} { set nports 3 }
if {$freq eq ""}   { set freq 333 }
if {$bdir eq ""}   { set bdir "/tmp/kevbuild/big_r2a_bit" }
if {$gmax eq ""}   { set gmax 96 }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]

switch $nports {
    3 { set slaves {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP3_FPD}; set auxs S_AXI_HP2_FPD }
    4 { set slaves {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP2_FPD S_AXI_HP3_FPD}; set auxs S_AXI_HPC0_FPD }
    default { error "NPORTS 3 or 4" }
}
# aux_axi (activation fetch + result write-back, a few MB per token) gets the port the weight
# stream does not use: HP2 shares a DDR-controller port with HP1, HPC0 is the fallback.
# PS slave-port index (PSU__USE__S_AXI_GPn) and clock pin of each S_AXI_*_FPD interface
array set gpn {S_AXI_HPC0_FPD 0 S_AXI_HPC1_FPD 1 S_AXI_HP0_FPD 2 S_AXI_HP1_FPD 3 S_AXI_HP2_FPD 4 S_AXI_HP3_FPD 5}
array set clk {S_AXI_HPC0_FPD saxihpc0_fpd_aclk S_AXI_HPC1_FPD saxihpc1_fpd_aclk S_AXI_HP0_FPD saxihp0_fpd_aclk \
               S_AXI_HP1_FPD saxihp1_fpd_aclk S_AXI_HP2_FPD saxihp2_fpd_aclk S_AXI_HP3_FPD saxihp3_fpd_aclk}

file mkdir $bdir
create_project gemv_axi "$bdir/gemv_axi" -part $part -force
set_property board_part $board [current_project]
set srcs [list "$root/fabric/big/rtl/gemv_axi.v" "$root/fabric/big/rtl/gemv_stream.v"]
foreach f {axi_rd_stream.v axi_aux_dma.v} {
    if {[file exists "$root/fabric/big/rtl/$f"]} { lappend srcs "$root/fabric/big/rtl/$f" }
}
add_files -norecurse $srcs
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} [get_bd_cells ps]
set cfg [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq]
foreach s [concat $slaves $auxs] {
    lappend cfg CONFIG.PSU__USE__S_AXI_GP$gpn($s) {1} CONFIG.PSU__SAXIGP$gpn($s)__DATA_WIDTH {128}
}
set_property -dict $cfg [get_bd_cells ps]

set g [create_bd_cell -type module -reference gemv_axi eng]
set_property -dict [list CONFIG.NPORTS $nports CONFIG.GMAX $gmax] [get_bd_cells eng]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/eng/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /eng/S_AXI]

# direct master -> PS slave port connections, no interconnect in the data path
set k 0
foreach s $slaves {
    connect_bd_intf_net [get_bd_intf_pins eng/m${k}_axi] [get_bd_intf_pins ps/$s]
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/$clk($s)]
    incr k
}
if {[llength [get_bd_intf_pins -quiet eng/aux_axi]]} {
    connect_bd_intf_net [get_bd_intf_pins eng/aux_axi] [get_bd_intf_pins ps/$auxs]
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/$clk($auxs)]
}

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */eng/*] }
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces eng/*]] {
    puts "ADDRSEG $seg [get_property offset $seg] [get_property range $seg]"
}
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/*]] {
    puts "ADDRSEG $seg [get_property offset $seg] [get_property range $seg]"
}
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
puts "BD_BUILD_GEMV_AXI_OK nports=$nports freq=$freq gmax=$gmax"

set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
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
set whs [get_property SLACK [get_timing_paths -delay_type min -nworst 1]]
set bit "$bdir/gemv_axi/gemv_axi.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv_axi.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv_axi.bit" -quiet
    puts "BITSTREAM -> $bdir/gemv_axi.bit.bin (wns=$wns whs=$whs)"
} else { puts "NO_BITSTREAM" }
puts "GEMV_AXI_BUILD_DONE nports=$nports freq=$freq wns=$wns whs=$whs"
