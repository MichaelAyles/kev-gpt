# Mamba-2 WAVE engine bitstream — PS+PL block design around mamba_pipe_axi (the
# pipelined NC-stream engine, config-B NST=32, DBG=0). Build-only; run
# impl_mamba_pipe.tcl afterwards for synth/impl/bitstream.
#   vivado -mode batch -source build_bd_mamba_pipe.tcl -tclargs <FREQMHZ> <BDIR> <NC> <NST> <TMAX> <QH>
# seed.mem is only the sim/fallback init — the real rsqrt seed loads over AXI
# (WSEL_SEED=15), so the bitstream does NOT depend on it baking.
set freq [lindex $argv 0]
set bdir [lindex $argv 1]
set nc   [lindex $argv 2]
set nst  [lindex $argv 3]
set tmax [lindex $argv 4]
set qh   [lindex $argv 5]
set dbg  [lindex $argv 6]
if {$freq eq ""} { set freq 100 }
if {$bdir eq ""} { set bdir "/tmp/kevbuild/mamba_pipe_bit" }
if {$nc   eq ""} { set nc 2 }
if {$nst  eq ""} { set nst 32 }
if {$tmax eq ""} { set tmax 2 }
if {$qh   eq ""} { set qh 16 }
if {$dbg  eq ""} { set dbg 0 }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]

file mkdir $bdir
create_project mamba_pipe_pl "$bdir/mamba_pipe_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/mamba_pipe_axi.v" \
    "$root/fabric/stage3/rtl/mamba_pipe.sv" \
    "$root/fabric/stage3/rtl/gemv_i4i8.sv" \
    "$root/fabric/stage3/rtl/conv_silu.sv" \
    "$root/fabric/stage3/rtl/ssm_scan_row.sv" \
    "$root/fabric/stage3/rtl/rmsnorm_gated.sv" \
    "$root/fabric/stage3/seed.mem"]
set_property file_type {Memory Initialization Files} [get_files "$root/fabric/stage3/seed.mem"]
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq] [get_bd_cells ps]

set g [create_bd_cell -type module -reference mamba_pipe_axi eng]
set_property -dict [list CONFIG.C_S_AXI_ADDR_WIDTH {8} \
    CONFIG.NC $nc CONFIG.NST $nst CONFIG.QH $qh CONFIG.DBG $dbg \
    CONFIG.TMAX $tmax CONFIG.T_TOKENS $tmax] [get_bd_cells eng]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/eng/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /eng/S_AXI]

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */eng/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
puts "BD_BUILD_MAMBA_PIPE_OK freq=$freq nc=$nc nst=$nst tmax=$tmax qh=$qh"
