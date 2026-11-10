# R2a stretch: out-of-context synth (and optionally place+route) of the streaming GEMV core,
# for a first resource + timing estimate. No board, no bitstream; the numbers are PROJECTED.
#   vivado -mode batch -source fabric/big/tcl/ooc_gemv_stream.tcl \
#          -tclargs <NPORTS> <GMAX> <PERIOD_NS> <BDIR> <synth|impl> [ACT_RAM_STYLE]
# Defaults: 3 96 3.0 /tmp/kevbuild/gemv_stream_ooc synth distributed
# STATUS (2026-11-10): written but NEVER RUN -- Vivado could not check out its node-locked
# licence on the day (see the R2a report), so treat this script as untested. The RTL itself
# does parse and elaborate in Vivado's front end (xvlog -sv + xelab, no warnings).
# Only clk is constrained, so the report is register-to-register timing inside the core;
# the port-side paths (s_ready/s_valid skid, act_* write port, m_*) are registered in the RTL
# but their I/O timing is not covered here.
set nports [lindex $argv 0]
set gmax   [lindex $argv 1]
set period [lindex $argv 2]
set bdir   [lindex $argv 3]
set stage  [lindex $argv 4]
set ram    [lindex $argv 5]
if {$nports eq ""} { set nports 3 }
if {$gmax   eq ""} { set gmax 96 }
if {$period eq ""} { set period 3.0 }
if {$bdir   eq ""} { set bdir "/tmp/kevbuild/gemv_stream_ooc" }
if {$stage  eq ""} { set stage synth }
if {$ram    eq ""} { set ram distributed }

set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
file mkdir $bdir

set xdc "$bdir/ooc.xdc"
set f [open $xdc w]
puts $f "create_clock -name clk -period $period \[get_ports clk\]"
close $f

read_verilog "$root/fabric/big/rtl/gemv_stream.v"
read_xdc -mode out_of_context $xdc
synth_design -top gemv_stream -part $part -mode out_of_context \
    -generic NPORTS=$nports -generic GMAX=$gmax -generic ACT_RAM_STYLE=$ram

proc summary {tag bdir} {
    report_utilization    -file "$bdir/util_$tag.rpt"
    report_utilization    -hierarchical -file "$bdir/util_hier_$tag.rpt"
    report_timing_summary -file "$bdir/timing_$tag.rpt"
    report_timing -max_paths 10 -nworst 1 -file "$bdir/paths_$tag.rpt"
    set wns [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
    set lut  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == CLB && REF_NAME =~ LUT*}]]
    set lram [llength [get_cells -hier -filter {PRIMITIVE_GROUP == CLB && REF_NAME =~ RAM*}]]
    set ff   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == REGISTER}]]
    set dsp  [llength [get_cells -hier -filter {REF_NAME =~ DSP48E2*}]]
    set bram [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
    set c8   [llength [get_cells -hier -filter {REF_NAME == CARRY8}]]
    puts "OOC_GEMV_STREAM stage=$tag wns=$wns lut_cells=$lut lutram_cells=$lram ff=$ff carry8=$c8 dsp=$dsp bram=$bram"
}

summary synth $bdir
write_checkpoint -force "$bdir/synth.dcp"

if {$stage eq "impl"} {
    opt_design
    place_design
    phys_opt_design
    route_design
    summary route $bdir
    write_checkpoint -force "$bdir/route.dcp"
}
puts "OOC_GEMV_STREAM_DONE nports=$nports gmax=$gmax period=$period stage=$stage ram=$ram"
