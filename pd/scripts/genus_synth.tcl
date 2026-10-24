# genus_synth.tcl - logic synthesis of one GPU block on ASAP7 (template)
#
# Run from the pd/ folder (the file lists point to the RTL in ../sim/), for example:
#   DESIGN=cu PERIOD=2.5 genus -files scripts/genus_synth.tcl -log logs/cu
# DESIGN : cu, tex_unit, rop, prim_unit, rasterizer, display_engine, gpu_core, gpu_top
# PERIOD : clock period in ns (2.5 = 400 MHz target; try 5.0 first for the rasterizer)
#
# EDIT the library path below to where ASAP7 is installed on your machine.

set DESIGN [expr {[info exists env(DESIGN)] ? $env(DESIGN) : "cu"}]
set PERIOD [expr {[info exists env(PERIOD)] ? $env(PERIOD) : 2.5}]

# ---- ASAP7 standard cells (7.5-track, regular Vt, typical corner) ----
set ASAP7_LIB_DIR "/path/to/asap7sc7p5t_28/LIB/NLDM"          ;# <-- EDIT ME
set_db init_lib_search_path $ASAP7_LIB_DIR
set_db library [glob -nocomplain $ASAP7_LIB_DIR/*RVT_TT*.lib*]
# ASAP7 library time unit is ps in some releases: check 'report_units' and the
# clock period below if timing numbers look 1000x off.

file mkdir reports outputs logs

# ---- read and elaborate the RTL ----
set fh [open filelists/$DESIGN.f r]
set files [split [string trim [read $fh]] "\n"]
close $fh
read_hdl -sv $files
elaborate $DESIGN
check_design -unresolved

# ---- constraints ----
create_clock -name clk -period $PERIOD [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_input_delay  [expr {0.2 * $PERIOD}] -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay [expr {0.2 * $PERIOD}] -clock clk [all_outputs]
set_false_path -from [get_ports rst]

# ---- synthesis ----
set_db syn_generic_effort medium
set_db syn_map_effort     medium
set_db syn_opt_effort     medium
syn_generic
syn_map
syn_opt

# ---- results ----
report_area   > reports/${DESIGN}_area.rpt
report_timing -max_paths 10 > reports/${DESIGN}_timing.rpt
report_power  > reports/${DESIGN}_power.rpt
report_gates  > reports/${DESIGN}_gates.rpt
write_hdl     > outputs/${DESIGN}_netlist.v
write_sdc     > outputs/${DESIGN}.sdc
puts "Done: see reports/${DESIGN}_*.rpt and outputs/${DESIGN}_netlist.v"
