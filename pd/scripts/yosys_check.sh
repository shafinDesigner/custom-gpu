#!/bin/sh
# yosys_check.sh - free synthesizability check + rough cell count (no ASAP7 needed)
# Usage (from the pd/ folder):  apt-get install -y yosys   then   sh scripts/yosys_check.sh cu
D=${1:-cu}
yosys -q -l "yosys_$D.log" -p "read_verilog -sv $(tr '\n' ' ' < filelists/$D.f); synth -top $D; stat" \
  && echo "OK: $D is synthesizable (cell counts in yosys_$D.log)" \
  || echo "PROBLEM: see yosys_$D.log"
