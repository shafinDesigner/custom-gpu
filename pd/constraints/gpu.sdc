# gpu.sdc - basic constraints for any GPU block (400 MHz target)
# Every block has one clock port 'clk' and a synchronous reset 'rst'.
create_clock -name clk -period 2.5 [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_input_delay  0.5 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.5 -clock clk [all_outputs]
set_false_path -from [get_ports rst]
