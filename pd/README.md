# Custom GPU - physical design package (ASAP7)

Scripts and constraints for synthesis and place-and-route of the GPU.
The RTL itself lives in `../sim/` (one copy only, the same files the tests verify);
the file lists here point to it. Testbenches (`tb_*.v`) are never part of synthesis.

## Folder

| Folder | Contents |
|---|---|
| `filelists/` | One file list per synthesis top (`<top>.f`), pointing to `../sim/*.v` |
| `constraints/gpu.sdc` | Basic clock / IO constraints, 400 MHz |
| `scripts/genus_synth.tcl` | Genus synthesis template for ASAP7 (edit the library path) |
| `scripts/yosys_check.sh` | Free synthesizability check with Yosys (works in Colab) |

## Synthesis tops

| Top | Contains | Registers (estimate) | Notes |
|---|---|---|---|
| `cu` | One compute unit | ~3,000 bits | Good first block. ~9 multipliers of 32x32 (4 lanes) |
| `tex_unit` | Texture unit | ~700 bits | Small. Bilinear blend multipliers (8x9 bit) |
| `rop` | Render output unit | ~700 bits | Small |
| `prim_unit` | Primitive unit | ~1,600 bits | 32x32 multipliers, 68-bit edge function |
| `rasterizer` | Rasterizer | ~5,200 bits | **Largest arithmetic**: see below |
| `display_engine` | Display engine | ~82,000 bits | **Line buffer must become SRAM** |
| `gpu_core` | 2 CUs + 2 texture units | ~8,000 bits | Good intermediate top |
| `gpu_top` | The complete GPU | ~196,000 bits incl. all blocks | **Needs SRAM macros** first |

Register estimates are counted from the RTL declarations (before synthesis) with default
parameters (720p display, `MAXV` = 256).

## Things to know before PD

**1. Two large memories must be SRAM macros, not flip-flops**

| Memory | Block | Size | As flip-flops |
|---|---|---|---|
| `lbuf` (display line buffers) | `display_engine` | 2,560 x 32 | 81,920 flops |
| `vob` (vertex output buffer) | `gpu_top` | 2,816 x 32 | 90,112 flops |
| `vreg` (vector registers) | `cu` | 64 x 32 per CU | 2,048 flops (OK as flops, or a small register-file macro) |

ASAP7 has no SRAM compiler. Options: use the "fakeram" ASAP7 macros from the OpenROAD
flow scripts, or, for a first trial, reduce the parameters:
`gpu_top #(.MAXV(64), .H_ACTIVE(160), .V_ACTIVE(90), ...)` makes both memories small.

**2. The rasterizer has very wide arithmetic**

It keeps results exact for any 32-bit input, so some products are up to 118 bits wide and
many are computed in one cycle (12 edge functions, 12 weight products, 12 attribute products).
Expect a large area and long paths. Plan: synthesize it at a relaxed clock first
(`PERIOD=5.0`, 200 MHz), measure, then narrow the widths (guard band) or split the work over
more cycles in a later RTL version.

**3. Other notes**

- One clock (`clk`), synchronous active-high reset (`rst`), no latches intended.
- The display engine uses the core clock (a real chip needs a 74.25 MHz pixel clock domain
  with an asynchronous FIFO; not in this version).
- Memory interfaces are simple request/grant ports; there is no DRAM controller.
- ASAP7: check whether your LEF/GDS release uses the 4x scaled dimensions for place-and-route.

## Recommended order

1. `cu` - first full RTL-to-GDS run on your own design
2. `tex_unit`, `rop` - small blocks, quick wins
3. `prim_unit`
4. `rasterizer` - relaxed clock first, then improve
5. `gpu_core` - two CUs and two texture units together
6. `display_engine` and `gpu_top` - after SRAM macros (or reduced parameters)

## How to run

Run everything from this `pd/` folder. Free check (no ASAP7 needed):
```
apt-get install -y yosys
sh scripts/yosys_check.sh cu
```

Genus on ASAP7 (after editing the library path in the script):
```
DESIGN=cu PERIOD=2.5 genus -files scripts/genus_synth.tcl
```
Reports go to `reports/`, the netlist and SDC to `outputs/` - ready for Innovus.
