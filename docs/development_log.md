# Development log

How the GPU was built, in order, with the problems found along the way.

## 1. Learning the pipeline in software (`learning/`)

A software renderer in 7 steps: edge-function triangle fill, barycentric color
interpolation, 3D transform with perspective and culling, depth buffer, perspective-correct
textures, bilinear filtering and mipmaps, and finally fixed-point (integer-only) math, the way
the hardware calculates. A separate study showed why the top-left fill rule is needed
(shared edges drawn twice or left as holes otherwise).

## 2. Specification

Architecture specification v0.1: targets (720p30, 2 CUs x 4 lanes), number formats,
programming model, ISA (52 instructions, 4 formats), command list and state registers,
block specifications, memory map, performance budget, verification plan, roadmap.

## 3. Tools and golden model

| Tool | Result |
|---|---|
| Assembler + disassembler | Round-trip checked |
| Instruction simulator | 25 instruction tests |
| Full GPU golden model | Command list in, frames out; textured animated scenes |

Design bugs found by the tools before any hardware existed:

| # | Found by | Problem | Fix |
|---|---|---|---|
| 1 | Assembler | Spec example used a constant where the format needs a register | Load constant into a scalar register |
| 2 | Simulator | `V_CMUL`: white x white = 254, colors darkened | `(a x (b + 1)) >> 8` |
| 3 | Golden model | Alpha 255 did not give exactly the source color | `a' = a + (a >> 7)` |
| 4 | Textured test | `V_KILL` before `V_TEX` gave wrong mip levels at triangle edges | Sample before kill; helper-lane semantics noted as open question |

## 4. RTL, block by block

Each block was first emulated in Python with its exact bit widths, then written in Verilog
and compared with the golden model.

| Step | Block | Test |
|---|---|---|
| Compute unit v0 | Non-pipelined CU | 20 programs, bit-exact on the first run |
| Compute unit v1 | Special function unit (bit-serial divider, square root) | Algorithms proven on 120,014 values first; 30 programs |
| 2-CU core | Work distributor, memory arbiter | First image from hardware: 2304 / 2304 pixels |
| Texture unit | Mipmaps, bilinear, RGB565 | 416 requests with random memory stalls |
| Rasterizer | Setup, quads, early depth, interpolation | ~21,000 quads |
| Render output unit | Depth, blending, color mask, clear | 2,500 random operations, whole buffers |
| Primitive unit | Near plane, projection, culling | 720 random triangles |
| Complete GPU | Command processor, input assembler, all blocks | 2 scenes, 4 frames, every pixel |
| Display engine | 720p timing, flip at vertical blank | Cycle-exact; real 720p60 frame |
| GPU + display | Video output | Pictures on screen in order, no tearing |

## 5. Verilog problems found by running the tests

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| Testbench memory too small | Every pixel black, every pixel written twice | Constants at 0x20000 wrapped to address 0 | Larger memory + out-of-range address alarm |
| Rasterizer would not compile | Syntax errors on 9 lines | `cover` is a SystemVerilog keyword | Renamed to `covmask`; all files scanned for keywords |

## 6. Lessons

- Build the golden model before the hardware, and make it use the hardware's number formats.
- Prove arithmetic algorithms in Python before writing them in RTL.
- Randomized tests with random stalls find timing and handshake bugs that directed tests miss.
- The testbench and the model must agree on everything, including the memory map.
- Measure before optimizing: the cycle counts point to one dominant bottleneck.
