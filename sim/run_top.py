# run_top.py                                                         Phase 8, stage 4
# THE WHOLE GPU. A driver (this script) writes a command list, shaders, constants,
# vertex/index buffers and textures into memory. Then the GPU runs on its own:
# command processor -> input assembler -> vertex waves -> primitive unit -> rasterizer
# -> pixel waves (+ texture units) -> render output unit -> SWAP -> display engine.
# The golden model (GPU.start in gpu_model.py) runs the SAME memory image.
# Checked: every pixel of every finished frame, AND the video output: the pictures that
# appear on the "screen" must be exactly the golden frames, in order, none skipped,
# none torn, with no display underflow.
#
# Needs: assembler.py, simulator.py, gpu_model.py,
#        cu.v, tex_unit.v, prim_unit.v, rasterizer.v, rop.v, display_engine.v,
#        gpu_top.v, tb_gpu_top.v
# Use:   python run_top.py

import shutil
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont
from simulator import Memory
from assembler import assemble
import gpu_model as g

W, H, FOCAL = 160, 90, 90
BUFFERS = [("FB_A_ADDR", 0x140000), ("FB_B_ADDR", 0x160000),
           ("DEPTH_ADDR", 0x180000), ("STENCIL_ADDR", 0x1A0000)]   # inside 2 MB


def common_state(cl):
    for name, value in BUFFERS + [("VS_ADDR", g.VS_ADDR), ("PS_ADDR", g.PS_ADDR),
                                  ("VB_STRIDE", 24), ("VB_ATTRS", 6), ("VS_OUT_ATTRS", 3),
                                  ("VIEWPORT_W", W), ("VIEWPORT_H", H),
                                  ("PROJ_FOCAL", g.to_fx(FOCAL)), ("PROJ_CX", g.to_fx(W / 2)),
                                  ("PROJ_CY", g.to_fx(H / 2)), ("CULL_MODE", 1)]:
        cl.set_state(name, value)


def scene_uv_cubes():
    """Frame 1: UV cube, depth off. Frame 2: two intersecting cubes, depth on."""
    mem = Memory()
    mem.write_block(g.VS_ADDR, assemble(g.CUBE_VS)[0])
    mem.write_block(g.PS_ADDR, assemble(g.UV_PS)[0])
    g.write_constants(mem, g.CONST_ADDR, *g.cube_matrix(35, -25))
    g.write_constants(mem, g.SMALL_CUBE_CONST, *g.cube_matrix(70, -25, scale=0.55, offset=(1.0, 0.35, -0.3)))
    verts, indices = g.cube_vertex_data()
    for k, attrs in enumerate(verts):
        mem.write_block(g.VB_ADDR + k * 24, attrs)
    g.write_indices(mem, g.IB_ADDR, indices)
    cl = g.CommandList()
    common_state(cl)
    cl.set_state("VB_ADDR", g.VB_ADDR)
    cl.set_state("IB_ADDR", g.IB_ADDR)
    cl.set_state("CONST_ADDR", g.CONST_ADDR)
    cl.set_state("DEPTH_MODE", 0)
    cl.clear()
    cl.draw(count=len(indices))
    cl.swap()
    cl.set_state("DEPTH_MODE", 1)
    cl.clear(z=0)
    cl.draw(count=len(indices))
    cl.set_state("CONST_ADDR", g.SMALL_CUBE_CONST)
    cl.draw(count=len(indices))
    cl.swap()
    cl.fence(0x00F000, 0x600D)
    cl.end()
    mem.write_block(g.CMD_ADDR, cl.words)
    return mem


def scene_textured(angles=(30, 75)):
    """Textured cube spinning over a textured, mipmapped floor."""
    mem = Memory()
    mem.write_block(g.VS_ADDR, assemble(g.CUBE_VS)[0])
    mem.write_block(g.PS_ADDR, assemble(g.TEXTURED_PS)[0])
    tex_info = g.upload_texture(mem, g.TEX_ADDR, g.make_texture())
    cube_v, cube_i = g.cube_vertex_data()
    for k, attrs in enumerate(cube_v):
        mem.write_block(g.VB_ADDR + k * 24, attrs)
    g.write_indices(mem, g.IB_ADDR, cube_i)
    floor_v, floor_i = g.floor_vertex_data()
    for k, attrs in enumerate(floor_v):
        mem.write_block(g.FLOOR_VB + k * 24, attrs)
    g.write_indices(mem, g.FLOOR_IB, floor_i)
    g.write_constants(mem, g.FLOOR_CONST, *g.cube_matrix(0, -25))
    for f, angle in enumerate(angles):
        g.write_constants(mem, g.FRAME_CONSTS + 0x40 * f, *g.cube_matrix(angle, -25))
    cl = g.CommandList()
    common_state(cl)
    cl.set_state("DEPTH_MODE", 1)
    cl.set_state("TEX0_ADDR", g.TEX_ADDR)
    cl.set_state("TEX0_INFO", tex_info)
    for f in range(len(angles)):
        cl.clear(z=0)
        cl.set_state("VB_ADDR", g.FLOOR_VB)
        cl.set_state("IB_ADDR", g.FLOOR_IB)
        cl.set_state("CONST_ADDR", g.FLOOR_CONST)
        cl.draw(count=6)
        cl.set_state("VB_ADDR", g.VB_ADDR)
        cl.set_state("IB_ADDR", g.IB_ADDR)
        cl.set_state("CONST_ADDR", g.FRAME_CONSTS + 0x40 * f)
        cl.draw(count=36)
        cl.swap()
    cl.fence(0x00F000, 0x600D)
    cl.end()
    mem.write_block(g.CMD_ADDR, cl.words)
    return mem


def image_file(mem, path):
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def golden(mem):
    gpu = g.GPU(mem)
    gpu.start(g.CMD_ADDR)
    frames = []
    for f in gpu.frames:
        raw = f.tobytes()                                   # R, G, B, R, G, B, ...
        frames.append([tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)])
    return frames, gpu.stats, mem.load(0x00F000)


def run_rtl(path):
    out = subprocess.run(["vvp", "top_sim", f"+image={path}"], capture_output=True, text=True).stdout
    frames, stats, fence, status, bad, shown, video = [], None, None, "NO_OUTPUT", 0, [], None
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "FRAME":
            frames.append([])
        elif p[0] == "P" and frames:
            try:
                w = int(p[1], 16)
                frames[-1].append((w & 255, (w >> 8) & 255, (w >> 16) & 255))
            except ValueError:
                frames[-1].append(None)
        elif p[0] == "SHOWN":
            shown.append([])
        elif p[0] == "V" and shown:
            try:
                c = int(p[1], 16)
                shown[-1].append((c >> 16, (c >> 8) & 255, c & 255))
            except ValueError:
                shown[-1].append(None)
        elif p[0] == "VIDEO":
            video = [int(x) for x in p[1:4]]
        elif p[0] == "FENCE":
            fence = int(p[1], 16)
        elif p[0] == "STATS":
            stats = [int(x) for x in p[1:]]
        elif p[0] == "STATUS":
            status = p[1]
        elif p[0] == "BADCOUNT":
            bad = int(p[1])
    return frames, stats, fence, status, bad, shown, video


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake:
        if not shutil.which("iverilog"):
            print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
            return
        build = subprocess.run(["iverilog", "-g2012", "-o", "top_sim", "cu.v", "tex_unit.v", "prim_unit.v",
                                "rasterizer.v", "rop.v", "display_engine.v", "gpu_top.v", "tb_gpu_top.v"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
            return

    all_frames, all_ok = [], True
    for name, short, builder in [("UV cubes (depth off, then 2 cubes with depth)", "UV cubes", scene_uv_cubes),
                                 ("Textured cube over a mipmapped floor", "Textured", scene_textured)]:
        exp_frames, gs, exp_fence = golden(builder())
        if fake:
            got_frames, stats, fence, status, bad = exp_frames, None, exp_fence, "DONE", 0
            shown, video = list(exp_frames), [len(exp_frames), 0, 0]
        else:
            image_file(builder(), "scene.hex")
            got_frames, stats, fence, status, bad, shown, video = run_rtl("scene.hex")

        print(f"===== Full GPU: {name} =====")
        print(f"  RTL status        : {status}{'  !! ' + str(bad) + ' accesses outside memory' if bad else ''}")
        if stats:
            cyc, cmds, vw, pw, quads, drop, written, failed, tri_out, tri_back = stats
            print(f"  Cycles            : {cyc}")
            print(f"  Commands          : RTL {cmds}, golden {gs['commands']}")
            print(f"  Vertex waves      : RTL {vw}, golden {gs['vertex_waves']}")
            print(f"  Triangles         : RTL {tri_out} kept / {tri_back} culled, golden "
                  f"{gs['triangles_out']} / {gs['culled_back']}")
            print(f"  Quads             : RTL {quads} ({drop} early-depth drops), golden "
                  f"{gs['quads']} ({gs['quads_early_dropped']})")
            print(f"  Pixel waves       : RTL {pw}, golden {gs['pixel_waves']}")
            print(f"  Pixels written    : RTL {written} ({failed} failed depth), golden "
                  f"{gs['pixels_written']} ({gs['pixels_depth_failed']})")
        print(f"  Fence             : RTL 0x{fence or 0:X}, golden 0x{exp_fence:X}")
        ok = status == "DONE" and fence == exp_fence and len(got_frames) == len(exp_frames)
        for f, exp in enumerate(exp_frames):
            got = got_frames[f] if f < len(got_frames) else []
            bad_px = sum(1 for i in range(len(exp)) if i >= len(got) or got[i] != exp[i])
            ok &= bad_px == 0
            print(f"  Frame {f + 1}           : {len(exp) - bad_px} / {len(exp)} pixels match")
            all_frames.append((f"{short}, frame {f + 1}", exp, got))
        # ---- what appeared on the screen ----
        match = [next((k + 1 for k, e in enumerate(exp_frames) if pic == e), None) for pic in shown]
        in_order = match == list(range(1, len(exp_frames) + 1))
        if video:
            print(f"  Display           : {video[1]} display frames, underflow {video[2]}")
        print(f"  On screen         : {len(shown)} different pictures shown -> "
              f"{', '.join('frame ' + str(m) if m else 'UNKNOWN (torn?)' for m in match) or 'none'}")
        ok &= in_order and bool(video) and video[2] == 0
        print(f"  Screen check      : {'every frame shown in order, none torn' if in_order else 'WRONG'}")
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}\n")

    # picture: golden (top) vs RTL (bottom) for every frame
    S = 2
    sheet = Image.new("RGB", (len(all_frames) * (W * S + 10), 2 * (H * S + 30) + 10), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (title, exp, got) in enumerate(all_frames):
        for row, data, label in [(0, exp, "golden"), (1, got, "RTL")]:
            img = Image.new("RGB", (W, H))
            img.putdata([px if px else (255, 0, 255) for px in data] + [(255, 0, 255)] * (W * H - len(data)))
            x, y = k * (W * S + 10), row * (H * S + 30)
            sheet.paste(img.resize((W * S, H * S), Image.NEAREST), (x, y + 24))
            draw.text((x + 4, y + 4), f"{title} ({label})", fill=(44, 44, 42), font=font(12))
    sheet.save("full_gpu_result.png")
    print("Saved full_gpu_result.png")
    print("ALL PASS: the complete GPU matches the golden model, frames AND video output"
          if all_ok else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
