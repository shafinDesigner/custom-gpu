# run_textured.py                                                      Phase 3 + 6
# A TEXTURED quad drawn by your hardware: 2 triangles, each pixel shader
#   1. computes the 3 edge functions (software rasterizer),
#   2. turns them into weights and blends the corners' (u, v),
#   3. samples the texture with V_TEX (texture unit: bilinear + mipmaps),
#   4. kills pixels outside the triangle (AFTER V_TEX, so all 4 lanes have
#      correct u, v for the quad's mip level), and exports the color.
# The golden model (simulator.py + gpu_model.py texture unit) must match every pixel.
#
# Needs: assembler.py, simulator.py, gpu_model.py, run_phase3.py,
#        cu.v, tex_unit.v, gpu_core.v, tb_gpu_core.v
# Use:   python run_textured.py

import shutil
import subprocess
import sys
from PIL import Image, ImageDraw
from assembler import assemble
from simulator import Wave, Memory, PROGRAM_BASE, u32
from gpu_model import GPU, upload_texture, make_texture
from run_phase3 import W, H, CONST_BASE, BACKGROUND, edge, is_top_left, write_image, to_image, font

FX = 1 << 16
TEX_ADDR = 0x100000
TEX_SIZE = 64

TEXTURED_PS = """\
; Textured triangle: software rasterizer + V_TEX, one 2x2 quad per wave
; v0 = pixel x, v1 = pixel y
; s1 = constants: a0 b0 c0  a1 b1 c1  a2 b2 c2  KW  uA vA  uB vB  uC vC
;      edge i: E = a*x + b*y + c ;  weight = E x KW (16.16: 0 to 1)
        S_LOAD   s3,  [s1 + 0]
        S_LOAD   s4,  [s1 + 4]
        S_LOAD   s5,  [s1 + 8]
        S_LOAD   s6,  [s1 + 12]
        S_LOAD   s7,  [s1 + 16]
        S_LOAD   s8,  [s1 + 20]
        S_LOAD   s9,  [s1 + 24]
        S_LOAD   s10, [s1 + 28]
        S_LOAD   s11, [s1 + 32]
        V_MUL    v2, v0, s3           ; E0
        V_MUL    v3, v1, s4
        V_ADD    v2, v2, v3
        V_ADD    v2, v2, s5
        V_MUL    v4, v0, s6           ; E1
        V_MUL    v3, v1, s7
        V_ADD    v4, v4, v3
        V_ADD    v4, v4, s8
        V_MUL    v5, v0, s9           ; E2
        V_MUL    v3, v1, s10
        V_ADD    v5, v5, v3
        V_ADD    v5, v5, s11
        S_LOAD   s12, [s1 + 36]       ; KW
        V_MULFX  v6, v2, s12          ; weight of corner A
        V_MULFX  v7, v4, s12          ; weight of corner B
        V_MULFX  v8, v5, s12          ; weight of corner C
        S_LOAD   s13, [s1 + 40]       ; uA
        S_LOAD   s14, [s1 + 44]       ; vA
        V_MULFX  v9,  v6, s13         ; u = wA x uA
        V_MULFX  v10, v6, s14         ; v = wA x vA
        S_LOAD   s13, [s1 + 48]       ; uB
        S_LOAD   s14, [s1 + 52]       ; vB
        V_MADFX  v9,  v7, s13, v9     ;   + wB x uB
        V_MADFX  v10, v7, s14, v10
        S_LOAD   s13, [s1 + 56]       ; uC
        S_LOAD   s14, [s1 + 60]       ; vC
        V_MADFX  v9,  v8, s13, v9     ;   + wC x uC
        V_MADFX  v10, v8, s14, v10
        V_TEX    v11, v9, v10, 0      ; texture slot 0
        ; KILL only AFTER V_TEX: killed lanes stop computing, so their u, v
        ; would be wrong and the quad's mip level with them
        V_SLT    v12, v2, s0          ; outside if any E < 0
        V_SLT    v13, v4, s0
        V_OR     v12, v12, v13
        V_SLT    v13, v5, s0
        V_OR     v12, v12, v13
        V_KILL   v12
        EXPORT   COLOR, v11
        HALT
"""

# A quad made of 2 triangles sharing an edge, with texture corners (u, v)
P0, P1, P2, P3 = (6.25, 4.0), (58.75, 9.5), (55.5, 31.25), (20.125, 33.75)
UV0, UV1, UV2, UV3 = (0, 0), (1, 0), (1, 1), (0, 1)
TRIANGLES = [((P0, UV0), (P1, UV1), (P3, UV3)),
             ((P1, UV1), (P2, UV2), (P3, UV3))]


def triangle_constants(corners):
    """DRIVER: 16 words: 3 edges (a, b, c), KW, then (u, v) of corners A, B, C."""
    (a, uva), (b, uvb), (c, uvc) = [((round(x * 16), round(y * 16)), uv) for (x, y), uv in corners]
    area = edge(a, b, c)
    if area > 0:                                    # same winding for every triangle
        (b, uvb), (c, uvc) = (c, uvc), (b, uvb)
        area = -area
    words = []
    for p1, p2 in [(b, c), (c, a), (a, b)]:         # edge i -> weight of corner i
        A, B = p2[1] - p1[1], -(p2[0] - p1[0])
        C = -(A * p1[0] + B * p1[1])
        bias = 0 if is_top_left(p1, p2) else -1     # top-left rule
        words += [u32(16 * A), u32(16 * B), u32(8 * A + 8 * B + C + bias)]
    words.append((1 << 32) // (-area))              # KW: E x KW (16.16) = E / total
    for uv in (uva, uvb, uvc):
        words += [u32(round(uv[0] * FX)), u32(round(uv[1] * FX))]
    return words


def build_memory():
    mem = Memory()
    mem.write_block(PROGRAM_BASE, assemble(TEXTURED_PS)[0])
    for d, tri in enumerate(TRIANGLES):
        mem.write_block(CONST_BASE + 0x40 * d, triangle_constants(tri))
    texture = make_texture().resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
    info = upload_texture(mem, TEX_ADDR, texture)       # bilinear, repeat, mipmaps
    mem.store(0x0F010, TEX_ADDR)                         # slot 0 address
    mem.store(0x0F020, info)                             # slot 0 settings
    return mem, info


def golden(mem, info):
    gpu = GPU(mem)                                       # only its texture unit is used here
    gpu.state["TEX0_ADDR"], gpu.state["TEX0_INFO"] = TEX_ADDR, info
    fb, writes = [BACKGROUND] * (W * H), [0] * (W * H)
    waves = instructions = 0
    for d in range(len(TRIANGLES)):
        number = 0
        for qy in range(0, H, 2):
            for qx in range(0, W, 2):
                wave = Wave(mem, pc=PROGRAM_BASE, wave_type=1, texture_sampler=gpu.texture_unit)
                wave.v[0] = [qx, qx + 1, qx, qx + 1]
                wave.v[1] = [qy, qy, qy + 1, qy + 1]
                wave.s[1] = CONST_BASE + 0x40 * d
                wave.s[2] = number
                number += 1
                wave.run()
                waves += 1
                instructions += wave.count
                for lane, color in enumerate(wave.exports.get("COLOR", [None] * 4)):
                    if color is not None:
                        i = (qy + lane // 2) * W + qx + lane % 2
                        fb[i] = color
                        writes[i] += 1
    return {"fb": fb, "overdraw": sum(1 for n in writes if n > 1), "waves": waves,
            "instructions": instructions, "mips": gpu.stats["mip_levels_used"]}


def rtl(mem):
    build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "core_sim", "cu.v", "tex_unit.v",
                            "gpu_core.v", "tb_gpu_core.v"], capture_output=True, text=True)
    if build.returncode != 0:
        print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
        sys.exit(1)
    if build.stderr.strip():
        print("Compiler warnings:\n" + build.stderr)
    write_image(mem, "textured.hex")
    out = subprocess.run(["vvp", "core_sim", "+image=textured.hex", f"+draws={len(TRIANGLES)}"],
                         capture_output=True, text=True).stdout
    r = {"status": "NO_OUTPUT", "fb": [None] * (W * H)}
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "STATUS":
            r["status"] = p[1]
        elif p[0] == "CYCLES":
            r["cycles"] = int(p[1])
        elif p[0] == "WAVES":
            r["waves_cu"] = (int(p[1]), int(p[2]))
        elif p[0] == "CONFLICTS":
            r["conflicts"] = int(p[1])
        elif p[0] == "OVERDRAW":
            r["overdraw"] = int(p[1])
        elif p[0] == "BADCOUNT":
            r["bad_addresses"] = int(p[1])
        elif p[0] == "FB":
            try:
                r["fb"][int(p[1])] = int(p[2], 16)
            except ValueError:
                r["fb"][int(p[1])] = None
    return r


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake and not shutil.which("iverilog"):
        print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
        return
    mem, info = build_memory()
    gold = golden(mem, info)
    if fake:
        got = {"status": "DONE", "fb": list(gold["fb"]), "overdraw": gold["overdraw"]}
    else:
        got = rtl(build_memory()[0])

    bad = [i for i in range(W * H) if got["fb"][i] != gold["fb"][i]]
    print("===== Textured quad: 2 CUs + 2 texture units =====")
    print(f"  RTL status        : {got['status']}")
    if "waves_cu" in got:
        w0, w1 = got["waves_cu"]
        print(f"  Waves             : CU0 {w0} + CU1 {w1} = {w0 + w1}   (golden: {gold['waves']})")
    print(f"  Instructions      : {gold['instructions']} ({gold['instructions'] / gold['waves']:.0f} per wave)")
    if got.get("cycles"):
        print(f"  Cycles            : {got['cycles']}  ({got['cycles'] / gold['waves']:.1f} per wave)")
        print(f"  Memory conflicts  : {got['conflicts']} cycles where both sides wanted memory")
    used = gold["mips"]
    total = sum(used.values())
    print("  Mip levels used   : " + ", ".join(f"level {k}: {100 * v / total:.0f}%"
                                             for k, v in sorted(used.items())))
    if got.get("bad_addresses"):
        print(f"  !! ALARM          : {got['bad_addresses']} accesses outside testbench memory")
    print(f"  Overdraw          : RTL {got.get('overdraw')}, golden {gold['overdraw']}")
    ok = not bad and got["status"] == "DONE" and got.get("overdraw") == 0
    print(f"  Pixels            : {W * H - len(bad)} / {W * H} match the golden model "
          f"-> {'PASS' if ok else 'FAIL'}")
    for i in bad[:5]:
        g = "undefined" if got["fb"][i] is None else f"0x{got['fb'][i]:08X}"
        print(f"      pixel ({i % W}, {i // W}): RTL {g}, golden 0x{gold['fb'][i]:08X}")

    diff = [0xFF0000FF if i in set(bad) else 0xFF000000 for i in range(W * H)]
    panels = [(to_image(gold["fb"]), "Golden model (Python)"),
              (to_image(got["fb"]), "Your RTL: 2 CUs + 2 texture units"),
              (to_image(diff), f"Difference: {len(bad)} pixels")]
    pw, ph = W * 8, H * 8
    sheet = Image.new("RGB", (pw * 3 + 20, ph + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (img, title) in enumerate(panels):
        sheet.paste(img, (k * (pw + 10), 34))
        draw.text((k * (pw + 10) + 6, 8), title, fill=(44, 44, 42), font=font(16))
    sheet.save("textured_result.png")
    print("  Saved textured_result.png")


if __name__ == "__main__":
    main()
