# run_phase3.py                                                          Phase 3
# The first image drawn by YOUR hardware.
#   - A pixel shader (your ISA) rasterizes a triangle in software: it computes the
#     3 edge functions for its 2x2 quad, kills pixels outside, and blends 3 colors.
#   - gpu_core.v runs it on 2 CUs for every quad of a 64 x 36 screen, twice
#     (2 triangles sharing an edge).
#   - The golden model (simulator.py) does the same; every pixel must match.
#
# Needs: assembler.py, simulator.py, cu.v, tex_unit.v, gpu_core.v, tb_gpu_core.v
# Use:   python run_phase3.py

import shutil
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont
from assembler import assemble
from simulator import Wave, Memory, PROGRAM_BASE, u32

W, H = 64, 36                     # must match tb_gpu_core.v
CONST_BASE = 0x20000              # draw d uses CONST_BASE + 0x40 * d
BACKGROUND = 0xFF1E1E1E
SUB = 4                           # 12.4 screen positions

# =====================================================================
# 1. THE PIXEL SHADER: a software rasterizer in your ISA
# =====================================================================
TRIANGLE_PS = """\
; Software rasterizer for one triangle, one 2x2 quad per wave.
; v0 = pixel x, v1 = pixel y  (set by the work distributor)
; s1 = constants: a0 b0 c0  a1 b1 c1  a2 b2 c2  K
;      edge i:  E = a*x + b*y + c   (>= 0 inside; top-left rule already in c)
;      color:   channel = (E x K) >> 16   (K = 255 x 65536 / triangle size)
        S_LOAD   s3,  [s1 + 0]
        S_LOAD   s4,  [s1 + 4]
        S_LOAD   s5,  [s1 + 8]
        S_LOAD   s6,  [s1 + 12]
        S_LOAD   s7,  [s1 + 16]
        S_LOAD   s8,  [s1 + 20]
        S_LOAD   s9,  [s1 + 24]
        S_LOAD   s10, [s1 + 28]
        S_LOAD   s11, [s1 + 32]
        S_LOAD   s12, [s1 + 36]
        V_MUL    v2, v0, s3           ; E0 = a0*x + b0*y + c0
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
        V_SLT    v6, v2, s0           ; outside if any E < 0
        V_SLT    v7, v4, s0
        V_OR     v6, v6, v7
        V_SLT    v7, v5, s0
        V_OR     v6, v6, v7
        V_KILL   v6                   ; turn off lanes outside the triangle
        S_ADDI   s13, s0, 16
        V_MUL    v8, v2, s12          ; red   = weight of corner A
        V_SRA    v8, v8, s13
        V_MUL    v9, v4, s12          ; green = weight of corner B
        V_SRA    v9, v9, s13
        V_MUL    v10, v5, s12         ; blue  = weight of corner C
        V_SRA    v10, v10, s13
        V_PACK   v11, v8, v9, v10
        EXPORT   COLOR, v11
        HALT
"""

# Two triangles sharing the edge (58.75, 9.5) - (20.125, 33.75)
TRIANGLES = [
    [(6.25, 4.0), (58.75, 9.5), (20.125, 33.75)],
    [(58.75, 9.5), (55.5, 31.25), (20.125, 33.75)],
]


# =====================================================================
# 2. DRIVER: turn a triangle into the shader's constants
# =====================================================================
def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def is_top_left(p1, p2):
    A, B = p2[1] - p1[1], -(p2[0] - p1[0])
    return A > 0 or (A == 0 and B > 0)


def triangle_constants(corners):
    a, b, c = [(round(x * 16), round(y * 16)) for x, y in corners]     # 12.4
    area = edge(a, b, c)
    if area > 0:                                    # same winding for every triangle
        b, c = c, b
        area = -area
    words = []
    for p1, p2 in [(b, c), (c, a), (a, b)]:         # edge i -> weight of corner i
        A, B = p2[1] - p1[1], -(p2[0] - p1[0])      # E(p) = A*px + B*py + C (12.4 units)
        C = -(A * p1[0] + B * p1[1])
        bias = 0 if is_top_left(p1, p2) else -1     # top-left rule
        # pixel center: px = 16x + 8, py = 16y + 8  ->  E = 16A*x + 16B*y + (8A + 8B + C)
        words += [u32(16 * A), u32(16 * B), u32(8 * A + 8 * B + C + bias)]
    words.append((255 << 16) // (-area))            # K
    return words


def build_memory():
    mem = Memory()
    mem.write_block(PROGRAM_BASE, assemble(TRIANGLE_PS)[0])
    for d, tri in enumerate(TRIANGLES):
        mem.write_block(CONST_BASE + 0x40 * d, triangle_constants(tri))
    return mem


# =====================================================================
# 3. GOLDEN MODEL: same work, in simulator.py
# =====================================================================
def golden(mem):
    fb, writes = [BACKGROUND] * (W * H), [0] * (W * H)
    waves = instructions = 0
    for d in range(len(TRIANGLES)):
        number = 0
        for qy in range(0, H, 2):
            for qx in range(0, W, 2):
                wave = Wave(mem, pc=PROGRAM_BASE, wave_type=1)
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
            "instructions": instructions}


# =====================================================================
# 4. RTL run
# =====================================================================
def write_image(mem, path):
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def rtl(mem):
    build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "core_sim",
                            "cu.v", "tex_unit.v", "gpu_core.v", "tb_gpu_core.v"], capture_output=True, text=True)
    if build.returncode != 0:
        print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
        sys.exit(1)
    if build.stderr.strip():
        print("Compiler warnings:\n" + build.stderr)
    write_image(mem, "scene.hex")
    out = subprocess.run(["vvp", "core_sim", "+image=scene.hex", f"+draws={len(TRIANGLES)}"],
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
        elif p[0] == "BADADDR":
            r.setdefault("bad_examples", []).append(" ".join(p[1:]))
        elif p[0] == "FB":
            try:
                r["fb"][int(p[1])] = int(p[2], 16)
            except ValueError:
                r["fb"][int(p[1])] = None          # x / z = undefined
    return r


# =====================================================================
# 5. Compare and save the picture
# =====================================================================
def to_image(fb, scale=8):
    img = Image.new("RGB", (W, H))
    img.putdata([(c & 255, (c >> 8) & 255, (c >> 16) & 255) if c is not None else (255, 0, 255)
                 for c in fb])
    return img.resize((W * scale, H * scale), Image.NEAREST)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake and not shutil.which("iverilog"):
        print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
        return
    mem = build_memory()
    gold = golden(mem)
    got = ({"status": "DONE", "fb": list(gold["fb"]), "overdraw": gold["overdraw"],
            "cycles": 0, "waves_cu": (gold["waves"] // 2, gold["waves"] // 2), "conflicts": 0}
           if fake else rtl(build_memory()))

    bad = [i for i in range(W * H) if got["fb"][i] != gold["fb"][i]]
    print("===== Phase 3: two CUs draw two triangles =====")
    print(f"  RTL status        : {got['status']}")
    if "waves_cu" in got:
        w0, w1 = got["waves_cu"]
        print(f"  Waves             : CU0 {w0} + CU1 {w1} = {w0 + w1}   (golden: {gold['waves']})")
        print(f"  Instructions      : {gold['instructions']} ({gold['instructions'] // gold['waves']} per wave)")
    if got.get("cycles"):
        print(f"  Cycles            : {got['cycles']}  ({got['cycles'] / gold['waves']:.1f} per wave, "
              f"2 CUs in parallel)")
        print(f"  Memory conflicts  : {got['conflicts']} cycles where both CUs wanted memory")
    if got.get("bad_addresses"):
        print(f"  !! ALARM          : {got['bad_addresses']} accesses OUTSIDE testbench memory, e.g. "
              f"{', '.join(got.get('bad_examples', []))}")
    print(f"  Overdraw          : RTL {got.get('overdraw')}, golden {gold['overdraw']} "
          f"(pixels written twice; must be 0)")
    ok = not bad and got["status"] == "DONE" and got.get("overdraw") == 0
    print(f"  Pixels            : {W * H - len(bad)} / {W * H} match the golden model "
          f"-> {'PASS' if ok else 'FAIL'}")
    for i in bad[:5]:
        g = "undefined" if got["fb"][i] is None else f"0x{got['fb'][i]:08X}"
        print(f"      pixel ({i % W}, {i // W}): RTL {g}, golden 0x{gold['fb'][i]:08X}")

    # Picture: golden | RTL | difference (red = mismatch)
    diff = [0xFF0000FF if i in set(bad) else 0xFF000000 for i in range(W * H)]
    panels = [(to_image(gold["fb"]), "Golden model (Python)"),
              (to_image(got["fb"]), "Your RTL (2 CUs)"),
              (to_image(diff), f"Difference: {len(bad)} pixels")]
    pw, ph = W * 8, H * 8
    sheet = Image.new("RGB", (pw * 3 + 20, ph + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (img, title) in enumerate(panels):
        sheet.paste(img, (k * (pw + 10), 34))
        draw.text((k * (pw + 10) + 6, 8), title, fill=(44, 44, 42), font=font(16))
    sheet.save("phase3_result.png")
    print("  Saved phase3_result.png")


if __name__ == "__main__":
    main()
