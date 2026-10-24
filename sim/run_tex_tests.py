# run_tex_tests.py                                                       Phase 6 (early)
# Tests tex_unit.v against the golden texture unit (texture_unit() in gpu_model.py)
# with directed tests + hundreds of RANDOM requests covering:
#   RGBA8 and RGB565, repeat and clamp, point and bilinear, every mip level,
#   vertex waves (level 0) and pixel waves (level from the quad), huge and negative u, v.
#
# Needs: assembler.py, simulator.py, gpu_model.py, tex_unit.v, tb_tex.v
# Use:   python run_tex_tests.py

import random
import shutil
import subprocess
import sys
from simulator import Memory, u32
from gpu_model import GPU, upload_texture, make_texture

FX = 1 << 16
TEX_A = 0x100000            # 128 x 128 RGBA8 "GPU" texture, 8 mip levels
TEX_B = 0x140000            # 4 x 4 RGBA8, random colors, 1 level
TEX_C = 0x150000            # 8 x 8 RGB565, random colors, 4 mip levels
N_RANDOM = 400


def info(size_log2, levels, rgb565=0, clamp=0, bilinear=1, mipmaps=1):
    return size_log2 | levels << 4 | rgb565 << 8 | clamp << 9 | bilinear << 10 | mipmaps << 11


# The 4 slots: same or different textures with different settings
SLOTS = [
    (TEX_A, info(7, 8)),                                    # bilinear, repeat, mipmaps
    (TEX_A, info(7, 8, clamp=1, bilinear=0)),               # point, clamp, mipmaps
    (TEX_B, info(2, 1, clamp=1, mipmaps=0)),                # tiny, bilinear, clamp
    (TEX_C, info(3, 4, rgb565=1)),                          # RGB565, bilinear, mipmaps
]


def build_memory(rng):
    mem = Memory()
    upload_texture(mem, TEX_A, make_texture())
    mem.write_block(TEX_B, [rng.getrandbits(32) for _ in range(16)])
    halves = [rng.getrandbits(16) for _ in range(64 + 16 + 4 + 1)]          # 8x8 + 4x4 + 2x2 + 1x1
    halves.append(0)
    mem.write_block(TEX_C, [halves[i] | halves[i + 1] << 16 for i in range(0, len(halves), 2)])
    for s, (addr, inf) in enumerate(SLOTS):
        mem.store(0x0F010 + 4 * s, addr)
        mem.store(0x0F020 + 4 * s, inf)
    return mem


def quad(base, dx, dy):
    """u (or v) for the 4 lanes of a 2x2 quad: TL, TR, BL, BR."""
    return [u32(base), u32(base + dx), u32(base + dy), u32(base + dx + dy)]


def make_requests(rng):
    reqs = []
    c = lambda i: (i * 2 + 1) * FX // 8                      # texel centers of the 4x4 texture
    # ---- directed tests ----
    reqs.append(("point: texel center", 1, 0, [c(1)] * 4, [c(2)] * 4))
    reqs.append(("bilinear: texel centers of 4x4", 2, 0, [c(0), c(1), c(2), c(3)], [c(3), c(2), c(1), c(0)]))
    reqs.append(("bilinear: halfway between texels", 2, 0, [FX // 2] * 4, [c(0)] * 4))
    reqs.append(("clamp: far outside", 2, 0, [u32(-5 * FX), 7 * FX, u32(-FX), 3 * FX], [0, FX, 2 * FX, u32(-2 * FX)]))
    reqs.append(("repeat: u + 1.0 and u - 1.0", 0, 0, [FX // 3, FX // 3 + FX, u32(FX // 3 - FX), FX // 3], [FX // 5] * 4))
    reqs.append(("RGB565 bilinear", 3, 0, [c(0), c(1), FX // 2, 3 * FX // 7], [c(2), FX // 3, c(1), c(3)]))
    for level in range(9):                                   # pixel quads with growing footprint
        step = (1 << level) * FX // 128                      # 2^level texels per pixel
        reqs.append((f"mip: {1 << level} texel(s) per pixel", 0, 1,
                     quad(FX // 7, step, 0), quad(FX // 9, 0, step)))
    reqs.append(("huge u, v (overflow edges)", 0, 1, [0x7FFFFFFF, 0x80000000, 0x7FFF0000, 0x80010000],
                 [0x80000000, 0x7FFFFFFF, 0x12345678, 0xEDCBA987]))
    # ---- random tests ----
    for k in range(N_RANDOM):
        slot = rng.randrange(4)
        wave = rng.randrange(2)
        kind = rng.random()
        if kind < 0.1:                                       # anything at all
            us = [rng.getrandbits(32) for _ in range(4)]
            vs = [rng.getrandbits(32) for _ in range(4)]
        else:                                                # a real-looking quad
            scale = 1 << rng.randrange(0, 22)                # tiny to huge steps
            du, dv = rng.randint(-scale, scale), rng.randint(-scale, scale)
            bu, bv = rng.randint(-4 * FX, 4 * FX), rng.randint(-4 * FX, 4 * FX)
            us, vs = quad(bu, du, rng.randint(-scale, scale)), quad(bv, rng.randint(-scale, scale), dv)
        reqs.append((f"random {k}", slot, wave, us, vs))
    return reqs


def golden(mem, reqs):
    gpu = GPU(mem)
    for s, (addr, inf) in enumerate(SLOTS):
        gpu.state[f"TEX{s}_ADDR"], gpu.state[f"TEX{s}_INFO"] = addr, inf
    out = [gpu.texture_unit(slot, us, vs, wave) for _, slot, wave, us, vs in reqs]
    return out, gpu.stats["mip_levels_used"]


def write_image(mem, reqs, path):
    mem.store(0x0F000, len(reqs))
    for k, (_, slot, wave, us, vs) in enumerate(reqs):
        mem.write_block(0x0F100 + 64 * k, [slot | wave << 8] + list(us) + list(vs))
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def run_rtl(path):
    build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "tex_sim", "tex_unit.v", "tb_tex.v"],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
        sys.exit(1)
    if build.stderr.strip():
        print("Compiler warnings:\n" + build.stderr)
    out = subprocess.run(["vvp", "tex_sim", f"+image={path}"], capture_output=True, text=True).stdout
    results, info_lines = {}, {}
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "RESULT":
            try:
                results[int(p[1])] = [int(x, 16) for x in p[2:6]]
            except ValueError:
                results[int(p[1])] = None                    # x / z
        elif p and p[0] in ("CYCLES", "BADCOUNT", "STATUS"):
            info_lines[p[0]] = p[1]
    return results, info_lines


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake and not shutil.which("iverilog"):
        print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
        return
    rng = random.Random(2026)
    mem = build_memory(rng)
    reqs = make_requests(rng)
    expected, levels = golden(build_memory(random.Random(2026)), reqs)
    write_image(mem, reqs, "tex.hex")
    if fake:
        got, extra = {k: e for k, e in enumerate(expected)}, {"STATUS": "DONE", "CYCLES": "0"}
    else:
        got, extra = run_rtl("tex.hex")

    print(f"===== Texture unit RTL vs golden model: {len(reqs)} requests =====")
    fails = [k for k in range(len(reqs)) if got.get(k) != expected[k]]
    for k, (name, slot, wave, *_ ) in enumerate(reqs):
        if not name.startswith("random"):
            print(f"  {'PASS' if k not in fails else 'FAIL'}  {name}  (slot {slot}, "
                  f"{'pixel' if wave else 'vertex'} wave)")
    rnd = [k for k, r in enumerate(reqs) if r[0].startswith("random")]
    print(f"  {'PASS' if not set(rnd) & set(fails) else 'FAIL'}  {len(rnd)} random requests "
          f"({len(set(rnd) & set(fails))} mismatches)")
    for k in fails[:5]:
        g = got.get(k)
        print(f"      request {k} ({reqs[k][0]}):")
        print(f"        RTL      {[hex(x) for x in g] if g else g}")
        print(f"        expected {[hex(x) for x in expected[k]]}")
    print(f"\n  Mip levels exercised : {sorted(levels)}")
    if extra.get("CYCLES") and extra["CYCLES"] != "0":
        print(f"  Cycles               : {extra['CYCLES']} "
              f"({int(extra['CYCLES']) / len(reqs):.1f} per request, with random memory waits)")
    if extra.get("BADCOUNT", "0") != "0":
        print(f"  !! ALARM: {extra['BADCOUNT']} reads outside testbench memory")
    print(f"\n  {len(reqs) - len(fails)} / {len(reqs)} requests match the golden texture unit exactly")


if __name__ == "__main__":
    main()
