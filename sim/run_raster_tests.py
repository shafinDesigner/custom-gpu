# run_raster_tests.py                                               Phase 5, stage 1
# Tests rasterizer.v against the golden rasterizer (GPU.rasterizer in gpu_model.py).
# Every quad the rasterizer hands to pixel shading must match exactly:
#   position, coverage mask (incl. early depth), closeness per lane, every attribute.
# Two runs: early depth OFF, and early depth ON with a random depth buffer.
#
# Needs: assembler.py, simulator.py, gpu_model.py, rasterizer.v, tb_raster.v
# Use:   python run_raster_tests.py

import random
import shutil
import subprocess
import sys
from simulator import Memory, u32
from gpu_model import GPU, edge, write16

W, H = 64, 36
DEPTH_ADDR = 0x080000
N_TRIANGLES = 150


# ---------------------------------------------------------------------
# Random triangles (screen space, 12.4) of many kinds
# ---------------------------------------------------------------------
def random_triangles(rng):
    tris = []

    def px(v):                                   # pixels -> 12.4
        return int(round(v * 16))

    def add(pts):
        if edge(*pts) == 0:                      # the primitive unit never sends these
            return
        n = rng.randint(0, 8)
        tris.append({"pts": pts,
                     "z": [rng.randint(2000, 65000) for _ in range(3)],
                     "attrs": [[rng.randint(-(1 << 22), 1 << 22) for _ in range(n)] for _ in range(3)]})

    while len(tris) < N_TRIANGLES:
        kind = rng.random()
        if kind < 0.35:                          # normal, partly off-screen
            add([(px(rng.uniform(-6, W + 6)), px(rng.uniform(-6, H + 6))) for _ in range(3)])
        elif kind < 0.50:                        # tiny
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            add([(px(cx + rng.uniform(-2, 2)), px(cy + rng.uniform(-2, 2))) for _ in range(3)])
        elif kind < 0.60:                        # huge, far off-screen
            add([(px(rng.uniform(-3000, 3000)), px(rng.uniform(-3000, 3000))) for _ in range(3)])
        elif kind < 0.70:                        # sliver
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            add([(px(x), px(y)), (px(rng.uniform(0, W)), px(rng.uniform(0, H))),
                 (px(x + rng.uniform(-0.3, 0.3)), px(y + rng.uniform(-0.3, 0.3)))])
        elif kind < 0.85:                        # edges exactly through pixel centers
            pts = [(rng.randrange(-4, W + 4) * 16 + 8, rng.randrange(-4, H + 4) * 16 + 8)
                   for _ in range(3)]
            if rng.random() < 0.5:               # make one edge horizontal or vertical
                pts[1] = (pts[1][0], pts[0][1]) if rng.random() < 0.5 else (pts[0][0], pts[1][1])
            add(pts)
        else:                                    # two triangles sharing an edge
            a, b = [(px(rng.uniform(0, W)), px(rng.uniform(0, H))) for _ in range(2)]
            c, d = [(px(rng.uniform(-4, W + 4)), px(rng.uniform(-4, H + 4))) for _ in range(2)]
            add([a, b, c])
            add([b, a, d])
    return tris[:N_TRIANGLES]


# ---------------------------------------------------------------------
# Golden model: GPU.rasterizer, recording what it sends to pixel shading
# ---------------------------------------------------------------------
def golden(tris, depth_mode, depth_values):
    mem = Memory()
    for i, d in enumerate(depth_values):
        write16(mem, DEPTH_ADDR + 2 * i, d)
    gpu = GPU(mem)
    gpu.state.update({"VIEWPORT_W": W, "VIEWPORT_H": H, "DEPTH_MODE": depth_mode,
                      "DEPTH_ADDR": DEPTH_ADDR})
    quads = []

    def record(lanes, cover, close, attrs, prim_id):
        quads.append((prim_id, lanes[0][0], lanes[0][1], sum(1 << l for l in range(4) if cover[l]),
                      list(close), [list(r) for r in attrs]))
    gpu.pixel_shading = record
    for t, tri in enumerate(tris):
        verts = [{"x": x, "y": y, "inv_z": z, "attrs_z": [tri["attrs"][v][k] for k in range(len(tri["attrs"][0]))]}
                 for v, ((x, y), z) in enumerate(zip(tri["pts"], tri["z"]))]
        gpu.rasterizer(verts, t)
    return quads, gpu.stats


def write_image(tris, depth_mode, depth_values, path):
    mem = Memory()
    for i, d in enumerate(depth_values):
        write16(mem, DEPTH_ADDR + 2 * i, d)
    mem.write_block(0x0F000, [len(tris), W, H, depth_mode, DEPTH_ADDR])
    for t, tri in enumerate(tris):
        n = len(tri["attrs"][0])
        words = [t, n] + [u32(p[0]) for p in tri["pts"]] + [u32(p[1]) for p in tri["pts"]] + tri["z"]
        for v in range(3):
            words += [u32(a) for a in tri["attrs"][v]] + [0] * (8 - n)
        mem.write_block(0x40000 + 256 * t, words)
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def run_rtl(path, tris):
    out = subprocess.run(["vvp", "raster_sim", f"+image={path}"], capture_output=True, text=True).stdout
    quads, stats = [], None
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "QUAD":
            try:
                prim = int(p[1])
                n = len(tris[prim]["attrs"][0])
                close = [int(x, 16) for x in p[5:9]]
                flat = [int(x, 16) for x in p[9:9 + 32]]
                attrs = [flat[4 * k: 4 * k + 4] for k in range(n)]
                quads.append((prim, int(p[2]), int(p[3]), int(p[4], 16), close, attrs))
            except (ValueError, IndexError):
                quads.append(("undefined", line.strip()[:80]))
        elif p and p[0] == "STATS":
            stats = [int(x) for x in p[1:4]]
    return quads, stats


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake:
        if not shutil.which("iverilog"):
            print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
            return
        build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "raster_sim", "rasterizer.v",
                                "tb_raster.v"], capture_output=True, text=True)
        if build.returncode != 0:
            print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
            return
        if build.stderr.strip():
            print("Compiler warnings:\n" + build.stderr)

    rng = random.Random(2027)
    tris = random_triangles(rng)
    all_ok = True
    for depth_mode in (0, 1):
        depth_values = [rng.randint(0, 40000) for _ in range(W * H)]
        expected, gstats = golden(tris, depth_mode, depth_values)
        if fake:
            got, stats = list(expected), [gstats["quads"], gstats["quads_early_dropped"], 0]
        else:
            write_image(tris, depth_mode, depth_values, "raster.hex")
            got, stats = run_rtl("raster.hex", tris)

        title = "early depth OFF" if depth_mode == 0 else "early depth ON (random depth buffer)"
        print(f"===== Rasterizer RTL vs golden: {len(tris)} triangles, {title} =====")
        mismatch = None
        for k in range(max(len(got), len(expected))):
            g = got[k] if k < len(got) else None
            e = expected[k] if k < len(expected) else None
            if g != e:
                mismatch = (k, g, e)
                break
        ok = mismatch is None
        all_ok &= ok
        covered = sum(bin(q[3]).count("1") for q in expected)
        print(f"  Quads shaded      : RTL {len(got)}, golden {len(expected)} ({covered} pixels)")
        if stats:
            print(f"  Quads tested      : RTL {stats[0]}, golden {gstats['quads']}")
            print(f"  Early-depth drops : RTL {stats[1]}, golden {gstats['quads_early_dropped']}")
            if stats[2]:
                print(f"  Cycles            : {stats[2]} (with random memory and output waits)")
        if ok:
            print("  PASS: every quad matches (position, mask, closeness, attributes)\n")
        else:
            k, g, e = mismatch
            print(f"  FAIL at quad {k}:")
            print(f"     RTL      {g}")
            print(f"     expected {e}\n")
    print("ALL PASS" if all_ok else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
