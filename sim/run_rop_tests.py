# run_rop_tests.py                                                  Phase 5, stage 2
# Tests rop.v against the golden render output unit (render_output() and clear()
# in gpu_model.py). A random sequence of clears and overlapping quads, with random
# depth mode, blend mode and color mask, is replayed on both. At the end the WHOLE
# color, depth and stencil buffers must be identical.
#
# Needs: assembler.py, simulator.py, gpu_model.py, rop.v, tb_rop.v
# Use:   python run_rop_tests.py

import random
import shutil
import subprocess
import sys
from simulator import Memory
from gpu_model import GPU

W, H = 64, 36
FB, DEPTH, STENCIL = 0x040000, 0x060000, 0x070000
N_OPS = 2500


def random_ops(rng):
    ops = [("clear", 0, 0, 0xF, (7, 0xFF1E1E1E, 0, 0))]                 # start clean
    for _ in range(N_OPS - 1):
        depth_mode = rng.choice([0, 1, 1, 1, 2])
        blend_mode = rng.choice([0, 0, 1, 2, 3])
        color_mask = 0xF if rng.random() < 0.7 else rng.randrange(16)
        if rng.random() < 0.01:                                          # occasional clear
            ops.append(("clear", depth_mode, blend_mode, color_mask,
                        (rng.randrange(1, 8), rng.getrandbits(32), rng.randrange(0, 30000),
                         rng.randrange(256))))
            continue
        x, y = rng.randrange(0, W, 2), rng.randrange(0, H, 2)
        mask = rng.randrange(1, 16)
        close = [rng.choice([rng.randrange(1, 40000), rng.randrange(40000, 70000),
                             rng.randrange(70000, 1 << 32)]) for _ in range(4)]
        colors = []
        for _ in range(4):
            c = rng.getrandbits(32)
            alpha = rng.choice([0, 255, 128, 127, rng.randrange(256)])  # edge alphas often
            colors.append((c & 0x00FFFFFF) | alpha << 24)
        ops.append(("quad", depth_mode, blend_mode, color_mask, (x, y, mask, close, colors)))
    return ops


def golden(ops):
    mem = Memory()
    gpu = GPU(mem)
    gpu.state.update({"VIEWPORT_W": W, "VIEWPORT_H": H, "FB_A_ADDR": FB,
                      "DEPTH_ADDR": DEPTH, "STENCIL_ADDR": STENCIL})
    for kind, dm, bm, cm, data in ops:
        gpu.state.update({"DEPTH_MODE": dm, "BLEND_MODE": bm, "COLOR_MASK": cm})
        if kind == "clear":
            gpu.clear(*data)
        else:
            x, y, mask, close, colors = data
            lanes = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
            gpu.render_output(lanes, [bool(mask >> l & 1) for l in range(4)], close, colors)
    n = W * H
    return {"FB": mem.read_block(FB, n), "DEPTH": mem.read_block(DEPTH, (n + 1) // 2),
            "STENCIL": mem.read_block(STENCIL, (n + 3) // 4),
            "written": gpu.stats["pixels_written"], "failed": gpu.stats["pixels_depth_failed"]}


def write_image(ops, path):
    mem = Memory()
    mem.write_block(0x0F000, [len(ops), W, H, FB, DEPTH, STENCIL])
    for k, (kind, dm, bm, cm, data) in enumerate(ops):
        head = (1 if kind == "clear" else 0) | dm << 8 | bm << 12 | cm << 16
        if kind == "clear":
            words = [head] + list(data)
        else:
            x, y, mask, close, colors = data
            words = [head, x, y, mask] + close + colors
        mem.write_block(0x0F100 + 64 * k, words)
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def run_rtl(path):
    out = subprocess.run(["vvp", "rop_sim", f"+image={path}"], capture_output=True, text=True).stdout
    n = W * H
    r = {"FB": [None] * n, "DEPTH": [None] * ((n + 1) // 2), "STENCIL": [None] * ((n + 3) // 4)}
    for line in out.splitlines():
        p = line.split()
        if p and p[0] in ("FB", "DEPTH", "STENCIL"):
            try:
                r[p[0]][int(p[1])] = int(p[2], 16)
            except ValueError:
                pass                                        # x / z stays None
        elif p and p[0] == "STATS":
            r["written"], r["failed"], r["cycles"] = (int(x) for x in p[1:4])
    return r


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake:
        if not shutil.which("iverilog"):
            print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
            return
        build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "rop_sim", "rop.v", "tb_rop.v"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
            return
        if build.stderr.strip():
            print("Compiler warnings:\n" + build.stderr)

    ops = random_ops(random.Random(2028))
    exp = golden(ops)
    if fake:
        got = dict(exp, cycles=0)
    else:
        write_image(ops, "rop.hex")
        got = run_rtl("rop.hex")

    quads = sum(1 for o in ops if o[0] == "quad")
    print(f"===== Render output unit RTL vs golden: {quads} quads, {len(ops) - quads} clears =====")
    print(f"  Pixels written    : RTL {got.get('written')}, golden {exp['written']}")
    print(f"  Depth test failed : RTL {got.get('failed')}, golden {exp['failed']}")
    if got.get("cycles"):
        print(f"  Cycles            : {got['cycles']} (with random memory waits)")
    ok = True
    for name in ("FB", "DEPTH", "STENCIL"):
        bad = [i for i in range(len(exp[name])) if got[name][i] != exp[name][i]]
        ok &= not bad
        label = {"FB": "Color buffer", "DEPTH": "Depth buffer", "STENCIL": "Stencil buffer"}[name]
        print(f"  {label:17s} : {len(exp[name]) - len(bad)} / {len(exp[name])} words match"
              f"{'' if not bad else '  <-- FAIL'}")
        for i in bad[:3]:
            g = "undefined" if got[name][i] is None else f"0x{got[name][i]:08X}"
            print(f"      word {i}: RTL {g}, golden 0x{exp[name][i]:08X}")
    ok &= got.get("written") == exp["written"] and got.get("failed") == exp["failed"]
    print("\n  ALL PASS" if ok else "\n  FAIL")


if __name__ == "__main__":
    main()
