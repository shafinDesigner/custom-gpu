# run_prim_tests.py                                                 Phase 5, stage 3
# Tests prim_unit.v against the golden primitive unit (primitive_unit() in gpu_model.py).
# Groups ("draws") of random triangles, each group with random projection, near plane,
# cull mode and attribute count. Every triangle handed to the rasterizer must match:
# primitive number, screen x/y (12.4), 1/z, and every attribute x 1/z.
#
# Needs: assembler.py, simulator.py, gpu_model.py, prim_unit.v, tb_prim.v
# Use:   python run_prim_tests.py

import random
import shutil
import subprocess
from simulator import Memory, u32
from gpu_model import GPU

FX = 1 << 16
N_GROUPS = 12
TRIS_PER_GROUP = 60


def fx(v):
    return u32(int(round(v * FX)))


def random_groups(rng):
    groups, tris = [], []
    for g in range(N_GROUPS):
        state = {"PROJ_CX": fx(rng.uniform(0, 64)), "PROJ_CY": fx(rng.uniform(0, 36)),
                 "PROJ_FOCAL": fx(rng.choice([rng.uniform(5, 60), rng.uniform(60, 500)])),
                 "NEAR_Z": fx(rng.choice([1.0, 1.0, rng.uniform(0.3, 3.0), -0.5, 0.0])),
                 "CULL_MODE": rng.choice([0, 1, 1, 2]), "VS_OUT_ATTRS": rng.randint(0, 8)}
        first = len(tris)
        for _ in range(TRIS_PER_GROUP):
            kind = rng.random()
            if kind < 0.6:                                # normal scene triangle
                cx, cy, cz = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 12)
                verts = [(cx + rng.uniform(-1.5, 1.5), cy + rng.uniform(-1.5, 1.5),
                          cz + rng.uniform(-0.8, 0.8)) for _ in range(3)]
                pos = [[fx(x), fx(y), fx(z)] for x, y, z in verts]
            elif kind < 0.7:                              # zero area (two equal corners)
                p = [fx(rng.uniform(-2, 2)), fx(rng.uniform(-2, 2)), fx(rng.uniform(1.5, 8))]
                q = [fx(rng.uniform(-2, 2)), fx(rng.uniform(-2, 2)), fx(rng.uniform(1.5, 8))]
                pos = [p, list(p), q]
            elif kind < 0.85:                             # huge coordinates
                pos = [[fx(rng.uniform(-30000, 30000)), fx(rng.uniform(-30000, 30000)),
                        fx(rng.uniform(1.0, 30000))] for _ in range(3)]
            else:                                         # anything at all
                pos = [[rng.getrandbits(32) for _ in range(3)] for _ in range(3)]
            attrs = [[rng.getrandbits(32) for _ in range(8)] for _ in range(3)]
            tris.append((pos, attrs))
        groups.append((state, first, TRIS_PER_GROUP))
    return groups, tris


def golden(groups, tris):
    out, totals = [], [0, 0, 0, 0, 0]
    for state, first, n in groups:
        gpu = GPU(Memory())
        gpu.state.update(state)
        shaded = []
        for pos, attrs in tris[first:first + n]:
            for v in range(3):
                shaded.append({"pos": pos[v], "attrs": attrs[v][:state["VS_OUT_ATTRS"]]})
        result = gpu.primitive_unit(shaded)
        group_out = []
        for prim, tri in enumerate(result):
            group_out.append((prim, [u32(p["x"]) for p in tri], [u32(p["y"]) for p in tri],
                              [p["inv_z"] for p in tri], [[u32(a) for a in p["attrs_z"]] for p in tri]))
        out.append(group_out)
        s = gpu.stats
        for i, key in enumerate(["triangles_in", "culled_near", "culled_zero", "culled_back",
                                 "triangles_out"]):
            totals[i] += s[key]
    return out, totals


def write_image(groups, tris, path):
    mem = Memory()
    mem.store(0x0F000, len(groups))
    for g, (st, first, n) in enumerate(groups):
        mem.write_block(0x0F100 + 64 * g, [n, first, st["PROJ_CX"], st["PROJ_CY"], st["PROJ_FOCAL"],
                                           st["NEAR_Z"], st["CULL_MODE"], st["VS_OUT_ATTRS"]])
    for t, (pos, attrs) in enumerate(tris):
        words = [pos[v][0] for v in range(3)] + [pos[v][1] for v in range(3)] + [pos[v][2] for v in range(3)]
        for v in range(3):
            words += attrs[v]
        mem.write_block(0x40000 + 256 * t, words)
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def run_rtl(path, groups):
    out = subprocess.run(["vvp", "prim_sim", f"+image={path}"], capture_output=True, text=True).stdout
    result, stats, g = [], None, -1
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "GROUP":
            g = int(p[1])
            result.append([])
        elif p[0] == "TRI":
            try:
                n_attrs = groups[g][0]["VS_OUT_ATTRS"]
                w = [int(x, 16) for x in p[2:2 + 9 + 24]]
                attrs = [w[9 + 8 * v: 9 + 8 * v + n_attrs] for v in range(3)]
                result[g].append((int(p[1]), w[0:3], w[3:6], w[6:9], attrs))
            except (ValueError, IndexError):
                result[g].append(("undefined", line[:80]))
        elif p[0] == "STATS":
            stats = [int(x) for x in p[1:6]]
    return result, stats


def main():
    import sys
    fake = "--fake-rtl" in sys.argv
    if not fake:
        if not shutil.which("iverilog"):
            print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
            return
        build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "prim_sim", "prim_unit.v", "tb_prim.v"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("===== Verilog compile FAILED =====\n" + build.stdout + build.stderr)
            return
        if build.stderr.strip():
            print("Compiler warnings:\n" + build.stderr)

    groups, tris = random_groups(random.Random(2029))
    expected, totals = golden(groups, tris)
    if fake:
        got, stats = expected, totals
    else:
        write_image(groups, tris, "prim.hex")
        got, stats = run_rtl("prim.hex", groups)

    print(f"===== Primitive unit RTL vs golden: {len(tris)} triangles in {len(groups)} draws =====")
    names = ["in", "near-plane culled", "zero-area culled", "back/front culled", "out"]
    for i, name in enumerate(names):
        print(f"  {name:18s}: RTL {stats[i] if stats else '?'}, golden {totals[i]}")
    ok = stats == totals
    for g in range(len(groups)):
        e, r = expected[g], got[g] if g < len(got) else []
        if e != r:
            ok = False
            k = next(i for i in range(max(len(e), len(r))) if i >= len(e) or i >= len(r) or e[i] != r[i])
            print(f"  FAIL in draw {g} (cull mode {groups[g][0]['CULL_MODE']}), triangle {k}:")
            print(f"     RTL      {r[k] if k < len(r) else None}")
            print(f"     expected {e[k] if k < len(e) else None}")
            break
    print("\n  ALL PASS: every output triangle matches" if ok else "\n  FAIL")


if __name__ == "__main__":
    main()
