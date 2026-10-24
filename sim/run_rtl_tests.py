# run_rtl_tests.py                                                       Phase 2
# Runs the SAME programs on:
#   1. your Verilog compute unit (cu.v, through Icarus Verilog), and
#   2. the golden model (simulator.py)
# and compares every scalar register, vector register, the mask, every memory word,
# and the instruction count. Any difference is a hardware bug (or a model bug).
#
# Needs: assembler.py, simulator.py, cu.v, tb_cu.v, and vector_add.asm (from the assembler demo)
# Use:   python run_rtl_tests.py

import os
import shutil
import subprocess
import sys
from assembler import assemble
from simulator import Wave, Memory, PROGRAM_BASE, TESTS, u32
from assembler import EXPORT_TARGETS

# Instructions the RTL does not have yet (added in the next steps)
NOT_BUILT_YET = ("V_TEX",)

FX = 1 << 16

# Extra tests with tricky values: each lane loads its own input from memory
LOAD_LANES = """\
        V_LANEID v1
        S_ADDI   s1, s0, 2
        V_SHL    v1, v1, s1          ; lane x 4
        V_LOAD   v2, [v1 + 0x400]    ; input A, one per lane
        V_LOAD   v3, [v1 + 0x410]    ; input B, one per lane
"""
SFU_PROGRAM = LOAD_LANES + """\
        V_RCP    v4, v2
        V_RSQ    v5, v2
        HALT
"""
COLOR_PROGRAM = LOAD_LANES + """\
        V_CMUL   v4, v2, v3
        V_CADD   v5, v2, v3
        V_UNPK   v6, v2, 2
        V_PACK   v7, v2, v3, v6      ; big and negative values get clamped
        EXPORT   COLOR, v4
        EXPORT   ATTR3, v5
        V_LANEID v8
        S_ADDI   s2, s0, 1
        V_SLT    v9, v8, s2          ; lane 0 -> 1
        V_KILL   v9                  ; turn lane 0 off
        V_RCP    v10, v3             ; SFU skips the killed lane
        EXPORT   ATTR0, v7           ; only lanes 1-3 are exported
        HALT
"""
EXTRA_CASES = [
    ("SFU: 3.0, -2.0, 0, tiny", SFU_PROGRAM, [3 * FX, u32(-2 * FX), 0, 1], [0, 0, 0, 0]),
    ("SFU: largest, most negative, 7, big", SFU_PROGRAM,
     [0x7FFFFFFF, 0x80000000, 7, 123456789], [0, 0, 0, 0]),
    ("SFU: 1/3, 10.0, 0.001, 65535", SFU_PROGRAM, [FX // 3, 10 * FX, 66, 65535], [0, 0, 0, 0]),
    ("Colors, clamps, KILL + EXPORT", COLOR_PROGRAM,
     [0x12345678, 0xFFFFFFFF, 0x00000000, 0x80FF7F01],
     [0x9ABCDEF0, 0xFFFFFFFF, 0xFFFFFFFF, 0x7F01FF80]),
]


# ---------------------------------------------------------------------
# Test cases: the simulator's instruction tests + vector_add with data
# ---------------------------------------------------------------------
def all_cases():
    cases, skipped = [], []
    for name, source, _ in TESTS:
        if any(op in source for op in NOT_BUILT_YET):
            skipped.append(name)
        else:
            cases.append((name, source, None))
    for name, source, a, b in EXTRA_CASES:
        def setup(mem, a=a, b=b):
            mem.write_block(0x400, a)
            mem.write_block(0x410, b)
        cases.append((name, source, setup))
    if os.path.exists("vector_add.asm"):
        with open("vector_add.asm") as f:
            source = f.read()

        def setup(mem):
            mem.write_block(0x1000, [i + 1 for i in range(16)])
            mem.write_block(0x2000, [100 * (i + 1) for i in range(16)])
        cases.append(("vector_add.asm: C = A + B", source, setup))
    return cases, skipped


# ---------------------------------------------------------------------
# Golden model run
# ---------------------------------------------------------------------
def python_run(source, setup):
    mem = Memory()
    mem.write_block(PROGRAM_BASE, assemble(source)[0])
    if setup:
        setup(mem)
    image = dict(mem.words)                        # memory BEFORE running = what RTL gets
    wave = Wave(mem).run()
    expected = {
        "status": "DONE",
        "count": wave.count,
        "mask": sum(1 << l for l in range(4) if wave.mask[l]),
        "sreg": list(wave.s),
        "vreg": [list(r) for r in wave.v],
        "mem": {a: v for a, v in mem.words.items() if v},
        "exports": {EXPORT_TARGETS[name]: values for name, values in wave.exports.items()},
    }
    return image, expected


# ---------------------------------------------------------------------
# RTL run
# ---------------------------------------------------------------------
def write_image(image, path):
    """$readmemh format: '@' + word address, then the word."""
    with open(path, "w") as f:
        for addr in sorted(image):
            if image[addr]:
                f.write(f"@{addr >> 2:x}\n{image[addr]:08x}\n")


def parse(text):
    r = {"status": "NO_OUTPUT", "count": None, "cycles": None, "mask": None,
         "sreg": [0] * 16, "vreg": [[0] * 4 for _ in range(16)], "mem": {}, "exports": {}}
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        try:
            if p[0] == "STATUS":
                r["status"] = p[1]
            elif p[0] == "COUNT":
                r["count"] = int(p[1])
            elif p[0] == "CYCLES":
                r["cycles"] = int(p[1])
            elif p[0] == "MASK":
                r["mask"] = int(p[1], 16)
            elif p[0] == "SREG":
                r["sreg"][int(p[1])] = int(p[2], 16)
            elif p[0] == "VREG":
                r["vreg"][int(p[1])][int(p[2])] = int(p[3], 16)
            elif p[0] == "MEM":
                r["mem"][int(p[1], 16)] = int(p[2], 16)
            elif p[0] == "EXPORT":                 # target, mask, lane 0..3; the last one wins
                mask = int(p[2], 16)
                r["exports"][int(p[1])] = [int(p[3 + l], 16) if mask >> l & 1 else None
                                           for l in range(4)]
        except ValueError:                         # 'x' or 'z' in a value = undefined bits
            r.setdefault("undefined", []).append(line.strip())
    return r


def rtl_run(image):
    write_image(image, "image.hex")
    result = subprocess.run(["vvp", "cu_sim", "+image=image.hex"],
                            capture_output=True, text=True)
    return parse(result.stdout)


def fake_rtl_run(expected):
    """Only for testing THIS script without Verilog: pretend the RTL is perfect."""
    lines = [f"STATUS {expected['status']}", f"COUNT {expected['count']}", "CYCLES 0",
             f"MASK {expected['mask']:x}"]
    lines += [f"SREG {i} {v:08x}" for i, v in enumerate(expected["sreg"])]
    lines += [f"VREG {r} {l} {expected['vreg'][r][l]:08x}" for r in range(16) for l in range(4)]
    lines += [f"MEM {a:08x} {v:08x}" for a, v in sorted(expected["mem"].items())]
    for t, vals in expected["exports"].items():
        m = sum(1 << l for l in range(4) if vals[l] is not None)
        lines.append(f"EXPORT {t} {m:x} " + " ".join(f"{(v or 0):08x}" for v in vals))
    return parse("\n".join(lines))


# ---------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------
def compare(exp, got):
    diffs = []
    if got.get("undefined"):
        diffs += [f"undefined value (x/z): {u}" for u in got["undefined"][:3]]
    if got["status"] != exp["status"]:
        diffs.append(f"status: RTL {got['status']}, expected {exp['status']}")
    if got["count"] != exp["count"]:
        diffs.append(f"instruction count: RTL {got['count']}, expected {exp['count']}")
    if got["mask"] != exp["mask"]:
        diffs.append(f"mask: RTL {got['mask']}, expected {exp['mask']}")
    for i in range(16):
        if got["sreg"][i] != exp["sreg"][i]:
            diffs.append(f"s{i}: RTL 0x{got['sreg'][i]:08X}, expected 0x{exp['sreg'][i]:08X}")
    for r in range(16):
        if got["vreg"][r] != exp["vreg"][r]:
            diffs.append(f"v{r}: RTL {[hex(x) for x in got['vreg'][r]]}, "
                         f"expected {[hex(x) for x in exp['vreg'][r]]}")
    if got["exports"] != exp["exports"]:
        diffs.append(f"exports: RTL {got['exports']}, expected {exp['exports']}")
    for a in sorted(set(got["mem"]) | set(exp["mem"])):
        g, e = got["mem"].get(a, 0), exp["mem"].get(a, 0)
        if g != e:
            diffs.append(f"memory 0x{a:05X}: RTL 0x{g:08X}, expected 0x{e:08X}")
    return diffs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    fake = "--fake-rtl" in sys.argv
    if not fake:
        if not shutil.which("iverilog"):
            print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
            return
        build = subprocess.run(["iverilog", "-g2012", "-Wall", "-o", "cu_sim", "cu.v", "tb_cu.v"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("===== Verilog compile FAILED =====")
            print(build.stdout + build.stderr)
            return
        if build.stderr.strip():
            print("Compiler warnings:\n" + build.stderr)

    cases, skipped = all_cases()
    print(f"===== RTL vs golden model: {len(cases)} programs =====")
    passed = 0
    for name, source, setup in cases:
        image, expected = python_run(source, setup)
        got = fake_rtl_run(expected) if fake else rtl_run(image)
        diffs = compare(expected, got)
        cycles = f"{got['cycles']} cycles" if got.get("cycles") else ""
        if diffs:
            print(f"  FAIL  {name}")
            for d in diffs[:8]:
                print(f"          {d}")
        else:
            passed += 1
            print(f"  PASS  {name:45s} {expected['count']:3d} instructions  {cycles}")
    print(f"\n  {passed} / {len(cases)} programs match the golden model exactly")
    if skipped:
        print(f"  Skipped (not in the RTL yet): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
