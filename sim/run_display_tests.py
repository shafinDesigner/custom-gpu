# run_display_tests.py                                              Phase 9, stage 5
# Tests display_engine.v in 3 ways:
#   1. Small screen (32 x 18), 5 frames, frame flips at awkward moments, random memory
#      stalls: EVERY video signal must match the golden model on EVERY clock cycle.
#   2. Stress: memory denied 85% of the time -> the underflow alarm MUST go off.
#   3. Real 720p60 timing: one whole frame measured (1650 x 750 = 1,237,500 cycles).
#
# Needs: simulator.py, display_engine.v, tb_display.v
# Use:   python run_display_tests.py

import shutil
import subprocess
import sys
from simulator import Memory

SMALL = dict(H_ACTIVE=32, H_FP=8, H_SYNC=8, H_BP=32, V_ACTIVE=18, V_FP=2, V_SYNC=2, V_BP=2)
P720 = dict(H_ACTIVE=1280, H_FP=110, H_SYNC=40, H_BP=220, V_ACTIVE=720, V_FP=5, V_SYNC=5, V_BP=20)
BUF_A, BUF_B, BUF_C = 0x140000, 0x160000, 0x180000


def totals(p):
    return (p["H_ACTIVE"] + p["H_FP"] + p["H_SYNC"] + p["H_BP"],
            p["V_ACTIVE"] + p["V_FP"] + p["V_SYNC"] + p["V_BP"])


# ---------------------------------------------------------------------
# Golden model: the display engine, one clock cycle at a time
# ---------------------------------------------------------------------
def golden(p, n_cycles, front, swaps, mem):
    HT, VT = totals(p)
    HA, VA = p["H_ACTIVE"], p["V_ACTIVE"]
    h, v, pending, flips, out = 0, VA, None, 0, []
    swap_at = dict(swaps)
    for n in range(n_cycles):
        visible = h < HA and v < VA
        hs = HA + p["H_FP"] <= h < HA + p["H_FP"] + p["H_SYNC"]
        vs = VA + p["V_FP"] <= v < VA + p["V_FP"] + p["V_SYNC"]
        rgb = mem.load(front + 4 * (v * HA + h)) & 0xFFFFFF if visible else 0
        out.append((int(visible), int(hs), int(vs), rgb & 255, (rgb >> 8) & 255, rgb >> 16))
        # ---- clock edge ----
        if h == 0 and v == VA and pending is not None:      # flip at the start of vertical blank
            front, pending = pending, None
            flips += 1
        if n in swap_at:                                    # newest finished frame wins
            pending = swap_at[n]
        h += 1
        if h == HT:
            h, v = 0, (v + 1) % VT
    return out, flips


def frame_buffers(p):
    """Three easy-to-recognize pictures."""
    mem = Memory()
    W, H = p["H_ACTIVE"], p["V_ACTIVE"]
    for y in range(H):
        for x in range(W):
            i = 4 * (y * W + x)
            mem.store(BUF_A + i, 0xFF000000 | (8 * x) | (14 * y) << 8 | 128 << 16)        # gradient
            mem.store(BUF_B + i, 0xFFEEEDFE if (x // 4 + y // 4) % 2 else 0xFF7F77DD)    # checker
            mem.store(BUF_C + i, 0xFF4BE5 << 0 if x == y else 0xFF201E1E)                # diagonal
    return mem


def write_image(mem, n_cycles, front, swaps, deny, path):
    mem.write_block(0x0F000, [n_cycles, len(swaps), front, deny])
    for k, (c, a) in enumerate(swaps):
        mem.write_block(0x0F100 + 8 * k, [c, a])
    with open(path, "w") as f:
        for addr in sorted(mem.words):
            if mem.words[addr]:
                f.write(f"@{addr >> 2:x}\n{mem.words[addr]:08x}\n")


def build(params, name):
    cmd = ["iverilog", "-g2012", "-o", name] + [f"-Ptb_display.{k}={v}" for k, v in params.items()] + \
          ["display_engine.v", "tb_display.v"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("===== Verilog compile FAILED =====\n" + r.stdout + r.stderr)
        sys.exit(1)


def run(sim, path, summary=False):
    args = ["vvp", sim, f"+image={path}"] + (["+summary"] if summary else [])
    out = subprocess.run(args, capture_output=True, text=True).stdout
    stream, end, summ = [], None, None
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "S":
            try:
                c = int(p[4], 16)
                stream.append((int(p[1]), int(p[2]), int(p[3]), c >> 16, (c >> 8) & 255, c & 255))
            except ValueError:
                stream.append(("undefined",))
        elif p and p[0] == "END":
            end = [int(x) for x in p[1:4]]
        elif p and p[0] == "SUMMARY":
            summ = [int(x) for x in p[1:6]]
    return stream, end, summ


def main():
    fake = "--fake-rtl" in sys.argv
    if not fake and not shutil.which("iverilog"):
        print("Icarus Verilog not found. In Colab run:  !apt-get install -y iverilog")
        return
    HT, VT = totals(SMALL)
    frame = HT * VT
    n_cycles = 5 * frame
    # The engine starts at (h=0, v=V_ACTIVE), so flip moments are at cycles 0, frame, 2*frame ...
    # swaps: mid-frame, two in one frame (newest wins), one cycle BEFORE a flip moment
    # (shown at once), and exactly ON a flip moment (shown one frame later)
    swaps = [(700, BUF_B), (frame + 300, BUF_C), (frame + 310, BUF_A),
             (2 * frame - 1, BUF_C), (3 * frame, BUF_B)]
    ok_all = True

    # ---------------- test 1: cycle-exact ----------------
    mem = frame_buffers(SMALL)
    exp, exp_flips = golden(SMALL, n_cycles, BUF_A, swaps, mem)
    if fake:
        got, end = exp, [0, exp_flips, 0]
    else:
        build(SMALL, "disp_small")
        write_image(frame_buffers(SMALL), n_cycles, BUF_A, swaps, 25, "disp.hex")
        got, end, _ = run("disp_small", "disp.hex")
    first_bad = next((i for i in range(len(exp)) if i >= len(got) or got[i] != exp[i]), None)
    ok = first_bad is None and end and end[0] == 0 and end[1] == exp_flips
    ok_all &= bool(ok)
    print(f"===== Test 1: small screen {SMALL['H_ACTIVE']}x{SMALL['V_ACTIVE']} "
          f"({HT}x{VT} with blanking), 5 frames, 25% memory stalls =====")
    print(f"  Cycles compared   : {len(exp)}  (every DE, HSYNC, VSYNC, R, G, B)")
    print(f"  Frame flips       : RTL {end[1] if end else '?'}, golden {exp_flips}")
    print(f"  Underflow alarm   : {end[0] if end else '?'} (must be 0)")
    if first_bad is not None:
        g = got[first_bad] if first_bad < len(got) else None
        print(f"  FIRST MISMATCH at cycle {first_bad}: RTL {g}, golden {exp[first_bad]}")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # ---------------- test 2: stress ----------------
    if not fake:
        write_image(frame_buffers(SMALL), 2 * frame, BUF_A, [], 85, "disp.hex")
        _, end2, _ = run("disp_small", "disp.hex")
        ok2 = bool(end2) and end2[0] == 1
        ok_all &= ok2
        print("===== Test 2: memory denied 85% of the time =====")
        print(f"  Underflow alarm   : {end2[0] if end2 else '?'} (must be 1: lines cannot be read in time)")
        print(f"  {'PASS' if ok2 else 'FAIL'}\n")

    # ---------------- test 3: real 720p60 timing ----------------
    HT7, VT7 = totals(P720)
    expected = [HT7 * VT7, P720["H_ACTIVE"] * P720["V_ACTIVE"], P720["H_SYNC"] * VT7,
                P720["V_SYNC"] * HT7, VT7]
    if fake:
        summ, end3 = expected, [0, 0, 0]
    else:
        print("===== Test 3: 720p60 timing (simulating ~1.3 million cycles, please wait) =====")
        build(P720, "disp_720")
        write_image(Memory(), 3 * HT7 * VT7, BUF_A, [], 0, "disp720.hex")
        _, end3, summ = run("disp_720", "disp720.hex", summary=True)
    names = ["Cycles per frame", "Visible pixels", "HSYNC high cycles", "VSYNC high cycles", "Lines (HSYNC pulses)"]
    ok3 = summ == expected and end3 and end3[0] == 0
    ok_all &= bool(ok3)
    if fake:
        print("===== Test 3: 720p60 timing =====")
    for name, got_v, exp_v in zip(names, summ or [None] * 5, expected):
        print(f"  {name:21s}: RTL {got_v}, expected {exp_v}")
    print(f"  At 74.25 MHz: {74_250_000 / expected[0]:.2f} frames per second")
    print(f"  {'PASS' if ok3 else 'FAIL'}\n")
    print("ALL PASS" if ok_all else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
