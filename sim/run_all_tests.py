# run_all_tests.py
# Runs EVERY test of the GPU project in order and prints one summary table.
# A full log is saved to all_tests_log.txt.
#
# Use:   python run_all_tests.py            -> everything (about 10-20 minutes)
#        python run_all_tests.py --quick    -> skips the two slowest tests
#        python run_all_tests.py --no-rtl   -> only the Python tests (no Verilog)

import os
import re
import shutil
import subprocess
import sys
import time

QUICK = "--quick" in sys.argv
NO_RTL = "--no-rtl" in sys.argv


def ratio_ok(pattern):
    """Pass if a line like '30 / 30 programs match' has equal numbers."""
    def check(out):
        m = re.findall(pattern, out)
        return bool(m) and all(a == b for a, b in m)
    return check


def has(*texts):
    return lambda out: all(t in out for t in texts)


# name, command, files needed, how to recognize a pass, uses Verilog, slow
TESTS = [
    ("Assembler: demo programs + round trip",
     ["assembler.py"], ["assembler.py"],
     lambda o: "MISMATCH" not in o and o.count("Round-trip check (disassemble -> assemble): OK") == 2,
     False, False),
    ("Instruction simulator: 25 tests + vector add",
     ["simulator.py"], ["simulator.py", "vector_add.asm", "pixel_shader.asm"],
     has("25 / 25 tests passed", "PASS: C = A + B"), False, False),
    ("SFU algorithms: 120,014 values",
     ["sfu_check.py"], ["sfu_check.py"], has("Algorithms are bit-exact"), False, False),
    ("Compute unit RTL: 30 programs",
     ["run_rtl_tests.py"], ["cu.v", "tb_cu.v"],
     ratio_ok(r"(\d+) / (\d+) programs match"), True, False),
    ("Texture unit RTL: 416 requests",
     ["run_tex_tests.py"], ["tex_unit.v", "tb_tex.v", "gpu_model.py"],
     ratio_ok(r"(\d+) / (\d+) requests match"), True, False),
    ("2 CUs: first image (2 triangles)",
     ["run_phase3.py"], ["cu.v", "tex_unit.v", "gpu_core.v", "tb_gpu_core.v"],
     has("-> PASS"), True, False),
    ("2 CUs + 2 texture units: textured quad",
     ["run_textured.py"], ["run_phase3.py", "gpu_core.v", "tb_gpu_core.v"],
     has("-> PASS"), True, False),
    ("Rasterizer RTL: 150 triangles x 2",
     ["run_raster_tests.py"], ["rasterizer.v", "tb_raster.v"], has("ALL PASS"), True, True),
    ("Render output unit RTL: 2,500 operations",
     ["run_rop_tests.py"], ["rop.v", "tb_rop.v"], has("ALL PASS"), True, False),
    ("Primitive unit RTL: 720 triangles",
     ["run_prim_tests.py"], ["prim_unit.v", "tb_prim.v"], has("ALL PASS"), True, False),
    ("Display engine RTL: cycle-exact + 720p",
     ["run_display_tests.py"], ["display_engine.v", "tb_display.v"], has("ALL PASS"), True, False),
    ("COMPLETE GPU: 2 scenes, frames + video out",
     ["run_top.py"], ["gpu_top.v", "tb_gpu_top.v", "rasterizer.v", "rop.v", "prim_unit.v",
                      "display_engine.v"],
     has("ALL PASS: the complete GPU"), True, True),
]

COMMON = ["assembler.py", "simulator.py"]


def ensure_iverilog():
    if shutil.which("iverilog"):
        return True
    print("Icarus Verilog not found - trying to install it (works in Colab) ...")
    subprocess.run(["apt-get", "install", "-y", "-qq", "iverilog"], capture_output=True)
    if shutil.which("iverilog"):
        print("Installed.\n")
        return True
    print("Could not install it. In Colab run:  !apt-get install -y iverilog\n")
    return False


def main():
    have_iverilog = False if NO_RTL else ensure_iverilog()
    results = []
    log = open("all_tests_log.txt", "w")
    print(f"{'#':>2}  {'Test':45s} {'Result':8s} {'Time':>7s}")
    print("-" * 68)
    for n, (name, cmd, needs, ok_check, rtl, slow) in enumerate(TESTS, 1):
        missing = [f for f in COMMON + needs if not os.path.exists(f)]
        if rtl and NO_RTL:
            result, secs = "skipped", 0.0
        elif slow and QUICK:
            result, secs = "skipped", 0.0
        elif missing:
            result, secs = "MISSING", 0.0
            log.write(f"\n===== {name}: missing files {missing}\n")
        elif rtl and not have_iverilog:
            result, secs = "no iverilog", 0.0
        else:
            t0 = time.time()
            p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
            secs = time.time() - t0
            out = p.stdout + p.stderr
            log.write(f"\n===== {name} ({secs:.0f} s) =====\n{out}\n")
            log.flush()
            result = "PASS" if p.returncode == 0 and ok_check(out) else "FAIL"
        results.append((name, result, missing))
        print(f"{n:>2}  {name:45s} {result:8s} {secs:6.0f}s")
    log.close()

    passed = sum(1 for _, r, _ in results if r == "PASS")
    failed = [(n, m) for n, r, m in results if r in ("FAIL", "MISSING")]
    print("-" * 68)
    print(f"{passed} passed, {len(failed)} failed or missing, "
          f"{len(results) - passed - len(failed)} skipped")
    for name, missing in failed:
        print(f"  - {name}" + (f"  (missing: {', '.join(missing)})" if missing else ""))
    print("Full output of every test: all_tests_log.txt")
    if not failed and passed == len(results):
        print("\nEVERYTHING PASSES: tools, golden model and every RTL block of the GPU.")


if __name__ == "__main__":
    main()
