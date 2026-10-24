# simulator.py
# Phase 1, task 2: the INSTRUCTION SIMULATOR.
# A Python model of ONE wave (4 lanes) running on a compute unit, following spec sections 5 and 6.
# Your RTL compute unit must later match this, instruction by instruction.
#
# Needs assembler.py in the same folder.
# Use:  python simulator.py            -> runs all tests and demos
#       python simulator.py prog.asm   -> runs your program with a trace

import math
import sys
from assembler import assemble, disassemble, sign_extend, EXPORT_TARGETS

MASK32 = 0xFFFFFFFF
LANES = 4
PROGRAM_BASE = 0x010000              # shader programs live here (spec section 9)
TARGET_NAMES = {v: k for k, v in EXPORT_TARGETS.items()}


# =====================================================================
# 1. 32-BIT HELPERS
# Registers hold 32-bit patterns. s32() reads a pattern as a signed number.
# =====================================================================
def u32(x):
    return x & MASK32


def s32(x):
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


# ---- ALU operations (shared by scalar and vector instructions) ----
def op_add(a, b):   return a + b
def op_sub(a, b):   return a - b
def op_and(a, b):   return a & b
def op_or(a, b):    return a | b
def op_xor(a, b):   return a ^ b
def op_shl(a, b):   return a << (b & 31)
def op_shr(a, b):   return u32(a) >> (b & 31)              # logical: fills with 0
def op_sra(a, b):   return s32(a) >> (b & 31)              # arithmetic: keeps the sign
def op_mul(a, b):   return a * b                           # low 32 bits kept by u32()
def op_mulfx(a, b): return (s32(a) * s32(b)) >> 16         # 16.16 fixed-point multiply
def op_slt(a, b):   return 1 if s32(a) < s32(b) else 0
def op_min(a, b):   return a if s32(a) < s32(b) else b
def op_max(a, b):   return a if s32(a) > s32(b) else b

SCALAR_ALU = {0x08: op_add, 0x09: op_sub, 0x0A: op_and, 0x0B: op_or, 0x0C: op_xor,
              0x0D: op_shl, 0x0E: op_shr, 0x0F: op_sra, 0x10: op_mul, 0x11: op_mulfx,
              0x12: op_slt}
VECTOR_ALU = {0x20: op_add, 0x21: op_sub, 0x22: op_mul, 0x23: op_mulfx, 0x25: op_min,
              0x26: op_max, 0x27: op_and, 0x28: op_or, 0x29: op_xor, 0x2A: op_shl,
              0x2B: op_shr, 0x2C: op_sra, 0x2D: op_slt}


# ---- Special function unit (16.16 in, 16.16 out) ----
def sfu_rcp(a):
    """1 / a. Divide by zero gives the largest positive number."""
    a = s32(a)
    if a == 0:
        return 0x7FFFFFFF
    r = min((1 << 32) // abs(a), 0x7FFFFFFF)
    return u32(-r if a < 0 else r)


def sfu_rsq(a):
    """1 / square root of a. Zero or negative gives the largest positive number."""
    a = s32(a)
    if a <= 0:
        return 0x7FFFFFFF
    return min(math.isqrt((1 << 48) // a), 0x7FFFFFFF)


# ---- RGBA8 colors: R in bits 7-0, G 15-8, B 23-16, A 31-24 ----
def unpack(c):
    return [(c >> (8 * i)) & 255 for i in range(4)]


def pack(ch):
    return ch[0] | ch[1] << 8 | ch[2] << 16 | ch[3] << 24


def clamp255(x):
    return max(0, min(255, s32(x)))


# =====================================================================
# 2. MEMORY  (byte addresses, 32-bit words, must be 4-byte aligned)
# =====================================================================
class SimError(Exception):
    pass


class Memory:
    def __init__(self):
        self.words = {}                          # address -> 32-bit value (missing = 0)

    def _check(self, addr):
        if addr % 4:
            raise SimError(f"unaligned memory access at 0x{addr:X}")

    def load(self, addr):
        self._check(addr)
        return self.words.get(addr, 0)

    def store(self, addr, value):
        self._check(addr)
        self.words[addr] = u32(value)

    def write_block(self, addr, values):
        for i, v in enumerate(values):
            self.store(addr + 4 * i, v)

    def read_block(self, addr, count):
        return [self.load(addr + 4 * i) for i in range(count)]


# =====================================================================
# 3. ONE WAVE = 4 lanes running one program
# =====================================================================
class Wave:
    def __init__(self, memory, pc=PROGRAM_BASE, cu_id=0, wave_type=0, frame=0,
                 texture_sampler=None):
        self.mem = memory
        self.pc = pc
        self.s = [0] * 16                                  # scalar registers
        self.v = [[0] * LANES for _ in range(16)]          # vector registers
        self.mask = [True] * LANES                         # execution mask
        self.special = {0: cu_id, 1: wave_type, 2: frame, 3: LANES}
        self.texture_sampler = texture_sampler             # hook for V_TEX
        self.exports = {}                                  # target name -> 4 values
        self.halted = False
        self.count = 0                                     # instructions executed

    # ---- register writes (s0 stays 0; inactive lanes are not written) ----
    def set_s(self, i, value):
        if i != 0:
            self.s[i] = u32(value)

    def set_v(self, i, values):
        for lane in range(LANES):
            if self.mask[lane]:
                self.v[i][lane] = u32(values[lane])

    def operand_b(self, rb, bs):
        """Second vector source: a vector register, or a scalar copied to all lanes."""
        return [self.s[rb]] * LANES if bs else self.v[rb]

    # ---- execute ONE instruction ----
    def step(self):
        word = self.mem.load(self.pc)
        op = word >> 26
        rd, ra, rb = (word >> 22) & 15, (word >> 18) & 15, (word >> 14) & 15
        bs, rc = (word >> 13) & 1, (word >> 9) & 15
        imm = sign_extend(word & 0x3FFFF, 18)
        uimm = word & 0x3FFFF
        next_pc = self.pc + 4
        S, V = self.s, self.v

        if op == 0x00:                                           # NOP
            pass
        elif op == 0x01:                                         # HALT
            self.halted = True
        elif op == 0x02:                                         # S_JMP
            next_pc += 4 * sign_extend(word & 0x3FFFFFF, 26)
        elif op in (0x03, 0x04, 0x05):                           # branches
            a, b = s32(S[ra]), s32(S[rb])
            taken = {0x03: a == b, 0x04: a != b, 0x05: a < b}[op]
            if taken:
                next_pc += 4 * sign_extend(word & 0x3FFF, 14)
        elif op == 0x06:                                         # S_SPR
            self.set_s(rd, self.special.get(uimm, 0))
        elif op == 0x07:                                         # V_KILL
            for lane in range(LANES):
                if V[ra][lane] != 0:
                    self.mask[lane] = False
        elif op in SCALAR_ALU:
            self.set_s(rd, SCALAR_ALU[op](S[ra], S[rb]))
        elif op == 0x13:                                         # S_ADDI
            self.set_s(rd, S[ra] + imm)
        elif op == 0x14:                                         # S_ORI
            self.set_s(rd, S[ra] | (uimm & 0xFFFF))
        elif op == 0x15:                                         # S_LUI
            self.set_s(rd, (uimm & 0xFFFF) << 16)
        elif op == 0x18:                                         # S_LOAD
            self.set_s(rd, self.mem.load(u32(S[ra] + imm)))
        elif op == 0x19:                                         # S_STORE
            self.mem.store(u32(S[ra] + imm), S[rd])
        elif op in VECTOR_ALU:
            b = self.operand_b(rb, bs)
            self.set_v(rd, [VECTOR_ALU[op](V[ra][l], b[l]) for l in range(LANES)])
        elif op == 0x24:                                         # V_MADFX
            b = self.operand_b(rb, bs)
            self.set_v(rd, [V[rc][l] + op_mulfx(V[ra][l], b[l]) for l in range(LANES)])
        elif op == 0x2E:                                         # V_SEL
            b = self.operand_b(rb, bs)
            self.set_v(rd, [V[ra][l] if V[rc][l] != 0 else b[l] for l in range(LANES)])
        elif op == 0x2F:                                         # V_ADDI
            self.set_v(rd, [V[ra][l] + imm for l in range(LANES)])
        elif op == 0x30:                                         # V_BCAST
            self.set_v(rd, [S[ra]] * LANES)
        elif op == 0x31:                                         # V_LANEID
            self.set_v(rd, list(range(LANES)))
        elif op == 0x34:                                         # V_RCP
            self.set_v(rd, [sfu_rcp(V[ra][l]) for l in range(LANES)])
        elif op == 0x35:                                         # V_RSQ
            self.set_v(rd, [sfu_rsq(V[ra][l]) for l in range(LANES)])
        elif op == 0x38:                                         # V_LOAD
            self.set_v(rd, [self.mem.load(u32(V[ra][l] + imm)) if self.mask[l] else 0
                            for l in range(LANES)])
        elif op == 0x39:                                         # V_STORE
            for l in range(LANES):
                if self.mask[l]:
                    self.mem.store(u32(V[ra][l] + imm), V[rd][l])
        elif op == 0x3A:                                         # V_TEX
            if self.texture_sampler is None:
                raise SimError("V_TEX used, but no texture sampler is connected")
            # All 4 lanes sample (helper lanes too, for the quad's mip level);
            # only active lanes are written.
            colors = self.texture_sampler(rc, V[ra], V[rb], self.special[1])
            self.set_v(rd, colors)
        elif op == 0x3B:                                         # V_CMUL
            # (x * (y + 1)) >> 8: multiplying by white (255) leaves a color unchanged
            self.set_v(rd, [pack([(x * (y + 1)) >> 8 for x, y in zip(unpack(V[ra][l]), unpack(V[rb][l]))])
                            for l in range(LANES)])
        elif op == 0x3C:                                         # V_CADD
            self.set_v(rd, [pack([min(255, x + y) for x, y in zip(unpack(V[ra][l]), unpack(V[rb][l]))])
                            for l in range(LANES)])
        elif op == 0x3D:                                         # V_PACK
            self.set_v(rd, [pack([clamp255(V[ra][l]), clamp255(V[rb][l]), clamp255(V[rc][l]), 255])
                            for l in range(LANES)])
        elif op == 0x3E:                                         # V_UNPK
            self.set_v(rd, [(V[ra][l] >> (8 * (uimm & 3))) & 255 for l in range(LANES)])
        elif op == 0x3F:                                         # EXPORT
            name = TARGET_NAMES.get(uimm, str(uimm))
            self.exports[name] = [V[ra][l] if self.mask[l] else None for l in range(LANES)]
        else:
            raise SimError(f"reserved opcode 0x{op:02X} at PC 0x{self.pc:06X}")

        self.pc = next_pc
        self.count += 1

    # ---- run until HALT ----
    def run(self, max_steps=100_000, trace=False):
        while not self.halted:
            if self.count >= max_steps:
                raise SimError(f"no HALT after {max_steps} instructions (endless loop?)")
            if trace:
                before_s, before_v = list(self.s), [list(r) for r in self.v]
                pc, text = self.pc, disassemble(self.mem.load(self.pc))
            self.step()
            if trace:
                changes = [f"s{i}={s32(self.s[i])}" for i in range(16) if self.s[i] != before_s[i]]
                changes += [f"v{i}={[s32(x) for x in self.v[i]]}" for i in range(16)
                            if self.v[i] != before_v[i]]
                print(f"  {pc:06X}  {text:30s} {'  '.join(changes)}")
        return self


def load_and_run(source, setup=None, trace=False, **wave_options):
    """Assemble source, put it in memory, run one wave. Returns the finished wave."""
    words, _ = assemble(source)
    mem = Memory()
    mem.write_block(PROGRAM_BASE, words)
    wave = Wave(mem, **wave_options)
    if setup:
        setup(mem, wave)
    return wave.run(trace=trace)


# =====================================================================
# 4. INSTRUCTION TESTS  (verification plan, level 1)
# =====================================================================
FX = 1 << 16          # 1.0 in 16.16

TESTS = [
    ("S_ADD 5 + 7",
     "S_ADDI s1, s0, 5\nS_ADDI s2, s0, 7\nS_ADD s3, s1, s2\nHALT",
     lambda w: w.s[3] == 12),
    ("S_SUB wraps below zero",
     "S_ADDI s1, s0, 3\nS_SUB s2, s0, s1\nHALT",
     lambda w: s32(w.s[2]) == -3),
    ("s0 always reads 0",
     "S_ADDI s0, s0, 9\nHALT",
     lambda w: w.s[0] == 0),
    ("S_LUI + S_ORI build 3.25",
     "S_LUI s1, 3\nS_ORI s1, s1, 0x4000\nHALT",
     lambda w: w.s[1] == int(3.25 * FX)),
    ("S_MULFX 1.5 x 2.0 = 3.0",
     "S_LUI s1, 1\nS_ORI s1, s1, 0x8000\nS_LUI s2, 2\nS_MULFX s3, s1, s2\nHALT",
     lambda w: w.s[3] == 3 * FX),
    ("S_SRA keeps the sign",
     "S_ADDI s1, s0, -16\nS_ADDI s2, s0, 2\nS_SRA s3, s1, s2\nHALT",
     lambda w: s32(w.s[3]) == -4),
    ("S_SHR fills with zeros",
     "S_ADDI s1, s0, -16\nS_ADDI s2, s0, 2\nS_SHR s3, s1, s2\nHALT",
     lambda w: w.s[3] == u32(-16) >> 2),
    ("S_SLT is signed",
     "S_ADDI s1, s0, -1\nS_SLT s2, s1, s0\nHALT",
     lambda w: w.s[2] == 1),
    ("S_BLT loop counts to 10",
     "S_ADDI s2, s0, 10\nloop: S_ADDI s1, s1, 1\nS_BLT s1, s2, loop\nHALT",
     lambda w: w.s[1] == 10),
    ("S_JMP skips an instruction",
     "S_JMP skip\nS_ADDI s1, s0, 99\nskip: HALT",
     lambda w: w.s[1] == 0),
    ("S_STORE then S_LOAD",
     "S_ADDI s1, s0, 1234\nS_STORE s1, [s0 + 0x100]\nS_LOAD s2, [s0 + 0x100]\nHALT",
     lambda w: w.s[2] == 1234),
    ("S_SPR reads lane count",
     "S_SPR s1, 3\nHALT",
     lambda w: w.s[1] == 4),
    ("V_LANEID",
     "V_LANEID v1\nHALT",
     lambda w: w.v[1] == [0, 1, 2, 3]),
    ("V_ADD with scalar broadcast",
     "V_LANEID v1\nS_ADDI s1, s0, 10\nV_ADD v2, v1, s1\nHALT",
     lambda w: w.v[2] == [10, 11, 12, 13]),
    ("V_MADFX: 1.0 + 2.0 x lane",
     "V_LANEID v1\nS_ADDI s1, s0, 16\nV_SHL v1, v1, s1\nS_LUI s2, 2\nS_LUI s3, 1\n"
     "V_BCAST v3, s3\nV_MADFX v4, v1, s2, v3\nHALT",
     lambda w: w.v[4] == [1 * FX, 3 * FX, 5 * FX, 7 * FX]),
    ("V_MAX is signed",
     "V_LANEID v1\nV_ADDI v2, v1, -2\nV_MAX v3, v2, s0\nHALT",
     lambda w: w.v[3] == [0, 0, 0, 1]),
    ("V_SLT + V_SEL choose without branching",
     "V_LANEID v1\nS_ADDI s1, s0, 2\nV_SLT v2, v1, s1\nS_ADDI s2, s0, 100\nV_BCAST v3, s2\n"
     "V_SEL v4, v3, v1, v2\nHALT",
     lambda w: w.v[4] == [100, 100, 2, 3]),
    ("V_RCP 1 / 4.0 = 0.25",
     "S_LUI s1, 4\nV_BCAST v1, s1\nV_RCP v2, v1\nHALT",
     lambda w: w.v[2] == [FX // 4] * 4),
    ("V_RSQ 1 / sqrt(4.0) = 0.5",
     "S_LUI s1, 4\nV_BCAST v1, s1\nV_RSQ v2, v1\nHALT",
     lambda w: w.v[2] == [FX // 2] * 4),
    ("V_PACK clamps to 0..255",
     "V_LANEID v1\nS_ADDI s1, s0, 300\nV_BCAST v2, s1\nS_ADDI s2, s0, -5\nV_BCAST v3, s2\n"
     "V_PACK v4, v2, v1, v3\nV_UNPK v5, v4, 1\nHALT",
     lambda w: w.v[4] == [pack([255, l, 0, 255]) for l in range(4)] and w.v[5] == [0, 1, 2, 3]),
    ("V_CMUL: gray x white stays gray",
     "S_LUI s1, 0xFF80\nS_ORI s1, s1, 0x8080\nV_BCAST v1, s1\nS_LUI s2, 0xFFFF\n"
     "S_ORI s2, s2, 0xFFFF\nV_BCAST v2, s2\nV_CMUL v3, v1, v2\nHALT",
     lambda w: w.v[3] == [pack([128, 128, 128, 255])] * 4),
    ("V_CADD clamps at 255",
     "S_LUI s1, 0xFF80\nS_ORI s1, s1, 0x8080\nV_BCAST v1, s1\nV_CADD v2, v1, v1\nHALT",
     lambda w: w.v[2] == [pack([255, 255, 255, 255])] * 4),
    ("V_KILL: killed lanes are not written",
     "V_LANEID v1\nS_ADDI s1, s0, 2\nV_SLT v2, v1, s1\nV_KILL v2\nS_ADDI s2, s0, 7\n"
     "V_BCAST v3, s2\nHALT",
     lambda w: w.v[3] == [0, 0, 7, 7] and w.mask == [False, False, True, True]),
    ("V_STORE then V_LOAD, own address per lane",
     "V_LANEID v1\nS_ADDI s1, s0, 2\nV_SHL v1, v1, s1\nV_ADDI v2, v1, 50\n"
     "V_STORE v2, [v1 + 0x200]\nV_LOAD v3, [v1 + 0x200]\nHALT",
     lambda w: w.v[3] == [50, 54, 58, 62]),
    ("EXPORT COLOR",
     "V_LANEID v1\nEXPORT COLOR, v1\nHALT",
     lambda w: w.exports["COLOR"] == [0, 1, 2, 3]),
]


def run_tests():
    passed = 0
    for name, source, check in TESTS:
        try:
            ok = check(load_and_run(source))
        except Exception as e:
            ok, name = False, f"{name}  ({e})"
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {passed} / {len(TESTS)} tests passed\n")
    return passed == len(TESTS)


# =====================================================================
# 5. DEMOS
# =====================================================================
def demo_vector_add():
    with open("vector_add.asm") as f:
        source = f.read()
    A = [i + 1 for i in range(16)]                       # 1, 2, 3 ... 16
    B = [100 * (i + 1) for i in range(16)]               # 100, 200 ... 1600

    def setup(mem, wave):
        mem.write_block(0x1000, A)
        mem.write_block(0x2000, B)

    wave = load_and_run(source, setup)
    C = wave.mem.read_block(0x3000, 16)
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  C = {C}")
    ok = C == [a + b for a, b in zip(A, B)]
    print(f"  {'PASS' if ok else 'FAIL'}: C = A + B   ({wave.count} instructions executed)\n")


def demo_pixel_shader():
    with open("pixel_shader.asm") as f:
        source = f.read()

    def fake_texture(slot, u, v, wave_type):
        """Stand-in texture unit: purple/white checkerboard, 1 square per 1.0 of u and v."""
        return [0xFFDDD7FE if ((s32(u[l]) >> 16) + (s32(v[l]) >> 16)) % 2 == 0 else 0xFFDD777F
                for l in range(LANES)]

    def setup(mem, wave):
        wave.v[3] = [0, FX, 0, FX]                       # u per pixel of the quad
        wave.v[4] = [0, 0, FX, FX]                       # v per pixel
        wave.v[5] = [FX, FX // 2, FX // 4, 0]            # brightness 1.0, 0.5, 0.25, 0

    wave = load_and_run(source, setup, wave_type=1, texture_sampler=fake_texture)
    names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    for lane, color in enumerate(wave.exports["COLOR"]):
        r, g, b, a = unpack(color)
        print(f"  lane {lane} ({names[lane]:12s}) brightness {s32(wave.v[5][lane]) / FX:4.2f}"
              f"  -> COLOR 0x{color:08X}  (R {r:3d}, G {g:3d}, B {b:3d})")
    print()


def demo_trace():
    source = "S_ADDI s1, s0, 5\nS_ADDI s2, s0, 7\nS_ADD s3, s1, s2\nV_LANEID v1\nV_ADD v2, v1, s3\nHALT"
    load_and_run(source, trace=True)
    print()


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if a.endswith(".asm")]
    if files:
        for path in files:
            with open(path) as f:
                print(f"===== Trace of {path} =====")
                wave = load_and_run(f.read(), trace=True)
                print(f"  HALT after {wave.count} instructions")
    else:
        print("===== 1. Instruction tests =====")
        run_tests()
        print("===== 2. Trace: watch registers change =====")
        demo_trace()
        print("===== 3. vector_add.asm =====")
        demo_vector_add()
        print("===== 4. pixel_shader.asm (with a stand-in texture) =====")
        demo_pixel_shader()
