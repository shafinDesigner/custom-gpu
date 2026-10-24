# assembler.py
# Phase 1, task 1: the ASSEMBLER for your GPU's instruction set (ISA v0.1).
#   Shader text  "V_ADD v1, v2, v3"  ->  32-bit machine word  0x80484000
# Follows spec section 6 exactly. Also includes a DISASSEMBLER (machine word -> text),
# used to double-check every encoding now, and to debug the simulator later.
#
# Use:
#   python assembler.py myshader.asm     -> myshader.hex, myshader.bin, myshader.lst
#   python assembler.py                  -> runs the built-in demo programs
#   In Colab: paste this file in a cell and run it (runs the demo).

import re
import struct
import sys

# =====================================================================
# 1. THE INSTRUCTION TABLE  (spec section 6)
#    mnemonic: (opcode, format, [operand kinds in the order you write them])
# =====================================================================
INSTRUCTIONS = {
    # ---- Control and special ----
    "NOP":      (0x00, "R", []),
    "HALT":     (0x01, "R", []),
    "S_JMP":    (0x02, "J", ["offset"]),
    "S_BEQ":    (0x03, "B", ["sa", "sb", "offset"]),
    "S_BNE":    (0x04, "B", ["sa", "sb", "offset"]),
    "S_BLT":    (0x05, "B", ["sa", "sb", "offset"]),
    "S_SPR":    (0x06, "I", ["sd", "spr"]),
    "V_KILL":   (0x07, "R", ["va"]),
    # ---- Scalar ALU and memory ----
    "S_ADD":    (0x08, "R", ["sd", "sa", "sb"]),
    "S_SUB":    (0x09, "R", ["sd", "sa", "sb"]),
    "S_AND":    (0x0A, "R", ["sd", "sa", "sb"]),
    "S_OR":     (0x0B, "R", ["sd", "sa", "sb"]),
    "S_XOR":    (0x0C, "R", ["sd", "sa", "sb"]),
    "S_SHL":    (0x0D, "R", ["sd", "sa", "sb"]),
    "S_SHR":    (0x0E, "R", ["sd", "sa", "sb"]),
    "S_SRA":    (0x0F, "R", ["sd", "sa", "sb"]),
    "S_MUL":    (0x10, "R", ["sd", "sa", "sb"]),
    "S_MULFX":  (0x11, "R", ["sd", "sa", "sb"]),
    "S_SLT":    (0x12, "R", ["sd", "sa", "sb"]),
    "S_ADDI":   (0x13, "I", ["sd", "sa", "imm18"]),
    "S_ORI":    (0x14, "I", ["sd", "sa", "u16"]),
    "S_LUI":    (0x15, "I", ["sd", "u16"]),
    "S_LOAD":   (0x18, "I", ["sd", "smem"]),
    "S_STORE":  (0x19, "I", ["ss", "smem"]),
    # ---- Vector ALU (the last register may be scalar: BS = 1) ----
    "V_ADD":    (0x20, "R", ["vd", "va", "vbs"]),
    "V_SUB":    (0x21, "R", ["vd", "va", "vbs"]),
    "V_MUL":    (0x22, "R", ["vd", "va", "vbs"]),
    "V_MULFX":  (0x23, "R", ["vd", "va", "vbs"]),
    "V_MADFX":  (0x24, "R", ["vd", "va", "vbs", "vc"]),
    "V_MIN":    (0x25, "R", ["vd", "va", "vbs"]),
    "V_MAX":    (0x26, "R", ["vd", "va", "vbs"]),
    "V_AND":    (0x27, "R", ["vd", "va", "vbs"]),
    "V_OR":     (0x28, "R", ["vd", "va", "vbs"]),
    "V_XOR":    (0x29, "R", ["vd", "va", "vbs"]),
    "V_SHL":    (0x2A, "R", ["vd", "va", "vbs"]),
    "V_SHR":    (0x2B, "R", ["vd", "va", "vbs"]),
    "V_SRA":    (0x2C, "R", ["vd", "va", "vbs"]),
    "V_SLT":    (0x2D, "R", ["vd", "va", "vbs"]),
    "V_SEL":    (0x2E, "R", ["vd", "va", "vbs", "vc"]),
    "V_ADDI":   (0x2F, "I", ["vd", "va", "imm18"]),
    "V_BCAST":  (0x30, "I", ["vd", "sa"]),
    "V_LANEID": (0x31, "R", ["vd"]),
    # ---- Special function unit and vector memory ----
    "V_RCP":    (0x34, "R", ["vd", "va"]),
    "V_RSQ":    (0x35, "R", ["vd", "va"]),
    "V_LOAD":   (0x38, "I", ["vd", "vmem"]),
    "V_STORE":  (0x39, "I", ["vs", "vmem"]),
    # ---- Graphics ----
    "V_TEX":    (0x3A, "R", ["vd", "va", "vb", "slot"]),
    "V_CMUL":   (0x3B, "R", ["vd", "va", "vb"]),
    "V_CADD":   (0x3C, "R", ["vd", "va", "vb"]),
    "V_PACK":   (0x3D, "R", ["vd", "va", "vb", "vc"]),
    "V_UNPK":   (0x3E, "I", ["vd", "va", "ch"]),
    "EXPORT":   (0x3F, "I", ["target", "va"]),
}

# EXPORT targets (spec section 6, graphics table)
EXPORT_TARGETS = {"POS_X": 0, "POS_Y": 1, "POS_Z": 2, "COLOR": 16}
EXPORT_TARGETS.update({f"ATTR{i}": 8 + i for i in range(8)})

# Register operand kinds: which field they go into, and which register file is allowed
REGISTER_KINDS = {
    "sd": ("rd", "s"), "ss": ("rd", "s"), "vd": ("rd", "v"), "vs": ("rd", "v"),
    "sa": ("ra", "s"), "va": ("ra", "v"),
    "sb": ("rb", "s"), "vb": ("rb", "v"), "vbs": ("rb", "vs"),
    "vc": ("rc", "v"),
}

# How each operand kind is shown in error messages
SHOWN_AS = {"vbs": "vb|sb", "smem": "[sa + imm]", "vmem": "[va + imm]",
            "offset": "label", "imm18": "imm", "u16": "imm16"}


class AsmError(Exception):
    pass


# =====================================================================
# 2. SMALL PARSING HELPERS
# =====================================================================
def parse_number(text):
    try:
        return int(text.strip(), 0)          # accepts 42, -7, 0x1F, 0b101
    except ValueError:
        raise AsmError(f"'{text.strip()}' is not a number")


def check_range(value, low, high, what):
    if not low <= value <= high:
        raise AsmError(f"{what} {value} is out of range ({low} to {high})")
    return value


def parse_register(text, allowed):
    """'v3' -> ('v', 3). allowed = 's', 'v' or 'vs'."""
    m = re.fullmatch(r"([sSvV])(\d+)", text.strip())
    if not m:
        raise AsmError(f"expected a register, got '{text.strip()}'")
    kind, number = m.group(1).lower(), int(m.group(2))
    if number > 15:
        raise AsmError(f"register {text.strip()} does not exist (only 0 to 15)")
    if kind not in allowed:
        wanted = {"s": "a scalar (s)", "v": "a vector (v)"}[allowed]
        raise AsmError(f"expected {wanted} register, got '{text.strip()}'")
    return kind, number


def parse_memory(text, allowed):
    """'[v1 + 0x1000]' -> (register number, offset)."""
    m = re.fullmatch(r"\[\s*([sSvV]\d+)\s*(?:([+-])\s*(\w+))?\s*\]", text.strip())
    if not m:
        raise AsmError(f"expected a memory address like [{allowed}1 + 16], got '{text.strip()}'")
    _, reg = parse_register(m.group(1), allowed)
    offset = parse_number(m.group(3)) if m.group(3) else 0
    if m.group(2) == "-":
        offset = -offset
    return reg, check_range(offset, -(1 << 17), (1 << 17) - 1, "memory offset")


def usage(mnemonic):
    kinds = INSTRUCTIONS[mnemonic][2]
    return mnemonic + (" " + ", ".join(SHOWN_AS.get(k, k) for k in kinds) if kinds else "")


# =====================================================================
# 3. ENCODE ONE INSTRUCTION  (spec section 6, encoding formats)
#   R: opcode[31:26] rd[25:22] ra[21:18] rb[17:14] BS[13] rc[12:9] 0[8:0]
#   I: opcode[31:26] rd[25:22] ra[21:18] imm18[17:0]
#   B: opcode[31:26] -[25:22]  ra[21:18] rb[17:14] off14[13:0]
#   J: opcode[31:26] off26[25:0]
# =====================================================================
def encode(index, mnemonic, operands, labels):
    if mnemonic not in INSTRUCTIONS:
        raise AsmError(f"unknown instruction '{mnemonic}'")
    opcode, fmt, kinds = INSTRUCTIONS[mnemonic]
    if len(operands) != len(kinds):
        raise AsmError(f"needs {len(kinds)} operand(s): {usage(mnemonic)}")

    f = {"rd": 0, "ra": 0, "rb": 0, "rc": 0, "bs": 0, "imm": 0, "off": 0}
    for kind, text in zip(kinds, operands):
        if kind in REGISTER_KINDS:
            field, allowed = REGISTER_KINDS[kind]
            if kind == "vbs" and not re.fullmatch(r"\s*[sSvV]\d+\s*", text):
                raise AsmError(f"'{text.strip()}' must be a register (v or s); "
                               f"to use a constant, put it in an s register first")
            reg_kind, number = parse_register(text, allowed)
            f[field] = number
            if kind == "vbs" and reg_kind == "s":
                f["bs"] = 1                      # broadcast scalar to all lanes
        elif kind == "imm18":
            f["imm"] = check_range(parse_number(text), -(1 << 17), (1 << 17) - 1, "immediate")
        elif kind == "u16":
            f["imm"] = check_range(parse_number(text), 0, 0xFFFF, "16-bit value")
        elif kind in ("spr", "ch"):
            f["imm"] = check_range(parse_number(text), 0, 3, "special/channel number")
        elif kind == "slot":
            f["rc"] = check_range(parse_number(text), 0, 3, "texture slot")
        elif kind == "target":
            name = text.strip().upper()
            if name in EXPORT_TARGETS:
                f["imm"] = EXPORT_TARGETS[name]
            else:
                raise AsmError(f"unknown export target '{text.strip()}' "
                               f"(use POS_X, POS_Y, POS_Z, ATTR0-ATTR7, COLOR)")
        elif kind in ("smem", "vmem"):
            f["ra"], f["imm"] = parse_memory(text, "s" if kind == "smem" else "v")
        elif kind == "offset":
            name = text.strip()
            if name in labels:
                f["off"] = labels[name] - (index + 1)   # relative to the NEXT instruction
            elif re.fullmatch(r"[+-]?(0x)?[0-9A-Fa-f]+", name):
                f["off"] = parse_number(name)
            else:
                raise AsmError(f"'{name}' is not a label or a number")
            bits = 14 if fmt == "B" else 26
            check_range(f["off"], -(1 << (bits - 1)), (1 << (bits - 1)) - 1, "branch distance")

    word = opcode << 26
    if fmt == "R":
        word |= f["rd"] << 22 | f["ra"] << 18 | f["rb"] << 14 | f["bs"] << 13 | f["rc"] << 9
    elif fmt == "I":
        word |= f["rd"] << 22 | f["ra"] << 18 | (f["imm"] & 0x3FFFF)
    elif fmt == "B":
        word |= f["ra"] << 18 | f["rb"] << 14 | (f["off"] & 0x3FFF)
    else:  # J
        word |= f["off"] & 0x3FFFFFF
    return word


# =====================================================================
# 4. ASSEMBLE A WHOLE PROGRAM  (two passes)
#   Pass 1: find every label and remember its instruction number
#   Pass 2: encode each instruction (labels are now known)
# =====================================================================
class AssemblyFailed(Exception):
    def __init__(self, errors):
        super().__init__("\n".join(errors))
        self.errors = errors


def split_operands(text):
    return [part.strip() for part in text.split(",")] if text.strip() else []


def assemble(source):
    program, labels, errors = [], {}, []

    # ---- Pass 1 ----
    for line_no, line in enumerate(source.splitlines(), 1):
        code = line.split(";", 1)[0].strip()          # remove comments
        while True:                                   # labels like "loop:"
            m = re.match(r"^([A-Za-z_]\w*)\s*:\s*(.*)$", code)
            if not m:
                break
            name, code = m.group(1), m.group(2)
            if name in labels:
                errors.append(f"line {line_no}: label '{name}' is defined twice")
            labels[name] = len(program)
        if not code:
            continue
        parts = code.split(None, 1)
        program.append((line_no, parts[0].upper(),
                        split_operands(parts[1] if len(parts) > 1 else ""), line.rstrip()))

    # ---- Pass 2 ----
    words, listing = [], []
    for index, (line_no, mnemonic, operands, text) in enumerate(program):
        try:
            word = encode(index, mnemonic, operands, labels)
            words.append(word)
            listing.append(f"{index * 4:06X}  {word:08X}  {text.strip()}")
        except AsmError as e:
            errors.append(f"line {line_no}: {e}\n        -> {text.strip()}")

    if errors:
        raise AssemblyFailed(errors)
    return words, listing


# =====================================================================
# 5. DISASSEMBLER  (32-bit word -> text). Used to check the assembler.
# =====================================================================
OPCODES = {op: (name, fmt, kinds) for name, (op, fmt, kinds) in INSTRUCTIONS.items()}
TARGET_NAMES = {v: k for k, v in EXPORT_TARGETS.items()}


def sign_extend(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def disassemble(word):
    op = word >> 26
    if op not in OPCODES:
        return f".word 0x{word:08X}   ; reserved opcode 0x{op:02X}"
    name, fmt, kinds = OPCODES[op]
    rd, ra, rb = (word >> 22) & 15, (word >> 18) & 15, (word >> 14) & 15
    bs, rc = (word >> 13) & 1, (word >> 9) & 15
    imm = sign_extend(word & 0x3FFFF, 18)
    uimm = word & 0x3FFFF
    off = sign_extend(word & 0x3FFF, 14) if fmt == "B" else sign_extend(word & 0x3FFFFFF, 26)

    def mem(kind_letter):
        return f"[{kind_letter}{ra} {'-' if imm < 0 else '+'} {abs(imm)}]"

    shown = {
        "sd": f"s{rd}", "ss": f"s{rd}", "vd": f"v{rd}", "vs": f"v{rd}",
        "sa": f"s{ra}", "va": f"v{ra}", "sb": f"s{rb}", "vb": f"v{rb}",
        "vbs": f"{'s' if bs else 'v'}{rb}", "vc": f"v{rc}",
        "imm18": str(imm), "u16": str(uimm), "spr": str(uimm), "ch": str(uimm),
        "slot": str(rc), "target": TARGET_NAMES.get(uimm, str(uimm)),
        "smem": mem("s"), "vmem": mem("v"), "offset": str(off),
    }
    return name + (" " + ", ".join(shown[k] for k in kinds) if kinds else "")


# =====================================================================
# 6. SAVE THE RESULT
#   .hex : one word per line  (Verilog can load it with $readmemh)
#   .bin : raw little-endian words (for the simulator and DRAM images)
#   .lst : address, machine word, source line (for you to read)
# =====================================================================
def save(words, listing, base_name):
    with open(base_name + ".hex", "w") as f:
        f.write("\n".join(f"{w:08x}" for w in words) + "\n")
    with open(base_name + ".bin", "wb") as f:
        f.write(struct.pack(f"<{len(words)}I", *words))
    with open(base_name + ".lst", "w") as f:
        f.write("ADDR    WORD      SOURCE\n" + "\n".join(listing) + "\n")


def round_trip_check(words):
    """Disassemble every word, assemble that text again: must give the same word."""
    for w in words:
        again = encode(0, *_split_line(disassemble(w)), {})
        if again != w:
            return f"MISMATCH: {w:08X} -> '{disassemble(w)}' -> {again:08X}"
    return "OK"


def _split_line(text):
    parts = text.split(None, 1)
    return parts[0], split_operands(parts[1] if len(parts) > 1 else "")


# =====================================================================
# 7. DEMO PROGRAMS
# =====================================================================
VECTOR_ADD = """\
; Vector add: C[i] = A[i] + B[i] for 16 numbers, on one CU (4 lanes)
; A at 0x1000, B at 0x2000, C at 0x3000 (byte addresses, 4 bytes per number)
        S_ADDI   s3, s0, 2          ; s3 = 2 (shift amount: x4 bytes)
        V_LANEID v0                 ; v0 = [0, 1, 2, 3]
        V_SHL    v0, v0, s3         ; v0 = lane x 4 = [0, 4, 8, 12]
        S_ADDI   s1, s0, 64         ; s1 = end: 16 numbers x 4 bytes
        S_ADDI   s2, s0, 0          ; s2 = byte offset, starts at 0
loop:   V_ADD    v1, v0, s2         ; v1 = offset + lane x 4   (s2 broadcast)
        V_LOAD   v2, [v1 + 0x1000]  ; load A
        V_LOAD   v3, [v1 + 0x2000]  ; load B
        V_ADD    v4, v2, v3         ; add
        V_STORE  v4, [v1 + 0x3000]  ; store C
        S_ADDI   s2, s2, 16         ; next 4 numbers (4 lanes x 4 bytes)
        S_BLT    s2, s1, loop       ; repeat until done
        HALT
"""

PIXEL_SHADER = """\
; Textured, lit pixel shader (spec section 6 example, corrected)
; inputs: v3 = u (ATTR0), v4 = v (ATTR1), v5 = brightness (ATTR2, 16.16)
        V_TEX    v11, v3, v4, 0     ; texel RGBA8 from texture slot 0
        S_ADDI   s3, s0, 8          ; s3 = 8 (shift amount)
        V_SRA    v12, v5, s3        ; brightness 16.16 -> 0..256
        V_PACK   v13, v12, v12, v12 ; gray RGBA8 (clamped at 255)
        V_CMUL   v14, v11, v13      ; texel x brightness, per channel
        EXPORT   COLOR, v14
        HALT
"""

MISTAKES = """\
; This program has mistakes ON PURPOSE, to show the error messages
        V_SRA    v12, v5, 8         ; the original spec line: 8 is not a register
        V_ADD    v1, v2, v16        ; only v0-v15 exist
        S_ADDI   s1, s0, 200000     ; too big for 18 bits
        S_BLT    s1, s2, nowhere    ; this label does not exist
        V_ADD    v1, s2, v3         ; first source must be a vector register
        V_MULT   v1, v2, v3         ; misspelled instruction
        HALT
"""


def demo():
    for name, source in [("vector_add", VECTOR_ADD), ("pixel_shader", PIXEL_SHADER)]:
        words, listing = assemble(source)
        save(words, listing, name)
        with open(name + ".asm", "w") as f:
            f.write(source)
        print(f"===== {name}.asm  ({len(words)} instructions) =====")
        print("ADDR    WORD      SOURCE")
        print("\n".join(listing))
        print(f"Round-trip check (disassemble -> assemble): {round_trip_check(words)}")
        print(f"Saved {name}.hex, {name}.bin, {name}.lst\n")

    print("===== mistakes.asm  (errors on purpose) =====")
    try:
        assemble(MISTAKES)
    except AssemblyFailed as failure:
        print(f"{len(failure.errors)} errors found:")
        for e in failure.errors:
            print("  " + e)


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if a.endswith(".asm")]
    if not files:
        demo()
    for path in files:
        with open(path) as f:
            source = f.read()
        try:
            words, listing = assemble(source)
        except AssemblyFailed as failure:
            print(f"{path}: {len(failure.errors)} error(s)")
            for e in failure.errors:
                print("  " + e)
            continue
        save(words, listing, path[:-4])
        print("\n".join(listing))
        print(f"{path}: {len(words)} instructions, round-trip {round_trip_check(words)}")
