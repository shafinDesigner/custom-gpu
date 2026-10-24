# sfu_check.py                                                            Phase 2
# Before writing Verilog for the special function unit, prove the HARDWARE ALGORITHMS
# give exactly the same answers as the golden model (simulator.py: sfu_rcp, sfu_rsq).
#
# The functions below copy the RTL step by step: one loop pass = one clock cycle.

import random
from simulator import sfu_rcp, sfu_rsq, s32, u32

MAX_POS = 0x7FFFFFFF


def hw_divide(top_bit, divisor):
    """Restoring division of (1 << top_bit) by divisor, one quotient bit per cycle.
    Dividend is a single 1 followed by zeros, so we shift in '1' first, then '0's."""
    rem, quo = 0, 0
    first = 1
    for _ in range(top_bit + 1):             # 33 cycles for 2^32, 49 cycles for 2^48
        rem = (rem << 1) | first             # bring down the next dividend bit
        first = 0
        if rem >= divisor:                   # does the divisor fit?
            rem -= divisor
            quo = (quo << 1) | 1
        else:
            quo = quo << 1
    return quo


def hw_isqrt(n):
    """Digit-by-digit integer square root, 2 bits of n per cycle (25 cycles for n < 2^50)."""
    res, bit = 0, 1 << 48
    for _ in range(25):
        trial = res + bit
        if n >= trial:
            n -= trial
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def hw_rcp(a):
    x = s32(a)
    if x == 0:
        return MAX_POS
    q = min(hw_divide(32, abs(x)), MAX_POS)
    return u32(-q if x < 0 else q)


def hw_rsq(a):
    x = s32(a)
    if x <= 0:
        return MAX_POS
    return min(hw_isqrt(hw_divide(48, x)), MAX_POS)


def check(name, hw, golden, values):
    bad = [(v, hw(v), golden(v)) for v in values if hw(v) != golden(v)]
    print(f"  {'PASS' if not bad else 'FAIL'}  {name}: {len(values)} values, {len(bad)} mismatches")
    for v, h, g in bad[:5]:
        print(f"        input 0x{u32(v):08X}: hardware 0x{h:08X}, golden 0x{g:08X}")
    return not bad


if __name__ == "__main__":
    random.seed(1)
    edges = [0, 1, 2, 3, 0xFFFF, 0x10000, 0x10001, MAX_POS, 0x80000000, 0x80000001,
             0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000]
    values = edges + [random.getrandbits(32) for _ in range(100_000)]
    values += [random.randint(1, 1 << 20) for _ in range(20_000)]        # small numbers
    print("===== SFU hardware algorithms vs golden model =====")
    ok = check("V_RCP  (33-cycle divider)", hw_rcp, sfu_rcp, values)
    ok &= check("V_RSQ  (49-cycle divider + 25-cycle square root)", hw_rsq, sfu_rsq, values)
    print("  Algorithms are bit-exact: safe to build in Verilog." if ok else "  FIX THE ALGORITHM FIRST.")
