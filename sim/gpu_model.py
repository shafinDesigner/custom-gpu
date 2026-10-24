# gpu_model.py
# Phase 1, task 4: the FULL GPU MODEL (golden reference), built in 3 parts.
#   Part A: command processor, input assembler, vertex shading on YOUR ISA,
#           primitive unit (projection, culling).
#   Part B: rasterizer (2x2 quads, top-left rule, early depth, interpolation),
#           pixel shading on YOUR ISA, render output unit, clear, frame buffers, swap.
#   Part C: texture unit (bilinear + mipmaps, RGBA8 / RGB565, repeat / clamp), animation.
#
# Needs assembler.py and simulator.py in the same folder.

import math
from PIL import Image, ImageDraw, ImageFont
from assembler import assemble
from simulator import Wave, Memory, s32, u32, sfu_rcp, pack, unpack, MASK32

FX = 1 << 16                      # 1.0 in 16.16
SUB = 4                           # screen positions: 12.4 (spec section 4)


def to_fx(x):
    return u32(int(round(x * FX)))


def mulfx(a, b):
    return (s32(a) * s32(b)) >> 16


# =====================================================================
# 1. SPEC TABLES  (section 7)
# =====================================================================
CMD_NOP, CMD_SET_STATE, CMD_CLEAR, CMD_DRAW = 0x00, 0x01, 0x02, 0x03
CMD_SWAP, CMD_FENCE, CMD_WAIT_VBLANK, CMD_JUMP, CMD_END = 0x04, 0x05, 0x06, 0x07, 0xFF

STATE_REGS = {
    "VS_ADDR": 0x00, "PS_ADDR": 0x01, "CONST_ADDR": 0x02, "VB_ADDR": 0x03,
    "VB_STRIDE": 0x04, "VB_ATTRS": 0x05, "IB_ADDR": 0x06, "VS_OUT_ATTRS": 0x07,
    "FB_A_ADDR": 0x10, "FB_B_ADDR": 0x11, "DEPTH_ADDR": 0x12, "STENCIL_ADDR": 0x13,
    "VIEWPORT_W": 0x14, "VIEWPORT_H": 0x15, "PROJ_FOCAL": 0x16, "PROJ_CX": 0x17,
    "PROJ_CY": 0x18, "NEAR_Z": 0x19, "CULL_MODE": 0x20, "DEPTH_MODE": 0x21,
    "STENCIL_CTRL": 0x22, "BLEND_MODE": 0x23, "COLOR_MASK": 0x24,
    "TEX0_ADDR": 0x30, "TEX1_ADDR": 0x31, "TEX2_ADDR": 0x32, "TEX3_ADDR": 0x33,
    "TEX0_INFO": 0x34, "TEX1_INFO": 0x35, "TEX2_INFO": 0x36, "TEX3_INFO": 0x37,
}
STATE_NAMES = {v: k for k, v in STATE_REGS.items()}

DEFAULT_STATE = {
    "VS_ADDR": 0, "PS_ADDR": 0, "CONST_ADDR": 0, "VB_ADDR": 0, "VB_STRIDE": 32,
    "VB_ATTRS": 3, "IB_ADDR": 0, "VS_OUT_ATTRS": 0,
    "FB_A_ADDR": 0x500000, "FB_B_ADDR": 0x900000, "DEPTH_ADDR": 0xD00000,
    "STENCIL_ADDR": 0xED0000, "VIEWPORT_W": 1280, "VIEWPORT_H": 720,
    "PROJ_FOCAL": 0, "PROJ_CX": to_fx(640), "PROJ_CY": to_fx(360), "NEAR_Z": to_fx(1.0),
    "CULL_MODE": 1, "DEPTH_MODE": 1, "STENCIL_CTRL": 0, "BLEND_MODE": 0, "COLOR_MASK": 0xF,
    "TEX0_ADDR": 0, "TEX1_ADDR": 0, "TEX2_ADDR": 0, "TEX3_ADDR": 0,
    "TEX0_INFO": 0, "TEX1_INFO": 0, "TEX2_INFO": 0, "TEX3_INFO": 0,
}


def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def is_top_left(p1, p2):
    """Top-left rule (spec 8.8): left edge (inside to the right) or flat top edge (inside below)."""
    A, B = p2[1] - p1[1], -(p2[0] - p1[0])
    return A > 0 or (A == 0 and B > 0)


def lerp8(c1, c2, f):
    """Blend two RGBA8 colors with an 8-bit fraction f (0..255), per channel."""
    a, b = unpack(c1), unpack(c2)
    return pack([(x * (256 - f) + y * f) >> 8 for x, y in zip(a, b)])


def read16(mem, addr):
    word = mem.load(addr & ~3)
    return (word >> 16) & 0xFFFF if addr & 2 else word & 0xFFFF


def write16(mem, addr, value):
    a = addr & ~3
    word = mem.load(a)
    if addr & 2:
        word = (word & 0x0000FFFF) | (value & 0xFFFF) << 16
    else:
        word = (word & 0xFFFF0000) | (value & 0xFFFF)
    mem.store(a, word)


# =====================================================================
# 2. THE GPU
# =====================================================================
class GPU:
    def __init__(self, memory):
        self.mem = memory
        self.state = dict(DEFAULT_STATE)
        self.frames_done = 0
        self.next_cu = 0                       # work distributor: round-robin
        self.stats = {"commands": 0, "draws": 0, "vertices": 0, "vertex_waves": 0,
                      "vs_instructions": 0, "waves_per_cu": [0, 0], "vs_waves_per_cu": [0, 0], "triangles_in": 0,
                      "culled_back": 0, "culled_near": 0, "culled_zero": 0, "triangles_out": 0}
        self.triangles = []                    # Part A: primitive unit output (for checking)
        self.culled = []
        self.projected = []                    # every projected vertex, in draw order
        self.back_is_a = True                  # draw into buffer A first
        self.frames = []                       # finished frames as images (for viewing)
        self.write_count = {}                  # debug: how often each pixel was written
        self.stats.update({"quads": 0, "quads_early_dropped": 0, "pixel_waves": 0,
                           "ps_instructions": 0, "helper_lanes": 0, "pixels_written": 0,
                           "pixels_depth_failed": 0, "texture_samples": 0, "texel_reads": 0,
                           "mip_levels_used": {}})

    # ---------------- 8.1 HOST INTERFACE ----------------
    def start(self, cmd_addr):
        """Host writes CMD_ADDR and CMD_START = 1."""
        self.command_processor(cmd_addr)
        return "idle"

    # ---------------- 8.2 COMMAND PROCESSOR ----------------
    def command_processor(self, addr):
        while True:
            header = self.mem.load(addr)
            code, nwords, param = header >> 24, (header >> 16) & 0xFF, header & 0xFFFF
            payload = self.mem.read_block(addr + 4, nwords)
            addr += 4 * (1 + nwords)
            self.stats["commands"] += 1

            if code == CMD_NOP:
                pass
            elif code == CMD_SET_STATE:
                self.state[STATE_NAMES[param]] = payload[0]
            elif code == CMD_CLEAR:
                self.clear(param, *payload)
            elif code == CMD_DRAW:
                self.draw(count=payload[0], first=payload[1], indexed=bool(param & 1))
            elif code == CMD_SWAP:
                self.swap_buffers()
            elif code == CMD_FENCE:
                self.mem.store(payload[0], payload[1])      # all work is already done here
            elif code == CMD_WAIT_VBLANK:
                pass                                        # no display timing in the model
            elif code == CMD_JUMP:
                addr = payload[0]
            elif code == CMD_END:
                return
            else:
                raise ValueError(f"unknown command 0x{code:02X}")

    # ---------------- buffers ----------------
    def back_addr(self):
        return self.state["FB_A_ADDR"] if self.back_is_a else self.state["FB_B_ADDR"]

    def pixel_count(self):
        return self.state["VIEWPORT_W"] * self.state["VIEWPORT_H"]

    def clear(self, flags, color, depth, stencil):
        n = self.pixel_count()
        if flags & 1:                                       # color: one word per pixel
            base = self.back_addr()
            for i in range(n):
                self.mem.store(base + 4 * i, color)
        if flags & 2:                                       # depth: 16 bits, two per word
            base = self.state["DEPTH_ADDR"]
            for i in range((n + 1) // 2):
                self.mem.store(base + 4 * i, (depth & 0xFFFF) * 0x10001)
        if flags & 4:                                       # stencil: 8 bits, four per word
            base = self.state["STENCIL_ADDR"]
            for i in range((n + 3) // 4):
                self.mem.store(base + 4 * i, (stencil & 0xFF) * 0x01010101)
        self.write_count = {}

    def swap_buffers(self):
        """The finished back buffer becomes the front buffer (shown); draw into the other."""
        self.frames.append(self.read_frame(self.back_addr()))
        self.back_is_a = not self.back_is_a
        self.frames_done += 1

    def read_frame(self, base):
        st = self.state
        data = bytearray()
        for w in self.mem.read_block(base, self.pixel_count()):
            data += bytes((w & 255, (w >> 8) & 255, (w >> 16) & 255))   # R, G, B
        return Image.frombytes("RGB", (st["VIEWPORT_W"], st["VIEWPORT_H"]), bytes(data))

    def read_depth_image(self):
        st = self.state
        vals = [read16(self.mem, st["DEPTH_ADDR"] + 2 * i) for i in range(self.pixel_count())]
        used = [v for v in vals if v > 0]
        lo, hi = (min(used), max(used)) if used else (0, 1)
        img = Image.new("L", (st["VIEWPORT_W"], st["VIEWPORT_H"]))
        img.putdata([0 if v == 0 else int(60 + 195 * (v - lo) / max(1, hi - lo)) for v in vals])
        return img

    # ---------------- THE DRAW: the whole pipeline ----------------
    def draw(self, count, first, indexed):
        self.stats["draws"] += 1
        vertices = self.input_assembler(count, first, indexed)
        shaded = self.vertex_shading(vertices)
        triangles = self.primitive_unit(shaded)
        self.triangles.extend(triangles)
        for prim_id, tri in enumerate(triangles):
            self.rasterizer(tri, prim_id)

    # ---------------- 8.3 INPUT ASSEMBLER ----------------
    def read_index(self, i):
        addr = self.state["IB_ADDR"] + 2 * i                # 16-bit indices
        word = self.mem.load(addr & ~3)
        return (word >> 16) & 0xFFFF if addr & 2 else word & 0xFFFF

    def input_assembler(self, count, first, indexed):
        """Returns one list of attributes per vertex, in draw order (no reuse in v0.1)."""
        st = self.state
        vertices = []
        for i in range(first, first + count):
            k = self.read_index(i) if indexed else i
            base = st["VB_ADDR"] + k * st["VB_STRIDE"]
            vertices.append(self.mem.read_block(base, st["VB_ATTRS"]))
        self.stats["vertices"] += len(vertices)
        return vertices

    # ---------------- 8.4 WORK DISTRIBUTOR + 8.5 COMPUTE UNITS: vertex waves ----------------
    def vertex_shading(self, vertices):
        st = self.state
        shaded = []
        for start in range(0, len(vertices), 4):          # batches of 4 = one wave
            batch = vertices[start:start + 4]
            cu = self.next_cu
            self.next_cu = 1 - self.next_cu
            wave = Wave(self.mem, pc=st["VS_ADDR"], cu_id=cu, wave_type=0,
                        frame=self.frames_done, texture_sampler=self.texture_unit)
            wave.mask = [lane < len(batch) for lane in range(4)]   # unused lanes off
            wave.s[1] = st["CONST_ADDR"]
            wave.s[2] = start
            for a in range(st["VB_ATTRS"]):                # inputs v0..v7
                wave.v[a] = [batch[l][a] if l < len(batch) else 0 for l in range(4)]
            wave.run()
            self.stats["vertex_waves"] += 1
            self.stats["waves_per_cu"][cu] += 1
            self.stats["vs_waves_per_cu"][cu] += 1
            self.stats["vs_instructions"] += wave.count
            for l in range(len(batch)):
                shaded.append({
                    "pos": [wave.exports[k][l] for k in ("POS_X", "POS_Y", "POS_Z")],
                    "attrs": [wave.exports[f"ATTR{a}"][l] for a in range(st["VS_OUT_ATTRS"])],
                })
        return shaded

    # ---------------- 8.7 PRIMITIVE UNIT ----------------
    def primitive_unit(self, shaded):
        st = self.state
        out = []
        for t in range(0, len(shaded) - 2, 3):
            tri = shaded[t:t + 3]
            self.stats["triangles_in"] += 1
            # Near plane: v0.1 discards the whole triangle (spec 8.7)
            if any(s32(v["pos"][2]) < s32(st["NEAR_Z"]) for v in tri):
                self.stats["culled_near"] += 1
                continue
            screen = []
            for v in tri:
                x, y, z = v["pos"]
                inv_z = sfu_rcp(z)                                       # reciprocal unit
                # every result is a 32-bit hardware value: s32() wraps like the RTL
                sx = s32(s32(st["PROJ_CX"]) + s32(mulfx(mulfx(x, st["PROJ_FOCAL"]), inv_z)))
                sy = s32(s32(st["PROJ_CY"]) - s32(mulfx(mulfx(y, st["PROJ_FOCAL"]), inv_z)))
                screen.append({
                    "x": sx >> (16 - SUB), "y": sy >> (16 - SUB),         # 12.4
                    "inv_z": inv_z,                                       # 0.16
                    "attrs_z": [s32(mulfx(a, inv_z)) for a in v["attrs"]],  # attr x 1/z
                })
            self.projected.extend(screen)
            a, b, c = ((p["x"], p["y"]) for p in screen)
            area = edge(a, b, c)
            if area == 0:
                self.stats["culled_zero"] += 1
                continue
            # Front faces have area < 0 (corners counter-clockwise from outside, y down)
            if (st["CULL_MODE"] == 1 and area > 0) or (st["CULL_MODE"] == 2 and area < 0):
                self.stats["culled_back"] += 1
                self.culled.append(screen)
                continue
            self.stats["triangles_out"] += 1
            out.append(screen)
        return out


    # ---------------- 8.8 RASTERIZER ----------------
    def rasterizer(self, tri, prim_id):
        st = self.state
        W, H = st["VIEWPORT_W"], st["VIEWPORT_H"]
        a, b, c = tri
        pts = [(v["x"], v["y"]) for v in (a, b, c)]
        area = edge(*pts)
        if area > 0:                                    # make every triangle wind the same way
            b, c = c, b
            pts = [pts[0], pts[2], pts[1]]
            area = -area
        pa, pb, pc = pts
        inv_area = (1 << 48) // (-area)                 # SETUP: one divide per triangle
        edges = [(pb, pc), (pc, pa), (pa, pb)]          # edge i -> weight of vertex i
        bias = [0 if is_top_left(p1, p2) else -1 for p1, p2 in edges]   # top-left rule
        verts = (a, b, c)

        # Bounding box in pixels, starting on even x and y (2x2 quads)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        min_x, min_y = max(min(xs) >> SUB, 0) & ~1, max(min(ys) >> SUB, 0) & ~1
        max_x, max_y = min((max(xs) >> SUB) + 1, W), min((max(ys) >> SUB) + 1, H)

        for qy in range(min_y, max_y, 2):
            for qx in range(min_x, max_x, 2):
                lanes = [(qx, qy), (qx + 1, qy), (qx, qy + 1), (qx + 1, qy + 1)]
                centers = [((x << SUB) + 8, (y << SUB) + 8) for x, y in lanes]
                E = [[-edge(p1, p2, ctr) for p1, p2 in edges] for ctr in centers]
                cover = [x < W and y < H and all(e + bb >= 0 for e, bb in zip(Ei, bias))
                         for (x, y), Ei in zip(lanes, E)]
                if not any(cover):
                    continue
                self.stats["quads"] += 1

                # Weights (0.16) and closeness 1/z (0.16) for ALL 4 lanes (helper lanes too)
                w = [[(e * inv_area) >> 32 for e in Ei] for Ei in E]
                # closeness saturates at 32 bits (it goes into a 32-bit register)
                close = [min(0xFFFFFFFF, max(1, (wi[0] * a["inv_z"] + wi[1] * b["inv_z"] +
                                                 wi[2] * c["inv_z"]) >> 16))
                         for wi in w]

                # EARLY DEPTH TEST: hidden pixels never reach the shader
                if st["DEPTH_MODE"] != 0:
                    for l, (x, y) in enumerate(lanes):
                        if cover[l] and min(close[l], 0xFFFF) <= read16(self.mem, self.depth_addr(x, y)):
                            cover[l] = False
                    if not any(cover):
                        self.stats["quads_early_dropped"] += 1
                        continue

                # Perspective-correct attributes: (blended attr/z) x (1 / blended 1/z)
                attrs = []
                for k in range(len(a["attrs_z"])):
                    row = []
                    for l in range(4):
                        az = (w[l][0] * a["attrs_z"][k] + w[l][1] * b["attrs_z"][k] +
                              w[l][2] * c["attrs_z"][k]) >> 16
                        row.append(u32(mulfx(az, sfu_rcp(close[l]))))   # reciprocal unit
                    attrs.append(row)
                self.pixel_shading(lanes, cover, close, attrs, prim_id)

    def depth_addr(self, x, y):
        return self.state["DEPTH_ADDR"] + 2 * (y * self.state["VIEWPORT_W"] + x)

    # ---------------- 8.4 + 8.5: pixel waves on the compute units ----------------
    def pixel_shading(self, lanes, cover, close, attrs, prim_id):
        st = self.state
        cu = self.next_cu
        self.next_cu = 1 - self.next_cu
        wave = Wave(self.mem, pc=st["PS_ADDR"], cu_id=cu, wave_type=1, frame=self.frames_done,
                    texture_sampler=self.texture_unit)
        wave.mask = list(cover)                         # uncovered lanes = helper lanes
        wave.s[1] = st["CONST_ADDR"]
        wave.s[2] = prim_id
        wave.v[0] = [x for x, _ in lanes]               # pixel x
        wave.v[1] = [y for _, y in lanes]               # pixel y
        wave.v[2] = list(close)                         # closeness 1/z
        for k, row in enumerate(attrs[:8]):
            wave.v[3 + k] = row                         # ATTR0..7
        wave.run()
        self.stats["pixel_waves"] += 1
        self.stats["waves_per_cu"][cu] += 1
        self.stats["ps_instructions"] += wave.count
        self.stats["helper_lanes"] += 4 - sum(cover)
        colors = wave.exports.get("COLOR", [None] * 4)
        self.render_output(lanes, wave.mask, close, colors)

    # ---------------- 8.6 TEXTURE UNIT ----------------
    def texture_unit(self, slot, us, vs, wave_type):
        """Executes V_TEX for one wave: returns 4 RGBA8 colors (one per lane)."""
        st = self.state
        base, info = st[f"TEX{slot}_ADDR"], st[f"TEX{slot}_INFO"]
        size_log2 = info & 0xF
        levels = max(1, (info >> 4) & 0xF)
        rgb565, clamp = (info >> 8) & 1, (info >> 9) & 1
        bilinear, mipmaps = (info >> 10) & 1, (info >> 11) & 1

        # MIP LEVEL: one per quad, from the differences across the quad (pixel waves only)
        level = 0
        if mipmaps and wave_type == 1:
            u, v = [s32(x) for x in us], [s32(x) for x in vs]
            d = max(abs(u[1] - u[0]), abs(v[1] - v[0]),      # lane 0 -> 1: one pixel right
                    abs(u[2] - u[0]), abs(v[2] - v[0]))      # lane 0 -> 2: one pixel down
            texels = (d << size_log2) >> 16                  # footprint in whole texels
            level = texels.bit_length() - 1 if texels >= 1 else 0   # log2 = highest 1 bit
            level = min(level, levels - 1, size_log2)
        hist = self.stats["mip_levels_used"]
        hist[level] = hist.get(level, 0) + 1

        bpp = 2 if rgb565 else 4
        size0 = 1 << size_log2
        level_base = base + sum((size0 >> i) ** 2 * bpp for i in range(level))
        size = size0 >> level

        def texel(tx, ty):
            if clamp:
                tx, ty = min(max(tx, 0), size - 1), min(max(ty, 0), size - 1)
            else:                                            # repeat (size is a power of two)
                tx, ty = tx & (size - 1), ty & (size - 1)
            index = ty * size + tx
            self.stats["texel_reads"] += 1
            if not rgb565:
                return self.mem.load(level_base + 4 * index)
            c = read16(self.mem, level_base + 2 * index)
            r, g, b = (c >> 11) & 31, (c >> 5) & 63, c & 31
            return pack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), 255])

        out = []
        for l in range(4):
            u, v = s32(us[l]), s32(vs[l])
            if not bilinear:                                 # point: nearest texel
                out.append(texel((u * size) >> 16, (v * size) >> 16))
                continue
            tx = u * size - (FX >> 1)                        # 16.16, measured from texel centers
            ty = v * size - (FX >> 1)
            x0, y0 = tx >> 16, ty >> 16
            fx, fy = (tx >> 8) & 255, (ty >> 8) & 255        # 8-bit blend fractions
            top = lerp8(texel(x0, y0), texel(x0 + 1, y0), fx)
            bottom = lerp8(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fx)
            out.append(lerp8(top, bottom, fy))
        self.stats["texture_samples"] += 4
        return out

    # ---------------- 8.9 RENDER OUTPUT UNIT ----------------
    def render_output(self, lanes, mask, close, colors):
        st = self.state
        W = st["VIEWPORT_W"]
        for l, (x, y) in enumerate(lanes):
            if not mask[l] or colors[l] is None:
                continue
            if st["DEPTH_MODE"] != 0:
                d = min(close[l], 0xFFFF)
                if d <= read16(self.mem, self.depth_addr(x, y)):
                    self.stats["pixels_depth_failed"] += 1
                    continue
                if st["DEPTH_MODE"] == 1:
                    write16(self.mem, self.depth_addr(x, y), d)
            addr = self.back_addr() + 4 * (y * W + x)
            src = colors[l]
            if st["BLEND_MODE"] != 0:
                dst = self.mem.load(addr)
                s_ch, d_ch = unpack(src), unpack(dst)
                if st["BLEND_MODE"] == 1:               # alpha: a' = a + (a >> 7) maps 0..255 -> 0..256
                    ap = s_ch[3] + (s_ch[3] >> 7)
                    src = pack([(sv * ap + dv * (256 - ap)) >> 8 for sv, dv in zip(s_ch, d_ch)])
                else:                                   # additive, clamped at 255
                    src = pack([min(255, sv + dv) for sv, dv in zip(s_ch, d_ch)])
            if st["COLOR_MASK"] != 0xF:
                dst = self.mem.load(addr)
                keep = sum(0xFF << (8 * i) for i in range(4) if not st["COLOR_MASK"] >> i & 1)
                src = (src & ~keep & MASK32) | (dst & keep)
            self.mem.store(addr, src)
            self.stats["pixels_written"] += 1
            self.write_count[(x, y)] = self.write_count.get((x, y), 0) + 1


# =====================================================================
# 3. DRIVER HELPERS (the host side: what a CPU driver would do)
# =====================================================================
class CommandList:
    def __init__(self):
        self.words = []

    def _cmd(self, code, param=0, payload=()):
        self.words.append(code << 24 | len(payload) << 16 | (param & 0xFFFF))
        self.words.extend(u32(p) for p in payload)

    def set_state(self, name, value):
        self._cmd(CMD_SET_STATE, STATE_REGS[name], [value])

    def clear(self, color=True, depth=True, stencil=False, rgba=0xFF1E1E1E, z=0, s=0):
        self._cmd(CMD_CLEAR, color | depth << 1 | stencil << 2, [rgba, z, s])

    def draw(self, count, first=0, indexed=True):
        self._cmd(CMD_DRAW, int(indexed), [count, first])

    def swap(self):
        self._cmd(CMD_SWAP)

    def fence(self, addr, value):
        self._cmd(CMD_FENCE, 0, [addr, value])

    def end(self):
        self._cmd(CMD_END)


def write_indices(mem, addr, indices):
    """16-bit indices, two per 32-bit word, little-endian."""
    padded = list(indices) + [0] * (len(indices) % 2)
    mem.write_block(addr, [padded[i] | padded[i + 1] << 16 for i in range(0, len(padded), 2)])


# =====================================================================
# 4. THE TEST SCENE: a cube, transformed by YOUR vertex shader
# =====================================================================
CUBE_VS = """\
; Cube vertex shader: 3x4 matrix (rotate + tilt + move) from the constant buffer
; inputs : v0 = x, v1 = y, v2 = z, v3 = u, v4 = v, v5 = brightness   (all 16.16)
; s1     : constant buffer = m00 m01 m02  m10 m11 m12  m20 m21 m22  tx ty tz
        S_LOAD   s3,  [s1 + 0]
        S_LOAD   s4,  [s1 + 4]
        S_LOAD   s5,  [s1 + 8]
        S_LOAD   s6,  [s1 + 12]
        S_LOAD   s7,  [s1 + 16]
        S_LOAD   s8,  [s1 + 20]
        S_LOAD   s9,  [s1 + 24]
        S_LOAD   s10, [s1 + 28]
        S_LOAD   s11, [s1 + 32]
        S_LOAD   s12, [s1 + 36]
        S_LOAD   s13, [s1 + 40]
        S_LOAD   s14, [s1 + 44]
        V_MULFX  v8, v0, s3             ; x' = m00*x
        V_MADFX  v8, v1, s4, v8         ;    + m01*y
        V_MADFX  v8, v2, s5, v8         ;    + m02*z
        V_ADD    v8, v8, s12            ;    + tx
        V_MULFX  v9, v0, s6             ; y'
        V_MADFX  v9, v1, s7, v9
        V_MADFX  v9, v2, s8, v9
        V_ADD    v9, v9, s13
        V_MULFX  v10, v0, s9            ; z'
        V_MADFX  v10, v1, s10, v10
        V_MADFX  v10, v2, s11, v10
        V_ADD    v10, v10, s14
        EXPORT   POS_X, v8
        EXPORT   POS_Y, v9
        EXPORT   POS_Z, v10
        EXPORT   ATTR0, v3              ; u
        EXPORT   ATTR1, v4              ; v
        EXPORT   ATTR2, v5              ; brightness
        HALT
"""

# 24 vertices (4 per face, so each face has its own u, v and brightness)
CUBE_CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
CUBE_FACES = [(0, 1, 2, 3), (5, 4, 7, 6), (1, 5, 6, 2), (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)]
FACE_UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
FACE_BRIGHTNESS = [1.0, 0.75, 0.85, 0.7, 0.95, 0.6]

# Memory addresses (spec section 9)
CMD_ADDR, VS_ADDR, PS_ADDR = 0x000000, 0x010000, 0x010400
CONST_ADDR, VB_ADDR, IB_ADDR = 0x020000, 0x030000, 0x038000

# Demo viewport: 16:9 like 720p, but small so Python is fast
WIDTH, HEIGHT, FOCAL, DISTANCE = 320, 180, 180, 4.5


def cube_vertex_data():
    verts = []
    for face, quad in enumerate(CUBE_FACES):
        for corner, (u, v) in zip(quad, FACE_UV):
            x, y, z = CUBE_CORNERS[corner]
            verts.append([to_fx(x), to_fx(y), to_fx(z), to_fx(u), to_fx(v),
                          to_fx(FACE_BRIGHTNESS[face])])
    indices = []
    for f in range(6):
        b = 4 * f
        indices += [b, b + 1, b + 2, b, b + 2, b + 3]
    return verts, indices


def cube_matrix(spin_deg, tilt_deg, scale=1.0, offset=(0.0, 0.0, 0.0)):
    """Driver side: scale, spin around Y, place at offset, tilt the camera, move away.
    Returns a 3x3 matrix m and translation t (floats; the driver converts to 16.16)."""
    c, s = math.cos(math.radians(spin_deg)), math.sin(math.radians(spin_deg))
    ct, st = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    rot_y = [[c * scale, 0, s * scale], [0, scale, 0], [-s * scale, 0, c * scale]]
    tilt_x = [[1, 0, 0], [0, ct, -st], [0, st, ct]]
    m = [[sum(tilt_x[i][k] * rot_y[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    t = [sum(tilt_x[i][k] * offset[k] for k in range(3)) for i in range(3)]
    t[2] += DISTANCE
    return m, t


def write_constants(mem, addr, m, t):
    mem.write_block(addr, [to_fx(m[i][j]) for i in range(3) for j in range(3)] +
                    [to_fx(x) for x in t])


def build_scene(mem, spin_deg=35, tilt_deg=-25):
    """Everything the host writes into GPU memory before pressing start."""
    mem.write_block(VS_ADDR, assemble(CUBE_VS)[0])
    mem.write_block(PS_ADDR, assemble(UV_PS)[0])
    m, t = cube_matrix(spin_deg, tilt_deg)
    write_constants(mem, CONST_ADDR, m, t)
    verts, indices = cube_vertex_data()
    for k, attrs in enumerate(verts):
        mem.write_block(VB_ADDR + k * 24, attrs)
    write_indices(mem, IB_ADDR, indices)

    cl = CommandList()
    for name, value in [("VS_ADDR", VS_ADDR), ("PS_ADDR", PS_ADDR), ("CONST_ADDR", CONST_ADDR),
                        ("VB_ADDR", VB_ADDR), ("VB_STRIDE", 24), ("VB_ATTRS", 6),
                        ("IB_ADDR", IB_ADDR), ("VS_OUT_ATTRS", 3),
                        ("VIEWPORT_W", WIDTH), ("VIEWPORT_H", HEIGHT),
                        ("PROJ_FOCAL", to_fx(FOCAL)), ("PROJ_CX", to_fx(WIDTH / 2)),
                        ("PROJ_CY", to_fx(HEIGHT / 2)), ("CULL_MODE", 1)]:
        cl.set_state(name, value)
    cl.clear()
    cl.draw(count=len(indices))
    cl.swap()
    cl.end()
    mem.write_block(CMD_ADDR, cl.words)
    return m, t, verts, indices


UV_PS = """\
; UV debug pixel shader: red = u, green = v, blue = 128, then face brightness
; inputs: v3 = u (ATTR0), v4 = v (ATTR1), v5 = brightness (ATTR2), all 16.16
        S_ADDI   s3, s0, 255
        S_ADDI   s4, s0, 16
        V_MUL    v6, v3, s3             ; u x 255   (still 16.16)
        V_SRA    v6, v6, s4             ; -> 0..255
        V_MUL    v7, v4, s3             ; v x 255
        V_SRA    v7, v7, s4             ; -> 0..255
        S_ADDI   s5, s0, 128
        V_BCAST  v8, s5                 ; blue = 128
        V_PACK   v9, v6, v7, v8         ; RGBA8 (u, v, 128)
        S_ADDI   s6, s0, 8
        V_SRA    v10, v5, s6            ; brightness 16.16 -> 0..256
        V_PACK   v11, v10, v10, v10     ; gray
        V_CMUL   v12, v9, v11           ; color x brightness
        EXPORT   COLOR, v12
        HALT
"""

SMALL_CUBE_CONST = CONST_ADDR + 0x40


def build_part_b(mem):
    """Two frames: (1) one cube, depth test OFF; (2) two intersecting cubes, depth test ON."""
    mem.write_block(VS_ADDR, assemble(CUBE_VS)[0])
    mem.write_block(PS_ADDR, assemble(UV_PS)[0])
    write_constants(mem, CONST_ADDR, *cube_matrix(35, -25))
    write_constants(mem, SMALL_CUBE_CONST, *cube_matrix(70, -25, scale=0.55, offset=(1.0, 0.35, -0.3)))
    verts, indices = cube_vertex_data()
    for k, attrs in enumerate(verts):
        mem.write_block(VB_ADDR + k * 24, attrs)
    write_indices(mem, IB_ADDR, indices)

    cl = CommandList()
    for name, value in [("VS_ADDR", VS_ADDR), ("PS_ADDR", PS_ADDR), ("VB_ADDR", VB_ADDR),
                        ("VB_STRIDE", 24), ("VB_ATTRS", 6), ("IB_ADDR", IB_ADDR),
                        ("VS_OUT_ATTRS", 3), ("VIEWPORT_W", WIDTH), ("VIEWPORT_H", HEIGHT),
                        ("PROJ_FOCAL", to_fx(FOCAL)), ("PROJ_CX", to_fx(WIDTH / 2)),
                        ("PROJ_CY", to_fx(HEIGHT / 2)), ("CULL_MODE", 1)]:
        cl.set_state(name, value)
    # Frame 1: one cube, no depth test (a convex cube needs only culling)
    cl.set_state("CONST_ADDR", CONST_ADDR)
    cl.set_state("DEPTH_MODE", 0)
    cl.clear()
    cl.draw(count=len(indices))
    cl.swap()
    # Frame 2: big cube + small cube poking through it, depth test ON
    cl.set_state("DEPTH_MODE", 1)
    cl.clear(z=0)
    cl.draw(count=len(indices))
    cl.set_state("CONST_ADDR", SMALL_CUBE_CONST)
    cl.draw(count=len(indices))
    cl.swap()
    cl.end()
    mem.write_block(CMD_ADDR, cl.words)


def convex_hull(points):
    pts = sorted(set(points))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def strictly_inside(hull, x, y, margin=0.05):
    n = len(hull)
    for i in range(n):
        (x1, y1), (x2, y2) = hull[i], hull[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross / math.hypot(x2 - x1, y2 - y1) < margin:
            return False
    return True


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def part_b():
    mem = Memory()
    build_part_b(mem)
    gpu = GPU(mem)

    # Run frame 1 alone first, to check it before frame 2 overwrites the debug counters
    gpu.start(CMD_ADDR)
    s = gpu.stats
    print("\n===== Part B: rasterizer -> pixel shading -> render output unit =====")
    print(f"  Frames            : {gpu.frames_done} swapped "
          f"(buffer A, then buffer B)")
    print(f"  Rasterizer        : {s['quads']} quads, {s['quads_early_dropped']} dropped by early depth")
    print(f"  Pixel shading     : {s['pixel_waves']} pixel waves, {s['ps_instructions']} instructions "
          f"on YOUR ISA, {s['helper_lanes']} helper lanes")
    print(f"  Render output     : {s['pixels_written']} pixels written, "
          f"{s['pixels_depth_failed']} failed the late depth test")
    return gpu


def check_frame1():
    """Frame 1 alone: every pixel inside the cube written exactly once (top-left rule)."""
    mem = Memory()
    build_part_b(mem)
    gpu = GPU(mem)
    # stop after frame 1: run the command list but replace frame 2 with END
    words = []
    addr = CMD_ADDR
    while True:
        header = mem.load(addr)
        n = (header >> 16) & 0xFF
        words += mem.read_block(addr, 1 + n)
        addr += 4 * (1 + n)
        if header >> 24 == CMD_SWAP:
            break
    words.append(CMD_END << 24)
    mem.write_block(CMD_ADDR, words)
    gpu.start(CMD_ADDR)
    hull = convex_hull([(p["x"] / 16, p["y"] / 16) for tri in gpu.triangles for p in tri])
    twice = sum(1 for n in gpu.write_count.values() if n > 1)
    holes = sum(1 for y in range(HEIGHT) for x in range(WIDTH)
                if strictly_inside(hull, x + 0.5, y + 0.5) and (x, y) not in gpu.write_count)
    print(f"  Frame 1 check     : {twice} pixels written twice, {holes} holes inside the cube "
          f"({'PASS' if twice == holes == 0 else 'FAIL'})")


# =====================================================================
# 5. PART A CHECKS
# =====================================================================
def float_screen_position(corner, m, t):
    x, y, z = (sum(m[i][j] * corner[j] for j in range(3)) + t[i] for i in range(3))
    return WIDTH / 2 + x * FOCAL / z, HEIGHT / 2 - y * FOCAL / z


def part_a():
    mem = Memory()
    m, t, verts, indices = build_scene(mem)
    gpu = GPU(mem)
    gpu.start(CMD_ADDR)

    s = gpu.stats
    print("===== Part A: command processor -> input assembler -> vertex shading -> primitive unit =====")
    print(f"  Command processor : {s['commands']} commands, {s['draws']} draw, {gpu.frames_done} swap")
    print(f"  Input assembler   : {s['vertices']} vertices fetched (36 indices, no reuse)")
    print(f"  Vertex shading    : {s['vertex_waves']} waves of 4 lanes "
          f"(CU0: {s['vs_waves_per_cu'][0]}, CU1: {s['vs_waves_per_cu'][1]}), "
          f"{s['vs_instructions']} instructions on YOUR ISA")
    print(f"  Primitive unit    : {s['triangles_in']} triangles in -> {s['triangles_out']} kept, "
          f"{s['culled_back']} back faces culled, {s['culled_near']} near-plane, "
          f"{s['culled_zero']} zero-area")

    # Compare fixed-point screen positions with an exact float calculation
    worst = 0.0
    for k, p in zip(indices, gpu.projected):
        ex, ey = float_screen_position(CUBE_CORNERS[CUBE_FACES[k // 4][k % 4]], m, t)
        worst = max(worst, abs(p["x"] / 16 - ex), abs(p["y"] / 16 - ey))
    print(f"  Accuracy check    : largest error vs exact float = {worst:.3f} pixel "
          f"({'PASS' if worst < 0.1 else 'FAIL'}, limit 0.1)")

    # Picture: culled triangles dim, kept triangles bright
    S = 3
    img = Image.new("RGB", (WIDTH * S, HEIGHT * S), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for tri in gpu.culled:
        draw.polygon([(p["x"] * S / 16, p["y"] * S / 16) for p in tri], outline=(90, 90, 86))
    for tri in gpu.triangles:
        draw.polygon([(p["x"] * S / 16, p["y"] * S / 16) for p in tri], outline=(175, 169, 236))
        for p in tri:
            x, y = p["x"] * S / 16, p["y"] * S / 16
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(239, 159, 39))
    img.save("part_a_geometry.png")
    print("  Saved part_a_geometry.png  (bright = kept, dim = culled back faces)")
    return gpu, img


# =====================================================================
# 7. PART C: TEXTURES
# =====================================================================
TEX_ADDR = 0x100000                       # texture region (spec section 9)


def make_texture(size=128):
    """The same 'GPU' checkerboard texture as Step 1."""
    tex = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(tex)
    check = size // 8
    for row in range(8):
        for col in range(8):
            color = (238, 237, 254) if (row + col) % 2 == 0 else (127, 119, 221)
            draw.rectangle([col * check, row * check, (col + 1) * check - 1,
                            (row + 1) * check - 1], fill=color)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(229, 75, 74), width=4)
    draw.rectangle([22, 44, 106, 84], fill=(60, 52, 137))
    draw.text((size // 2, size // 2), "GPU", fill=(255, 255, 255), font=font(32), anchor="mm")
    return tex


def upload_texture(mem, addr, image):
    """DRIVER: write an RGBA8 texture and all its mip levels; return the TEXn_INFO value.
    Each smaller level = rounded average of 2x2 texels of the level above."""
    size = image.width
    level = [[pack([r, g, b, 255]) for r, g, b in [image.getpixel((x, y)) for x in range(size)]]
             for y in range(size)]
    levels = 0
    while True:
        mem.write_block(addr, [c for row in level for c in row])
        addr += 4 * size * size
        levels += 1
        if size == 1:
            break
        half = size // 2
        smaller = []
        for y in range(half):
            row = []
            for x in range(half):
                quad = [unpack(level[2 * y + dy][2 * x + dx]) for dy in (0, 1) for dx in (0, 1)]
                row.append(pack([(sum(ch) + 2) >> 2 for ch in zip(*quad)]))
            smaller.append(row)
        level, size = smaller, half
    size_log2 = image.width.bit_length() - 1
    return size_log2 | levels << 4 | 1 << 10 | 1 << 11        # RGBA8, repeat, bilinear, mipmaps


TEXTURED_PS = """\
; Textured, lit pixel shader (spec section 6 example)
; inputs: v3 = u (ATTR0), v4 = v (ATTR1), v5 = brightness (ATTR2, 16.16)
        V_TEX    v11, v3, v4, 0         ; texel RGBA8 from texture slot 0
        S_ADDI   s3, s0, 8
        V_SRA    v12, v5, s3            ; brightness 16.16 -> 0..256
        V_PACK   v13, v12, v12, v12     ; gray RGBA8 (clamped at 255)
        V_CMUL   v14, v11, v13          ; texel x brightness, per channel
        EXPORT   COLOR, v14
        HALT
"""

FLOOR_VB, FLOOR_IB = VB_ADDR + 0x1000, IB_ADDR + 0x100
FRAME_CONSTS = CONST_ADDR + 0x100        # one constant block per frame (64 bytes each)
FLOOR_CONST = CONST_ADDR + 0x80


def floor_vertex_data():
    """A floor under the cube: x from -6 to 6, z from -3 (near) to 10 (far), at y = -1.
    The texture repeats every 2 units."""
    corners = [(-6, -3), (6, -3), (6, 10), (-6, 10)]          # near-left ... far-left
    verts = [[to_fx(x), to_fx(-1), to_fx(z), to_fx((x + 6) / 2), to_fx((10 - z) / 2), to_fx(0.8)]
             for x, z in corners]
    return verts, [0, 1, 2, 0, 2, 3]


def build_part_c(mem, frames=24):
    mem.write_block(VS_ADDR, assemble(CUBE_VS)[0])
    mem.write_block(PS_ADDR, assemble(TEXTURED_PS)[0])
    tex_info = upload_texture(mem, TEX_ADDR, make_texture())

    cube_v, cube_i = cube_vertex_data()
    for k, attrs in enumerate(cube_v):
        mem.write_block(VB_ADDR + k * 24, attrs)
    write_indices(mem, IB_ADDR, cube_i)
    floor_v, floor_i = floor_vertex_data()
    for k, attrs in enumerate(floor_v):
        mem.write_block(FLOOR_VB + k * 24, attrs)
    write_indices(mem, FLOOR_IB, floor_i)

    write_constants(mem, FLOOR_CONST, *cube_matrix(0, -25))
    for f in range(frames):
        write_constants(mem, FRAME_CONSTS + 0x40 * f, *cube_matrix(f * 360 / frames, -25))

    cl = CommandList()
    for name, value in [("VS_ADDR", VS_ADDR), ("PS_ADDR", PS_ADDR), ("VB_STRIDE", 24),
                        ("VB_ATTRS", 6), ("VS_OUT_ATTRS", 3),
                        ("VIEWPORT_W", WIDTH), ("VIEWPORT_H", HEIGHT),
                        ("PROJ_FOCAL", to_fx(FOCAL)), ("PROJ_CX", to_fx(WIDTH / 2)),
                        ("PROJ_CY", to_fx(HEIGHT / 2)), ("CULL_MODE", 1), ("DEPTH_MODE", 1),
                        ("TEX0_ADDR", TEX_ADDR), ("TEX0_INFO", tex_info)]:
        cl.set_state(name, value)
    for f in range(frames):                                   # the whole animation, one list
        cl.clear(rgba=0xFF1E1E1E, z=0)
        cl.set_state("VB_ADDR", FLOOR_VB)
        cl.set_state("IB_ADDR", FLOOR_IB)
        cl.set_state("CONST_ADDR", FLOOR_CONST)
        cl.draw(count=6)
        cl.set_state("VB_ADDR", VB_ADDR)
        cl.set_state("IB_ADDR", IB_ADDR)
        cl.set_state("CONST_ADDR", FRAME_CONSTS + 0x40 * f)
        cl.draw(count=36)
        cl.swap()
    cl.fence(0x00F000, 0x600D)                                # tell the host we are done
    cl.end()
    mem.write_block(CMD_ADDR, cl.words)
    return tex_info


def texture_unit_tests():
    """Unit tests for the texture unit on a tiny 4x4 texture."""
    mem = Memory()
    gpu = GPU(mem)
    colors = [pack([16 * i, 255 - 16 * i, 7 * i, 255]) for i in range(16)]   # 16 different texels
    mem.write_block(TEX_ADDR, colors)
    base_info = 2 | 1 << 4                                        # 4x4, 1 level
    gpu.state["TEX0_ADDR"] = TEX_ADDR
    center = lambda i: to_fx((i + 0.5) / 4)                       # u or v at a texel center
    results = []

    def sample(info, us, vs, wave_type=0):
        gpu.state["TEX0_INFO"] = info
        return gpu.texture_unit(0, [u32(u) for u in us], [u32(v) for v in vs], wave_type)

    out = sample(base_info, [center(1)] * 4, [center(2)] * 4)
    results.append(("point: texel center gives that texel", out[0] == colors[2 * 4 + 1]))
    out = sample(base_info | 1 << 10, [center(1)] * 4, [center(2)] * 4)
    results.append(("bilinear: texel center gives that texel exactly", out[0] == colors[9]))
    out = sample(base_info | 1 << 10, [to_fx(2 / 4)] * 4, [center(0)] * 4)
    half = pack([(a + b) >> 1 for a, b in zip(unpack(colors[1]), unpack(colors[2]))])
    results.append(("bilinear: halfway between two texels = their average", out[0] == half))
    out_a = sample(base_info, [center(1)] * 4, [center(0)] * 4)
    out_b = sample(base_info, [center(1) + FX] * 4, [center(0)] * 4)
    results.append(("repeat: u + 1.0 gives the same texel", out_a == out_b))
    out = sample(base_info | 1 << 9, [to_fx(1.7)] * 4, [center(0)] * 4)
    results.append(("clamp: u = 1.7 gives the last texel", out[0] == colors[3]))
    mem.write_block(TEX_ADDR + 0x100, [0x0000FFFF])               # RGB565: texel 0 white, texel 1 black
    gpu.state["TEX0_ADDR"] = TEX_ADDR + 0x100
    out = sample(2 | 1 << 4 | 1 << 8, [center(0)] * 4, [center(0)] * 4)
    results.append(("RGB565 white expands to 255, 255, 255", out[0] == 0xFFFFFFFF))
    gpu.state["TEX0_ADDR"] = TEX_ADDR
    gpu.stats["mip_levels_used"] = {}
    step = to_fx(4 / 4)                                           # 1 pixel = 4 texels
    sample(2 | 3 << 4 | 1 << 11, [0, step, 0, step], [0, 0, step, step], wave_type=1)
    results.append(("mipmaps: 4 texels per pixel picks level 2",
                    gpu.stats["mip_levels_used"] == {2: 1}))
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {sum(ok for _, ok in results)} / {len(results)} texture unit tests passed")


def part_c(frames=24):
    print("\n===== Part C: texture unit + animation =====")
    texture_unit_tests()
    mem = Memory()
    build_part_c(mem, frames)
    gpu = GPU(mem)
    gpu.start(CMD_ADDR)
    s = gpu.stats
    pixels = WIDTH * HEIGHT * frames
    print(f"  Frames rendered   : {gpu.frames_done} from ONE command list "
          f"(fence value 0x{mem.load(0x00F000):X} = done)")
    print(f"  Pixel shading     : {s['pixel_waves']} waves x "
          f"{s['ps_instructions'] / max(1, s['pixel_waves']):.0f} instructions "
          f"(each wave shades a 2x2 quad at once)")
    print(f"  Texture unit      : {s['texture_samples']} samples, {s['texel_reads']} texel reads")
    used = s["mip_levels_used"]
    total = sum(used.values())
    print("  Mip levels used   : " + ", ".join(f"level {k}: {100 * v / total:.0f}%"
                                             for k, v in sorted(used.items())
                                             if 100 * v / total >= 0.5))
    gpu.frames[0].save("textured_cube_gpu.png")
    big = [f.resize((WIDTH * 2, HEIGHT * 2), Image.NEAREST) for f in gpu.frames]
    big[0].save("textured_cube_gpu.gif", save_all=True, append_images=big[1:],
                duration=80, loop=0)
    print("  Saved textured_cube_gpu.png and textured_cube_gpu.gif")
    return gpu



if __name__ == "__main__":
    part_a()
    gpu = part_b()
    check_frame1()

    S = 2
    frames = [f.resize((WIDTH * S, HEIGHT * S), Image.NEAREST) for f in gpu.frames]
    depth = gpu.read_depth_image().convert("RGB").resize((WIDTH * S, HEIGHT * S), Image.NEAREST)
    panels = [(frames[0], "Frame 1: UV shader, one cube"),
              (frames[1], "Frame 2: two cubes, depth test"),
              (depth, "Frame 2: depth buffer (near = bright)")]
    sheet = Image.new("RGB", (WIDTH * S * 3 + 20, HEIGHT * S + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (img, title) in enumerate(panels):
        sheet.paste(img, (i * (WIDTH * S + 10), 34))
        draw.text((i * (WIDTH * S + 10) + 6, 8), title, fill=(44, 44, 42), font=font(16))
    sheet.save("part_b_frames.png")
    print("  Saved part_b_frames.png")

    part_c()
