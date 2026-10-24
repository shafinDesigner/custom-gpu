# fixed_point.py
# Step 1, part 7: FIXED-POINT MATH - the whole pipeline with only whole numbers (integers),
# the way your GPU hardware will calculate. Then compare with the normal float version.
#
# Both versions use the SAME steps. Only the number type is different:
#   FLOAT version : Python decimals (like 0.375)
#   FIXED version : integers only, with an agreed position for the "decimal point"

import math
from PIL import Image, ImageDraw, ImageFont, ImageChops

WIDTH = 320
HEIGHT = 240
BACKGROUND = (30, 30, 30)
FOCAL = 300
DISTANCE = 4.5
TEX_SIZE = 128

ANGLE_STEPS = 1024          # a full circle = 1024 steps (hardware-friendly)
SPIN = 100                  # cube rotation, in steps  (about 35 degrees)
TILT = -71                  # camera tilt, in steps    (about -25 degrees)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---- Texture + mipmaps (same as part 6) ----
def make_texture():
    tex = Image.new("RGB", (TEX_SIZE, TEX_SIZE))
    draw = ImageDraw.Draw(tex)
    check = TEX_SIZE // 8
    for row in range(8):
        for col in range(8):
            color = (238, 237, 254) if (row + col) % 2 == 0 else (127, 119, 221)
            draw.rectangle([col * check, row * check,
                            (col + 1) * check - 1, (row + 1) * check - 1], fill=color)
    draw.rectangle([0, 0, TEX_SIZE - 1, TEX_SIZE - 1], outline=(229, 75, 74), width=4)
    draw.rectangle([22, 44, 106, 84], fill=(60, 52, 137))
    draw.text((TEX_SIZE // 2, TEX_SIZE // 2), "GPU", fill=(255, 255, 255),
              font=font(32), anchor="mm")
    return tex


def build_mipmaps(tex):
    levels = [tex]
    while levels[-1].width > 1:
        half = levels[-1].width // 2
        levels.append(levels[-1].resize((half, half), Image.BOX))
    return [(img.width, img.load()) for img in levels]


MIPS = build_mipmaps(make_texture())


def fetch(level, tx, ty):
    size, texels = MIPS[level]
    return texels[tx % size, ty % size]


# ---- Cube (same as part 5) ----
CUBE_VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
CUBE_FACES = [(0, 1, 2, 3), (5, 4, 7, 6), (1, 5, 6, 2),
              (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)]
FACE_UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
FACE_BRIGHTNESS = [1.0, 0.75, 0.85, 0.7, 0.95, 0.6]


# #####################################################################
#                       FLOAT VERSION (reference)
# #####################################################################
def f_vertex_shader(v):
    x, y, z = v
    a = SPIN * 2 * math.pi / ANGLE_STEPS
    x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
    t = TILT * 2 * math.pi / ANGLE_STEPS
    y, z = y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t)
    return (x, y, z + DISTANCE)


def f_project(v):
    x, y, z = v
    inv_z = 1.0 / z
    return (WIDTH / 2 + x * FOCAL * inv_z, HEIGHT / 2 - y * FOCAL * inv_z, inv_z)


def f_edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def f_sample(u, v, level):
    size = MIPS[level][0]
    x, y = u * size - 0.5, v * size - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    lerp = lambda c1, c2, t: tuple(a + (b - a) * t for a, b in zip(c1, c2))
    top = lerp(fetch(level, x0, y0), fetch(level, x0 + 1, y0), fx)
    bottom = lerp(fetch(level, x0, y0 + 1), fetch(level, x0 + 1, y0 + 1), fx)
    return lerp(top, bottom, fy)


def f_mip_level(uv, uv_r, uv_d):
    footprint = max(abs(uv_r[0] - uv[0]), abs(uv_r[1] - uv[1]),
                    abs(uv_d[0] - uv[0]), abs(uv_d[1] - uv[1])) * TEX_SIZE
    level = int(footprint).bit_length() - 1 if footprint >= 1 else 0
    return min(level, len(MIPS) - 1)


def f_draw_triangle(pixels, depth, a, b, c, uvs, brightness):
    area = f_edge(a, b, c)
    if area >= 0:
        return
    uz = [uvs[i][0] * p[2] for i, p in enumerate((a, b, c))]
    vz = [uvs[i][1] * p[2] for i, p in enumerate((a, b, c))]

    def uv_at(px, py):
        w = (f_edge(b, c, (px, py)) / area, f_edge(c, a, (px, py)) / area,
             f_edge(a, b, (px, py)) / area)
        close = w[0] * a[2] + w[1] * b[2] + w[2] * c[2]
        if close <= 0:
            close = 1e-6
        u = (w[0] * uz[0] + w[1] * uz[1] + w[2] * uz[2]) / close
        v = (w[0] * vz[0] + w[1] * vz[1] + w[2] * vz[2]) / close
        return w, close, (u, v)

    for y in range(max(int(min(a[1], b[1], c[1])), 0), min(int(max(a[1], b[1], c[1])) + 1, HEIGHT)):
        for x in range(max(int(min(a[0], b[0], c[0])), 0), min(int(max(a[0], b[0], c[0])) + 1, WIDTH)):
            w, close, uv = uv_at(x + 0.5, y + 0.5)
            if min(w) < 0 or close <= depth[y][x]:
                continue
            depth[y][x] = close
            level = f_mip_level(uv, uv_at(x + 1.5, y + 0.5)[2], uv_at(x + 0.5, y + 1.5)[2])
            color = f_sample(uv[0], uv[1], level)
            pixels[x, y] = tuple(int(ch * brightness) for ch in color)


# #####################################################################
#                *** NEW ***  FIXED-POINT VERSION (hardware)
# #####################################################################
# Format "16.16": a 32-bit integer where the lower 16 bits are the fraction.
#   1.0  ->  65536        0.5 -> 32768        2.25 -> 147456
FRAC = 16
ONE = 1 << FRAC                     # 65536 means 1.0


def to_fx(number):
    """Convert a normal number to fixed-point (done once, e.g. by the driver)."""
    return int(round(number * ONE))


def fx_mul(a, b):
    """Multiply two fixed-point numbers: multiply, then shift away 16 extra fraction bits."""
    return (a * b) >> FRAC


def fx_recip(a):
    """RECIPROCAL UNIT: 1 / a in fixed-point. (Hardware will do this with a small
    iterative circuit; here Python's integer division gives the exact answer.)"""
    return (ONE * ONE) // a


# sin/cos from a LOOKUP TABLE: hardware reads a small ROM instead of calculating
SIN_TABLE = [to_fx(math.sin(2 * math.pi * i / ANGLE_STEPS)) for i in range(ANGLE_STEPS)]


def fx_sin(step):
    return SIN_TABLE[step % ANGLE_STEPS]


def fx_cos(step):
    return SIN_TABLE[(step + ANGLE_STEPS // 4) % ANGLE_STEPS]


SUB = 4                             # screen positions keep 4 fraction bits ("subpixels")


def x_vertex_shader(v):
    x, y, z = (to_fx(c) for c in v)
    c, s = fx_cos(SPIN), fx_sin(SPIN)
    x, z = fx_mul(x, c) + fx_mul(z, s), -fx_mul(x, s) + fx_mul(z, c)
    c, s = fx_cos(TILT), fx_sin(TILT)
    y, z = fx_mul(y, c) - fx_mul(z, s), fx_mul(y, s) + fx_mul(z, c)
    return (x, y, z + to_fx(DISTANCE))


def x_project(v):
    x, y, z = v
    inv_z = fx_recip(z)                                   # 16.16
    sx = to_fx(WIDTH / 2) + fx_mul(x * FOCAL, inv_z)      # 16.16 screen x
    sy = to_fx(HEIGHT / 2) - fx_mul(y * FOCAL, inv_z)
    return (sx >> (FRAC - SUB), sy >> (FRAC - SUB), inv_z)   # keep 4 subpixel bits


def x_edge(p1, p2, p):
    # Same formula. Inputs are integers -> result is an exact integer. No rounding at all.
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def x_lerp8(c1, c2, f):
    """Blend two 8-bit colors with an 8-bit fraction f (0..255): pure integer math."""
    return tuple((a * (256 - f) + b * f) >> 8 for a, b in zip(c1, c2))


def x_sample(u, v, level):
    size = MIPS[level][0]
    tx = u * size - ONE // 2                  # 16.16 texel position, measured from centers
    ty = v * size - ONE // 2
    x0, y0 = tx >> FRAC, ty >> FRAC           # whole texel
    fx, fy = (tx >> 8) & 255, (ty >> 8) & 255  # 8-bit fraction between texels
    top = x_lerp8(fetch(level, x0, y0), fetch(level, x0 + 1, y0), fx)
    bottom = x_lerp8(fetch(level, x0, y0 + 1), fetch(level, x0 + 1, y0 + 1), fx)
    return x_lerp8(top, bottom, fy)


def x_mip_level(uv, uv_r, uv_d):
    footprint = max(abs(uv_r[0] - uv[0]), abs(uv_r[1] - uv[1]),
                    abs(uv_d[0] - uv[0]), abs(uv_d[1] - uv[1])) * TEX_SIZE
    texels = footprint >> FRAC
    level = texels.bit_length() - 1 if texels >= 1 else 0   # log2 = count the bits!
    return min(level, len(MIPS) - 1)


def x_draw_triangle(pixels, depth, a, b, c, uvs, brightness):
    area = x_edge(a, b, c)
    if area >= 0:
        return
    # TRIANGLE SETUP: one reciprocal per triangle (extra bits for precision),
    # so each pixel only needs a MULTIPLY instead of a divide.
    inv_area = (1 << 48) // (-area)
    uvs_fx = [(to_fx(u), to_fx(v)) for u, v in uvs]
    uz = [fx_mul(uvs_fx[i][0], p[2]) for i, p in enumerate((a, b, c))]
    vz = [fx_mul(uvs_fx[i][1], p[2]) for i, p in enumerate((a, b, c))]
    bright8 = int(round(brightness * 256))

    def uv_at(px, py):
        e = (-x_edge(b, c, (px, py)), -x_edge(c, a, (px, py)), -x_edge(a, b, (px, py)))
        w = [(ei * inv_area) >> 32 for ei in e]                   # weights in 0.16
        close = (w[0] * a[2] + w[1] * b[2] + w[2] * c[2]) >> FRAC
        if close <= 0:
            close = 1
        r = fx_recip(close)                                       # RECIPROCAL UNIT
        u = fx_mul((w[0] * uz[0] + w[1] * uz[1] + w[2] * uz[2]) >> FRAC, r)
        v = fx_mul((w[0] * vz[0] + w[1] * vz[1] + w[2] * vz[2]) >> FRAC, r)
        return e, close, (u, v)

    step = 1 << SUB                       # one pixel = 16 subpixel units
    half = step // 2                      # pixel center
    min_x = max(min(a[0], b[0], c[0]) >> SUB, 0)
    max_x = min((max(a[0], b[0], c[0]) >> SUB) + 1, WIDTH)
    min_y = max(min(a[1], b[1], c[1]) >> SUB, 0)
    max_y = min((max(a[1], b[1], c[1]) >> SUB) + 1, HEIGHT)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            px, py = x * step + half, y * step + half
            e, close, uv = uv_at(px, py)
            if min(e) < 0:                # inside test on exact integers
                continue
            depth16 = min(close, 0xFFFF)  # 16-bit depth buffer value
            if depth16 <= depth[y][x]:
                continue
            depth[y][x] = depth16
            level = x_mip_level(uv, uv_at(px + step, py)[2], uv_at(px, py + step)[2])
            color = x_sample(uv[0], uv[1], level)
            pixels[x, y] = tuple((ch * bright8) >> 8 for ch in color)


# #####################################################################
#                       Render both and compare
# #####################################################################
def render(vertex_shader, project, draw_triangle):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    pixels = image.load()
    depth = [[0] * WIDTH for _ in range(HEIGHT)]
    screen = [project(vertex_shader(v)) for v in CUBE_VERTICES]
    for face, quad in enumerate(CUBE_FACES):
        q = [screen[i] for i in quad]
        for i0, i1, i2 in [(0, 1, 2), (0, 2, 3)]:
            draw_triangle(pixels, depth, q[i0], q[i1], q[i2],
                          (FACE_UV[i0], FACE_UV[i1], FACE_UV[i2]), FACE_BRIGHTNESS[face])
    return image


float_img = render(f_vertex_shader, f_project, f_draw_triangle)
fixed_img = render(x_vertex_shader, x_project, x_draw_triangle)

# Difference picture, made 10x brighter so small differences become visible
diff = ImageChops.difference(float_img, fixed_img)
diff_view = diff.point(lambda v: min(255, v * 10))

# Statistics
total = WIDTH * HEIGHT
raw = diff.tobytes()                                   # R,G,B,R,G,B,...
diffs = [max(raw[i:i + 3]) for i in range(0, len(raw), 3)]
same = sum(1 for d in diffs if d == 0)
small = sum(1 for d in diffs if 0 < d <= 3)

canvas = Image.new("RGB", (WIDTH * 3 + 20, HEIGHT + 34), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for i, (img, title) in enumerate([(float_img, "Float (decimals)"),
                                  (fixed_img, "Fixed-point (integers only)"),
                                  (diff_view, "Difference x10")]):
    canvas.paste(img, (i * (WIDTH + 10), 34))
    draw.text((i * (WIDTH + 10) + 10, 8), title, fill=(44, 44, 42), font=font(16))
canvas.save("fixed_vs_float.png")
fixed_img.save("cube_fixed_point.png")

print("Done! Saved fixed_vs_float.png and cube_fixed_point.png")
print(f"Exactly the same : {same / total:6.2%} of pixels")
print(f"Differ by 1-3    : {small / total:6.2%}  (out of 255, invisible to the eye)")
print(f"Differ by more   : {(total - same - small) / total:6.2%}")
print(f"Biggest difference: {max(diffs)} out of 255")

try:
    display(canvas)
except NameError:
    pass
