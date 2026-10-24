# texture_filtering.py
# Step 1, part 6: SMOOTH TEXTURES
#   - Bilinear filtering: fixes blocky textures up close
#   - Mipmaps:            fixes noisy, shimmering textures far away
# Builds on textures.py. New parts are marked with  *** NEW ***

import math
from PIL import Image, ImageDraw, ImageFont

WIDTH = 320
HEIGHT = 240
BACKGROUND = (30, 30, 30)
FOCAL = 300
TEX_SIZE = 128


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---- Texture (same as part 5) ----
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


TEXTURE = make_texture()


# =====================================================================
# *** NEW *** MIPMAPS: smaller and smaller copies of the texture
# Each level is half the size. Each new texel = average of 4 texels above it.
# =====================================================================
def build_mipmaps(tex):
    levels = [tex]
    while levels[-1].width > 1:
        half = levels[-1].width // 2
        levels.append(levels[-1].resize((half, half), Image.BOX))   # BOX = average of 2x2
    return [(img.width, img.load()) for img in levels], levels


MIPS, MIP_IMAGES = build_mipmaps(TEXTURE)     # 128, 64, 32, 16, 8, 4, 2, 1


def fetch(level, tx, ty):
    size, texels = MIPS[level]
    return texels[tx % size, ty % size]       # wrap around (repeat)


# ---- Point sampling (part 5): just take the nearest texel ----
def sample_point(u, v):
    return fetch(0, int(math.floor(u * TEX_SIZE)), int(math.floor(v * TEX_SIZE)))


# =====================================================================
# *** NEW *** BILINEAR FILTERING: blend the 4 nearest texels
# =====================================================================
def lerp(c1, c2, t):
    """Blend two colors: t=0 gives c1, t=1 gives c2."""
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def sample_bilinear(u, v, level=0):
    size = MIPS[level][0]
    x = u * size - 0.5                  # -0.5: measure from texel CENTERS
    y = v * size - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0             # how far between texels (0 to 1)

    top    = lerp(fetch(level, x0, y0),     fetch(level, x0 + 1, y0),     fx)
    bottom = lerp(fetch(level, x0, y0 + 1), fetch(level, x0 + 1, y0 + 1), fx)
    return lerp(top, bottom, fy)        # 4 texel reads + 3 blends


# =====================================================================
# *** NEW *** PICK THE MIPMAP LEVEL
# How many texels does one screen pixel cover? 1 -> level 0, 2 -> level 1, 4 -> level 2 ...
# =====================================================================
def pick_mip_level(uv, uv_right, uv_down):
    du_dx = (uv_right[0] - uv[0]) * TEX_SIZE
    dv_dx = (uv_right[1] - uv[1]) * TEX_SIZE
    du_dy = (uv_down[0] - uv[0]) * TEX_SIZE
    dv_dy = (uv_down[1] - uv[1]) * TEX_SIZE
    footprint = max(math.hypot(du_dx, dv_dx), math.hypot(du_dy, dv_dy))
    if footprint <= 1:
        return 0
    return min(int(round(math.log2(footprint))), len(MIPS) - 1)


# ---- Pipeline helpers ----
def project(v):
    x, y, z = v
    inv_z = 1.0 / z
    return (WIDTH / 2 + x * FOCAL * inv_z, HEIGHT / 2 - y * FOCAL * inv_z, inv_z)


def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# =====================================================================
# RASTERIZER + PIXEL SHADER, with a choice of texture filter
# mode = "point", "bilinear", or "mipmap" (bilinear + mipmaps)
# =====================================================================
def draw_triangle(pixels, depth, a, b, c, uv_a, uv_b, uv_c, mode):
    area = edge(a, b, c)
    if area >= 0:
        return
    uz = (uv_a[0] * a[2], uv_b[0] * b[2], uv_c[0] * c[2])
    vz = (uv_a[1] * a[2], uv_b[1] * b[2], uv_c[1] * c[2])

    def uv_at(px, py):
        """Perspective-correct (u, v) at any screen point (same math as part 5)."""
        w_a = edge(b, c, (px, py)) / area
        w_b = edge(c, a, (px, py)) / area
        w_c = edge(a, b, (px, py)) / area
        closeness = w_a * a[2] + w_b * b[2] + w_c * c[2]
        u = (w_a * uz[0] + w_b * uz[1] + w_c * uz[2]) / closeness
        v = (w_a * vz[0] + w_b * vz[1] + w_c * vz[2]) / closeness
        return (u, v), (w_a, w_b, w_c), closeness

    min_x = max(int(min(a[0], b[0], c[0])), 0)
    max_x = min(int(max(a[0], b[0], c[0])) + 1, WIDTH)
    min_y = max(int(min(a[1], b[1], c[1])), 0)
    max_y = min(int(max(a[1], b[1], c[1])) + 1, HEIGHT)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            uv, w, closeness = uv_at(x + 0.5, y + 0.5)
            if w[0] < 0 or w[1] < 0 or w[2] < 0:
                continue
            if closeness <= depth[y][x]:
                continue
            depth[y][x] = closeness

            if mode == "point":
                color = sample_point(*uv)
            elif mode == "bilinear":
                color = sample_bilinear(*uv)
            else:
                # *** NEW *** look at the neighbor pixels (in hardware: the 2x2 quad)
                uv_right, _, _ = uv_at(x + 1.5, y + 0.5)
                uv_down, _, _ = uv_at(x + 0.5, y + 1.5)
                level = pick_mip_level(uv, uv_right, uv_down)
                color = sample_bilinear(*uv, level=level)

            pixels[x, y] = tuple(int(ch) for ch in color)


# =====================================================================
# SCENES
# =====================================================================
def render(corners, uvs, mode):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    pixels = image.load()
    depth = [[0.0] * WIDTH for _ in range(HEIGHT)]
    screen = [project(v) for v in corners]
    for i0, i1, i2 in [(0, 1, 2), (0, 2, 3)]:
        draw_triangle(pixels, depth, screen[i0], screen[i1], screen[i2],
                      uvs[i0], uvs[i1], uvs[i2], mode)
    return image


def look_down(v, angle=math.radians(-15)):
    """Tilt the camera down a little so the floor fills more of the picture."""
    x, y, z = v
    c, s = math.cos(angle), math.sin(angle)
    return (x, y * c - z * s, y * s + z * c)


def floor_scene(mode, scroll=0.0):
    """A long floor. The texture repeats 6 x 14 times. scroll = move forward."""
    corners = [look_down(v) for v in
               [(-3, -1, 1.2), (3, -1, 1.2), (3, -1, 14), (-3, -1, 14)]]
    uvs = [(0, 14 + scroll), (6, 14 + scroll), (6, scroll), (0, scroll)]
    return render(corners, uvs, mode)


def closeup_scene(mode):
    """A wall right in front of the camera, showing only a small part of the texture."""
    corners = [(-1.4, -1.05, 2), (1.4, -1.05, 2), (1.4, 1.05, 2), (-1.4, 1.05, 2)]
    uvs = [(0.30, 0.72), (0.70, 0.72), (0.70, 0.42), (0.30, 0.42)]
    return render(corners, uvs, mode)


def side_by_side(images, titles):
    canvas = Image.new("RGB", (WIDTH * len(images) + 10 * (len(images) - 1), HEIGHT + 34),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (img, title) in enumerate(zip(images, titles)):
        canvas.paste(img, (i * (WIDTH + 10), 34))
        draw.text((i * (WIDTH + 10) + 10, 8), title, fill=(44, 44, 42), font=font(16))
    return canvas


# ===================== Outputs =====================
# 1. The mipmap chain
strip = Image.new("RGB", (sum(m.width for m in MIP_IMAGES[:6]) + 10 * 5, TEX_SIZE), (255, 255, 255))
x = 0
for m in MIP_IMAGES[:6]:
    strip.paste(m, (x, TEX_SIZE - m.height))
    x += m.width + 10
strip.save("mipmaps.png")

# 2. Up close: point vs bilinear
side_by_side([closeup_scene("point"), closeup_scene("bilinear")],
             ["Point sampling (blocky)", "Bilinear filtering (smooth)"]).save("closeup_compare.png")

# 3. Far away: point vs bilinear vs mipmaps
side_by_side([floor_scene("point"), floor_scene("bilinear"), floor_scene("mipmap")],
             ["Point sampling", "Bilinear only", "Bilinear + mipmaps"]).save("far_compare.png")

# 4. Moving floor: shimmering vs smooth (takes ~30 seconds)
frames = [side_by_side([floor_scene("point", i / 16), floor_scene("mipmap", i / 16)],
                       ["Point sampling", "Bilinear + mipmaps"]) for i in range(16)]
frames[0].save("moving_floor.gif", save_all=True, append_images=frames[1:],
               duration=80, loop=0)

print("Done! Saved mipmaps.png, closeup_compare.png, far_compare.png, moving_floor.gif")

try:
    from IPython.display import Image as ShowImage
    display(Image.open("closeup_compare.png"))
    display(Image.open("far_compare.png"))
    display(ShowImage("moving_floor.gif"))
except Exception:
    pass
