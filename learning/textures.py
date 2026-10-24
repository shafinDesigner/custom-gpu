# textures.py
# Step 1, part 5: TEXTURES - put a picture onto 3D surfaces.
# Builds on depth_buffer.py. New parts are marked with  *** NEW ***

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


# =====================================================================
# *** NEW *** THE TEXTURE: a small picture stored in memory
# =====================================================================
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
TEXELS = TEXTURE.load()            # fast access to texture pixels ("texels")


# *** NEW *** TEXTURE UNIT: read one texel at position (u, v)
# u and v go from 0 to 1 across the texture. Bigger values wrap around (repeat).
def sample_texture(u, v):
    tx = int(math.floor(u * TEX_SIZE)) % TEX_SIZE
    ty = int(math.floor(v * TEX_SIZE)) % TEX_SIZE
    return TEXELS[tx, ty]


# =====================================================================
# Pipeline helpers (same ideas as before)
# =====================================================================
def project(v):
    x, y, z = v
    inv_z = 1.0 / z
    return (WIDTH / 2 + x * FOCAL * inv_z, HEIGHT / 2 - y * FOCAL * inv_z, inv_z)


def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# =====================================================================
# RASTERIZER + PIXEL SHADER + RENDER OUTPUT UNIT
# Each corner now has: screen x, screen y, 1/z, and texture position (u, v)
# =====================================================================
def draw_triangle(pixels, depth, a, b, c, uv_a, uv_b, uv_c,
                  brightness=1.0, perspective_correct=True):
    area = edge(a, b, c)
    if area >= 0:
        return

    # *** NEW *** For perspective correction, prepare u/z and v/z at each corner
    uz = (uv_a[0] * a[2], uv_b[0] * b[2], uv_c[0] * c[2])
    vz = (uv_a[1] * a[2], uv_b[1] * b[2], uv_c[1] * c[2])

    min_x = max(int(min(a[0], b[0], c[0])), 0)
    max_x = min(int(max(a[0], b[0], c[0])) + 1, WIDTH)
    min_y = max(int(min(a[1], b[1], c[1])), 0)
    max_y = min(int(max(a[1], b[1], c[1])) + 1, HEIGHT)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            p = (x + 0.5, y + 0.5)
            w_a = edge(b, c, p) / area
            w_b = edge(c, a, p) / area
            w_c = edge(a, b, p) / area
            if w_a < 0 or w_b < 0 or w_c < 0:
                continue

            closeness = w_a * a[2] + w_b * b[2] + w_c * c[2]
            if closeness <= depth[y][x]:
                continue
            depth[y][x] = closeness

            # *** NEW *** Find the texture position for this pixel
            if perspective_correct:
                # Blend u/z and v/z, then divide by blended 1/z  (needs DIVISION)
                u = (w_a * uz[0] + w_b * uz[1] + w_c * uz[2]) / closeness
                v = (w_a * vz[0] + w_b * vz[1] + w_c * vz[2]) / closeness
            else:
                # Simple blend of u and v (fast but WRONG in 3D: warps)
                u = w_a * uv_a[0] + w_b * uv_b[0] + w_c * uv_c[0]
                v = w_a * uv_a[1] + w_b * uv_b[1] + w_c * uv_c[1]

            # *** NEW *** PIXEL SHADER: read texture, apply simple lighting
            r, g, bl = sample_texture(u, v)
            pixels[x, y] = (int(r * brightness), int(g * brightness), int(bl * brightness))


# =====================================================================
# SCENE 1: a long textured floor (best for seeing the warping problem)
# =====================================================================
def render_floor(perspective_correct):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    pixels = image.load()
    depth = [[0.0] * WIDTH for _ in range(HEIGHT)]
    # Floor corners (x, y, z): near-left, near-right, far-right, far-left
    NL, NR, FR, FL = (-2.5, -1, 1.5), (2.5, -1, 1.5), (2.5, -1, 9), (-2.5, -1, 9)
    # Texture repeats 3 times across and 5 times into the distance
    uv = {NL: (0, 5), NR: (3, 5), FR: (3, 0), FL: (0, 0)}
    for t in [(NL, NR, FR), (NL, FR, FL)]:
        a, b, c = (project(v) for v in t)
        draw_triangle(pixels, depth, a, b, c, uv[t[0]], uv[t[1]], uv[t[2]],
                      perspective_correct=perspective_correct)
    return image


# =====================================================================
# SCENE 2: a spinning textured cube
# =====================================================================
CUBE_VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
# Each face as 4 corners, counter-clockwise from outside:
# bottom-left, bottom-right, top-right, top-left (as seen on that face)
CUBE_FACES = [(0, 1, 2, 3), (5, 4, 7, 6), (1, 5, 6, 2),
              (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)]
FACE_UV = [(0, 1), (1, 1), (1, 0), (0, 0)]      # matching texture corners
FACE_BRIGHTNESS = [1.0, 0.75, 0.85, 0.7, 0.95, 0.6]   # simple flat lighting
TILT = math.radians(-25)


def cube_vertex_shader(v, angle):
    x, y, z = v
    c, s = math.cos(angle), math.sin(angle)
    x, z = x * c + z * s, -x * s + z * c
    c, s = math.cos(TILT), math.sin(TILT)
    y, z = y * c - z * s, y * s + z * c
    return (x, y, z + 4.5)


def render_cube(angle):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    pixels = image.load()
    depth = [[0.0] * WIDTH for _ in range(HEIGHT)]
    screen = [project(cube_vertex_shader(v, angle)) for v in CUBE_VERTICES]
    for face, quad in enumerate(CUBE_FACES):
        q = [screen[i] for i in quad]
        for i0, i1, i2 in [(0, 1, 2), (0, 2, 3)]:      # a square = 2 triangles
            draw_triangle(pixels, depth, q[i0], q[i1], q[i2],
                          FACE_UV[i0], FACE_UV[i1], FACE_UV[i2],
                          brightness=FACE_BRIGHTNESS[face])
    return image


# ===================== Outputs =====================
TEXTURE.save("texture.png")

wrong = render_floor(perspective_correct=False)
right = render_floor(perspective_correct=True)
compare = Image.new("RGB", (WIDTH * 2 + 10, HEIGHT + 34), (255, 255, 255))
compare.paste(wrong, (0, 34))
compare.paste(right, (WIDTH + 10, 34))
draw = ImageDraw.Draw(compare)
draw.text((10, 8), "Simple blend (warped)", fill=(163, 45, 45), font=font(16))
draw.text((WIDTH + 20, 8), "Perspective-correct", fill=(15, 110, 86), font=font(16))
compare.save("texture_compare.png")

render_cube(math.radians(35)).save("textured_cube.png")
frames = [render_cube(math.radians(i * 10)) for i in range(36)]
frames[0].save("textured_cube.gif", save_all=True, append_images=frames[1:],
               duration=70, loop=0)

print("Done! Saved texture.png, texture_compare.png, textured_cube.png, textured_cube.gif")

try:
    from IPython.display import Image as ShowImage
    display(compare)
    display(ShowImage("textured_cube.gif"))
except Exception:
    pass
