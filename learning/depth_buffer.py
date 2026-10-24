# depth_buffer.py
# Step 1, part 4: two cubes that overlap correctly using a DEPTH BUFFER.
# Builds on cube_3d.py. New parts are marked with  *** NEW ***

import math
from PIL import Image, ImageDraw, ImageFont

WIDTH = 320
HEIGHT = 240
BACKGROUND = (30, 30, 30)
FOCAL = 300
CAMERA_DISTANCE = 6
TILT = math.radians(-20)

# ---- Cube model (same as part 3) ----
CUBE_VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
# (corner, corner, corner, face number)  face: 0 near, 1 far, 2 right, 3 left, 4 top, 5 bottom
CUBE_TRIANGLES = [
    (0, 1, 2, 0), (0, 2, 3, 0),
    (5, 4, 7, 1), (5, 7, 6, 1),
    (1, 5, 6, 2), (1, 6, 2, 2),
    (4, 0, 3, 3), (4, 3, 7, 3),
    (3, 2, 6, 4), (3, 6, 7, 4),
    (4, 5, 1, 5), (4, 1, 0, 5),
]

BIG_COLORS = [(229, 75, 74), (99, 153, 34), (55, 138, 221),
              (239, 159, 39), (127, 119, 221), (29, 158, 117)]
SMALL_COLORS = [(250, 199, 117), (250, 199, 117), (239, 159, 39),
                (239, 159, 39), (255, 230, 170), (255, 230, 170)]


# ---- The scene: a big spinning cube + a small cube orbiting around it ----
def scene(t):
    big = {"scale": 1.0, "spin": t * 0.5, "position": (0, 0, 0), "colors": BIG_COLORS}
    small = {"scale": 0.55, "spin": t * 2.0,
             "position": (1.7 * math.cos(t), 0.2, 1.7 * math.sin(t)),
             "colors": SMALL_COLORS}
    return [big, small]


# ---- VERTEX SHADER: scale, spin, place in the world, tilt camera ----
def vertex_shader(v, obj):
    s = obj["scale"]
    x, y, z = v[0] * s, v[1] * s, v[2] * s
    c, si = math.cos(obj["spin"]), math.sin(obj["spin"])
    x, z = x * c + z * si, -x * si + z * c
    px, py, pz = obj["position"]
    x, y, z = x + px, y + py, z + pz
    c, si = math.cos(TILT), math.sin(TILT)
    y, z = y * c - z * si, y * si + z * c
    z += CAMERA_DISTANCE
    return (x, y, z)


# ---- PRIMITIVE UNIT: project to the screen ----
def project(v):
    x, y, z = v
    inv_z = 1.0 / z                       # *** NEW *** keep 1/z for the depth test
    return (WIDTH / 2 + x * FOCAL * inv_z,
            HEIGHT / 2 - y * FOCAL * inv_z,
            inv_z)


def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# ---- RASTERIZER + RENDER OUTPUT UNIT (with depth test) ----
def draw_triangle(image, depth, a, b, c, color, use_depth):
    area = edge(a, b, c)
    if area >= 0:                         # back-face culling
        return

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

            # *** NEW *** How close is this pixel? Blend 1/z with the weights.
            # Bigger 1/z = closer to the camera.
            closeness = w_a * a[2] + w_b * b[2] + w_c * c[2]

            # *** NEW *** DEPTH TEST: is something closer already drawn here?
            if use_depth:
                if closeness <= depth[y][x]:
                    continue                  # hidden -> skip this pixel
                depth[y][x] = closeness       # remember the new closest value

            image.putpixel((x, y), color)


# ---- One full frame ----
def render_frame(t, use_depth=True):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    # *** NEW *** depth buffer: one number per pixel, 0 = "nothing here yet"
    depth = [[0.0] * WIDTH for _ in range(HEIGHT)]
    for obj in scene(t):
        screen = [project(vertex_shader(v, obj)) for v in CUBE_VERTICES]
        for i0, i1, i2, face in CUBE_TRIANGLES:
            draw_triangle(image, depth, screen[i0], screen[i1], screen[i2],
                          obj["colors"][face], use_depth)
    return image, depth


# ---- *** NEW *** Turn the depth buffer into a picture (close = white) ----
def depth_to_image(depth):
    values = [v for row in depth for v in row if v > 0]
    low, high = min(values), max(values)
    img = Image.new("L", (WIDTH, HEIGHT), 0)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            v = depth[y][x]
            if v > 0:
                img.putpixel((x, y), int(60 + 195 * (v - low) / (high - low)))
    return img


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ===================== Outputs =====================
T = math.radians(40)   # moment when the small cube is partly BEHIND the big one

wrong, _ = render_frame(T, use_depth=False)
right, depth = render_frame(T, use_depth=True)

# 1. Side-by-side comparison
compare = Image.new("RGB", (WIDTH * 2 + 10, HEIGHT + 34), (255, 255, 255))
compare.paste(wrong, (0, 34))
compare.paste(right, (WIDTH + 10, 34))
draw = ImageDraw.Draw(compare)
draw.text((10, 8), "Without depth buffer (wrong)", fill=(163, 45, 45), font=font(16))
draw.text((WIDTH + 20, 8), "With depth buffer (correct)", fill=(15, 110, 86), font=font(16))
compare.save("depth_compare.png")

# 2. What the depth buffer looks like
depth_to_image(depth).save("depth_view.png")

# 3. Animation with the depth buffer on
frames = [render_frame(math.radians(i * 10))[0] for i in range(36)]
frames[0].save("two_cubes.gif", save_all=True, append_images=frames[1:],
               duration=70, loop=0)

print("Done! Saved depth_compare.png, depth_view.png, two_cubes.gif")

try:
    from IPython.display import Image as ShowImage
    display(compare)
    display(ShowImage("two_cubes.gif"))
except Exception:
    pass
