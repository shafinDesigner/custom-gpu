# cube_3d.py
# Step 1, part 3: draw a 3D cube.
# Each section is named after the GPU block that will do this job in hardware.

import math
from PIL import Image

WIDTH = 320
HEIGHT = 240
BACKGROUND = (30, 30, 30)
FOCAL = 300          # "zoom" of the camera
DISTANCE = 4.5       # how far the cube is from the camera

# =====================================================================
# INPUT DATA (what the Input assembler would fetch from memory)
# =====================================================================
# 8 corners of a cube, from -1 to +1 on each axis (x right, y up, z away)
VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),   # 0-3: near face
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),   # 4-7: far face
]

# 12 triangles (2 per face). Each is 3 corner numbers + a face color.
# Corners are listed counter-clockwise when seen from OUTSIDE the cube.
TRIANGLES = [
    (0, 1, 2, (229, 75, 74)),  (0, 2, 3, (229, 75, 74)),    # near   - red
    (5, 4, 7, (99, 153, 34)),  (5, 7, 6, (99, 153, 34)),    # far    - green
    (1, 5, 6, (55, 138, 221)), (1, 6, 2, (55, 138, 221)),   # right  - blue
    (4, 0, 3, (239, 159, 39)), (4, 3, 7, (239, 159, 39)),   # left   - orange
    (3, 2, 6, (127, 119, 221)), (3, 6, 7, (127, 119, 221)), # top    - purple
    (4, 5, 1, (29, 158, 117)), (4, 1, 0, (29, 158, 117)),   # bottom - teal
]


# =====================================================================
# VERTEX SHADER (runs on your shader cores)
# Rotates each 3D point and moves it in front of the camera.
# =====================================================================
def vertex_shader(v, angle_y, angle_x):
    x, y, z = v
    # Rotate around the vertical (Y) axis
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    x, z = x * cy + z * sy, -x * sy + z * cy
    # Rotate around the horizontal (X) axis (tilt)
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    y, z = y * cx - z * sx, y * sx + z * cx
    # Move the cube away from the camera
    z += DISTANCE
    return (x, y, z)


# =====================================================================
# PRIMITIVE UNIT: viewport transform (3D -> 2D screen pixels)
# Far things look smaller: divide by z. This is "perspective".
# =====================================================================
def project(v):
    x, y, z = v
    screen_x = WIDTH / 2 + x * FOCAL / z
    screen_y = HEIGHT / 2 - y * FOCAL / z    # minus: screen y goes down
    return (screen_x, screen_y)


def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# =====================================================================
# RASTERIZER + RENDER OUTPUT UNIT
# Same edge-function method as parts 1 and 2.
# =====================================================================
def draw_triangle(image, a, b, c, color):
    area = edge(a, b, c)

    # PRIMITIVE UNIT: back-face culling.
    # Triangles facing away from the camera have the opposite sign -> skip them.
    if area >= 0:
        return

    # Bounding box: only check pixels near the triangle (much faster)
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
            if w_a >= 0 and w_b >= 0 and w_c >= 0:
                image.putpixel((x, y), color)   # RENDER OUTPUT UNIT writes the pixel


# =====================================================================
# ONE FULL FRAME = the whole pipeline
# =====================================================================
def render_frame(angle_y, angle_x):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    # Vertex shader + projection for all 8 corners
    screen = [project(vertex_shader(v, angle_y, angle_x)) for v in VERTICES]
    # Primitive assembly + rasterization for all 12 triangles
    for i0, i1, i2, color in TRIANGLES:
        draw_triangle(image, screen[i0], screen[i1], screen[i2], color)
    return image


# ---- Still image ----
still = render_frame(math.radians(35), math.radians(-25))
still.save("cube.png")

# ---- Spinning animation (36 frames) ----
frames = [render_frame(math.radians(i * 10), math.radians(-25)) for i in range(36)]
frames[0].save("cube.gif", save_all=True, append_images=frames[1:],
               duration=60, loop=0)

print("Done! Saved cube.png and cube.gif")

# Show in Colab / Jupyter
try:
    from IPython.display import Image as ShowImage
    display(ShowImage("cube.gif"))
except Exception:
    pass
