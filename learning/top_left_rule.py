# top_left_rule.py
# Phase 1, task 3: the TOP-LEFT RULE (spec section 8.8).
# When a pixel center lies EXACTLY on an edge shared by two triangles,
# exactly one of the two triangles must draw it: never both, never neither.
#
# This program draws 8 triangles around one center point (a "fan") and counts
# how many times each pixel is drawn, using three different rules.

from PIL import Image, ImageDraw, ImageFont

SIZE = 33            # image is 33 x 33 pixels
SCALE = 8            # enlarge each pixel 8x so you can see them
SUB = 4              # 12.4 fixed point: 4 subpixel bits (spec section 4)
ONE_PIXEL = 1 << SUB


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def to_sub(x, y):
    """Pixel coordinates -> 12.4 integers (like the primitive unit's output)."""
    return (round(x * ONE_PIXEL), round(y * ONE_PIXEL))


def edge(p1, p2, p):
    """Same edge function as always: exact integers."""
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# =====================================================================
# THE THREE RULES for a pixel center exactly ON an edge (edge value = 0)
# =====================================================================
def is_top_left(p1, p2):
    """*** THE TOP-LEFT RULE ***
    For the edge p1 -> p2 (with the inside on the positive side):
      inside direction = (A, B) = (dy, -dx)
      LEFT edge : the inside is to its RIGHT  ->  A > 0
      TOP edge  : the edge is flat (A == 0) and the inside is BELOW it -> B > 0
    (screen y points down)"""
    A = p2[1] - p1[1]
    B = -(p2[0] - p1[0])
    return A > 0 or (A == 0 and B > 0)


def pixel_inside(values, edges, rule):
    for e, (p1, p2) in zip(values, edges):
        if e > 0:
            continue                       # clearly inside this edge
        if e < 0:
            return False                   # clearly outside
        # e == 0: the pixel center is exactly on this edge
        if rule == "inclusive":            # count it (old reference: >= 0)
            continue
        if rule == "exclusive":            # never count it (> 0)
            return False
        if rule == "top-left" and not is_top_left(p1, p2):
            return False
    return True


def draw_count(triangles, rule):
    """Return how many times each pixel is drawn."""
    count = [[0] * SIZE for _ in range(SIZE)]
    owner = [[-1] * SIZE for _ in range(SIZE)]
    for t, (a, b, c) in enumerate(triangles):
        # Make every triangle wind the same way (inside = positive edge values)
        if edge(a, b, c) > 0:
            b, c = c, b
        edges = [(b, c), (c, a), (a, b)]
        for y in range(SIZE):
            for x in range(SIZE):
                p = (x * ONE_PIXEL + ONE_PIXEL // 2, y * ONE_PIXEL + ONE_PIXEL // 2)
                values = [-edge(p1, p2, p) for p1, p2 in edges]   # positive inside
                if pixel_inside(values, edges, rule):
                    count[y][x] += 1
                    owner[y][x] = t
    return count, owner


# =====================================================================
# THE TEST SCENE: 8 triangles around a center that sits on a pixel center,
# so the shared edges (horizontal, vertical and diagonal) run exactly
# through rows of pixel centers - the hardest case.
# =====================================================================
center = (16.5, 16.5)
ring = [(0, 0), (16.5, 0), (33, 0), (33, 16.5), (33, 33), (16.5, 33), (0, 33), (0, 16.5)]
TRIANGLES = [(to_sub(*center), to_sub(*ring[i]), to_sub(*ring[(i + 1) % 8])) for i in range(8)]

GAP = (226, 75, 74)          # red   : drawn 0 times (a hole!)
DOUBLE = (239, 159, 39)      # amber : drawn 2+ times (overdraw!)
TRI_COLORS = [(127, 119, 221), (175, 169, 236)]   # alternating purples: drawn once


def render(rule):
    count, owner = draw_count(TRIANGLES, rule)
    img = Image.new("RGB", (SIZE, SIZE))
    gaps = doubles = 0
    for y in range(SIZE):
        for x in range(SIZE):
            if count[y][x] == 0:
                img.putpixel((x, y), GAP)
                gaps += 1
            elif count[y][x] > 1:
                img.putpixel((x, y), DOUBLE)
                doubles += 1
            else:
                img.putpixel((x, y), TRI_COLORS[owner[y][x] % 2])
    return img.resize((SIZE * SCALE, SIZE * SCALE), Image.NEAREST), gaps, doubles


panels = [("inclusive", "Old rule: on edge = inside"),
          ("exclusive", "On edge = outside"),
          ("top-left", "Top-left rule")]
W = SIZE * SCALE
canvas = Image.new("RGB", (W * 3 + 40, W + 70), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for i, (rule, title) in enumerate(panels):
    img, gaps, doubles = render(rule)
    x0 = i * (W + 20)
    canvas.paste(img, (x0, 34))
    draw.text((x0, 8), title, fill=(44, 44, 42), font=font(16))
    result = "PERFECT" if gaps == doubles == 0 else f"{doubles} drawn twice, {gaps} holes"
    draw.text((x0, W + 42), result, fill=(15, 110, 86) if result == "PERFECT" else (163, 45, 45),
              font=font(15))
    print(f"{rule:10s}: {doubles:3d} pixels drawn twice, {gaps:3d} holes")

canvas.save("top_left_rule.png")
print("Saved top_left_rule.png")

try:
    display(canvas)
except NameError:
    pass
