# triangle.py
# Step 1, part 1: draw one filled triangle using edge functions.
# This is the same method your GPU's rasterizer will use in hardware.

from PIL import Image

# ---- 1. Image size (in pixels) ----
WIDTH = 320
HEIGHT = 200

# ---- 2. Colors as (Red, Green, Blue), each from 0 to 255 ----
BACKGROUND = (30, 30, 30)          # dark gray
TRIANGLE_COLOR = (127, 119, 221)   # purple

# ---- 3. The three corners of the triangle, as (x, y) ----
# x goes right, y goes DOWN (that's how images work)
A = (40, 20)
B = (290, 70)
C = (110, 180)


# ---- 4. Edge function ----
def edge(p1, p2, p):
    """Tells which side of the line p1 -> p2 the point p is on.
    Positive = one side, negative = other side, zero = exactly on the line."""
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# ---- 5. Inside test ----
def inside_triangle(p):
    """A point is inside if it is on the same side of all 3 edges."""
    e1 = edge(A, B, p)
    e2 = edge(B, C, p)
    e3 = edge(C, A, p)
    all_positive = e1 >= 0 and e2 >= 0 and e3 >= 0
    all_negative = e1 <= 0 and e2 <= 0 and e3 <= 0
    return all_positive or all_negative


# ---- 6. Make a blank image ----
image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)

# ---- 7. Check every pixel ----
colored = 0
for y in range(HEIGHT):
    for x in range(WIDTH):
        center = (x + 0.5, y + 0.5)        # test the middle of the pixel
        if inside_triangle(center):
            image.putpixel((x, y), TRIANGLE_COLOR)
            colored += 1

# ---- 8. Save the picture ----
image.save("triangle.png")
print(f"Done! Checked {WIDTH * HEIGHT} pixels, colored {colored}.")
