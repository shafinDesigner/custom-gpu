# triangle_color.py
# Step 1, part 2: each corner has its own color, blended smoothly.
# The blending uses the SAME edge function as part 1.

from PIL import Image

# ---- 1. Image size and background ----
WIDTH = 320
HEIGHT = 200
BACKGROUND = (30, 30, 30)

# ---- 2. Corners, each with its own color (R, G, B) ----
A = (40, 20)
B = (290, 70)
C = (110, 180)

COLOR_A = (255, 0, 0)    # red
COLOR_B = (0, 255, 0)    # green
COLOR_C = (0, 0, 255)    # blue


# ---- 3. Edge function (same as part 1) ----
def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# ---- 4. Whole-triangle value (used to turn edge results into weights) ----
area = edge(A, B, C)

# ---- 5. Blank image ----
image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)

# ---- 6. Check every pixel ----
for y in range(HEIGHT):
    for x in range(WIDTH):
        p = (x + 0.5, y + 0.5)

        # Weight of each corner = edge function of the OPPOSITE side, divided by area.
        # Close to a corner -> that corner's weight is near 1.
        weight_a = edge(B, C, p) / area
        weight_b = edge(C, A, p) / area
        weight_c = edge(A, B, p) / area

        # Inside the triangle only if all weights are 0 or more
        if weight_a >= 0 and weight_b >= 0 and weight_c >= 0:
            # ---- 7. Mix the 3 colors using the weights ----
            red   = weight_a * COLOR_A[0] + weight_b * COLOR_B[0] + weight_c * COLOR_C[0]
            green = weight_a * COLOR_A[1] + weight_b * COLOR_B[1] + weight_c * COLOR_C[1]
            blue  = weight_a * COLOR_A[2] + weight_b * COLOR_B[2] + weight_c * COLOR_C[2]
            image.putpixel((x, y), (int(red), int(green), int(blue)))

# ---- 8. Save (and show it if running in Colab) ----
image.save("triangle_color.png")
print("Done! Saved triangle_color.png")

try:
    display(image)       # works in Colab / Jupyter
except NameError:
    pass                 # normal Python: just open the PNG file
