import os
import cv2
import numpy as np
import random
import json

OUT_DIR = "data"
IM_DIR = os.path.join(OUT_DIR, "images")
ANN_DIR = os.path.join(OUT_DIR, "annotations")
NUM_IMAGES = 200
SIZE = 512

os.makedirs(IM_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

def clamp(v, a, b):
    return max(a, min(b, v))

for i in range(NUM_IMAGES):
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)  # black background

    # draw grid lines (white)
    spacing = random.randint(24, 40)
    thickness = random.randint(2, 4)
    for x in range(0, SIZE, spacing):
        cv2.line(img, (x, 0), (x, SIZE), 255, thickness)
    for y in range(0, SIZE, spacing):
        cv2.line(img, (0, y), (SIZE, y), 255, thickness)

    holes = []
    num_holes = random.randint(1, 6)
    for _ in range(num_holes):
        r = random.randint(10, 40)
        cx = random.randint(r + 4, SIZE - r - 4)
        cy = random.randint(r + 4, SIZE - r - 4)
        # draw hole as filled black circle (same as background)
        cv2.circle(img, (cx, cy), r, 0, -1)
        x1 = clamp(cx - r - 2, 0, SIZE - 1)
        y1 = clamp(cy - r - 2, 0, SIZE - 1)
        x2 = clamp(cx + r + 2, 0, SIZE - 1)
        y2 = clamp(cy + r + 2, 0, SIZE - 1)
        holes.append([x1, y1, x2 - x1, y2 - y1])

    # slight blur to make it less synthetic
    img = cv2.GaussianBlur(img, (3, 3), 0)

    fname = f"net_{i:04d}.png"
    cv2.imwrite(os.path.join(IM_DIR, fname), img)

    ann = {"image": fname, "width": SIZE, "height": SIZE, "boxes": holes}
    with open(os.path.join(ANN_DIR, f"net_{i:04d}.json"), "w") as f:
        json.dump(ann, f)

    if (i + 1) % 20 == 0:
        print(f"Generated {i+1}/{NUM_IMAGES}")

print("Dataset generation complete.")
