import os
import cv2
import numpy as np
import random
import yaml

OUT_DIR = "data_realistic"
IM_DIR_TRAIN = os.path.join(OUT_DIR, "images", "train")
IM_DIR_VAL = os.path.join(OUT_DIR, "images", "val")
LBL_DIR_TRAIN = os.path.join(OUT_DIR, "labels", "train")
LBL_DIR_VAL = os.path.join(OUT_DIR, "labels", "val")

NUM_IMAGES = 200
SIZE = 512

for d in [IM_DIR_TRAIN, IM_DIR_VAL, LBL_DIR_TRAIN, LBL_DIR_VAL]:
    os.makedirs(d, exist_ok=True)

def clamp(v, a, b):
    return max(a, min(b, v))

def generate_concrete_background(size):
    # Base concrete color (grey)
    bg = np.full((size, size, 3), (130, 130, 130), dtype=np.uint8)
    
    # Add random noise for texture
    noise = np.random.randn(size, size, 3) * 25
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    
    # Slight blur to soften the noise
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    
    return bg

# Net colors in BGR format
net_colors = [
    (34, 139, 34),   # Green
    (30, 30, 30),    # Black
    (208, 224, 64),  # Turquoise
    (255, 100, 0),   # Blue
]

for i in range(NUM_IMAGES):
    # 80/20 train/val split
    is_val = random.random() < 0.2
    im_out_dir = IM_DIR_VAL if is_val else IM_DIR_TRAIN
    lbl_out_dir = LBL_DIR_VAL if is_val else LBL_DIR_TRAIN

    # Generate background
    bg = generate_concrete_background(SIZE)

    # Net layer mask
    net_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    
    # Draw grid lines on mask
    spacing = random.randint(24, 40)
    thickness = random.randint(2, 4)
    for x in range(0, SIZE, spacing):
        cv2.line(net_mask, (x, 0), (x, SIZE), 1, thickness)
    for y in range(0, SIZE, spacing):
        cv2.line(net_mask, (0, y), (SIZE, y), 1, thickness)

    holes_yolo = []
    num_holes = random.randint(1, 6)
    for _ in range(num_holes):
        r = random.randint(10, 40)
        cx = random.randint(r + 4, SIZE - r - 4)
        cy = random.randint(r + 4, SIZE - r - 4)
        
        # Erase net mask where hole is
        cv2.circle(net_mask, (cx, cy), r, 0, -1)
        
        # Calculate bounding box for YOLO
        x1 = clamp(cx - r - 2, 0, SIZE - 1)
        y1 = clamp(cy - r - 2, 0, SIZE - 1)
        x2 = clamp(cx + r + 2, 0, SIZE - 1)
        y2 = clamp(cy + r + 2, 0, SIZE - 1)
        
        bw = x2 - x1
        bh = y2 - y1
        bx = x1 + bw / 2.0
        by = y1 + bh / 2.0
        
        nx = bx / SIZE
        ny = by / SIZE
        nw = bw / SIZE
        nh = bh / SIZE
        
        # Class 0 is 'hole'
        holes_yolo.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

    # Choose color and apply to background where mask is 1
    color = random.choice(net_colors)
    bg[net_mask == 1] = color
    
    # Apply slight blur to make the net look less artificially sharp
    bg = cv2.GaussianBlur(bg, (3, 3), 0)

    # Save image
    fname_base = f"net_{i:04d}"
    cv2.imwrite(os.path.join(im_out_dir, f"{fname_base}.png"), bg)

    # Save label
    with open(os.path.join(lbl_out_dir, f"{fname_base}.txt"), "w") as f:
        f.write("\n".join(holes_yolo))

    if (i + 1) % 20 == 0:
        print(f"Generated {i+1}/{NUM_IMAGES}")

# Write yaml file for YOLO
data_yaml = {
    'path': os.path.abspath(OUT_DIR).replace('\\', '/'),
    'train': 'images/train',
    'val': 'images/val',
    'names': ['hole']
}

with open(os.path.join(OUT_DIR, 'dataset.yaml'), 'w') as f:
    yaml.dump(data_yaml, f)

print(f"Dataset generation complete. Files saved in {OUT_DIR}")
