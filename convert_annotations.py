import os
import json
import shutil
from glob import glob
import random

SRC_IMG_DIR = 'data/images'
SRC_ANN_DIR = 'data/annotations'
OUT_IMG_DIR = 'data/yolo/images'
OUT_LABEL_DIR = 'data/yolo/labels'

TRAIN_DIR = os.path.join(OUT_IMG_DIR, 'train')
VAL_DIR = os.path.join(OUT_IMG_DIR, 'val')
TRAIN_LABEL_DIR = os.path.join(OUT_LABEL_DIR, 'train')
VAL_LABEL_DIR = os.path.join(OUT_LABEL_DIR, 'val')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

ann_files = sorted(glob(os.path.join(SRC_ANN_DIR, '*.json')))
random.shuffle(ann_files)
split = int(0.8 * len(ann_files))
train_files = ann_files[:split]
val_files = ann_files[split:]

def convert_one(json_path, out_img_dir, out_label_dir):
    with open(json_path, 'r') as f:
        ann = json.load(f)
    img_name = ann['image']
    w = ann['width']
    h = ann['height']
    boxes = ann.get('boxes', [])

    # copy image
    src_img = os.path.join(SRC_IMG_DIR, img_name)
    dst_img = os.path.join(out_img_dir, img_name)
    shutil.copyfile(src_img, dst_img)

    # write label file in YOLO format (class x_center y_center width height) normalized
    label_name = os.path.splitext(img_name)[0] + '.txt'
    lines = []
    for box in boxes:
        x, y, bw, bh = box
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        nx = cx / w
        ny = cy / h
        nw = bw / w
        nh = bh / h
        # single class 0
        lines.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

    with open(os.path.join(out_label_dir, label_name), 'w') as f:
        f.write('\n'.join(lines))

for j in train_files:
    convert_one(j, TRAIN_DIR, TRAIN_LABEL_DIR)
for j in val_files:
    convert_one(j, VAL_DIR, VAL_LABEL_DIR)

# write dataset yaml
data_yaml = {
    'path': 'data/yolo',
    'train': 'images/train',
    'val': 'images/val',
    'names': ['hole']
}

import yaml
with open('data/yolov8_dataset.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print(f'Converted {len(train_files)} train and {len(val_files)} val samples. Wrote data/yolov8_dataset.yaml')
