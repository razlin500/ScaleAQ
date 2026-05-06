import base64
import os
import random

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO


MODEL_PATH = os.path.join("runs", "detect", "train-3", "weights", "best.pt")
VAL_IMAGE_DIR = os.path.join("data_realistic", "images", "val")


def _load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_yolo.py first."
        )
    return YOLO(MODEL_PATH)


def _run_detection(filepath: str):
    """Load an image from disk, run YOLO inference, return (b64_annotated, count)."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")
    results = model.predict(source=img, conf=0.25, device='cpu', verbose=False)
    result = results[0]
    out = result.plot()
    count = len(result.boxes) if result.boxes is not None else 0
    _, buf = cv2.imencode('.png', out)
    b64 = base64.b64encode(buf).decode('ascii')
    return f'data:image/png;base64,{b64}', count


model = _load_model()

app = Flask(__name__)


@app.route('/val_images/<path:filename>')
def val_image(filename):
    """Serve raw images from data_realistic/images/val."""
    return send_from_directory(VAL_IMAGE_DIR, filename)


@app.route('/', methods=['GET'])
def index():
    all_images = sorted(f for f in os.listdir(VAL_IMAGE_DIR) if f.endswith('.png'))

    # Pick 1 hero image for the scrollytelling section
    hero_filename = random.choice(all_images)
    hero_path = os.path.join(VAL_IMAGE_DIR, hero_filename)
    hero_annotated, hero_count = _run_detection(hero_path)

    # Pick 20 images for the gallery (excluding the hero)
    remaining = [f for f in all_images if f != hero_filename]
    gallery_images = random.sample(remaining, min(20, len(remaining)))

    return render_template(
        'index.html',
        hero_filename=hero_filename,
        hero_annotated=hero_annotated,
        hero_count=hero_count,
        gallery_images=gallery_images,
    )


@app.route('/api/detect_path', methods=['POST'])
def api_detect_path():
    """Detect holes in a pre-existing val image given by filename."""
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400

    filename = os.path.basename(data['filename'])  # safety: strip any path components
    filepath = os.path.join(VAL_IMAGE_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Image not found'}), 404

    try:
        annotated_image, count = _run_detection(filepath)
    except ValueError as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'annotated_image': annotated_image, 'count': count})


if __name__ == '__main__':
    app.run(debug=True)
