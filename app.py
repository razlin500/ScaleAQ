import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO


MODEL_PATH = os.path.join("runs", "detect", "train-3", "weights", "best.pt")


def _load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_yolo.py first."
        )
    return YOLO(MODEL_PATH)


model = _load_model()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', annotated_image=None)

@app.route('/detect', methods=['POST'])
def detect():
    f = request.files.get('image')
    if not f:
        return render_template('index.html', annotated_image=None, error='No file uploaded')

    data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return render_template('index.html', annotated_image=None, error='Could not read image')

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = model.predict(source=rgb, conf=0.25, device='cpu', verbose=False)

    result = results[0]
    out = result.plot()

    count = len(result.boxes) if result.boxes is not None else 0

    _, buf = cv2.imencode('.png', out)
    b64 = base64.b64encode(buf).decode('ascii')
    data_uri = f'data:image/png;base64,{b64}'
    return render_template('index.html', annotated_image=data_uri, count=count)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'No file uploaded'}), 400

    data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = model.predict(source=rgb, conf=0.25, device='cpu', verbose=False)

    result = results[0]
    out = result.plot()

    count = len(result.boxes) if result.boxes is not None else 0

    _, buf = cv2.imencode('.png', out)
    b64 = base64.b64encode(buf).decode('ascii')
    data_uri = f'data:image/png;base64,{b64}'
    
    return jsonify({'annotated_image': data_uri, 'count': count})

if __name__ == '__main__':
    app.run(debug=True)
