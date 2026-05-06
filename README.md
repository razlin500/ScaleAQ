# Net Hole Detection

This project is a local proof-of-concept for detecting holes in fish-farm nets.
Because there was no suitable real-world dataset available, the repo first generates
realistic synthetic training images featuring various net colors (green, black, turquoise, blue)
over simulated concrete floor backgrounds, complete with circular holes.
Those generated annotations are output directly into YOLO format and used to train a lightweight
YOLOv8 model. A modern Flask app then provides a professional, browser-based interface for
uploading images, viewing galleries of examples, and performing real-time hole detection analysis.

## What it does

- Generates realistic synthetic net images with different colors, backgrounds, and randomly placed circular holes.
- Automatically creates corresponding YOLO label annotations.
- Trains a local YOLOv8 model using Ultralytics.
- Serves a modern, interactive Flask GUI with a sleek design to upload an image and view annotated detections in real-time.
- Runs entirely on a local machine; no cloud service is required.

## Project structure

- `generate_dataset.py` - creates the realistic synthetic images and YOLO bounding-box annotations.
- `train_yolo.py` - trains a YOLOv8 model on the generated dataset.
- `eval_yolo.py` - runs inference on held-out validation images.
- `app.py` - Flask app that loads the trained model and serves the modern front-end for detections in the browser.
- `detector.py` - core logic for loading the model and predicting bounding boxes.
- `data_realistic/` - generated images, labels, and dataset metadata.
- `runs/` - YOLO training and evaluation outputs.
- `static/` & `templates/` - frontend assets (CSS, JS, HTML) for the modern web interface.

## Setup

Create and activate a Python virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical workflow

Generate synthetic data:

```powershell
python generate_dataset.py
```

Train the model:

```powershell
python train_yolo.py
```

Run a quick evaluation:

```powershell
python eval_yolo.py
```

Start the application:

```powershell
python app.py
```

Then open http://127.0.0.1:5000/ in your browser to access the web UI, explore the gallery, and upload images to test the detection model.

## Notes

- The dataset is synthetic and intended for proof-of-concept only.
- The model is trained locally with Ultralytics YOLOv8.
- The web app is designed to be interactive and visually appealing, demonstrating how a production UI might look and function.
