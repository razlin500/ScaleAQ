# Net Hole Detection Mockup

This project is a local proof-of-concept for detecting holes in fish-farm nets.
Because there was no suitable real-world dataset available, the repo first generates
synthetic training images with a simple grid-like net pattern and circular holes.
Those annotations are converted into YOLO format and used to train a lightweight
YOLOv8 model. A small Flask app then provides a browser-based interface for
uploading images and viewing detections.

## What it does

- Generates synthetic net images with randomly placed circular holes.
- Converts the synthetic annotations into YOLO labels.
- Trains a local YOLOv8 model using Ultralytics.
- Serves a simple Flask GUI to upload an image and view annotated detections.
- Runs entirely on a local machine; no cloud service is required.

## Project structure

- `generate_dataset.py` - creates the synthetic images and bounding-box annotations.
- `convert_annotations.py` - converts the annotations into YOLO format.
- `train_yolo.py` - trains a YOLOv8 model on the generated dataset.
- `eval_yolo.py` - runs inference on held-out validation images.
- `app.py` - Flask mockup that loads the trained model and shows detections in the browser.
- `data/` - generated images, labels, and dataset metadata.
- `runs/` - YOLO training and evaluation outputs.

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

Convert annotations to YOLO format:

```powershell
python convert_annotations.py
```

Train the model:

```powershell
python train_yolo.py
```

Run a quick evaluation:

```powershell
python eval_yolo.py
```

Start the Flask mockup:

```powershell
python app.py
```

Then open http://127.0.0.1:5000/ in your browser and upload one of the generated images from `data/images`.

## Notes

- The dataset is synthetic and intended for proof-of-concept only.
- The model is trained locally with Ultralytics YOLOv8.
- The app is intentionally minimal so it is easy to test, modify, and extend.
