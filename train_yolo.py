import os
from ultralytics import YOLO

DATA_YAML = 'data_realistic/dataset.yaml'
EPOCHS = 50

def main():
    if not os.path.exists(DATA_YAML):
        raise SystemExit(f"Dataset YAML not found: {DATA_YAML}. Run convert_annotations.py first.")

    # Use yolov8n small model for quick experiments
    model = YOLO('yolov8n.pt')

    # Train on CPU explicitly (Windows + AMD GPU likely unsupported by CUDA)
    model.train(data=DATA_YAML, epochs=EPOCHS, device='cpu')

if __name__ == '__main__':
    main()
