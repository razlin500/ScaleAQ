import os
from glob import glob
from ultralytics import YOLO


def main():
    weights = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
    val_images = sorted(glob(os.path.join('data', 'yolo', 'images', 'val', '*.png')))

    if not os.path.exists(weights):
        raise SystemExit(f'Missing weights: {weights}')
    if not val_images:
        raise SystemExit('No validation images found.')

    model = YOLO(weights)

    # Run inference on a small held-out sample and save annotated outputs.
    sample = val_images[:10]
    results = model.predict(source=sample, conf=0.25, save=True, project='runs', name='eval', device='cpu')

    print(f'Evaluated {len(results)} images.')
    print(f'Results saved under: {os.path.join("runs", "eval")}')


if __name__ == '__main__':
    main()
