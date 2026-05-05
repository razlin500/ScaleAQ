import cv2
import numpy as np
from typing import List, Tuple

def detect_holes_bboxes(img: np.ndarray, min_area: int = 150, max_area: int = 8000,
                        circularity_thresh: float = 0.60) -> List[Tuple[int,int,int,int]]:
    """Detect circular hole-like dark regions and return bounding boxes (x,y,w,h).

    Filters contours by area, circularity and aspect ratio to prefer circular holes
    and avoid square/rectangular grid cells.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # binary: net lines will be white(255), background and holes black(0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    # invert so holes become white blobs on black background
    holes_mask = cv2.bitwise_not(thresh)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_area = gray.shape[0] * gray.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area or area > 0.9 * img_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))

        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / float(h) if h > 0 else 0
        extent = area / float(w * h) if (w * h) > 0 else 0

        # approximate polygon and count vertices — squares give 4 vertices, circles many
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        verts = len(approx)

        # Accept contours that either have good circularity or many polygon vertices
        # (many verts helps distinguish circles from rectangles/squares), and
        # have reasonable extent and aspect ratio.
        if (circularity >= circularity_thresh or verts >= 6) and 0.4 <= aspect <= 2.5 and extent > 0.4:
            boxes.append((x, y, w, h))
    return boxes

def annotate_image(img: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for (x, y, w, h) in boxes:
        # green box for circular holes
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

if __name__ == "__main__":
    # quick local smoke test if run directly
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python detector.py <image_path>")
        raise SystemExit(1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    boxes = detect_holes_bboxes(img)
    print(f"Found {len(boxes)} holes")
    out = annotate_image(img, boxes)
    cv2.imwrite(path.replace('.png', '_annotated.png'), out)
