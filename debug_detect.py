import cv2
import numpy as np

img = cv2.imread('data/images/net_0000.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5,5), 0)
_, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
holes_mask = cv2.bitwise_not(thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel, iterations=1)
contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Total contours:', len(contours))
import json
ann = json.load(open('data/annotations/net_0000.json','r'))
gt_boxes = ann['boxes']
print('Ground-truth boxes:', gt_boxes)
import math
img_area = img.shape[0]*img.shape[1]
for i,c in enumerate(sorted(contours, key=lambda c: -cv2.contourArea(c))[:40]):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    circ = 4.0*math.pi*(area/(perim*perim)) if perim>0 else 0
    x,y,w,h = cv2.boundingRect(c)
    aspect = w/h if h>0 else 0
    extent = area/(w*h) if (w*h)>0 else 0
    approx = cv2.approxPolyDP(c, 0.02*perim, True)
    verts = len(approx)
    (mx,my), mr = cv2.minEnclosingCircle(c)
    circle_area = math.pi * (mr*mr)
    circle_ratio = area / circle_area if circle_area>0 else 0
    # check overlap with any GT box (center within gt box)
    cx = x + w/2
    cy = y + h/2
    overlapped = [j for j,bb in enumerate(gt_boxes) if (cx>=bb[0] and cx<=bb[0]+bb[2] and cy>=bb[1] and cy<=bb[1]+bb[3])]
    # Apply detector heuristics (same as detector.py)
    accept = False
    if (circ >= 0.60 or verts >= 6) and 0.4 <= aspect <= 2.5 and extent > 0.4:
        accept = True
    print(i, 'area=', int(area), 'perim=', int(perim), 'circ={:.2f}'.format(circ), 'circle_ratio={:.2f}'.format(circle_ratio), 'verts=', verts, 'aspect={:.2f}'.format(aspect), 'extent={:.2f}'.format(extent), 'bbox=', (x,y,w,h), 'overlaps_gt=', overlapped, 'ACCEPT' if accept else '')
