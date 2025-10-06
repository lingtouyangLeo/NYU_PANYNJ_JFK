import numpy as np
import cv2
from collections import defaultdict

# Map raw YOLO class names to reporting categories
CLASS_TO_CATEGORY = {
    "car": "passenger_car",
    "van": "passenger_car",
    "bus": "large_vehicle",
    "truck": "large_vehicle",
    "pedestrian": "people",
    "people": "people",
}
VALID_CLASSES = set(CLASS_TO_CATEGORY.keys())
CATEGORIES = ["passenger_car", "large_vehicle", "people"]

def run_yolo(model, frame):
    # Safe wrapper: handle empty results
    res = model.track(frame, persist=True, verbose=False, iou=0.15)
    if not res or res[0].boxes is None or res[0].boxes.xywh is None:
        return np.zeros((0,4)), [], []
    boxes = res[0].boxes.xywh.cpu().numpy()
    tids = [] if res[0].boxes.id is None else res[0].boxes.id.int().cpu().tolist()
    clsi = res[0].boxes.cls.int().cpu().tolist()
    names = model.names
    return boxes, tids, [names[c] for c in clsi]

def assign_lane_for_box(box, lane_contours):
    x, y, w, h = box
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    xs = [x1 + w/6, x1 + w/2, x2 - w/6]
    ys = [y1 + h/6, y1 + h/2, y2 - h/6]
    grid = [(int(px), int(py)) for px in xs for py in ys]

    hits = defaultdict(int)
    for pt in grid:
        for ln, contours in lane_contours.items():
            for c in contours:
                if cv2.pointPolygonTest(c, pt, False) >= 0:
                    hits[ln] += 1

    if not hits:
        center = (int(x), int(y))
        for ln, contours in lane_contours.items():
            for c in contours:
                if cv2.pointPolygonTest(c, center, False) >= 0:
                    return ln
        return None

    max_hit = max(hits.values())
    best = [ln for ln, cnt in hits.items() if cnt == max_hit]
    if len(best) == 1:
        return best[0]

    center = (int(x), int(y))
    for ln, contours in lane_contours.items():
        for c in contours:
            if cv2.pointPolygonTest(c, center, False) >= 0:
                return ln
    return best[0]

def filter_and_label(boxes, tids, classes, lane_contours):
    f_boxes, f_ids, f_cats, f_lanes = [], [], [], []
    for i, b in enumerate(boxes):
        if i >= len(tids) or i >= len(classes):
            continue
        raw = classes[i]
        if raw not in VALID_CLASSES:
            continue
        cat = CLASS_TO_CATEGORY[raw]
        ln = assign_lane_for_box(b, lane_contours)
        if ln is None:
            continue
        f_boxes.append(b)
        f_ids.append(tids[i])
        f_cats.append(cat)
        f_lanes.append(ln)
    return f_boxes, f_ids, f_cats, f_lanes
