import csv

TSV_FIELDS = [
    "frame","timestamp","track_id","class",
    "bbox_x","bbox_y","bbox_w","bbox_h",
    "assigned_lane","perspective_change","homography_matrix"
]

def append_records(records, frame_idx, fps, boxes, tids, cats, lanes,
                   perspective_change, matrix_str):
    for i, b in enumerate(boxes):
        x, y, w, h = b
        records.append({
            "frame": frame_idx,
            "timestamp": frame_idx / fps,
            "track_id": tids[i],
            "class": cats[i],
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "assigned_lane": lanes[i],
            "perspective_change": perspective_change,
            "homography_matrix": matrix_str if perspective_change else ""
        })

def write_tsv(path, records):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        wr.writeheader()
        for r in records:
            wr.writerow(r)
