# main.py  (compatible replacement; interface unchanged)
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

from .config import (
    VIDEO_PATH, MASK_DIR, REF_IMAGE_PATH, MODEL_PATH, TEXT_AREA_HEIGHT,
    TILT_ANGLE_THRESHOLD_DEG, CHECK_TILT_EVERY_SECS, SIMILARITY_THRESHOLD, RECHECK_BATCH_ENABLED,
    DWELL_TIME_THRESHOLD_SECS, DISTANCE_THRESHOLD_PX, SHOW_WINDOW,
    make_output_dir, pick_device
)
from .homography import (
    find_homography_and_warp_masks, detect_camera_tilt, recheck_perspective_for_one_second, matrix_to_str
)
from .detection import run_yolo, filter_and_label, CATEGORIES
from .drawing import draw_lane_overlay, draw_detections, compose_text_area
from .io_utils import append_records, write_tsv

def main():
    tag = Path(VIDEO_PATH).stem
    out_dir = make_output_dir(tag)
    out_video = str(out_dir / "output_video.mp4")
    out_tsv   = str(out_dir / "detections.tsv")
    print(f"[IO] Output dir: {out_dir}")

    device = pick_device()
    print(f"[Device] {device}")

    model = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)  # keep default backend
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h = height + TEXT_AREA_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, out_h))

    # ---- first frame + initial homography ----
    ok, first = cap.read()
    if not ok:
        print("Could not read the first frame."); return
    ref = cv2.imread(REF_IMAGE_PATH)
    if ref is None:
        print(f"Could not read reference image {REF_IMAGE_PATH}."); return

    lane_contours, H = find_homography_and_warp_masks(ref, first, MASK_DIR)
    if not lane_contours:
        print("Warning: initial lane_contours is empty. Check your masks/homography.")

    lane_colors = {ln: np.random.randint(50,256,(3,)).tolist() for ln in lane_contours.keys()}
    reference_frame_for_tilt = first.copy()

    # start from frame #1 (we already consumed #0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    last_move_by_id = {}
    records = []
    frame_idx = 0

    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    with tqdm(total=max(total-1, 0), desc="Processing Video") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            perspective_change = False
            matrix_str = ""

            # ---- tilt cadence ----
            if CHECK_TILT_EVERY_SECS and int(CHECK_TILT_EVERY_SECS) > 0:
                if frame_idx % int(round(CHECK_TILT_EVERY_SECS * fps)) == 0:
                    tilt_deg, tilted = detect_camera_tilt(reference_frame_for_tilt, frame, TILT_ANGLE_THRESHOLD_DEG)
                    if tilt_deg is not None:
                        print(f"Frame {frame_idx}: tilt angle = {tilt_deg:.2f}°")
                    if tilted:
                        print(f"Frame {frame_idx}: TILT DETECTED -> recomputing homography...")
                        new_lc, new_H = find_homography_and_warp_masks(ref, frame, MASK_DIR)
                        if RECHECK_BATCH_ENABLED and new_H is not None and new_lc:
                            stable_H, stable_lc = recheck_perspective_for_one_second(
                                cap=cap, ref_image=ref, fps=fps, frame_number=frame_idx,
                                first_M=new_H, first_lane_contours=new_lc, mask_dir=MASK_DIR,
                                find_homography_func=find_homography_and_warp_masks,
                                current_frame=frame, similarity_threshold=SIMILARITY_THRESHOLD
                            )
                            if stable_H is not None and stable_lc:
                                H = stable_H
                                lane_contours = stable_lc
                                lane_colors = {ln: np.random.randint(50,256,(3,)).tolist()
                                               for ln in lane_contours.keys()}
                                reference_frame_for_tilt = frame.copy()
                                perspective_change = True
                                matrix_str = matrix_to_str(stable_H)
                            else:
                                print("WARNING: No stable homography in re-check. Keeping previous transform.")
                        else:
                            if new_H is not None and new_lc:
                                H = new_H
                                lane_contours = new_lc
                                lane_colors = {ln: np.random.randint(50,256,(3,)).tolist()
                                               for ln in lane_contours.keys()}
                                reference_frame_for_tilt = frame.copy()
                                perspective_change = True
                                matrix_str = matrix_to_str(new_H)
                            else:
                                print("Homography for tilt frame is None. Lane contours unchanged.")

            if not lane_contours:
                panel = np.zeros((TEXT_AREA_HEIGHT, width, 3), dtype=np.uint8)
                writer.write(np.vstack((frame, panel)))
                pbar.update(1)
                continue

            vis = draw_lane_overlay(frame.copy(), lane_contours, lane_colors, alpha=0.3)

            boxes, tids, classes = run_yolo(model, frame)
            f_boxes, f_ids, f_cats, f_lanes = filter_and_label(boxes, tids, classes, lane_contours)

            lane_counts = defaultdict(lambda: defaultdict(int))
            lane_over = defaultdict(set)
            for i, _ in enumerate(f_boxes):
                ln = f_lanes[i]; cat = f_cats[i]
                lane_counts[ln][cat] += 1

            draw_detections(
                vis, frame_idx, fps,
                f_boxes, f_ids, f_cats, f_lanes,
                last_move_by_id,
                dwell_secs=DWELL_TIME_THRESHOLD_SECS,
                dist_thresh_px=DISTANCE_THRESHOLD_PX
            )

            for i, b in enumerate(f_boxes):
                tid = f_ids[i]; cat = f_cats[i]; ln = f_lanes[i]
                if cat != "people":
                    last_frame, _ = last_move_by_id.get(tid, (frame_idx, (int(b[0]), int(b[1]))))
                    dt = (frame_idx - last_frame) / fps
                    if dt >= DWELL_TIME_THRESHOLD_SECS:
                        lane_over[ln].add(tid)

            panel = compose_text_area(width, TEXT_AREA_HEIGHT, lane_contours, lane_counts, lane_over, CATEGORIES)
            writer.write(np.vstack((vis, panel)))

            append_records(records, frame_idx, fps, f_boxes, f_ids, f_cats, f_lanes, perspective_change, matrix_str)

            if SHOW_WINDOW:
                cv2.imshow("FHV Tracking", np.vstack((vis, panel)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    write_tsv(out_tsv, records)
    print(f"[Done] Video -> {out_video}")
    print(f"[Done] TSV   -> {out_tsv}")

if __name__ == "__main__":
    main()
