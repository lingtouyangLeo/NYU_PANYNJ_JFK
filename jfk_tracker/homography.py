# homography.py  (drop-in replacement)
import cv2
import numpy as np
import os
import json
from pathlib import Path

# ------------------------------- #
#            Tilt utils           #
# ------------------------------- #

def calculate_tilt_angle(M):
    if M is None:
        return None
    R = M[0:2, 0:2]
    scale = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if scale == 0:
        return None
    Rn = R / scale
    phi = np.arctan2(Rn[1,0], Rn[0,0])
    return np.degrees(phi)

def _sift_match(gray1, gray2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None, None, []
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.7 * n.distance]
    return kp1, kp2, good

def detect_camera_tilt(reference_frame, new_frame, angle_threshold_deg):
    if reference_frame is None or new_frame is None:
        return None, False
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    kp1, kp2, good = _sift_match(ref_gray, new_gray)
    if kp1 is None or len(good) < 20:
        return None, False

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 10)
    tilt = calculate_tilt_angle(M)
    if tilt is None:
        return None, False
    return tilt, (abs(tilt) > angle_threshold_deg)

# ------------------------------- #
#      Mask tuning                #
# ------------------------------- #

def _load_mask_tuning(mask_dir: str):
    """
    Optional JSON at <mask_dir>/mask_tuning.json
    {
      "global": {"dilate_px":6, "erode_px":0, "smooth_iters":1, "offset":[0,0]},
      "lanes": {
        "1": {"dilate_px":4, "offset":[-3,0]},
        "5": {"erode_px":3, "smooth_iters":2}
      }
    }
    Missing keys are treated as 0 / [0,0].
    """
    cfg_path = Path(mask_dir) / "mask_tuning.json"
    if not cfg_path.exists():
        return {"global":{}, "lanes":{}}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "global" not in cfg: cfg["global"] = {}
        if "lanes"  not in cfg: cfg["lanes"]  = {}
        return cfg
    except Exception as e:
        print(f"[mask_tuning] Failed to read {cfg_path}: {e}")
        return {"global":{}, "lanes":{}}

def _apply_shape_tuning(mask_bin, tuning):
    """
    Apply dilation/erosion, smoothing, and integer offset (dx, dy) on a binary mask (0/255).
    """
    dilate_px = int(max(0, tuning.get("dilate_px", 0)))
    erode_px  = int(max(0, tuning.get("erode_px", 0)))
    smooth_it = int(max(0, tuning.get("smooth_iters", 0)))
    dx, dy    = tuning.get("offset", [0,0]) if isinstance(tuning.get("offset", [0,0]), (list, tuple)) else (0,0)
    dx, dy    = int(dx), int(dy)

    out = mask_bin.copy()

    # dilation / erosion
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        out = cv2.dilate(out, k, iterations=1)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        out = cv2.erode(out, k, iterations=1)

    # smoothing (closing then opening)
    if smooth_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        for _ in range(smooth_it):
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k, iterations=1)

    # integer pixel offset using warpAffine
    if dx != 0 or dy != 0:
        H, W = out.shape[:2]
        A = np.float32([[1,0,dx],[0,1,dy]])
        out = cv2.warpAffine(out, A, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return out

def _read_and_prepare_masks(mask_dir: str, ref_w: int, ref_h: int):
    """Read all *.png masks; return list of (lane_id, mask_bin HxW)."""
    items = []
    for mf in sorted(os.listdir(mask_dir)):
        if not mf.lower().endswith(".png"):
            continue
        # lane id from "lane_1.png" -> 1 ; else keep filename
        try:
            lane_id = int(mf.split('_')[1].split('.')[0])
        except Exception:
            lane_id = mf
        m = cv2.imread(os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if (m.shape[1], m.shape[0]) != (ref_w, ref_h):
            m = cv2.resize(m, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        items.append((lane_id, m))
    return items

# ------------------------------- #
#   Homography + tuned warping    #
# ------------------------------- #

def find_homography_and_warp_masks(ref_image, current_frame, mask_dir):
    """
    1) 计算 ref -> current 的单应性 H
    2) 读取 mask_dir 下的 *.png 掩膜（ref 坐标系）
    3) 在 ref 空间内按 mask_tuning.json 进行调形（加宽/缩窄/平移/平滑）
    4) 用 H 把调形后的掩膜 warp 到当前帧，提取轮廓
    """
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    kp1, kp2, good = _sift_match(ref_gray, cur_gray)
    if kp1 is None or len(good) < 20:
        return {}, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if M is None:
        return {}, None

    h2, w2 = current_frame.shape[:2]
    ref_h, ref_w = ref_image.shape[:2]

    # 读取 masks & tuning
    tuning = _load_mask_tuning(mask_dir)
    g_tune = tuning.get("global", {})
    lane_tune = tuning.get("lanes", {})

    lane_contours = {}
    for lane_id, mask_ref in _read_and_prepare_masks(mask_dir, ref_w, ref_h):
        # 先按 global 调形，再按 lane 覆盖
        tuned = _apply_shape_tuning(mask_ref, g_tune)
        lt = lane_tune.get(str(lane_id), {})
        if lt:
            tuned = _apply_shape_tuning(tuned, lt)

        # warp to current frame
        warped = cv2.warpPerspective(tuned, M, (w2, h2), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_contours[lane_id] = contours

    return lane_contours, M

# ------------------------------- #
#   Rotation helper (optional)    #
# ------------------------------- #

def rotation_homography(angle_deg, cx, cy):
    """3x3 homography: rotation by angle_deg around (cx, cy)."""
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,   1]], dtype=np.float32)
    T2 = np.array([[1, 0,  cx],
                   [0, 1,  cy],
                   [0, 0,   1]], dtype=np.float32)
    return (T2 @ R @ T1).astype(np.float32)

# ------------------------------- #
#      Similarity / logging       #
# ------------------------------- #

def homography_distance(M1, M2):
    if M1 is None or M2 is None:
        return float("inf")
    return np.linalg.norm(M1.flatten() - M2.flatten())

def matrix_to_str(M):
    if M is None:
        return ""
    return ",".join(f"{v:.4f}" for v in M.flatten())

def recheck_perspective_for_one_second(
    cap, ref_image, fps, frame_number,
    first_M, first_lane_contours,
    mask_dir, find_homography_func,
    current_frame, similarity_threshold
):
    print(f"[Tilt @ frame {frame_number}] -> Starting 1-second re-check logic...")
    if first_M is None or not first_lane_contours:
        print("WARNING: The first homography or lane_contours is None. Can't re-check.")
        return None, None

    while True:
        Ms = [first_M]
        lane_contours_list = [first_lane_contours]

        for _ in range(int(fps) - 1):
            ok, f = cap.read()
            if not ok:
                print("Reached end of video during re-check. No stable homography found.")
                return None, None
            lc, M = find_homography_func(ref_image, f, mask_dir)
            Ms.append(M); lane_contours_list.append(lc)

        first_batch_M = Ms[0]
        similar = sum(1 for m in Ms if homography_distance(m, first_batch_M) < similarity_threshold)
        ratio = similar / len(Ms)
        print(f">> {similar}/{len(Ms)} homographies are similar (ratio={ratio:.2f}).")

        if ratio > 0.5:
            print(">> Majority similar. Accepting the first matrix for subsequent frames.")
            return first_batch_M, lane_contours_list[0]
        else:
            print(">> Not similar. Attempting another 1-second batch...")

        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print("No more frames left in video. Exiting re-check logic.")
            return None, None
