from pathlib import Path
import re
import torch

# ------------- Paths -------------
# Resolve project root as the parent of this package folder
ROOT = Path(__file__).resolve().parent.parent

# Set concrete defaults that match your current workspace
VIDEO_PATH = str(ROOT / "NYU_PANYNJ" / "Arrival" / "Asheque Rahman - Terminal 4 _ Arrivals _ Wednesday May 28 2025 _ 2pm to 3pm.avi")
MASK_DIR = str(ROOT / "masks" / "masks_Asheque Rahman - Terminal 4 _ Arrivals _ Wednesday May 28 2025 _ 2pm to 3pm")
REF_IMAGE_PATH = str(ROOT / "masks" / "ref_imgs" / "Asheque Rahman - Terminal 4 _ Arrivals _ Wednesday May 28 2025 _ 2pm to 3pm_ref.jpg")
MODEL_PATH = str(ROOT / "Yolo_Models" / "yolo11l-general.pt") # "best.pt" or "yolo11s-visdrone.pt" or "yolo11l-general.pt"

# ------------- Visualization -------------
TEXT_AREA_HEIGHT = 240
SHOW_WINDOW = True # set True to cv2.imshow while running 

# ------------- Homography / Tilt -------------
TILT_ANGLE_THRESHOLD_DEG = 10 # consider 'tilt' if |angle| > 10
CHECK_TILT_EVERY_SECS = 1
SIMILARITY_THRESHOLD = 20.0 # norm(Mi - M0) < threshold => “similar”
RECHECK_BATCH_ENABLED = True # 1-second “majority similar” check

# ------------- Dwell thresholds -------------
DWELL_TIME_THRESHOLD_SECS = 90 # seconds of “little movement”
DISTANCE_THRESHOLD_PX = 40 # px movement to reset dwell timer

def make_output_dir(video_name: str, prefix="yolo11l_output_"):
    # Clean video name: keep only letters, numbers, dot, underscore, dash
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', "_", video_name)[:60]

    i = 0
    while Path(f"{prefix}{i}_{safe_name}").exists():
        i += 1
    out = Path(f"{prefix}{i}_{safe_name}")
    out.mkdir(parents=True, exist_ok=True)
    return out

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

