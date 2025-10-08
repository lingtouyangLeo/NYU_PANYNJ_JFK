# JFK Airport Vehicle Tracking System

A computer vision system for tracking and analyzing vehicle traffic at JFK Airport Terminal 4 using YOLO object detection and lane-based analysis.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Processing a New Video](#processing-a-new-video)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Output](#output)

## Overview

This system processes video footage from JFK Airport to:
- Detect and track vehicles (passenger cars, large vehicles)
- Separate traffic by lane using custom-drawn lane masks
- Identify vehicles that dwell beyond a threshold time
- Generate annotated output videos with lane-by-lane statistics
- Export detection data to TSV format for analysis

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster processing)
- Conda or virtual environment

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lingtouyangLeo/NYU_PANYNJ_JFK.git
cd jfk
```

2. Install dependencies:
```bash
pip install opencv-python numpy ultralytics torch tqdm
```

3. Download YOLO model:
Place your YOLO model (e.g., `yolo11l-general.pt`) in the `Yolo_Models/` directory.

## Processing a New Video

Follow these steps to process a new video:

### Step 1: Extract Reference Image

Extract the first frame from your video to use as a reference for lane drawing:

```bash
python -m jfk_tracker.make_ref_img --video "path/to/your/video.avi"
```

**Output:** `masks/ref_imgs/video_name_ref.jpg`

### Step 2: Draw Lane Boundaries

Use the interactive lane drawing tool to manually define each lane:

1. Update the video name in `jfk_tracker/_draw_lanes.py`:
```python
video_name = "your_video_name"  # without _ref.jpg suffix
```

2. Run the drawing tool:
```bash
python jfk_tracker/_draw_lanes.py
```

3. **Interactive Drawing Instructions:**
   - **Left Click**: Add points to define the current lane boundary (minimum 3 points)
   - **Right Click**: Complete current lane and start next lane
   - **Keyboard Shortcuts:**
     - `u`: Undo last point
     - `c`: Clear current lane
     - `r`: Reset all lanes
     - `s`: **Save** colored reference image and lane data
     - `q` or `ESC`: Quit without saving

4. Draw all 9 lanes (or adjust `num_lanes` as needed)

**Output:** 
- `masks/masks_video_name/video_name_colored_ref.jpg`
- `masks/masks_video_name/lane_data.json`

### Step 3: Generate Lane Masks

Generate binary masks for each lane from the colored reference image:

1. Update the video name in `jfk_tracker/_make_mask.py`:
```python
video_name = "your_video_name"
num_lanes = 9  # adjust based on your setup
```

2. Run the mask generation script:
```bash
python jfk_tracker/_make_mask.py
```

3. **Interactive Mask Generation** (repeat for each lane):
   - **First Click**: Select the color of the lane by clicking on it
   - **Second Click**: Click again within the same colored area to flood-fill
   - The mask will be automatically saved as `mask_X.png`
   - Script will prompt you to continue with the next lane
   - Press any key after all lanes are completed

**Tips:**
- Click in the **center** of each colored lane region (avoid edges)
- If a mask is too large/small, adjust `tolerance` parameter (default: 15)
- Press `Ctrl+C` or close window to exit early if needed

**Output:** `masks/masks_video_name/mask_1.png` through `mask_9.png`

### Step 4: Verify Mask Quality

Check that each mask covers an appropriate area:

```bash
python -c "import cv2; import numpy as np; [print(f'Lane {i}: {np.count_nonzero(cv2.imread(f\"masks/masks_video_name/mask_{i}.png\", 0)):,} pixels') for i in range(1, 10)]"
```

Each lane should typically cover 3-10% of the total frame area. If a mask is too large (>20%), regenerate it with a lower tolerance value.

### Step 5: Update Configuration

Edit `jfk_tracker/config.py` to point to your new video and masks:

```python
VIDEO_PATH = str(ROOT / "NYU_PANYNJ" / "Arrival" / "your_video.avi")
MASK_DIR = str(ROOT / "masks" / "masks_your_video_name")
REF_IMAGE_PATH = str(ROOT / "masks" / "ref_imgs" / "your_video_name_ref.jpg")
MODEL_PATH = str(ROOT / "Yolo_Models" / "yolo11l-general.pt")
```

### Step 6: Run Main Processing Pipeline

Process the video with vehicle detection and tracking:

```bash
python -m jfk_tracker.main
```

The script will:
1. Load the video and reference image
2. Compute homography and warp lane masks to each frame
3. Detect and track vehicles using YOLO
4. Assign vehicles to lanes based on mask regions
5. Monitor dwell times for stationary vehicles
6. Generate annotated output video with lane statistics
7. Export detection data to TSV

**Output:**
- `yolo11l_output_X_video_name/output_video.mp4`: Annotated video with lane overlays and detection boxes
- `yolo11l_output_X_video_name/detections.tsv`: Frame-by-frame detection data

## Configuration

Key parameters in `jfk_tracker/config.py`:

### Detection & Tracking
- `MODEL_PATH`: Path to YOLO model file
- `CATEGORIES`: Vehicle types to detect (passenger_car, large_vehicle, people)

### Lane Masks
- `MASK_DIR`: Directory containing lane mask PNG files
- `REF_IMAGE_PATH`: Reference image for homography

### Dwell Detection
- `DWELL_TIME_THRESHOLD_SECS`: Time (seconds) before a vehicle is flagged as dwelling (default: 90)
- `DISTANCE_THRESHOLD_PX`: Minimum movement (pixels) to reset dwell timer (default: 40)

### Homography & Tilt Detection
- `TILT_ANGLE_THRESHOLD_DEG`: Angle threshold to trigger homography recalculation (default: 10°)
- `CHECK_TILT_EVERY_SECS`: Frequency to check for camera tilt (default: 1 second)
- `SIMILARITY_THRESHOLD`: Threshold for homography matrix similarity (default: 20.0)

### Visualization
- `SHOW_WINDOW`: Display live preview during processing (default: True)
- `TEXT_AREA_HEIGHT`: Height of statistics panel at bottom of video (default: 240)

## Project Structure

```
jfk/
├── jfk_tracker/              # Main package
│   ├── config.py            # Configuration settings
│   ├── main.py              # Main processing pipeline
│   ├── detection.py         # YOLO detection and tracking
│   ├── homography.py        # Homography and mask warping
│   ├── drawing.py           # Visualization functions
│   ├── io_utils.py          # File I/O utilities
│   ├── _draw_lanes.py       # Interactive lane drawing tool
│   ├── _make_mask.py        # Interactive mask generation tool
│   └── make_ref_img.py      # Reference image extraction
├── masks/                   # Lane masks and reference images
│   ├── ref_imgs/           # Reference images from videos
│   └── masks_*/            # Lane mask directories (one per video)
├── Yolo_Models/            # YOLO model weights
├── NYU_PANYNJ/             # Video files
│   ├── Arrival/
│   ├── Departure/
│   └── ...
└── output/                 # Processing results (generated)
```

## Output

### Video Output

The annotated output video includes:
- Lane overlays with semi-transparent colors
- Green dots for tracked vehicles
- Orange/red boxes for dwelling vehicles with time display
- Bottom panel with lane-by-lane statistics:
  - Vehicle counts by category (passenger_car, large_vehicle, people)
  - Count of vehicles over dwell threshold

### TSV Output

The detections TSV file contains:
- Frame number and timestamp
- Track ID, category, lane assignment
- Bounding box coordinates (x, y, w, h)
- Perspective change indicators

## Troubleshooting

### Common Issues

**Q: Lane masks are too large or include unwanted areas**
- A: Lower the `tolerance` parameter in `_make_mask.py` (try 10 or 5)
- A: Click more precisely in the center of each colored lane region

**Q: Homography fails or lanes don't align**
- A: Ensure reference image matches video resolution
- A: Check that first frame has sufficient feature points for matching

**Q: Vehicles not being assigned to correct lanes**
- A: Verify mask files cover the intended areas using Step 4
- A: Ensure lane colors in colored_ref image are pure (no blending)

**Q: Processing is very slow**
- A: Disable live preview: Set `SHOW_WINDOW = False` in config.py
- A: Use GPU acceleration: Install CUDA and PyTorch with CUDA support
- A: Use a smaller/faster YOLO model (e.g., yolo11s instead of yolo11l)

## License

[Add your license information here]

## Contact

[Add contact information here]
