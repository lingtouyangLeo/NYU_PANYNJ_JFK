import argparse
from pathlib import Path
import cv2

# Usage:
# powershell:
#   python -m jfk_tracker.make_ref_img --video "path/to/video.mp4" --out-dir "masks/ref_imgs"
# If --out-dir is omitted, it defaults to project_root/masks/ref_imgs
# The output filename will be <video_basename>_ref.jpg

def extract_first_frame(video_path: Path, out_dir: Path) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    # Build output filename
    stem = video_path.stem
    out_path = out_dir / f"{stem}_ref.jpg"

    # Write JPEG
    success = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise RuntimeError(f"Failed to write reference image to {out_path}")

    print(f"[OK] Wrote reference image: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract first frame as reference image for homography/masks.")
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--out-dir', default=None, help='Directory to store reference image (default: masks/ref_imgs under project root)')
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    # Determine default output directory relative to repo root (this file is jfk_tracker/make_ref_img.py)
    if args.out_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / 'masks' / 'ref_imgs'
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    extract_first_frame(video_path, out_dir)


if __name__ == '__main__':
    main()
