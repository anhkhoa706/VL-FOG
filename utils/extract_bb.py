import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import pickle
import gzip
import hashlib
import contextlib
from ultralytics import YOLO
import cv2
import torch
import argparse

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR       = "data/AVA_Dataset"
CATEGORIES     = ["freeway", "road"]
SPLIT          = "train"  # Will be overwritten by argument
YOLO_WEIGHTS   = "yolo11x_trained.pt"                 # use the newest, largest YOLOv11x
VEHICLE_CLASSES = {0, 1, 2, 3}                        # COCO IDs: car, motorcycle, bus, truck

# ── UTILITY: suppress YOLO prints ───────────────────────────────────────────────
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr in this context."""
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = os.sys.stdout, os.sys.stderr
        os.sys.stdout, os.sys.stderr = devnull, devnull
        try:
            yield
        finally:
            os.sys.stdout, os.sys.stderr = old_out, old_err

# ── CACHING UTILITIES ──────────────────────────────────────────────────────────
def compute_cache_key(category, video_name):
    """Compute cache key based only on category and video name for better integration."""
    key_data = f"{category}_{video_name}"
    return hashlib.md5(key_data.encode()).hexdigest()

def load_bb_cache(cache_dir, key):
    path = os.path.join(cache_dir, f"bb_{key}.pkl.gz")
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def save_bb_cache(cache_dir, key, bb_data):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"bb_{key}.pkl.gz")
    with gzip.open(path, 'wb') as f:
        pickle.dump(bb_data, f)

# ── DETECTION ─────────────────────────────────────────────────────────────────
def detect_vehicle_boxes(frame, det_model):
    """
    Returns a list of (x1,y1,x2,y2) for vehicles in BGR `frame`.
    """
    with suppress_stdout_stderr():
        result = det_model(frame, verbose=False)[0]
    return [tuple(box.cpu().numpy().astype(int))
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls)
            if int(cls) in VEHICLE_CLASSES]

# ── MAIN: extract BBs & print counts for each video ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vehicle bounding boxes using YOLO.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (train or test)")
    args = parser.parse_args()
    SPLIT = args.split

    det_model = YOLO(YOLO_WEIGHTS, verbose=False)
    total_new = 0
    total_frames_with_vehicles = 0

    for category in CATEGORIES:
        split_dir = os.path.join(BASE_DIR, category, SPLIT)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Missing directory: {split_dir}")
            continue
        bb_cache_dir = os.path.join(split_dir, ".bb_cache")
        os.makedirs(bb_cache_dir, exist_ok=True)

        videos = [d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d))]

        for vid in videos:
            folder = os.path.join(split_dir, vid)
            frames = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith((".jpg", ".png"))])
            if not frames:
                print(f"[SKIP] {category}/{vid}: no frames found")
                continue

            frame_paths = [os.path.join(folder, f) for f in frames]
            key = compute_cache_key(category, vid)
            
            if load_bb_cache(bb_cache_dir, key) is not None:
                print(f"[SKIP] {category}/{vid}: already cached")
                continue

            print(f"[COMPUTE] {category}/{vid}")
            bb_data = []  # List of (frame_name, boxes) pairs
            frames_with_vehicles = 0
            
            for i, (frame_name, frame_path) in enumerate(zip(frames, frame_paths)):
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARN] Could not read frame {i+1}/{len(frames)}: {frame_path}")
                    continue
                
                boxes = detect_vehicle_boxes(img, det_model)
                if boxes:  # Only include frames with vehicles
                    bb_data.append((frame_name, boxes))
                    frames_with_vehicles += 1
                

            if bb_data:  # Only save if we found any frames with vehicles
                save_bb_cache(bb_cache_dir, key, bb_data)
                total_frames_with_vehicles += frames_with_vehicles

                # Count BBs per frame and print result
                counts = [len(boxes) for _, boxes in bb_data]
                print(f"[RESULT] {category}/{vid}:")
                print(f"  - Frames with vehicles: {frames_with_vehicles}/{len(frames)}")
                print(f"  - BB count per frame (with vehicles): {counts}")
                print(f"  - First 3 frames with vehicles: {[name for name, _ in bb_data[:3]]}")
                
                total_new += 1

    print(f"BB extraction complete:")
    print(f"- New caches created: {total_new}")
    print(f"- Total frames with vehicles: {total_frames_with_vehicles}") 