import os
import cv2
import numpy as np
import pickle
import gzip
import hashlib


# ── CONFIG ─────────────────────────────────────────────────────────────────────
LAST_N = 16  # Number of last frames to process

def load_bb_data(frame_dir, bb_cache_dir=None):
    """
    Load bounding box data from bb_cache using consistent hash key.
    Returns list of (frame_name, boxes) pairs.
    """
    if bb_cache_dir is None:
        bb_cache_dir = os.path.join(frame_dir, '.bb_cache')
    
    category = frame_dir.split(os.sep)[-3]  # 'road' or 'freeway'
    video_name = os.path.basename(frame_dir)

    cache_key = hashlib.md5(f"{category}_{video_name}".encode()).hexdigest()
    bb_path = os.path.join(bb_cache_dir, f"bb_{cache_key}.pkl.gz")

    if not os.path.exists(bb_path):
        raise FileNotFoundError(f"Bounding-box cache not found: {bb_path}")

    with gzip.open(bb_path, 'rb') as f:
        return pickle.load(f)

def sample_last_n_frames(bb_data, last_n=16):
    """
    Sample the last N frames from the bounding box data.
    Skip frames without boxes and ensure we get enough frames with vehicles.
    If there are fewer than N frames with vehicles, use linspace to duplicate frames.
    """
    if not bb_data:
        return []
    
    # Filter out frames without boxes (empty boxes list)
    frames_with_boxes = [(frame_name, boxes) for frame_name, boxes in bb_data if boxes]
    
    if not frames_with_boxes:
        return []
    
    if len(frames_with_boxes) >= last_n:
        # If we have N or more frames with boxes, take the last N
        return frames_with_boxes[-last_n:]
    else:
        # If we have fewer than N frames with boxes, use linspace to duplicate
        num_frames = len(frames_with_boxes)
        if num_frames == 0:
            return []
        
        # Create indices using linspace to evenly distribute the frames
        indices = np.linspace(0, num_frames - 1, last_n, dtype=int)
        
        # Map indices to frames, handling edge cases
        result = []
        for idx in indices:
            if idx < num_frames:
                result.append(frames_with_boxes[idx])
            else:
                # Fallback to last frame if index is out of bounds
                result.append(frames_with_boxes[-1])
        
        return result

def get_frame_pairs(frame_dir, bb_data, skip=3):
    """
    Get frame pairs that are 'skip' frames apart in the sampled sequence.
    Only includes frames that have vehicles detected.
    """
    # Sample the last N frames (only frames with boxes)
    sampled_bb_data = sample_last_n_frames(bb_data, LAST_N)
    
    if not sampled_bb_data:
        return []
    
    frame_names = [frame_name for frame_name, _ in sampled_bb_data]
    frame_boxes = [boxes for _, boxes in sampled_bb_data]
    
    # Create pairs of frames that are 'skip' frames apart in the sampled sequence
    pairs = []
    for i in range(len(frame_names) - skip):
        curr_frame = frame_names[i]
        next_frame = frame_names[i + skip]
        
        pairs.append({
            'prev_frame': os.path.join(frame_dir, curr_frame),
            'next_frame': os.path.join(frame_dir, next_frame),
            'boxes': frame_boxes[i + skip]  # Use boxes from the later frame
        })
    
    return pairs

def compute_optical_flow_histograms(frame_pairs, bins=16, flow_range=(0, 20), top_k=3):
    """
    Compute optical flow histogram using top-K most active boxes (ranked by std + mean).
    Only processes frames that have vehicles detected.
    """
    frame_level_hists = []

    for pair in frame_pairs:
        prev = cv2.imread(pair['prev_frame'], cv2.IMREAD_GRAYSCALE)
        nxt = cv2.imread(pair['next_frame'], cv2.IMREAD_GRAYSCALE)
        boxes = pair['boxes']

        if prev is None or nxt is None:
            print(f"Warning: Could not read frames: {pair['prev_frame']} or {pair['next_frame']}")
            continue

        box_hists = []
        box_scores = []

        for (x1, y1, x2, y2) in boxes:
            crop1 = prev[y1:y2, x1:x2]
            crop2 = nxt[y1:y2, x1:x2]

            if crop1.size == 0 or crop2.size == 0:
                continue

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                crop1, crop2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Compute histogram and motion score
            hist, _ = np.histogram(mag.flatten(), bins=bins, range=flow_range, density=True)
            motion_score = np.mean(mag) + np.std(mag)

            box_hists.append(hist)
            box_scores.append(motion_score)

        # Top-K selection per frame
        if box_hists:
            k = min(top_k, len(box_scores))
            top_indices = np.argsort(box_scores)[-k:]
            top_hists = [box_hists[j] for j in top_indices]
            frame_avg_hist = np.mean(top_hists, axis=0)
            frame_level_hists.append(frame_avg_hist)

    # Final video-level histogram
    if frame_level_hists:
        video_hist = np.mean(frame_level_hists, axis=0)
        return video_hist
    else:
        raise RuntimeError("No valid frame pairs found to compute optical flow histogram.")

def get_optical_flow_feature_bb(frame_dir, bins=16, flow_range=(0, 20), skip=3, bb_cache_dir=None):
    """
    Wrapper function that handles loading frames with vehicles and computing optical flow.
    Only processes the last N frames that have vehicles detected.
    """
    # Load bounding box data
    bb_data = load_bb_data(frame_dir, bb_cache_dir)
    if not bb_data:
        raise ValueError(f"No frames with vehicles found in {frame_dir}")

    # Get valid frame pairs from sampled frames
    frame_pairs = get_frame_pairs(frame_dir, bb_data, skip=skip)
    if not frame_pairs:
        raise ValueError(f"No valid frame pairs found in {frame_dir} with skip={skip}")

    # Compute optical flow histograms
    return compute_optical_flow_histograms(frame_pairs, bins=bins, flow_range=flow_range)

def get_optical_flow_feature(frame_paths, bins=16, flow_range=(0, 20)):
    """
    Compute optical flow feature for a list of frame paths.
    """
    flows = []
    for i in range(len(frame_paths) - 3):
        prev = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
        next = cv2.imread(frame_paths[i + 3], cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(mag)
    all_mag = np.concatenate([m.flatten() for m in flows])
    hist, _ = np.histogram(all_mag, bins=bins, range=flow_range, density=True)
    return hist