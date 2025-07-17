import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.model_loader import load_clip_fusion_model
# from data.dataset import AVADataset
from data.dataset import AVADataset
from data.augmentation import get_video_transform
from utils.config import load_config

def scan_folders(folder_root):
    # Collect folder names only (subdirectories)
    return sorted([
        f for f in os.listdir(folder_root)
        if os.path.isdir(os.path.join(folder_root, f)) and not f.startswith(".")
    ])

def create_dataloader(folder_list, frame_root, config):
    df = pd.DataFrame({"file_name": folder_list, "risk": [0] * len(folder_list)})
    dataset = AVADataset(
        dataframe=df,
        root_dir=frame_root,
        max_frames=config["dataset"]["max_frames"],
        flow_bins=config["dataset"]["flow_bins"],
        transform=get_video_transform(train=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )
    return loader, df

def run_inference(model, loader, best_threshold, config):
    model.eval()
    probs = []
    device = config["training"]["device"]

    with torch.no_grad():
        for frames, _, clip_id, flow_feats in tqdm(loader, desc="Testing"):
            frames = frames.to(device)
            flow_feats = flow_feats.to(device)

            is_road_flag = torch.tensor(
                [1 if s.startswith("road") else 0 for s in clip_id],
                dtype=torch.long,
                device=device
            )

            output = model(frames, flow_feats, is_road_flag)
            batch_probs = torch.softmax(output, dim=1)[:, 1]  # probability for class 1
            probs.extend(batch_probs.cpu().tolist())

    preds = [1 if p > best_threshold else 0 for p in probs]
    return preds

def main():
    config = load_config(path="config/training_config_stepLR.yaml")
    model_path = config["test"]["model_path"]
    output_csv = config["test"]["output_csv"]

    # Load model
    model = load_clip_fusion_model(config)
    checkpoint = torch.load(model_path, map_location=config["training"]["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config["training"]["device"])

    all_results = []

    for domain in ["road", "freeway"]:  # Order: road then freeway
        frame_root = config["test"]["frame_root"][domain]
        folder_list = scan_folders(frame_root)
        loader, df = create_dataloader(folder_list, frame_root, config)
        preds = run_inference(model, loader, checkpoint["best_threshold"], config)
        df["risk"] = preds
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

if __name__ == "__main__":
    main()