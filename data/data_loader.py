import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split

from data.dataset import AVADataset
from data.augmentation import get_video_transform

# === Load and combine freeway and road datasets === #
def load_combined_datasets(config):
    train_sets = []
    val_sets = []

    print("📊 Analyzing class distribution for balanced splitting...")
    
    for domain in ["freeway", "road"]:
        csv_path = config["dataset"]["csv_path"][domain]
        frame_root = config["dataset"]["frame_root"][domain]
        df = pd.read_csv(csv_path)

        # Print original distribution
        original_counts = df['risk'].value_counts().sort_index()
        print(f"\n{domain.title()} dataset: {len(df)} samples")
        for class_val, count in original_counts.items():
            percentage = count / len(df) * 100
            print(f"  Class {class_val}: {count} samples ({percentage:.1f}%)")

        # Handle full training mode vs normal split with stratified splitting
        train_split = config["dataset"]["train_split"]
        
        if train_split == 1.0:
            # Full training mode: use 95% for training, 5% for monitoring (stratified)
            print(f"🎯 Full training mode: Using 95% training, 5% validation (stratified)")
            train_df, val_df = train_test_split(
                df, 
                test_size=0.05, 
                random_state=42, 
                stratify=df['risk']
            )
        else:
            # Normal training mode with specified split (stratified)
            val_size = 1.0 - train_split
            train_df, val_df = train_test_split(
                df, 
                test_size=val_size, 
                random_state=42, 
                stratify=df['risk']
            )

        # Reset indices after splitting
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Print stratified distribution
        train_counts = train_df['risk'].value_counts().sort_index()
        val_counts = val_df['risk'].value_counts().sort_index()
        
        print(f"  ✅ Training split ({len(train_df)} samples):")
        for class_val, count in train_counts.items():
            percentage = count / len(train_df) * 100
            print(f"    Class {class_val}: {count} samples ({percentage:.1f}%)")
            
        print(f"  ✅ Validation split ({len(val_df)} samples):")
        for class_val, count in val_counts.items():
            percentage = count / len(val_df) * 100
            print(f"    Class {class_val}: {count} samples ({percentage:.1f}%)")

        train_dataset = AVADataset(
            dataframe=train_df,
            root_dir=frame_root,
            max_frames=config["dataset"]["max_frames"],
            flow_bins=config["dataset"]["flow_bins"],
            transform=get_video_transform(train=True)
        )

        val_dataset = AVADataset(
            dataframe=val_df,
            root_dir=frame_root,
            max_frames=config["dataset"]["max_frames"],
            flow_bins=config["dataset"]["flow_bins"],
            transform=get_video_transform(train=False)
        )

        train_sets.append(train_dataset)
        val_sets.append(val_dataset)

    # Now safely concatenate after correct transforms are assigned
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )

    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    
    # Print final combined distribution summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"📊 Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print("✅ Stratified splitting complete - class balance maintained!")
    
    return train_loader, val_loader 


def load_splitted_datasets(config):
    train_sets = []
    val_sets = []
    all_frame_roots = config["dataset"]["frame_root"]

    print("📊 Analyzing class distribution for balanced splitting...")
    
    train_df = pd.read_csv('data/train_df.csv')
    val_df = pd.read_csv('data/val_df.csv')

    # Print stratified distribution
    train_counts = train_df['risk'].value_counts().sort_index()
    val_counts = val_df['risk'].value_counts().sort_index()
    
    print(f"\n  ✅ Training split ({len(train_df)} samples):")
    for class_val, count in train_counts.items():
        percentage = count / len(train_df) * 100
        print(f"    Class {class_val}: {count} samples ({percentage:.1f}%)")
        
    print(f"\n  ✅ Validation split ({len(val_df)} samples):")
    for class_val, count in val_counts.items():
        percentage = count / len(val_df) * 100
        print(f"    Class {class_val}: {count} samples ({percentage:.1f}%)")

    for domain, frame_root in all_frame_roots.items():
        # Print stratified distribution
        print(f"\n📊 Analyzing class distribution for {domain}...")
        train_counts = train_df[train_df['file_name'].str.contains(domain)]['risk'].value_counts().sort_index()
        val_counts = val_df[val_df['file_name'].str.contains(domain)]['risk'].value_counts().sort_index()
        
        print(f"  - Training split ({train_counts.sum()} samples):")
        for class_val, count in train_counts.items():
            print(f"    Class {class_val}: {count} samples")
            
        print(f"\n  - Validation split ({val_counts.sum()} samples):")
        for class_val, count in val_counts.items():
            print(f"    Class {class_val}: {count} samples")

        train_dataset = AVADataset(
            dataframe=train_df[train_df['file_name'].str.contains(domain)],
            root_dir=frame_root,
            max_frames=config["dataset"]["max_frames"],
            flow_bins=config["dataset"]["flow_bins"],
            transform=get_video_transform(train=True)
        )
        
        val_dataset = AVADataset(
            dataframe=val_df[val_df['file_name'].str.contains(domain)],
            root_dir=frame_root,
            max_frames=config["dataset"]["max_frames"],
            flow_bins=config["dataset"]["flow_bins"],
            transform=get_video_transform(train=False)
        )

        train_sets.append(train_dataset)
        val_sets.append(val_dataset)

    # Now safely concatenate after correct transforms are assigned
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )

    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    
    # Print final combined distribution summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"📊 Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    return train_loader, val_loader 
