"""
Setup utilities for training initialization and directory management.
"""

import os
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from models.model_loader import load_clip_fusion_model
from data.data_loader import load_splitted_datasets
from .schedulers import create_scheduler
from .class_weights import calculate_class_weights


def setup_directories(config):
    """Create training directories."""
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir)
    
    log_path = os.path.join(run_dir, "log.txt")
    model_path = os.path.join(run_dir, "best_model.pth")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    return run_dir, log_path, model_path, tensorboard_dir


def initialize_training(config):
    """Initialize model, data, optimizer, loss, and scheduler."""
    # Load data and model
    # train_loader, val_loader = load_combined_datasets(config)
    train_loader, val_loader = load_splitted_datasets(config)
    model = load_clip_fusion_model(config)

    # Setup class weights
    class_weights = calculate_class_weights(train_loader, config)
    device = config["training"]["device"]
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Initialize training components
    optimizer = optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    return model, train_loader, val_loader, optimizer, criterion, scheduler, class_weights 