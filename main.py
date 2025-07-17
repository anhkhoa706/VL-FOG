"""
Main training script for CLIPFusionNet model.
"""

import os
import warnings
import argparse

# Set environment variables and suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow processor.*")

# Project imports
from utils.config import load_config
from utils.setup import setup_directories, initialize_training
from utils.log import log_training_info, log
from utils.training import train_epoch, save_best_model
from utils.board import setup_tensorboard, log_training_config
from utils.seed import set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CLIPFusionNet model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    set_seed(21)
    
    run_dir, log_path, model_path, tensorboard_dir = setup_directories(config)
    model, train_loader, val_loader, optimizer, criterion, scheduler, class_weights = initialize_training(config)
    writer = setup_tensorboard(config, tensorboard_dir)
    
    # Training variables
    best_score = -1
    patience = config["training"]["patience"]
    wait = 0
    
    try:
        with open(log_path, "w") as log_file:
            # Log training message from config
            if "training_message" in config:
                log(config["training_message"], log_file)
                log("="*60, log_file)
            
            # Log setup info
            log_training_info(log_file, train_loader, val_loader, class_weights, config, scheduler)
            
            if writer is not None:
                log_training_config(writer, config)
            
            # Training loop
            for epoch in range(config["training"]["epochs"]):
                # Get evaluation results including predictions
                score, threshold, y_true, y_pred = train_epoch(model, train_loader, val_loader, optimizer, criterion, 
                                                   scheduler, config, log_file, epoch, writer)
                
                # Check for improvement
                if score > best_score:
                    best_score = score
                    wait = 0
                    save_best_model(model, config, log_file, epoch, score, threshold, model_path, writer, y_true, y_pred)
                else:
                    wait += 1
                    log(f"⚠️ No improvement. Patience: {wait}/{patience}", log_file)

                # Early stopping
                if wait >= patience:
                    log("⏹️ Early stopping triggered.", log_file)
                    break

            # Training complete
            log(f"\n🎉 Training complete. Best model: {model_path}", log_file)
            print(f"📊 TensorBoard logs: {tensorboard_dir}")
            
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()