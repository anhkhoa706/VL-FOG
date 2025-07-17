import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def create_scheduler(optimizer, config, train_loader):
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: configuration dictionary
        train_loader: training dataloader (not used in simple version)
        
    Returns:
        scheduler object or None
    """
    scheduler_config = config["training"].get("scheduler", {})
    scheduler_type = scheduler_config.get("type", None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type == "step":
        # StepLR: Decays learning rate by gamma every step_size epochs
        step_size = int(scheduler_config.get("step_size", 10))
        gamma = float(scheduler_config.get("gamma", 0.5))
        
        # Validation
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if gamma <= 0 or gamma > 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
            
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',  # We want to maximize our score
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("step_size", 5)),
            threshold=float(scheduler_config.get("threshold", 1e-4)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    
    return None 