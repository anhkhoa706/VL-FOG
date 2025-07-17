import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(dataloader, config):
    """
    Calculate class weights for imbalanced dataset.
    Args:
        dataloader: DataLoader to analyze
        config: configuration dictionary
    Returns:
        torch.Tensor: class weights or None if not using class weights
    """
    class_weights_setting = config["training"].get("class_weights", None)
    
    if class_weights_setting is None:
        return None
    
    if isinstance(class_weights_setting, list):
        # Manual class weights provided
        return torch.tensor(class_weights_setting, dtype=torch.float32)
    
    if class_weights_setting == "auto":
        # Calculate class weights automatically
        all_labels = []
        for _, labels, _, _ in dataloader:
            all_labels.extend(labels.tolist())
        
        unique_classes = np.unique(all_labels)
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=all_labels
        )
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    return None 