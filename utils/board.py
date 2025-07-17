"""
TensorBoard utilities for logging training metrics and model monitoring.
Provides functions for confusion matrix visualization, model weight tracking,
and configuration logging.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns


def setup_tensorboard(config, tensorboard_dir):
    """Setup TensorBoard logging."""
    if not config["training"].get("use_tensorboard", True):
        print("📊 TensorBoard logging disabled")
        return None
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("📊 TensorBoard logging enabled")
    print(f"📊 Logs: {tensorboard_dir}")
    print(f"📊 View: tensorboard --logdir {tensorboard_dir}")
    return writer


def create_confusion_matrix_figure(y_true, y_pred, class_names=None, normalize=False):
    """
    Create a confusion matrix figure for TensorBoard logging.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: List of class names for labeling
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    if class_names is None:
        class_names = ['Normal', 'Risk']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    figure = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    return figure


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, epoch, phase='validation'):
    """
    Log confusion matrix figures to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        y_true: True labels
        y_pred: Predicted labels
        epoch: Current epoch
        phase: Phase name (e.g., 'validation', 'best_model')
    """
    try:
        class_names = ['Normal', 'Risk']
        
        # Create and log normalized confusion matrix
        norm_fig = create_confusion_matrix_figure(y_true, y_pred, class_names, normalize=True)
        writer.add_figure(f'{phase}/Confusion_Matrix_Normalized', norm_fig, epoch)
        plt.close(norm_fig)
        
        # Create and log raw confusion matrix
        raw_fig = create_confusion_matrix_figure(y_true, y_pred, class_names, normalize=False)
        writer.add_figure(f'{phase}/Confusion_Matrix_Raw', raw_fig, epoch)
        plt.close(raw_fig)
        
    except Exception as e:
        print(f"⚠️ Could not log confusion matrix: {e}")


def log_model_weights_and_gradients(model, writer, epoch):
    """
    Log model weights and gradients to TensorBoard for training monitoring.
    
    Args:
        model: PyTorch model
        writer: TensorBoard SummaryWriter
        epoch: Current epoch
    """
    try:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Log weight histograms
                writer.add_histogram(f'Weights/{name}', param.data, epoch)
                # Log gradient histograms
                writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
                # Log gradient norms
                grad_norm = param.grad.data.norm(2)
                writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)
    except Exception as e:
        print(f"⚠️ Could not log model weights/gradients: {e}")


def log_training_config(writer, config):
    """
    Log training configuration to TensorBoard for reference.
    
    Args:
        writer: TensorBoard SummaryWriter
        config: Configuration dictionary
    """
    try:
        writer.add_text('Config/Learning_Rate', str(config["training"]["lr"]))
        writer.add_text('Config/Model_Version', config["model"]["version"])
        writer.add_text('Config/Batch_Size', str(config["dataset"]["batch_size"]))
        writer.add_text('Config/Scheduler_Type', config["training"]["scheduler"]["type"])
        writer.add_text('Config/Patience', str(config["training"]["patience"]))
        
        # Log additional config details if available
        if "label_smoothing" in config["training"]:
            writer.add_text('Config/Label_Smoothing', str(config["training"]["label_smoothing"]))
            
    except Exception as e:
        print(f"⚠️ Could not log config to TensorBoard: {e}") 