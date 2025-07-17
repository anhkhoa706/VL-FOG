import torch
from tqdm import tqdm
from .metrics import calculate_metrics, find_best_threshold_weighted

def train_one_epoch(model, dataloader, optimizer, criterion, config):
    """
    Train the model for a single epoch.
    Args:
        model: the model to train
        dataloader: DataLoader with training samples
        optimizer: optimizer (e.g., Adam)
        criterion: loss function
        config: configuration dictionary

    Returns:
        Average training loss over the epoch
    """
    model.train()
    total_loss = 0
    device = config["training"]["device"]

    for frames, label, clip_id, flow_feats in tqdm(dataloader, desc="Training"):
        frames = frames.to(device)              # [B, T, C, H, W]
        label = label.to(device)
        flow_feats = flow_feats.to(device) 
        # [B, flow_bins]
        
        is_road_flag = torch.tensor([1 if s.startswith("road") else 0 for s in clip_id],
            dtype=torch.long,
            device=device)
        
        output = model(frames, flow_feats, is_road_flag)      # Forward pass
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, config, log_file):
    """
    Evaluate the model on validation set and compute metrics.
    Args:
        model: the model to evaluate
        dataloader: DataLoader with validation samples
        config: configuration dictionary
        log_file: file handle for logging
    
    Returns:
        (final_score, acc, f1, auc, targets, preds)
    """
    model.eval()
    targets, probs = [], []
    device = config["training"]["device"]

    with torch.no_grad():
        for frames, label, clip_id, flow_feats in tqdm(dataloader, desc="Evaluating"):
            frames = frames.to(device)
            label = label.to(device)
            flow_feats = flow_feats.to(device)

            is_road_flag = torch.tensor([1 if s.startswith("road") else 0 for s in clip_id],
            dtype=torch.long,
            device=device)

            output = model(frames, flow_feats, is_road_flag)
            batch_probs = torch.softmax(output, dim=1)[:, 1]  # probability for class 1

            targets.extend(label.cpu().tolist())
            probs.extend(batch_probs.cpu().tolist())

    # Find best threshold for final score
    best_threshold = find_best_threshold_weighted(targets, probs)

    # Apply threshold to compute final predictions
    preds = [1 if p > best_threshold else 0 for p in probs]

    final_score, acc, f1, auc = calculate_metrics(targets, preds, probs, log_file)
    
    return final_score, best_threshold, acc, f1, auc, targets, preds

def train_epoch(model, train_loader, val_loader, optimizer, criterion, scheduler, config, log_file, epoch, writer):
    """Train and evaluate one epoch."""
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from .board import log_model_weights_and_gradients
    from .log import log
    
    current_lr = optimizer.param_groups[0]['lr']
    log(f"\nEpoch {epoch+1}/{config['training']['epochs']} | LR: {current_lr:.2e}", log_file)

    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config)
    
    # Evaluation - always get predictions since computation is the same
    score, threshold, acc, f1, auc, y_true, y_pred = evaluate(model, val_loader, config, log_file)
    
    log(f"Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Threshold: {threshold:.2f} | Score: {score:.4f}", log_file)

    # TensorBoard logging
    if writer is not None:
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Validation/Accuracy', acc, epoch)
        writer.add_scalar('Validation/F1_Score', f1, epoch)
        writer.add_scalar('Validation/AUC', auc, epoch)
        writer.add_scalar('Validation/Final_Score', score, epoch)
        
        # Log model weights periodically
        if epoch % 5 == 0:
            log_model_weights_and_gradients(model, writer, epoch)

    # Update scheduler
    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(-score)  # Negative because higher is better
        else:
            scheduler.step()

    return score, threshold, y_true, y_pred

def save_best_model(model, config, log_file, epoch, score, threshold, model_path, writer, y_true=None, y_pred=None):
    """Save best model and log confusion matrix."""
    from .board import log_confusion_matrix_to_tensorboard
    from .log import log
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_threshold': threshold,  # Save threshold
    }, model_path)
    log(f"✅ New best model saved at epoch {epoch+1}", log_file)
    
    if writer is not None:
        writer.add_scalar('Training/Best_Score', score, epoch)
        
        # Log confusion matrix for best model using provided predictions
        if y_true is not None and y_pred is not None:
            print("📊 Generating confusion matrix...")
            try:
                log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, epoch, 'best_model')
                print(f"📊 Confusion matrix logged for epoch {epoch+1}")
            except Exception as e:
                print(f"⚠️ Could not generate confusion matrix: {e}")
        else:
            print("⚠️ No predictions provided for confusion matrix")
