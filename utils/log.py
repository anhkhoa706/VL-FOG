# === Logging helper === #
def log(message, log_file=None):
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def log_training_info(log_file, train_loader, val_loader, class_weights, config, scheduler):
    """Log training setup information."""
    log(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}", log_file)
    
    if class_weights is not None:
        log(f"Class weights: {class_weights.tolist()}", log_file)
    else:
        log("No class weights", log_file)
    
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    log(f"Label smoothing: {label_smoothing}", log_file)
    log(f"Learning rate: {config['training']['lr']}", log_file)
    
    if scheduler is not None:
        scheduler_type = config['training']['scheduler']['type']
        log(f"Scheduler: {scheduler_type}", log_file)