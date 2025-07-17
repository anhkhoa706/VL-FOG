import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (optional, can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 