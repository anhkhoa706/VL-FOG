import yaml
import torch

# === Load configuration from YAML and determine device === #
def load_config(path="config/training_config_stepLR.yaml"):
    print(f"Loading configuration from {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config