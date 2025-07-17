from transformers import CLIPModel, CLIPProcessor
from models.clip_fusion_net import ClipFusionNet
import torch

def load_clip_fusion_model(config):
    """
    Load SimplifiedClipFusionNet based on configuration.
    """
    # Load CLIP model
    clip_model_name = config["clip"]["model"]
    if "local_path" in config["clip"] and config["clip"]["local_path"]:
        clip_model = CLIPModel.from_pretrained(config["clip"]["local_path"])
        processor = CLIPProcessor.from_pretrained(config["clip"]["local_path"])
    else:
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    # Attach processor to model for convenience
    clip_model.processor = processor
    
    print("⚡ Loading SimplifiedClipFusionNet - Fast & Effective!")
    model = ClipFusionNet(
        clip_model=clip_model,
        text_prompts=config["dataset"]["prompts"],
        flow_bins=config["dataset"]["flow_bins"],
        hidden_dim=config.get("model", {}).get("hidden_dim", 256),
        num_classes=config.get("model", {}).get("num_classes", 2),
        dropout=config.get("model", {}).get("dropout", 0.2),
        max_frames=config["dataset"]["max_frames"],
        freeze_clip=config["clip"].get("freeze_clip", True)
    )

    # Move to device
    device_str = config["training"]["device"]
    
    # Try to use specified device, fallback to CPU if not available
    try:
        device = torch.device(device_str)
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    except:
        print("Warning: Specified device not available. Using CPU")
        device = torch.device("cpu")
        config["training"]["device"] = "cpu"

    model = model.to(device)

    # Load pretrained weights if specified
    if "pretrained_path" in config["clip"] and config["clip"]["pretrained_path"]:
        try:
            print(f"Loading pretrained weights from {config['clip']['pretrained_path']}")
            state_dict = torch.load(config["clip"]["pretrained_path"], map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
    
    print("✅ =================Done loading model ===============================")
    return model
