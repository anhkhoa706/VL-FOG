import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 3)
        )

    def forward(self, v, t, f):
        x = torch.cat([v, t, f], dim=1)        # [B, 3*dim]
        gates = torch.sigmoid(self.gate_mlp(x))  # [B, 3]
        weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        fused = weights[:, 0:1] * v + weights[:, 1:2] * t + weights[:, 2:3] * f
        return fused                           # [B, dim]


class ClipFusionNet(nn.Module):
    def __init__(self, 
                 clip_model, 
                 text_prompts, 
                 flow_bins=16,
                 domain_emb_dim=8,
                 hidden_dim=256,
                 num_classes=2,
                 dropout=0.2,
                 max_frames=32,
                 freeze_clip=True):
        super().__init__()
        self.clip_model = clip_model
        self.max_frames = max_frames
        self.text_prompts = text_prompts
        self.flow_bins = flow_bins

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Encode prompts once
        prompt_embed = self._encode_prompts(text_prompts).detach()
        self.register_buffer("prompt_embed", prompt_embed)

        # LSTM temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Flow
        self.flow_processor = nn.Sequential(
            nn.Linear(flow_bins, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Projections
        self.vision_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(len(text_prompts), hidden_dim)
        self.flow_proj = nn.Linear(hidden_dim, hidden_dim)

        # 🔁 Gated fusion
        self.gated_fusion = GatedFusion(hidden_dim)

        # 🛣️ Domain embedding
        self.domain_emb = nn.Embedding(2, domain_emb_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + domain_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        print(f"✅ Initialized with Gated Fusion + Domain Embedding")
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - Text prompts: {len(text_prompts)}")
        print(f"   - Flow bins: {flow_bins}")

    def _encode_prompts(self, prompts):
        processor = self.clip_model.processor
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        device = next(self.clip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_frames(self, images):
        images = images.to(next(self.clip_model.parameters()).device)
        with torch.no_grad():
            feats = self.clip_model.get_image_features(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(self, vision_frames, flow_feats, is_road_flag):
        B, T, C, H, W = vision_frames.shape
        device = next(self.parameters()).device

        # Encode vision
        vision_frames = vision_frames.view(B * T, C, H, W)
        vision_feats = self._encode_frames(vision_frames)
        vision_feats = vision_feats.view(B, T, -1)  # [B, T, 512]

        # LSTM
        lstm_out, _ = self.temporal_lstm(vision_feats)  # [B, T, 512]
        temporal_feats = lstm_out.mean(dim=1)           # [B, 512]
        vision_proj = self.vision_proj(temporal_feats)  # [B, hidden_dim]

        # Text similarity
        prompt_embed = self.prompt_embed.to(device)
        text_sim = torch.matmul(vision_feats, prompt_embed.T)  # [B, T, num_prompts]
        text_feats = text_sim.mean(dim=1)                      # [B, num_prompts]
        text_proj = self.text_proj(text_feats)                 # [B, hidden_dim]

        # Flow
        flow_proj = self.flow_proj(self.flow_processor(flow_feats.to(device)))  # [B, hidden_dim]

        # 🔁 Gated fusion of vision, text, and flow
        fused = self.gated_fusion(vision_proj, text_proj, flow_proj)            # [B, hidden_dim]

        # 🛣️ Append domain embedding
        dom = self.domain_emb(is_road_flag.long().to(device))                   # [B, domain_emb_dim]
        final_feat = torch.cat([fused, dom], dim=1)                             # [B, hidden_dim + domain_emb_dim]

        # Classification
        logits = self.classifier(final_feat)  # [B, num_classes]
        return logits