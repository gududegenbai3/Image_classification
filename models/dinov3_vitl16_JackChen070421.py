"""DINOv3 ViT-L/16 frozen features + MLP head; config and weights are external.

Feature definition follows the training pipeline used for this submission:
[CLS] and the mean of patch tokens (register tokens excluded), each L2-normalized,
concatenated and L2-normalized again. The checkpoint stores only the 4.2 MB head;
the frozen backbone is restored from config["model"]["pretrained_dir"], which is
pinned by the 40-character revision in the config.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, features):
        return self.fc2(self.drop(self.act(self.fc1(self.norm(features)))))


class Classifier(nn.Module):
    def __init__(self, architecture, num_classes, freeze_backbone, hidden, dropout,
                 pretrained_dir):
        super().__init__()
        self.backbone = AutoModel.from_config(architecture)
        self.feature_dim = architecture.hidden_size * 2
        self.head = MLPHead(self.feature_dim, num_classes, hidden, dropout)
        self.num_register_tokens = int(getattr(architecture, "num_register_tokens", 0) or 0)
        self.pretrained_dir = str(pretrained_dir)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, inputs):
        hidden = self.backbone(pixel_values=inputs).last_hidden_state
        cls_token = hidden[:, 0]
        patch_tokens = (hidden[:, 1 + self.num_register_tokens:]
                        if self.num_register_tokens else hidden[:, 1:])
        features = torch.cat([
            F.normalize(cls_token.float(), dim=-1),
            F.normalize(patch_tokens.mean(dim=1).float(), dim=-1),
        ], dim=-1)
        return self.head(F.normalize(features, dim=-1))


def _backbone_from_dir(directory):
    return AutoModel.from_pretrained(
        str(directory), local_files_only=True, use_safetensors=True
    )


def build_model(config):
    # No network access and no pretrained weights here: the caller prepares them.
    model_config = config["model"]
    architecture = AutoConfig.from_pretrained(
        model_config["pretrained_dir"], local_files_only=True
    )
    return Classifier(
        architecture,
        config["data"]["num_classes"],
        model_config["freeze_backbone"],
        int(model_config.get("head_hidden", 512)),
        float(model_config.get("head_dropout", 0.3)),
        model_config["pretrained_dir"],
    )


def load_weights(model, checkpoint_path, *, device):
    path = Path(checkpoint_path)
    if path.is_dir():
        # Training entry point: a prepared pretrained directory, head stays fresh.
        model.backbone = _backbone_from_dir(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "head_state_dict" in checkpoint:
            # Frozen-backbone submission: only the head is stored in final.pt.
            model.backbone = _backbone_from_dir(model.pretrained_dir)
            model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            raise ValueError("Checkpoint must contain head_state_dict or model_state_dict")
    if model.freeze_backbone:
        model.backbone.requires_grad_(False)
    model.train(False)
    return model.to(device).eval()


def predict(model, inputs):
    return model(inputs)
