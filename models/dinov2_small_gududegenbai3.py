"""DINOv2 CLS features + linear head; all settings/weights are external."""

from pathlib import Path

import torch
from torch import nn
from transformers import Dinov2Config, Dinov2Model


class Classifier(nn.Module):
    def __init__(self, architecture, num_classes, freeze_backbone):
        super().__init__()
        self.backbone = Dinov2Model(architecture)
        self.head = nn.Linear(architecture.hidden_size, num_classes)
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
        features = self.backbone(pixel_values=inputs).last_hidden_state[:, 0]
        return self.head(features)


def build_model(config):
    # Runtime resolves this path before passing the config. No network calls.
    architecture = Dinov2Config.from_pretrained(
        config["model"]["pretrained_dir"], local_files_only=True
    )
    return Classifier(architecture, config["data"]["num_classes"],
                      config["model"]["freeze_backbone"])


def load_weights(model, checkpoint_path, *, device):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        model.backbone = Dinov2Model.from_pretrained(
            checkpoint_path, local_files_only=True, use_safetensors=True
        )
        model.backbone.requires_grad_(not model.freeze_backbone)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def predict(model, inputs):
    return model(inputs)
