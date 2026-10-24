"""DINOv3 classifier adapter; architecture, feature recipe and weights are external."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Classifier(nn.Module):
    def __init__(self, architecture, config):
        super().__init__()
        settings = config["model"]
        self.model_key = config["model_key"]
        self.settings = deepcopy(settings)
        self.pretrained_dir = Path(settings["pretrained_dir"])
        self.n_last_blocks = settings.get("n_last_blocks", 1)
        self.trainable_last_blocks = settings.get("trainable_last_blocks", 0)
        self.feature_quantization = settings.get("feature_quantization", "none")
        self.backbone_precision = settings.get("backbone_precision", "float32")
        if self.n_last_blocks not in (1, 4) or self.n_last_blocks > architecture.num_hidden_layers:
            raise ValueError("n_last_blocks must be 1 or 4 and fit the architecture")
        if self.feature_quantization not in ("none", "float16_roundtrip"):
            raise ValueError("Unknown feature quantization recipe")
        if self.backbone_precision not in ("float32", "float16_autocast"):
            raise ValueError("Unknown backbone precision")
        if not 0 <= self.trainable_last_blocks <= architecture.num_hidden_layers:
            raise ValueError("Invalid trainable_last_blocks")
        if settings["freeze_backbone"] != (self.trainable_last_blocks == 0):
            raise ValueError("freeze_backbone must agree with trainable_last_blocks")
        self.backbone = DINOv3ViTModel(architecture)
        self.backbone.requires_grad_(False)
        if self.trainable_last_blocks:
            for layer in self.backbone.layer[-self.trainable_last_blocks:]:
                layer.requires_grad_(True)
        features = architecture.hidden_size * (self.n_last_blocks + 1)
        self.head = nn.Linear(features, config["data"]["num_classes"])
        self.register_buffer("feature_mean", torch.zeros(features))
        self.register_buffer("feature_std", torch.ones(features))
        self._pretrained_ready = False
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Keep all frozen behavior, especially RoPE coordinate augmentation, off.
        self.backbone.eval()
        if self.trainable_last_blocks:
            for layer in self.backbone.layer[-self.trainable_last_blocks:]:
                layer.train(mode)
        self.head.train(mode)
        return self

    def features(self, inputs):
        context = (torch.autocast("cuda", dtype=torch.float16)
                   if inputs.is_cuda and self.backbone_precision == "float16_autocast"
                   else nullcontext())
        with context:
            tokens = []

            def capture(module, arguments, output):
                tokens.append(self.backbone.norm(output[:, :1])[:, 0].float())

            handles = []
            if self.n_last_blocks > 1:
                handles = [layer.register_forward_hook(capture)
                           for layer in self.backbone.layer[-self.n_last_blocks:]]
            try:
                hidden = self.backbone(pixel_values=inputs).last_hidden_state
            finally:
                for handle in handles:
                    handle.remove()
            if self.n_last_blocks == 1:
                tokens = [hidden[:, 0].float()]
            if len(tokens) != self.n_last_blocks:
                raise RuntimeError("Intermediate-layer capture is incomplete")
            patch_start = 1 + self.backbone.config.num_register_tokens
            patch_mean = hidden[:, patch_start:].float().mean(dim=1)
            result = torch.cat([*tokens, patch_mean], dim=1)
        if self.feature_quantization == "float16_roundtrip":
            # Reproduce the two historical frozen-feature caches, not an int8 model.
            result = result.half().float()
        return result

    def forward(self, inputs):
        features = self.features(inputs)
        with torch.autocast(inputs.device.type, enabled=False):
            return self.head((features - self.feature_mean) / self.feature_std)

    def partial_keys(self):
        return {name for name, parameter in self.named_parameters() if parameter.requires_grad} | {
            "feature_mean", "feature_std"}


def build_model(config):
    """Construct parameters from a local architecture file; never load weights here."""
    architecture = DINOv3ViTConfig.from_pretrained(
        config["model"]["pretrained_dir"], local_files_only=True)
    return Classifier(architecture, config)


def load_weights(model, checkpoint_path, *, device):
    """Load an explicit pretrained directory, compact run state, or full run state."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        weights = checkpoint_path / "model.safetensors"
        expected = model.settings.get("pretrained_sha256")
        if expected and _sha256(weights) != expected:
            raise ValueError("Pretrained weights SHA256 does not match the configuration")
        model.backbone.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        model._pretrained_ready = True
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_key") != model.model_key:
            raise ValueError("Checkpoint belongs to another model_key")
        saved = checkpoint.get("config", {}).get("model", {})
        for name in ("n_last_blocks", "trainable_last_blocks", "feature_quantization", "backbone_precision"):
            if saved.get(name) != model.settings.get(name):
                raise ValueError(f"Checkpoint feature/precision configuration differs: {name}")
        state = checkpoint["model_state_dict"]
        if checkpoint.get("checkpoint_format") == "dinov3_external_backbone_partial_v1":
            if set(state) != model.partial_keys():
                raise ValueError("Compact checkpoint does not contain precisely the expected parameters")
            if not model._pretrained_ready:
                model = load_weights(model, model.pretrained_dir, device="cpu")
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=True)
            model._pretrained_ready = True
        if not torch.isfinite(model.feature_std).all() or not (model.feature_std > 0).all():
            raise ValueError("Feature standard deviations must be finite and positive")
    return model.to(device).eval()


def predict(model, inputs):
    """Return raw [batch_size, num_classes] logits; caller controls gradients."""
    return model(inputs)
