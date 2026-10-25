"""Frozen timm ConvNeXt V2 and an independently saved linear classifier."""

from pathlib import Path
import hashlib

import torch
from torch import nn
import timm
from safetensors.torch import load_file


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(config['model']['name'], pretrained=False,
                                          num_classes=0, **config['model'].get('model_kwargs', {}))
        if self.backbone.num_features != config['model']['feature_dim']:
            raise ValueError('Configured feature dimension differs from the model')
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(self.backbone.num_features, config['data']['num_classes'])
        self.pretrained_ready = False
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs):
        with torch.no_grad():
            features = self.backbone(inputs)
        return self.head(features.float())


def build_model(config):
    return Classifier(config)


def load_weights(model, checkpoint_path, *, device):
    path = Path(checkpoint_path)
    if path.is_dir():
        weights = path / 'model.safetensors'
        if sha256(weights) != model.config['model']['pretrained_sha256']:
            raise ValueError('Pretrained weight SHA256 mismatch')
        state = load_file(str(weights), device='cpu')
        # The downloaded model has an ImageNet classifier; retain all feature layers.
        state.pop('head.fc.weight', None)
        state.pop('head.fc.bias', None)
        model.backbone.load_state_dict(state, strict=True)
        model.pretrained_ready = True
    else:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('model_key') != model.config['model_key'] or saved.get('format') != 'convnext_head_v1':
            raise ValueError('Wrong checkpoint identity or format')
        if saved['class_to_idx'] != model.config['class_to_idx']:
            raise ValueError('Checkpoint class mapping differs')
        for key in ['name', 'model_kwargs', 'feature_dim', 'pretrained_sha256', 'revision']:
            if saved['config']['model'].get(key) != model.config['model'].get(key):
                raise ValueError(f'Checkpoint model configuration differs: {key}')
        if not model.pretrained_ready:
            model = load_weights(model, model.config['model']['pretrained_dir'], device='cpu')
        model.head.load_state_dict(saved['head_state_dict'], strict=True)
    return model.to(device).eval()


def predict(model, inputs):
    return model(inputs)
