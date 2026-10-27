"""Explicit model construction and lossless sharded checkpoint loading."""
from pathlib import Path
import hashlib
import torch
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.settings = config['model']
        self.model_key = config['model_key']
        self.count = config['data']['num_classes']
        self.tail = self.settings['trainable_last_blocks']
        self.family = self.settings['family']
        if self.family == 'dinov3':
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            architecture = DINOv3ViTConfig(**self.settings['architecture'])
            self.backbone = DINOv3ViTModel(architecture)
            self.head = nn.Linear(architecture.hidden_size, self.count)
        elif self.family == 'dinov2_timm':
            import timm
            self.backbone = timm.create_model(self.settings['timm_name'], pretrained=False, num_classes=self.count)
        else:
            raise ValueError('Unsupported model family')
        if not 0 <= self.tail <= len(self.layers):
            raise ValueError('Invalid trainable_last_blocks')
        self.backbone.requires_grad_(False)
        for layer in self.layers[len(self.layers)-self.tail:]:
            layer.requires_grad_(True)
        (self.head if self.family == 'dinov3' else self.backbone.head).requires_grad_(True)
        self.train(False)

    @property
    def layers(self):
        return self.backbone.layer if self.family == 'dinov3' else self.backbone.blocks

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for layer in self.layers[len(self.layers)-self.tail:]:
            layer.train(mode)
        (self.head if self.family == 'dinov3' else self.backbone.head).train(mode)
        return self

    def forward(self, inputs):
        if self.family == 'dinov3':
            return self.head(self.backbone(pixel_values=inputs).last_hidden_state[:, 0])
        return self.backbone(inputs)


def build_model(config):
    return Classifier(config)


def load_weights(model, checkpoint_path, *, device):
    path = Path(checkpoint_path)
    if path.is_dir():
        from safetensors.torch import load_file
        base = path / 'model.safetensors'
        if digest(base) != model.settings['pretrained_sha256']:
            raise ValueError('Pretrained SHA256 mismatch')
        model.backbone.load_state_dict(load_file(str(base)), strict=True)
        return model.to(device).eval()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('model_key') != model.model_key:
        raise ValueError('Checkpoint model identity mismatch')
    original = saved['config']['model']
    for key in ('family', 'trainable_last_blocks', 'architecture', 'timm_name', 'pretrained_sha256'):
        if original.get(key) != model.settings.get(key):
            raise ValueError('Checkpoint configuration mismatch: ' + key)
    mapping = saved['class_to_idx']
    if sorted(mapping.values()) != list(range(model.count)):
        raise ValueError('Invalid class mapping')
    if saved.get('checkpoint_format') == 'lossless_shards_v1':
        state = {}
        for shard in saved['shards']:
            name = shard['file']
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Shard must be a sibling file')
            part = path.parent / name
            if digest(part) != shard['sha256']:
                raise ValueError('Shard SHA256 mismatch: ' + name)
            tensors = torch.load(part, map_location='cpu', weights_only=True)
            if state.keys() & tensors.keys():
                raise ValueError('Duplicate tensor in checkpoint shards')
            state.update(tensors)
        if saved['storage'] == 'partial':
            load_weights(model, Path(model.settings['pretrained_dir']), device='cpu')
            expected = {n for n,p in model.named_parameters() if p.requires_grad}
            if set(state) != expected:
                raise ValueError('Partial checkpoint tensor set mismatch')
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).eval()


def predict(model, inputs):
    return model(inputs)
