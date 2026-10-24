"""Offline tiny-model checks of compact weights, pooling and frozen behavior."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from pipeline.model_api import load_adapter
from pipeline.config import ROOT

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_RUNTIME, "Install requirements.txt for DINOv3 runtime checks")
class DINOv3SubmissionTests(unittest.TestCase):
    def test_features_compact_reload_and_freezing(self):
        import torch
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        torch.set_num_threads(1)
        adapter = load_adapter(ROOT, "dinov3_vitl16_yjqsh233")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            architecture = DINOv3ViTConfig(hidden_size=8, num_hidden_layers=4,
                                           num_attention_heads=2, intermediate_size=16,
                                           image_size=8, patch_size=4, num_register_tokens=2)
            DINOv3ViTModel(architecture).save_pretrained(root)
            template = {"model_key": "dinov3_vitl16_yjqsh233", "data": {"num_classes": 2},
                        "model": {"pretrained_dir": str(root), "freeze_backbone": True,
                                  "n_last_blocks": 1, "trainable_last_blocks": 0,
                                  "feature_quantization": "none", "backbone_precision": "float32"}}
            pixels = torch.randn(2, 3, 8, 8)
            for layers in (1, 4):
                with self.subTest(layers=layers):
                    config = deepcopy(template)
                    config["model"]["n_last_blocks"] = layers
                    model = adapter.load_weights(adapter.build_model(config), root, device="cpu")
                    with torch.inference_mode():
                        output = model.backbone(pixel_values=pixels, output_hidden_states=True)
                        states = output.hidden_states[-layers:]
                        expected = torch.cat([*(model.backbone.norm(x)[:, 0] for x in states),
                                              output.last_hidden_state[:, 3:].mean(1)], dim=1)
                        torch.testing.assert_close(model.features(pixels), expected)
                        before = adapter.predict(model, pixels)
                    checkpoint = root / f"head_{layers}.pt"
                    torch.save({"model_key": config["model_key"], "config": config,
                                "checkpoint_format": "dinov3_external_backbone_partial_v1",
                                "model_state_dict": {k: v for k, v in model.state_dict().items()
                                                     if k in model.partial_keys()}}, checkpoint)
                    restored = adapter.load_weights(adapter.build_model(config), checkpoint, device="cpu")
                    with torch.inference_mode():
                        torch.testing.assert_close(adapter.predict(restored, pixels), before)
                    restored.train()
                    self.assertFalse(restored.backbone.training)
                    self.assertTrue(restored.head.training)
                    self.assertTrue(all(not p.requires_grad for p in restored.backbone.parameters()))
            config = deepcopy(template)
            config["model"].update(freeze_backbone=False, trainable_last_blocks=2)
            model = adapter.load_weights(adapter.build_model(config), root, device="cpu")
            model.train()
            self.assertFalse(model.backbone.rope_embeddings.training)
            self.assertFalse(model.backbone.norm.training)
            for index, block in enumerate(model.backbone.layer):
                self.assertEqual(block.training, index >= 2)
                self.assertTrue(all(p.requires_grad == (index >= 2) for p in block.parameters()))


if __name__ == "__main__":
    unittest.main()
