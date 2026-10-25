"""Validate the AutoDL CUDA stack; never download weights or read datasets."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions-only", action="store_true",
                        help="Check package versions only; does not validate the GPU")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/convnextv2_base_environment.json").read_text())
    expected = {}
    for line in (root / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            requirement = Requirement(line.strip())
            expected[requirement.name] = requirement.specifier
    versions = {name: importlib.metadata.version(name) for name in expected}
    print(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2), flush=True)
    if platform.python_version_tuple()[:2] != tuple(config["python"].split(".")):
        raise RuntimeError(f"Expected the image's Python {config['python']}")
    for name, specifier in expected.items():
        if versions[name] not in specifier:
            raise RuntimeError(f"Expected {name}{specifier}, found {versions[name]}")

    import torch
    import torchvision
    import timm

    if args.versions_only:
        print("Package versions/imports OK; GPU and model execution NOT checked")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: check the GPU instance, host driver and installed torch wheel")
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split('.')[:2])) < (12, 8):
        raise RuntimeError(f"Blackwell needs a supported CUDA build (target >=12.8), found {torch.version.cuda}")
    capability = torch.cuda.get_device_capability(0)
    if list(capability) != config["gpu_compute_capability"]:
        raise RuntimeError(f"GPU capability {capability} differs from the configured RTX PRO 6000 Blackwell target")
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": properties.name, "vram_gib": properties.total_memory / 1024**3,
                      "compute_capability": capability, "compiled_architectures": torch.cuda.get_arch_list(),
                      "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                      "bf16_supported": torch.cuda.is_bf16_supported()}, indent=2), flush=True)

    # Random architecture weights are sufficient for a kernel/gradient smoke test.
    torch.manual_seed(42)
    backbone = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
    backbone.requires_grad_(False).eval().cuda()
    data_config = timm.data.resolve_model_data_config(backbone)
    if list(data_config["input_size"]) != config["input_size"]:
        raise RuntimeError("Model preprocessing input size differs from configs/convnextv2_base_environment.json")
    from PIL import Image
    transform = timm.data.create_transform(**data_config, is_training=False)
    image = Image.new("RGB", (32, 32), (64, 128, 192))
    inputs = transform(image).unsqueeze(0).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=torch.cuda.is_bf16_supported()):
        features = backbone(inputs).float()
    if tuple(features.shape) != (1, config["feature_dim"]):
        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
    head = torch.nn.Linear(config["feature_dim"], config["num_classes"]).cuda()
    logits = head(features)
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device="cuda"))
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                         for p in head.parameters()):
        raise RuntimeError("Non-finite classification head loss/gradients")
    if any(p.grad is not None or p.requires_grad for p in backbone.parameters()):
        raise RuntimeError("Backbone unexpectedly receives gradients")
    # Exercise a torchvision CUDA operator too, detecting mismatched binary wheels.
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 9., 9.]], device="cuda")
    torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "features_shape": list(features.shape),
                      "logits_shape": list(logits.shape), "backbone_frozen": True,
                      "head_gradients_finite": True, "pretrained_weights_loaded": False,
                      "real_training_or_evaluation": "not_run"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError, ValueError) as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
