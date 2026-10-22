"""Shared train/evaluate functions; model submissions only implement adapters."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time

from .config import repo_path
from .data import ImageSamples, scan_split
from .metrics import ClassificationMetrics
from .model_api import load_adapter


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def inspect_data(config, root):
    train_dir = repo_path(root, config["data"]["train_dir"])
    test_dir = repo_path(root, config["data"]["test_dir"])
    if train_dir == test_dir or train_dir in test_dir.parents or test_dir in train_dir.parents:
        raise ValueError("Training and test data must use separate directories")
    train, mapping = scan_split(train_dir)
    test, _ = scan_split(test_dir, mapping)
    if len(mapping) != config["data"]["num_classes"]:
        raise ValueError("Configured class count differs from dataset")
    return train, test, mapping


def runtime_config(config, root):
    resolved = deepcopy(config)
    resolved["model"]["pretrained_dir"] = str(repo_path(root, config["model"]["pretrained_dir"]))
    return resolved


def choose_device(name):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def reserve_run(config, root, *, training):
    result_dir = repo_path(root, config["output"]["result_dir"]) / config["run_id"]
    checkpoint_dir = repo_path(root, config["output"]["checkpoint_dir"]) / config["run_id"]
    if result_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("run_id already exists; choose a new run_id to preserve results")
    checkpoint_created = False
    try:
        if training:
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_created = True
        result_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        if checkpoint_created:
            checkpoint_dir.rmdir()
        raise
    return result_dir, checkpoint_dir


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment(root, device, config):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for name in ["torch", "transformers", "Pillow", "safetensors"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": str(device), "dependencies": versions,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "model_source_sha256": file_hash(root / "models" / f"{config['model_key']}.py"),
    }


def score(model, adapter, loader, device, num_classes):
    import torch
    from torch.nn import functional as F

    metrics = ClassificationMetrics(num_classes)
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = adapter.predict(model, inputs)
            if tuple(logits.shape) != (targets.shape[0], num_classes):
                raise ValueError("predict must return [batch_size, num_classes] logits")
            loss = F.cross_entropy(logits, targets, reduction="sum")
            metrics.update(targets.tolist(), logits.argmax(1).tolist(), loss.item())
    return metrics.compute()


def check_checkpoint(checkpoint, config, mapping):
    if checkpoint.get("model_key") != config["model_key"]:
        raise ValueError("Checkpoint belongs to another model_key")
    if checkpoint.get("class_to_idx") != mapping:
        raise ValueError("Checkpoint class mapping differs from dataset")
    saved = checkpoint.get("config", {})
    for key in ["model", "preprocessing"]:
        if saved.get(key) != config[key]:
            raise ValueError(f"Checkpoint {key} configuration differs")


def run(config, root, *, training, checkpoint_path=None):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor

    # macOS exposes temporary directories through both /var and /private/var.
    root = root.resolve()
    # Validate data, weights, and interfaces before reserving an output directory.
    train_samples, test_samples, mapping = inspect_data(config, root)
    adapter = load_adapter(root, config["model_key"])
    device = choose_device(config["runtime"]["device"])
    random.seed(config["runtime"]["seed"])
    torch.manual_seed(config["runtime"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["runtime"]["seed"])
    resolved = runtime_config(config, root)
    processor = AutoImageProcessor.from_pretrained(
        resolved["model"]["pretrained_dir"], local_files_only=True, use_fast=False
    )
    model = adapter.build_model(resolved)
    if checkpoint_path is not None:
        weights = repo_path(root, checkpoint_path)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        check_checkpoint(checkpoint, config, mapping)
    else:
        if not training:
            raise ValueError("Evaluation requires --checkpoint")
        weights = repo_path(root, config["model"]["pretrained_dir"])
    model = adapter.load_weights(model, weights, device=device)
    batch_size = config["training"]["batch_size"]
    test_loader = DataLoader(ImageSamples(test_samples, processor), batch_size=batch_size,
                             shuffle=False, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    if training and not params:
        raise ValueError("Model has no trainable parameters")
    result_dir, checkpoint_dir = reserve_run(config, root, training=training)
    history = []
    try:
        write_json(result_dir / "status.json", {"state": "running"})
        write_json(result_dir / "config_snapshot.json", config)
        write_json(result_dir / "class_to_idx.json", mapping)
        info = environment(root, device, config)
        info["input_weights"] = str(weights.relative_to(root))
        if weights.is_file():
            info["input_weights_sha256"] = file_hash(weights)
        write_json(result_dir / "environment.json", info)
        if training:
            train_loader = DataLoader(ImageSamples(train_samples, processor),
                                      batch_size=batch_size, shuffle=True, num_workers=0)
            optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"],
                                          weight_decay=config["training"]["weight_decay"])
            for epoch in range(1, config["training"]["epochs"] + 1):
                start = time.monotonic()
                model.train()
                totals = ClassificationMetrics(len(mapping))
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = adapter.predict(model, inputs)
                    if tuple(logits.shape) != (targets.shape[0], len(mapping)):
                        raise ValueError("predict must return [batch_size, num_classes] logits")
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                    totals.update(targets.tolist(), logits.detach().argmax(1).tolist(),
                                  loss.item() * targets.shape[0])
                values = totals.compute()
                record = {"epoch": epoch, "train_accuracy": values["accuracy"],
                          "train_loss": values["cross_entropy_loss"],
                          "seconds": time.monotonic() - start}
                history.append(record)
                write_json(result_dir / "history.json", history)
                print(json.dumps(record), flush=True)
            weights = checkpoint_dir / "final.pt"
            torch.save({"model_state_dict": model.state_dict(), "model_key": config["model_key"],
                        "run_id": config["run_id"], "class_to_idx": mapping, "config": config}, weights)
        # Evaluate persisted parameters, so scores refer to the actual deliverable.
        model = adapter.load_weights(model, weights, device=device)
        scores = score(model, adapter, test_loader, device, len(mapping))
        scores.update({"model_key": config["model_key"], "run_id": config["run_id"], "split": "test",
                       "checkpoint": str(weights.relative_to(root)),
                       "checkpoint_sha256": file_hash(weights)})
        write_json(result_dir / "metrics.json", scores)
        if not training:
            write_json(result_dir / "history.json", [])
        write_json(result_dir / "status.json", {"state": "complete", "operation": "train" if training else "evaluate"})
        return scores
    except Exception as exc:
        write_json(result_dir / "status.json", {"state": "failed", "error": str(exc)})
        raise

