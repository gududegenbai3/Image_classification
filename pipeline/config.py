"""Validate submissions before importing model code or creating outputs."""

import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def repo_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Expected a repository-relative path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def load_config(path, *, root=ROOT):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    model = config["model"]
    name, user = model["pretrained_name"], config["github_username"]
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", name):
        raise ValueError("pretrained_name must use lowercase letters, numbers and underscores")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", user):
        raise ValueError("Invalid GitHub username")
    key = config["model_key"]
    if key != f"{name}_{user}":
        raise ValueError("model_key must equal <pretrained_name>_<github_username>")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config["run_id"]):
        raise ValueError("run_id must be a single directory name")
    if Path(path).stem != key:
        raise ValueError("Config filename must match model_key")
    if not (root / "models" / f"{key}.py").is_file():
        raise ValueError(f"Missing models/{key}.py")
    for field in ["train_dir", "test_dir"]:
        repo_path(root, config["data"][field])
    if config["data"]["train_dir"] == config["data"]["test_dir"]:
        raise ValueError("Training and test directories must differ")
    count = config["data"]["num_classes"]
    if type(count) is not int or count < 2:
        raise ValueError("num_classes must be an integer >= 2")
    for field in ["epochs", "batch_size"]:
        value = config["training"][field]
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{field} must be a positive integer")
    for field in ["learning_rate", "weight_decay"]:
        value = config["training"][field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid training.{field}")
    if config["training"]["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if config["runtime"].get("dtype") != "float32":
        raise ValueError("This runner supports float32 only")
    if type(model.get("freeze_backbone")) is not bool:
        raise ValueError("model.freeze_backbone must be a boolean")
    if type(config["runtime"]["seed"]) is not int:
        raise ValueError("runtime.seed must be an integer")
    if config["preprocessing"].get("source") != "local_huggingface_processor":
        raise ValueError("Expected local_huggingface_processor preprocessing")
    repo_path(root, model["pretrained_dir"])
    if not isinstance(model["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("A fixed pretrained revision (40-character commit SHA) is required")
    for field, prefix in [("checkpoint_dir", "checkpoints"), ("result_dir", "results")]:
        expected = f"{prefix}/{key}"
        if config["output"][field] != expected:
            raise ValueError(f"output.{field} must be {expected}")
        repo_path(root, expected)
    return config

