"""Usage: python -m pipeline {validate,prepare,train,evaluate} --config ..."""

import argparse
import ast
import json
from pathlib import Path

from .config import ROOT, load_config, repo_path
from .runner import inspect_data, run


def main():
    parser = argparse.ArgumentParser(description="图像分类协作训练与评估")
    parser.add_argument("command", choices=["validate", "prepare", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", help="使用新运行目录，不覆盖历史成果")
    parser.add_argument("--checkpoint", help="仓库相对权重路径；evaluate 必填")
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        if args.run_id:
            import re
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
                raise ValueError("Invalid run_id")
            config["run_id"] = args.run_id
        if args.checkpoint and args.command not in {"train", "evaluate"}:
            raise ValueError("--checkpoint only applies to train/evaluate")
        if args.command == "validate":
            train, test, mapping = inspect_data(config, ROOT)
            source = ROOT / "models" / f"{config['model_key']}.py"
            tree = ast.parse(source.read_text(encoding="utf-8"))
            names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            if not {"build_model", "load_weights", "predict"} <= names:
                raise ValueError("Model must define build_model, load_weights, predict")
            print(json.dumps({"model_key": config["model_key"], "classes": len(mapping),
                              "train_samples": len(train), "test_samples": len(test),
                              "validation": "config, paths, class mapping, interface declarations",
                              "runtime_execution": "not performed"}, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            from huggingface_hub import snapshot_download
            destination = repo_path(ROOT, config["model"]["pretrained_dir"])
            snapshot_download(repo_id=config["model"]["model_id"], revision=config["model"]["revision"],
                              local_dir=destination,
                              allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "*.safetensors.index.json"])
            print(f"Prepared local pretrained files: {destination}")
        else:
            if args.command == "evaluate" and not args.checkpoint:
                raise ValueError("evaluate requires --checkpoint")
            scores = run(config, ROOT, training=args.command == "train", checkpoint_path=args.checkpoint)
            print(json.dumps({key: value for key, value in scores.items() if key != "confusion_matrix"}, indent=2))
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
