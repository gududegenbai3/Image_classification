from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pipeline.config import ROOT, load_config, repo_path
from pipeline.data import scan_split
from pipeline.metrics import ClassificationMetrics
from pipeline.model_api import load_adapter
from pipeline.runner import check_checkpoint, reserve_run
from tools.sync_docs import synchronize


class MetricsTests(unittest.TestCase):
    def test_unequal_batches_use_sample_weighted_loss_and_true_rows(self):
        scores = ClassificationMetrics(2)
        scores.update([0, 0, 0], [0, 1, 1], 3.0)
        scores.update([1], [1], 2.0)
        self.assertEqual(scores.compute(), {
            "samples": 4, "accuracy": 0.5, "cross_entropy_loss": 1.25,
            "confusion_matrix": [[1, 2], [0, 1]],
        })

    def test_invalid_scores_are_rejected(self):
        scores = ClassificationMetrics(2)
        with self.assertRaises(ValueError):
            scores.compute()
        with self.assertRaises(ValueError):
            scores.update([0], [2], 1.0)
        with self.assertRaises(ValueError):
            scores.update([0], [0], float("nan"))
        self.assertEqual(scores.samples, 0)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/dinov2_small_gududegenbai3.json").read_text())
        (self.root / "models").mkdir()
        self.model_path = self.root / "models" / f"{self.config['model_key']}.py"
        self.model_path.write_text("raise RuntimeError('validation must not import submissions')\n")
        self.config_path = self.root / f"{self.config['model_key']}.json"

    def load(self):
        self.config_path.write_text(json.dumps(self.config))
        return load_config(self.config_path, root=self.root)

    def test_validation_does_not_execute_submission(self):
        self.assertEqual(self.load()["model_key"], self.config["model_key"])

    def test_identity_and_unsafe_paths_are_rejected(self):
        self.config["github_username"] = "another-user"
        with self.assertRaises(ValueError):
            self.load()
        with self.assertRaises(ValueError):
            repo_path(self.root, "../other-model/final.pt")

    def test_existing_experiments_are_never_overwritten(self):
        results, checkpoints = reserve_run(self.config, self.root, training=True)
        marker = results / "metrics.json"
        marker.write_text('{"original": true}')
        with self.assertRaises(FileExistsError):
            reserve_run(self.config, self.root, training=True)
        self.assertEqual(marker.read_text(), '{"original": true}')
        self.assertTrue(checkpoints.is_dir())

    def test_reversed_class_mapping_is_rejected(self):
        for split, classes in [("train", ["apple", "pear"]), ("test", ["apple", "orange"])]:
            for name in classes:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                (folder / "sample.png").touch()
        _, mapping = scan_split(self.root / "train")
        with self.assertRaises(ValueError):
            scan_split(self.root / "test", mapping)

    def test_foreign_weights_and_incompatible_config_are_rejected(self):
        mapping = {"apple": 0, "pear": 1}
        checkpoint = {"model_key": self.config["model_key"], "class_to_idx": mapping,
                      "config": deepcopy(self.config)}
        check_checkpoint(checkpoint, self.config, mapping)
        checkpoint["config"]["model"]["revision"] = "different-revision"
        with self.assertRaises(ValueError):
            check_checkpoint(checkpoint, self.config, mapping)
        checkpoint["model_key"] = "another-model"
        with self.assertRaises(ValueError):
            check_checkpoint(checkpoint, self.config, mapping)

    def test_hyphenated_usernames_load_by_file_and_bad_signatures_fail(self):
        key = "dinov2_small_some-user"
        path = self.root / "models" / f"{key}.py"
        path.write_text("def build_model(config): return config\n"
                        "def load_weights(model, checkpoint_path, *, device): return model\n"
                        "def predict(model, inputs): return inputs\n")
        adapter = load_adapter(self.root, key)
        self.assertEqual(adapter.predict(None, [1, 2]), [1, 2])
        path.write_text("def build_model(config): return config\n"
                        "def load_weights(model): return model\n"
                        "def predict(model, inputs): return inputs\n")
        with self.assertRaises(ValueError):
            load_adapter(self.root, key)


class DocumentationTests(unittest.TestCase):
    def test_precommit_updates_docs_then_requires_review_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tools").mkdir()
            shutil.copyfile(ROOT / "tools/sync_docs.py", root / "tools/sync_docs.py")
            protected = "<!-- OWNER_ONLY_RULES_START -->\nKeep these rules.\n<!-- OWNER_ONLY_RULES_END -->"
            for name in ["AGENTS.md", "README.md"]:
                (root / name).write_text(protected + "\n<!-- PROJECT_STRUCTURE_START -->\nold\n<!-- PROJECT_STRUCTURE_END -->\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            result = subprocess.run(["sh", str(ROOT / ".githooks/pre-commit")], cwd=root,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Keep these rules.", (root / "AGENTS.md").read_text())
            subprocess.run(["git", "-C", str(root), "add", "README.md", "AGENTS.md"], check=True)
            result = subprocess.run(["sh", str(ROOT / ".githooks/pre-commit")], cwd=root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_structure_sync_preserves_owner_rules_and_can_use_git_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            protected = "<!-- OWNER_ONLY_RULES_START -->\nDo not change me.\n<!-- OWNER_ONLY_RULES_END -->"
            for name in ["AGENTS.md", "README.md"]:
                (root / name).write_text(protected + "\n<!-- PROJECT_STRUCTURE_START -->\nold\n<!-- PROJECT_STRUCTURE_END -->\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            (root / "new_model.py").touch()
            synchronize(root, staged=True)
            self.assertNotIn("new_model.py", (root / "README.md").read_text())
            synchronize(root)
            self.assertIn("new_model.py", (root / "README.md").read_text())
            self.assertIn(protected, (root / "AGENTS.md").read_text())
            self.assertEqual(synchronize(root, check=True), [])


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ["torch", "transformers", "PIL"])


@unittest.skipUnless(HAS_RUNTIME, "Install requirements.txt for the offline DINOv2 integration test")
class OfflineTrainingTests(unittest.TestCase):
    def test_train_reload_evaluate_and_prevent_overwrite(self):
        import torch
        from PIL import Image
        from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
        from pipeline.model_api import load_adapter
        from pipeline.runner import run

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "configs/dinov2_small_gududegenbai3.json").read_text())
            config["data"]["num_classes"] = 2
            config["training"].update(epochs=1, batch_size=3)
            config["runtime"]["device"] = "cpu"
            config["model"]["pretrained_dir"] = ".cache/tiny-dinov2"
            pretrained = root / config["model"]["pretrained_dir"]
            pretrained.mkdir(parents=True)
            architecture = Dinov2Config(hidden_size=8, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=16,
                                        image_size=8, patch_size=4)
            Dinov2Model(architecture).save_pretrained(pretrained)
            BitImageProcessor(size={"shortest_edge": 8},
                              crop_size={"height": 8, "width": 8}).save_pretrained(pretrained)
            (root / "models").mkdir()
            key = config["model_key"]
            shutil.copyfile(ROOT / "models" / f"{key}.py", root / "models" / f"{key}.py")
            for split in ["train", "test"]:
                for name, color in [("apple", "red"), ("pear", "green")]:
                    folder = root / "dataset" / split / name
                    folder.mkdir(parents=True)
                    for index in range(2):
                        Image.new("RGB", (8, 8), color).save(folder / f"{index}.png")
            scores = run(config, root, training=True)
            result = root / config["output"]["result_dir"] / config["run_id"]
            self.assertEqual(scores["samples"], 4)
            self.assertEqual(sum(map(sum, scores["confusion_matrix"])), 4)
            self.assertEqual(json.loads((result / "status.json").read_text())["state"], "complete")
            self.assertTrue((result / "environment.json").is_file())
            checkpoint = f"{config['output']['checkpoint_dir']}/{config['run_id']}/final.pt"
            config["run_id"] = "eval01"
            reevaluated = run(config, root, training=False, checkpoint_path=checkpoint)
            for name in ["accuracy", "cross_entropy_loss", "confusion_matrix"]:
                self.assertEqual(scores[name], reevaluated[name])
            with self.assertRaises(FileExistsError):
                run(config, root, training=False, checkpoint_path=checkpoint)
            adapter = load_adapter(root, key)
            resolved = deepcopy(config)
            resolved["model"]["pretrained_dir"] = str(pretrained)
            model = adapter.build_model(resolved)
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertTrue(model.head.training)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
