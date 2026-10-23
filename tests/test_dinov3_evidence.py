"""Validate measured submission evidence without importing a model runtime."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
KEY = "dinov3_vitl16_yjqsh233"


class DINOv3EvidenceTests(unittest.TestCase):
    def test_predictions_metrics_and_checkpoint_provenance_agree(self):
        folder = ROOT / "results" / KEY
        identity = json.loads((folder / "dataset_identity.json").read_text())
        self.assertEqual(identity["checked_png_images"], 60000)
        self.assertEqual(identity["exact_git_blob_matches"], 60000)
        runs = sorted(folder.glob("*/metrics.json"))
        self.assertEqual(len(runs), 4)
        for path in runs:
            with self.subTest(run=path.parent.name):
                metrics = json.loads(path.read_text())
                read = lambda name: json.loads((path.parent / name).read_text())
                mapping = read("class_to_idx.json")
                self.assertEqual(mapping, {name: i for i, name in enumerate(sorted(mapping))})
                self.assertEqual(len(mapping), 100)
                self.assertEqual(metrics["model_key"], KEY)
                self.assertEqual(metrics["run_id"], path.parent.name)
                predicted = read("predictions.json")
                truth, guesses = predicted["targets"], predicted["predictions"]
                self.assertEqual(len(truth), metrics["samples"])
                self.assertEqual(len(truth), len(guesses))
                matrix = [[0] * 100 for _ in range(100)]
                for actual, guess in zip(truth, guesses):
                    self.assertIs(type(actual), int)
                    self.assertIs(type(guess), int)
                    self.assertTrue(0 <= actual < 100 and 0 <= guess < 100)
                    matrix[actual][guess] += 1
                self.assertEqual(matrix, metrics["confusion_matrix"])
                self.assertEqual(sum(map(sum, matrix)), metrics["samples"])
                self.assertEqual(sum(matrix[i][i] for i in range(100)) / len(truth), metrics["accuracy"])
                self.assertGreaterEqual(metrics["cross_entropy_loss"], 0)
                self.assertAlmostEqual(metrics["cross_entropy_loss"], metrics["original_cross_entropy_loss"], places=5)
                checkpoint = ROOT / metrics["checkpoint"]
                digest = hashlib.sha256()
                with checkpoint.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                self.assertEqual(digest.hexdigest(), metrics["checkpoint_sha256"])
                environment = read("environment.json")
                self.assertEqual(digest.hexdigest(), environment["exported_checkpoint_sha256"])
                self.assertEqual(hashlib.sha256((ROOT / "models" / (KEY + ".py")).read_bytes()).hexdigest(),
                                 environment["model_source_sha256"])
                if "epoch3" in path.parent.name:
                    self.assertEqual(metrics["split"], "validation")
                    self.assertEqual(metrics["samples"], 5000)
                    self.assertEqual(read("status.json")["official_test_evaluation"], "not_run")
                else:
                    self.assertEqual(metrics["split"], "test")
                    self.assertEqual(metrics["samples"], 10000)


if __name__ == "__main__":
    unittest.main()
