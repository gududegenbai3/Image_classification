"""Check the imported AutoDL run without training or changing its evidence."""

import hashlib
import json
import math
from pathlib import PurePosixPath
import unittest

from pipeline.convnext import ROOT, grouped_split


KEY = 'convnextv2_base_gududegenbai3'
RUN = 'convnext_20261025_143320'


class ConvNeXtEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = ROOT / 'results' / KEY / RUN

    @classmethod
    def read(cls, name):
        return json.loads((cls.results / name).read_text())

    def test_predictions_reproduce_confusion_matrix_and_accuracy(self):
        metrics = self.read('metrics.json')
        mapping = self.read('class_to_idx.json')
        predictions = self.read('predictions.json')
        samples = self.read('samples.json')
        self.assertEqual(sorted(mapping.values()), list(range(100)))
        self.assertEqual((metrics['model_key'], metrics['run_id'], metrics['split']),
                         (KEY, RUN, 'test'))
        self.assertEqual(metrics['samples'], 10000)
        targets, guesses = predictions['targets'], predictions['predictions']
        self.assertEqual(len(targets), 10000)
        self.assertEqual(len(guesses), 10000)
        self.assertEqual(targets, [mapping[PurePosixPath(p).parent.name] for p in samples['test']])
        matrix = [[0] * 100 for _ in range(100)]
        for target, guess in zip(targets, guesses):
            self.assertIs(type(guess), int)
            self.assertTrue(0 <= guess < 100)
            matrix[target][guess] += 1
        self.assertEqual(matrix, metrics['confusion_matrix'])
        self.assertEqual(sum(map(sum, matrix)), metrics['samples'])
        accuracy = sum(matrix[i][i] for i in range(100)) / metrics['samples']
        self.assertEqual(accuracy, metrics['accuracy'])
        self.assertEqual(accuracy, 0.8844)
        self.assertTrue(math.isfinite(metrics['cross_entropy_loss']))
        self.assertGreater(metrics['cross_entropy_loss'], 0)

    def test_original_group_split_and_selected_epochs(self):
        config = self.read('config_snapshot.json')
        mapping = self.read('class_to_idx.json')
        samples = self.read('samples.json')
        split = self.read('split.json')
        self.assertEqual(len(samples['original']), 50000)
        self.assertEqual(len(samples['mirror']), 50000)
        self.assertEqual([str(PurePosixPath(p).relative_to(config['data']['original_dir']))
                          for p in samples['original']],
                         [str(PurePosixPath(p).relative_to(config['data']['mirror_dir']))
                          for p in samples['mirror']])
        labels = [mapping[PurePosixPath(p).parent.name] for p in samples['original']]
        train, validation = grouped_split(labels, config['data']['validation_fraction'],
                                          config['runtime']['seed'])
        self.assertEqual(train, split['selection_train_original_indices'])
        self.assertEqual(validation, split['validation_original_indices'])
        self.assertEqual((len(train), len(validation)), (45000, 5000))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(sorted(train + validation), list(range(50000)))
        selection = self.read('selection_history.json')
        self.assertEqual(len(selection), config['training']['max_epochs'])
        best = max(selection, key=lambda r: (r['validation']['accuracy'],
                                             -r['validation']['cross_entropy_loss']))
        self.assertEqual(best['epoch'], 5)
        self.assertEqual(self.read('metrics.json')['selected_epochs'], best['epoch'])
        self.assertEqual(len(self.read('history.json')), best['epoch'])
        self.assertEqual(self.read('status.json'), {'state': 'complete', 'selected_epochs': 5})

    def test_checkpoint_and_original_source_hashes(self):
        metrics = self.read('metrics.json')
        environment = self.read('environment.json')
        provenance = self.results / 'provenance'
        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest(ROOT / metrics['checkpoint']), metrics['checkpoint_sha256'])
        self.assertEqual(digest(provenance / f'{KEY}.py.txt'), environment['model_source_sha256'])
        self.assertEqual(digest(ROOT / 'models' / f'{KEY}.py'), environment['model_source_sha256'])
        self.assertEqual(digest(provenance / 'convenext/train_head.py.txt'), environment['runner_sha256'])
        pretrained = json.loads((provenance / 'pretrained.json').read_text())
        self.assertEqual(pretrained['revision'], self.read('config_snapshot.json')['model']['revision'])
        self.assertEqual(pretrained['weights_sha256'], environment['pretrained_sha256'])
        manifest = json.loads((provenance / 'migration_manifest.json').read_text())
        for record in manifest['files']:
            with self.subTest(path=record['destination']):
                destination = (ROOT / record['destination']).resolve()
                self.assertTrue(destination.is_relative_to(ROOT))
                self.assertEqual(digest(destination), record['sha256'])


if __name__ == '__main__':
    unittest.main()
