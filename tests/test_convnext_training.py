"""Offline coverage of paired splitting, cached-head training and checkpoint loading."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from pipeline.convnext import ROOT, grouped_split, inspect_data, train, file_hash, write_json

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ['torch', 'torchvision', 'timm', 'safetensors', 'PIL'])


class GroupingTests(unittest.TestCase):
    def test_stratified_original_groups_are_disjoint_and_complete(self):
        labels = [0]*10+[1]*10
        training, validation = grouped_split(labels, 0.2, 42)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(sorted(training+validation), list(range(20)))
        self.assertEqual([labels[i] for i in validation].count(0), 2)
        self.assertEqual([labels[i] for i in validation].count(1), 2)
        rows = training+[i+len(labels) for i in training]
        forbidden = set(validation+[i+len(labels) for i in validation])
        self.assertFalse(set(rows) & forbidden)


@unittest.skipUnless(HAS_RUNTIME, 'Install torch/torchvision/timm for offline integration coverage')
class TrainingTests(unittest.TestCase):
    def test_cached_training_reload_metrics_and_overwrite_protection(self):
        import torch
        import timm
        from PIL import Image
        from safetensors.torch import save_file
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = json.loads((ROOT/'configs/convnextv2_base_gududegenbai3.json').read_text())
            config['model'].update(model_kwargs={'depths': [1,1,1,1], 'dims': [8,16,32,64]}, feature_dim=64)
            config['runtime'].update(device='cpu', workers=0, precision='fp32', feature_batch_size=4)
            config['training'].update(max_epochs=2, head_batch_size=3)
            config['data'].update(num_classes=2, expected_original_samples=8, expected_test_samples=4,
                                  validation_fraction=0.25)
            for split in ['dataset/train', 'dataset/test', 'dataset_mirror/train']:
                for name, color in [('apple','red'), ('pear','green')]:
                    folder = root/split/name
                    folder.mkdir(parents=True)
                    for i in range(2 if 'test' in split else 4):
                        image = Image.new('RGB', (32,32), color)
                        if 'mirror' in split:
                            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        image.save(folder/f'{i}.png')
            original, mirror, test, mapping = inspect_data(config, root)
            key = config['model_key']
            (root/'models').mkdir()
            shutil.copyfile(ROOT/'models'/f'{key}.py', root/'models'/f'{key}.py')
            pretrained = root/'.cache'/key/'local-test'
            pretrained.mkdir(parents=True)
            architecture = timm.create_model(config['model']['name'], pretrained=False,
                                             **config['model']['model_kwargs'])
            save_file(architecture.state_dict(), pretrained/'model.safetensors')
            write_json(pretrained/'config.json', {'test_fixture': True})
            write_json(pretrained.parent/'pretrained.json', {
                'repository': config['model']['repository'], 'revision': 'a'*40,
                'directory': str(pretrained.relative_to(root)),
                'weights_sha256': file_hash(pretrained/'model.safetensors'),
                'config_sha256': file_hash(pretrained/'config.json'),
            })
            metrics = train(config, root, 'offline_test')
            results = root/'results'/key/'offline_test'
            self.assertEqual(metrics['samples'], 4)
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 4)
            self.assertEqual(metrics['accuracy'], sum(metrics['confusion_matrix'][i][i] for i in range(2))/4)
            self.assertEqual(json.loads((results/'status.json').read_text())['state'], 'complete')
            split = json.loads((results/'split.json').read_text())
            self.assertFalse(set(split['selection_train_original_indices']) & set(split['validation_original_indices']))
            self.assertEqual(len(json.loads((results/'history.json').read_text())), metrics['selected_epochs'])
            spec = importlib.util.spec_from_file_location('convnext_offline', root/'models'/f'{key}.py')
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            runtime = json.loads((results/'config_snapshot.json').read_text())
            runtime['model']['pretrained_dir'] = str(pretrained)
            model = adapter.load_weights(adapter.build_model(runtime), root/metrics['checkpoint'], device='cpu')
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertTrue(model.head.training)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            from pipeline.convnext import evaluate
            cached = torch.load(root/'.cache'/'features'/key/'offline_test'/'test.pt', weights_only=True)
            again, _ = evaluate(model.head, cached['features'], cached['labels'], config, torch.device('cpu'))
            self.assertEqual(again['confusion_matrix'], metrics['confusion_matrix'])
            self.assertEqual(again['cross_entropy_loss'], metrics['cross_entropy_loss'])
            with self.assertRaises(FileExistsError):
                train(config, root, 'offline_test')
            (root/config['data']['mirror_dir']/'pear'/'0.png').unlink()
            with self.assertRaises(ValueError):
                inspect_data(config, root)


if __name__ == '__main__':
    unittest.main()
