"""Offline checks for the two submitted Henry checkpoints; no real reevaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
RUNS={
    "dinov2_small_henry-liyingjie": "last6_ls01_epoch16",
    "dinov3_vitl16_henry-liyingjie": "last1_ls005_epoch18",
}

def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

HAS_RUNTIME=all(importlib.util.find_spec(name) for name in
                ["torch","timm","transformers","safetensors"])


class HenryEvidenceTests(unittest.TestCase):
    def test_dinov2_stays_an_explicitly_incomplete_archive(self):
        key="dinov2_small_henry-liyingjie"
        folder=ROOT/"results"/key/RUNS[key]
        metrics=read(folder/"metrics.json")
        status=read(folder/"status.json")
        self.assertEqual(metrics["accuracy"],0.9124)
        self.assertIsNone(metrics["confusion_matrix"])
        self.assertEqual(status["state"],"archived")
        self.assertIn("incomplete",status["evaluation_evidence"])
        self.assertFalse(read(folder/"environment.json")["reevaluated"])
        self.assertEqual(digest(ROOT/metrics["checkpoint"]),metrics["checkpoint_sha256"])

    def test_dinov3_matrix_and_recorded_sources_agree(self):
        key="dinov3_vitl16_henry-liyingjie"
        folder=ROOT/"results"/key/RUNS[key]
        metrics=read(folder/"metrics.json")
        mapping=read(folder/"class_to_idx.json")
        matrix=metrics["confusion_matrix"]
        self.assertEqual(metrics["split"],"test")
        self.assertEqual(metrics["samples"],10000)
        self.assertEqual(sorted(mapping.values()),list(range(100)))
        self.assertEqual(len(matrix),100)
        self.assertTrue(all(len(row)==100 for row in matrix))
        self.assertTrue(all(type(v) is int and v>=0 for row in matrix for v in row))
        self.assertEqual(sum(map(sum,matrix)),10000)
        self.assertEqual(sum(matrix[i][i] for i in range(100))/10000,metrics["accuracy"])
        self.assertEqual(metrics["accuracy"],0.9366)
        self.assertGreater(metrics["cross_entropy_loss"],0)
        self.assertEqual(digest(ROOT/metrics["checkpoint"]),metrics["checkpoint_sha256"])
        environment=read(folder/"environment.json")
        self.assertEqual(digest(ROOT/"models"/(key+".py")),environment["model_source_sha256"])
        self.assertEqual(digest(ROOT/"pipeline/henry_submission.py"),environment["evaluation_source_sha256"])
        self.assertIn("unsmoothed cross entropy",metrics["evaluation"])
        self.assertEqual(read(folder/"status.json")["state"],"complete")


@unittest.skipUnless(HAS_RUNTIME,"Install runtime dependencies for checkpoint loading checks")
class HenryLoadingTests(unittest.TestCase):
    def test_original_shard_manifests_and_configs(self):
        import torch
        for key,run in RUNS.items():
            with self.subTest(model=key):
                folder=ROOT/"checkpoints"/key/run
                saved=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
                self.assertEqual(saved["model_key"],key)
                self.assertEqual(saved["checkpoint_format"],"lossless_shards_v1")
                self.assertEqual(saved["config"],read(ROOT/"results"/key/run/"config_snapshot.json"))
                self.assertEqual(saved["class_to_idx"],read(ROOT/"results"/key/run/"class_to_idx.json"))
                self.assertEqual({entry["file"] for entry in saved["shards"]},
                                 {p.name for p in folder.glob("weights-*.pt")})
                self.assertTrue(saved["shards"])
                for entry in saved["shards"]:
                    self.assertEqual(Path(entry["file"]).name,entry["file"])
                    self.assertEqual(digest(folder/entry["file"]),entry["sha256"])

    def test_tiny_full_and_partial_reload_freezing_and_corrupt_shards(self):
        import torch
        from torch import nn
        from safetensors.torch import save_file
        from transformers import DINOv3ViTConfig
        from pipeline.model_api import load_adapter

        torch.set_num_threads(1)

        class TinyDINOv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks=nn.ModuleList([
                    nn.Sequential(nn.Linear(4,4),nn.Dropout(0.1)),
                    nn.Sequential(nn.Linear(4,4),nn.Dropout(0.1)),
                ])
                self.head=nn.Linear(4,2)
            def forward(self,x):
                for block in self.blocks:
                    x=block(x)
                return self.head(x)

        def export(folder,adapter,model,config,storage,state):
            entries=list(state.items())
            cut=max(1,len(entries)//2)
            shards=[]
            for i,part in enumerate([entries[:cut],entries[cut:]],1):
                if not part:
                    continue
                path=folder/f"weights-{i:02}.pt"
                torch.save(dict(part),path)
                shards.append({"file":path.name,"sha256":adapter.digest(path)})
            final=folder/"final.pt"
            torch.save({"model_key":config["model_key"],"config":config,
                        "class_to_idx":{"a":0,"b":1},
                        "checkpoint_format":"lossless_shards_v1",
                        "storage":storage,"shards":shards},final)
            return final,folder/shards[0]["file"]

        with tempfile.TemporaryDirectory() as temporary:
            for key in RUNS:
                with self.subTest(model=key),patch("timm.create_model",side_effect=lambda *a,**k:TinyDINOv2()):
                    folder=Path(temporary)/key
                    folder.mkdir()
                    v3="dinov3" in key
                    settings={"family":"dinov3" if v3 else "dinov2_timm",
                              "trainable_last_blocks":1}
                    if v3:
                        settings["architecture"]=DINOv3ViTConfig(
                            hidden_size=8,num_hidden_layers=2,num_attention_heads=2,
                            intermediate_size=16,image_size=8,patch_size=4,
                            num_register_tokens=2).to_dict()
                    else:
                        settings["timm_name"]="tiny_offline_fixture"
                    config={"model_key":key,"data":{"num_classes":2},"model":settings}
                    adapter=load_adapter(ROOT,key)
                    model=adapter.build_model(config)
                    if v3:
                        base=folder/"backbone"
                        base.mkdir()
                        save_file(model.backbone.state_dict(),base/"model.safetensors")
                        settings.update(pretrained_dir=str(base),
                                        pretrained_sha256=adapter.digest(base/"model.safetensors"))
                    model.train()
                    for index,layer in enumerate(model.layers):
                        self.assertEqual(layer.training,index==len(model.layers)-1)
                        self.assertTrue(all(p.requires_grad==(index==len(model.layers)-1)
                                            for p in layer.parameters()))
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.requires_grad:
                                parameter.add_(0.25)
                    model.eval()
                    inputs=torch.randn(2,3,8,8) if v3 else torch.randn(2,4)
                    with torch.inference_mode():
                        expected=adapter.predict(model,inputs)
                    names={name for name,p in model.named_parameters() if p.requires_grad}
                    state={name:value for name,value in model.state_dict().items()
                           if not v3 or name in names}
                    final,first=export(folder,adapter,model,config,"partial" if v3 else "full",state)
                    restored=adapter.load_weights(adapter.build_model(config),final,device="cpu")
                    with torch.inference_mode():
                        torch.testing.assert_close(adapter.predict(restored,inputs),expected)
                    with first.open("ab") as stream:
                        stream.write(b"corrupt")
                    with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                        adapter.load_weights(adapter.build_model(config),final,device="cpu")


if __name__=="__main__":
    unittest.main()
