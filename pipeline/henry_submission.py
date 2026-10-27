"""Evaluate henry-liyingjie's saved runs with their original PIL preprocessing."""
import argparse
from copy import deepcopy
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from .metrics import ClassificationMetrics

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

class Images(Dataset):
    def __init__(self, samples, preprocessing):
        self.samples=samples
        self.transform=transforms.Compose([
            transforms.Resize(tuple(preprocessing['size']),interpolation=getattr(transforms.InterpolationMode,preprocessing['interpolation'].upper())),
            transforms.ToTensor(),transforms.Normalize(preprocessing['mean'],preprocessing['std'])])
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        path,label=self.samples[i]
        with Image.open(path) as image: image=self.transform(image.convert('RGB'))
        return image,label

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New result directory; existing directories are rejected')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'dataset/test')
    parser.add_argument('--pretrained-dir',type=Path)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    if args.output.exists():raise FileExistsError('Use a new output directory')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['config']!=config:raise ValueError('Configuration differs from checkpoint')
    mapping=checkpoint['class_to_idx']
    if sorted(p.name for p in args.data_dir.iterdir() if p.is_dir())!=sorted(mapping):raise ValueError('Test classes differ')
    samples=[(p,mapping[c]) for c in sorted(mapping) for p in sorted((args.data_dir/c).iterdir()) if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    if len(samples)!=10000:raise ValueError('Expected exactly 10,000 test images')
    module_path=ROOT/'models'/f"{config['model_key']}.py"
    spec=importlib.util.spec_from_file_location('submission_adapter',module_path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    resolved=deepcopy(config)
    resolved['model']['pretrained_dir']=str(args.pretrained_dir or ROOT/config['model']['pretrained_dir'])
    torch.set_num_threads(4)
    model=adapter.build_model(resolved)
    model=adapter.load_weights(model,args.checkpoint,device=args.device)
    loader=DataLoader(Images(samples,config['preprocessing']),batch_size=args.batch_size,shuffle=False,num_workers=0)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'status.json',{'state':'running'})
    metrics=ClassificationMetrics(len(mapping));start=time.monotonic()
    try:
        with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8') as stream,torch.inference_mode():
            writer=csv.writer(stream);writer.writerow(['path','target','prediction','cross_entropy'])
            offset=0
            for inputs,targets in loader:
                logits=adapter.predict(model,inputs.to(args.device))
                loss=F.cross_entropy(logits,targets.to(args.device),reduction='none')
                predictions=logits.argmax(1).cpu().tolist();losses=loss.cpu().tolist()
                metrics.update(targets.tolist(),predictions,sum(losses))
                for j,(target,pred,ce) in enumerate(zip(targets.tolist(),predictions,losses)):
                    writer.writerow([samples[offset+j][0].relative_to(args.data_dir).as_posix(),target,pred,ce])
                offset+=len(targets)
                if offset%800==0 or offset==len(samples):print(f'{config["run_id"]}: {offset}/{len(samples)}, {time.monotonic()-start:.0f}s',flush=True)
        result=metrics.compute();result.update(model_key=config['model_key'],run_id=config['run_id'],split='test',checkpoint=f"checkpoints/{config['model_key']}/{config['run_id']}/final.pt",checkpoint_sha256=adapter.digest(args.checkpoint),evaluation='PIL direct resize, FP32, eval mode, no augmentation, unsmoothed cross entropy')
        write(args.output/'metrics.json',result)
        write(args.output/'class_to_idx.json',mapping)
        write(args.output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'device':args.device,'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,'dependencies':{n:importlib.metadata.version(n) for n in ['torch','torchvision','timm','transformers','Pillow','safetensors']},'model_source_sha256':adapter.digest(module_path),'evaluation_source_sha256':adapter.digest(Path(__file__)),'checkpoint_sha256':adapter.digest(args.checkpoint),'elapsed_seconds':time.monotonic()-start})
        write(args.output/'config_snapshot.json',config)
        write(args.output/'status.json',{'state':'complete','operation':'fixed_checkpoint_evaluation'})
        print(json.dumps({k:v for k,v in result.items() if k!='confusion_matrix'}),flush=True)
    except Exception as error:
        write(args.output/'status.json',{'state':'failed','error':str(error)});raise
if __name__=='__main__':main()
