"""Inspect ImageFolder-style data without changing images or splits."""

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def scan_split(directory, class_to_idx=None):
    directory = Path(directory)
    classes = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class directories: {directory}")
    mapping = {name: i for i, name in enumerate(classes)}
    if class_to_idx is not None and mapping != class_to_idx:
        raise ValueError(f"Class mapping differs between splits: {directory}")
    samples = []
    for name in classes:
        images = sorted(p for p in (directory / name).iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Empty class: {directory / name}")
        samples.extend((path, mapping[name]) for path in images)
    return samples, mapping


class ImageSamples:
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")
        return pixels["pixel_values"][0], label

