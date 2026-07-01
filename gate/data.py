"""ImageNet val loader: sample N requests (fixed seed) and preprocess with the
timm ViT-B/16 standard transform (resize 256 bicubic -> center-crop 224,
normalize with mean/std 0.5).

All three runtimes consume the SAME N requests in the SAME order, so exit masks
and batch composition are identical across runtimes.
"""
from __future__ import annotations

import numpy as np

from .util import Config


def _build_transform(cfg: Config):
    from PIL import Image
    from torchvision import transforms

    interp = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }.get(cfg.data.get("interpolation", "bicubic"), Image.BICUBIC)

    return transforms.Compose([
        transforms.Resize(int(cfg.data.resize), interpolation=interp),
        transforms.CenterCrop(int(cfg.data.crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.data.mean), std=list(cfg.data.std)),
    ])


def load_requests(cfg: Config, n: int | None = None):
    """Return (images[N,3,H,W] float32, labels[N] int64, paths[N]).

    Requests are a fixed-seed random sample of the ImageNet val set.
    """
    from torchvision.datasets import ImageFolder

    n = int(cfg.data.num_requests if n is None else n)
    tf = _build_transform(cfg)
    ds = ImageFolder(cfg.data.imagenet_val_dir, transform=tf)

    rng = np.random.default_rng(int(cfg.arrivals.seed))
    total = len(ds)
    if n > total:
        raise ValueError(f"num_requests={n} exceeds val set size {total}")
    idx = rng.choice(total, size=n, replace=False)

    images = np.empty((n, 3, int(cfg.data.crop), int(cfg.data.crop)), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    paths = []
    for j, i in enumerate(idx):
        img, lbl = ds[int(i)]
        images[j] = img.numpy()
        labels[j] = lbl
        paths.append(ds.samples[int(i)][0])
    return images, labels, paths


def iter_batches(images: np.ndarray, batch: int, drop_last: bool = True):
    """Yield (start_idx, batch_images) for full batches of size `batch`.

    Leftover partial batch is dropped (spec: never-full batches are dropped).
    """
    n = images.shape[0]
    full = (n // batch) * batch if drop_last else n
    for s in range(0, full, batch):
        yield s, images[s : s + batch]
