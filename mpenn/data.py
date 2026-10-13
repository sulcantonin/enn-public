"""CIFAR-10 in-distribution loaders and the seven OOD test sets of Table 1.

Datasets live under ``data/`` at the repository root; override with the
``MPENN_DATA`` environment variable. ``scripts/download_data.py`` fetches them.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

OOD_DATASETS = ('CIFAR-100', 'SVHN', 'MNIST', 'FashionMNIST', 'Textures(DTD)', 'Gaussian', 'Uniform')


def data_root() -> str:
    default = Path(__file__).resolve().parent.parent / 'data'
    return os.environ.get('MPENN_DATA', str(default))


def default_num_workers() -> int:
    """DataLoader workers: ``MPENN_NUM_WORKERS`` if set, else min(4, usable CPUs - 1).

    Uses CPU affinity rather than ``os.cpu_count()`` so SLURM allocations with
    few CPUs are not oversubscribed.
    """
    raw = os.environ.get('MPENN_NUM_WORKERS', '').strip()
    if raw:
        return max(0, int(raw))
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    return max(0, min(4, n - 1))


def _loader(ds, batch_size, shuffle, num_workers, pin_memory=None, persistent_workers=None):
    if num_workers is None:
        num_workers = default_num_workers()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kw['persistent_workers'] = True if persistent_workers is None else persistent_workers
    return DataLoader(ds, **kw)


def get_train_loader(batch_size=128, num_workers: Optional[int] = None,
                     max_train_samples: Optional[int] = None, subset_seed: int = 0):
    """CIFAR-10 train set with random-crop + horizontal-flip augmentation."""
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_root(), train=True, download=True, transform=tfm)
    if max_train_samples is not None and max_train_samples < len(ds):
        g = torch.Generator().manual_seed(int(subset_seed))
        perm = torch.randperm(len(ds), generator=g).tolist()
        ds = Subset(ds, perm[:int(max_train_samples)])
    return _loader(ds, batch_size, True, num_workers)


def get_test_loader(batch_size=128, num_workers: Optional[int] = None):
    """CIFAR-10 test set (10,000 images, fixed order = CIFAR-10H order)."""
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = torchvision.datasets.CIFAR10(root=data_root(), train=False, download=True, transform=tfm)
    return _loader(ds, batch_size, False, num_workers)


class _NoiseDataset(Dataset):
    """Synthetic OOD images, drawn once at construction from the global torch RNG.

    'gaussian': N(0, 1) per pixel; 'uniform': U[0, 1) per pixel. Both are then
    normalised with the CIFAR-10 statistics (no clipping).
    """

    def __init__(self, kind: str, n: int = 10000, shape=(3, 32, 32)):
        self.data = torch.randn(n, *shape) if kind == 'gaussian' else torch.rand(n, *shape)
        for c in range(3):
            self.data[:, c] = (self.data[:, c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
        self.targets = torch.zeros(n, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def get_ood_loaders(batch_size=128, num_workers: Optional[int] = None, max_samples: int = 10000,
                    names: Iterable[str] = OOD_DATASETS) -> Dict[str, DataLoader]:
    """OOD test sets, each capped at the first ``max_samples`` images.

    Near-OOD: CIFAR-100 test. Far-OOD: SVHN test, MNIST test, Fashion-MNIST
    test, DTD test split, Gaussian noise, Uniform noise. Grayscale sets are
    replicated to 3 channels; all are resized/cropped to 32x32 and normalised
    with the CIFAR-10 statistics. The noise sets consume the global torch RNG,
    so their content depends on the seed set before this call.
    """
    import torchvision
    import torchvision.transforms as T

    if num_workers is None:
        num_workers = default_num_workers()
    names = list(names)
    unknown = set(names) - set(OOD_DATASETS)
    if unknown:
        raise ValueError(f'unknown OOD sets: {sorted(unknown)}; choose from {OOD_DATASETS}')

    tf_color = T.Compose([
        T.Resize(32), T.CenterCrop(32), T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tf_gray = T.Compose([
        T.Resize(32), T.CenterCrop(32), T.Grayscale(num_output_channels=3),
        T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    root = data_root()
    builders = {
        'CIFAR-100': lambda: torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tf_color),
        'SVHN': lambda: torchvision.datasets.SVHN(root, split='test', download=True, transform=tf_color),
        'MNIST': lambda: torchvision.datasets.MNIST(root, train=False, download=True, transform=tf_gray),
        'FashionMNIST': lambda: torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=tf_gray),
        'Textures(DTD)': lambda: torchvision.datasets.DTD(root, split='test', download=True, transform=tf_color),
    }

    loaders = {}
    for name in OOD_DATASETS:  # fixed order keeps the RNG draws of the noise sets reproducible
        if name not in names:
            continue
        if name in builders:
            try:
                ds = builders[name]()
            except Exception as e:  # keep going if one optional download fails
                print(f'  [skip] {name}: {e}')
                continue
            if len(ds) > max_samples:
                ds = Subset(ds, list(range(max_samples)))
            loaders[name] = _loader(ds, batch_size, False, num_workers)
        else:
            kind = 'gaussian' if name == 'Gaussian' else 'uniform'
            loaders[name] = DataLoader(_NoiseDataset(kind, 10000), batch_size=batch_size, num_workers=0)
    return loaders
