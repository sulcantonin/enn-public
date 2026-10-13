#!/usr/bin/env python3
"""Download every dataset the experiments need into ``data/`` (or $MPENN_DATA).

Idempotent: datasets already present are verified and skipped.

    python scripts/download_data.py                 # everything
    python scripts/download_data.py --check         # verify only
    python scripts/download_data.py --only cifar10 cifar10h
    python scripts/download_data.py --skip dtd snli

  cifar10        CIFAR-10 train/test           (~170 MB)  in-distribution
  cifar100       CIFAR-100 test                (~170 MB)  near-OOD
  svhn           SVHN test                     (~65 MB)   far-OOD
  mnist          MNIST test                    (~65 MB)   far-OOD
  fashionmnist   Fashion-MNIST test            (~85 MB)   far-OOD
  dtd            Describable Textures          (~600 MB)  far-OOD
  cifar10h       CIFAR-10H label distributions (~1 MB)    Sect. 5.2
  chaosnli       ChaosNLI-SNLI                 (~1 MB)    Sect. 5.2
  snli           SNLI (Hugging Face)           (~100 MB)  training the ChaosNLI model

Gaussian and Uniform noise OOD sets are generated on the fly.
"""

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('MPENN_DATA', ROOT / 'data'))

# CIFAR-10H (Peterson et al., 2019): 10000 x 10 normalised human label counts,
# rows in CIFAR-10 test-set order.
CIFAR10H_URLS = [
    'https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy',
    'https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy',
]

# ChaosNLI (Nie et al., 2020). The official archive link from
# https://github.com/easonnie/ChaosNLI is tried first; if it is unavailable the
# SNLI file is taken from a community mirror on the Hugging Face Hub and checked
# against the SHA-256 below (1,514 pairs).
CHAOSNLI_URL = os.environ.get('CHAOSNLI_URL', 'https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1')
CHAOSNLI_MIRROR = ('earino/chaosnli', 'raw/chaosNLI_snli.jsonl')
CHAOSNLI_SNLI_SHA256 = '99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1'
CHAOSNLI_TARGET = DATA / 'ChaosNLI' / 'data' / 'chaosNLI_snli.jsonl'

SNLI_HF_ID = 'stanfordnlp/snli'


def ok(m):
    print(f'  [ok]   {m}')


def warn(m):
    print(f'  [warn] {m}')


def fail(m):
    print(f'  [FAIL] {m}')


def size_of(p: Path) -> str:
    n = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.0f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def tv_dataset(name: str, check_only: bool) -> bool:
    import torchvision
    specs = {
        'cifar10': [(torchvision.datasets.CIFAR10, dict(train=True)), (torchvision.datasets.CIFAR10, dict(train=False))],
        'cifar100': [(torchvision.datasets.CIFAR100, dict(train=False))],
        'svhn': [(torchvision.datasets.SVHN, dict(split='test'))],
        'mnist': [(torchvision.datasets.MNIST, dict(train=False))],
        'fashionmnist': [(torchvision.datasets.FashionMNIST, dict(train=False))],
        'dtd': [(torchvision.datasets.DTD, dict(split='test'))],
    }
    for cls, kw in specs[name]:
        try:
            cls(root=str(DATA), download=False, **kw)
            continue
        except Exception:
            pass
        if check_only:
            fail(f'{name}: missing')
            return False
        print(f'  downloading {name} ...', flush=True)
        try:
            cls(root=str(DATA), download=True, **kw)
        except Exception as e:
            fail(f'{name}: {type(e).__name__}: {e}')
            return False
    ok(f'{name}')
    return True


def get_cifar10h(check_only: bool) -> bool:
    import numpy as np
    target = DATA / 'cifar10h-probs.npy'

    def valid() -> bool:
        try:
            a = np.load(target)
        except Exception:
            return False
        return a.shape == (10000, 10) and np.allclose(a.sum(axis=1), 1.0, atol=1e-4)

    if target.exists() and valid():
        ok('cifar10h')
        return True
    if check_only:
        fail('cifar10h: missing')
        return False
    DATA.mkdir(parents=True, exist_ok=True)
    for url in CIFAR10H_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                target.write_bytes(r.read())
            if valid():
                ok(f'cifar10h (from {url})')
                return True
        except Exception as e:
            warn(f'cifar10h: {type(e).__name__} from {url}')
    target.unlink(missing_ok=True)
    fail('cifar10h: get data/cifar10h-probs.npy from https://github.com/jcpeterson/cifar-10h')
    return False


def get_chaosnli(check_only: bool) -> bool:
    def n_pairs(p: Path) -> int:
        with p.open() as f:
            return sum(1 for line in f if line.strip())

    if CHAOSNLI_TARGET.exists():
        ok(f'chaosnli ({n_pairs(CHAOSNLI_TARGET)} SNLI pairs)')
        return True
    if check_only:
        fail(f'chaosnli: missing ({CHAOSNLI_TARGET})')
        return False
    CHAOSNLI_TARGET.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(CHAOSNLI_URL, headers={'User-Agent': 'Wget/1.21'})
        with urllib.request.urlopen(req, timeout=300) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            member = next(n for n in z.namelist() if n.endswith('chaosNLI_snli.jsonl'))
            CHAOSNLI_TARGET.write_bytes(z.read(member))
        ok(f'chaosnli from the official archive ({n_pairs(CHAOSNLI_TARGET)} SNLI pairs)')
        return True
    except Exception as e:
        warn(f'chaosnli: official archive unavailable ({type(e).__name__}); trying the Hugging Face mirror')

    try:
        from huggingface_hub import hf_hub_download
        repo, filename = CHAOSNLI_MIRROR
        path = Path(hf_hub_download(repo, filename, repo_type='dataset'))
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != CHAOSNLI_SNLI_SHA256:
            fail(f'chaosnli: mirror file hash {digest} does not match the expected {CHAOSNLI_SNLI_SHA256}')
            return False
        CHAOSNLI_TARGET.write_bytes(blob)
        ok(f'chaosnli from hf.co/datasets/{repo} ({n_pairs(CHAOSNLI_TARGET)} SNLI pairs, SHA-256 verified)')
        return True
    except Exception as e:
        fail(f'chaosnli: {type(e).__name__}: {e}\n'
             f'         place chaosNLI_snli.jsonl (ChaosNLI v1.0) at {CHAOSNLI_TARGET}')
        return False


def get_snli(check_only: bool) -> bool:
    """SNLI lives in the Hugging Face cache (HF_HOME), not under data/."""
    if check_only:
        ok('snli: not checked (Hugging Face caches it on first use)')
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        fail('snli: needs `pip install -e ".[nli]"`')
        return False
    try:
        d = load_dataset(SNLI_HF_ID)
        ok(f'snli (train={len(d["train"])}, validation={len(d["validation"])})')
        return True
    except Exception as e:
        fail(f'snli: {type(e).__name__}: {e}')
        return False


DATASETS = {
    'cifar10': lambda c: tv_dataset('cifar10', c),
    'cifar100': lambda c: tv_dataset('cifar100', c),
    'svhn': lambda c: tv_dataset('svhn', c),
    'mnist': lambda c: tv_dataset('mnist', c),
    'fashionmnist': lambda c: tv_dataset('fashionmnist', c),
    'dtd': lambda c: tv_dataset('dtd', c),
    'cifar10h': get_cifar10h,
    'chaosnli': get_chaosnli,
    'snli': get_snli,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true', help='verify only; download nothing')
    p.add_argument('--only', nargs='+', choices=list(DATASETS))
    p.add_argument('--skip', nargs='+', choices=list(DATASETS), default=[])
    args = p.parse_args()

    names = args.only or [n for n in DATASETS if n not in args.skip]
    DATA.mkdir(parents=True, exist_ok=True)
    print(f'{"Verifying" if args.check else "Fetching"} {len(names)} dataset(s) in {DATA}')
    results = {}
    for n in names:
        try:
            results[n] = bool(DATASETS[n](args.check))
        except Exception as e:
            fail(f'{n}: {type(e).__name__}: {e}')
            results[n] = False
    bad = [n for n, good in results.items() if not good]
    print(f'\n{len(names) - len(bad)}/{len(names)} ready, {size_of(DATA)} on disk'
          + (f'; incomplete: {", ".join(bad)}' if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
