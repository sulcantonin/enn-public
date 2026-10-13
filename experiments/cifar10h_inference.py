#!/usr/bin/env python3
"""Run a Yager and a Dempster MP-ENN checkpoint on the CIFAR-10 test set and cache the outputs.

The cache feeds ``cifar10h_analysis.py`` and ``make_figure2.py`` (Sect. 5.2,
Fig. 2), which then need no GPU. Arrays (test-set order = CIFAR-10H order):
  by, py, vy : Yager    belief (N, 10), plausibility (N, 10), vacuity (N,)
  bd, pd, vd : Dempster belief, plausibility, vacuity
  labels     : CIFAR-10 test labels (N,)

The paper used the Yager model of ``cifar10_mpenn.py --suite main`` (seed 0)
and the Dempster model of ``--suite ablation-combine``:

    python experiments/cifar10h_inference.py \
        --yager-checkpoint runs/cifar10/main_seed0/best_model.pth \
        --dempster-checkpoint runs/cifar10/rule_dempster/best_model.pth \
        --output runs/cifar10/interp_cache.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mpenn.data import get_test_loader
from mpenn.training import default_device, load_enn


@torch.no_grad()
def infer(model, loader, device):
    bel, pl, vac, labels = [], [], [], []
    for x, y in loader:
        out = model(x.to(device))
        bel.append(out['belief'].float().cpu().numpy())
        pl.append(out['plausibility'].float().cpu().numpy())
        vac.append(out['uncertainty'].float().cpu().numpy())
        labels.append(y.numpy())
    return tuple(map(np.concatenate, (bel, pl, vac, labels)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--yager-checkpoint', required=True)
    p.add_argument('--dempster-checkpoint', required=True)
    p.add_argument('--prior-weight', type=float, default=10.0)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default='runs/cifar10/interp_cache.npz')
    args = p.parse_args()

    device = args.device or default_device()
    loader = get_test_loader(args.batch_size, num_workers=0)
    arrays = {}
    for rule, ckpt, tag in (('yager', args.yager_checkpoint, 'y'), ('dempster', args.dempster_checkpoint, 'd')):
        print(f'{rule:8s} checkpoint: {ckpt}')
        model = load_enn(ckpt, device, num_classes=10, num_sources=3,
                         prior_weight=args.prior_weight, combination_rule=rule)
        bel, pl, vac, labels = infer(model, loader, device)
        arrays.update({f'b{tag}': bel, f'p{tag}': pl, f'v{tag}': vac})
        acc = 100.0 * float((bel.argmax(1) == labels).mean())
        print(f'  test accuracy {acc:.2f}%   mean m(Theta) {vac.mean():.4f}')
    arrays['labels'] = labels

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
