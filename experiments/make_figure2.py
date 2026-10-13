#!/usr/bin/env python3
"""Fig. 2: mean [Bel, Pl] interval on the true class vs. CIFAR-10H annotator entropy.

All 10,000 CIFAR-10 test images, grouped into the five fixed entropy bins of
``cifar10h_analysis.py``; each point is plotted at the bin's mean entropy. The
true class is the CIFAR-10 test label. The interval width Pl - Bel is m(Theta).

    python experiments/make_figure2.py --cache results/cifar10/interp_cache.npz \
        --output runs/figures/interp_interval.pdf
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cifar10h_analysis import annotator_entropy, entropy_bins
from mpenn.data import data_root

YAGER_C, DEMPSTER_C = '#EA580C', '#2563EB'


def cifar10_test_labels() -> np.ndarray:
    import torchvision
    return np.array(torchvision.datasets.CIFAR10(data_root(), train=False, download=True).targets)


def binned_interval(H, bel, pl, labels):
    """Per bin: mean H, mean Bel and mean Pl of the true class."""
    rows = np.arange(len(labels))
    bel_true, pl_true = bel[rows, labels], pl[rows, labels]
    out = []
    for lo, hi, m in entropy_bins(H):
        if m.any():
            out.append({'lo': lo, 'hi': hi, 'n': int(m.sum()), 'mean_H': float(H[m].mean()),
                        'bel': float(bel_true[m].mean()), 'pl': float(pl_true[m].mean())})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default='results/cifar10/interp_cache.npz')
    p.add_argument('--human', default=os.path.join(data_root(), 'cifar10h-probs.npy'))
    p.add_argument('--output', default='runs/figures/interp_interval.pdf')
    args = p.parse_args()

    z = np.load(args.cache)
    labels = z['labels'] if 'labels' in z.files else cifar10_test_labels()
    H = annotator_entropy(np.load(args.human))
    curves = {
        'yager': binned_interval(H, z['by'], z['py'], labels),
        'dempster': binned_interval(H, z['bd'], z['pd'], labels),
    }
    for rule, rows in curves.items():
        widths = ', '.join(f'{r["pl"] - r["bel"]:.3f}' for r in rows)
        print(f'{rule:8s} interval width per bin: {widths}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 1.8))
    styles = {
        'yager': dict(color=YAGER_C, marker='o', ls='-', label=r'Yager  $[\mathrm{Bel},\mathrm{Pl}]$'),
        'dempster': dict(color=DEMPSTER_C, marker='s', ls='--', label=r'Dempster $[\mathrm{Bel},\mathrm{Pl}]$'),
    }
    for rule, rows in curves.items():
        st = styles[rule]
        x = [r['mean_H'] for r in rows]
        bel = [r['bel'] for r in rows]
        pl = [r['pl'] for r in rows]
        ax.fill_between(x, bel, pl, color=st['color'], alpha=0.25, lw=0)
        ax.plot(x, pl, color=st['color'], marker=st['marker'], ls=st['ls'], ms=3.5, lw=1.5, label=st['label'])
        ax.plot(x, bel, color=st['color'], marker=st['marker'], ls=st['ls'], ms=3.5, lw=1.5)
    ax.set_xlabel('human annotator entropy (CIFAR-10H)', fontsize=8)
    ax.set_ylabel(r'$[\mathrm{Bel},\mathrm{Pl}]$ on true class', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower left')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    with open(out.with_suffix('.json'), 'w') as f:
        json.dump(curves, f, indent=2)
    print(f'wrote {out} and {out.with_suffix(".json")}')


if __name__ == '__main__':
    main()
