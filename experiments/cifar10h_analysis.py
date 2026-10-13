#!/usr/bin/env python3
"""Vacuity vs. CIFAR-10H annotator disagreement, Yager and Dempster (Sect. 5.2).

Needs only the cache written by ``cifar10h_inference.py`` and the CIFAR-10H
label distributions; no GPU. Human disagreement of an image is the Shannon
entropy H (nats) of its CIFAR-10H label distribution. Reports:

  * Spearman rho(H, m(Theta)) for both rules, with a percentile bootstrap CI;
  * mean m(Theta) in five fixed entropy bins with edges 0, 0.05, 0.15, 0.30,
    0.50 nats (the first bin is exactly the unanimous images);
  * mean m(Theta) on the most contested images (H >= 90th/95th/99th percentile)
    and the Yager/Dempster ratio;
  * the AUROC of m(Theta) separating images with H >= P90 from images with
    H <= P10 (P10 = 0 here, so the low group is every unanimous image).

    python experiments/cifar10h_analysis.py --cache results/cifar10/interp_cache.npz \
        --output runs/cifar10/human_disagreement_report.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

from mpenn.data import data_root

BIN_EDGES = (0.0, 0.05, 0.15, 0.30, 0.50)


def annotator_entropy(human: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) per row. Probabilities are sorted first so that label
    distributions equal up to a class permutation get bit-identical entropies;
    otherwise percentile cut-offs can split such ties differently across platforms."""
    p = np.sort(np.clip(human, 1e-12, 1), axis=1)
    return -(p * np.log(p)).sum(1)


def entropy_bins(H: np.ndarray):
    """Boolean masks for the five fixed entropy bins (last bin is open-ended)."""
    edges = np.array(list(BIN_EDGES) + [H.max() + 1e-6])
    for lo, hi in zip(edges[:-1], edges[1:]):
        yield float(lo), float(hi), (H >= lo) & (H < hi)


def separation_auroc(scores_low: np.ndarray, scores_high: np.ndarray) -> float:
    """P(score of a contested image > score of an uncontested one); ties count 1/2."""
    u = stats.mannwhitneyu(scores_high, scores_low, alternative='two-sided').statistic
    return float(u / (len(scores_high) * len(scores_low)))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple:
    """95% percentile bootstrap CI of Spearman rho, resampling images."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        vals[b] = stats.spearmanr(x[idx], y[idx]).correlation
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default='results/cifar10/interp_cache.npz')
    p.add_argument('--human', default=os.path.join(data_root(), 'cifar10h-probs.npy'))
    p.add_argument('--output', default='runs/cifar10/human_disagreement_report.json')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    z = np.load(args.cache)
    vy, vd = z['vy'].astype(np.float64), z['vd'].astype(np.float64)
    H = annotator_entropy(np.load(args.human))
    assert len(H) == len(vy) == len(vd), 'cache and CIFAR-10H are misaligned'
    rules = (('yager', vy), ('dempster', vd))
    rep = {'n_images': int(len(H))}
    print(f'n = {len(H)} CIFAR-10 test images with CIFAR-10H annotations\n')

    print('Spearman rho(annotator entropy, m(Theta))')
    for name, v in rules:
        r = stats.spearmanr(H, v)
        lo, hi = bootstrap_ci(H, v, args.n_boot, args.seed)
        rep[f'spearman_{name}'] = {'rho': float(r.correlation), 'p': float(r.pvalue), 'ci95': [lo, hi]}
        print(f'  {name:9s} rho = {r.correlation:.3f}  [95% CI {lo:.3f}, {hi:.3f}]  p = {r.pvalue:.2e}')

    rep['mean_vacuity'] = {'yager': float(vy.mean()), 'dempster': float(vd.mean()),
                           'ratio': float(vy.mean() / vd.mean())}
    print(f'\nMean m(Theta): Yager {vy.mean():.4f}  Dempster {vd.mean():.4f}  '
          f'ratio {vy.mean() / vd.mean():.1f}x')

    print('\nMean m(Theta) per annotator-entropy bin')
    bins = []
    for lo, hi, m in entropy_bins(H):
        if not m.any():
            continue
        b = {'lo': lo, 'hi': hi, 'n': int(m.sum()), 'mean_H': float(H[m].mean()),
             'yager': float(vy[m].mean()), 'dempster': float(vd[m].mean()),
             'yager_sem': float(vy[m].std() / np.sqrt(m.sum())),
             'dempster_sem': float(vd[m].std() / np.sqrt(m.sum())),
             'ratio': float(vy[m].mean() / vd[m].mean())}
        bins.append(b)
        print(f'  H in [{lo:.2f}, {hi:.2f})  n={b["n"]:5d}  Yager {b["yager"]:.4f}  '
              f'Dempster {b["dempster"]:.4f}  ratio {b["ratio"]:.1f}x')
    rep['entropy_bins'] = bins

    print('\nMost contested images')
    rep['contested'] = {}
    for q in (90, 95, 99):
        m = H >= np.percentile(H, q)
        rep['contested'][f'top{100 - q}pct'] = {
            'n': int(m.sum()), 'yager': float(vy[m].mean()), 'dempster': float(vd[m].mean()),
            'ratio': float(vy[m].mean() / vd[m].mean())}
        print(f'  top {100 - q:2d}% (n={m.sum():4d}): Yager {vy[m].mean():.4f}  '
              f'Dempster {vd[m].mean():.4f}  ratio {vy[m].mean() / vd[m].mean():.1f}x')

    print('\nTop decile (H >= P90) vs bottom (H <= P10)')
    m_hi, m_lo = H >= np.percentile(H, 90), H <= np.percentile(H, 10)
    rep['responsiveness'] = {}
    for name, v in rules:
        rel = float(v[m_hi].mean() / v.mean())
        sep = separation_auroc(v[m_lo], v[m_hi])
        rep['responsiveness'][name] = {'top_decile_over_mean': rel, 'decile_separation_auroc': sep,
                                       'n_high': int(m_hi.sum()), 'n_low': int(m_lo.sum())}
        print(f'  {name:9s} top-decile mean = {rel:.2f}x overall mean; separation AUROC = {sep:.3f}')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
