#!/usr/bin/env python3
"""Worst-case combination-layer derivative as a function of conflict (Sect. 3.3).

Two sources on F with K = 3: m_1 puts mass p on {c_1}, m_2 puts mass p on
{c_2}, the rest on Theta, so kappa = p^2. For each p the script reports
max_{A,B} |d m_comb(A) / d m_1(B)| for Yager's rule and for Dempster's rule
without the (1 - kappa) clamp. Dempster's tracks 1/(1 - kappa); Yager's is 1
at every conflict level (Theorem 1). Computed in float32, as in the paper.

    python experiments/gradient_vs_conflict.py --output runs/cifar10/gradient_vs_conflict.json
"""

import argparse
import json
from pathlib import Path

import torch

from mpenn.combine import dempster_combine, yager_combine


def max_abs_jacobian(rule, m1, m2, num_classes) -> float:
    """max over (A, B) of |d m_comb(A) / d m_1(B)|, one autograd call per output entry."""
    m1 = m1.clone().requires_grad_(True)
    out, _ = rule(m1, m2, num_classes)
    worst = 0.0
    for a in range(num_classes + 1):
        g, = torch.autograd.grad(out[0, a], m1, retain_graph=True)
        g = torch.nan_to_num(g, nan=0.0, posinf=float('inf'), neginf=float('inf'))
        worst = max(worst, float(g.abs().max()))
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--num-classes', type=int, default=3)
    p.add_argument('--output', default='runs/cifar10/gradient_vs_conflict.json')
    args = p.parse_args()
    k = args.num_classes

    def dempster_unclamped(a, b, n):
        return dempster_combine(a, b, n, clamp_one_minus_k=False)

    rows = []
    print(f"{'p':>8} {'kappa':>10} {'Yager max|J|':>14} {'Dempster max|J|':>17} {'1/(1-kappa)':>13}")
    for pp in [0.5, 0.9, 0.99, 0.999, 0.9999]:
        m1 = torch.zeros(1, k + 1)
        m1[0, 0], m1[0, -1] = pp, 1 - pp
        m2 = torch.zeros(1, k + 1)
        m2[0, 1], m2[0, -1] = pp, 1 - pp
        kappa = float(m1[0, :k].sum() * m2[0, :k].sum() - (m1[0, :k] * m2[0, :k]).sum())
        jy = max_abs_jacobian(yager_combine, m1, m2, k)
        jd = max_abs_jacobian(dempster_unclamped, m1, m2, k)
        rows.append({'p': pp, 'kappa': kappa, 'yager_max_abs_jacobian': jy,
                     'dempster_max_abs_jacobian': jd, 'one_over_one_minus_kappa': 1 / (1 - kappa)})
        print(f'{pp:8.4f} {kappa:10.6f} {jy:14.4f} {jd:17.4f} {1 / (1 - kappa):13.1f}')

    out = {'num_classes': k, 'family': 'm1 = p on c_0, m2 = p on c_1, rest on Theta', 'rows': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
