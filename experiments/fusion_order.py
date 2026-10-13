#!/usr/bin/env python3
"""Fusion-order sensitivity of the Yager fold (Sect. 5.3, "Ablations").

Yager's rule is commutative but not associative, so ((m_1 + m_2) + m_3) can
depend on the order of the sources. One forward pass caches the discounted
per-source masses; every one of the |S|! orderings is then re-folded from the
cache. With |S| = 3 commutativity leaves 3 distinct outcomes (which source is
folded last); all 6 are enumerated and the distinct ones counted.

Per dataset the report gives accuracy and mean m(Theta) per ordering, the
per-image maximum pairwise L1 distance between orderings (mean, p99, max), the
prediction-flip rate, the order-averaged (symmetric) variant, and, for OOD
sets, the vacuity AUROC of each ordering against the matching ID ordering.
"mean_conflict" is the per-image sum of the two pairwise kappas of the fold.

    python experiments/fusion_order.py --checkpoint runs/cifar10/main_seed0/best_model.pth \
        --ood CIFAR-100 SVHN --output runs/cifar10/fusion_order_report.json
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from mpenn.combine import COMBINATION_RULES
from mpenn.data import OOD_DATASETS, get_ood_loaders, get_test_loader
from mpenn.training import default_device, load_enn


def auroc(id_scores, ood_scores) -> float:
    y = [0] * len(id_scores) + [1] * len(ood_scores)
    return float(roc_auc_score(y, list(id_scores) + list(ood_scores)))


def combine_in_order(source_masses, order, num_classes, rule='yager'):
    """Left-fold the cached per-source masses in the given permutation."""
    combine = COMBINATION_RULES[rule]
    m = source_masses[order[0]]
    total_k = torch.zeros(m.size(0))
    for j in order[1:]:
        m, k = combine(m, source_masses[j], num_classes)
        total_k = total_k + k
    return m, total_k


@torch.no_grad()
def cache_source_masses(model, loader, device, max_batches=None):
    """One pass of encoder + heads + discount; returns per-source masses (CPU) and labels."""
    per_source, labels = None, []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        out = model(x.to(device))
        if per_source is None:
            per_source = [[] for _ in out['source_masses']]
        for s, m in enumerate(out['source_masses']):
            per_source[s].append(m.float().cpu())
        labels.append(y)
    return [torch.cat(c) for c in per_source], torch.cat(labels)


def analyse(masses, labels, num_classes, rule='yager'):
    orders = list(itertools.permutations(range(len(masses))))
    per_order, mass_stack, vac_stack, pred_stack = {}, [], [], []
    for order in orders:
        m, k = combine_in_order(masses, order, num_classes, rule)
        vac, pred = m[:, num_classes], m[:, :num_classes].argmax(dim=1)
        entry = {'mean_vacuity': float(vac.mean()), 'std_vacuity': float(vac.std()),
                 'mean_conflict': float(k.mean())}
        if labels is not None:
            entry['accuracy'] = float((pred == labels).float().mean() * 100)
        per_order['->'.join(map(str, order))] = entry
        mass_stack.append(m)
        vac_stack.append(vac)
        pred_stack.append(pred)

    M, V, P = torch.stack(mass_stack), torch.stack(vac_stack), torch.stack(pred_stack)

    distinct = []
    for i in range(len(orders)):
        if not any(torch.allclose(M[i], M[j], atol=1e-6) for j in distinct):
            distinct.append(i)

    l1_max = torch.zeros(M.shape[1])
    vac_max = torch.zeros(M.shape[1])
    for i, j in itertools.combinations(range(len(orders)), 2):
        l1_max = torch.maximum(l1_max, (M[i] - M[j]).abs().sum(dim=-1))
        vac_max = torch.maximum(vac_max, (V[i] - V[j]).abs())

    m_sym = M.mean(dim=0)
    symmetric = {
        'mean_vacuity': float(m_sym[:, num_classes].mean()),
        'max_l1_to_any_ordering': float(max((m_sym - M[i]).abs().sum(dim=-1).mean()
                                            for i in range(len(orders)))),
    }
    if labels is not None:
        symmetric['accuracy'] = float((m_sym[:, :num_classes].argmax(dim=1) == labels).float().mean() * 100)

    spread = {
        'n_orderings': len(orders),
        'n_distinct_outcomes': len(distinct),
        'mean_max_pairwise_L1': float(l1_max.mean()),
        'p99_max_pairwise_L1': float(torch.quantile(l1_max, 0.99)),
        'max_max_pairwise_L1': float(l1_max.max()),
        'mean_max_pairwise_vacuity_diff': float(vac_max.mean()),
        'max_max_pairwise_vacuity_diff': float(vac_max.max()),
        'prediction_flip_rate': float((P != P[0]).any(dim=0).float().mean()),
        'mean_vacuity_across_orderings': float(V.mean()),
        'range_of_mean_vacuity': float(V.mean(dim=1).max() - V.mean(dim=1).min()),
    }
    return per_order, spread, symmetric, V


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rule', default='yager', choices=['yager', 'dempster'])
    p.add_argument('--num-sources', type=int, default=3)
    p.add_argument('--prior-weight', type=float, default=10.0)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-batches', type=int, default=None, help='cap batches per dataset (debugging)')
    p.add_argument('--ood', nargs='*', default=['CIFAR-100', 'SVHN'], choices=list(OOD_DATASETS))
    p.add_argument('--device', default=None)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = args.device or default_device()
    model = load_enn(args.checkpoint, device, num_classes=10, num_sources=args.num_sources,
                     prior_weight=args.prior_weight, combination_rule=args.rule)
    k = model.num_classes
    report = {'checkpoint': str(args.checkpoint), 'rule': args.rule,
              'num_sources': args.num_sources, 'datasets': {}}

    print('=== CIFAR-10 test (ID) ===')
    masses, labels = cache_source_masses(model, get_test_loader(args.batch_size), device, args.max_batches)
    per_order, spread, sym, v_id = analyse(masses, labels, k, args.rule)
    report['datasets']['CIFAR-10-test'] = {'per_ordering': per_order, 'spread': spread, 'symmetric': sym}
    print(json.dumps(spread, indent=2))

    if args.ood:
        loaders = get_ood_loaders(args.batch_size, names=args.ood)
        for name, loader in loaders.items():
            print(f'\n=== {name} (OOD) ===')
            masses, _ = cache_source_masses(model, loader, device, args.max_batches)
            per_order_o, spread_o, sym_o, v_ood = analyse(masses, None, k, args.rule)
            aurocs = {key: auroc(v_id[i].numpy(), v_ood[i].numpy()) for i, key in enumerate(per_order_o)}
            report['datasets'][name] = {
                'per_ordering': per_order_o, 'spread': spread_o, 'symmetric': sym_o,
                'vacuity_auroc_per_ordering': aurocs,
                'vacuity_auroc_symmetric': auroc(v_id.mean(0).numpy(), v_ood.mean(0).numpy()),
                'vacuity_auroc_range': max(aurocs.values()) - min(aurocs.values()),
            }
            print(json.dumps(spread_o, indent=2))
            print(f'  vacuity AUROC range across orderings: {report["datasets"][name]["vacuity_auroc_range"]:.5f}')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
