#!/usr/bin/env python3
"""MP-ENN-only CIFAR-10 runs: main model, Yager-vs-Dempster pair, ablations, synthetic study.

Suites
------
main              MP-ENN-Yager for ``--seeds`` seeds (default 200 epochs). Seed 0 is
                  the "representative seed" used by the CIFAR-10H and fusion-order
                  analyses (Sect. 5.2, 5.3).
ablation-combine  MP-ENN-Yager and MP-ENN-Dempster, one seed (default 60 epochs);
                  logs per-epoch pre-clip gradient norms. ``rule_dempster`` is the
                  Dempster model of the CIFAR-10H analysis. Also runs ``synthetic``.
ablation-W        prior weight W in {1, 5, 10, 20, 100}, CIFAR-100 vacuity AUROC.
ablation-lambda   lambda_unc in {0, 0.01, 0.1, 1}, CIFAR-100 vacuity AUROC.
synthetic         gradient of the combined mass w.r.t. m_1 over 200 random mass
                  pairs on F, Yager vs Dempster (Sect. 5.3: 0.21 vs 1.52).
all               everything above.

    python experiments/cifar10_mpenn.py --suite main --seeds 1 --output-dir runs/cifar10
    python experiments/cifar10_mpenn.py --suite ablation-combine --output-dir runs/cifar10

Every run is skipped if its checkpoint / result file already exists.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from mpenn.baselines import EnergyDetector, ENNVacuityDetector, MSPDetector
from mpenn.combine import dempster_combine, yager_combine
from mpenn.data import get_ood_loaders, get_test_loader, get_train_loader
from mpenn.metrics import expected_calibration_error, ood_metrics
from mpenn.training import DEFAULT_CONFIG, default_device, set_seed, train_enn

NEAR_OOD = ('CIFAR-100',)
DEFAULT_EPOCHS = {'main': 200}   # every other suite: 60
W_VALUES = (1.0, 5.0, 10.0, 20.0, 100.0)
LAMBDA_VALUES = (0.0, 0.01, 0.1, 1.0)


def make_config(args, seed: int, output_dir: str, epochs: int, **overrides) -> dict:
    return {**DEFAULT_CONFIG, 'epochs': epochs, 'batch_size': args.batch_size, 'seed': seed,
            'output_dir': output_dir, 'quiet_tqdm': args.quiet, **overrides}


def train_loader_for(args, seed):
    return get_train_loader(args.batch_size, max_train_samples=args.max_train_samples, subset_seed=seed)


@torch.no_grad()
def max_belief_ece(model, loader, device) -> float:
    """ECE with confidence max_i m({c_i}) (fused belief, not renormalised)."""
    model.eval()
    confs, preds, labels = [], [], []
    for x, y in loader:
        bel = model(x.to(device))['belief']
        confs.append(bel.max(dim=1).values.cpu().numpy())
        preds.append(bel.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return expected_calibration_error(*map(np.concatenate, (confs, preds, labels)), n_bins=15)


def evaluate_ood(model, test_loader, ood_loaders, device, with_softmax: bool = True) -> dict:
    """per_ood[ood_name][detector] = {AUROC, FPR95, AUPR_IN, AUPR_OUT}."""
    detectors = {'ENN-Vacuity': ENNVacuityDetector(model)}
    if with_softmax:
        detectors = {'MSP': MSPDetector(model), 'Energy': EnergyDetector(model), **detectors}
    id_scores = {name: d.score(test_loader, device) for name, d in detectors.items()}
    return {
        ood_name: {name: ood_metrics(id_scores[name], d.score(loader, device))
                   for name, d in detectors.items()}
        for ood_name, loader in ood_loaders.items()
    }


def near_far_summary(per_ood: dict, det: str = 'ENN-Vacuity') -> dict:
    near = [v[det]['AUROC'] for k, v in per_ood.items() if k in NEAR_OOD]
    far = [v[det]['AUROC'] for k, v in per_ood.items() if k not in NEAR_OOD]
    return {'mean_near_auroc': float(np.mean(near)) if near else None,
            'mean_far_auroc': float(np.mean(far)) if far else None}


# ─────────────────────────────────────────────────────────────────────────────
def run_main(args):
    epochs = args.epochs or DEFAULT_EPOCHS['main']
    for s in range(args.seed, args.seed + args.seeds):
        out_dir = os.path.join(args.output_dir, f'main_seed{s}')
        result_path = os.path.join(out_dir, 'main_seed_results.json')
        if os.path.isfile(result_path):
            print(f'  main seed {s}: results exist ({result_path})')
            continue
        cfg = make_config(args, s, out_dir, epochs)
        set_seed(s)
        train_loader = train_loader_for(args, s)
        test_loader = get_test_loader(args.batch_size)
        ood_loaders = get_ood_loaders(args.batch_size, max_samples=args.ood_max_samples)
        model, info = train_enn(cfg, train_loader, test_loader, args.device)
        per_ood = evaluate_ood(model, test_loader, ood_loaders, args.device)
        row = {
            'seed': s,
            'epochs': epochs,
            'best_test_acc': info['best_test_acc'],
            'ece': max_belief_ece(model, test_loader, args.device),
            'near_far': near_far_summary(per_ood),
            'per_ood': per_ood,
        }
        with open(result_path, 'w') as f:
            json.dump(row, f, indent=2)
        with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
            json.dump(info['history'], f, indent=2)

    # Summary over every main seed present in the output directory.
    rows = []
    for path in sorted(glob.glob(os.path.join(args.output_dir, 'main_seed*', 'main_seed_results.json'))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        return
    agg = {}
    for name in sorted({k for r in rows for k in r['per_ood']}):
        au = [r['per_ood'][name]['ENN-Vacuity']['AUROC'] for r in rows if name in r['per_ood']]
        agg[name] = {'AUROC_mean': float(np.mean(au)), 'AUROC_std': float(np.std(au))}
    summary = {
        'seeds': [r['seed'] for r in rows],
        'enn_vacuity_auroc': agg,
        'accuracy_mean': float(np.mean([r['best_test_acc'] for r in rows])),
        'accuracy_std': float(np.std([r['best_test_acc'] for r in rows])),
        'ece_mean': float(np.mean([r['ece'] for r in rows])),
        'ece_std': float(np.std([r['ece'] for r in rows])),
    }
    with open(os.path.join(args.output_dir, 'main_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def run_ablation_combine(args):
    report_path = os.path.join(args.output_dir, 'ablation_combine_report.json')
    if os.path.isfile(report_path):
        print(f'  ablation-combine: results exist ({report_path})')
        return
    epochs = args.epochs or 60
    ood_loaders = get_ood_loaders(args.batch_size, max_samples=args.ood_max_samples)
    rows = []
    for rule in ('yager', 'dempster'):
        out_dir = os.path.join(args.output_dir, f'rule_{rule}')
        cfg = make_config(args, args.seed, out_dir, epochs, combination_rule=rule)
        set_seed(args.seed)
        train_loader = train_loader_for(args, args.seed)
        test_loader = get_test_loader(args.batch_size)
        model, info = train_enn(cfg, train_loader, test_loader, args.device)
        with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
            json.dump(info['history'], f, indent=2)
        per_ood = evaluate_ood(model, test_loader, ood_loaders, args.device, with_softmax=False)
        gn = info['history']['grad_norm']
        rows.append({
            'rule': rule,
            'epochs': epochs,
            'best_test_acc': info['best_test_acc'],
            'pre_clip_grad_norm_mean': float(np.mean(gn)) if gn else None,
            'pre_clip_grad_norm_max': float(np.max(gn)) if gn else None,
            'ece_max_belief': max_belief_ece(model, test_loader, args.device),
            'per_ood_enn_vacuity': {k: v['ENN-Vacuity'] for k, v in per_ood.items()},
        })
    syn = run_synthetic(args)
    with open(report_path, 'w') as f:
        json.dump({'runs': rows, 'synthetic': syn}, f, indent=2)
    print(f'wrote {report_path}')


def _vacuity_sweep(args, name: str, values, out_fmt: str, key: str, cfg_key: str):
    """Train one model per value of ``cfg_key``; report CIFAR-100 vacuity AUROC."""
    result_path = os.path.join(args.output_dir, name)
    if os.path.isfile(result_path):
        print(f'  {name}: results exist ({result_path})')
        return
    epochs = args.epochs or 60
    ood_loader = get_ood_loaders(args.batch_size, max_samples=args.ood_max_samples,
                                 names=['CIFAR-100'])['CIFAR-100']
    test_loader = get_test_loader(args.batch_size)
    rows = []
    for v in values:
        cfg = make_config(args, args.seed, os.path.join(args.output_dir, out_fmt.format(v)), epochs,
                          **{cfg_key: v})
        set_seed(args.seed)
        model, info = train_enn(cfg, train_loader_for(args, args.seed), test_loader, args.device)
        det = ENNVacuityDetector(model)
        id_s, ood_s = det.score(test_loader, args.device), det.score(ood_loader, args.device)
        rows.append({
            key: v,
            'test_acc': info['best_test_acc'],
            'AUROC_CIFAR100': ood_metrics(id_s, ood_s)['AUROC'],
            'id_mean_vacuity': float(np.mean(id_s)),
            'ood_mean_vacuity': float(np.mean(ood_s)),
        })
    with open(result_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


def run_ablation_w(args):
    _vacuity_sweep(args, 'ablation_prior_W.json', W_VALUES, 'W_{:g}', 'W', 'prior_weight')


def run_ablation_lambda(args):
    _vacuity_sweep(args, 'ablation_lambda_unc.json', LAMBDA_VALUES, 'lambda_{:g}', 'lambda_unc', 'lambda_unc')


def synthetic_combine_gradient_study(num_classes: int = 10, n_trials: int = 200) -> dict:
    """L2 norm of d m_comb({c_1}) / d m_1 for random mass pairs on F.

    m_1, m_2 are U[0,1)^{K+1} draws normalised to sum 1 (fresh draws for each
    rule). Dempster is evaluated without the (1 - kappa) clamp.
    """
    k = num_classes
    norms = {'yager': [], 'dempster': []}
    conflicts = []
    for _ in range(n_trials):
        for rule in ('yager', 'dempster'):
            m1 = torch.rand(1, k + 1)
            m2 = torch.rand(1, k + 1)
            m1 = (m1 / m1.sum()).requires_grad_(True)
            m2 = m2 / m2.sum()
            if rule == 'yager':
                out, kappa = yager_combine(m1, m2, k)
            else:
                out, kappa = dempster_combine(m1, m2, k, clamp_one_minus_k=False)
                conflicts.append(float(kappa.mean()))
            grad, = torch.autograd.grad(out[:, 0].sum(), m1)
            norms[rule].append(float(grad.norm()))
    return {
        'n_trials': n_trials,
        'num_classes': k,
        'mean_grad_norm_yager': float(np.mean(norms['yager'])),
        'mean_grad_norm_dempster': float(np.mean(norms['dempster'])),
        'max_grad_norm_yager': float(np.max(norms['yager'])),
        'max_grad_norm_dempster': float(np.max(norms['dempster'])),
        'mean_conflict': float(np.mean(conflicts)),
    }


def run_synthetic(args) -> dict:
    torch.manual_seed(args.seed)
    syn = synthetic_combine_gradient_study(num_classes=10, n_trials=args.synthetic_trials)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'synthetic_combine_gradients.json')
    with open(path, 'w') as f:
        json.dump(syn, f, indent=2)
    print(json.dumps(syn, indent=2))
    return syn


SUITES = {
    'main': run_main,
    'ablation-combine': run_ablation_combine,
    'ablation-W': run_ablation_w,
    'ablation-lambda': run_ablation_lambda,
    'synthetic': run_synthetic,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', default='main', choices=list(SUITES) + ['all'])
    p.add_argument('--output-dir', default='runs/cifar10')
    p.add_argument('--epochs', type=int, default=None,
                   help='training epochs (default: 200 for main, 60 for the other suites)')
    p.add_argument('--seed', type=int, default=0, help='first seed (main) / the seed (other suites)')
    p.add_argument('--seeds', type=int, default=1, help='number of consecutive seeds for --suite main')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-train-samples', type=int, default=None,
                   help='train on a random CIFAR-10 subset (debugging)')
    p.add_argument('--ood-max-samples', type=int, default=10000)
    p.add_argument('--synthetic-trials', type=int, default=200)
    p.add_argument('--device', default=None)
    p.add_argument('--quiet', action='store_true', help='no tqdm progress bars')
    args = p.parse_args()
    args.device = args.device or default_device()
    os.makedirs(args.output_dir, exist_ok=True)

    for name in (list(SUITES) if args.suite == 'all' else [args.suite]):
        if args.suite == 'all' and name == 'synthetic':
            continue  # already run inside ablation-combine
        print(f'\n=== {name} ===', flush=True)
        SUITES[name](args)


if __name__ == '__main__':
    main()
