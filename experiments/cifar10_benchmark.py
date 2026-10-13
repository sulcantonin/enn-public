#!/usr/bin/env python3
"""Table 1: five-seed CIFAR-10 OOD benchmark of MP-ENN against the baselines.

Per seed, 11 models are trained (MP-ENN-Yager, MP-ENN-Dempster, EDL, DUQ,
SNGP, MC Dropout, 5 ensemble members). The Softmax rows (MSP, Energy, ODIN)
are post-hoc scores on MP-ENN-Yager's pre-softplus head-0 logits, so they share
its accuracy and ECE. OOD sets: CIFAR-100, SVHN, MNIST, Fashion-MNIST, DTD,
Gaussian and Uniform noise; macro AUROC/FPR95 average over the seven.

With more than one visible GPU, every training job runs as its own subprocess
on one GPU; evaluation then reloads the checkpoints. Finished runs are skipped,
so an interrupted benchmark resumes where it stopped.

    python experiments/cifar10_benchmark.py --output-dir runs/benchmark
    python experiments/make_table1.py --bench-dir runs/benchmark

Outputs: seed_<s>/seed_results.json, table_core_ood.json,
table_id_quality.json, table_efficiency.json, run_config.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mpenn.baselines import (
    DeepEnsembleDetector,
    DUQDetector,
    EDLDetector,
    EnergyDetector,
    ENNVacuityDetector,
    MCDropoutDetector,
    MSPDetector,
    ODINDetector,
    SNGPDetector,
    train_duq_model,
    train_edl_model,
    train_sngp_model,
    train_softmax_classifier,
)
from mpenn.data import get_ood_loaders, get_test_loader, get_train_loader
from mpenn.metrics import expected_calibration_error, ood_metrics
from mpenn.training import DEFAULT_CONFIG, set_seed, train_enn

METHOD_ORDER = [
    'MP-ENN-Yager',
    'MP-ENN-Dempster',
    'Softmax+MSP',
    'Softmax+Energy',
    'Softmax+ODIN',
    'EDL',
    'DUQ',
    'SNGP',
    'MC Dropout',
    'Deep Ensemble',
]
SINGLE_MODEL_JOBS = ['enn_yager', 'enn_dempster', 'edl', 'duq', 'sngp', 'mc_dropout']
FORWARD_PASSES = {'MC Dropout': 30}   # Deep Ensemble: ensemble size; everything else: 1


# ─────────────────────────────────────────────────────────────────────────────
# Training (shared by the sequential path and the per-GPU workers)
# ─────────────────────────────────────────────────────────────────────────────
def _enn_config(rule: str, seed: int, seed_out: str, args) -> dict:
    return {**DEFAULT_CONFIG, 'combination_rule': rule, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'seed': seed,
            'output_dir': os.path.join(seed_out, f'enn_{rule}'), 'quiet_tqdm': args.quiet}


def _baseline_config(args, output_dir: str, **extra) -> dict:
    return {'num_classes': 10, 'epochs': args.epochs, 'lr': 1e-3, 'weight_decay': 1e-4,
            'pretrained': True, 'quiet_tqdm': args.quiet, 'output_dir': output_dir, **extra}


def train_job(job: str, seed: int, seed_out: str, train_loader, test_loader, device: str, args):
    """Train (or load, if its checkpoint exists) one model. Returns (model, info)."""
    if job in ('enn_yager', 'enn_dempster'):
        rule = job.split('_', 1)[1]
        return train_enn(_enn_config(rule, seed, seed_out, args), train_loader, test_loader, device)
    if job == 'edl':
        return train_edl_model(_baseline_config(args, os.path.join(seed_out, 'edl')),
                               train_loader, test_loader, device)
    if job == 'duq':
        return train_duq_model(_baseline_config(args, os.path.join(seed_out, 'duq')),
                               train_loader, test_loader, device)
    if job == 'sngp':
        return train_sngp_model(_baseline_config(args, os.path.join(seed_out, 'sngp')),
                                train_loader, test_loader, device)
    if job == 'mc_dropout':
        cfg = _baseline_config(args, os.path.join(seed_out, 'mc_dropout'), dropout_rate=0.1)
        return train_softmax_classifier(cfg, train_loader, test_loader, device)
    if job.startswith('ensemble_'):
        i = int(job.split('_', 1)[1])
        set_seed(seed * 100 + i)
        cfg = _baseline_config(args, os.path.join(seed_out, f'ensemble_member_{i}'), dropout_rate=0.0)
        return train_softmax_classifier(cfg, train_loader, test_loader, device)
    raise ValueError(f'unknown job: {job}')


def _job_checkpoint(job: str, seed_out: str) -> str:
    sub = f'ensemble_member_{job.split("_", 1)[1]}' if job.startswith('ensemble_') else job
    return os.path.join(seed_out, sub, 'best_model.pth')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _logit_model_stats(model, loader, device):
    """Accuracy, ECE (max softmax) and NLL for a model returning logits."""
    model.eval()
    confs, preds, labels, nll_sum = [], [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)['logits']
        probs = F.softmax(logits, dim=-1)
        confs.append(probs.max(dim=1).values.cpu().numpy())
        preds.append(probs.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        nll_sum += float(F.cross_entropy(logits, y, reduction='sum').item())
    confs, preds, labels = map(np.concatenate, (confs, preds, labels))
    return {'accuracy': 100.0 * float((preds == labels).mean()),
            'ece': expected_calibration_error(confs, preds, labels, n_bins=15),
            'nll': nll_sum / len(labels)}


@torch.no_grad()
def _enn_stats(model, loader, device):
    """Accuracy, ECE and NLL of the fused belief; confidence = max_i m({c_i}) (not renormalised)."""
    model.eval()
    confs, preds, labels, vacs, nll_sum = [], [], [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        bel = out['belief']
        confs.append(bel.max(dim=1).values.cpu().numpy())
        preds.append(bel.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        vacs.append(out['uncertainty'].cpu().numpy())
        m_true = bel[torch.arange(y.size(0), device=device), y]
        nll_sum += float((-torch.log(m_true.clamp_min(1e-8))).sum().item())
    confs, preds, labels, vacs = map(np.concatenate, (confs, preds, labels, vacs))
    return {'accuracy': 100.0 * float((preds == labels).mean()),
            'ece': expected_calibration_error(confs, preds, labels, n_bins=15),
            'nll': nll_sum / len(labels), 'mean_id_vacuity': float(vacs.mean())}


def _param_count(models) -> int:
    return int(sum(p.numel() for m in models for p in m.parameters() if p.requires_grad))


def _detector_latency(detector, loader, device, max_batches: int = 20) -> float:
    """Wall-clock ms per test batch (batch 128) over ``max_batches`` batches, data loading included."""
    start = time.perf_counter()
    batches = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(batch)
    detector.score(batches, device)
    return 1000.0 * (time.perf_counter() - start) / max(len(batches), 1)


def run_seed(seed: int, args) -> dict:
    """Train/load every method for one seed and score all OOD sets."""
    set_seed(seed)
    seed_out = os.path.join(args.output_dir, f'seed_{seed}')
    result_path = os.path.join(seed_out, 'seed_results.json')
    if os.path.isfile(result_path):
        print(f'  seed {seed}: results exist, loading {result_path}')
        with open(result_path) as f:
            return json.load(f)
    os.makedirs(seed_out, exist_ok=True)

    device = args.device
    if args.smoke_synthetic:
        train_loader, test_loader, ood_loaders = _synthetic_smoke_loaders(args.batch_size)
    else:
        train_loader = get_train_loader(args.batch_size, args.num_workers,
                                        max_train_samples=args.max_train_samples, subset_seed=seed)
        test_loader = get_test_loader(args.batch_size, args.num_workers)
        ood_loaders = get_ood_loaders(args.batch_size, args.num_workers, max_samples=args.ood_max_samples)

    detectors, id_stats, efficiency = {}, {}, {}

    print('  MP-ENN (Yager)', flush=True)
    enn_yager, _ = train_job('enn_yager', seed, seed_out, train_loader, test_loader, device, args)
    detectors['MP-ENN-Yager'] = ENNVacuityDetector(enn_yager)
    id_stats['MP-ENN-Yager'] = {**_enn_stats(enn_yager, test_loader, device),
                                'param_count': _param_count([enn_yager])}
    efficiency['MP-ENN-Yager'] = {'train_wall_time_s': None}

    print('  MP-ENN (Dempster)', flush=True)
    enn_dempster, _ = train_job('enn_dempster', seed, seed_out, train_loader, test_loader, device, args)
    detectors['MP-ENN-Dempster'] = ENNVacuityDetector(enn_dempster)
    id_stats['MP-ENN-Dempster'] = {**_enn_stats(enn_dempster, test_loader, device),
                                   'param_count': _param_count([enn_dempster])}
    efficiency['MP-ENN-Dempster'] = {'train_wall_time_s': None}

    # Post-hoc scores on MP-ENN-Yager: same model, so same ID accuracy / ECE.
    for name, det in (('Softmax+MSP', MSPDetector(enn_yager)),
                      ('Softmax+Energy', EnergyDetector(enn_yager)),
                      ('Softmax+ODIN', ODINDetector(enn_yager))):
        detectors[name] = det
        id_stats[name] = {k: id_stats['MP-ENN-Yager'][k] for k in ('accuracy', 'ece', 'nll', 'param_count')}
        efficiency[name] = {'train_wall_time_s': 0.0}

    for job, name, det_cls in (('edl', 'EDL', EDLDetector),
                               ('duq', 'DUQ', DUQDetector),
                               ('sngp', 'SNGP', SNGPDetector)):
        print(f'  {name}', flush=True)
        model, info = train_job(job, seed, seed_out, train_loader, test_loader, device, args)
        detectors[name] = det_cls(model)
        id_stats[name] = {**_logit_model_stats(model, test_loader, device), 'param_count': info.param_count}
        efficiency[name] = {'train_wall_time_s': info.train_wall_time_s}

    print('  MC Dropout', flush=True)
    mc_model, mc_info = train_job('mc_dropout', seed, seed_out, train_loader, test_loader, device, args)
    detectors['MC Dropout'] = MCDropoutDetector(mc_model, mc_samples=30)
    id_stats['MC Dropout'] = {**_logit_model_stats(mc_model, test_loader, device),
                              'param_count': mc_info.param_count}
    efficiency['MC Dropout'] = {'train_wall_time_s': mc_info.train_wall_time_s}

    print(f'  Deep Ensemble ({args.ensemble_size})', flush=True)
    members, ens_time = [], 0.0
    for i in range(args.ensemble_size):
        m, info = train_job(f'ensemble_{i}', seed, seed_out, train_loader, test_loader, device, args)
        members.append(m)
        ens_time += info.train_wall_time_s
    detectors['Deep Ensemble'] = DeepEnsembleDetector(members)
    # ID accuracy / ECE of the ensemble row are those of member 0.
    id_stats['Deep Ensemble'] = {**_logit_model_stats(members[0], test_loader, device),
                                 'param_count': _param_count(members)}
    efficiency['Deep Ensemble'] = {'train_wall_time_s': ens_time}

    print('  scoring OOD sets', flush=True)
    per_method = {}
    for method in METHOD_ORDER:
        det = detectors[method]
        id_scores = det.score(test_loader, device)
        per_ood = {name: ood_metrics(id_scores, det.score(loader, device))
                   for name, loader in ood_loaders.items()}
        per_method[method] = {
            'id': id_stats[method],
            'ood': per_ood,
            'macro_auroc': float(np.mean([m['AUROC'] for m in per_ood.values()])),
            'macro_fpr95': float(np.mean([m['FPR95'] for m in per_ood.values()])),
            'efficiency': efficiency[method],
        }
    for method in METHOD_ORDER:
        per_method[method]['efficiency']['latency_ms_per_batch'] = _detector_latency(
            detectors[method], test_loader, device)

    with open(result_path, 'w') as f:
        json.dump(per_method, f, indent=2)
    return per_method


def _mean_std(rows: Sequence[dict], fields: Sequence[str]) -> dict:
    out = {}
    for f in fields:
        vals = [r[f] for r in rows if r.get(f) is not None]
        out[f'{f}_mean'] = float(np.mean(vals)) if vals else None
        out[f'{f}_std'] = float(np.std(vals)) if vals else None
    return out


def aggregate(seed_rows: List[dict], out_dir: str, ensemble_size: int) -> None:
    """Mean and (population) std over seeds -> table_*.json."""
    core, id_quality, eff = {'methods': {}}, {'methods': {}}, {'methods': {}}
    for method in METHOD_ORDER:
        rows = [r[method] for r in seed_rows if method in r]
        if not rows:
            continue
        per_ood = {}
        for name in sorted(rows[0]['ood']):
            au = [x['ood'][name]['AUROC'] for x in rows]
            fp = [x['ood'][name]['FPR95'] for x in rows]
            per_ood[name] = {'AUROC_mean': float(np.mean(au)), 'AUROC_std': float(np.std(au)),
                             'FPR95_mean': float(np.mean(fp)), 'FPR95_std': float(np.std(fp))}
        core['methods'][method] = {'per_ood': per_ood, **_mean_std(rows, ['macro_auroc', 'macro_fpr95'])}

        id_rows = [x['id'] for x in rows]
        id_quality['methods'][method] = {
            **_mean_std(id_rows, ['accuracy', 'ece', 'nll']),
            'param_count_mean': float(np.mean([x['param_count'] for x in id_rows])),
        }

        erows = [x['efficiency'] for x in rows]
        times = [x['train_wall_time_s'] for x in erows if x['train_wall_time_s'] is not None]
        eff['methods'][method] = {
            'forward_passes': ensemble_size if method == 'Deep Ensemble' else FORWARD_PASSES.get(method, 1),
            'train_wall_time_s_mean': float(np.mean(times)) if times else None,
            'latency_ms_per_batch_mean': float(np.mean([x['latency_ms_per_batch'] for x in erows])),
            'latency_ms_per_batch_std': float(np.std([x['latency_ms_per_batch'] for x in erows])),
        }
    for name, obj in (('table_core_ood.json', core), ('table_id_quality.json', id_quality),
                      ('table_efficiency.json', eff)):
        with open(os.path.join(out_dir, name), 'w') as f:
            json.dump(obj, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU scheduling: one training subprocess per GPU
# ─────────────────────────────────────────────────────────────────────────────
def _all_jobs(args) -> List[str]:
    return SINGLE_MODEL_JOBS + [f'ensemble_{i}' for i in range(args.ensemble_size)]


def _pending_jobs(args) -> list:
    pending = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        seed_out = os.path.join(args.output_dir, f'seed_{seed}')
        pending += [(job, seed) for job in _all_jobs(args)
                    if not os.path.isfile(_job_checkpoint(job, seed_out))]
    return pending


def _worker(args) -> None:
    """Subprocess entry point: train one job for one seed on the single visible GPU."""
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else args.device
    seed = args._worker_seed
    set_seed(seed)
    seed_out = os.path.join(args.output_dir, f'seed_{seed}')
    os.makedirs(seed_out, exist_ok=True)
    train_loader = get_train_loader(args.batch_size, args.num_workers,
                                    max_train_samples=args.max_train_samples, subset_seed=seed)
    test_loader = get_test_loader(args.batch_size, args.num_workers)
    train_job(args._worker_job, seed, seed_out, train_loader, test_loader, device, args)


def run_parallel_training(args, num_gpus: int) -> None:
    pending = _pending_jobs(args)
    total = args.seeds * len(_all_jobs(args))
    if not pending:
        print('  all models already trained', flush=True)
        return
    print(f'  parallel training: {len(pending)}/{total} jobs on {num_gpus} GPUs', flush=True)

    active, failed = {}, []
    queue = list(pending)
    while queue or active:
        for gpu in range(num_gpus):
            if gpu in active or not queue:
                continue
            job, seed = queue.pop(0)
            seed_out = os.path.join(args.output_dir, f'seed_{seed}')
            os.makedirs(seed_out, exist_ok=True)
            cmd = [sys.executable, os.path.abspath(__file__), '--_worker',
                   '--_worker-job', job, '--_worker-seed', str(seed),
                   '--output-dir', args.output_dir, '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--ensemble-size', str(args.ensemble_size),
                   '--quiet']
            if args.max_train_samples is not None:
                cmd += ['--max-train-samples', str(args.max_train_samples)]
            if args.num_workers is not None:
                cmd += ['--num-workers', str(args.num_workers)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            log_f = open(os.path.join(seed_out, f'train_{job}.log'), 'w')
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            active[gpu] = (proc, job, seed, log_f)
            print(f'    [GPU {gpu}] {job} seed={seed}', flush=True)

        done = []
        for gpu, (proc, job, seed, log_f) in active.items():
            ret = proc.poll()
            if ret is None:
                continue
            log_f.close()
            if ret != 0:
                print(f'    [GPU {gpu}] FAILED {job} seed={seed} (exit {ret}, see {log_f.name})', flush=True)
                failed.append((job, seed, ret))
            else:
                print(f'    [GPU {gpu}] done   {job} seed={seed}', flush=True)
            done.append(gpu)
        for gpu in done:
            del active[gpu]
        if active and not done:
            time.sleep(2)

    if failed:
        raise RuntimeError('training failed for: ' + '; '.join(f'{j} seed={s} (exit {r})' for j, s, r in failed))


# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_smoke_loaders(batch_size: int):
    """Tiny random tensors standing in for every dataset (code-path check only)."""
    def loader(x, y, shuffle=False):
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    train = loader(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)), shuffle=True)
    test = loader(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
    ood = {}
    for name, scale in [('CIFAR-100', 1.1), ('SVHN', 1.2), ('MNIST', 1.3), ('FashionMNIST', 1.4),
                        ('Textures(DTD)', 1.5), ('Gaussian', 2.0), ('Uniform', None)]:
        x = torch.rand(64, 3, 32, 32) if scale is None else scale * torch.randn(64, 3, 32, 32)
        ood[name] = loader(x, torch.zeros(64, dtype=torch.long))
    return train, test, ood


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default='runs/benchmark')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--ensemble-size', type=int, default=5)
    p.add_argument('--ood-max-samples', type=int, default=10000)
    p.add_argument('--max-train-samples', type=int, default=None,
                   help='train on a random subset of CIFAR-10 (debugging)')
    p.add_argument('--num-workers', type=int, default=None,
                   help='DataLoader workers (default: MPENN_NUM_WORKERS or CPU affinity)')
    p.add_argument('--num-gpus', type=int, default=None,
                   help='GPUs for parallel training (default: all visible)')
    p.add_argument('--device', default=None)
    p.add_argument('--quiet', action='store_true', help='no tqdm progress bars')
    p.add_argument('--quick', action='store_true',
                   help='5 epochs, 2048 train images, 1024 OOD images, 1 seed, 2 ensemble members')
    p.add_argument('--smoke-synthetic', action='store_true',
                   help='random tensors instead of datasets (code-path check)')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--_worker-job', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_worker-seed', type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args._worker:
        _worker(args)
        return

    torch.backends.cudnn.benchmark = True
    if args.quick:
        args.epochs = min(args.epochs, 5)
        args.max_train_samples = args.max_train_samples or 2048
        args.ood_max_samples = min(args.ood_max_samples, 1024)
        args.ensemble_size = min(args.ensemble_size, 2)
        args.seeds = min(args.seeds, 1)

    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if args.device.startswith('cuda') else 1
    num_gpus = max(num_gpus, 1)
    os.makedirs(args.output_dir, exist_ok=True)

    if num_gpus > 1 and not args.smoke_synthetic:
        run_parallel_training(args, num_gpus)

    seed_rows = []
    for s in range(args.seed_start, args.seed_start + args.seeds):
        print(f'\n=== seed {s} ===', flush=True)
        seed_rows.append(run_seed(s, args))

    aggregate(seed_rows, args.output_dir, args.ensemble_size)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items() if not k.startswith('_')}, f, indent=2)
    print(f'\nwrote benchmark tables to {args.output_dir}')


if __name__ == '__main__':
    main()
