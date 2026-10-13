#!/usr/bin/env python3
"""Corrupted-head stress test of the learned discount (Sect. 5.3, "Ablations").

MP-ENN is trained with Gaussian noise (std ``--sigma``) added to the encoder
features that feed evidence head ``--corrupt-idx`` (0-indexed; the paper's
"head 2" is index 1). The noise is resampled on every forward pass, in
training and at test time. In this experiment *all* heads, including head 0,
are discounted (``discount_all_sources=True``), so every head has an alpha_s.

Reported alpha_s is the mean over the training batches of an epoch (train mode).

    python experiments/source_degradation.py --sigma 1.5 --epochs 60 --seed 0 \
        --output-dir runs/degraded/seed0

Outputs: alpha_per_epoch.csv, final_alpha.csv, final_model.pth.
"""

import argparse
import csv
import os

import torch
import torch.optim as optim

from mpenn.data import get_test_loader, get_train_loader
from mpenn.model import ENN, compute_enn_loss
from mpenn.training import default_device, evaluate_accuracy, set_seed


class CorruptedENN(ENN):
    """ENN whose head ``corrupt_idx`` sees features + sigma * N(0, I)."""

    def __init__(self, *args, corrupt_idx: int = 1, sigma: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrupt_idx = corrupt_idx
        self.sigma = sigma

    def head_input(self, s, features):
        if s == self.corrupt_idx:
            return features + self.sigma * torch.randn_like(features)
        return features


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corrupt-idx', type=int, default=1)
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-sources', type=int, default=3)
    p.add_argument('--prior-weight', type=float, default=10.0)
    p.add_argument('--max-train-samples', type=int, default=None, help='debugging')
    p.add_argument('--device', default=None)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or default_device()

    model = CorruptedENN(
        num_classes=10, num_sources=args.num_sources, prior_weight=args.prior_weight,
        corrupt_idx=args.corrupt_idx, sigma=args.sigma, discount_all_sources=True,
    ).to(device)
    train_loader = get_train_loader(args.batch_size, max_train_samples=args.max_train_samples,
                                    subset_seed=args.seed)
    test_loader = get_test_loader(args.batch_size)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(os.path.join(args.output_dir, 'alpha_per_epoch.csv'), 'w', newline='') as log_f:
        log = csv.writer(log_f)
        log.writerow(['epoch', 'train_loss', 'test_acc', *(f'alpha_s{i}' for i in range(args.num_sources))])
        for epoch in range(args.epochs):
            model.train()
            running_loss, alpha_sum, n = 0.0, [0.0] * args.num_sources, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = compute_enn_loss(out, y)
                opt.zero_grad()
                loss['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running_loss += float(loss['total'].detach())
                for i in range(args.num_sources):
                    alpha_sum[i] += float(out['source_alphas'][i].mean().detach())
                n += 1
            sched.step()
            test_acc = evaluate_accuracy(model, test_loader, device)
            alphas = [a / max(n, 1) for a in alpha_sum]
            log.writerow([epoch, running_loss / max(n, 1), test_acc, *alphas])
            log_f.flush()
            print(f'[epoch {epoch:3d}] loss={running_loss / max(n, 1):.4f} acc={test_acc:.2f} alpha={alphas}',
                  flush=True)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    final_csv = os.path.join(args.output_dir, 'final_alpha.csv')
    with open(final_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'final_alpha', 'corrupted'])
        for i, a in enumerate(alphas):
            w.writerow([i, a, int(i == args.corrupt_idx)])
    print(f'wrote {final_csv}')


if __name__ == '__main__':
    main()
