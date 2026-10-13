#!/usr/bin/env python3
"""Fine-tune RoBERTa-base + MP-ENN head (K = 3, W = 3, |S| = 3, Yager) on SNLI (Sect. 5.2).

Trained on hard SNLI labels only; the best epoch by SNLI validation accuracy
is kept. The paper's model: 2 epochs, batch size 64, AdamW (lr 2e-5, weight
decay 0.01), cosine schedule over all steps, gradient-norm clip 1.0,
max length 128 (91.7% SNLI validation accuracy).

    python experiments/chaosnli_train.py --epochs 2 --batch-size 64 --output-dir runs/chaosnli
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    raise SystemExit('ChaosNLI experiments need: pip install -e ".[nli]"')

from mpenn.nli import ENN_NLI, compute_enn_loss_nli

SNLI_HF_ID = 'stanfordnlp/snli'


class NLICollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        encoded = self.tokenizer(
            [ex['premise'] for ex in batch], [ex['hypothesis'] for ex in batch],
            padding=True, truncation=True, max_length=self.max_length, return_tensors='pt',
        )
        return {**encoded, 'labels': torch.tensor([ex['label'] for ex in batch], dtype=torch.long)}


def _forward(model, batch, device):
    token_type_ids = batch.get('token_type_ids')
    return model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                 token_type_ids.to(device) if token_type_ids is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', default='roberta-base')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-sources', type=int, default=3)
    p.add_argument('--combination-rule', default='yager', choices=['yager', 'dempster'])
    p.add_argument('--lambda-unc', type=float, default=0.1)
    p.add_argument('--lambda-aux', type=float, default=0.1)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-train-samples', type=int, default=None, help='debugging')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', default='runs/chaosnli')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, 'best_model.pth')
    config_path = os.path.join(args.output_dir, 'train_config.json')
    if os.path.isfile(best_path) and os.path.isfile(config_path):
        with open(config_path) as f:
            print(f'checkpoint exists ({best_path}); best val acc {json.load(f).get("best_val_acc")}%')
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone)
    hidden_dim = backbone.config.hidden_size
    model = ENN_NLI(backbone, hidden_dim=hidden_dim, num_sources=args.num_sources, prior_weight=3.0,
                    combination_rule=args.combination_rule, use_discounting=True).to(device)

    snli = load_dataset(SNLI_HF_ID)
    train_ds = snli['train'].filter(lambda x: x['label'] != -1)   # drop pairs without a gold label
    val_ds = snli['validation'].filter(lambda x: x['label'] != -1)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    collator = NLICollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collator, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_acc, history = 0.0, []
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        pbar = tqdm(train_loader, desc=f'epoch {epoch + 1}/{args.epochs}')
        for batch in pbar:
            labels = batch['labels'].to(device)
            outputs = _forward(model, batch, device)
            loss = compute_enn_loss_nli(outputs, labels, lambda_unc=args.lambda_unc,
                                        lambda_aux=args.lambda_aux)
            optimizer.zero_grad()
            loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            correct += (outputs['belief'].argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f'{loss["total"].item():.3f}', acc=f'{100 * correct / total:.1f}%')
        train_acc = 100 * correct / total

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['labels'].to(device)
                val_correct += (_forward(model, batch, device)['belief'].argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total
        print(f'  epoch {epoch + 1}: train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%')
        history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    with open(config_path, 'w') as f:
        json.dump({**vars(args), 'best_val_acc': best_acc, 'hidden_dim': hidden_dim}, f, indent=2)
    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'best SNLI validation accuracy {best_acc:.2f}%; checkpoint {best_path}')


if __name__ == '__main__':
    main()
