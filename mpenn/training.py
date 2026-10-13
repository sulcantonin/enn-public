"""Training / evaluation loops for MP-ENN on CIFAR-10."""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import ENN, compute_enn_loss

# Hyperparameters shared by every CIFAR-10 MP-ENN run in the paper (Sect. 5.1).
DEFAULT_CONFIG = {
    'num_classes': 10,
    'num_sources': 3,
    'prior_weight': None,          # None -> W = K
    'lambda_unc': 0.1,
    'lambda_aux': 0.1,
    'use_discounting': True,
    'symmetric_fusion': False,
    'combination_rule': 'yager',
    'head_hidden_dim': 256,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'quiet_tqdm': False,
}


def default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_enn(config: dict, pretrained: bool = True) -> ENN:
    return ENN(
        num_classes=config['num_classes'],
        num_sources=config['num_sources'],
        prior_weight=config['prior_weight'],
        use_discounting=config['use_discounting'],
        symmetric_fusion=config.get('symmetric_fusion', False),
        combination_rule=config.get('combination_rule', 'yager'),
        head_hidden_dim=config.get('head_hidden_dim', 256),
        pretrained=pretrained,
    )


def load_enn(checkpoint: str, device: str, **model_kwargs) -> ENN:
    """Build an ENN (no ImageNet download) and load a state_dict strictly."""
    model = ENN(pretrained=False, **model_kwargs).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def train_one_epoch(model, loader, optimizer, device, config, scaler=None) -> Tuple[dict, float]:
    """One epoch of Eq. 6 with gradient-norm clipping at 1.0.

    Returns (metrics, mean pre-clip gradient L2 norm over the epoch's batches).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    grad_norms = []
    use_amp = scaler is not None
    for data, target in tqdm(loader, desc='Train', leave=False, disable=config.get('quiet_tqdm', False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(data)
            loss = compute_enn_loss(outputs, target,
                                    lambda_unc=config['lambda_unc'],
                                    lambda_aux=config['lambda_aux'])
        if scaler is not None:
            scaler.scale(loss['total']).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss['total'].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        grad_norms.append(float(gn.detach()))
        total_loss += loss['total'].item()
        correct += (outputs['belief'].argmax(dim=1) == target).sum().item()
        total += target.size(0)
    return (
        {'loss': total_loss / len(loader), 'accuracy': 100.0 * correct / total},
        float(np.mean(grad_norms)) if grad_norms else 0.0,
    )


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    """Top-1 accuracy (%) of argmax_i Bel({c_i})."""
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        correct += (model(data)['belief'].argmax(dim=1) == target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


def train_enn(config: dict, train_loader, test_loader, device: str) -> Tuple[nn.Module, dict]:
    """Train MP-ENN with Adam + cosine LR; keep the epoch with the best test accuracy.

    Writes ``best_model.pth`` (state_dict) to ``config['output_dir']``. If that
    file already exists, training is skipped and the checkpoint is loaded.
    Returns (model, info) where info['history'] holds per-epoch train loss,
    test accuracy and mean pre-clip gradient norm.
    """
    out_dir = config['output_dir']
    best_path = os.path.join(out_dir, 'best_model.pth')
    os.makedirs(out_dir, exist_ok=True)

    model = build_enn(config).to(device)

    if os.path.isfile(best_path):
        print(f'  Skipping training, loading existing checkpoint: {best_path}')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        acc = evaluate_accuracy(model, test_loader, device)
        history = {'grad_norm': [], 'train_loss': [], 'test_acc': [acc]}
        return model, {'history': history, 'best_test_acc': acc, 'best_path': best_path}

    torch.backends.cudnn.benchmark = True
    opt = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
    scaler = torch.amp.GradScaler('cuda') if device.startswith('cuda') else None

    best_acc, best_state = 0.0, None
    history = {'grad_norm': [], 'train_loss': [], 'test_acc': []}
    for epoch in range(config['epochs']):
        tr, gn = train_one_epoch(model, train_loader, opt, device, config, scaler=scaler)
        acc = evaluate_accuracy(model, test_loader, device)
        sched.step()
        history['grad_norm'].append(gn)
        history['train_loss'].append(tr['loss'])
        history['test_acc'].append(acc)
        print(f'  epoch {epoch + 1:3d}/{config["epochs"]}  loss={tr["loss"]:.4f}  '
              f'test_acc={acc:.2f}  grad_norm={gn:.3f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_path)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, {'history': history, 'best_test_acc': best_acc, 'best_path': best_path}
