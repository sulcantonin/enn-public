"""Baseline models, their trainers, and the OOD detectors used in Table 1.

Every detector's ``score`` follows the same convention: higher = more OOD.
All baselines share the MP-ENN encoder (ImageNet-initialised ResNet-18 with a
3x3 stem) and the MP-ENN optimiser settings.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from .model import make_cifar_resnet18


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainInfo:
    best_test_acc: float
    best_path: str
    train_wall_time_s: float
    param_count: int
    history: Dict[str, List[float]]


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class SoftmaxClassifier(nn.Module):
    """ResNet-18 + linear softmax head (MC Dropout model and ensemble members)."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = make_cifar_resnet18(pretrained)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        return {'logits': self.classifier(self.dropout(features)), 'features': features}


class EDLModel(nn.Module):
    """Evidential deep learning (Sensoy et al., 2018): Dirichlet alpha = softplus(z) + 1."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = make_cifar_resnet18(pretrained)
        self.evidence_head = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        evidence = F.softplus(self.evidence_head(features))
        alpha = evidence + 1.0
        probs = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        uncertainty = self.num_classes / alpha.sum(dim=-1).clamp_min(1e-8)
        return {
            'features': features,
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty,
            'logits': torch.log(probs.clamp_min(1e-8)),
        }


def _edl_kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) )."""
    ones = torch.ones((1, num_classes), device=alpha.device, dtype=alpha.dtype)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return (first + second).squeeze(-1)


def edl_loss(alpha: torch.Tensor, y: torch.Tensor, epoch: int, annealing_steps: int = 10) -> torch.Tensor:
    """Digamma (Bayes risk of cross-entropy) loss + annealed KL regulariser."""
    num_classes = alpha.size(1)
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    s = alpha.sum(dim=1, keepdim=True)
    loglikelihood = torch.sum(y_onehot * (torch.digamma(s) - torch.digamma(alpha)), dim=1)
    anneal = min(1.0, epoch / max(annealing_steps, 1))
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    return (loglikelihood + anneal * _edl_kl_divergence(alpha_tilde, num_classes)).mean()


class DUQModel(nn.Module):
    """DUQ-style model (van Amersfoort et al., 2020): RBF kernel to learned class prototypes."""

    def __init__(self, num_classes: int = 10, embedding_dim: int = 128, sigma: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = make_cifar_resnet18(pretrained)
        self.embedding = nn.Linear(feat_dim, embedding_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.05)
        self.sigma = sigma
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        z = self.embedding(features)
        dists = torch.cdist(z, self.prototypes, p=2.0).pow(2)
        kernel = torch.exp(-dists / (2 * (self.sigma ** 2)))
        kernel_norm = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {
            'features': features,
            'embedding': z,
            'kernel_distances': kernel,
            'probs': kernel_norm,
            'logits': torch.log(kernel_norm.clamp_min(1e-8)),
        }


def duq_gradient_penalty(model: DUQModel, x: torch.Tensor) -> torch.Tensor:
    """Two-sided penalty (||d max_c K_c / dx|| - 1)^2."""
    x = x.clone().detach().requires_grad_(True)
    score = model(x)['kernel_distances'].max(dim=1).values
    grad = torch.autograd.grad(score.sum(), x, create_graph=True)[0].flatten(start_dim=1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


class RandomFeatureGPLayer(nn.Module):
    """Random Fourier features followed by a linear output layer."""

    def __init__(self, in_dim: int, out_dim: int, num_features: int = 1024, lengthscale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('omega', torch.randn(in_dim, num_features) / max(lengthscale, 1e-8))
        self.register_buffer('bias', 2 * math.pi * torch.rand(num_features))
        self.beta = nn.Linear(num_features, out_dim)

    def forward(self, x):
        phi = math.sqrt(2.0 / self.num_features) * torch.cos(x @ self.omega + self.bias)
        return {'gp_mean': self.beta(phi), 'phi': phi}


class SNGPModel(nn.Module):
    """SNGP-style model (Liu et al., 2020): spectral-normalised encoder + RFF output layer."""

    def __init__(self, num_classes: int = 10, gp_features: int = 1024, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = make_cifar_resnet18(pretrained)
        for module in self.backbone.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    spectral_norm(module)
                except Exception:
                    pass
        self.gp_layer = RandomFeatureGPLayer(feat_dim, num_classes, num_features=gp_features)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        logits = self.gp_layer(features)['gp_mean']
        return {'features': features, 'gp_mean': logits, 'logits': logits}


# ─────────────────────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────────────────────
class BaseDetector:
    name = 'Detector'

    def score(self, loader, device: str) -> np.ndarray:
        raise NotImplementedError


def enn_head_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pre-softplus output of MP-ENN head 0, used as logits by the post-hoc Softmax scores."""
    return model.heads[0][:-1](model.backbone(x))


class ENNVacuityDetector(BaseDetector):
    """MP-ENN output vacuity m(Theta)."""
    name = 'MP-ENN-Vacuity'

    def __init__(self, enn_model):
        self.model = enn_model

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        return np.concatenate([self.model(x.to(device))['uncertainty'].cpu().numpy() for x, _ in loader])


class MSPDetector(BaseDetector):
    """1 - max softmax probability (Hendrycks & Gimpel, 2017) on MP-ENN head-0 logits."""
    name = 'Softmax+MSP'

    def __init__(self, enn_model):
        self.model = enn_model

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        out = []
        for x, _ in loader:
            probs = F.softmax(enn_head_logits(self.model, x.to(device)), dim=-1)
            out.append((1.0 - probs.max(dim=1).values).cpu().numpy())
        return np.concatenate(out)


class EnergyDetector(BaseDetector):
    """Energy score -T logsumexp(z / T) (Liu et al., 2020) on MP-ENN head-0 logits."""
    name = 'Softmax+Energy'

    def __init__(self, enn_model, temperature: float = 1.0):
        self.model = enn_model
        self.temperature = temperature

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        t = self.temperature
        out = []
        for x, _ in loader:
            logits = enn_head_logits(self.model, x.to(device))
            out.append((-t * torch.logsumexp(logits / t, dim=1)).cpu().numpy())
        return np.concatenate(out)


class ODINDetector(BaseDetector):
    """ODIN (Liang et al., 2018), T = 1000, eps = 0.0014, on MP-ENN head-0 logits."""
    name = 'Softmax+ODIN'

    def __init__(self, enn_model, temperature: float = 1000.0, eps: float = 0.0014):
        self.model = enn_model
        self.temperature = temperature
        self.eps = eps

    def score(self, loader, device):
        self.model.eval()
        out = []
        for x, _ in loader:
            x = x.to(device).clone().detach().requires_grad_(True)
            scaled = enn_head_logits(self.model, x) / self.temperature
            loss = F.cross_entropy(scaled, scaled.argmax(dim=1))
            grad = torch.autograd.grad(loss, x)[0]
            x_pert = (x - self.eps * torch.sign(grad)).detach()
            with torch.no_grad():
                probs = F.softmax(enn_head_logits(self.model, x_pert) / self.temperature, dim=-1)
                out.append((1.0 - probs.max(dim=1).values).cpu().numpy())
        return np.concatenate(out)


class EDLDetector(BaseDetector):
    """EDL vacuity K / sum(alpha)."""
    name = 'EDL'

    def __init__(self, model: EDLModel):
        self.model = model

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        return np.concatenate([self.model(x.to(device))['uncertainty'].cpu().numpy() for x, _ in loader])


class DUQDetector(BaseDetector):
    """1 - max_c K_c(x)."""
    name = 'DUQ'

    def __init__(self, model: DUQModel):
        self.model = model

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        out = []
        for x, _ in loader:
            k = self.model(x.to(device))['kernel_distances']
            out.append((1.0 - k.max(dim=1).values).cpu().numpy())
        return np.concatenate(out)


class SNGPDetector(BaseDetector):
    """1 - max softmax of the SNGP output layer."""
    name = 'SNGP'

    def __init__(self, model: SNGPModel):
        self.model = model

    @torch.no_grad()
    def score(self, loader, device):
        self.model.eval()
        out = []
        for x, _ in loader:
            probs = F.softmax(self.model(x.to(device))['gp_mean'], dim=-1)
            out.append((1.0 - probs.max(dim=1).values).cpu().numpy())
        return np.concatenate(out)


def _predictive_entropy(prob_list) -> torch.Tensor:
    mean_probs = torch.stack(prob_list, dim=0).mean(dim=0)
    return -(mean_probs * torch.log(mean_probs.clamp_min(1e-8))).sum(dim=1)


class MCDropoutDetector(BaseDetector):
    """Entropy of the mean of ``mc_samples`` stochastic softmax passes.

    The model is put in train mode so dropout is active (BatchNorm then also
    uses batch statistics).
    """
    name = 'MC Dropout'

    def __init__(self, model: SoftmaxClassifier, mc_samples: int = 30):
        self.model = model
        self.mc_samples = mc_samples

    def score(self, loader, device):
        self.model.train()
        out = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                probs = [F.softmax(self.model(x)['logits'], dim=-1) for _ in range(self.mc_samples)]
                out.append(_predictive_entropy(probs).cpu().numpy())
        return np.concatenate(out)


class DeepEnsembleDetector(BaseDetector):
    """Entropy of the mean softmax over the ensemble members."""
    name = 'Deep Ensemble'

    def __init__(self, models: Sequence[SoftmaxClassifier]):
        self.models = list(models)

    def score(self, loader, device):
        for m in self.models:
            m.eval()
        out = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                probs = [F.softmax(m(x)['logits'], dim=-1) for m in self.models]
                out.append(_predictive_entropy(probs).cpu().numpy())
        return np.concatenate(out)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def _eval_classifier(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    """Accuracy (%) and mean NLL of ``model(x)['logits']``."""
    model.eval()
    total, correct, nll_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)['logits']
            nll_sum += float(F.cross_entropy(logits, y, reduction='sum').item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))
    return 100.0 * correct / max(total, 1), nll_sum / max(total, 1)


def _train_classifier_model(model, train_loader, test_loader, device, config, loss_fn) -> TrainInfo:
    """Adam + cosine LR + grad-norm clip 1.0, keeping the best-test-accuracy epoch.

    Skips training if the checkpoint already exists in ``config['output_dir']``.
    """
    torch.backends.cudnn.benchmark = True
    lr = float(config.get('lr', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-4))
    epochs = int(config.get('epochs', 200))
    quiet = bool(config.get('quiet_tqdm', False))
    out_dir = config['output_dir']
    ckpt_path = os.path.join(out_dir, config.get('checkpoint_name', 'best_model.pth'))
    lambda_gp = float(config.get('duq_lambda_gp', 0.5))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(ckpt_path):
        print(f'  Skipping training, loading existing checkpoint: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        test_acc, _ = _eval_classifier(model, test_loader, device)
        return TrainInfo(float(test_acc), ckpt_path, 0.0, _param_count(model),
                         {'train_loss': [], 'test_acc': [test_acc]})

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    # DUQ's gradient penalty needs double backprop, so it trains in fp32.
    use_amp = device.startswith('cuda') and loss_fn != 'duq'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc, best_state = -1.0, None
    history = {'train_loss': [], 'test_acc': []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x, y in tqdm(train_loader, desc='Train', leave=False, disable=quiet):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                if loss_fn == 'ce':
                    loss = F.cross_entropy(model(x)['logits'], y)
                elif loss_fn == 'edl':
                    loss = edl_loss(model(x)['alpha'], y, epoch=epoch,
                                    annealing_steps=max(epochs // 10, 1))
                elif loss_fn == 'duq':
                    probs = model(x)['probs']
                    loss = (F.nll_loss(torch.log(probs.clamp_min(1e-8)), y)
                            + lambda_gp * duq_gradient_penalty(model, x))
                else:
                    raise ValueError(f'Unknown loss_fn: {loss_fn}')
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            loss_sum += float(loss.item()) * int(y.size(0))
            n_seen += int(y.size(0))

        scheduler.step()
        test_acc, _ = _eval_classifier(model, test_loader, device)
        history['train_loss'].append(loss_sum / max(n_seen, 1))
        history['test_acc'].append(test_acc)
        print(f'  epoch {epoch:3d}/{epochs}  loss={history["train_loss"][-1]:.4f}  '
              f'test_acc={test_acc:.2f}', flush=True)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), ckpt_path)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return TrainInfo(float(best_acc), ckpt_path, float(time.time() - t0), _param_count(model), history)


def train_softmax_classifier(config, train_loader, test_loader, device):
    model = SoftmaxClassifier(
        num_classes=config.get('num_classes', 10),
        dropout_rate=float(config.get('dropout_rate', 0.0)),
        pretrained=bool(config.get('pretrained', True)),
    ).to(device)
    return model, _train_classifier_model(model, train_loader, test_loader, device, config, 'ce')


def train_edl_model(config, train_loader, test_loader, device):
    model = EDLModel(
        num_classes=config.get('num_classes', 10),
        pretrained=bool(config.get('pretrained', True)),
    ).to(device)
    return model, _train_classifier_model(model, train_loader, test_loader, device, config, 'edl')


def train_duq_model(config, train_loader, test_loader, device):
    model = DUQModel(
        num_classes=config.get('num_classes', 10),
        embedding_dim=int(config.get('duq_embedding_dim', 128)),
        sigma=float(config.get('duq_sigma', 0.3)),
        pretrained=bool(config.get('pretrained', True)),
    ).to(device)
    return model, _train_classifier_model(model, train_loader, test_loader, device, config, 'duq')


def train_sngp_model(config, train_loader, test_loader, device):
    model = SNGPModel(
        num_classes=config.get('num_classes', 10),
        gp_features=int(config.get('sngp_gp_features', 1024)),
        pretrained=bool(config.get('pretrained', True)),
    ).to(device)
    return model, _train_classifier_model(model, train_loader, test_loader, device, config, 'ce')
