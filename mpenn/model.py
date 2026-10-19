"""MP-ENN for CIFAR-10 (paper Sect. 3).

Pipeline: ResNet-18 encoder -> |S| evidence heads -> mass on F (Eq. 1)
-> learned alpha-discount (Sect. 3.4) -> Yager (or Dempster) left fold (Eq. 5)
-> output mass, [Bel, Pl] and vacuity m(Theta).
"""

import torch
import torch.nn as nn

from .layers import BeliefLayer, EvidenceToMass  # noqa: F401


def make_cifar_resnet18(pretrained: bool = True):
    """torchvision ResNet-18 adapted to 32x32 inputs; returns (backbone, feature_dim).

    With ``pretrained=True`` the ImageNet-1k weights are loaded; the replaced
    3x3 stem convolution is always randomly initialised.
    """
    import torchvision  # lazy: only the CIFAR ResNet needs it

    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = torchvision.models.resnet18(weights=weights)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class ENN(BeliefLayer):
    """Mass-Primitive Evidential Neural Network: ResNet-18 + ``BeliefLayer``.

    Takes the arguments of ``BeliefLayer`` (minus ``in_features``) plus:

    pretrained : load ImageNet weights into the encoder. Irrelevant when a
        trained checkpoint is loaded afterwards, so evaluation code passes False
        to avoid the download.
    """

    def __init__(self, num_classes: int = 10, num_sources: int = 3, pretrained: bool = True, **kwargs):
        backbone, feature_dim = make_cifar_resnet18(pretrained)
        super().__init__(feature_dim, num_classes, num_sources, **kwargs)
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone(x)
        out = super().forward(features)
        out['features'] = features                             # (B, D)
        return out


def compute_enn_loss(outputs, targets, lambda_unc: float = 0.1, lambda_aux: float = 0.1):
    """Training objective (paper Eq. 6).

        L = -log(m({c_y}) + eps) + lambda_unc * m(Theta)
            + (lambda_aux / |S|) * sum_s -log(m_s({c_y}) + eps)

    The per-source term is evaluated on ``outputs['source_masses']``, i.e. on
    the masses after alpha-discounting (head 0 is undiscounted).
    """
    belief = outputs['belief']
    vacuity = outputs['uncertainty']
    batch = targets.shape[0]
    rows = torch.arange(batch, device=targets.device)
    eps = 1e-8

    loss_task = -torch.log(belief[rows, targets] + eps).mean()
    loss_unc = lambda_unc * vacuity.mean()

    loss_aux = torch.tensor(0.0, device=targets.device)
    for m in outputs['source_masses']:
        loss_aux = loss_aux + (-torch.log(m[:, :-1][rows, targets] + eps).mean())
    loss_aux = lambda_aux * loss_aux / len(outputs['source_masses'])

    total = loss_task + loss_unc + loss_aux
    return {
        'total': total,
        'task': loss_task.item(),
        'unc': loss_unc.item(),
        'aux': loss_aux.item(),
    }
