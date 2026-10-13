"""MP-ENN: neural networks with differentiable belief-function layers.

Code for "Neural Networks with Differentiable Belief-Function Layers"
(A. Sulc, BELIEF 2026).
"""

from .combine import (
    dempster_combine,
    discount_mass,
    fusion_combine_chain,
    fusion_yager_symmetric,
    yager_combine,
)
from .model import ENN, EvidenceToMass, compute_enn_loss

__all__ = [
    'ENN',
    'EvidenceToMass',
    'compute_enn_loss',
    'dempster_combine',
    'discount_mass',
    'fusion_combine_chain',
    'fusion_yager_symmetric',
    'yager_combine',
]
