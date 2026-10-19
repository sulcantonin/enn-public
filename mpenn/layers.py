"""Drop-in belief-function layers for any feature extractor.

``BeliefLayer`` maps a ``(B, D)`` feature tensor to a Dempster-Shafer mass
function on F = {{c_1}, ..., {c_K}, Theta}: |S| evidence heads, Eq. 1, a learned
alpha-discount, and a Yager (or Dempster) fold. ``ENN`` in ``model.py`` is this
layer on top of a ResNet-18; use ``BeliefLayer`` directly to put it on top of
your own backbone.

Only depends on PyTorch.
"""

import os

import torch
import torch.nn as nn

from .combine import (
    assert_valid_mass,
    discount_mass,
    fusion_combine_chain,
    fusion_yager_symmetric,
)

# Set MPENN_DEBUG=1 to check the mass-function axioms on every forward pass.
_DEBUG = bool(int(os.environ.get('MPENN_DEBUG', '0')))


class EvidenceToMass(nn.Module):
    """Eq. 1: m({c_i}) = e_i / (E + W),  m(Theta) = W / (E + W),  E = sum_i e_i."""

    def __init__(self, num_classes: int, prior_weight: float):
        super().__init__()
        self.num_classes = num_classes
        self.prior_weight = prior_weight

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        total = evidence.sum(dim=-1, keepdim=True)
        m_singletons = evidence / (total + self.prior_weight)
        m_theta = self.prior_weight / (total + self.prior_weight)
        return torch.cat([m_singletons, m_theta], dim=-1)


class BeliefLayer(nn.Module):
    """Features ``(B, D)`` -> fused mass function, belief, plausibility, vacuity.

    Parameters
    ----------
    in_features : D, the feature dimension of the backbone.
    num_classes : K.
    num_sources : number of evidence heads |S|.
    prior_weight : W in Eq. 1; defaults to K.
    use_discounting : learn alpha_s with the reliability gate g_phi.
    symmetric_fusion : average the Yager fold over all |S|! orderings.
    combination_rule : 'yager' or 'dempster'.
    head_hidden_dim : width of the hidden layer of each evidence head.
    discount_all_sources : also discount head 0 (the paper keeps alpha_1 = 1).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 10,
        num_sources: int = 3,
        prior_weight: float = None,
        use_discounting: bool = True,
        symmetric_fusion: bool = False,
        combination_rule: str = 'yager',
        head_hidden_dim: int = 256,
        discount_all_sources: bool = False,
    ):
        super().__init__()
        if combination_rule not in ('yager', 'dempster'):
            raise ValueError("combination_rule must be 'yager' or 'dempster'")
        if symmetric_fusion and combination_rule != 'yager':
            raise ValueError('symmetric_fusion is only implemented for Yager')
        self.feature_dim = in_features
        self.num_classes = num_classes
        self.num_sources = num_sources
        self.prior_weight = prior_weight if prior_weight is not None else float(num_classes)
        self.use_discounting = use_discounting
        self.symmetric_fusion = symmetric_fusion
        self.combination_rule = combination_rule
        self.discount_all_sources = discount_all_sources

        # Evidence heads: non-negative evidence e_{i,s} via a final softplus.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden_dim, num_classes),
                nn.Softplus(),
            )
            for _ in range(num_sources)
        ])

        # Reliability gate: alpha_s = sigmoid(g_phi([e_s; e_bar; e_s - e_bar])).
        if use_discounting:
            self.reliability_net = nn.Sequential(
                nn.Linear(num_classes * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        self.evidence_to_mass = EvidenceToMass(num_classes, self.prior_weight)

    def reliability(self, evidence_s: torch.Tensor, mean_evidence: torch.Tensor) -> torch.Tensor:
        """Per-sample alpha_s in (0, 1), shape (B, 1)."""
        feat = torch.cat([evidence_s, mean_evidence, evidence_s - mean_evidence], dim=-1)
        return self.reliability_net(feat)

    def head_input(self, s: int, features: torch.Tensor) -> torch.Tensor:
        """Features fed to head ``s``; override to corrupt or reroute a head."""
        return features

    def forward(self, features: torch.Tensor):
        # All evidence vectors first: the gate needs the consensus e_bar.
        source_evidences = [head(self.head_input(s, features)) for s, head in enumerate(self.heads)]
        mean_evidence = torch.stack(source_evidences, dim=0).mean(dim=0)

        source_masses, source_alphas = [], []
        for s, evidence in enumerate(source_evidences):
            mass = self.evidence_to_mass(evidence)
            if self.use_discounting and (self.discount_all_sources or s > 0):
                alpha = self.reliability(evidence, mean_evidence)
                mass = discount_mass(mass, alpha, self.num_classes)
                source_alphas.append(alpha.squeeze(-1))
            else:
                source_alphas.append(torch.ones(features.size(0), device=features.device))
            if _DEBUG:
                assert_valid_mass(mass)
            source_masses.append(mass)

        if self.symmetric_fusion:
            combined, total_conflict, pairwise_conflicts = fusion_yager_symmetric(
                source_masses, self.num_classes)
        else:
            combined, total_conflict, pairwise_conflicts = fusion_combine_chain(
                source_masses, self.num_classes, rule=self.combination_rule)
        if _DEBUG:
            assert_valid_mass(combined)

        belief = combined[:, :self.num_classes]          # Bel({c_i}) = m({c_i})
        vacuity = combined[:, self.num_classes]          # m(Theta)
        plausibility = belief + vacuity.unsqueeze(-1)    # Pl({c_i}) = m({c_i}) + m(Theta)

        return {
            'mass': combined,                                  # (B, K+1)
            'belief': belief,                                  # (B, K)
            'plausibility': plausibility,                      # (B, K)
            'uncertainty': vacuity,                            # (B,)
            'evidence': torch.stack(source_evidences, dim=0),  # (S, B, K)
            'source_masses': source_masses,                    # S x (B, K+1), after discounting
            'source_alphas': source_alphas,                    # S x (B,)
            'conflict': total_conflict,                        # (B,), summed over fold steps
            'pairwise_conflicts': pairwise_conflicts,          # (S-1) x (B,)
        }
