"""MP-ENN head on a transformer encoder for 3-class NLI (ChaosNLI experiment).

Same construction as the vision model (evidence -> mass -> alpha-discount
-> left fold), with K = 3 and W = 3 by default. Heads read the first-token
([CLS]/<s>) representation of the encoder.
"""

import torch
import torch.nn as nn

from .combine import COMBINATION_RULES, discount_mass
from .model import EvidenceToMass

NLI_LABELS = ['entailment', 'neutral', 'contradiction']
NUM_CLASSES = 3


class ENN_NLI(nn.Module):
    """MP-ENN on top of a HuggingFace encoder that returns ``last_hidden_state``."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 768,
        num_sources: int = 3,
        prior_weight: float = 3.0,
        combination_rule: str = 'yager',
        use_discounting: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = NUM_CLASSES
        self.num_sources = num_sources
        self.prior_weight = prior_weight
        self.combination_rule = combination_rule
        self.use_discounting = use_discounting

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, NUM_CLASSES),
                nn.Softplus(),
            )
            for _ in range(num_sources)
        ])
        self.evidence_to_mass = EvidenceToMass(NUM_CLASSES, prior_weight)

        if use_discounting and num_sources > 1:
            self.reliability_net = nn.Sequential(
                nn.Linear(NUM_CLASSES * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        else:
            self.reliability_net = None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0, :]

        source_evidence = [head(hidden) for head in self.heads]
        mean_evidence = torch.stack(source_evidence, dim=0).mean(dim=0)
        source_masses = [self.evidence_to_mass(e) for e in source_evidence]

        # Head 0 is the undiscounted reference; heads 1.. are discounted by alpha_s.
        if self.reliability_net is not None:
            for s in range(1, self.num_sources):
                feat = torch.cat(
                    [source_evidence[s], mean_evidence, source_evidence[s] - mean_evidence],
                    dim=-1,
                )
                alpha = self.reliability_net(feat)
                source_masses[s] = discount_mass(source_masses[s], alpha, self.num_classes)

        combine = COMBINATION_RULES[self.combination_rule]
        combined = source_masses[0]
        total_conflict = torch.zeros(combined.size(0), device=combined.device)
        for s in range(1, self.num_sources):
            combined, k = combine(combined, source_masses[s], self.num_classes)
            total_conflict = total_conflict + k

        belief = combined[:, :self.num_classes]
        vacuity = combined[:, self.num_classes]
        return {
            'belief': belief,
            'plausibility': belief + vacuity.unsqueeze(1),
            'uncertainty': vacuity,
            'conflict': total_conflict,
            'combined_mass': combined,
            'source_masses': source_masses,
        }


def compute_enn_loss_nli(outputs, targets, lambda_unc=0.1, lambda_aux=0.1, eps=1e-8):
    """Eq. 6 for the NLI model (aux term on the discounted per-source masses)."""
    belief = outputs['belief']
    vacuity = outputs['uncertainty']
    source_masses = outputs['source_masses']
    batch, num_classes = belief.shape
    rows = torch.arange(batch, device=belief.device)

    task_loss = -torch.log(belief[rows, targets] + eps).mean()
    unc_loss = vacuity.mean()
    aux_loss = torch.tensor(0.0, device=belief.device)
    for sm in source_masses:
        aux_loss = aux_loss + (-torch.log(sm[:, :num_classes][rows, targets] + eps).mean())
    aux_loss = aux_loss / len(source_masses)

    total = task_loss + lambda_unc * unc_loss + lambda_aux * aux_loss
    return {'total': total, 'task': task_loss, 'unc': unc_loss, 'aux': aux_loss}
