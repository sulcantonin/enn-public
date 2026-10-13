"""Belief combination on the focal set F = {{c_1}, ..., {c_K}, Theta}.

A mass function is stored as a (B, K+1) tensor: columns 0..K-1 hold the
singleton masses m({c_i}), column K holds the ignorance mass m(Theta).

Pure PyTorch; no torchvision dependency.
"""

from itertools import permutations

import torch


def _intersection_terms(m1, m2, num_classes):
    """Unnormalised intersection on F and the conflict mass kappa.

    Returns (m_tilde_singletons, kappa) where
        m_tilde({c_i}) = m1(c_i) m2(c_i) + m1(c_i) m2(Theta) + m1(Theta) m2(c_i)
        kappa          = sum_{i != j} m1(c_i) m2(c_j).
    """
    m1_s, m1_theta = m1[:, :num_classes], m1[:, num_classes:]
    m2_s, m2_theta = m2[:, :num_classes], m2[:, num_classes:]

    m_tilde_s = m1_s * m2_s + m1_s * m2_theta + m1_theta * m2_s

    sum_m1 = m1_s.sum(dim=1, keepdim=True)
    sum_m2 = m2_s.sum(dim=1, keepdim=True)
    dot_prod = (m1_s * m2_s).sum(dim=1, keepdim=True)
    kappa = sum_m1 * sum_m2 - dot_prod
    return m_tilde_s, kappa


def yager_combine(m1, m2, num_classes):
    """Yager's rule (paper Eqs. 3-4): conflict is reassigned to Theta.

    Returns the combined (B, K+1) mass and the conflict kappa (B,).
    """
    m_new_s, kappa = _intersection_terms(m1, m2, num_classes)
    m_new_theta = m1[:, num_classes:] * m2[:, num_classes:] + kappa
    combined = torch.cat([m_new_s, m_new_theta], dim=1)
    return combined, kappa.squeeze(-1)


def dempster_combine(m1, m2, num_classes, eps=1e-8, clamp_one_minus_k: bool = True):
    """Dempster's rule: the intersection renormalised by 1/(1 - kappa).

    ``clamp_one_minus_k=True`` clamps (1 - kappa) at ``eps`` and is what all
    end-to-end Dempster training uses. Pass ``False`` to expose the raw
    (1 - kappa)^-1 behaviour (gradient-vs-conflict study, synthetic study).

    Returns the combined (B, K+1) mass and the conflict kappa (B,).
    """
    m_tilde_s, kappa = _intersection_terms(m1, m2, num_classes)
    m_tilde_theta = m1[:, num_classes:] * m2[:, num_classes:]

    one_minus_k = 1.0 - kappa
    if clamp_one_minus_k:
        one_minus_k = one_minus_k.clamp(min=eps)
    combined = torch.cat([m_tilde_s / one_minus_k, m_tilde_theta / one_minus_k], dim=1)
    return combined, kappa.squeeze(-1)


COMBINATION_RULES = {'yager': yager_combine, 'dempster': dempster_combine}


def fusion_combine_chain(source_masses, num_classes, rule='yager'):
    """Left fold over the sources in index order (paper Eq. 5).

    Returns (combined mass, summed conflict over the |S|-1 steps, list of
    per-step conflicts).
    """
    combine = COMBINATION_RULES[rule]
    combined = source_masses[0]
    total_conflict = torch.zeros(combined.size(0), device=combined.device)
    pairwise_conflicts = []
    for i in range(1, len(source_masses)):
        combined, k_i = combine(combined, source_masses[i], num_classes)
        total_conflict = total_conflict + k_i
        pairwise_conflicts.append(k_i)
    return combined, total_conflict, pairwise_conflicts


def fusion_yager_symmetric(source_masses, num_classes):
    """Order-free variant: average the Yager left fold over all |S|! orderings."""
    num_sources = len(source_masses)
    if num_sources == 0:
        raise ValueError('fusion_yager_symmetric: empty source list')
    batch = source_masses[0].size(0)
    device = source_masses[0].device
    if num_sources == 1:
        return source_masses[0], torch.zeros(batch, device=device), []

    acc = torch.zeros_like(source_masses[0])
    k_acc = torch.zeros(batch, device=device)
    n_perm = 0
    for perm in permutations(range(num_sources)):
        m = source_masses[perm[0]]
        total_k = torch.zeros(batch, device=device)
        for j in perm[1:]:
            m, k_i = yager_combine(m, source_masses[j], num_classes)
            total_k = total_k + k_i
        acc = acc + m
        k_acc = k_acc + total_k
        n_perm += 1
    return acc / n_perm, k_acc / n_perm, []


def discount_mass(mass, alpha, num_classes):
    """Shafer's alpha-discount on F (paper Sect. 3.4).

    m^alpha(F) = alpha m(F) for F != Theta, and m^alpha(Theta) = 1 - alpha (1 - m(Theta)).
    ``alpha`` broadcasts against the batch, e.g. shape (B, 1).
    """
    m_s = mass[:, :num_classes]
    m_theta = mass[:, num_classes:]
    return torch.cat([alpha * m_s, 1.0 - alpha * (1.0 - m_theta)], dim=1)


def assert_valid_mass(mass: torch.Tensor, atol: float = 1e-5) -> None:
    """Raise if ``mass`` has negative entries or rows that do not sum to 1."""
    if (mass < -atol).any():
        raise AssertionError(f"mass has negative entries (min={mass.min().item()})")
    s = mass.sum(dim=-1)
    if not torch.allclose(s, torch.ones_like(s), atol=atol):
        raise AssertionError(
            f"mass does not sum to 1 (mean={s.mean().item():.6f}, "
            f"max abs dev={float((s - 1).abs().max()):.2e})"
        )
