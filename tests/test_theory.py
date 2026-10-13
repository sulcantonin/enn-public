"""Numerical checks of the structural results (paper Sects. 3-4).

    pytest tests/          # or: python tests/test_theory.py

Covers: validity of every mass in the pipeline, Bel <= Pl and duality on F,
commutativity and non-associativity of Yager's rule, Theorem 1 (|partial| <= 1,
column-stochastic Jacobian, l1 non-expansiveness), the 1/(1 - kappa) growth of
Dempster's derivative, the alpha-discount, the symmetric fold, and a short
end-to-end training smoke test.
"""

import torch
from torch.autograd.functional import jacobian

from mpenn.combine import (
    assert_valid_mass,
    dempster_combine,
    discount_mass,
    fusion_combine_chain,
    fusion_yager_symmetric,
    yager_combine,
)
from mpenn.model import ENN, compute_enn_loss


def _random_mass(k, batch=1):
    m = torch.rand(batch, k + 1)
    return m / m.sum(dim=-1, keepdim=True)


def _yager(m1, m2, k):
    return yager_combine(m1, m2, k)[0]


def l1(a, b):
    return (a - b).abs().sum(dim=-1)


def test_combined_masses_are_valid():
    torch.manual_seed(0)
    for k in (2, 5, 10):
        m1, m2 = _random_mass(k, 64), _random_mass(k, 64)
        assert_valid_mass(_yager(m1, m2, k))
        assert_valid_mass(dempster_combine(m1, m2, k)[0])
        assert_valid_mass(discount_mass(m1, torch.rand(64, 1), k))


def test_belief_plausibility_duality():
    torch.manual_seed(0)
    k = 5
    for _ in range(100):
        m = _random_mass(k)
        m_s, m_theta = m[0, :k], m[0, k]
        assert (m_s <= m_s + m_theta).all()
        a = torch.rand(k) > 0.5
        if a.any() and not a.all():
            pl_a = m_s[a].sum() + m_theta
            bel_not_a = m_s[~a].sum()
            assert abs(float(pl_a - (1.0 - bel_not_a))) < 1e-5


def test_yager_commutative():
    torch.manual_seed(0)
    for _ in range(200):
        m1, m2 = _random_mass(5), _random_mass(5)
        assert l1(_yager(m1, m2, 5), _yager(m2, m1, 5)).item() < 1e-5


def test_yager_not_associative():
    k = 3
    m1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    m2 = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    m3 = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
    left = _yager(_yager(m1, m2, k), m3, k)
    right = _yager(m1, _yager(m2, m3, k), k)
    assert l1(left, torch.tensor([[0.45, 0.05, 0.05, 0.45]])).item() < 1e-5
    assert l1(right, torch.tensor([[0.55, 0.0, 0.0, 0.45]])).item() < 1e-5


def test_theorem1_yager_jacobian_bounded_and_column_stochastic():
    torch.manual_seed(0)
    k = 10
    for _ in range(100):
        m1, m2 = _random_mass(k), _random_mass(k)
        J = jacobian(lambda x: _yager(x.unsqueeze(0), m2, k).squeeze(0), m1.squeeze(0))
        assert float(J.abs().max()) <= 1.0 + 1e-5
        assert float(J.min()) >= -1e-7
        assert torch.allclose(J.sum(dim=0), torch.ones(k + 1), atol=1e-5)


def test_theorem1_yager_l1_nonexpansive():
    torch.manual_seed(0)
    k = 4
    m2 = _random_mass(k)
    for _ in range(100):
        m1, m1p = _random_mass(k), _random_mass(k)
        assert l1(_yager(m1, m2, k), _yager(m1p, m2, k)).item() <= l1(m1, m1p).item() + 1e-5


def test_dempster_derivative_tracks_inverse_one_minus_kappa():
    k = 3
    for p in (0.5, 0.9, 0.99):
        m1 = torch.zeros(1, k + 1)
        m1[0, 0], m1[0, -1] = p, 1 - p
        m2 = torch.zeros(1, k + 1)
        m2[0, 1], m2[0, -1] = p, 1 - p
        kappa = p * p
        J = jacobian(lambda x: dempster_combine(x.unsqueeze(0), m2, k, clamp_one_minus_k=False)[0].squeeze(0),
                     m1.squeeze(0))
        assert abs(float(J.abs().max()) - 1 / (1 - kappa)) < 1e-3 / (1 - kappa)
        Jy = jacobian(lambda x: _yager(x.unsqueeze(0), m2, k).squeeze(0), m1.squeeze(0))
        assert abs(float(Jy.abs().max()) - 1.0) < 1e-6


def test_discount_is_alpha_contractive():
    torch.manual_seed(0)
    k = 4
    for _ in range(50):
        m, mp = _random_mass(k), _random_mass(k)
        alpha = torch.rand(1, 1)
        lhs = l1(discount_mass(m, alpha, k), discount_mass(mp, alpha, k)).item()
        assert lhs <= float(alpha) * l1(m, mp).item() + 1e-5


def test_symmetric_fold_is_order_invariant():
    torch.manual_seed(0)
    k = 3
    masses = [_random_mass(k) for _ in range(3)]
    ref = fusion_yager_symmetric(masses, k)[0]
    other = fusion_yager_symmetric([masses[1], masses[2], masses[0]], k)[0]
    assert l1(ref, other).item() < 1e-5
    # The fixed-order fold generally differs, but only through the last source.
    a = fusion_combine_chain([masses[0], masses[1], masses[2]], k)[0]
    b = fusion_combine_chain([masses[1], masses[0], masses[2]], k)[0]
    assert l1(a, b).item() < 1e-5


def test_enn_forward_and_training_smoke():
    torch.manual_seed(0)
    for rule in ('yager', 'dempster'):
        model = ENN(num_classes=10, num_sources=3, combination_rule=rule, pretrained=False)
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
        for _ in range(3):
            out = model(x)
            assert_valid_mass(out['mass'])
            assert torch.allclose(out['plausibility'] - out['belief'],
                                  out['uncertainty'].unsqueeze(-1).expand(-1, 10))
            loss = compute_enn_loss(out, y)
            assert torch.isfinite(loss['total'])
            opt.zero_grad()
            loss['total'].backward()
            opt.step()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'all {len(tests)} checks passed')
