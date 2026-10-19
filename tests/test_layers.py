"""BeliefLayer works on bare features, and ENN is exactly BeliefLayer + ResNet-18."""

import torch

from mpenn import BeliefLayer, ENN, compute_enn_loss
from mpenn.combine import assert_valid_mass


def test_belief_layer_on_arbitrary_features():
    torch.manual_seed(0)
    layer = BeliefLayer(in_features=32, num_classes=5, num_sources=3)
    out = layer(torch.randn(16, 32))
    assert out['mass'].shape == (16, 6)
    assert out['plausibility'].shape == (16, 5)
    assert_valid_mass(out['mass'])
    loss = compute_enn_loss(out, torch.randint(0, 5, (16,)))['total']
    loss.backward()
    assert all(p.grad is not None for p in layer.parameters())


def test_enn_is_belief_layer_plus_backbone():
    torch.manual_seed(0)
    model = ENN(num_classes=10, num_sources=3, pretrained=False).eval()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    layer = BeliefLayer(model.feature_dim, 10, 3).eval()
    layer.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith('backbone.')})
    ref = layer(out['features'])
    assert torch.allclose(out['mass'], ref['mass'], atol=1e-6)
