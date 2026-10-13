# Neural Networks with Differentiable Belief-Function Layers

Code for the BELIEF 2026 paper *Neural Networks with Differentiable Belief-Function Layers* (Antonin Sulc).

The paper's model is the Mass-Primitive Evidential Neural Network (MP-ENN). Every mass from the evidence heads to the output is a valid Dempster–Shafer mass function on the focal set F = {{c_1}, …, {c_K}, Θ}. The heads are combined with **Yager's rule**, which sends conflict to the ignorance set Θ instead of renormalising by 1/(1−κ). The rule is bilinear, and every partial derivative of the combined mass is bounded by 1 (Theorem 1). The output mass m(Θ) is the network's "I do not know" signal.

```
ResNet-18 ──► |S| evidence heads ──► mass on F (Eq. 1) ──► learned α-discount ──► Yager fold (Eq. 5) ──► [Bel, Pl], m(Θ)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[nli]"        # drop [nli] if you do not need the ChaosNLI experiment
```

For a specific CUDA build, install `torch`/`torchvision` first (e.g. from `https://download.pytorch.org/whl/cu121`) and then run the line above. Python ≥ 3.9 is required. The original runs used Python 3.10, PyTorch 2.x and CUDA 12.1.

## Data and pretrained weights

```bash
python scripts/download_data.py      # CIFAR-10/100, SVHN, MNIST, Fashion-MNIST, DTD, CIFAR-10H, ChaosNLI, SNLI
python scripts/download_models.py    # ImageNet ResNet-18 and roberta-base, for offline compute nodes
```

Datasets go to `data/` by default; set `MPENN_DATA` to use another location. The official ChaosNLI archive link no longer works. When it fails, `download_data.py` takes `chaosNLI_snli.jsonl` from a [community mirror](https://huggingface.co/datasets/earino/chaosnli) and checks its SHA-256. The Gaussian and Uniform noise OOD sets are generated on the fly.

## Using the model

```python
import torch
from mpenn import ENN, compute_enn_loss

model = ENN(num_classes=10, num_sources=3, combination_rule='yager')   # W = K by default
out = model(torch.randn(8, 3, 32, 32))
out['mass']          # (8, 11)  fused mass: m({c_1}), ..., m({c_10}), m(Θ)
out['belief']        # (8, 10)  Bel({c_i}) = m({c_i})
out['plausibility']  # (8, 10)  Pl({c_i}) = m({c_i}) + m(Θ)
out['uncertainty']   # (8,)     vacuity m(Θ)
loss = compute_enn_loss(out, torch.randint(0, 10, (8,)))['total']     # Eq. 6
```

The combination rules are in `mpenn/combine.py` (`yager_combine`, `dempster_combine`, `discount_mass`, `fusion_combine_chain`). They work on any `(B, K+1)` mass tensor.

## Reproducing the paper

`scripts/reproduce.sh <stage>` runs each stage below. Outputs go to `runs/` (or `$RUNS`), and finished work is skipped.

| Paper | Command | Hardware |
|---|---|---|
| Theorem 1, DST axioms | `pytest tests` | CPU, seconds |
| Sect. 3.3: derivative vs conflict (1.33 → 4999.2) | `python experiments/gradient_vs_conflict.py` | CPU, seconds |
| Sect. 5.3: 200 random mass pairs, ‖∇‖ Yager vs Dempster | `python experiments/cifar10_mpenn.py --suite synthetic` | CPU, seconds |
| Sect. 5.2 CIFAR-10H numbers + Fig. 2 (from the paper's cached inference) | `bash scripts/reproduce.sh cpu` | CPU, ~1 min |
| Table 1 | `bash scripts/reproduce.sh table1` | GPU: 5 seeds × 11 trainings × 200 epochs |
| MP-ENN-Yager seed 0, Yager/Dempster pair, W and λ_unc ablations | `bash scripts/reproduce.sh cifar10` | GPU |
| Sect. 5.2 CIFAR-10H + Fig. 2 from your own checkpoints | `bash scripts/reproduce.sh cifar10h` | after `cifar10` |
| Fusion-order sensitivity | `bash scripts/reproduce.sh fusion` | after `cifar10` |
| Learned-discount stress test (α under a corrupted head) | `bash scripts/reproduce.sh degradation` | GPU, 3 × 60 epochs |
| Sect. 5.2 ChaosNLI | `bash scripts/reproduce.sh chaosnli` | GPU, RoBERTa-base, 2 epochs on SNLI |

`results/` holds the raw outputs behind the numbers in the paper. [`results/README.md`](results/README.md) maps each file to the claims it supports. `results/cifar10/interp_cache.npz` is the test-set inference of the two CIFAR-10H models, so the Sect. 5.2 CIFAR-10H numbers and Fig. 2 can be regenerated without a GPU.

Retrained models will not match the paper digit for digit. GPU training here is not deterministic (cuDNN autotuning, mixed precision). The paper's ChaosNLI model and synthetic-gradient study came from unseeded runs; both scripts now take `--seed`.

### Which runs feed which section

- **Table 1** comes from `experiments/cifar10_benchmark.py`, which trains MP-ENN-Yager, MP-ENN-Dempster, EDL, DUQ, SNGP, MC Dropout and five ensemble members per seed. Softmax + MSP/Energy/ODIN are post-hoc scores on MP-ENN-Yager's pre-softplus head-0 logits, so those rows share its accuracy and ECE.
- **CIFAR-10H and fusion order** (Sects. 5.2, 5.3) use the seed-0 model of `cifar10_mpenn.py --suite main` as MP-ENN-Yager. The CIFAR-10H Dempster model is `rule_dempster` from `--suite ablation-combine` (60 epochs).
- **W and λ_unc ablations** use one seed, 60 epochs each, and report CIFAR-100 vacuity AUROC.

## Repository layout

```
mpenn/                  the library
  combine.py            Yager / Dempster on F, α-discount, left fold, symmetric fold
  model.py              EvidenceToMass (Eq. 1), ENN, compute_enn_loss (Eq. 6)
  nli.py                MP-ENN head on a transformer encoder (ChaosNLI)
  training.py           training loop and checkpoint loading
  baselines.py          EDL, DUQ, SNGP, MC Dropout, Deep Ensemble, post-hoc Softmax scores
  data.py, metrics.py   CIFAR-10 and OOD loaders; AUROC, FPR95, ECE
experiments/            one script per experiment (see table above)
scripts/                data/weight download, reproduce.sh
tests/                  numerical checks of the theory
results/                stored outputs behind the paper's numbers
```

## Implementation details

These choices are fixed in the code and some are not spelled out in the paper:

- **Encoder.** torchvision ResNet-18 initialised from ImageNet-1k weights. The stem is replaced by a randomly initialised 3×3, stride-1 convolution and the max-pool is removed. The same encoder is used by every CIFAR-10 method.
- **Evidence heads.** `Linear(512, 256) – ReLU – Dropout(0.1) – Linear(256, K) – Softplus`; |S| = 3, W = K.
- **Reliability gate.** α_s = σ(g_φ([e_s; ē; e_s − ē])), with g_φ a `3K → 128 → 1` MLP. Head 0 is undiscounted (α_1 = 1). The corrupted-head stress test discounts all three heads.
- **Loss.** Eq. 6 with λ_unc = λ_aux = 0.1 and ε = 1e-8. The per-head term is evaluated on the masses *after* α-discounting.
- **Dempster.** In training, 1 − κ is clamped at 1e-8. The clamp is off in the two gradient studies.
- **Optimisation (CIFAR-10).** Adam (lr 1e-3, weight decay 1e-4), batch 128, cosine schedule, gradient-norm clip 1.0, mixed precision on CUDA (DUQ trains in fp32). Every method keeps the epoch with the best CIFAR-10 test accuracy.
- **Optimisation (ChaosNLI).** AdamW (lr 2e-5, weight decay 0.01), batch 64, max length 128, 2 epochs, best epoch by SNLI validation accuracy.
- **OOD sets.** First 10,000 test images of CIFAR-100, SVHN, MNIST and Fashion-MNIST; the DTD test split (1,880 images); 10,000 Gaussian N(0, 1) and Uniform U[0, 1) images. All are normalised with the CIFAR-10 statistics. Macro scores weight the seven sets equally.
- **ECE.** 15 bins. MP-ENN's confidence is max_i m({c_i}), not renormalised. The Deep Ensemble's ID accuracy and ECE are those of member 0.

## Citation

```bibtex
@inproceedings{sulc2026mpenn,
  title     = {Neural Networks with Differentiable Belief-Function Layers},
  author    = {Sulc, Antonin},
  booktitle = {Belief Functions: Theory and Applications (BELIEF 2026)},
  publisher = {Springer},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The datasets keep their own licenses. ChaosNLI is CC BY-NC 4.0.
