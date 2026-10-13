# Stored results

These are the raw outputs behind the numbers in the paper. They were produced by earlier versions of the scripts in `experiments/`, so some JSON keys differ slightly from what the current scripts write.

## `cifar10/`

| File | Paper | Contents |
|---|---|---|
| `gradient_vs_conflict.json` | Sect. 3.3 | Largest \|∂m_comb(A)/∂m_1(B)\| for the two-singleton family (K = 3, float32). Dempster (no clamp) gives 1.33 at κ = 0.25 and 4999.2 at κ = 0.9998, tracking 1/(1−κ). Yager gives 1.00 at every κ. |
| `synthetic_combine_gradients.json` | Sect. 5.3 | 200 random mass pairs on F (K = 10). ‖∂m_comb({c_1})/∂m_1‖ is 0.21 for Yager and 1.52 for Dempster, with mean κ = 0.75. This came from an unseeded run; with `--seed 0` the current script gives 0.21 vs 1.36, and seeds vary roughly 6.6–8× in the ratio. Here 1 − κ ≥ 0.08, so clamped and unclamped Dempster agree. |
| `interp_cache.npz` | Sect. 5.2, Fig. 2 | Test-set inference (CIFAR-10H order). `by, py, vy` are belief, plausibility and m(Θ) of MP-ENN-Yager: `cifar10_mpenn.py --suite main`, seed 0, 94.46% test accuracy. `bd, pd, vd` are the same for MP-ENN-Dempster: `--suite ablation-combine`, seed 0, 60 epochs, 94.24%. `experiments/cifar10h_inference.py` reproduces it bit for bit from those checkpoints. |
| `human_disagreement_report.json` | Sect. 5.2 | CIFAR-10H analysis of the cache. Spearman ρ is 0.278 (Yager) vs 0.263 (Dempster), with bootstrap CIs. Binned m(Θ) runs 0.008 → 0.043 (Yager) and 0.002 → 0.005 (Dempster). On the top decile it is 0.044 vs 0.005 (9×), with separation AUROC 0.77. The current script sorts label probabilities before computing the entropy (see `annotator_entropy`), which changes ρ only in the 4th decimal (0.2777 vs 0.2776). |
| `fusion_order_report.json` | Sect. 5.3, ablations | All 3! fold orders of the seed-0 Yager model on CIFAR-10 test and CIFAR-100. There are 3 distinct outcomes. The mean of the per-image max pairwise ℓ1 is 0.002 (CIFAR-10) and 0.012 (CIFAR-100), with no class flips on CIFAR-10. Mean summed conflict is 0.022 vs 0.144, and the vacuity AUROC range across orders is 0.0004. |
| `ablation_prior_W.json` | Sect. 5.3, ablations | W ∈ {1, 5, 10, 20, 100}, seed 0, 60 epochs. CIFAR-100 vacuity AUROC (`AUROC`) stays between 0.855 and 0.866. |
| `ablation_lambda_unc.json` | Sect. 5.3, ablations | λ_unc ∈ {0, 0.01, 0.1, 1}, seed 0, 60 epochs. CIFAR-100 vacuity AUROC falls from 0.865 to 0.850. |

## `degraded/seed{0,1,2}/`

These are from the corrupted-head stress test (`experiments/source_degradation.py --sigma 1.5 --corrupt-idx 1 --epochs 60`). `alpha_per_epoch.csv` gives each epoch's mean α_s over the training batches. `final_alpha.csv` gives the last epoch.

Per head (head 1, head 2 = corrupted, head 3), every seed rounds to α = (0.99998, 0.99995, 0.99997). The corrupted head has the lowest α in all three seeds.

## `chaosnli/`

| File | Contents |
|---|---|
| `train_config.json`, `train_history.json` | RoBERTa-base + MP-ENN (K = 3, W = 3, Yager), 2 epochs on SNLI, batch 64. Best SNLI validation accuracy is 91.7%. The run was unseeded. |
| `chaosnli_report.json` | 1,514 ChaosNLI-SNLI pairs. Spearman ρ(annotator entropy, m(Θ)) = 0.346. Mean m(Θ) is 0.076 on the most contested quartile and 0.036 on the least. `spearman_softmax_entropy` (0.390) and `spearman_energy` (0.346) are defined in `experiments/chaosnli_eval.py`. |

## Not included

The per-seed outputs of the Table 1 benchmark (`cifar10_benchmark.py`) are not stored here. Run `bash scripts/reproduce.sh table1` to regenerate them. Checkpoints (`*.pth`) are not part of the repository.
