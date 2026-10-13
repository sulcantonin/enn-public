#!/usr/bin/env bash
# Reproduce the paper's results, stage by stage. Run from the repository root.
#
#   bash scripts/reproduce.sh cpu          # no GPU: tests, Sect. 3.3 / 5.3 gradient studies,
#                                          # Sect. 5.2 CIFAR-10H numbers + Fig. 2 from results/
#   bash scripts/reproduce.sh table1       # Table 1 (5 seeds x 11 trainings; uses all visible GPUs)
#   bash scripts/reproduce.sh cifar10      # MP-ENN main seed 0, Yager/Dempster pair, W and lambda_unc sweeps
#   bash scripts/reproduce.sh cifar10h     # Sect. 5.2 CIFAR-10H + Fig. 2 from your own checkpoints (after cifar10)
#   bash scripts/reproduce.sh fusion       # fusion-order sensitivity (after cifar10)
#   bash scripts/reproduce.sh degradation  # corrupted-head stress test, 3 seeds
#   bash scripts/reproduce.sh chaosnli     # Sect. 5.2 ChaosNLI (RoBERTa-base on SNLI)
#   bash scripts/reproduce.sh all          # every GPU stage above, in order
#
# Outputs go to $RUNS (default: runs/). Every stage skips work whose outputs exist.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${RUNS:-runs}"
STAGE="${1:-cpu}"
PY="${PYTHON:-python}"

log() { echo -e "\n\033[0;34m[$(date +%H:%M:%S)] $*\033[0m"; }

stage_cpu() {
    log "unit tests (Theorem 1, DST axioms, smoke test)"
    $PY -m pytest -q tests
    log "Sect. 3.3: worst-case derivative vs conflict"
    $PY experiments/gradient_vs_conflict.py --output "$RUNS/cifar10/gradient_vs_conflict.json"
    log "Sect. 5.3: 200 random mass pairs, Yager vs Dempster gradient norm"
    $PY experiments/cifar10_mpenn.py --suite synthetic --output-dir "$RUNS/cifar10"
    log "Sect. 5.2: CIFAR-10H analysis and Fig. 2 from the paper's cached inference"
    $PY experiments/cifar10h_analysis.py --cache results/cifar10/interp_cache.npz \
        --output "$RUNS/cifar10/human_disagreement_report.json"
    $PY experiments/make_figure2.py --cache results/cifar10/interp_cache.npz \
        --output "$RUNS/figures/interp_interval.pdf"
}

stage_table1() {
    log "Table 1: CIFAR-10 OOD benchmark (5 seeds, 200 epochs, ensemble of 5)"
    $PY experiments/cifar10_benchmark.py --output-dir "$RUNS/benchmark" --quiet
    $PY experiments/make_table1.py --bench-dir "$RUNS/benchmark" --output "$RUNS/benchmark/table1.tex"
}

stage_cifar10() {
    log "MP-ENN-Yager, seed 0, 200 epochs (representative seed of Sect. 5.2/5.3)"
    $PY experiments/cifar10_mpenn.py --suite main --seed 0 --seeds 1 --output-dir "$RUNS/cifar10" --quiet
    log "Yager vs Dempster pair, 60 epochs"
    $PY experiments/cifar10_mpenn.py --suite ablation-combine --output-dir "$RUNS/cifar10" --quiet
    log "prior weight W and lambda_unc sweeps, 60 epochs"
    $PY experiments/cifar10_mpenn.py --suite ablation-W --output-dir "$RUNS/cifar10" --quiet
    $PY experiments/cifar10_mpenn.py --suite ablation-lambda --output-dir "$RUNS/cifar10" --quiet
}

stage_cifar10h() {
    log "Sect. 5.2: CIFAR-10H from the checkpoints in $RUNS/cifar10"
    $PY experiments/cifar10h_inference.py \
        --yager-checkpoint "$RUNS/cifar10/main_seed0/best_model.pth" \
        --dempster-checkpoint "$RUNS/cifar10/rule_dempster/best_model.pth" \
        --output "$RUNS/cifar10/interp_cache.npz"
    $PY experiments/cifar10h_analysis.py --cache "$RUNS/cifar10/interp_cache.npz" \
        --output "$RUNS/cifar10/human_disagreement_report.json"
    $PY experiments/make_figure2.py --cache "$RUNS/cifar10/interp_cache.npz" \
        --output "$RUNS/figures/interp_interval.pdf"
}

stage_fusion() {
    log "fusion-order sensitivity on MP-ENN-Yager seed 0"
    $PY experiments/fusion_order.py --checkpoint "$RUNS/cifar10/main_seed0/best_model.pth" \
        --ood CIFAR-100 SVHN --output "$RUNS/cifar10/fusion_order_report.json"
}

stage_degradation() {
    for seed in 0 1 2; do
        if [ -f "$RUNS/degraded/seed$seed/final_alpha.csv" ]; then continue; fi
        log "corrupted-head stress test, sigma 1.5, seed $seed"
        $PY experiments/source_degradation.py --sigma 1.5 --corrupt-idx 1 --epochs 60 \
            --seed "$seed" --output-dir "$RUNS/degraded/seed$seed"
    done
    cat "$RUNS"/degraded/seed*/final_alpha.csv
}

stage_chaosnli() {
    log "ChaosNLI: RoBERTa-base + MP-ENN on SNLI, then evaluation"
    $PY experiments/chaosnli_train.py --epochs 2 --batch-size 64 --output-dir "$RUNS/chaosnli"
    $PY experiments/chaosnli_eval.py --checkpoint "$RUNS/chaosnli/best_model.pth" --output-dir "$RUNS/chaosnli"
}

case "$STAGE" in
    cpu)          stage_cpu ;;
    table1)       stage_table1 ;;
    cifar10)      stage_cifar10 ;;
    cifar10h)     stage_cifar10h ;;
    fusion)       stage_fusion ;;
    degradation)  stage_degradation ;;
    chaosnli)     stage_chaosnli ;;
    all)          stage_cpu; stage_cifar10; stage_cifar10h; stage_fusion
                  stage_degradation; stage_chaosnli; stage_table1 ;;
    *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
log "done; outputs in $RUNS/"
