#!/usr/bin/env python3
"""Vacuity vs. ChaosNLI annotator disagreement (Sect. 5.2, 1,514 SNLI pairs, 100 labels each).

Human disagreement of a pair is the Shannon entropy (nats) of its normalised
ChaosNLI ``label_counter``. Reported Spearman correlations with it:

  * vacuity          m(Theta) of the fused output mass;
  * softmax_entropy  entropy of the fused belief renormalised over the 3 classes,
                     Bel / sum(Bel);
  * energy           -log(sum_i Bel({c_i})). With Yager fusion the output mass
                     sums to 1, so this equals -log(1 - m(Theta)), a monotone
                     transform of the vacuity (identical Spearman rho).

Also: Pearson rho for vacuity, mean Jensen-Shannon distance between the human
distribution and the renormalised belief, and mean vacuity on the most (>= Q75)
and least (<= Q25) contested quartiles.

    python experiments/chaosnli_eval.py --checkpoint runs/chaosnli/best_model.pth \
        --chaosnli-path data/ChaosNLI/data/chaosNLI_snli.jsonl --output-dir runs/chaosnli
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    raise SystemExit('ChaosNLI experiments need: pip install -e ".[nli]"')

from mpenn.data import data_root
from mpenn.nli import ENN_NLI, NUM_CLASSES

LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2, 'e': 0, 'n': 1, 'c': 2}


def premise_hypothesis(ex: dict):
    """ChaosNLI nests premise/hypothesis under ``example``."""
    inner = ex.get('example') if isinstance(ex.get('example'), dict) else ex
    return (inner.get('premise', inner.get('sentence1', '')),
            inner.get('hypothesis', inner.get('sentence2', '')))


def human_distribution(label_counter: dict) -> np.ndarray:
    dist = np.zeros(NUM_CLASSES)
    for label, count in label_counter.items():
        idx = LABEL_MAP.get(label.lower())
        if idx is not None:
            dist[idx] = count
    return dist / dist.sum() if dist.sum() > 0 else np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)


def entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def load_model(checkpoint: str, config_path: str, device: str):
    with open(config_path) as f:
        config = json.load(f)
    name = config.get('backbone', 'roberta-base')
    tokenizer = AutoTokenizer.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name)
    model = ENN_NLI(backbone, hidden_dim=config.get('hidden_dim', backbone.config.hidden_size),
                    num_sources=config.get('num_sources', 3), prior_weight=3.0,
                    combination_rule=config.get('combination_rule', 'yager'), use_discounting=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Tolerate only the position_ids buffer, whose persistence differs across transformers versions.
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing + unexpected if not k.endswith('position_ids')]
    if bad:
        raise RuntimeError(f'checkpoint does not match the model: {bad[:10]}')
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def infer(model, tokenizer, examples, device, batch_size=128, max_length=128):
    belief, vacuity = [], []
    for start in range(0, len(examples), batch_size):
        pairs = [premise_hypothesis(ex) for ex in examples[start:start + batch_size]]
        enc = tokenizer([p for p, _ in pairs], [h for _, h in pairs], padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
        out = model(enc['input_ids'], enc['attention_mask'], enc.get('token_type_ids'))
        belief.append(out['belief'].cpu().numpy())
        vacuity.append(out['uncertainty'].cpu().numpy())
    return np.concatenate(belief), np.concatenate(vacuity)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default=None, help='train_config.json (default: next to the checkpoint)')
    p.add_argument('--chaosnli-path', default=os.path.join(data_root(), 'ChaosNLI', 'data', 'chaosNLI_snli.jsonl'))
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='runs/chaosnli')
    args = p.parse_args()
    config = args.config or os.path.join(os.path.dirname(args.checkpoint), 'train_config.json')
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.checkpoint, config, args.device)
    with open(args.chaosnli_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f'{len(examples)} ChaosNLI pairs from {args.chaosnli_path}')

    belief, vac = infer(model, tokenizer, examples, args.device, batch_size=args.batch_size)
    bel_sum = belief.sum(axis=1)
    model_probs = belief / np.maximum(bel_sum, 1e-10)[:, None]
    softmax_ent = -(model_probs * np.log(np.clip(model_probs, 1e-10, 1.0))).sum(axis=1)
    energy = -np.log(np.clip(bel_sum, 1e-10, None))

    human = [human_distribution(ex.get('label_counter', ex.get('label_dist', {}))) for ex in examples]
    H = np.array([entropy(h) for h in human])
    jsd = np.array([jensenshannon(h, m) for h, m in zip(human, model_probs)])

    sp_vac = stats.spearmanr(H, vac)
    pe_vac = stats.pearsonr(H, vac)
    sp_sme = stats.spearmanr(H, softmax_ent)
    sp_eng = stats.spearmanr(H, energy)
    high, low = H >= np.percentile(H, 75), H <= np.percentile(H, 25)

    report = {
        'n_examples': len(examples),
        'spearman_vacuity': {'r': float(sp_vac[0]), 'p': float(sp_vac[1])},
        'pearson_vacuity': {'r': float(pe_vac[0]), 'p': float(pe_vac[1])},
        'spearman_softmax_entropy': {'r': float(sp_sme[0]), 'p': float(sp_sme[1])},
        'spearman_energy': {'r': float(sp_eng[0]), 'p': float(sp_eng[1])},
        'mean_jsd': float(jsd.mean()),
        'mean_vacuity_all': float(vac.mean()),
        'mean_vacuity_high_ambiguity': float(vac[high].mean()),
        'mean_vacuity_low_ambiguity': float(vac[low].mean()),
    }
    print(json.dumps(report, indent=2))
    out_path = os.path.join(args.output_dir, 'chaosnli_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
