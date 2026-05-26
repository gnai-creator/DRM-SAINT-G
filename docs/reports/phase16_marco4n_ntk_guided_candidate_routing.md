# Phase 16 Marco 4N - NTK Residual/Saturation Routing Plan

Status: **4N-A completed; 4N-B completed for seeds 42, 7, and 123; 4N-C offline score ablation completed**.

## Decision From Marco 4M

Marco 4M tested the NTK-Mirror-inspired raw activation-gate score:

```text
score(block) = sum(abs(grad_h * h))
```

The completed seeds show that raw NTK is stable but not sufficient as a router:

```text
seed 42:
  accepted_grafts: 5
  stage 1 selected blocks.4, raw NTK rank 1, approved
  stage 2 selected blocks.2, raw NTK rank 3, approved fifth graft
  stage 3 selected blocks.3, raw NTK rank 2, rejected

seed 7:
  accepted_grafts: 4
  stage 1 selected blocks.4, raw NTK rank 1, approved
  stage 2 selected blocks.2, raw NTK rank 3, rejected with gain ~0

seed 123:
  accepted_grafts: 4
  stage 1 selected blocks.3, raw NTK rank 2, approved with seed-42-scale gain
  stage 2 selected blocks.4, raw NTK rank 1, rejected with gain 0
```

Therefore the raw score appears to measure global activation sensitivity rather
than marginal utility after previous grafts have already been accepted. Seed 123
is the strongest counterexample: raw NTK top-1 preferred `blocks.4`, while the
useful first-stage candidate was `blocks.3` at raw NTK rank 2.

Direct promotion of raw NTK is rejected for now:

```text
raw ntk_prefilter: rejected as next step
raw ntk_score_blend: rejected as next step
```

The useful question for 4N becomes:

```text
which target still has unexplored marginal utility after previous grafts?
```

## 4N-A - NTK Residual/Saturation Analysis

Status: **implemented / completed for seeds 42, 7, and 123**.

4N-A does not change training or routing. It joins Marco 4M artifacts and writes
an offline feature table for candidate routing analysis.

Implemented files:

```text
saint/adapters/drm_grafting_ntk_analysis.py
scripts/analyze_phase16_ntk_residual_saturation.py
tests/test_ntk_residual_saturation_analysis.py
```

Inputs per run directory:

```text
summary.json
stage_metrics.json
candidate_metrics.json
ntk_activation_probe_metrics.json
```

Outputs:

```text
ntk_candidate_joined_metrics.json
ntk_stage_feature_table.csv
ntk_run_summaries.json
ntk_routing_analysis.md
```

Derived features include:

```text
raw_ntk_score:
  ntk_activation_score from Marco 4M

ntk_rank:
  raw NTK rank within stage

ntk_delta_from_previous_stage:
  ntk_score(stage, target) - ntk_score(stage - 1, target)

ntk_delta_pct_from_previous_stage:
  ntk_delta_from_previous_stage / ntk_score(stage - 1, target)

accepted_grafts_on_target_before_stage:
  number of already accepted grafts assigned to that target before the stage

saturation_adjusted_ntk:
  ntk_score / (1 + accepted_grafts_on_target_before_stage)

best_candidate_composed_gain:
  best observed candidate_composed_gain for the same stage and target
```

The completed pass over seeds 42, 7, and 123 produces the same conservative
recommendation for every seed:

```text
reject_raw_ntk_prefilter
test_saturation_adjusted_ntk
test_residual_delta_ntk
include_target_saturation_features
```

4N-A artifacts:

```text
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123/ntk_candidate_joined_metrics.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123/ntk_stage_feature_table.csv
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123/ntk_run_summaries.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123/ntk_routing_analysis.md
```

## 4N-A Command - Seed 42 + Seed 7

Run this from the canonical SAINT-G repo:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_residual_saturation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7
```

Expected terminal summary:

```text
wrote 15 joined rows to /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7
seed=42 composed_loss=10.414523839950562 accepted_grafts=5 recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
seed=7 composed_loss=10.386313915252686 accepted_grafts=4 recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
```

## 4N-A Command - Completed Seeds 42 + 7 + 123

Seed 123 has completed. Run 4N-A with three `--run-dir` arguments. The script
fails fast if any run directory is missing `summary.json`, `stage_metrics.json`,
`candidate_metrics.json`, or `ntk_activation_probe_metrics.json`.

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_residual_saturation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123
```

## Current 4N-A Seed 42 + Seed 7 + Seed 123 Observation

The generated `ntk_routing_analysis.md` shows:

```text
seed 42 selected top-1 rate: 0.333
seed 7 selected top-1 rate: 0.500
seed 123 selected top-1 rate: 0.500
```

Critical selected-target rows:

```text
seed 42 stage 1:
  selected target: blocks.4
  decision: approved
  raw NTK rank: 1
  best_candidate_composed_gain: 0.001549959

seed 42 stage 2:
  selected target: blocks.2
  decision: approved
  raw NTK rank: 3
  raw NTK score: 2.03331
  saturation_adjusted_ntk: 2.03331
  best_candidate_composed_gain: 0.000100613

seed 7 stage 2:
  selected target: blocks.2
  decision: rejected
  raw NTK rank: 3
  raw NTK score: 2.02389
  saturation_adjusted_ntk: 2.02389
  best_candidate_composed_gain: 0.000000238419

seed 123 stage 1:
  selected target: blocks.3
  decision: approved
  raw NTK rank: 2
  raw NTK score: 3.08944
  saturation_adjusted_ntk: 3.08944
  best_candidate_composed_gain: 0.00165391

seed 123 stage 2:
  selected target: blocks.4
  decision: rejected
  raw NTK rank: 1
  raw NTK score: 4.19210
  saturation_adjusted_ntk: 4.19210
  best_candidate_composed_gain: 0.0
```

This confirms two separate failures of raw NTK as a direct router:

```text
1. It does not separate the useful seed-42 stage-2 blocks.2 fifth graft from the
   rejected seed-7 stage-2 blocks.2 candidate.
2. It misses the useful seed-123 stage-1 blocks.3 candidate because raw top-1
   still prefers blocks.4.
```

The more promising signal is composition state, especially target saturation:

```text
blocks.4 raw NTK can remain highest after stage 1,
but extra grafts on a saturated target may have near-zero marginal gain.
```

After saturation adjustment, saturated targets drop strongly:

```text
seed 42 stage 2 blocks.4:
  raw_ntk: 4.10929
  accepted_grafts_on_target_before_stage: 4
  saturation_adjusted_ntk: 0.821858

seed 7 stage 2 blocks.4:
  raw_ntk: 4.13932
  accepted_grafts_on_target_before_stage: 4
  saturation_adjusted_ntk: 0.827863

seed 123 stage 2 blocks.3:
  raw_ntk: 3.13342
  accepted_grafts_on_target_before_stage: 4
  saturation_adjusted_ntk: 0.626684
```

This supports 4N-B as a conservative hybrid: preserve
`composed_gain_orthogonal` as the primary decision source, and use NTK-derived
features only for candidate ordering, warnings, anti-saturation, or tie-breaks.

## 4N-B - Conservative NTK-Hybrid Routing

Status: **implemented as an explicit experiment mode**.

4N-B keeps the Marco 4M/4N-A lesson: raw NTK is not a router by itself. The new
mode is intentionally conservative: candidate composed gain remains the primary
signal, while NTK-derived terms only provide small bonuses/penalties and broaden
the candidate set so useful raw-rank-2/rank-3 targets are not dropped.

Implemented files:

```text
saint/adapters/drm_grafting_graftblock_routed_utils.py
saint/adapters/drm_grafting_graftblock_routed.py
scripts/benchmark_drm_g_phase16_graftblock.py
tests/test_ntk_hybrid_routing.py
```

New score mode:

```text
--candidate-score-mode composed_gain_ntk_hybrid_conservative
```

New knobs:

```text
--ntk-hybrid-saturation-weight       small bonus for ntk / (1 + accepted_on_target)
--ntk-hybrid-residual-delta-weight   small bonus for abs(ntk_t - ntk_t-1)
--ntk-hybrid-anti-saturation-penalty penalty per already accepted graft on target
--ntk-hybrid-keep-ranks              keep at least one candidate from raw NTK ranks <= N
```

Implemented routing score:

```text
saturation_adjusted_ntk = ntk_activation_score / (1 + accepted_grafts_on_target)
residual_delta_ntk = abs(ntk_activation_score_t - ntk_activation_score_t_minus_1)
anti_saturation_penalty = anti_saturation_penalty_weight * accepted_grafts_on_target

candidate_score =
  candidate_composed_gain
  - orthogonal_penalty
  - anti_saturation_penalty
  + saturation_weight * saturation_adjusted_ntk
  + residual_delta_weight * residual_delta_ntk
```

Candidate-set safety rule:

```text
stage_candidates = top candidate_top_k by hybrid candidate_score
                 + best representative for each raw NTK rank <= ntk_hybrid_keep_ranks
```

This preserves the known useful non-top-1 cases from 4N-A:

```text
seed 42 stage 2 blocks.2: raw NTK rank 3, approved fifth graft
seed 123 stage 1 blocks.3: raw NTK rank 2, approved main gain
```

It also downweights saturated targets such as post-stage-1 `blocks.4` by using
both `ntk / (1 + accepted_on_target)` and a direct anti-saturation penalty.

### 4N-B Training Command - Seeds 42, 7, and 123

Run the three seeds sequentially from the canonical SAINT-G repo with one shell
block:

```bash
cd /home/rato/dev/ai/SAINT-G

for seed in 42 7 123; do
  python \
    scripts/benchmark_drm_g_phase16_graftblock.py \
    --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed${seed} \
    --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
    --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
    --device cuda \
    --seeds ${seed} \
    --graft-count 24 \
    --hidden-size 25889 \
    --stage-size 4 \
    --post-first-stage-size 1 \
    --max-stages 8 \
    --stage-accept-min-gain 0.0 \
    --steps 100000000 \
    --max-train-seconds 1800 \
    --eval-every-steps 5000 \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 0.00001 \
    --batch-size 2 \
    --seq-len 128 \
    --validation-batches 4 \
    --train-batches 4096 \
    --learning-rate 0.0000003 \
    --lr-decay 0.02 \
    --training-mode validation_routed_staged \
    --candidate-targets blocks.2 blocks.3 blocks.4 \
    --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
    --candidate-init-scales 0.001 0.005 0.01 \
    --candidate-activations silu \
    --candidate-score-mode composed_gain_ntk_hybrid_conservative \
    --orthogonal-penalty 0.00001 \
    --candidate-probe-steps 2000 \
    --candidate-probe-max-train-seconds 300 \
    --candidate-top-k 8 \
    --ntk-activation-probe-batches 4 \
    --ntk-activation-probe-split train \
    --ntk-hybrid-saturation-weight 0.00002 \
    --ntk-hybrid-residual-delta-weight 0.00001 \
    --ntk-hybrid-anti-saturation-penalty 0.00005 \
    --ntk-hybrid-keep-ranks 3
done
```


## 4N-B Safety Requirements

Do not implement a raw NTK top-1 prefilter as the next experiment. The evidence
from seeds 42, 7, and 123 says that would over-favor `blocks.4`, may discard the
useful seed-42 `blocks.2` stage-2 candidate, and may discard the useful seed-123
`blocks.3` stage-1 candidate.

Before 4N-B routes automatically, the rule should pass these checks offline:

```text
- it keeps seed 42 stage-2 blocks.2 in the candidate set;
- it keeps seed 123 stage-1 blocks.3 in the candidate set despite raw NTK rank 2;
- it does not over-prioritize blocks.4 after blocks.4 already has 4 grafts;
- it does not treat raw NTK top-1 as sufficient when the candidate gain is zero;
- it explains why seed 7 blocks.2 had near-zero candidate gain.
```

## 4N-B / 4N-C Completed Result

The completed 4N-B run and 4N-C offline score ablation are documented in:

```text
docs/reports/phase16_marco4n_b_c_results.md
```

Short verdict:

```text
4N-B passed technically and found a diversified 6-graft seed-42 route, but it did
not solve multi-seed robustness. Seeds 7 and 123 still stop at 4 grafts. 4N-C
shows that the current NTK-hybrid bonus can assign positive score to zero-gain
candidates, so future routing should gate NTK bonus behind positive composed gain
or an epsilon.
```

4N-C command:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_hybrid_score_ablation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123
```

## Relationship to NTK-Mirror

This is not a direct port of `ntkmirror`. SAINT-G still uses graft blocks. The
borrowed idea is the activation-gate sensitivity score:

```text
dL/ds = grad_h * h
```

Marco 4N-A shows that this raw signal is useful as a diagnostic column, but it
must be combined with composition state before it can become a routing rule.
