# Phase 16 Marco 4P-B - Cost-Aware Dense Graft Routing

Status: **implemented and executed on CUDA for seed 42 short run**.

## Objective

Marco 4P-B moves the 4P-A offline efficiency score into the real routed/staged benchmark loop.

The test question is:

```text
Can dense GraftBlock routing preserve useful 4N-B-style gain while requiring both positive composed gain and positive cost-aware efficiency score before accepting a stage?
```

## Implementation

New benchmark score mode:

```text
--candidate-score-mode composed_gain_cost_aware
```

Implemented in:

```text
saint/adapters/drm_grafting_graftblock_routed_utils.py
saint/adapters/drm_grafting_graftblock_routed.py
scripts/benchmark_drm_g_phase16_graftblock.py
```

TDD coverage added in:

```text
tests/test_ntk_hybrid_routing.py
```

## Score and gate

```text
efficiency_score = candidate_composed_gain
  - redundancy_penalty
  - lambda_ntk_risk * ntk_hybrid_penalty
  - lambda_params * log1p(params_per_graft)
  - lambda_bytes  * log1p(checkpoint_bytes_delta)
  - lambda_time   * probe_seconds
```

Acceptance gate:

```text
candidate_composed_gain > stage_accept_min_gain
and
cost_aware_accept == true
```

For this run `stage_accept_min_gain=0.0`, so the effective gate is:

```text
candidate_composed_gain > 0
and
efficiency_score > 0
```

## Command

Executed from `/home/rato/dev/ai/SAINT-G`:

```bash
.venv/bin/python scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4p_b_cost_aware_short_seed42 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 42 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 180 \
  --eval-every-steps 200 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 1024 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_cost_aware \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 200 \
  --candidate-probe-max-train-seconds 90 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train \
  --ntk-hybrid-saturation-weight 0.00002 \
  --ntk-hybrid-residual-delta-weight 0.00001 \
  --ntk-hybrid-anti-saturation-penalty 0.00005 \
  --ntk-hybrid-keep-ranks 3 \
  --cost-aware-lambda-params 0.000001 \
  --cost-aware-lambda-bytes 0.0000005 \
  --cost-aware-lambda-time 0.0000001 \
  --cost-aware-lambda-ntk-risk 1.0
```

## Artifacts

```text
runs/phase16_marco4p_b_cost_aware_short_seed42/summary.json
runs/phase16_marco4p_b_cost_aware_short_seed42/stage_metrics.json
runs/phase16_marco4p_b_cost_aware_short_seed42/candidate_metrics.json
runs/phase16_marco4p_b_cost_aware_short_seed42/candidate_training_metrics.jsonl
runs/phase16_marco4p_b_cost_aware_short_seed42/ntk_activation_probe_metrics.json
runs/phase16_marco4p_b_cost_aware_short_seed42/results.md
runs/phase16_marco4p_b_cost_aware_short_seed42/composed_graft_checkpoint.pt
```

## Result

| metric | value |
|---|---:|
| base_loss | 10.4161744118 |
| composed_loss | 10.4145958424 |
| accumulated_gain | 0.0015785694 |
| accepted_groups | 1 |
| accepted_grafts | 4 |
| recompose_abs_diff | 0.0 |
| trainable_parameters_per_graft | 4,970,689 |
| composed_checkpoint_bytes | 477,210,951 |

Stage decisions:

| stage | selected_target | decision | gain | efficiency_score | target_by_graft |
|---:|---|---|---:|---:|---|
| 1 | blocks.4 | approved | 0.0015785694 | 0.0015427074 | 0-3 -> blocks.4 |
| 2 | blocks.3 | rejected | 0.0 | -0.0000251967 | unchanged |

Candidate rows:

```text
candidate_metrics rows: 71
cost_aware_accept rows: 13
positive candidate_score rows: 13
```

## Comparison against 4N-B seed 42

| run | composed_loss | accumulated_gain | accepted_grafts | target route | recompose_abs_diff |
|---|---:|---:|---:|---|---:|
| 4N-B seed42 | 10.4145286083 | 0.0016458035 | 6 | blocks.4 x4, blocks.3 x1, blocks.2 x1 | 0.0 |
| 4P-B short seed42 | 10.4145958424 | 0.0015785694 | 4 | blocks.4 x4 | 0.0 |

4P-B is slightly worse on raw loss than 4N-B seed42 by about `0.000067234`, but uses fewer accepted grafts:

```text
4N-B accepted_grafts: 6
4P-B accepted_grafts: 4
```

That is the desired behavior for this first cost-aware gate: preserve most of the useful seed42 improvement while rejecting zero-efficiency follow-up stages.

## Verdict

Marco 4P-B is **implemented, recomposable, and positive on seed 42 short CUDA run**.

Interpretation:

```text
The cost-aware gate keeps the high-value stage-1 blocks.4 graft group and rejects the next zero-gain/negative-efficiency stage.
```

Recommendation:

```text
Replicate 4P-B short on seeds 7 and 123 before attempting the full 2k-probe/1800s protocol.
```

Rationale:

```text
The full probe2k/max_train_seconds=1800 configuration is much slower because it can spend a long time in probe/deep candidates before writing final artifacts. The short 4P-B run is enough to validate the code path and the acceptance gate before spending that budget across seeds.
```
