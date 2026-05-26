# Phase 16 Marco 4P-B - Cost-Aware Dense Graft Routing

Status: **completed on short CUDA protocol for seeds 42, 7, and 123**.

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

TDD coverage:

```text
tests/test_ntk_hybrid_routing.py
tests/test_candidate_efficiency.py
tests/test_ntk_hybrid_ablation.py
tests/test_marco_names.py
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

For these runs `stage_accept_min_gain=0.0`, so the effective acceptance policy is:

```text
candidate_composed_gain > 0
and
efficiency_score > 0
```

## Short CUDA command

The multiseed short protocol used:

```bash
cd /home/rato/dev/ai/SAINT-G
source .venv/bin/activate

for seed in 42 7 123; do
  python scripts/benchmark_drm_g_phase16_graftblock.py \
    --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4p_b_cost_aware_short_seed${seed} \
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
done
```

## Artifacts

```text
runs/phase16_marco4p_b_cost_aware_short_seed42/
runs/phase16_marco4p_b_cost_aware_short_seed7/
runs/phase16_marco4p_b_cost_aware_short_seed123/
```

Each run produced:

```text
summary.json
stage_metrics.json
candidate_metrics.json
candidate_training_metrics.jsonl
ntk_activation_probe_metrics.json
results.md
composed_graft_checkpoint.pt
```

The JSON/JSONL/MD artifacts are intended to be committed with `git add -f`; the large `composed_graft_checkpoint.pt` files should stay local unless explicitly requested.

## 4P-B multiseed results

| seed | base_loss | composed_loss | gain | accepted_grafts | route | stage 2 | recompose_abs_diff |
|---:|---:|---:|---:|---:|---|---|---:|
| 42 | 10.4161744118 | 10.4145958424 | 0.0015785694 | 4 | blocks.4 x4 | rejected: gain 0.0000000000, score -0.0000251967 | 0.0 |
| 7 | 10.3868415356 | 10.3862791061 | 0.0005624294 | 4 | blocks.4 x4 | rejected: gain 0.0000000000, score -0.0000250908 | 0.0 |
| 123 | 10.4170150757 | 10.4153037071 | 0.0017113686 | 4 | blocks.3 x4 | rejected: gain 0.0000000000, score -0.0000249720 | 0.0 |

Aggregate:

```text
positive_runs: 3/3
exact_recomposition: 3/3
mean_composed_loss: 10.4053928852
mean_accumulated_gain: 0.0012841225
mean_accepted_grafts: 4.0000
total_accepted_grafts: 12
stage2_rejected: 3/3
```

## Comparison against 4N-B

| seed | 4N-B composed_loss | 4P-B composed_loss | delta 4P-B - 4N-B | 4N-B gain | 4P-B gain | grafts 4N-B -> 4P-B |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10.4145286083 | 10.4145958424 | +0.0000672341 | 0.0016458035 | 0.0015785694 | 6 -> 4 |
| 7 | 10.3863139153 | 10.3862791061 | -0.0000348092 | 0.0005276203 | 0.0005624294 | 4 -> 4 |
| 123 | 10.4153611660 | 10.4153037071 | -0.0000574589 | 0.0016539097 | 0.0017113686 | 4 -> 4 |

Aggregate comparison:

```text
4N-B mean composed_loss: 10.4054012299
4P-B mean composed_loss: 10.4053928852
mean_loss_delta_4P_minus_4N: -0.0000083447

4N-B mean gain: 0.0012757778
4P-B mean gain: 0.0012841225
mean_gain_delta_4P_minus_4N: +0.0000083446

4N-B total accepted_grafts: 14
4P-B total accepted_grafts: 12
```

## Interpretation

4P-B passed the short multiseed test.

The important behavior is not that 4P-B discovers deeper routed growth; it does not. The important behavior is that it turns the fragile 4N-B marginal-graft behavior into an explicit cost-aware policy:

```text
keep the first useful stage;
reject follow-up stages that have zero composed gain or negative efficiency score.
```

Seed 42 is the clearest example. 4N-B accepted two extra grafts after the first group:

```text
blocks.4 x4 + blocks.3 x1 + blocks.2 x1 = 6 grafts
```

4P-B rejected those marginal additions and kept:

```text
blocks.4 x4 = 4 grafts
```

The seed-42 loss cost was small:

```text
4P-B seed42 is worse than 4N-B seed42 by +0.0000672341
```

But across all three seeds, 4P-B is marginally better on mean loss and uses two fewer grafts total.

## Verdict

```text
Marco 4P-B is a success as a conservative cost-aware routing gate.
```

Criteria:

```text
positive_runs: passed, 3/3
exact_recomposition: passed, 3/3
mean loss vs 4N-B: passed, slightly better
accepted graft count vs 4N-B: passed, 12 vs 14
stage-2 rejection behavior: passed, 3/3 rejected at zero gain / negative score
```

## Recommended next marco: 4P-C

Do **not** spend full `probe2k/max_train_seconds=1800` budget yet.

The next recommended marco is:

```text
Marco 4P-C - Cost-Aware Calibration Sweep
```

Goal:

```text
Run short CUDA sweeps over cost-aware lambda settings to verify whether 4P-B's
success is robust to the exact penalty weights, and whether a softer cost model
changes target selection or accepted-stage behavior before spending full-protocol
budget.
```

Recommended first sweep:

```text
baseline:      lambda_params=1e-6,  lambda_bytes=5e-7,  lambda_time=1e-7, lambda_ntk_risk=1.0
no_time:       lambda_params=1e-6,  lambda_bytes=5e-7,  lambda_time=0,    lambda_ntk_risk=1.0
half_cost:     lambda_params=5e-7,  lambda_bytes=2.5e-7,lambda_time=5e-8, lambda_ntk_risk=1.0
low_ntk_risk:  lambda_params=1e-6,  lambda_bytes=5e-7,  lambda_time=1e-7, lambda_ntk_risk=0.5
```

Success criterion for 4P-C:

```text
The preferred calibration should preserve exact recomposition 3/3, avoid
accepting zero-gain stages, and keep mean composed_loss competitive with 4N-B
while maintaining <= 12 accepted grafts total across seeds 42/7/123.
```
