# Phase 16 Marco 4I - Residual/Orthogonal Routing

Status: **implemented, validated**.

## Goal

Marco 4H accepted five grafts, then stalled:

```text
accepted_grafts: 5
stage 3: rejected
```

Marco 4I tests whether later stages are being routed into redundant directions.
It keeps strict composed-loss acceptance, but changes candidate ranking to prefer
less redundant targets when gains are otherwise competitive.

## Method

New options:

```text
--candidate-score-mode composed_gain
--candidate-score-mode composed_gain_orthogonal
--orthogonal-penalty 0.00001
```

Default behavior remains:

```text
candidate_score = candidate_composed_gain
```

Orthogonal mode uses:

```text
candidate_score = candidate_composed_gain - redundancy_penalty
redundancy_penalty = orthogonal_penalty * accepted_grafts_on_same_target
```

Important rule:

```text
candidate_composed_gain > stage_accept_min_gain
```

is still required for approval. The orthogonal penalty only changes ranking; it
does not approve negative or zero-gain candidates.

## Dry-Runs

Permissive dry-run:

```text
runs/phase16_marco4i_orthogonal_dryrun
candidate_score_mode: composed_gain_orthogonal
orthogonal_penalty: 1e-5
recompose_abs_diff: 0.0
```

Strict dry-run:

```text
runs/phase16_marco4i_orthogonal_strict_dryrun
stage_accept_min_gain: 0.0
accepted_grafts: 0
stage_gain: 0.0
decision: rejected
recompose_abs_diff: 0.0
```

The strict run confirms that zero-gain candidates are not approved.

## Light CUDA Result

Run:

```text
runs/phase16_marco4i_light_orthogonal
```

Summary:

```text
base_loss: 10.416174411773682
composed_loss: 10.414714097976685
accumulated_gain: 0.0014603137969970703
accepted_groups: 2
accepted_grafts: 5
accepted_graft_ids: [0, 1, 2, 3, 4]
target_by_graft:
  0..3 -> blocks.2
  4    -> blocks.3
recompose_abs_diff: 0.0
```

Comparison against Marco 4H:

```text
4H composed_loss: 10.414670705795288
4I composed_loss: 10.414714097976685
delta_vs_4H: +0.000043392181397
```

Marco 4I did not improve the best current validation loss. Its useful result is
control behavior: at stage 3, the router rejected a redundant `blocks.3`
candidate with zero composed gain and negative orthogonal score.

```text
stage: 3
selected_target: blocks.3
stage_gain: 0.0
stage_score: -1e-05
redundancy_penalty: 1e-05
decision: rejected
```

Technical verdict:

```text
Marco 4I is valid as a redundancy-control mechanism.
It is not a quality improvement over Marco 4H.
```

## Recommended Command

```powershell
cd E:\dev\ai\SAINT-G

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\SAINT-G\runs\phase16_marco4i_orthogonal_24graft `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
  --stage-size 4 `
  --post-first-stage-size 1 `
  --max-stages 10 `
  --stage-accept-min-gain 0.0 `
  --steps 100000000 `
  --max-train-seconds 2400 `
  --eval-every-steps 5000 `
  --early-stopping-patience 3 `
  --early-stopping-min-delta 0.00001 `
  --batch-size 2 `
  --seq-len 128 `
  --validation-batches 4 `
  --train-batches 4096 `
  --learning-rate 0.0000003 `
  --lr-decay 0.02 `
  --training-mode validation_routed_staged `
  --candidate-targets blocks.0 blocks.1 blocks.2 blocks.3 blocks.4 blocks.5 `
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 `
  --candidate-init-scales 0.001 0.005 0.01 `
  --candidate-activations silu `
  --candidate-score-mode composed_gain_orthogonal `
  --orthogonal-penalty 0.00001
```

## Criteria

Marco 4I passes if:

```text
composed_loss <= 10.414714
accepted_grafts >= 5
recompose_abs_diff = 0.0
```

The next step should reduce candidate-search cost. Full grid routing is too
expensive because `max_train_seconds` is currently applied per candidate, not
per run.

## Next Step - Marco 4J

Marco 4J should implement candidate pruning / two-pass routing:

1. Probe many candidates cheaply.
2. Select top-k by `candidate_composed_gain` or orthogonal score.
3. Train only the selected candidates deeply.
4. Compare against Marco 4H.

This attacks the actual bottleneck observed in Marco 4I: exhaustive candidate
grids can run for many hours or days.
