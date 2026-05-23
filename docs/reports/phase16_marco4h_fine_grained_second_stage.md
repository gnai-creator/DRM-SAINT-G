# Phase 16 Marco 4H - Fine-Grained Second Stage

Status: **implemented, dry-run validated**.

## Goal

Marco 4G improved G1 but still rejected G2:

```text
accepted_grafts: 4
stage 2: rejected
```

Marco 4H tests whether the second stage fails because the remaining useful
updates are smaller than a 4-graft group.

## Method

The routed benchmark now supports:

```text
--post-first-stage-size
```

Behavior:

```text
stage 1: use --stage-size
stage 2+: use --post-first-stage-size
```

This keeps the best known first-stage shape while allowing G2 and later stages
to grow one or two grafts at a time.

Acceptance remains strict:

```text
approve only if candidate_composed_loss improves previous_composed_loss
```

## Dry-Run

Run:

```text
runs/phase16_marco4h_adaptive_stage_dryrun
```

Result:

```text
marco: 4h_fine_grained_second_stage
stage 1 size: 2
stage 2 size: 1
recompose_abs_diff: 0.0
```

The dry-run used permissive acceptance only to validate that adaptive stage
sizing, checkpoint composition, and target maps work.

## Recommended Command

```powershell
cd E:\dev\ai\SAINT-G
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\SAINT-G\runs\phase16_marco4h_fine_g2_24graft `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
  --stage-size 4 `
  --post-first-stage-size 1 `
  --max-stages 8 `
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
  --candidate-targets blocks.1 blocks.2 blocks.3 `
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 `
  --candidate-init-scales 0.001 0.005 0.01 `
  --candidate-activations silu
```

This tests the best known G1 pattern, then probes smaller follow-up grafts.

## Criteria

Marco 4H passes if:

```text
composed_loss < 10.414729
accepted_grafts > 4
recompose_abs_diff = 0.0
```

Strong pass:

```text
accepted_grafts >= 6
no composed regression
```

## 24-Graft Result

Run:

```text
runs/phase16_marco4h_fine_g2_24graft
```

Result:

| metric | value |
|---|---:|
| base_loss | 10.416174 |
| composed_loss | 10.414671 |
| accumulated_gain | 0.001504 |
| accepted_groups | 2 |
| accepted_grafts | 5 |
| recompose_abs_diff | 0.0 |

Stage summary:

| stage | grafts | target | lr | init_scale | decision | gain |
|---:|---:|---|---:|---:|---|---:|
| 1 | 0-3 | blocks.2 | 3e-8 | 0.01 | approved | 0.001450 |
| 2 | 4 | blocks.3 | 1e-7 | 0.01 | approved | 0.000054 |
| 3 | 5 | blocks.1 | 3e-8 | 0.001 | rejected | 0.000000 |

Comparison:

```text
Marco 4G light gain: 0.001446
Marco 4H gain:       0.001504
extra gain:          ~0.000058
```

Verdict:

```text
Marco 4H passed.
```

The result shows that G2 was not impossible. The previous `stage_size=4` was too
coarse after G1. Reducing follow-up stages to one graft allowed the router to
accept an additional graft on `blocks.3` without composed regression.

The next bottleneck is stage 3. After five accepted grafts, the current local
grid again fails to find a positive composed gain.
