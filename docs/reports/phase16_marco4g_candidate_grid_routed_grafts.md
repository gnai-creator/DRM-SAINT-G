# Phase 16 Marco 4G - Candidate Grid Routed Grafts

Status: **implemented, dry-run validated**.

## Goal

Marco 4F fixed routed acceptance, but it still accepted only one group:

```text
G1: accepted on blocks.2
G2: rejected with composed_gain 0.0
```

Marco 4G widens the candidate search so the second group is not limited to only
target selection.

## Candidate Grid

The routed benchmark now supports:

```text
--candidate-targets
--candidate-learning-rates
--candidate-init-scales
--candidate-activations
```

Each candidate is:

```text
target x learning_rate x init_scale x activation
```

Acceptance remains unchanged:

```text
approve only if candidate_composed_loss improves previous_composed_loss
```

The composed score still uses the candidate's best validation state, not the
final state.

## Dry-Run

Command shape:

```text
targets: blocks.1, blocks.2
learning_rates: 1e-7, 3e-7
init_scales: 0.01
activations: silu
```

Result:

```text
runs/phase16_marco4g_grid_dryrun
candidates: 4
accepted_groups: 0
accumulated_gain: 0.0
recompose_abs_diff: 0.0
```

This validates the grid runtime and artifacts. The tiny dry-run is not intended
to improve loss.

## Recommended Light Probe

Do not start with the full grid. The full grid below has:

```text
6 targets x 3 learning rates x 2 init scales x 2 activations = 72 candidates per stage
```

With up to 6 stages, that can become expensive. Start with this lighter probe:

```powershell
cd E:\dev\ai\SAINT-G
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\SAINT-G\runs\phase16_marco4g_light_probe_24graft `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
  --stage-size 4 `
  --max-stages 3 `
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
  --candidate-learning-rates 0.0000001 0.0000003 `
  --candidate-init-scales 0.01 `
  --candidate-activations silu
```

This tests:

```text
3 targets x 2 learning rates = 6 candidates per stage
```

Expected runtime is roughly tens of minutes for the first stages, instead of
many hours.

## Full 24-Graft Grid

Use this only after the light probe shows a promising region:

```powershell
cd E:\dev\ai\SAINT-G
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\SAINT-G\runs\phase16_marco4g_grid_24graft `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
  --stage-size 4 `
  --max-stages 6 `
  --stage-accept-min-gain 0.0 `
  --steps 100000000 `
  --max-train-seconds 14400 `
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
  --candidate-learning-rates 0.0000001 0.0000003 0.0000006 `
  --candidate-init-scales 0.005 0.01 `
  --candidate-activations silu gelu
```

## Criteria

Marco 4G passes if at least one is true:

- accumulated gain beats Marco 4F;
- more than one group is accepted with no composed regression;
- the grid identifies why later groups remain unproductive.

Checkpoint recomposition must remain exact.

## Light Probe Result

Run:

```text
runs/phase16_marco4g_light_probe_24graft
```

Result:

| metric | value |
|---|---:|
| base_loss | 10.416174 |
| composed_loss | 10.414729 |
| accumulated_gain | 0.001446 |
| accepted_groups | 1 |
| accepted_grafts | 4 |
| selected_target | blocks.2 |
| learning_rate | 1e-7 |
| init_scale | 0.01 |
| activation | silu |
| recompose_abs_diff | 0.0 |

Comparison:

```text
Marco 4F gain: 0.001366
Marco 4G light gain: 0.001446
extra gain: ~0.000079
```

Verdict:

```text
Marco 4G passed as an incremental improvement.
```

The grid found a better first group than Marco 4F, but still did not unlock a
second accepted group. The next question is whether G2 fails because there is no
remaining gain or because `stage_size=4` is too coarse after G1.

## Next: Marco 4H

Marco 4H should test fine-grained second-stage growth:

```text
G1: keep the best 4G setup
G2+: reduce stage size to 1 or 2
accept only by composed validation gain
```

Criterion:

```text
composed_loss < 10.414729
accepted_grafts > 4
recompose_abs_diff = 0.0
```

Strong criterion:

```text
accepted_grafts >= 6 or 8
no composed regression
```
