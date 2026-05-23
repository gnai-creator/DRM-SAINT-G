# Phase 16 Marco 4J - Two-Pass Candidate Pruning

Status: **implemented, pending CUDA run**.

## Goal

Marco 4I showed that full candidate grids are too expensive because
`max_train_seconds` is applied per candidate. With 54 candidates per stage, a
single stage can take many hours.

Marco 4J changes the search strategy:

```text
probe many candidates cheaply
  -> rank by composed validation gain
  -> keep top-k
  -> train only top-k deeply
  -> accept only if composed loss improves
```

## New CLI Options

```text
--candidate-probe-steps
--candidate-probe-max-train-seconds
--candidate-top-k
```

Example behavior:

```text
candidate grid: 6 targets * 3 lrs * 3 scales = 54 candidates
probe pass: train all 54 candidates briefly
deep pass: train only top 4 candidates
```

This keeps the broad search signal without paying full training cost for every
candidate.

## Acceptance Rule

The approval rule remains strict:

```text
candidate_composed_gain > stage_accept_min_gain
```

Probe ranking does not approve grafts. It only decides which candidates deserve
the expensive deep pass.

## Recommended Command

```powershell
cd E:\dev\ai\DRM-SAINT-G

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\DRM-SAINT-G\runs\phase16_marco4j_two_pass_24graft `
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
  --max-train-seconds 1800 `
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
  --orthogonal-penalty 0.00001 `
  --candidate-probe-steps 1000 `
  --candidate-probe-max-train-seconds 180 `
  --candidate-top-k 4
```

## Expected Runtime

Per stage, the upper bound becomes roughly:

```text
54 probe candidates * 180s = 2.7h
4 deep candidates * 1800s = 2.0h
total upper bound per stage = 4.7h
```

Early stopping should reduce this. The important difference is that the deep
pass is capped to top-k instead of every candidate.

For a faster probe:

```text
--candidate-targets blocks.1 blocks.2 blocks.3
--candidate-learning-rates 0.00000003 0.0000001
--candidate-init-scales 0.005 0.01
--candidate-probe-max-train-seconds 90
--candidate-top-k 3
```

## Criteria

Marco 4J is useful if it achieves at least one:

```text
composed_loss <= 10.414670705795288
accepted_grafts > 5 with recomposition exactness
same quality as 4H with lower routing time
```

It fails if:

```text
probe ranking consistently selects weak candidates
runtime remains close to full-grid routing
composed loss regresses against 4H
```

## Next Step After Run

After the CUDA run, compare:

```text
4H best composed_loss: 10.414670705795288
4I light composed_loss: 10.414714097976685
4J two-pass composed_loss: pending
```

If 4J improves speed without hurting quality, it becomes the default candidate
router for the next Phase 16 experiments.
