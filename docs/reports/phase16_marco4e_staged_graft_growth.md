# Phase 16 Marco 4E - Staged Graft Growth

Status: **implemented, dry-run validated**.

## Motivation

Marco 4D showed that 24 grafts trained simultaneously are not stable enough,
even with early stopping. However, 4 grafts produced a positive best checkpoint.

Observed 4-graft result:

| metric | value |
|---|---:|
| graft count | 4 |
| trainable params | 19,882,756 |
| CUDA peak | 746 MB |
| best eval step | 5,000 |
| best eval loss | 10.414828 |
| best eval gain | 0.001346 |
| best checkpoint size | 79.5 MB |
| best recompose abs diff | 0.0 |

This suggests:

```text
small approved graft groups > 24 simultaneous grafts
```

## Goal

Train grafts in accepted stages:

```text
G1: train 4 grafts
accept if best_eval_gain > 0
freeze G1

G2: add 4 grafts
train with G1 active/frozen
accept if validation does not regress
freeze G2

repeat until 24 grafts
```

## Procedure

1. Start from the frozen DRM 5M checkpoint.
2. Attach the first group of 4 grafts.
3. Train with early stopping and best checkpoint.
4. Accept the group only if `best_eval_gain > 0`.
5. Reload the accepted best checkpoint.
6. Freeze accepted graft parameters.
7. Attach the next group of 4 grafts.
8. Train the new group while previous groups remain active.
9. Reject or defer a group if it degrades validation.
10. Continue until 24 total grafts or no more groups are accepted.

## Acceptance Policy

Minimum policy:

```text
approve if best_eval_gain > 0
reject if best_eval_gain <= 0
```

Stronger policy:

```text
approve if best_eval_loss < previous_best_loss - min_delta
reject if validation regresses for patience evaluations
defer if gain is positive but below threshold
```

## Metrics

Per stage:

- base loss before stage;
- best eval loss;
- best eval gain;
- final loss;
- accepted/rejected/deferred;
- active graft count;
- frozen graft count;
- CUDA peak;
- stage time;
- checkpoint size;
- recomposition diff.

Final report:

- accumulated gain;
- distance to full 125M smoke;
- best stage;
- total checkpoint size;
- total train time;
- comparison against best isolated 4-graft run;
- comparison against 24 simultaneous graft run.

## Criteria

Marco 4E passes if:

```text
G1 improves validation
G1+G2 does not destroy G1
composed checkpoint reloads exactly
accumulated gain > best isolated 4-graft gain
```

It fails if:

- G1 cannot be reproduced;
- later stages consistently destroy earlier gains;
- composed checkpoint does not reload;
- accumulated gain remains below the isolated 4-graft result.

## Implementation

The graftblock benchmark now supports staged growth:

```text
--stage-size 4
--max-stages 6
--freeze-accepted-stages
--stage-accept-min-gain 0.0
--training-mode staged
```

The output should include:

```text
stage_metrics.json
composed_graft_checkpoint.pt
results.md
```

Dry-run:

```text
runs/phase16_marco4e_staged_dryrun
```

Dry-run result:

| metric | value |
|---|---:|
| accepted stages | 0 |
| accepted grafts | 0 |
| accumulated gain | 0.0 |
| checkpoint size | 3,151,043 bytes |
| recompose abs diff | 0.0 |

The dry-run used a small hidden size and short budget only to validate the
runtime path. It correctly produced the required artifacts and rejected a stage
with no positive gain.

## 24-Graft Staged Result

Run:

```text
runs/phase16_marco4e_staged_24graft
```

Result:

| metric | value |
|---|---:|
| base loss | 10.416174 |
| composed loss | 10.414818 |
| accumulated gain | 0.001357 |
| accepted stages | 1 |
| accepted grafts | 4 |
| CUDA peak | 1.30 GB |
| composed checkpoint size | 477,208,263 bytes |
| recomposed loss | 10.414818 |
| recompose abs diff | 0.0 |

Stage decisions:

| stage | graft ids | decision | gain | best step |
|---:|---|---|---:|---:|
| 1 | 0-3 | approved | 0.001357 | 5,000 |
| 2 | 4-7 | rejected | 0.000000 | 0 |

Interpretation:

The staged policy behaved as intended:

```text
G1 improved validation
G2 did not improve validation
G2 was rejected
the composed checkpoint preserved G1
the composed checkpoint reloads exactly
```

This passes the Marco 4E infrastructure and acceptance-policy test. It also
slightly beats the previous isolated 4-graft result:

```text
isolated 4-graft best gain: 0.001346
staged G1 accumulated gain: 0.001357
```

The gain is still small, but the result confirms that staged growth is a better
direction than training 24 grafts simultaneously.

## Next Step

Marco 4F should route staged candidates by validation gain.

Instead of fixed stage order:

```text
G1 = grafts 0-3
G2 = grafts 4-7
```

use validation-routed selection:

```text
test candidate graft groups across blocks.0..5
rank by best_eval_gain
accept top group
freeze accepted group
repeat
```

## Recommended Command

```powershell
cd E:\dev\ai\SAINT-G
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir runs\phase16_marco4e_staged_24graft `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
  --stage-size 4 `
  --max-stages 6 `
  --freeze-accepted-stages `
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
  --training-mode staged
```
