# Phase 16 Marco 4F - Validation-Routed Staged Grafts

Status: **passed**.

## Motivation

Marco 4E proved that staged growth works:

```text
stage 1, grafts 0-3: approved, gain 0.001357
stage 2, grafts 4-7: rejected, gain 0.000000
```

The composed checkpoint preserved the accepted group and recomposed exactly:

```text
recompose_abs_diff: 0.0
```

This means the next bottleneck is candidate selection. Fixed index order is too
weak. The next stage should not blindly try grafts 4-7; it should search for the
most useful group.

## Goal

Select graft groups by validation gain.

Instead of:

```text
G1 = grafts 0-3
G2 = grafts 4-7
```

use:

```text
candidate groups across blocks.0..5
train each briefly
rank by best_eval_gain
accept the best group
freeze accepted group
repeat
```

## Candidate Strategy

Candidate groups can vary by:

- target block;
- target order;
- stage seed;
- hidden size;
- learning rate;
- activation function;
- stage size.

Initial candidate set:

```text
blocks.0
blocks.1
blocks.2
blocks.3
blocks.4
blocks.5
```

Each candidate gets the same short budget. The first implementation scored the
local probe by `best_eval_gain`, but the real criterion must be the loss of the
composed checkpoint after adding that candidate.

Correct score:

```text
candidate_composed_gain = previous_composed_loss - candidate_composed_loss
```

The candidate is approved only if it improves the composed model. Local probe
gain is still reported, but it is diagnostic, not the acceptance metric.

The composed score is computed from the candidate's best observed validation
state, not from the final state after early stopping. This matters because a
candidate can briefly improve validation and then degrade before the patience
limit stops training.

Later, score should include cost:

```text
score = best_eval_gain / checkpoint_bytes
score = best_eval_gain / train_seconds
score = best_eval_gain / trainable_parameters
```

## Acceptance Policy

Minimum:

```text
approve top candidate if candidate_composed_gain > 0
reject all if no candidate improves validation
```

Stronger:

```text
approve if best_eval_loss < previous_best_loss - min_delta
defer if gain is positive but below threshold
reject if gain <= 0
```

## Outputs

The benchmark should produce:

```text
candidate_metrics.json
candidate_training_metrics.jsonl
stage_metrics.json
composed_graft_checkpoint.pt
summary.json
results.md
```

`candidate_metrics.json` must include:

```text
previous_composed_loss
candidate_composed_loss
candidate_composed_gain
candidate_target_by_graft
```

The composed checkpoint also stores `target_by_graft`, because routed stages can
place different graft groups on different DRM blocks.

Implemented mode:

```text
--training-mode validation_routed_staged
--candidate-targets blocks.0 blocks.1 ...
```

Fix dry-run:

```text
runs/phase16_marco4f_fix_dryrun
```

Dry-run artifacts:

```text
candidate_metrics.json
candidate_training_metrics.jsonl
stage_metrics.json
composed_graft_checkpoint.pt
summary.json
results.md
```

Dry-run result:

| metric | value |
|---|---:|
| candidates | 2 |
| accepted groups | 0 |
| accumulated gain | 0.0 |
| recompose_abs_diff | 0.0 |
| composed checkpoint size | 790,637 bytes |

The dry-run used tiny grafts and short training only to validate the runtime
path. It correctly rejected candidates that did not improve composed validation,
saved `candidate_composed_loss` in candidate metrics, stored the target map, and
recomposed with zero loss drift.

## Bug Found By The 24-Graft Run

The first full 4F run accepted all six groups, but final composed validation got
worse:

```text
base_loss: 10.416174
composed_loss: 10.440719
accumulated_gain: -0.024545
accepted_groups: 6
accepted_grafts: 24
recompose_abs_diff: 0.0
```

This was not a failure of recomposition. It exposed a metric bug:

```text
old rule: approve by local candidate gain
fixed rule: approve by composed validation gain
fixed state: compose from best_state_payload, not final_state_payload
```

The result remains useful because it proved the routed benchmark executes and
that checkpoint recomposition is exact, while showing that acceptance must be
tied to the composed model.

## Fixed 24-Graft Result

After fixing both acceptance and state selection, Marco 4F was rerun with:

```text
output: runs/phase16_marco4f_best_payload_24graft
graft_count: 24
stage_size: 4
candidate_targets: blocks.0..5
learning_rate: 3e-7
```

Result:

| metric | value |
|---|---:|
| base_loss | 10.416174 |
| composed_loss | 10.414808 |
| accumulated_gain | 0.001366 |
| accepted_groups | 1 |
| accepted_grafts | 4 |
| selected target | blocks.2 |
| recompose_abs_diff | 0.0 |

Stage summary:

| stage | selected target | decision | composed gain |
|---:|---|---|---:|
| 1 | blocks.2 | approved | 0.001366 |
| 2 | blocks.0 | rejected | 0.000000 |

Interpretation:

```text
Marco 4F passed.
```

It did not materially beat Marco 4E, but it fixed the router scientifically:

- candidates are accepted by composed validation loss;
- the best validation state is used for composition;
- the chosen target is stored per graft;
- recomposition remains exact.

The next limitation is not acceptance correctness. It is candidate diversity.
After the first useful group, the current candidate set cannot find a second
group with positive composed validation gain.

## Marco 4G Direction

Marco 4G should widen the candidate grid:

```text
target block x learning_rate x init_scale x activation
```

The goal is to find a positive second group after G1, not merely to repeat
fixed-target routing.

## Recommended Command

```powershell
cd E:\dev\ai\SAINT-G
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

.\.venv\Scripts\python.exe `
  scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir runs\phase16_marco4f_fix_24graft `
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
  --candidate-targets blocks.0 blocks.1 blocks.2 blocks.3 blocks.4 blocks.5
```

## Criteria

Marco 4F passes if:

```text
selected candidate beats fixed-order stage 2
accumulated gain > Marco 4E
checkpoint composed reloads exactly
VRAM remains controlled
```

It fails if:

- no candidate beats fixed order;
- candidate search costs more than the gain justifies;
- accepted stages regress after composition;
- checkpoint composition breaks.
