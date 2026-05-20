# Phase 16 Marco 4F - Validation-Routed Staged Grafts

Status: **planned**.

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

Each candidate gets the same short budget and is scored by:

```text
score = best_eval_gain
```

Later, score should include cost:

```text
score = best_eval_gain / checkpoint_bytes
score = best_eval_gain / train_seconds
score = best_eval_gain / trainable_parameters
```

## Acceptance Policy

Minimum:

```text
approve top candidate if best_eval_gain > 0
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
stage_metrics.json
composed_graft_checkpoint.pt
results.md
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
