# Phase 16 Marco 4C - 24-Graft Trainability

Status: **implemented, not yet quality-passing**.

## Goal

Make the 24-graft path trainable enough to compare against the full 125M smoke.

Required checks:

- run 24 grafts for more steps;
- train grafts progressively instead of all at once;
- add per-graft learning-rate or scale scheduling;
- compare 4, 8, 16 and 24 grafts at equal tokens;
- save a recomposable graft checkpoint;
- evaluate against the full 125M smoke validation loss.

## Implementation

Script:

```text
scripts/benchmark_drm_g_phase16_graftblock.py
```

New capabilities:

- `--graft-counts 4 8 16 24`;
- `--training-mode progressive`;
- `--lr-decay`;
- `--scale-warmup-grafts`;
- `--save-graft-checkpoint`;
- recomposed checkpoint evaluation with exact loss comparison;
- distance to the full 125M smoke validation loss.

The recomposable checkpoint format stores:

- metadata;
- target modules;
- each graft block's `up`, `down`, `scale`;
- shape and activation metadata.

## Full 125M Reference

```text
full_125m_smoke_loss = 9.049912414550782
```

This comes from:

```text
drm_transformer/checkpoints/multilingual_125m/smoke_100
```

## Equal-Step Sweep

Run:

```text
runs/phase16_marco4c_progressive_sweep
```

Settings:

```text
graft_counts: 4, 8, 16, 24
hidden_size: 25,889
steps: 16
training_mode: progressive
learning_rate: 0.0002
lr_decay: 0.01
batch_size: 2
seq_len: 128
validation_batches: 4
```

Results:

| grafts | trainable params | final loss | validation gain | distance to full 125M | CUDA peak |
|---:|---:|---:|---:|---:|---:|
| 4 | 19,882,756 | 10.415268 | 0.000906 | 1.365356 | 744 MB |
| 8 | 39,765,512 | 10.418468 | -0.002294 | 1.368556 | 1.28 GB |
| 16 | 79,531,024 | 10.429472 | -0.013298 | 1.379560 | 2.35 GB |
| 24 | 119,296,536 | 10.434869 | -0.018694 | 1.384956 | 3.36 GB |

Interpretation:

More capacity does not automatically improve validation. The 4-graft run is the
best point in this short sweep. The 24-graft run is stable in memory but
under-optimized or conflicting.

## 24-Graft Checkpoint Test

Run:

```text
runs/phase16_marco4c_24graft_positive_checkpoint
```

Settings:

```text
graft_count: 24
hidden_size: 25,889
steps: 4
training_mode: simultaneous
learning_rate: 0.0002
batch_size: 2
seq_len: 128
validation_batches: 4
```

Results:

| metric | value |
|---|---:|
| base loss | 10.416174 |
| final loss | 10.415910 |
| validation gain | 0.000265 |
| distance to full 125M smoke | 1.365997 |
| trainable params | 119,296,536 |
| effective params | 124,995,595 |
| CUDA peak | 3.43 GB |
| checkpoint size | 477,208,463 bytes |
| recomposed loss | 10.415910 |
| recompose abs diff | 0.0 |

The checkpoint path passes: the saved graft artifact reloads and reproduces the
same validation loss exactly on the tested slices.

## Verdict

Marco 4C passes the infrastructure requirements:

- 24 grafts run on CUDA;
- 24 grafts can be saved as a recomposable checkpoint;
- reload/recompose validation matches exactly;
- distance to full 125M is now measured;
- 4/8/16/24 comparison exists at equal steps.

Marco 4C does **not** pass the quality requirement yet:

- the best 24-graft run improves validation only slightly;
- more steps with progressive training degraded validation;
- 4 grafts beat 24 grafts in the equal-step sweep;
- the gap to full 125M remains about `1.366` validation-loss points.

## Next Step

Marco 4D should stop treating all grafts equally:

- select graft targets by validation gain before training;
- train grafts in accepted stages, not just progressively by index;
- freeze accepted grafts before adding the next group;
- add early stopping per graft group;
- compare against a 4-graft and 24-graft budget with equal wall-clock and tokens.
