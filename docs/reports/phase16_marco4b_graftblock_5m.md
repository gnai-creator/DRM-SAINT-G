# Phase 16 Marco 4B - DRM 5M + 5M GraftBlocks

Status: **operational smoke complete**.

## Goal

Test the concrete path:

```text
DRM 5M + 24 graft blocks of about 5M params each ~= DRM 125M effective budget
```

This is different from the earlier Phi hook benchmark. The Marco 4B unit is a
residual module attached to a DRM block output:

```text
h_out = h + scale * down(silu(up(h)))
```

For the multilingual DRM 5M config:

```text
d_model = 96
params_per_graft ~= 2 * d_model * hidden + 1
hidden ~= 25,889
params_per_graft ~= 4.97M
```

With 24 grafts:

```text
base DRM 5M params:      5,699,059
graft params:         119,296,536
effective total:      124,995,595
remaining gap:              4,405
```

## Script

```text
scripts/benchmark_drm_g_phase16_graftblock.py
```

The script writes:

- `results.json`;
- `summary.json`;
- `results.md`.

It also emits a planner table for 1, 2, 4, 8, 16 and 24 graft counts.

## Data and Checkpoint

```text
base checkpoint:
drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt

dataset:
drm_transformer/data/multilingual_125m
```

## Planner

| grafts | hidden | params/graft | graft params | effective total | remaining gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 621,359 | 119,300,929 | 119,300,929 | 124,999,988 | 12 |
| 2 | 310,679 | 59,650,369 | 119,300,738 | 124,999,797 | 203 |
| 4 | 155,339 | 29,825,089 | 119,300,356 | 124,999,415 | 585 |
| 8 | 77,669 | 14,912,449 | 119,299,592 | 124,998,651 | 1,349 |
| 16 | 38,834 | 7,456,129 | 119,298,064 | 124,997,123 | 2,877 |
| 24 | 25,889 | 4,970,689 | 119,296,536 | 124,995,595 | 4,405 |

The 24-graft plan is the cleanest match to the original idea:

```text
5M base + 5M graft * 24
```

## Smoke Results

All runs used:

```text
seed: 42
steps: 4
batch_size: 2
seq_len: 128
train_batches: 4
validation_batches: 4
learning_rate: 0.0002
targets: blocks.1..blocks.5
```

| run | grafts | hidden | trainable params | effective params | validation gain | CUDA peak |
|---|---:|---:|---:|---:|---:|---:|
| unit 5M | 1 | 26,042 | 5,000,065 | 10,699,124 | 0.000141 | 381 MB |
| stack 10M | 2 | 26,042 | 10,000,130 | 15,699,189 | 0.000334 | 495 MB |
| stack 20M | 4 | 26,042 | 20,000,260 | 25,699,319 | 0.000811 | 755 MB |
| exact 125M | 24 | 25,889 | 119,296,536 | 124,995,595 | 0.000261 | 3.43 GB |

## Interpretation

The path is now technically valid:

- the 5M checkpoint loads;
- graft blocks attach to real DRM block outputs;
- the exact 125M effective budget can be constructed;
- the 24-graft model runs on CUDA without OOM in a short smoke;
- validation gain is positive in the exact 125M smoke.

The result is not yet a quality win against full 125M. The validation gain is
small and the 24-block run did not beat the smaller 4-block run in this very
short training window. That suggests naive stacking can conflict or undertrain.

## Next Technical Step

Marco 4C has been implemented.

See:

```text
docs/reports/phase16_marco4c_graftblock_training.md
```

Summary:

- 24 grafts now save and reload as a recomposable checkpoint;
- recomposed validation loss matches exactly;
- 4/8/16/24 equal-step comparison exists;
- infrastructure passed;
- quality did not pass yet because naive 24-graft progressive training degraded
  validation.
