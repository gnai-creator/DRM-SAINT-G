# Phase 16 Marco 4D - Early Stopping and Best Graft Checkpoint

Status: **implemented and dry-run validated**.

## Why

The 4-hour 24-graft run proved that the infrastructure works, but it also
showed that long training without validation control can degrade the model.

Result from the 4-hour run:

| metric | value |
|---|---:|
| base loss | 10.416174 |
| final loss | 10.683245 |
| validation gain | -0.267071 |
| trained steps | 200,965 |
| train time | 14,400 s |
| CUDA peak | 3.43 GB |
| recomposed loss diff | 0.0 |

The correct conclusion is:

```text
DRM 5M + 24 grafts is efficient in VRAM and speed,
but long training needs validation-gated checkpoints and early stopping.
```

## Implementation

The graftblock benchmark now supports:

- periodic validation with `--eval-every-steps`;
- best checkpoint saving with `--save-best-checkpoint`;
- final checkpoint saving with `--save-graft-checkpoint`;
- early stopping with `--early-stopping-patience`;
- minimum validation improvement with `--early-stopping-min-delta`;
- `training_metrics.jsonl` with validation history.

Script:

```text
scripts/benchmark_drm_g_phase16_graftblock.py
```

## Dry Run

Run:

```text
runs/phase16_marco4d_earlystop_dryrun
```

Settings:

```text
graft_count: 24
hidden_size: 25,889
learning_rate: 3e-7
lr_decay: 0.02
training_mode: simultaneous
max_train_seconds: 30
eval_every_steps: 200
early_stopping_patience: 2
batch_size: 2
seq_len: 128
validation_batches: 4
```

Results:

| metric | value |
|---|---:|
| trained steps | 652 |
| final loss | 10.415352 |
| final validation gain | 0.000822 |
| best eval step | 600 |
| best eval loss | 10.415417 |
| best eval gain | 0.000757 |
| CUDA peak | 3.43 GB |
| best checkpoint size | 477,208,341 bytes |
| best recomposed loss | 10.415417 |
| best recompose abs diff | 0.0 |

The best checkpoint path passes: the artifact reloads and reproduces the
validation loss exactly on the tested slices.

## Recommended 4-Hour Command

Use a lower learning rate and validation gating:

```powershell
$env:PYTHONPATH="E:\dev\ai\SAINT-G"

E:\dev\ai\SAINT-G\.venv\Scripts\python.exe `
  E:\dev\ai\SAINT-G\scripts\benchmark_drm_g_phase16_graftblock.py `
  --output-dir E:\dev\ai\SAINT-G\runs\phase16_marco4d_24graft_4h_best `
  --checkpoint E:\dev\ai\drm_transformer\checkpoints\multilingual_5m\smoke_819k\final.pt `
  --data-dir E:\dev\ai\drm_transformer\data\multilingual_125m `
  --device cuda `
  --seeds 42 `
  --graft-count 24 `
  --hidden-size 25889 `
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
  --training-mode simultaneous `
  --save-best-checkpoint `
  --save-graft-checkpoint
```

## Next Experiment Matrix

Run the same protocol for:

```text
graft_count: 4, 8, 16, 24
learning_rate: 1e-7, 3e-7
```

Compare:

- best validation loss;
- final validation loss;
- best step;
- checkpoint size;
- CUDA peak;
- gain per parameter;
- distance to the full 125M smoke loss.

## Verdict

Marco 4D fixes the main flaw in Marco 4C: the final checkpoint no longer has to
represent the best point in the training curve. The next quality comparison
should use `best_eval_loss` and the `best_graft_checkpoint`, not only
`final_loss`.
