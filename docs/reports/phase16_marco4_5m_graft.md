# Phase 16 Marco 4 - DRM 5M Grafted Smoke

Status: **initial positive signal**.

## Context

This run starts the Phase 16 comparison:

```text
DRM full 125M smoke
vs
DRM 5M smoke
vs
DRM 5M + SAINT-G Phi graft
```

The full 125M baseline is not a complete pretraining run. It is a 100-step
operational smoke on the `data/multilingual_125m` dataset.

## Full 125M Smoke Reference

Source:

```text
drm_transformer/checkpoints/multilingual_125m/smoke_100
```

Metrics:

| metric | value |
|---|---:|
| steps | 100 |
| tokens | 819,200 |
| total time | 15,581.5 s |
| average tokens/s | 53 |
| validation loss | 9.049912 |
| validation perplexity | 8,517.79 |
| checkpoint size | ~1.83 GB each |

The run is valid as a full-model smoke, but too slow to use as a long baseline
without further optimization.

## DRM 5M Smoke Reference

Source:

```text
drm_transformer/checkpoints/multilingual_5m/smoke_819k
```

Command shape:

```text
5M config
data/multilingual_125m
819,200 tokens
50 optimizer steps
```

Metrics:

| metric | value |
|---|---:|
| steps | 50 |
| tokens | 819,200 |
| total time | 67.3 s |
| average tokens/s | 12,172 |
| validation loss | 10.438569 |
| validation perplexity | 34,151.74 |
| checkpoint size | ~68.5 MB each |

This gives the Phase 16 base gap:

```text
full 125M val_loss: 9.049912
DRM 5M val_loss:   10.438569
gap:                1.388657
```

## Graft Smoke

Script:

```text
scripts/benchmark_drm_g_phase16_5m_graft.py
```

Confirmed run:

```text
runs/phase16_marco4_5m_graft_ffn_confirm
```

Configuration:

| field | value |
|---|---|
| base checkpoint | `drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt` |
| dataset | `drm_transformer/data/multilingual_125m` |
| target | `blocks.5.ffn.down_proj` |
| learning rate | `0.0002` |
| train batches | 8 |
| validation batches | 8 |
| seeds | 42, 43, 44 |
| steps | 8 |

Results:

| method | mean base | mean final | mean gain | gain/param | positive | params |
|---|---:|---:|---:|---:|---:|---:|
| `phi_ls_full_rank` | 10.374700 | 10.374586 | 0.000114 | 1.236158e-08 | 3/3 | 9,216 |
| `phi_zero_full_rank` | 10.374700 | 10.374586 | 0.000114 | 1.234865e-08 | 3/3 | 9,216 |
| `phi_ls_residual_full_rank` | 10.374700 | 10.374603 | 0.000097 | 1.063443e-08 | 3/3 | 9,121 |
| `full_module_linear` | 10.374700 | 10.374725 | -0.000025 | -6.801673e-10 | 1/3 | 36,864 |
| `phi_ls_train_ab_half_rank` | 10.374700 | 10.376324 | -0.001623 | -1.409264e-07 | 0/3 | 11,520 |

Best single run:

| field | value |
|---|---:|
| method | `phi_zero_full_rank` |
| seed | 44 |
| validation gain | 0.000249 |
| gain/parameter | 2.699542e-08 |
| trainable parameters | 9,216 |

## Interpretation

Marco 4 has an initial positive signal:

- at least one Phi graft improves the DRM 5M checkpoint;
- the best Phi methods improve validation in 3/3 seeds;
- Phi uses 9,216 trainable parameters;
- the full-module control uses 36,864 parameters and is worse on average in
  this run;
- the gain is real but very small relative to the gap between 5M and full 125M.

This does **not** mean the 5M grafted model has reached 125M quality. It means
the Phase 16 grafting path is operational and can produce measurable validation
gain on the same dataset.

## Known Issues

- `metrics.json` in `drm_transformer` currently reports `final_loss: 0.0` after
  training because the trainer resets `running_loss` after the final log.
  The reliable final train loss is in `training_log.json`.
- The full 125M smoke is too slow for long baselines without profiling.
- The current graft benchmark trains small hooks, not a full progressive graft
  sequence.

## Next Step

Marco 4B implemented the explicit `5M + 5M graft * 24 ~= 125M` path.

See:

```text
docs/reports/phase16_marco4b_graftblock_5m.md
```

The next step is to train the 24-graft path progressively and compare 4, 8, 16
and 24 grafts with the same token budget.
