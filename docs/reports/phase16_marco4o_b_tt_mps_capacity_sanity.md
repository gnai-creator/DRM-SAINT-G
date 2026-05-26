# Phase 16 Marco 4O-B - TT/MPS Capacity Sanity Sweep

Status: **implemented and executed on CUDA for seed 42**.

## Objective

Marco 4O-B gives the TT/MPS adapter family one capacity-oriented sanity check after the first 4O smoke sweep failed at `adapter_width=128`.

The question is narrower than the original 4O question:

```text
If the TT/MPS bottleneck is made larger and candidate probes are longer, can stage 1 produce a positive composed gain?
```

If not, the TT/MPS path should be deprioritized for now.

## Implementation change

The benchmark now supports an explicit marco label:

```text
--marco-label 4o_b_tt_mps_capacity_sanity
```

This lets follow-up runs preserve their own `summary.json` marco names while reusing the same routed/staged implementation.

TDD coverage was added in:

```text
tests/test_marco_names.py
```

## Sweep design

TT/MPS capacity sweep:

```text
adapter_width: 256, 512
chi: 4, 8, 16
seed: 42
max_stages: 1
candidate_probe_steps: 200
candidate_top_k: 3
candidate_probe_max_train_seconds: 90
max_train_seconds for deep candidates: 180
```

Dense parameter-matched sanity controls:

```text
hidden_size 300  ~= width 256 / chi 16 TT-MPS params per graft
hidden_size 620  ~= width 512 / chi 16 TT-MPS params per graft
```

The dense controls use the same stage-1 routed/staged protocol and candidate grid, but `adapter_type=dense_graftblock`.

## Command pattern

Executed from `/home/rato/dev/ai/SAINT-G`.

TT/MPS runs:

```bash
for width in 256 512; do
  for chi in 4 8 16; do
    out="/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_b_tt_mps_capacity_seed42_w${width}_chi${chi}"
    .venv/bin/python scripts/benchmark_drm_g_phase16_graftblock.py \
      --output-dir "$out" \
      --marco-label 4o_b_tt_mps_capacity_sanity \
      --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
      --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
      --device cuda \
      --seeds 42 \
      --graft-count 24 \
      --adapter-type tt_mps \
      --tt-adapter-width "$width" \
      --tt-bond-dim "$chi" \
      --hidden-size "$width" \
      --stage-size 4 \
      --post-first-stage-size 1 \
      --max-stages 1 \
      --stage-accept-min-gain 0.0 \
      --steps 100000000 \
      --max-train-seconds 180 \
      --eval-every-steps 200 \
      --early-stopping-patience 2 \
      --early-stopping-min-delta 0.00001 \
      --batch-size 2 \
      --seq-len 128 \
      --validation-batches 4 \
      --train-batches 1024 \
      --learning-rate 0.0000003 \
      --lr-decay 0.02 \
      --training-mode validation_routed_staged \
      --candidate-targets blocks.2 blocks.3 blocks.4 \
      --candidate-learning-rates 0.0000001 0.0000003 \
      --candidate-init-scales 0.001 0.005 \
      --candidate-activations silu \
      --candidate-score-mode composed_gain_orthogonal \
      --orthogonal-penalty 0.00001 \
      --candidate-probe-steps 200 \
      --candidate-probe-max-train-seconds 90 \
      --candidate-top-k 3
  done
done
```

Dense controls used the same command with:

```text
--marco-label 4o_b_dense_parameter_matched_sanity
--hidden-size 300
```

and:

```text
--marco-label 4o_b_dense_parameter_matched_sanity
--hidden-size 620
```

## Artifacts

TT/MPS:

```text
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w256_chi4/
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w256_chi8/
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w256_chi16/
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w512_chi4/
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w512_chi8/
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w512_chi16/
```

Dense controls:

```text
runs/phase16_marco4o_b_dense_matched_seed42_h300_match_w256/
runs/phase16_marco4o_b_dense_matched_seed42_h620_match_w512/
```

Each run contains:

```text
summary.json
stage_metrics.json
candidate_metrics.json
candidate_training_metrics.jsonl
ntk_activation_probe_metrics.json
results.md
composed_graft_checkpoint.pt
```

## Results

| run | adapter | width | chi/hidden | params/graft | checkpoint bytes | composed_loss | accepted_grafts | recompose_abs_diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| w256 chi4 | TT/MPS | 256 | 4 | 51,201 | 4,954,731 | 10.416174 | 0 | 0.0 |
| w256 chi8 | TT/MPS | 256 | 8 | 53,249 | 5,151,339 | 10.416174 | 0 | 0.0 |
| w256 chi16 | TT/MPS | 256 | 16 | 57,345 | 5,544,619 | 10.416174 | 0 | 0.0 |
| w512 chi4 | TT/MPS | 512 | 4 | 103,425 | 9,968,235 | 10.416174 | 0 | 0.0 |
| w512 chi8 | TT/MPS | 512 | 8 | 108,545 | 10,459,755 | 10.416174 | 0 | 0.0 |
| w512 chi16 | TT/MPS | 512 | 16 | 118,785 | 11,442,859 | 10.416174 | 0 | 0.0 |
| h300 match_w256 | dense | - | 300 | 57,601 | 5,553,159 | 10.416174 | 0 | 0.0 |
| h620 match_w512 | dense | - | 620 | 119,041 | 11,451,399 | 10.416174 | 0 | 0.0 |

All stage-1 deep candidates were rejected. Some probe/candidate rows show tiny positive differences around `2e-7` to `7e-7`, but those are below the configured `early_stopping_min_delta=1e-5` and did not survive as an accepted stage-level composed gain.

## Interpretation

4O-B answers the capacity sanity question negatively under this protocol:

```text
Increasing TT/MPS adapter_width from 128 to 256/512 and increasing probes to 200 steps did not produce a meaningful positive stage-1 gain.
```

The dense parameter-matched controls also failed under the same short stage-1 protocol. That means this 4O-B result should be read as:

```text
The short stage-1 capacity sanity protocol did not find learnable small/medium adapters.
```

not as:

```text
TT/MPS is uniquely worse than dense matched adapters.
```

Against the previous 4N-B style dense run, 4O-B still loses clearly because it accepts zero grafts and does not move composed loss.

## Verdict

Marco 4O-B is **implemented, reproducible, and negative**.

Current recommendation:

```text
Deprioritize TT/MPS adapters for Phase 16 unless a future experiment changes the optimization protocol substantially.
```

The next useful direction is not another TT/MPS capacity increase. The next useful direction is:

```text
cost-aware dense graft routing / efficiency-aware candidate scoring
```

because dense graft blocks have already shown useful accepted-graft behavior in earlier 4K/4L/4N-B runs, while TT/MPS did not produce stage-1 gain even after the 4O-B capacity sanity sweep.
