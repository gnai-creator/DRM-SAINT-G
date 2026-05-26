# Phase 16 Marco 4O - TT/MPS Adapter Baseline

Status: **implemented and smoke-executed on CUDA for seed 42, chi 2/4/8/16; 4O-B capacity sanity follow-up also completed and negative**.

## Objective

Marco 4O tests whether a PyTorch-native Tensor-Train / MPS-style adapter can replace dense graft blocks under the existing validation-routed staged protocol.

The motivation came from the 4O-lite SVD anatomy result: accepted dense graft products looked low-rank-like, so the next controlled experiment was not to truncate existing grafts blindly, but to train an explicitly structured adapter family.

## Implementation

New adapter module:

```text
saint/adapters/drm_grafting_tt_adapter.py
```

Implemented components:

```text
TTLinear
TTGraftBlock
make_tt_graft_blocks
```

The adapter uses a projected tensorized bottleneck:

```text
x -> project_down -> TT/MPS core transform -> project_up -> residual add
```

This avoids requiring the DRM `d_model` axis itself to factor conveniently.

The existing routed/staged benchmark now accepts:

```text
--adapter-type tt_mps
--tt-adapter-width 128
--tt-bond-dim {2,4,8,16}
```

and reuses the same candidate grid, stage acceptance, recomposition, and checkpoint protocol as dense graft blocks.

## Smoke Sweep Command

Executed from `/home/rato/dev/ai/SAINT-G`:

```bash
for chi in 2 4 8 16; do
  out="/home/rato/dev/ai/SAINT-G/runs/phase16_marco4o_tt_mps_adapter_seed42_chi${chi}_smoke"
  .venv/bin/python \
    scripts/benchmark_drm_g_phase16_graftblock.py \
    --output-dir "$out" \
    --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
    --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
    --device cuda \
    --seeds 42 \
    --graft-count 24 \
    --adapter-type tt_mps \
    --tt-adapter-width 128 \
    --tt-bond-dim "$chi" \
    --hidden-size 128 \
    --stage-size 4 \
    --post-first-stage-size 1 \
    --max-stages 3 \
    --stage-accept-min-gain 0.0 \
    --steps 100000000 \
    --max-train-seconds 90 \
    --eval-every-steps 100 \
    --early-stopping-patience 2 \
    --early-stopping-min-delta 0.00001 \
    --batch-size 2 \
    --seq-len 128 \
    --validation-batches 4 \
    --train-batches 512 \
    --learning-rate 0.0000003 \
    --lr-decay 0.02 \
    --training-mode validation_routed_staged \
    --candidate-targets blocks.2 blocks.3 blocks.4 \
    --candidate-learning-rates 0.0000001 0.0000003 \
    --candidate-init-scales 0.001 0.005 \
    --candidate-activations silu \
    --candidate-score-mode composed_gain_orthogonal \
    --orthogonal-penalty 0.00001 \
    --candidate-probe-steps 40 \
    --candidate-probe-max-train-seconds 30 \
    --candidate-top-k 4
 done
```

## Artifacts

```text
runs/phase16_marco4o_tt_mps_adapter_seed42_chi2_smoke/
runs/phase16_marco4o_tt_mps_adapter_seed42_chi4_smoke/
runs/phase16_marco4o_tt_mps_adapter_seed42_chi8_smoke/
runs/phase16_marco4o_tt_mps_adapter_seed42_chi16_smoke/
```

Each directory contains:

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

| chi | adapter_width | params/graft | checkpoint bytes | base_loss | composed_loss | gain | accepted_grafts | recompose_abs_diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 25,217 | 2,460,139 | 10.416174 | 10.416174 | 0.000000 | 0 | 0.0 |
| 4 | 128 | 25,857 | 2,521,579 | 10.416174 | 10.416174 | 0.000000 | 0 | 0.0 |
| 8 | 128 | 27,137 | 2,644,459 | 10.416174 | 10.416174 | 0.000000 | 0 | 0.0 |
| 16 | 128 | 29,697 | 2,890,219 | 10.416174 | 10.416174 | 0.000000 | 0 | 0.0 |

All four smoke runs rejected stage 1. The best candidate in each run had zero composed gain, so no TT/MPS graft group was accepted.

## Interpretation

Marco 4O passes the infrastructure requirement:

```text
- PyTorch-native TT/MPS adapter implemented;
- adapter integrated into the existing routed/staged benchmark;
- checkpoint recomposition remains exact for the smoke runs;
- bond-dimension sweep generated reproducible CUDA artifacts.
```

It does **not** pass the modeling success criteria in this smoke configuration:

```text
- no positive validation gain;
- no accepted grafts;
- no evidence of fifth-graft recovery;
- no evidence yet that lower checkpoint bytes compensate for loss.
```

The likely bottleneck is not checkpoint/recomposition mechanics. The smoke configuration used a small `adapter_width=128`, short candidate probes, and conservative learning rates. Compared with the dense 4N-B run, this tests the structured-adapter mechanism but not its full capacity.

## Verdict

Marco 4O is **implemented and technically valid, but negative in the first seed-42 smoke sweep**.

The current result supports this conclusion:

```text
TT/MPS adapters are now a runnable baseline family, but chi 2/4/8/16 at width 128 did not improve the 5M checkpoint under the short routed/staged protocol.
```

## Recommended Next Marco

4O-B has now been run and documented:

```text
docs/reports/phase16_marco4o_b_tt_mps_capacity_sanity.md
runs/phase16_marco4o_b_tt_mps_capacity_seed42_w{256,512}_chi{4,8,16}/
runs/phase16_marco4o_b_dense_matched_seed42_h{300,620}_match_w{256,512}/
```

4O-B was also negative: width 256/512, chi 4/8/16, and parameter-matched dense controls all accepted zero grafts and kept `composed_loss=10.416174` with exact recomposition.

Current next direction:

```text
cost-aware dense graft routing / efficiency-aware candidate scoring
```

Do not keep increasing TT/MPS capacity under the same protocol unless a future optimizer/routing change gives a new reason to revisit it.
