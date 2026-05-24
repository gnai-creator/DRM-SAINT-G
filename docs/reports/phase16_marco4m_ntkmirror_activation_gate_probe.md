# Phase 16 Marco 4M - NTK-Mirror-Inspired Activation Gate Probe

Status: **implemented / ready for CUDA diagnostic runs**.

## Goal

Add an NTK-Mirror-inspired diagnostic probe to SAINT-G routed staged grafting so
we can rank candidate target blocks before the expensive candidate training
passes.

The probe computes a signed activation-gate sensitivity score for each candidate
target:

```text
score(block) = sum(abs(grad_h * h))
```

where `h` is the target module activation and `grad_h` is the teacher-forced loss
gradient with respect to that activation.

This mirrors the gate-selection signal described by `ntkmirror`:

```text
dL/ds_{layer,channel} = sum_t <dL/dh_{layer,t,channel}, h_{layer,t,channel}>
```

Marco 4M is diagnostic only. It does **not** replace the current
`composed_gain_orthogonal` pruning/routing rule yet.

## Why This Matters

Marco 4L showed seed-sensitive fifth-graft behavior:

```text
seed 42: 5 accepted grafts, stage 2 approved
seed 7:  4 accepted grafts, stage 2 rejected
seed 123: 4 accepted grafts, stage 2 rejected
```

The 4M question is whether a cheap NTK-style activation score explains or
predicts this difference before deep candidate training.

## Implementation

New script flags:

```text
--ntk-activation-probe-batches N
--ntk-activation-probe-split train|val
```

When `N > 0`, `validation_routed_staged` runs a diagnostic pass at each stage
before the normal candidate probe/deep passes. The run writes:

```text
ntk_activation_probe_metrics.json
```

and embeds the stage-local rows in each `stage_metrics.json` row under:

```text
ntk_activation_probe
```

Each row contains:

```text
stage
target
ntk_activation_score
mean_ntk_activation_score
ntk_rank
probe_batches
channel_count
top_channel
top_channel_score
split
```

## Recommended Diagnostic Runs

Use one invocation per seed and a distinct output directory. These runs preserve
the existing 4K/4L training recipe and add the NTK activation probe.

### Seed 42

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 42 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

### Seed 7

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 7 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

### Seed 123

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 123 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8 \
  --ntk-activation-probe-batches 4 \
  --ntk-activation-probe-split train
```

## Analysis Command

After runs finish:

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in (42, 7, 123):
    root = Path(f'/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed{seed}')
    summary_path = root / 'summary.json'
    ntk_path = root / 'ntk_activation_probe_metrics.json'
    if not summary_path.exists() or not ntk_path.exists():
        print(f'seed {seed}: missing artifacts')
        continue
    summary = json.loads(summary_path.read_text())
    rows = json.loads(ntk_path.read_text())
    print(f"\nseed {seed}: accepted_grafts={summary['accepted_grafts']} composed_loss={summary['composed_loss']}")
    for row in rows:
        print(
            f"  stage={row['stage']} rank={row['ntk_rank']} target={row['target']} "
            f"mean_ntk={row['mean_ntk_activation_score']:.6e} "
            f"top_channel={row['top_channel']}"
        )
PY
```

## Success Criteria

Marco 4M is useful if the NTK activation ranking explains or predicts at least
one of these outcomes:

```text
- seed 42 finds a second-stage target that seeds 7/123 do not;
- the stage-2 approved target appears near the top of the NTK ranking;
- rejected stage-2 runs show flat, low, or contradictory NTK rankings;
- NTK rank correlates with best deep candidate more reliably than the current probe.
```

If 4M is useful, promote the signal into routing in Marco 4N.
