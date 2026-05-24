# Phase 16 Marco 4L - 4K Multi-Seed Robustness

Status: **planned / recommended next CUDA runs**.

## Goal

Marco 4K produced the best grafted Phase 16 checkpoint so far on seed 42:

```text
4K seed 42 composed_loss: 10.414523839950562
4K accepted_grafts: 5
4K route: grafts 0-3 -> blocks.4, graft 4 -> blocks.2
4K recompose_abs_diff: 0.0
```

Marco 4L verifies whether that gain is robust or a seed-specific win before the
project promotes the 4K recipe into the full-vs-grafted comparison.

## Baselines To Beat

```text
4H: 10.414670705795288, 5 grafts
4I: 10.414714097976685, 5 grafts
4J: 10.414808034896850, 4 grafts
4K seed 42: 10.414523839950562, 5 grafts
```

Primary baseline for robustness is still 4H because it was the previous best
pre-4K grafted checkpoint with 5 accepted grafts.

## Important Runner Note

For `--training-mode validation_routed_staged`, the current script initializes
metadata from `args.seeds[0]` and then calls `run_validation_routed_staged(...)`
once. Therefore, do **not** pass multiple seeds in one invocation for Marco 4L.
Run one command per seed with a distinct `--output-dir`.

## Recommended Seeds

Use two additional seeds first:

```text
7
123
```

Together with the completed 4K seed 42, this gives a 3-seed robustness check:

```text
42, 7, 123
```

## Command - Seed 7

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed7 \
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
  --candidate-top-k 8
```

## Command - Seed 123

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed123 \
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
  --candidate-top-k 8
```

## Optional Sequential Runner

If the machine can be left unattended, run both new seeds sequentially from the
repo root:

```bash
cd /home/rato/dev/ai/SAINT-G

for SEED in 7 123; do
  python \
    scripts/benchmark_drm_g_phase16_graftblock.py \
    --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed${SEED} \
    --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
    --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
    --device cuda \
    --seeds ${SEED} \
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
    --candidate-top-k 8

done
```

Expected runtime is similar to Marco 4K per seed: several hours if the run stops
after stage 2/3, and potentially overnight if more stages are approved.

## Live Monitoring

```bash
ps -eo pid,etimes,cmd | grep -E 'benchmark_drm_g_phase16|phase16_marco4l' | grep -v grep || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

Inspect candidate progress for the active seed:

```bash
tail -n 20 /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed7/candidate_training_metrics.jsonl
```

Note: during probe passes, `candidate_training_metrics.jsonl` may not update
because `candidate_probe_steps=2000` is below `eval_every_steps=5000`. That does
not imply the run is stuck; check the process and GPU utilization together.

## Expected Artifacts Per Seed

```text
summary.json
stage_metrics.json
candidate_metrics.json
results.md
candidate_training_metrics.jsonl
composed_graft_checkpoint.pt
```

## Summary Command After Both Seeds Finish

Run this from anywhere after seed 7 and seed 123 complete:

```bash
python - <<'PY'
import json
from pathlib import Path

runs = {
    42: Path('/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft/summary.json'),
    7: Path('/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed7/summary.json'),
    123: Path('/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed123/summary.json'),
}

rows = []
for seed, path in runs.items():
    if not path.exists():
        print(f'seed {seed}: missing {path}')
        continue
    data = json.loads(path.read_text())
    rows.append(data['composed_loss'])
    route = ', '.join(f'{k}->{v}' for k, v in sorted(data['target_by_graft'].items(), key=lambda kv: int(kv[0])))
    print(
        f"seed {seed}: composed_loss={data['composed_loss']:.15f} "
        f"accepted_grafts={data['accepted_grafts']} "
        f"recompose_abs_diff={data['recompose_abs_diff']} "
        f"route={route}"
    )

if rows:
    mean = sum(rows) / len(rows)
    var = sum((x - mean) ** 2 for x in rows) / len(rows)
    print(f'mean_composed_loss={mean:.15f}')
    print(f'std_composed_loss={var ** 0.5:.15f}')
    print(f'beats_4h_mean={mean <= 10.414670705795288}')
PY
```

## Success Criteria

Marco 4L passes if either condition holds:

```text
mean composed_loss across available seeds <= 10.414670705795288
```

or:

```text
all completed seeds have accepted_grafts >= 5 and recompose_abs_diff = 0.0
```

Strong pass:

```text
all completed seeds beat 4H individually
and all completed seeds have accepted_grafts >= 5
and all completed seeds have recompose_abs_diff = 0.0
```

Failure:

```text
seed 42 is the only seed that beats 4H
or additional seeds reject before 5 accepted grafts
or any checkpoint fails exact recomposition
```

## Implementation Note For Later

The generated `summary.json` may continue to report `marco` as
`4j_two_pass_candidate_pruning` because `_marco_name()` currently labels any run
with `candidate_top_k > 0` as 4J. For Marco 4L, identify runs by their output
directories and this report until the helper is fixed.
