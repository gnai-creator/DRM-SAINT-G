# Phase 16 Marco 4K - Two-Pass Top-K 8 Probe 2K

Status: **planned / recommended next CUDA run**.

## Goal

Marco 4J validated the two-pass candidate-pruning infrastructure, but it did not
beat the best-quality Marco 4H checkpoint. The failure mode was stage 2 ranking:
4J selected `blocks.4` and rejected it, while Marco 4H previously found a useful
fifth graft in `blocks.3`.

Marco 4K keeps the two-pass router, but makes candidate pruning less aggressive:

```text
narrow target set to the historically useful region
increase probe steps from 1000 to 2000
increase probe max time from 180s to 300s
increase deep-pass top-k from 4 to 8
```

## Starting Point

Previous best quality remains Marco 4H:

```text
4H composed_loss: 10.414670705795288
4H accepted_grafts: 5
4H stage 1: blocks.2, grafts 0-3
4H stage 2: blocks.3, graft 4
4H recompose_abs_diff: 0.0
```

Marco 4J result:

```text
4J composed_loss: 10.41480803489685
4J accepted_grafts: 4
4J stage 1: blocks.2, grafts 0-3
4J stage 2: blocks.4 rejected
4J recompose_abs_diff: 0.0
```

## Hypothesis

The 4J probe was too shallow and `candidate_top_k=4` was too narrow after the
first accepted group. A better second-stage candidate, especially `blocks.3`, may
have been discarded before deep training.

Marco 4K tests whether a stronger probe and wider top-k can recover the fifth
graft while preserving the runtime savings of two-pass routing.

## Recommended Command

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft \
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
  --candidate-top-k 8
```

## Differences vs Marco 4J

```text
output-dir:
  phase16_marco4j_two_pass_24graft
  -> phase16_marco4k_two_pass_topk8_probe2k_24graft

candidate-targets:
  blocks.0 blocks.1 blocks.2 blocks.3 blocks.4 blocks.5
  -> blocks.2 blocks.3 blocks.4

candidate-probe-steps:
  1000 -> 2000

candidate-probe-max-train-seconds:
  180 -> 300

candidate-top-k:
  4 -> 8
```

## Expected Runtime

The candidate grid becomes smaller:

```text
3 targets * 3 learning rates * 3 init scales * 1 activation = 27 candidates
```

The upper bound per stage is:

```text
27 probe candidates * 300s = 2.25h
8 deep candidates * 1800s = 4.0h
total upper bound per stage = 6.25h
```

This upper bound is pessimistic. In Marco 4J, early stopping terminated deep
candidates much earlier than `max_train_seconds`.

## Success Criteria

Marco 4K passes if it achieves at least one:

```text
composed_loss <= 10.414670705795288
accepted_grafts >= 5 with recompose_abs_diff = 0.0
same or better quality than 4J while selecting blocks.3 in stage 2
```

Strong pass:

```text
composed_loss < 10.414670705795288
accepted_grafts > 5
recompose_abs_diff = 0.0
```

Failure:

```text
stage 2 still rejects after selecting a non-blocks.3 candidate
composed_loss remains worse than 4H
runtime approaches full-grid search without quality recovery
```

## Read After Run

Expected artifacts:

```text
/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft/summary.json
/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft/stage_metrics.json
/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft/candidate_metrics.json
/mnt/e/dev/ai/DRM-SAINT-G/runs/phase16_marco4k_two_pass_topk8_probe2k_24graft/results.md
```

Compare directly against:

```text
4H: 10.414670705795288, 5 grafts
4I: 10.414714097976685, 5 grafts
4J: 10.41480803489685, 4 grafts
```
