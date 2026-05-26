# Phase 16 Marco 4N-B / 4N-C Results

Status: **completed for seeds 42, 7, and 123**.

This report records the completed 4N-B conservative NTK-hybrid routing run and
the 4N-C offline score ablation. The goal was to test whether NTK-derived
saturation/residual features can recover useful marginal grafts beyond the first
accepted group without using raw NTK as a direct router.

## 4N-B Training Command

Run from the canonical SAINT-G repo:

```bash
cd /home/rato/dev/ai/SAINT-G

for seed in 42 7 123; do
  python \
    scripts/benchmark_drm_g_phase16_graftblock.py \
    --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed${seed} \
    --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
    --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
    --device cuda \
    --seeds ${seed} \
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
    --candidate-score-mode composed_gain_ntk_hybrid_conservative \
    --orthogonal-penalty 0.00001 \
    --candidate-probe-steps 2000 \
    --candidate-probe-max-train-seconds 300 \
    --candidate-top-k 8 \
    --ntk-activation-probe-batches 4 \
    --ntk-activation-probe-split train \
    --ntk-hybrid-saturation-weight 0.00002 \
    --ntk-hybrid-residual-delta-weight 0.00001 \
    --ntk-hybrid-anti-saturation-penalty 0.00005 \
    --ntk-hybrid-keep-ranks 3
done
```

## 4N-B Result Summary

| seed | base_loss | composed_loss | gain | accepted_grafts | route | recompose_abs_diff |
|---|---:|---:|---:|---:|---|---:|
| 42 | 10.4161744118 | 10.4145286083 | 0.0016458035 | 6 | 0-3 -> blocks.4, 4 -> blocks.3, 5 -> blocks.2 | 0.0 |
| 7 | 10.3868415356 | 10.3863139153 | 0.0005276203 | 4 | 0-3 -> blocks.4 | 0.0 |
| 123 | 10.4170150757 | 10.4153611660 | 0.0016539097 | 4 | 0-3 -> blocks.3 | 0.0 |

Aggregate:

```text
positive_runs: 3/3
exact_recomposition: 3/3
mean_accumulated_gain: 0.0012757778
mean_accepted_grafts: 4.6667
seeds_with_five_or_more_grafts: 1/3 (seed 42)
```

## 4N-B Interpretation

4N-B passed technically and confirmed a useful structural hypothesis, but did
not solve the multi-seed robustness problem.

Seed 42 is the important positive case:

```text
stage 1: blocks.4 approved, gain 0.0015499592, grafts 0-3
stage 2: blocks.3 approved, gain 0.0000808239, graft 4
stage 3: blocks.2 approved, gain 0.0000150204, graft 5
stage 4: blocks.3 rejected, gain 0.0
```

This shows that the conservative NTK-hybrid score can diversify routing beyond
`blocks.4` and accept a sixth graft with exact recomposition. However, the sixth
graft contributes only about 0.91% of the seed-42 total gain, so it should be
treated as fragile unless reproduced elsewhere.

Seeds 7 and 123 do not show the same extension:

```text
seed 7:   first group approved, second stage rejected, 4 grafts total
seed 123: first group approved, second stage rejected, 4 grafts total
```

Therefore 4N-B should be documented as a structural routing improvement, not as
a robust quality win.

Against the prior best seed-42 loss from Marco 4K:

```text
4K seed 42 composed_loss:   10.414523839950562
4N-B seed 42 composed_loss: 10.414528608322144
delta: +0.000004768371582 worse than 4K
```

So the best seed-42 checkpoint by raw composed loss remains 4K, while 4N-B is
the best seed-42 evidence for diversified growth.

## 4N-C Offline Score Ablation

4N-C is an offline analysis over the completed 4N-B artifacts. It does not train
or mutate checkpoints. It re-scores recorded deep candidates with alternate
policies and acceptance epsilons.

Implemented files:

```text
saint/adapters/drm_grafting_ntk_hybrid_ablation.py
scripts/analyze_phase16_ntk_hybrid_score_ablation.py
tests/test_ntk_hybrid_ablation.py
```

Generated artifacts:

```text
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123/ntk_hybrid_score_ablation_rows.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123/ntk_hybrid_score_ablation_summary.json
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123/ntk_hybrid_score_ablation_table.csv
/home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123/ntk_hybrid_score_ablation.md
```

### 4N-C Command

Run from the canonical SAINT-G repo:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_hybrid_score_ablation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123
```

Expected terminal summary:

```text
wrote 168 ablation rows to /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_c_offline_score_ablation_seed42_seed7_seed123
mean_gain=0.001275778 mean_grafts=4.667 seeds_ge5=42
recommendations=keep_4k_as_best_loss_checkpoint_for_seed42,treat_4n_b_as_structural_routing_improvement_not_robust_quality_win,gate_ntk_bonus_when_candidate_composed_gain_is_zero_or_below_epsilon,test_accept_min_gain_epsilon_2e-5_before_more_cuda,prefer_offline_ablation_or_4o_svd_before_new_large_routing_sweep
```

## 4N-C Findings

Key aggregate result:

```text
positive_runs: 3/3
exact_recompose_runs: 3/3
mean_accumulated_gain: 0.001275778
mean_accepted_grafts: 4.667
seeds_with_five_or_more_grafts: 42
zero_gain_positive_current_score_total: 23
```

The most important finding is that the current NTK-hybrid score can assign
positive score to zero-gain candidates:

```text
zero_gain_positive_current_score_total: 23
seed 42: 4
seed 7: 10
seed 123: 9
```

The final accept/reject gate protected the checkpoint because approval still
requires positive composed gain. But the ranking signal is optimistic: NTK bonus
can make a zero-gain candidate look attractive in the offline score. This is why
4N-C recommends gating the NTK bonus unless `candidate_composed_gain` is positive
or above a small epsilon.

Acceptance threshold ablation:

```text
threshold 0.0:     5 approved stage decisions in offline replay
threshold 0.00002: 4 approved stage decisions
threshold 0.00005: 4 approved stage decisions
```

The `0.00002` epsilon would reject seed-42 stage 3:

```text
seed 42 stage 3 gain: 0.0000150204
```

That is scientifically useful: the sixth graft is positive but below a modest
robustness threshold. If the goal is robust growth, do not count that sixth graft
as strong evidence yet.

## 4N-C Recommendation

Do not launch another large routing sweep immediately. The recommended next
routing rule, if continuing 4N, is:

```text
candidate_score = candidate_composed_gain
                - penalties
                + gated_ntk_bonus

where gated_ntk_bonus = ntk_bonus only if candidate_composed_gain > epsilon
```

Use one of these epsilons depending on intent:

```text
epsilon = 0.0      preserves current 4N-B behavior but allows tiny gains
epsilon = 0.00002 rejects the fragile seed-42 sixth graft
epsilon = 0.00005 is stricter and still keeps the main first-group wins
```

Recommended next marco:

```text
4O-lite SVD Anatomy
```

Rationale: before spending more CUDA on routing, inspect whether accepted grafts
use real capacity or are compressible/low-rank. That analysis is likely more
informative than another score tweak at this point.
