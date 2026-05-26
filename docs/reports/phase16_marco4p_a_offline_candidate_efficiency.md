# Phase 16 Marco 4P-A - Offline Candidate Efficiency Reranking

Status: **implemented and executed offline on existing 4N-B seed 42/7/123 artifacts**.

## Objective

Marco 4P-A starts the cost-aware dense graft routing line after the negative 4O/4O-B TT/MPS results.

The question is:

```text
If we re-rank already-probed dense 4N-B candidates by gain minus redundancy, NTK risk, parameter cost, checkpoint-byte cost, and probe-time cost, do we preserve the useful 4N-B stage winners?
```

No CUDA training is performed in 4P-A. The run uses completed 4N-B candidate/stage artifacts.

## Implementation

New analysis module:

```text
saint/adapters/drm_grafting_candidate_efficiency.py
```

New CLI:

```text
scripts/analyze_phase16_candidate_efficiency.py
```

TDD coverage:

```text
tests/test_candidate_efficiency.py
```

## Score

```text
efficiency_score = candidate_composed_gain
  - redundancy_penalty
  - lambda_ntk_risk * ntk_hybrid_penalty
  - lambda_params * log1p(params_per_graft)
  - lambda_bytes  * log1p(checkpoint_bytes_delta)
  - lambda_time   * probe_seconds
```

For the first completed 4P-A report, the dense graft cost is inferred from the 4N-B command shape:

```text
d_model: 96
hidden_size: 25889
params_per_graft: 4,970,689
checkpoint_bytes_delta: 19,882,756
```

Weights used:

```text
lambda_params: 0.000001
lambda_bytes:  0.0000005
lambda_time:   0.0000001
lambda_ntk_risk: 1.0
```

These weights are intentionally conservative. Parameter and byte terms are constant across equal-size dense candidates in this run, so they document cost but do not drive within-stage ranking. Time and NTK-risk terms can change the order.

## Command

Executed from `/home/rato/dev/ai/SAINT-G`:

```bash
.venv/bin/python scripts/analyze_phase16_candidate_efficiency.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4p_a_offline_candidate_efficiency_seed42_seed7_seed123 \
  --lambda-params 0.000001 \
  --lambda-bytes 0.0000005 \
  --lambda-time 0.0000001 \
  --lambda-ntk-risk 1.0
```

## Artifacts

```text
runs/phase16_marco4p_a_offline_candidate_efficiency_seed42_seed7_seed123/candidate_efficiency.md
runs/phase16_marco4p_a_offline_candidate_efficiency_seed42_seed7_seed123/candidate_efficiency_summary.json
runs/phase16_marco4p_a_offline_candidate_efficiency_seed42_seed7_seed123/candidate_efficiency_rows.json
runs/phase16_marco4p_a_offline_candidate_efficiency_seed42_seed7_seed123/candidate_efficiency_table.csv
```

## Results

| seed | 4N-B accumulated_gain | accepted_grafts | stages | efficiency_match_rate | best_efficiency_target | best_efficiency_score | recompose_abs_diff |
|---|---:|---:|---:|---:|---|---:|---:|
| 42 | 0.001645803 | 6 | 4 | 0.750 | blocks.4 | 0.001463747 | 0.0 |
| 7 | 0.000527620 | 4 | 2 | 1.000 | blocks.4 | 0.000424684 | 0.0 |
| 123 | 0.001653910 | 4 | 2 | 1.000 | blocks.3 | 0.001562765 | 0.0 |

Aggregate:

```text
exact_recompose_runs: 3/3
mean_accumulated_gain: 0.001275778
mean_accepted_grafts: 4.667
stage_decisions: 8
positive_efficiency_winner_count: 3
efficiency_actual_target_match_rate: 0.875
mean_winner_efficiency_score: 0.000380379
```

Stage winners by efficiency:

| seed | stage | winner | gain | redundancy | ntk_risk | probe_s | efficiency_score | actual |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 123 | 1 | blocks.3 | 0.001653910 | 0.000000000 | 0.000000000 | 673.232 | 0.001562765 | blocks.3/approved |
| 123 | 2 | blocks.4 | 0.000000000 | 0.000000000 | 0.000000000 | 427.890 | -0.000066611 | blocks.4/rejected |
| 42 | 1 | blocks.4 | 0.001546860 | 0.000000000 | 0.000000000 | 592.912 | 0.001463747 | blocks.4/approved |
| 42 | 2 | blocks.2 | 0.000099659 | 0.000000000 | 0.000000000 | 800.612 | -0.000004224 | blocks.3/approved |
| 42 | 3 | blocks.2 | 0.000000000 | 0.000000000 | 0.000000000 | 565.652 | -0.000080387 | blocks.2/approved |
| 42 | 4 | blocks.3 | 0.000000000 | 0.000060000 | 0.000050000 | 502.806 | -0.000184102 | blocks.3/rejected |
| 7 | 1 | blocks.4 | 0.000527620 | 0.000000000 | 0.000000000 | 791.149 | 0.000424684 | blocks.4/approved |
| 7 | 2 | blocks.3 | 0.000000000 | 0.000000000 | 0.000000000 | 490.140 | -0.000072836 | blocks.3/rejected |

## Interpretation

4P-A is a useful offline gate and mostly preserves the real 4N-B routing choices:

```text
7 of 8 stage winners match the actual 4N-B selected target under the conservative efficiency score.
```

The one mismatch is seed 42 stage 2:

```text
actual 4N-B: blocks.3/approved
efficiency rerank: blocks.2 with small gain but negative final efficiency_score after cost terms
```

Only 3 of 8 stage winners remain positive after subtracting the fixed dense-graft cost and probe-time cost. This suggests a CUDA 4P-B run should not blindly accept every efficiency winner. It should use cost-aware ranking plus a positive efficiency/gain gate.

## Verdict

Marco 4P-A is **implemented, reproducible, and useful as an offline filter**.

Current recommendation:

```text
Proceed to Marco 4P-B: short CUDA dense GraftBlock run with a real cost-aware candidate-score-mode, but keep the acceptance gate conservative.
```

Suggested 4P-B policy:

```text
candidate_score_mode: composed_gain_cost_aware
accept only if candidate_composed_gain > 0 and efficiency_score > 0
preserve the 4N-B conservative NTK penalties
compare directly against 4N-B seed 42 first, then replicate seed 7/123 only if seed 42 preserves useful gains
```
