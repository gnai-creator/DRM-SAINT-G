# Phase 16 Marco 4P-A - Offline Candidate Efficiency Reranking

Status: completed offline from Marco 4N-B artifacts; no CUDA training performed.

## Score

```text
efficiency_score = candidate_composed_gain
  - redundancy_penalty
  - lambda_ntk_risk * ntk_hybrid_penalty
  - lambda_params * log1p(params_per_graft)
  - lambda_bytes  * log1p(checkpoint_bytes_delta)
  - lambda_time   * probe_seconds
```

## Configuration

- params_per_graft: 4970689
- checkpoint_bytes_delta: 19882756
- weights: `{'lambda_params': 1e-06, 'lambda_bytes': 5e-07, 'lambda_time': 1e-07, 'lambda_ntk_risk': 1.0}`

## Run Summary

| seed | gain | accepted_grafts | stages | efficiency_match_rate | best_efficiency_target | best_efficiency_score | recompose_abs_diff |
|---|---:|---:|---:|---:|---|---:|---:|
| 42 | 0.001645803 | 6 | 4 | 0.750 | blocks.4 | 0.001463747 | 0.0 |
| 7 | 0.000527620 | 4 | 2 | 1.000 | blocks.4 | 0.000424684 | 0.0 |
| 123 | 0.001653910 | 4 | 2 | 1.000 | blocks.3 | 0.001562765 | 0.0 |

## Aggregate Verdict

- exact_recompose_runs: 3/3
- mean_accumulated_gain: 0.001275778
- mean_accepted_grafts: 4.667
- stage_decisions: 8
- positive_efficiency_winner_count: 3
- efficiency_actual_target_match_rate: 0.875
- mean_winner_efficiency_score: 0.000380379
- mean_winner_gain_per_million_params: 0.000096266

## Stage Winners by Efficiency Score

| seed | stage | winner | gain | redundancy | ntk_risk | probe_s | gain/M params | efficiency_score | actual |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 123 | 1 | blocks.3 | 0.001653910 | 0.000000000 | 0.000000000 | 673.232 | 0.000332732 | 0.001562765 | blocks.3/approved |
| 123 | 2 | blocks.4 | 0.000000000 | 0.000000000 | 0.000000000 | 427.890 | 0.000000000 | -0.000066611 | blocks.4/rejected |
| 42 | 1 | blocks.4 | 0.001546860 | 0.000000000 | 0.000000000 | 592.912 | 0.000311196 | 0.001463747 | blocks.4/approved |
| 42 | 2 | blocks.2 | 0.000099659 | 0.000000000 | 0.000000000 | 800.612 | 0.000020049 | -0.000004224 | blocks.3/approved |
| 42 | 3 | blocks.2 | 0.000000000 | 0.000000000 | 0.000000000 | 565.652 | 0.000000000 | -0.000080387 | blocks.2/approved |
| 42 | 4 | blocks.3 | 0.000000000 | 0.000060000 | 0.000050000 | 502.806 | 0.000000000 | -0.000184102 | blocks.3/rejected |
| 7 | 1 | blocks.4 | 0.000527620 | 0.000000000 | 0.000000000 | 791.149 | 0.000106146 | 0.000424684 | blocks.4/approved |
| 7 | 2 | blocks.3 | 0.000000000 | 0.000000000 | 0.000000000 | 490.140 | 0.000000000 | -0.000072836 | blocks.3/rejected |

## Recommendations

- use_4p_a_as_offline_gate_before_cuda_score_changes
- tune_lambda_params_bytes_time_against_4n_b_stage_winner_preservation
- prefer_cost_aware_score_only_if_it_preserves_positive_4n_b_winners
- next_cuda_step_4p_b_short_dense_cost_aware_routing_if_offline_match_rate_is_acceptable
