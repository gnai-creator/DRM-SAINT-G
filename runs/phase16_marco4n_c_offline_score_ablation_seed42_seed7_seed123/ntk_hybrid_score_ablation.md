# Phase 16 Marco 4N-C - Offline NTK-Hybrid Score Ablation

Status: completed offline from Marco 4N-B artifacts; no CUDA training performed.

## Run Summary

| seed | gain | accepted_grafts | stages | zero-gain positive current scores | route | recompose_abs_diff |
|---|---:|---:|---:|---:|---|---:|
| 42 | 0.001645803 | 6 | 4 | 4 | 0->blocks.4, 1->blocks.4, 2->blocks.4, 3->blocks.4, 4->blocks.3, 5->blocks.2 | 0.0 |
| 7 | 0.000527620 | 4 | 2 | 10 | 0->blocks.4, 1->blocks.4, 2->blocks.4, 3->blocks.4 | 0.0 |
| 123 | 0.001653910 | 4 | 2 | 9 | 0->blocks.3, 1->blocks.3, 2->blocks.3, 3->blocks.3 | 0.0 |

## Aggregate Verdict

- positive_runs: 3/3
- exact_recompose_runs: 3/3
- mean_accumulated_gain: 0.001275778
- mean_accepted_grafts: 4.667
- seeds_with_five_or_more_grafts: 42
- zero_gain_positive_current_score_total: 23

Interpretation: 4N-B is a structural routing improvement, especially for seed 42, but it did not solve multi-seed robustness. Seeds 7 and 123 still stop at four accepted grafts.

## Policy Summary

| policy | threshold | approve_count | zero_gain_winners | target_match_rate | mean_winner_gain |
|---|---:|---:|---:|---:|---:|
| composed_gain | 0.00000 | 5 | 3 | 0.875 | 0.000480771 |
| composed_gain | 0.00002 | 4 | 3 | 0.875 | 0.000480771 |
| composed_gain | 0.00005 | 4 | 3 | 0.875 | 0.000480771 |
| composed_gain_orthogonal | 0.00000 | 5 | 3 | 0.875 | 0.000480771 |
| composed_gain_orthogonal | 0.00002 | 4 | 3 | 0.875 | 0.000480771 |
| composed_gain_orthogonal | 0.00005 | 4 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_current | 0.00000 | 5 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_current | 0.00002 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_current | 0.00005 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_double_anti_saturation | 0.00000 | 5 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_double_anti_saturation | 0.00002 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_double_anti_saturation | 0.00005 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_gain_gated_bonus | 0.00000 | 5 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_gain_gated_bonus | 0.00002 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_gain_gated_bonus | 0.00005 | 4 | 3 | 1.000 | 0.000478417 |
| ntk_hybrid_half_bonus | 0.00000 | 5 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_half_bonus | 0.00002 | 4 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_half_bonus | 0.00005 | 4 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_no_bonus | 0.00000 | 5 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_no_bonus | 0.00002 | 4 | 3 | 0.875 | 0.000480771 |
| ntk_hybrid_no_bonus | 0.00005 | 4 | 3 | 0.875 | 0.000480771 |

## Recommendations

- keep_4k_as_best_loss_checkpoint_for_seed42
- treat_4n_b_as_structural_routing_improvement_not_robust_quality_win
- gate_ntk_bonus_when_candidate_composed_gain_is_zero_or_below_epsilon
- test_accept_min_gain_epsilon_2e-5_before_more_cuda
- prefer_offline_ablation_or_4o_svd_before_new_large_routing_sweep

## Stage Winners by Current and Gated Policy

| seed | stage | policy | threshold | winner | gain | score | would_approve | actual |
|---|---:|---|---:|---|---:|---:|---|---|
| 123 | 1 | composed_gain | 0.00000 | blocks.3 | 0.001653910 | 0.001653910 | true | blocks.3/approved |
| 123 | 1 | composed_gain | 0.00002 | blocks.3 | 0.001653910 | 0.001653910 | true | blocks.3/approved |
| 123 | 1 | ntk_hybrid_current | 0.00000 | blocks.3 | 0.001653910 | 0.001715478 | true | blocks.3/approved |
| 123 | 1 | ntk_hybrid_current | 0.00002 | blocks.3 | 0.001653910 | 0.001715478 | true | blocks.3/approved |
| 123 | 1 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.3 | 0.001653910 | 0.001715478 | true | blocks.3/approved |
| 123 | 1 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.3 | 0.001653910 | 0.001715478 | true | blocks.3/approved |
| 123 | 2 | composed_gain | 0.00000 | blocks.4 | 0.000000000 | 0.000000000 | false | blocks.4/rejected |
| 123 | 2 | composed_gain | 0.00002 | blocks.4 | 0.000000000 | 0.000000000 | false | blocks.4/rejected |
| 123 | 2 | ntk_hybrid_current | 0.00000 | blocks.4 | 0.000000000 | 0.000084546 | false | blocks.4/rejected |
| 123 | 2 | ntk_hybrid_current | 0.00002 | blocks.4 | 0.000000000 | 0.000084546 | false | blocks.4/rejected |
| 123 | 2 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.4 | 0.000000000 | 0.000000000 | false | blocks.4/rejected |
| 123 | 2 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.4 | 0.000000000 | 0.000000000 | false | blocks.4/rejected |
| 42 | 1 | composed_gain | 0.00000 | blocks.4 | 0.001549959 | 0.001549959 | true | blocks.4/approved |
| 42 | 1 | composed_gain | 0.00002 | blocks.4 | 0.001549959 | 0.001549959 | true | blocks.4/approved |
| 42 | 1 | ntk_hybrid_current | 0.00000 | blocks.4 | 0.001549959 | 0.001632037 | true | blocks.4/approved |
| 42 | 1 | ntk_hybrid_current | 0.00002 | blocks.4 | 0.001549959 | 0.001632037 | true | blocks.4/approved |
| 42 | 1 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.4 | 0.001549959 | 0.001632037 | true | blocks.4/approved |
| 42 | 1 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.4 | 0.001549959 | 0.001632037 | true | blocks.4/approved |
| 42 | 2 | composed_gain | 0.00000 | blocks.2 | 0.000099659 | 0.000099659 | true | blocks.3/approved |
| 42 | 2 | composed_gain | 0.00002 | blocks.2 | 0.000099659 | 0.000099659 | true | blocks.3/approved |
| 42 | 2 | ntk_hybrid_current | 0.00000 | blocks.3 | 0.000080824 | 0.000141767 | true | blocks.3/approved |
| 42 | 2 | ntk_hybrid_current | 0.00002 | blocks.3 | 0.000080824 | 0.000141767 | true | blocks.3/approved |
| 42 | 2 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.3 | 0.000080824 | 0.000141767 | true | blocks.3/approved |
| 42 | 2 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.3 | 0.000080824 | 0.000141767 | true | blocks.3/approved |
| 42 | 3 | composed_gain | 0.00000 | blocks.2 | 0.000015020 | 0.000015020 | true | blocks.2/approved |
| 42 | 3 | composed_gain | 0.00002 | blocks.2 | 0.000015020 | 0.000015020 | false | blocks.2/approved |
| 42 | 3 | ntk_hybrid_current | 0.00000 | blocks.2 | 0.000015020 | 0.000055796 | true | blocks.2/approved |
| 42 | 3 | ntk_hybrid_current | 0.00002 | blocks.2 | 0.000015020 | 0.000055796 | false | blocks.2/approved |
| 42 | 3 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.2 | 0.000015020 | 0.000055796 | true | blocks.2/approved |
| 42 | 3 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.2 | 0.000015020 | 0.000055796 | false | blocks.2/approved |
| 42 | 4 | composed_gain | 0.00000 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
| 42 | 4 | composed_gain | 0.00002 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
| 42 | 4 | ntk_hybrid_current | 0.00000 | blocks.3 | 0.000000000 | -0.000029538 | false | blocks.3/rejected |
| 42 | 4 | ntk_hybrid_current | 0.00002 | blocks.3 | 0.000000000 | -0.000029538 | false | blocks.3/rejected |
| 42 | 4 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.3 | 0.000000000 | -0.000060000 | false | blocks.3/rejected |
| 42 | 4 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.3 | 0.000000000 | -0.000060000 | false | blocks.3/rejected |
| 7 | 1 | composed_gain | 0.00000 | blocks.4 | 0.000527620 | 0.000527620 | true | blocks.4/approved |
| 7 | 1 | composed_gain | 0.00002 | blocks.4 | 0.000527620 | 0.000527620 | true | blocks.4/approved |
| 7 | 1 | ntk_hybrid_current | 0.00000 | blocks.4 | 0.000527620 | 0.000609783 | true | blocks.4/approved |
| 7 | 1 | ntk_hybrid_current | 0.00002 | blocks.4 | 0.000527620 | 0.000609783 | true | blocks.4/approved |
| 7 | 1 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.4 | 0.000527620 | 0.000609783 | true | blocks.4/approved |
| 7 | 1 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.4 | 0.000527620 | 0.000609783 | true | blocks.4/approved |
| 7 | 2 | composed_gain | 0.00000 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
| 7 | 2 | composed_gain | 0.00002 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
| 7 | 2 | ntk_hybrid_current | 0.00000 | blocks.3 | 0.000000000 | 0.000061010 | false | blocks.3/rejected |
| 7 | 2 | ntk_hybrid_current | 0.00002 | blocks.3 | 0.000000000 | 0.000061010 | false | blocks.3/rejected |
| 7 | 2 | ntk_hybrid_gain_gated_bonus | 0.00000 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
| 7 | 2 | ntk_hybrid_gain_gated_bonus | 0.00002 | blocks.3 | 0.000000000 | 0.000000000 | false | blocks.3/rejected |
