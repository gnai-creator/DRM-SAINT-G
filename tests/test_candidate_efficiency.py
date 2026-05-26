import math
import tempfile
import unittest
from pathlib import Path


class CandidateEfficiencyTests(unittest.TestCase):
    def test_efficiency_score_subtracts_redundancy_ntk_and_cost_terms(self):
        from saint.adapters.drm_grafting_candidate_efficiency import EfficiencyWeights, efficiency_score

        row = {
            "candidate_composed_gain": 0.25,
            "redundancy_penalty": 0.01,
            "ntk_hybrid_penalty": 0.02,
            "params_per_graft": 99,
            "checkpoint_bytes_delta": 999,
            "elapsed_s": 4.0,
        }
        weights = EfficiencyWeights(lambda_params=0.001, lambda_bytes=0.002, lambda_time=0.003)

        expected = 0.25 - 0.01 - 0.02 - 0.001 * math.log1p(99) - 0.002 * math.log1p(999) - 0.003 * 4.0

        self.assertAlmostEqual(efficiency_score(row, weights), expected)

    def test_rank_stage_prefers_lower_cost_when_gain_is_tied(self):
        from saint.adapters.drm_grafting_candidate_efficiency import EfficiencyWeights, rank_stage_candidates

        rows = [
            {
                "stage": 1,
                "candidate_target": "blocks.2",
                "candidate_composed_gain": 0.001,
                "redundancy_penalty": 0.0,
                "ntk_hybrid_penalty": 0.0,
                "params_per_graft": 10_000,
                "checkpoint_bytes_delta": 40_000,
                "elapsed_s": 3.0,
            },
            {
                "stage": 1,
                "candidate_target": "blocks.3",
                "candidate_composed_gain": 0.001,
                "redundancy_penalty": 0.0,
                "ntk_hybrid_penalty": 0.0,
                "params_per_graft": 1_000,
                "checkpoint_bytes_delta": 4_000,
                "elapsed_s": 1.0,
            },
        ]

        ranked = rank_stage_candidates(rows, EfficiencyWeights(lambda_params=0.0001, lambda_bytes=0.0001, lambda_time=0.0001))

        self.assertEqual(ranked[0]["candidate_target"], "blocks.3")
        self.assertGreater(ranked[0]["efficiency_score"], ranked[1]["efficiency_score"])

    def test_analyze_runs_writes_seed_and_stage_efficiency_summary(self):
        from saint.adapters.drm_grafting_candidate_efficiency import EfficiencyWeights, analyze_runs

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "phase16_marco4n_b_ntk_hybrid_topk8_probe2k_24graft_seed42"
            run.mkdir()
            (run / "summary.json").write_text(
                '{"base_loss": 10.0, "composed_loss": 9.9, "accumulated_gain": 0.1, "accepted_grafts": 4, "recompose_abs_diff": 0.0, "target_by_graft": {"0": "blocks.2"}}',
                encoding="utf-8",
            )
            (run / "stage_metrics.json").write_text(
                '[{"stage": 1, "selected_target": "blocks.2", "decision": "approved", "stage_gain": 0.1}]',
                encoding="utf-8",
            )
            (run / "candidate_metrics.json").write_text(
                '[{"stage": 1, "pass": "deep", "candidate_target": "blocks.2", "candidate_tag": "a", "candidate_composed_gain": 0.1, "candidate_score": 0.09, "redundancy_penalty": 0.0, "ntk_hybrid_penalty": 0.0, "elapsed_s": 2.0}]',
                encoding="utf-8",
            )

            rows, summary = analyze_runs([run], EfficiencyWeights(lambda_time=0.001), params_per_graft=1000, checkpoint_bytes_delta=4000)

        self.assertEqual(summary["marco"], "4p_a_offline_candidate_efficiency")
        self.assertEqual(summary["seeds"], ["42"])
        self.assertEqual(summary["run_summaries"][0]["best_efficiency_target"], "blocks.2")
        self.assertEqual(rows[0]["params_per_graft"], 1000)
        self.assertEqual(rows[0]["checkpoint_bytes_delta"], 4000)


if __name__ == "__main__":
    unittest.main()
