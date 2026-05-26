import unittest
from types import SimpleNamespace


class NTKHybridRoutingTests(unittest.TestCase):
    def test_marco_name_identifies_4p_b_cost_aware_routing(self):
        from saint.adapters.drm_grafting_graftblock_routed_utils import marco_name

        args = SimpleNamespace(
            adapter_type="dense_graftblock",
            ntk_activation_probe_batches=4,
            candidate_top_k=8,
            candidate_score_mode="composed_gain_cost_aware",
            marco_label="",
        )

        self.assertEqual(marco_name(args), "4p_b_cost_aware_dense_routing")

    def test_marco_name_identifies_4n_b_hybrid_routing(self):
        from saint.adapters.drm_grafting_graftblock_routed_utils import marco_name

        args = SimpleNamespace(
            ntk_activation_probe_batches=4,
            candidate_top_k=8,
            candidate_score_mode="composed_gain_ntk_hybrid_conservative",
        )

        self.assertEqual(marco_name(args), "4n_b_ntk_hybrid_conservative_routing")

    def test_ntk_hybrid_score_penalizes_saturated_targets(self):
        from saint.adapters.drm_grafting_graftblock_routed_utils import candidate_score

        args = SimpleNamespace(
            candidate_score_mode="composed_gain_ntk_hybrid_conservative",
            orthogonal_penalty=0.0,
            ntk_hybrid_saturation_weight=0.0001,
            ntk_hybrid_residual_delta_weight=0.0,
            ntk_hybrid_anti_saturation_penalty=0.00005,
        )
        accepted_target_map = {0: "blocks.4", 1: "blocks.4", 2: "blocks.4", 3: "blocks.4"}

        saturated_score, saturated_penalty, saturated_details = candidate_score(
            args,
            "blocks.4",
            0.0001,
            accepted_target_map,
            ntk_features={"ntk_activation_score": 4.0, "ntk_delta_abs": 0.0},
        )
        fresh_score, fresh_penalty, fresh_details = candidate_score(
            args,
            "blocks.2",
            0.0001,
            accepted_target_map,
            ntk_features={"ntk_activation_score": 2.0, "ntk_delta_abs": 0.0},
        )

        self.assertLess(saturated_score, fresh_score)
        self.assertGreater(saturated_penalty, fresh_penalty)
        self.assertLess(saturated_details["saturation_adjusted_ntk"], fresh_details["saturation_adjusted_ntk"])

    def test_cost_aware_score_requires_positive_gain_and_positive_efficiency(self):
        from saint.adapters.drm_grafting_graftblock_routed_utils import candidate_rank, candidate_score

        args = SimpleNamespace(
            candidate_score_mode="composed_gain_cost_aware",
            orthogonal_penalty=0.01,
            cost_aware_lambda_params=0.001,
            cost_aware_lambda_bytes=0.002,
            cost_aware_lambda_time=0.003,
            cost_aware_lambda_ntk_risk=1.0,
            cost_aware_params_per_graft=99,
            cost_aware_checkpoint_bytes_delta=999,
        )

        score, penalty, details = candidate_score(
            args,
            "blocks.4",
            0.25,
            {0: "blocks.4"},
            ntk_features={"ntk_hybrid_penalty": 0.02},
            elapsed_s=4.0,
        )

        self.assertAlmostEqual(
            score,
            0.25 - 0.01 - 0.02 - 0.001 * __import__("math").log1p(99) - 0.002 * __import__("math").log1p(999) - 0.003 * 4.0,
        )
        self.assertAlmostEqual(penalty, 0.01 + 0.02)
        self.assertTrue(details["cost_aware_accept"])
        rejected = dict(details, candidate_composed_gain=0.25, candidate_score=-0.001)
        self.assertEqual(candidate_rank(rejected, 0.0)[0], 0)

    def test_select_stage_candidates_preserves_rank_three_target_representative(self):
        from saint.adapters.drm_grafting_graftblock_routed_utils import select_stage_candidates

        args = SimpleNamespace(
            candidate_top_k=2,
            ntk_hybrid_keep_ranks=3,
            candidate_score_mode="composed_gain_ntk_hybrid_conservative",
        )
        probe_rows = [
            ({"candidate_target": "blocks.4", "candidate_composed_gain": 0.0010, "candidate_score": 0.0010}, {"target": "blocks.4", "tag": "b4-a"}),
            ({"candidate_target": "blocks.4", "candidate_composed_gain": 0.0009, "candidate_score": 0.0009}, {"target": "blocks.4", "tag": "b4-b"}),
            ({"candidate_target": "blocks.3", "candidate_composed_gain": 0.0008, "candidate_score": 0.0008}, {"target": "blocks.3", "tag": "b3"}),
            ({"candidate_target": "blocks.2", "candidate_composed_gain": 0.0001, "candidate_score": 0.0001}, {"target": "blocks.2", "tag": "b2"}),
        ]
        ntk_rows = [
            {"target": "blocks.4", "ntk_rank": 1},
            {"target": "blocks.3", "ntk_rank": 2},
            {"target": "blocks.2", "ntk_rank": 3},
        ]

        selected = select_stage_candidates(probe_rows, args, min_gain=0.0, ntk_rows=ntk_rows)
        selected_targets = [row["target"] for row in selected]

        self.assertIn("blocks.2", selected_targets)
        self.assertIn("blocks.3", selected_targets)
        self.assertEqual(selected_targets[:2], ["blocks.4", "blocks.4"])


if __name__ == "__main__":
    unittest.main()
