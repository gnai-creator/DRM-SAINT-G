import unittest


class NTKHybridAblationTests(unittest.TestCase):
    def test_gain_gated_bonus_removes_positive_score_from_zero_gain_candidate(self):
        from saint.adapters.drm_grafting_ntk_hybrid_ablation import rescore_candidate

        row = {
            "candidate_composed_gain": 0.0,
            "accepted_grafts_on_target_before_stage": 0,
            "redundancy_penalty": 0.0,
            "ntk_hybrid_bonus": 0.00008,
            "ntk_hybrid_penalty": 0.0,
        }

        self.assertGreater(rescore_candidate(row, "ntk_hybrid_current"), 0.0)
        self.assertEqual(rescore_candidate(row, "ntk_hybrid_gain_gated_bonus"), 0.0)

    def test_double_anti_saturation_penalizes_saturated_target_more(self):
        from saint.adapters.drm_grafting_ntk_hybrid_ablation import rescore_candidate

        row = {
            "candidate_composed_gain": 0.0001,
            "accepted_grafts_on_target_before_stage": 4,
            "redundancy_penalty": 0.00024,
            "ntk_hybrid_bonus": 0.00002,
            "ntk_hybrid_penalty": 0.0002,
        }

        current = rescore_candidate(row, "ntk_hybrid_current")
        double_anti = rescore_candidate(row, "ntk_hybrid_double_anti_saturation")

        self.assertLess(double_anti, current)

    def test_composed_gain_policy_ignores_ntk_bonus(self):
        from saint.adapters.drm_grafting_ntk_hybrid_ablation import rescore_candidate

        row = {
            "candidate_composed_gain": 0.00003,
            "accepted_grafts_on_target_before_stage": 0,
            "redundancy_penalty": 0.0,
            "ntk_hybrid_bonus": 0.00008,
        }

        self.assertEqual(rescore_candidate(row, "composed_gain"), 0.00003)
        self.assertGreater(rescore_candidate(row, "ntk_hybrid_current"), 0.00003)


if __name__ == "__main__":
    unittest.main()
