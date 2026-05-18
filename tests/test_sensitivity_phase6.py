import unittest

from saint.sensitivity import (
    evaluate_sensitivity_final,
    evaluate_sensitivity_success,
    run_sensitivity_sweep,
    score_sensitivity,
    summarize_sensitivity_by_regime,
    summarize_sensitivity_rows,
    train_mini_accumulated_sensitivity_delta,
    train_mini_block_sensitivity_delta,
    train_mini_sensitivity_delta,
)
from saint.transformer import make_mini_transformer_task


class SensitivityPhase6Tests(unittest.TestCase):
    def test_gradient_norm_scores_all_coordinates(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        scores = score_sensitivity(task, method="gradient_norm")

        total_params = sum(len(row) for matrix in task.base_weights.values() for row in matrix)
        self.assertEqual(len(scores), total_params)

    def test_unknown_sensitivity_method_fails(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        with self.assertRaises(ValueError):
            score_sensitivity(task, method="missing")

    def test_sensitivity_delta_trains_selected_budget(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_sensitivity_delta(
            task,
            method="gradient_norm",
            parameter_budget=12,
            steps=2,
        )

        self.assertEqual(result.parameter_count, 12)
        self.assertEqual(result.metadata["method"], "gradient_norm")

    def test_block_sensitivity_delta_uses_blocks(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_block_sensitivity_delta(
            task,
            method="gradient_norm",
            parameter_budget=12,
            steps=2,
        )

        self.assertEqual(result.metadata["block_size"], 2)
        self.assertLessEqual(result.parameter_count, 12)

    def test_accumulated_sensitivity_delta_trains(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_accumulated_sensitivity_delta(
            task,
            parameter_budget=12,
            warmup_steps=1,
            steps=2,
        )

        self.assertEqual(result.metadata["method"], "accumulated_gradient")

    def test_sensitivity_sweep_summarizes_methods(self):
        rows = run_sensitivity_sweep(
            seeds=(1,),
            delta_modes=("repeated",),
            methods=("random", "gradient_norm", "fisher"),
            parameter_budget=12,
            steps=2,
        )

        summaries = summarize_sensitivity_rows(rows)
        by_regime = summarize_sensitivity_by_regime(rows)
        methods = {row["method"] for row in summaries}

        self.assertIn("sensitivity_random", methods)
        self.assertIn("sensitivity_gradient_norm", methods)
        self.assertTrue(by_regime)

    def test_sensitivity_success_returns_decision(self):
        rows = run_sensitivity_sweep(
            seeds=(1,),
            delta_modes=("repeated",),
            methods=("random", "gradient_norm", "fisher", "gain_per_byte"),
            parameter_budget=12,
            steps=2,
        )

        decision = evaluate_sensitivity_success(rows)

        self.assertIn("passed", decision)
        self.assertIn("winner_count", decision)

    def test_sensitivity_final_returns_decision(self):
        rows = run_sensitivity_sweep(
            seeds=(1,),
            delta_modes=("repeated",),
            methods=("random", "gradient_norm", "fisher", "gain_per_byte"),
            parameter_budget=12,
            steps=2,
        )

        decision = evaluate_sensitivity_final(rows)

        self.assertIn("passed", decision)
        self.assertIn("sensitivity_saint_beats_default", decision)


if __name__ == "__main__":
    unittest.main()
