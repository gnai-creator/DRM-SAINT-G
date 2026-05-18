import unittest

from saint.training import (
    evaluate_phase4_success,
    evaluate_phase4_closure,
    make_linear_delta_task,
    run_linear_phase4_benchmark,
    run_linear_phase4_regime_sweep,
    run_linear_phase4_sweep,
    summarize_phase4_rows,
    train_budgeted_full_delta,
    train_block_budgeted_delta,
    train_full_delta,
    train_saint_dynamic_delta,
    train_saint_global_scaled_residual,
    train_saint_routed_delta,
)


class LinearTrainingTests(unittest.TestCase):
    def test_full_delta_learns_linear_task(self):
        task = make_linear_delta_task(rows=6, cols=6, train_samples=48, test_samples=16)

        result = train_full_delta(task, steps=160, learning_rate=0.35)

        self.assertEqual(result.name, "full_delta")
        self.assertLess(result.test_loss, 0.001)
        self.assertEqual(result.parameter_count, 36)

    def test_dense_delta_task_is_supported(self):
        task = make_linear_delta_task(
            rows=6,
            cols=6,
            train_samples=32,
            test_samples=12,
            delta_mode="dense",
        )

        result = train_full_delta(task, steps=80, learning_rate=0.35)

        self.assertLess(result.test_loss, 0.001)

    def test_saint_routed_delta_uses_fewer_parameters_than_full_delta(self):
        task = make_linear_delta_task(rows=8, cols=8, train_samples=64, test_samples=16)

        result = train_saint_routed_delta(task, steps=120)

        self.assertEqual(result.name, "saint_routed_delta")
        self.assertLess(result.parameter_count, 64)
        self.assertIn("frozen_regions", result.metadata)

    def test_budgeted_full_delta_respects_parameter_budget(self):
        task = make_linear_delta_task(rows=8, cols=8, train_samples=32, test_samples=12)

        result = train_budgeted_full_delta(task, parameter_budget=20, steps=80)

        self.assertEqual(result.parameter_count, 20)
        self.assertEqual(result.metadata["parameter_budget"], 20)

    def test_block_budgeted_delta_respects_block_budget(self):
        task = make_linear_delta_task(rows=8, cols=8, train_samples=32, test_samples=12)

        result = train_block_budgeted_delta(task, parameter_budget=20, steps=30)

        self.assertLessEqual(result.parameter_count, 20)
        self.assertEqual(result.metadata["block_size"], 2)

    def test_saint_dynamic_delta_uses_advanced_router(self):
        task = make_linear_delta_task(rows=8, cols=8, train_samples=32, test_samples=12)

        result = train_saint_dynamic_delta(task, parameter_budget=44, steps=30)

        self.assertEqual(result.metadata["residual_selection"], "real_marginal_gain_per_parameter")
        self.assertEqual(result.metadata["budget_allocator"], "dynamic_greedy")
        self.assertTrue(result.metadata["has_accumulated_sensitivity"])
        self.assertTrue(result.metadata["has_block_bias"])

    def test_saint_global_scaled_residual_uses_phase4_improvements(self):
        task = make_linear_delta_task(rows=8, cols=8, train_samples=32, test_samples=12)

        result = train_saint_global_scaled_residual(task, steps=30)

        self.assertEqual(result.metadata["scale_initialization"], "least_squares")
        self.assertTrue(result.metadata["residual_selected_after_warmup"])
        self.assertEqual(result.metadata["codebook_assignment"], "kmeans_gradient_signature")
        self.assertEqual(result.metadata["router"], "gain_per_parameter")

    def test_phase4_benchmark_returns_named_results(self):
        task = make_linear_delta_task(rows=6, cols=6, train_samples=48, test_samples=16)

        results = run_linear_phase4_benchmark(task)

        names = {result.name for result in results}
        self.assertIn("full_delta", names)
        self.assertIn("saint_routed_delta", names)
        self.assertIn("lora_rank_2", names)

    def test_phase4_sweep_summarizes_multiple_seeds(self):
        rows = run_linear_phase4_sweep(
            seeds=(1, 2),
            rows=6,
            cols=6,
            train_samples=32,
            test_samples=12,
        )

        summaries = summarize_phase4_rows(rows)
        methods = {summary["method"] for summary in summaries}

        self.assertIn("lora_rank_1", methods)
        self.assertIn("lora_rank_4", methods)
        self.assertIn("saint_routed_f25_c50", methods)
        self.assertIn("saint_global_capped", methods)
        self.assertIn("saint_global_scaled_residual", methods)
        self.assertIn("saint_dynamic_delta", methods)
        self.assertIn("lora_tuned_rank_2", methods)
        self.assertTrue(all("gain_per_parameter" in row for row in rows))

    def test_phase4_regime_sweep_includes_sizes_and_delta_modes(self):
        rows = run_linear_phase4_regime_sweep(
            seeds=(1,),
            sizes=(6,),
            delta_modes=("repeated", "dense"),
            train_samples=24,
            test_samples=8,
            steps=40,
            lora_steps=50,
        )

        modes = {row["delta_mode"] for row in rows}
        sizes = {row["rows"] for row in rows}

        self.assertEqual(modes, {"repeated", "dense"})
        self.assertEqual(sizes, {6})

    def test_phase4_criteria_returns_decision(self):
        rows = run_linear_phase4_sweep(
            seeds=(1,),
            rows=6,
            cols=6,
            train_samples=32,
            test_samples=12,
        )
        summaries = summarize_phase4_rows(rows)

        decision = evaluate_phase4_success(
            summaries,
            saint_method="saint_routed_f25_c50",
            compared_method="lora_rank_2",
        )

        self.assertEqual(decision.saint_method, "saint_routed_f25_c50")
        self.assertEqual(decision.compared_method, "lora_rank_2")
        self.assertIsInstance(decision.passed, bool)
        self.assertTrue(decision.reason)

    def test_phase4_closure_returns_decision(self):
        rows = run_linear_phase4_regime_sweep(
            seeds=(1,),
            sizes=(6,),
            delta_modes=("repeated",),
            train_samples=24,
            test_samples=8,
            steps=30,
            lora_steps=40,
        )

        decision = evaluate_phase4_closure(rows)

        self.assertEqual(decision["saint_method"], "saint_dynamic_delta")
        self.assertIn("budgeted_full_delta_passes", decision)


if __name__ == "__main__":
    unittest.main()
