import unittest

from saint.transformer import (
    evaluate_mini_transformer_closure,
    make_mini_transformer_task,
    run_mini_transformer_benchmark,
    run_mini_transformer_sweep,
    summarize_mini_transformer_rows,
    train_mini_budgeted_delta,
    train_mini_lora_delta,
    train_mini_saint_delta,
    train_mini_saint_per_matrix_delta,
)
from saint.transformer.model import combine_weights, distillation_loss


class MiniTransformerPhase5Tests(unittest.TestCase):
    def test_task_has_coupled_transformer_matrices(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2, delta_scale=3.0)

        self.assertIn("w_q", task.base_weights)
        self.assertIn("w_mlp2", task.base_weights)
        self.assertIn("w_head", task.base_weights)

    def test_target_model_improves_over_base(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        base_loss = distillation_loss(
            task.base_weights,
            task.target_weights,
            task.test_sequences,
        )
        target_loss = distillation_loss(
            task.target_weights,
            task.target_weights,
            task.test_sequences,
        )

        self.assertGreater(base_loss, target_loss)

    def test_budgeted_delta_trains_with_global_loss(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_budgeted_delta(task, parameter_budget=12, steps=2)

        self.assertEqual(result.parameter_count, 12)
        self.assertIn("gain_per_parameter", result.metadata)

    def test_saint_adapter_reports_global_loss_metadata(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_saint_delta(task, parameter_budget=24, steps=2)

        self.assertEqual(result.name, "mini_saint_dynamic_delta")
        self.assertTrue(result.metadata["global_loss"])
        self.assertGreater(result.parameter_count, 0)

    def test_lora_trains_selected_transformer_matrices(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_lora_delta(task, rank=1, steps=2)

        self.assertEqual(result.name, "mini_lora_rank_1")
        self.assertIn("w_q", result.metadata["target_matrices"])

    def test_saint_per_matrix_uses_separate_codebooks(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        result = train_mini_saint_per_matrix_delta(task, parameter_budget=24, steps=2)

        self.assertEqual(result.name, "mini_saint_per_matrix_delta")
        self.assertEqual(result.metadata["share_scope"], "matrix")

    def test_benchmark_returns_controls_and_saint(self):
        task = make_mini_transformer_task(train_samples=4, test_samples=2)

        results = run_mini_transformer_benchmark(task, steps=2, parameter_budget=24)
        names = {result.name for result in results}

        self.assertIn("mini_full_delta", names)
        self.assertIn("mini_lora_rank_1", names)
        self.assertIn("mini_lora_rank_2", names)
        self.assertIn("mini_budgeted_delta_for_saint", names)
        self.assertIn("mini_block_budgeted_delta_for_saint", names)
        self.assertIn("mini_saint_per_matrix_delta", names)
        self.assertIn("mini_saint_dynamic_delta", names)

    def test_sweep_summarizes_methods(self):
        rows = run_mini_transformer_sweep(
            seeds=(1,),
            delta_modes=("repeated",),
            steps=2,
            parameter_budget=24,
        )

        summaries = summarize_mini_transformer_rows(rows)

        self.assertTrue(rows)
        self.assertTrue(any(row["method"] == "mini_saint_dynamic_delta" for row in summaries))

    def test_phase5_closure_returns_decision(self):
        rows = run_mini_transformer_sweep(
            seeds=(1,),
            delta_modes=("repeated",),
            steps=2,
            parameter_budget=24,
        )

        decision = evaluate_mini_transformer_closure(rows)

        self.assertIn("passed", decision)
        self.assertIn("per_matrix", decision)


if __name__ == "__main__":
    unittest.main()
