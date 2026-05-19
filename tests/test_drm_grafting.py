from pathlib import Path
import unittest

from saint.adapters.drm_grafting import _read_simple_yaml
from saint.adapters.drm_grafting_decision import evaluate_graft_decision
from saint.config import RuntimeConfig


class DRMGraftingTests(unittest.TestCase):
    def test_reads_baseline_3_5m_config(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "drm_transformer" / "configs" / "baselines" / "small_3.5M.yaml"
        if not config_path.exists():
            self.skipTest("external drm_transformer baseline config not available")

        data = _read_simple_yaml(config_path)

        self.assertEqual(data["vocab_size"], 50000)
        self.assertEqual(data["d_model"], 64)
        self.assertEqual(data["n_layers"], 4)
        self.assertEqual(data["d_ff"], 256)

    def test_inspects_drm_graft_baseline_when_available(self):
        try:
            from saint.adapters.drm_grafting import inspect_graft_model
        except RuntimeError as exc:
            self.skipTest(str(exc))

        root = Path(__file__).resolve().parents[2]
        drm_root = root / "drm_transformer"
        if not drm_root.exists():
            self.skipTest("external drm_transformer repo not available")

        config = RuntimeConfig(
            task="drm_transformer",
            method="drm_g_saint_phi_graft",
            metadata={"drm_root": str(drm_root), "phi_rank": 8},
        )

        summary = inspect_graft_model(config)

        self.assertEqual(summary["adapter"], "drm_grafting")
        self.assertEqual(summary["d_model"], 64)
        self.assertEqual(summary["n_layers"], 4)
        self.assertEqual(summary["graft_parameters"], 64)
        self.assertGreater(summary["base_parameters"], 3_000_000)

    def test_graft_decision_rejects_when_dense_wins(self):
        decision = evaluate_graft_decision(
            {
                "validation_gain": 0.1,
                "validation_gain_per_parameter": 0.01,
                "dense_budget_gain": 0.2,
                "require_beats_dense": True,
            }
        )

        self.assertFalse(decision["approved"])
        self.assertEqual(decision["decision"], "reject")

    def test_graft_decision_approves_positive_gain(self):
        decision = evaluate_graft_decision(
            {
                "validation_gain": 0.1,
                "validation_gain_per_parameter": 0.01,
                "dense_budget_gain": 0.05,
                "require_beats_dense": True,
            }
        )

        self.assertTrue(decision["approved"])
        self.assertEqual(decision["decision"], "approve")

    def test_real_token_batch_loads_baseline_fixture_when_available(self):
        try:
            import torch
            from saint.adapters.drm_grafting_data import real_token_batch
        except ImportError as exc:
            self.skipTest(str(exc))

        root = Path(__file__).resolve().parents[2]
        data_dir = root / "drm_transformer" / "data" / "baseline"
        if not data_dir.exists():
            self.skipTest("external DRM token fixture not available")

        inputs, targets = real_token_batch(
            torch,
            {
                "drm_root": str(root / "drm_transformer"),
                "real_data_dir": "data/baseline",
                "batch_size": 1,
                "seq_len": 8,
                "validation_seed": 1032,
            },
            50000,
            "cpu",
            split="val",
            seed_key="validation_seed",
        )

        self.assertEqual(tuple(inputs.shape), (1, 8))
        self.assertEqual(tuple(targets.shape), (1, 8))

    def test_default_progressive_candidates(self):
        from saint.adapters.drm_grafting_progressive import _default_candidates

        candidates = _default_candidates({})

        self.assertEqual(candidates[0]["target_module"], "blocks.1")
        self.assertEqual(candidates[1]["target_module"], "blocks.2")
        self.assertEqual(candidates[2]["target_module"], "final_norm")


if __name__ == "__main__":
    unittest.main()
