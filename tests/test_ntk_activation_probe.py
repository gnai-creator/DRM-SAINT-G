import unittest
from types import SimpleNamespace


class NTKActivationProbeTests(unittest.TestCase):
    def test_marco_name_identifies_ntk_activation_probe(self):
        from saint.adapters.drm_grafting_graftblock_routed import _marco_name

        args = SimpleNamespace(
            ntk_activation_probe_batches=4,
            candidate_top_k=8,
            candidate_score_mode="composed_gain_orthogonal",
        )

        self.assertEqual(_marco_name(args), "4m_ntkmirror_activation_gate_probe")

    def test_activation_gate_scores_use_abs_grad_times_activation_per_target(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            self.skipTest(str(exc))

        from saint.adapters.drm_grafting_graftblock_routed import _activation_gate_scores_for_loss

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])
                with torch.no_grad():
                    self.blocks[0].weight.copy_(torch.eye(2))
                    self.blocks[1].weight.copy_(torch.eye(2))

            def forward(self, x):
                x = self.blocks[0](x)
                x = self.blocks[1](x)
                return x

        model = Tiny()
        x = torch.tensor([[1.0, -2.0]])

        def loss_fn():
            return model(x).sum()

        rows = _activation_gate_scores_for_loss(
            torch,
            model,
            ["blocks.0", "blocks.1"],
            loss_fn,
            stage=2,
            batch_index=3,
        )

        by_target = {row["target"]: row for row in rows}
        self.assertAlmostEqual(by_target["blocks.0"]["ntk_activation_score"], 3.0)
        self.assertAlmostEqual(by_target["blocks.1"]["ntk_activation_score"], 3.0)
        self.assertEqual(by_target["blocks.0"]["stage"], 2)
        self.assertEqual(by_target["blocks.0"]["batch_index"], 3)
        self.assertEqual(by_target["blocks.0"]["channel_count"], 2)
        self.assertEqual(by_target["blocks.0"]["top_channel"], 1)
        self.assertAlmostEqual(by_target["blocks.0"]["top_channel_score"], 2.0)

    def test_activation_gate_scores_restore_parameter_requires_grad(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            self.skipTest(str(exc))

        from saint.adapters.drm_grafting_graftblock_routed import _activation_gate_scores_for_loss

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        for param in model.parameters():
            param.requires_grad_(False)
        x = torch.tensor([[0.5, -0.25]])

        def loss_fn():
            return model(x).sum()

        _activation_gate_scores_for_loss(torch, model, ["0", "2"], loss_fn)

        self.assertTrue(all(not param.requires_grad for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
