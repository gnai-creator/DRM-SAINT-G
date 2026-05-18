import tempfile
from pathlib import Path
import unittest
import importlib.util

from saint.checkpoints import write_json
from saint.config import RuntimeConfig
from saint.runtime import inspect_runtime, merge_runtime, resume_runtime, train_runtime


def _write_hf_checkpoint(path: Path) -> None:
    write_json(
        path,
        {
            "model": {
                "model.layers.0.self_attn.q_proj.weight": [
                    [0.1, -0.2, 0.3, -0.4],
                    [0.2, 0.1, -0.1, -0.2],
                    [0.5, -0.3, 0.2, 0.1],
                    [-0.1, 0.4, -0.5, 0.2],
                ],
                "model.layers.0.self_attn.v_proj.weight": [
                    [0.2, -0.1, 0.1, 0.3],
                    [0.1, 0.3, -0.2, 0.2],
                    [-0.3, 0.2, 0.4, -0.1],
                    [0.2, -0.4, 0.1, 0.5],
                ],
                "model.layers.0.mlp.down_proj.weight": [
                    [0.2, -0.1],
                    [0.1, 0.3],
                ],
                "model.layers.0.norm.weight": [1.0, 1.0, 1.0, 1.0],
            }
        },
    )


class HuggingFacePhase13Tests(unittest.TestCase):
    def test_huggingface_json_checkpoint_smoke_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "hf_state.json"
            run_dir = Path(tmp) / "run"
            _write_hf_checkpoint(checkpoint)
            config = RuntimeConfig(
                experiment_name="hf_smoke",
                output_dir=str(run_dir),
                task="huggingface_causal_lm",
                method="hf_saint_delta_smoke",
                parameter_budget=8,
                metadata={
                    "model_name_or_path": str(checkpoint),
                    "max_dim": 4,
                    "max_matrices": 2,
                    "block_size": 2,
                    "checkpoint_dtype": "float16",
                    "checkpoint_shard_bytes": 128,
                },
            )

            inspected = inspect_runtime(config)
            result = train_runtime(config)
            resumed = resume_runtime(run_dir)
            merged = merge_runtime(run_dir)

            self.assertEqual(inspected["adapter"], "huggingface_causal_lm")
            self.assertEqual(len(inspected["matrices"]), 2)
            self.assertEqual(result["method"], "hf_saint_delta_smoke")
            self.assertEqual(result["metadata"]["marco"], "fase_13_marco_1")
            self.assertTrue(result["has_delta_payload"])
            self.assertTrue(resumed["resumed"])
            self.assertTrue(merged["merged"])
            self.assertTrue(merged["shape_validation"])
            self.assertIn(
                "model.layers.0.self_attn.q_proj.weight",
                merged["merged_weights"],
            )

    def test_huggingface_adapter_requires_model_source(self):
        config = RuntimeConfig(task="huggingface_causal_lm")

        with self.assertRaises(ValueError):
            inspect_runtime(config)

    def test_huggingface_autograd_requires_torch_or_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "hf_state.json"
            run_dir = Path(tmp) / "run"
            _write_hf_checkpoint(checkpoint)
            config = RuntimeConfig(
                experiment_name="hf_autograd_smoke",
                output_dir=str(run_dir),
                task="huggingface_causal_lm",
                method="hf_saint_autograd_smoke",
                steps=2,
                parameter_budget=8,
                metadata={
                    "model_name_or_path": str(checkpoint),
                    "max_dim": 4,
                    "max_matrices": 2,
                    "checkpoint_dtype": "float16",
                    "checkpoint_shard_bytes": 128,
                },
            )

            if importlib.util.find_spec("torch") is None:
                with self.assertRaises(RuntimeError):
                    train_runtime(config)
                return

            result = train_runtime(config)
            merged = merge_runtime(run_dir)

            self.assertEqual(result["metadata"]["marco"], "fase_13_marco_2")
            self.assertTrue(result["metadata"]["autograd"])
            self.assertLessEqual(
                result["train_loss"],
                result["metadata"]["initial_loss"],
            )
            self.assertTrue(merged["shape_validation"])


if __name__ == "__main__":
    unittest.main()
