import tempfile
from pathlib import Path
import unittest
import importlib.util

from saint.checkpoints import write_json
from saint.config import RuntimeConfig
from saint.adapters.huggingface_benchmark import benchmark_hf_saint_vs_full
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


def _write_tiny_hf_model(path: Path) -> bool:
    if importlib.util.find_spec("transformers") is None:
        return False
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "simple": 3,
        "ai": 4,
        "node": 5,
        "training": 6,
        "saint": 7,
        "trains": 8,
        "compact": 9,
        "deltas": 10,
        "small": 11,
        "local": 12,
        "causal": 13,
        "language": 14,
        "model": 15,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return True


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

    def test_huggingface_forward_real_model_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny_model"
            run_dir = Path(tmp) / "run"
            if not _write_tiny_hf_model(model_dir):
                self.skipTest("transformers is not installed")
            config = RuntimeConfig(
                experiment_name="hf_forward_smoke",
                output_dir=str(run_dir),
                task="huggingface_causal_lm",
                method="hf_saint_forward_smoke",
                steps=2,
                parameter_budget=8,
                metadata={
                    "model_name_or_path": str(model_dir),
                    "max_dim": 64,
                    "max_matrices": 4,
                    "checkpoint_dtype": "float16",
                    "checkpoint_shard_bytes": 256,
                    "device": "cpu",
                    "learning_rate": 0.001,
                    "max_length": 12,
                },
            )

            result = train_runtime(config)
            merged = merge_runtime(run_dir)

            self.assertEqual(result["metadata"]["marco"], "fase_13_marco_3")
            self.assertTrue(result["metadata"]["real_forward"])
            self.assertIn("perplexity", result["metadata"])
            self.assertLessEqual(
                result["train_loss"],
                result["metadata"]["initial_loss"],
            )
            self.assertIn("tokens_per_s", result["metadata"])
            self.assertIn("cuda_peak_bytes", result["metadata"])
            self.assertTrue(merged["shape_validation"])

    def test_huggingface_baseline_comparison_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny_model"
            run_dir = Path(tmp) / "bench"
            if not _write_tiny_hf_model(model_dir):
                self.skipTest("transformers is not installed")

            result = benchmark_hf_saint_vs_full(
                model_dir,
                run_dir,
                seeds=(31, 32),
                steps=1,
                parameter_budget=8,
                device="cpu",
            )
            rows = result["rows"]
            saint_rows = [
                row for row in rows
                if row["method"] == "hf_saint_forward_smoke"
            ]
            full_rows = [
                row for row in rows
                if row["method"] == "hf_full_finetune"
            ]

            self.assertEqual(result["seeds"], [31, 32])
            self.assertEqual(len(saint_rows), 2)
            self.assertEqual(len(full_rows), 2)
            self.assertTrue(all(row["checkpoint_merge"] for row in saint_rows))
            self.assertTrue(all(row["tokens_per_s"] > 0.0 for row in rows))
            self.assertTrue(all("cuda_peak_bytes" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
