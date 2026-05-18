import tempfile
from pathlib import Path
import unittest

from saint.checkpoints import (
    MANIFEST_FORMAT_VERSION,
    read_json,
    validate_checkpoint_bundle,
    write_json,
)
from saint.cli import main as cli_main
from saint.config import RuntimeConfig, config_from_dict, load_config, save_config
from saint.memory import estimate_runtime_memory
from saint.runtime import (
    estimate_runtime,
    inspect_runtime,
    merge_runtime,
    resume_runtime,
    train_runtime,
)


class RuntimePhase7Tests(unittest.TestCase):
    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1)

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(loaded.output_dir, config.output_dir)
            self.assertEqual(loaded.steps, 1)

    def test_config_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            config_from_dict({"missing": True})

    def test_memory_plan_fits_small_runtime(self):
        plan = estimate_runtime_memory(RuntimeConfig(parameter_budget=24))

        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.trainable_parameters, 24)

    def test_inspect_and_estimate_runtime(self):
        config = RuntimeConfig(parameter_budget=24)

        inspected = inspect_runtime(config)
        estimated = estimate_runtime(config)

        self.assertIn("matrices", inspected)
        self.assertTrue(estimated["fits_budget"])

    def test_drm_adapter_requires_checkpoint(self):
        config = RuntimeConfig(task="drm_transformer")

        with self.assertRaises(ValueError):
            inspect_runtime(config)

    def test_drm_checkpoint_smoke_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "drm_checkpoint.json"
            run_dir = Path(tmp) / "run"
            write_json(
                checkpoint,
                {
                    "model": {
                        "layers.0.attn.q_proj.weight": [
                            [0.1, -0.2, 0.3, -0.4],
                            [0.2, 0.1, -0.1, -0.2],
                            [0.5, -0.3, 0.2, 0.1],
                            [-0.1, 0.4, -0.5, 0.2],
                        ],
                        "layers.0.ffn.up_proj.weight": [
                            [0.2, -0.1],
                            [0.1, 0.3],
                        ],
                    }
                },
            )
            config = RuntimeConfig(
                experiment_name="drm_smoke",
                output_dir=str(run_dir),
                task="drm_transformer",
                method="drm_saint_delta_smoke",
                parameter_budget=8,
                metadata={
                    "checkpoint": str(checkpoint),
                    "max_dim": 4,
                    "max_matrices": 2,
                    "block_size": 2,
                },
            )

            inspected = inspect_runtime(config)
            result = train_runtime(config)
            resumed = resume_runtime(run_dir)
            merged = merge_runtime(run_dir)

            self.assertEqual(inspected["adapter"], "drm_transformer_checkpoint")
            self.assertEqual(len(inspected["matrices"]), 2)
            self.assertTrue(result["has_delta_payload"])
            self.assertEqual(result["format"], "saint_checkpoint")
            self.assertEqual(result["format_version"], MANIFEST_FORMAT_VERSION)
            self.assertNotIn("delta_payload", result)
            self.assertTrue((run_dir / "deltas.saintbin").exists())
            self.assertTrue((run_dir / "optimizer.saintopt").exists())
            self.assertEqual(result["metadata"]["marco"], "fase_9_marco_1")
            self.assertTrue(result["metadata"]["shape_validation"])
            self.assertTrue(resumed["resumed"])
            self.assertIn("optimizer_state", resumed)
            self.assertTrue(merged["merged"])
            self.assertTrue(merged["shape_validation"])
            self.assertIn("layers.0.attn.q_proj.weight", merged["merged_weights"])

    def test_train_resume_and_merge_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1, parameter_budget=24)

            result = train_runtime(config)
            resumed = resume_runtime(config.output_dir)
            merged = merge_runtime(config.output_dir)

            self.assertEqual(result["experiment_name"], config.experiment_name)
            self.assertTrue(result["has_delta_payload"])
            self.assertEqual(result["format"], "saint_checkpoint")
            self.assertEqual(result["format_version"], MANIFEST_FORMAT_VERSION)
            self.assertNotIn("delta_payload", result)
            self.assertTrue((Path(config.output_dir) / "checkpoint.json").exists())
            self.assertTrue((Path(config.output_dir) / "deltas.saintbin").exists())
            self.assertTrue((Path(config.output_dir) / "optimizer.saintopt").exists())
            self.assertTrue(resumed["resumed"])
            self.assertIn("optimizer_state", resumed)
            self.assertTrue(merged["merged"])
            self.assertIn("merged_weights", merged)

    def test_resume_rejects_corrupt_compact_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1, parameter_budget=24)

            train_runtime(config)
            delta_path = Path(config.output_dir) / "deltas.saintbin"
            with delta_path.open("ab") as handle:
                handle.write(b"corrupt")

            with self.assertRaises(ValueError):
                resume_runtime(config.output_dir)

    def test_checkpoint_manifest_v1_migrates_to_current_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1, parameter_budget=24)

            train_runtime(config)
            checkpoint_path = Path(config.output_dir) / "checkpoint.json"
            checkpoint = read_json(checkpoint_path)
            checkpoint["format_version"] = 1
            checkpoint.pop("compatibility", None)
            write_json(checkpoint_path, checkpoint)

            migrated = validate_checkpoint_bundle(config.output_dir)

            self.assertEqual(migrated["format_version"], MANIFEST_FORMAT_VERSION)
            self.assertEqual(migrated["migrated_from"]["format_version"], 1)
            self.assertEqual(
                migrated["compatibility"]["payload_format_version"],
                1,
            )

    def test_checkpoint_manifest_future_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1, parameter_budget=24)

            train_runtime(config)
            checkpoint_path = Path(config.output_dir) / "checkpoint.json"
            checkpoint = read_json(checkpoint_path)
            checkpoint["format_version"] = MANIFEST_FORMAT_VERSION + 100
            write_json(checkpoint_path, checkpoint)

            with self.assertRaises(ValueError):
                validate_checkpoint_bundle(config.output_dir)

    def test_sharded_dtype_checkpoint_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(
                output_dir=str(Path(tmp) / "run"),
                steps=1,
                parameter_budget=24,
                metadata={
                    "checkpoint_dtype": "float16",
                    "checkpoint_shard_bytes": 64,
                },
            )

            result = train_runtime(config)
            merged = merge_runtime(config.output_dir)
            delta_entry = next(
                entry for entry in result["files"] if entry["payload"] == "delta"
            )

            self.assertEqual(delta_entry["format"], "saint_matrix_shards")
            self.assertEqual(delta_entry["dtype"], "float16")
            self.assertGreater(delta_entry["shard_count"], 1)
            self.assertTrue(merged["shape_validation"])

    def test_partial_merge_selects_requested_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(
                output_dir=str(Path(tmp) / "run"),
                steps=1,
                parameter_budget=24,
                metadata={
                    "checkpoint_shard_bytes": 64,
                },
            )

            train_runtime(config)
            merged = merge_runtime(config.output_dir, matrix_names={"w_q"})

            self.assertTrue(merged["merged"])
            self.assertTrue(merged["partial"])
            self.assertEqual(merged["selected_matrices"], ["w_q"])
            self.assertEqual(set(merged["merged_weights"]), {"w_q"})
            self.assertTrue(merged["shape_validation"])

    def test_cli_commands_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            run_dir = Path(tmp) / "run"
            save_config(RuntimeConfig(output_dir=str(run_dir), steps=1, parameter_budget=24), config_path)

            self.assertEqual(cli_main(["inspect", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["estimate", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["train", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["resume", "--run", str(run_dir)]), 0)
            self.assertEqual(cli_main(["merge", "--run", str(run_dir)]), 0)


if __name__ == "__main__":
    unittest.main()
