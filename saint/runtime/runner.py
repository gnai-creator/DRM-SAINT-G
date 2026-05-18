"""Unified runtime runner for small SAINT experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from saint.adapters import inspect_model, make_task, run_method
from saint.checkpoints import (
    checkpoint_payload,
    read_json,
    require_delta_payload,
    require_optimizer_state,
    validate_checkpoint_bundle,
    write_checkpoint_bundle,
    write_json,
    write_jsonl,
    write_metrics,
)
from saint.config import RuntimeConfig, load_config, save_config
from saint.memory import estimate_runtime_memory
from saint.transformer.model import combine_weights


def _shape_summary(weights: dict) -> dict:
    return {
        name: {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
        }
        for name, matrix in weights.items()
    }


def _select_weights(weights: dict, matrix_names: set[str] | None) -> dict:
    if matrix_names is None:
        return weights
    missing = sorted(matrix_names - set(weights))
    if missing:
        raise ValueError(f"unknown matrices requested for merge: {missing}")
    return {name: weights[name] for name in weights if name in matrix_names}


def inspect_runtime(config: RuntimeConfig) -> dict:
    return inspect_model(config)


def estimate_runtime(config: RuntimeConfig) -> dict:
    return estimate_runtime_memory(config).__dict__


def train_runtime(config: RuntimeConfig) -> dict:
    start = perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_plan = estimate_runtime_memory(config)
    save_config(config, output_dir / "config.json")
    events = [
        {"event": "start", "experiment_name": config.experiment_name},
        {"event": "memory_plan", **memory_plan.__dict__},
    ]
    result = run_method(config)
    payload = checkpoint_payload(result, config, memory_plan)
    payload["elapsed_s_total"] = perf_counter() - start
    write_metrics(output_dir / "metrics.json", payload)
    manifest = write_checkpoint_bundle(output_dir, payload)
    events.append(
        {
            "event": "result",
            "method": result.name,
            "test_loss": result.test_loss,
            "parameter_count": result.parameter_count,
        }
    )
    events.append({"event": "complete"})
    write_jsonl(output_dir / "logs.jsonl", events)
    return manifest


def resume_runtime(run_dir: str | Path) -> dict:
    checkpoint = validate_checkpoint_bundle(run_dir)
    checkpoint["optimizer_state"] = require_optimizer_state(checkpoint, run_dir)
    checkpoint["resumed"] = True
    return checkpoint


def merge_runtime(run_dir: str | Path, matrix_names: set[str] | None = None) -> dict:
    run_path = Path(run_dir)
    checkpoint = validate_checkpoint_bundle(
        run_path,
        validate_payloads=matrix_names is None,
    )
    config = load_config(run_path / "config.json")
    task = make_task(config)
    base_weights = _select_weights(task.base_weights, matrix_names)
    delta_payload = require_delta_payload(
        checkpoint,
        run_path,
        matrix_names=set(base_weights),
    )
    merged_weights = combine_weights(base_weights, delta_payload)
    base_shapes = _shape_summary(base_weights)
    merged_shapes = _shape_summary(merged_weights)
    merged = {
        "experiment_name": checkpoint["experiment_name"],
        "method": checkpoint["method"],
        "parameter_count": checkpoint["parameter_count"],
        "merged_weights": merged_weights,
        "partial": matrix_names is not None,
        "selected_matrices": sorted(base_weights),
        "shape_validation": base_shapes == merged_shapes,
        "shapes": merged_shapes,
        "merged": True,
    }
    write_json(run_path / "merged.json", merged)
    return merged


def load_and_train(config_path: str | Path) -> dict:
    return train_runtime(load_config(config_path))


__all__ = [
    "estimate_runtime",
    "inspect_runtime",
    "load_and_train",
    "merge_runtime",
    "resume_runtime",
    "train_runtime",
]
