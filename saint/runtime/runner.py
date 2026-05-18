"""Unified runtime runner for small SAINT experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from saint.adapters import inspect_model, make_task, run_method
from saint.checkpoints import (
    checkpoint_payload,
    read_json,
    require_delta_payload,
    write_json,
    write_jsonl,
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
    write_json(output_dir / "metrics.json", payload)
    write_json(output_dir / "checkpoint.json", payload)
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
    return payload


def resume_runtime(run_dir: str | Path) -> dict:
    checkpoint = read_json(Path(run_dir) / "checkpoint.json")
    if checkpoint.get("has_delta_payload"):
        require_delta_payload(checkpoint)
    checkpoint["resumed"] = True
    return checkpoint


def merge_runtime(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    checkpoint = read_json(run_path / "checkpoint.json")
    config = load_config(run_path / "config.json")
    delta_payload = require_delta_payload(checkpoint)
    task = make_task(config)
    merged_weights = combine_weights(task.base_weights, delta_payload)
    base_shapes = _shape_summary(task.base_weights)
    merged_shapes = _shape_summary(merged_weights)
    merged = {
        "experiment_name": checkpoint["experiment_name"],
        "method": checkpoint["method"],
        "parameter_count": checkpoint["parameter_count"],
        "merged_weights": merged_weights,
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
