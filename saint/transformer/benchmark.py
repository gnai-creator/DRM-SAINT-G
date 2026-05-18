"""Benchmark helpers for phase-5 mini-transformer experiments."""

from __future__ import annotations

from saint.transformer.model import MiniTransformerTask, make_mini_transformer_task
from saint.transformer.lora import train_mini_lora_delta
from saint.transformer.saint_adapter import (
    train_mini_saint_delta,
    train_mini_saint_per_matrix_delta,
)
from saint.transformer.training import (
    MiniTransformerResult,
    train_mini_block_budgeted_delta,
    train_mini_budgeted_delta,
    train_mini_full_delta,
)


def run_mini_transformer_benchmark(
    task: MiniTransformerTask | None = None,
    *,
    steps: int = 8,
    parameter_budget: int = 48,
) -> list[MiniTransformerResult]:
    task = task or make_mini_transformer_task()
    saint = train_mini_saint_delta(task, parameter_budget=parameter_budget, steps=steps)
    saint_per_matrix = train_mini_saint_per_matrix_delta(
        task,
        parameter_budget=parameter_budget,
        steps=steps,
    )
    return [
        train_mini_full_delta(task, steps=steps),
        train_mini_lora_delta(task, rank=1, steps=steps),
        train_mini_lora_delta(task, rank=2, steps=steps, name="mini_lora_rank_2"),
        train_mini_budgeted_delta(
            task,
            parameter_budget=saint.parameter_count,
            steps=steps,
            name="mini_budgeted_delta_for_saint",
        ),
        train_mini_block_budgeted_delta(
            task,
            parameter_budget=saint.parameter_count,
            steps=steps,
            name="mini_block_budgeted_delta_for_saint",
        ),
        saint_per_matrix,
        saint,
    ]


def run_mini_transformer_sweep(
    *,
    seeds: tuple[int, ...] = (31, 32),
    delta_modes: tuple[str, ...] = ("repeated", "dense"),
    steps: int = 8,
    parameter_budget: int = 48,
    delta_scale: float = 3.0,
) -> list[dict]:
    rows = []
    for seed in seeds:
        for delta_mode in delta_modes:
            task = make_mini_transformer_task(
                seed=seed,
                delta_mode=delta_mode,
                delta_scale=delta_scale,
            )
            for result in run_mini_transformer_benchmark(
                task,
                steps=steps,
                parameter_budget=parameter_budget,
            ):
                rows.append(
                    {
                        "seed": seed,
                        "delta_mode": delta_mode,
                        "method": result.name,
                        "train_loss": result.train_loss,
                        "test_loss": result.test_loss,
                        "parameter_count": result.parameter_count,
                        "optimizer_state_values": result.optimizer_state_values,
                        "gain_per_parameter": result.metadata["gain_per_parameter"],
                        "metadata": result.metadata,
                    }
                )
    return rows


def summarize_mini_transformer_rows(rows: list[dict]) -> list[dict]:
    methods = sorted({row["method"] for row in rows})
    summaries = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        count = len(group)
        summaries.append(
            {
                "method": method,
                "runs": count,
                "avg_test_loss": sum(row["test_loss"] for row in group) / count,
                "avg_parameter_count": sum(row["parameter_count"] for row in group) / count,
                "avg_gain_per_parameter": sum(row["gain_per_parameter"] for row in group) / count,
            }
        )
    return sorted(summaries, key=lambda row: row["avg_test_loss"])


def evaluate_mini_transformer_closure(rows: list[dict]) -> dict:
    regimes = sorted({(row["seed"], row["delta_mode"]) for row in rows})
    wins = {
        "lora_rank_1": 0,
        "lora_rank_2": 0,
        "block_budgeted": 0,
        "budgeted": 0,
        "per_matrix": 0,
    }
    for seed, delta_mode in regimes:
        group = {
            row["method"]: row
            for row in rows
            if row["seed"] == seed and row["delta_mode"] == delta_mode
        }
        saint = group["mini_saint_dynamic_delta"]
        wins["lora_rank_1"] += saint["test_loss"] <= group["mini_lora_rank_1"]["test_loss"]
        wins["lora_rank_2"] += saint["test_loss"] <= group["mini_lora_rank_2"]["test_loss"]
        wins["block_budgeted"] += (
            saint["test_loss"] <= group["mini_block_budgeted_delta_for_saint"]["test_loss"]
        )
        wins["budgeted"] += saint["test_loss"] <= group["mini_budgeted_delta_for_saint"]["test_loss"]
        per_matrix = group["mini_saint_per_matrix_delta"]
        wins["per_matrix"] += (
            saint["test_loss"] <= per_matrix["test_loss"]
            or saint["gain_per_parameter"] >= per_matrix["gain_per_parameter"]
        )
    required = len(regimes)
    passed = (
        wins["lora_rank_1"] == required
        and wins["lora_rank_2"] == required
        and wins["block_budgeted"] >= required // 2
        and wins["budgeted"] >= required // 2
        and wins["per_matrix"] >= required // 2
    )
    return {
        "passed": passed,
        "regimes": required,
        **wins,
        "reason": "passed" if passed else "phase 5 closure thresholds not met",
    }


__all__ = [
    "run_mini_transformer_benchmark",
    "run_mini_transformer_sweep",
    "summarize_mini_transformer_rows",
    "evaluate_mini_transformer_closure",
]
