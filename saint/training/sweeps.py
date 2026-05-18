"""Benchmark sweeps for phase-4 linear experiments."""

from __future__ import annotations

from saint.training.data import LinearTask, make_linear_delta_task
from saint.training.methods import (
    train_block_scalar_delta,
    train_budgeted_full_delta,
    train_codebook_delta,
    train_full_delta,
    train_lora_delta,
    train_sparse_sensitivity_delta,
)
from saint.training.advanced_saint import train_saint_dynamic_delta
from saint.training.budgeted import train_block_budgeted_delta
from saint.training.lora_tuning import train_tuned_lora_delta
from saint.training.ops import TrainingResult
from saint.training.saint_delta import (
    train_saint_global_scaled_residual,
    train_saint_routed_delta,
)


def run_linear_phase4_benchmark(task: LinearTask | None = None) -> list[TrainingResult]:
    task = task or make_linear_delta_task()
    return [
        train_full_delta(task),
        train_lora_delta(task, rank=2),
        train_block_scalar_delta(task, block_size=2),
        train_codebook_delta(task, block_size=2),
        train_saint_routed_delta(task),
        train_sparse_sensitivity_delta(task, trainable_fraction=0.25),
    ]


def _row_from_result(
    result: TrainingResult,
    *,
    seed: int,
    rows: int,
    cols: int,
    delta_mode: str,
) -> dict:
    return {
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "delta_mode": delta_mode,
        "method": result.name,
        "train_loss": result.train_loss,
        "test_loss": result.test_loss,
        "weight_relative_l1_error": result.weight_relative_l1_error,
        "parameter_count": result.parameter_count,
        "optimizer_state_values": result.optimizer_state_values,
        "elapsed_s": result.elapsed_s,
        "gain_per_parameter": result.metadata["gain_per_parameter"],
        "metadata": result.metadata,
    }


def run_linear_phase4_sweep(
    *,
    seeds: tuple[int, ...] = (11, 12, 13, 14, 15),
    rows: int = 8,
    cols: int = 8,
    train_samples: int = 96,
    test_samples: int = 32,
    delta_mode: str = "repeated",
    steps: int = 240,
    lora_steps: int = 320,
) -> list[dict]:
    """Run a multi-seed phase-4 sweep and return flat result rows."""

    rows_out = []
    saint_budgets = (
        {
            "name": "saint_routed_f25_c50",
            "free_region_fraction": 0.25,
            "codebook_region_fraction": 0.50,
        },
        {
            "name": "saint_routed_f25_c25",
            "free_region_fraction": 0.25,
            "codebook_region_fraction": 0.25,
        },
        {
            "name": "saint_routed_f50_c25",
            "free_region_fraction": 0.50,
            "codebook_region_fraction": 0.25,
        },
        {
            "name": "saint_global_capped",
            "free_region_fraction": 0.125,
            "codebook_region_fraction": 0.375,
            "max_free_regions": 4,
            "max_prototypes": 16,
        },
    )
    lora_learning_rates = {1: 0.45, 2: 0.55, 4: 0.35}
    for seed in seeds:
        task = make_linear_delta_task(
            rows=rows,
            cols=cols,
            train_samples=train_samples,
            test_samples=test_samples,
            seed=seed,
            delta_mode=delta_mode,
        )
        results = [
            train_full_delta(task, steps=steps),
            train_lora_delta(
                task,
                rank=1,
                steps=lora_steps,
                learning_rate=lora_learning_rates[1],
            ),
            train_lora_delta(
                task,
                rank=2,
                steps=lora_steps,
                learning_rate=lora_learning_rates[2],
            ),
            train_lora_delta(
                task,
                rank=4,
                steps=lora_steps,
                learning_rate=lora_learning_rates[4],
            ),
            train_lora_delta(
                task,
                rank=8,
                steps=lora_steps,
                learning_rate=0.2,
            ),
            train_tuned_lora_delta(task, rank=1),
            train_tuned_lora_delta(task, rank=2),
            train_tuned_lora_delta(task, rank=4),
            train_tuned_lora_delta(task, rank=8),
            train_block_scalar_delta(task, block_size=2, steps=steps),
            train_codebook_delta(task, block_size=2, steps=steps),
            train_sparse_sensitivity_delta(task, trainable_fraction=0.25, steps=steps),
        ]
        saint_results = [
            train_saint_routed_delta(
                task,
                name=config["name"],
                free_region_fraction=config["free_region_fraction"],
                codebook_region_fraction=config["codebook_region_fraction"],
                max_free_regions=config.get("max_free_regions"),
                max_codebook_regions=config.get("max_codebook_regions"),
                max_prototypes=config.get("max_prototypes"),
                steps=steps,
            )
            for config in saint_budgets
        ]
        saint_results.append(
            train_saint_global_scaled_residual(
                task,
                steps=steps,
            )
        )
        saint_results.append(train_saint_dynamic_delta(task, steps=steps))
        results.extend(saint_results)
        results.extend(
            train_budgeted_full_delta(
                task,
                parameter_budget=result.parameter_count,
                steps=steps,
                name=f"budgeted_full_delta_for_{result.name}",
            )
            for result in saint_results
        )
        results.extend(
            train_block_budgeted_delta(
                task,
                parameter_budget=result.parameter_count,
                steps=steps,
                name=f"block_budgeted_delta_for_{result.name}",
            )
            for result in saint_results
        )
        for result in results:
            rows_out.append(
                _row_from_result(
                    result,
                    seed=seed,
                    rows=rows,
                    cols=cols,
                    delta_mode=delta_mode,
                )
            )
    return rows_out


def run_linear_phase4_regime_sweep(
    *,
    seeds: tuple[int, ...] = (11, 12),
    sizes: tuple[int, ...] = (8, 16, 32),
    delta_modes: tuple[str, ...] = ("repeated", "dense"),
    train_samples: int = 32,
    test_samples: int = 16,
    steps: int = 90,
    lora_steps: int = 140,
) -> list[dict]:
    rows_out = []
    for size in sizes:
        for delta_mode in delta_modes:
            rows_out.extend(
                run_linear_phase4_sweep(
                    seeds=seeds,
                    rows=size,
                    cols=size,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    delta_mode=delta_mode,
                    steps=steps,
                    lora_steps=lora_steps,
                )
            )
    return rows_out


def summarize_phase4_rows(rows: list[dict]) -> list[dict]:
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
                "avg_weight_relative_l1_error": sum(
                    row["weight_relative_l1_error"] for row in group
                ) / count,
                "avg_parameter_count": sum(row["parameter_count"] for row in group) / count,
                "avg_optimizer_state_values": sum(
                    row["optimizer_state_values"] for row in group
                ) / count,
                "avg_gain_per_parameter": sum(
                    row["gain_per_parameter"] for row in group
                ) / count,
                "avg_elapsed_s": sum(row["elapsed_s"] for row in group) / count,
            }
        )
    return sorted(summaries, key=lambda row: row["avg_test_loss"])
