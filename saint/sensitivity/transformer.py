"""Sensitivity maps for phase-6 mini-transformer experiments."""

from __future__ import annotations

from random import Random
from time import perf_counter

from saint.reconstruction.matrix_ops import shape, subtract
from saint.transformer.model import MiniTransformerTask, combine_weights, distillation_loss, zero_deltas
from saint.transformer.training import (
    Coord,
    MiniTransformerResult,
    _coords,
    _finite_gradient,
    _initial_gradients,
    _loss,
    _result,
)


def _target_deltas(task: MiniTransformerTask):
    return {
        name: subtract(task.target_weights[name], matrix)
        for name, matrix in task.base_weights.items()
    }


def _loss_with_delta_value(
    task: MiniTransformerTask,
    coord: Coord,
    value: float,
) -> float:
    deltas = zero_deltas(task.base_weights)
    name, row, col = coord
    deltas[name][row][col] = value
    weights = combine_weights(task.base_weights, deltas)
    return distillation_loss(weights, task.target_weights, task.train_sequences)


def _matrix_error_scores(task: MiniTransformerTask) -> dict[str, float]:
    deltas = _target_deltas(task)
    return {
        name: sum(abs(value) for row in matrix for value in row)
        for name, matrix in deltas.items()
    }


def _pattern_frequency_scores(task: MiniTransformerTask, block_size: int) -> dict[Coord, float]:
    deltas = _target_deltas(task)
    signatures = {}
    coord_to_signature = {}
    for name, matrix in deltas.items():
        rows, cols = shape(matrix)
        for row_start in range(0, rows, block_size):
            for col_start in range(0, cols, block_size):
                values = []
                coords = []
                for row in range(row_start, min(row_start + block_size, rows)):
                    for col in range(col_start, min(col_start + block_size, cols)):
                        values.append(round(matrix[row][col], 2))
                        coords.append((name, row, col))
                signature = tuple(values)
                signatures[signature] = signatures.get(signature, 0) + 1
                for coord in coords:
                    coord_to_signature[coord] = signature
    return {
        coord: float(signatures[signature])
        for coord, signature in coord_to_signature.items()
    }


def score_sensitivity(
    task: MiniTransformerTask,
    *,
    method: str,
    epsilon: float = 1e-4,
    block_size: int = 2,
    seed: int = 17,
) -> dict[Coord, float]:
    coords = _coords(task.base_weights)
    if method == "random":
        rng = Random(seed)
        return {coord: rng.random() for coord in coords}

    gradients = _initial_gradients(task, coords, epsilon=epsilon)
    target_deltas = _target_deltas(task)
    base_loss = _loss(task, zero_deltas(task.base_weights), train=True)
    matrix_scores = _matrix_error_scores(task)
    frequency_scores = _pattern_frequency_scores(task, block_size)
    scores = {}
    for coord in coords:
        name, row, col = coord
        grad = gradients[coord]
        base_value = task.base_weights[name][row][col]
        target_delta = target_deltas[name][row][col]
        if method == "gradient_norm":
            scores[coord] = abs(grad)
        elif method == "gradient_weight":
            scores[coord] = abs(grad * base_value)
        elif method == "mask_impact":
            scores[coord] = max(base_loss - _loss_with_delta_value(task, coord, target_delta), 0.0)
        elif method == "fisher":
            scores[coord] = grad * grad
        elif method == "activation_magnitude":
            scores[coord] = abs(base_value)
        elif method == "layer_error":
            scores[coord] = matrix_scores[name]
        elif method == "gain_per_byte":
            gain = max(base_loss - _loss_with_delta_value(task, coord, target_delta), 0.0)
            scores[coord] = gain
        elif method == "pattern_frequency":
            scores[coord] = frequency_scores.get(coord, 0.0) * abs(target_delta)
        else:
            raise ValueError(f"unknown sensitivity method: {method}")
    return scores


def train_mini_sensitivity_delta(
    task: MiniTransformerTask,
    *,
    method: str,
    parameter_budget: int = 48,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    seed: int = 17,
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    scores = score_sensitivity(task, method=method, epsilon=epsilon, seed=seed)
    trainable = sorted(scores, key=lambda coord: scores[coord], reverse=True)
    trainable = trainable[: max(1, min(parameter_budget, len(trainable)))]
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result(
        f"sensitivity_{method}",
        task,
        deltas,
        len(trainable),
        start,
        {"method": method, "parameter_budget": parameter_budget},
    )


def _block_coords_for_coord(task: MiniTransformerTask, coord: Coord, block_size: int) -> list[Coord]:
    name, row, col = coord
    row_start = (row // block_size) * block_size
    col_start = (col // block_size) * block_size
    rows, cols = shape(task.base_weights[name])
    return [
        (name, item_row, item_col)
        for item_row in range(row_start, min(row_start + block_size, rows))
        for item_col in range(col_start, min(col_start + block_size, cols))
    ]


def train_mini_block_sensitivity_delta(
    task: MiniTransformerTask,
    *,
    method: str,
    parameter_budget: int = 48,
    block_size: int = 2,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    seed: int = 17,
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    scores = score_sensitivity(task, method=method, epsilon=epsilon, seed=seed)
    blocks = {}
    for coord, score in scores.items():
        block = tuple(_block_coords_for_coord(task, coord, block_size))
        blocks[block] = blocks.get(block, 0.0) + score
    block_budget = max(1, parameter_budget // (block_size * block_size))
    selected = sorted(blocks, key=lambda block: blocks[block], reverse=True)[:block_budget]
    trainable = [coord for block in selected for coord in block]
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result(
        f"block_sensitivity_{method}",
        task,
        deltas,
        len(trainable),
        start,
        {"method": method, "parameter_budget": parameter_budget, "block_size": block_size},
    )


def train_mini_accumulated_sensitivity_delta(
    task: MiniTransformerTask,
    *,
    parameter_budget: int = 48,
    warmup_steps: int = 3,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    coords = _coords(task.base_weights)
    accumulated = {coord: 0.0 for coord in coords}
    for _ in range(warmup_steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in coords
        }
        for coord in coords:
            accumulated[coord] += abs(gradients[coord])
            matrix_name, row, col = coord
            deltas[matrix_name][row][col] -= learning_rate * gradients[coord]
    deltas = zero_deltas(task.base_weights)
    trainable = sorted(accumulated, key=lambda coord: accumulated[coord], reverse=True)
    trainable = trainable[: max(1, min(parameter_budget, len(trainable)))]
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result(
        "sensitivity_accumulated_gradient",
        task,
        deltas,
        len(trainable),
        start,
        {"method": "accumulated_gradient", "parameter_budget": parameter_budget},
    )


def run_sensitivity_sweep(
    *,
    seeds: tuple[int, ...] = (41, 42),
    delta_modes: tuple[str, ...] = ("repeated", "dense"),
    methods: tuple[str, ...] = (
        "random",
        "gradient_norm",
        "gradient_weight",
        "mask_impact",
        "fisher",
        "activation_magnitude",
        "layer_error",
        "gain_per_byte",
        "pattern_frequency",
    ),
    parameter_budget: int = 48,
    steps: int = 8,
    delta_scale: float = 3.0,
) -> list[dict]:
    from saint.transformer.model import make_mini_transformer_task

    rows = []
    for seed in seeds:
        for delta_mode in delta_modes:
            task = make_mini_transformer_task(seed=seed, delta_mode=delta_mode, delta_scale=delta_scale)
            results = []
            for method in methods:
                result = train_mini_sensitivity_delta(
                    task,
                    method=method,
                    parameter_budget=parameter_budget,
                    steps=steps,
                    seed=seed,
                )
                results.append(result)
            for method in ("gradient_norm", "fisher", "gain_per_byte"):
                results.append(
                    train_mini_block_sensitivity_delta(
                        task,
                        method=method,
                        parameter_budget=parameter_budget,
                        steps=steps,
                        seed=seed,
                    )
                )
            results.append(
                train_mini_accumulated_sensitivity_delta(
                    task,
                    parameter_budget=parameter_budget,
                    steps=steps,
                )
            )
            from saint.transformer.saint_adapter import train_mini_saint_delta

            results.extend(
                (
                    train_mini_saint_delta(
                        task,
                        parameter_budget=parameter_budget,
                        steps=steps,
                        name="mini_saint_default_for_sensitivity",
                    ),
                    train_mini_saint_delta(
                        task,
                        parameter_budget=parameter_budget,
                        steps=steps,
                        name="mini_saint_gradient_norm",
                        sensitivity_method="gradient_norm",
                    ),
                )
            )
            for result in results:
                rows.append(
                    {
                        "seed": seed,
                        "delta_mode": delta_mode,
                        "method": result.name,
                        "train_loss": result.train_loss,
                        "test_loss": result.test_loss,
                        "parameter_count": result.parameter_count,
                        "gain_per_parameter": result.metadata["gain_per_parameter"],
                    }
                )
    return rows


def summarize_sensitivity_rows(rows: list[dict]) -> list[dict]:
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
                "avg_gain_per_parameter": sum(row["gain_per_parameter"] for row in group) / count,
            }
        )
    return sorted(summaries, key=lambda row: row["avg_test_loss"])


def evaluate_sensitivity_success(rows: list[dict]) -> dict:
    summaries = {row["method"]: row for row in summarize_sensitivity_rows(rows)}
    random_loss = summaries["sensitivity_random"]["avg_test_loss"]
    winners = [
        method
        for method, summary in summaries.items()
        if method != "sensitivity_random" and summary["avg_test_loss"] < random_loss
    ]
    return {
        "passed": len(winners) >= 3,
        "random_avg_test_loss": random_loss,
        "winner_count": len(winners),
        "winners": winners,
        "reason": "passed" if len(winners) >= 3 else "fewer than 3 methods beat random",
    }


def summarize_sensitivity_by_regime(rows: list[dict]) -> list[dict]:
    summaries = []
    for delta_mode in sorted({row["delta_mode"] for row in rows}):
        for summary in summarize_sensitivity_rows(
            [row for row in rows if row["delta_mode"] == delta_mode]
        ):
            summaries.append({"delta_mode": delta_mode, **summary})
    return summaries


def evaluate_sensitivity_final(rows: list[dict]) -> dict:
    by_method = {row["method"]: row for row in summarize_sensitivity_rows(rows)}
    random_loss = by_method["sensitivity_random"]["avg_test_loss"]
    regime_passes = 0
    for delta_mode in sorted({row["delta_mode"] for row in rows}):
        decision = evaluate_sensitivity_success(
            [row for row in rows if row["delta_mode"] == delta_mode]
        )
        regime_passes += int(decision["passed"])
    block_best = min(
        row["avg_test_loss"]
        for method, row in by_method.items()
        if method.startswith("block_sensitivity_")
    )
    accumulated = by_method["sensitivity_accumulated_gradient"]["avg_test_loss"]
    saint_default = by_method["mini_saint_default_for_sensitivity"]
    saint_sensitive = by_method["mini_saint_gradient_norm"]
    passed = (
        regime_passes == len({row["delta_mode"] for row in rows})
        and block_best < random_loss
        and accumulated < random_loss
        and (
            saint_sensitive["avg_test_loss"] <= saint_default["avg_test_loss"]
            or saint_sensitive["avg_gain_per_parameter"] >= saint_default["avg_gain_per_parameter"]
        )
    )
    return {
        "passed": passed,
        "regime_passes": regime_passes,
        "block_best_beats_random": block_best < random_loss,
        "accumulated_beats_random": accumulated < random_loss,
        "sensitivity_saint_beats_default": (
            saint_sensitive["avg_test_loss"] <= saint_default["avg_test_loss"]
            or saint_sensitive["avg_gain_per_parameter"] >= saint_default["avg_gain_per_parameter"]
        ),
        "reason": "passed" if passed else "phase 6 final thresholds not met",
    }


__all__ = [
    "evaluate_sensitivity_final",
    "evaluate_sensitivity_success",
    "run_sensitivity_sweep",
    "score_sensitivity",
    "summarize_sensitivity_by_regime",
    "summarize_sensitivity_rows",
    "train_mini_accumulated_sensitivity_delta",
    "train_mini_block_sensitivity_delta",
    "train_mini_sensitivity_delta",
]
