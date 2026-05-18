"""Budgeted baselines for phase-4 linear experiments."""

from __future__ import annotations

from time import perf_counter

from saint.reconstruction.matrix_ops import shape, zeros
from saint.training.data import LinearTask
from saint.training.ops import TrainingResult, loss_and_delta_gradient, training_result


def _block_score(gradient, row_start: int, col_start: int, block_size: int) -> float:
    rows, cols = shape(gradient)
    total = 0.0
    for row in range(row_start, min(row_start + block_size, rows)):
        for col in range(col_start, min(col_start + block_size, cols)):
            total += abs(gradient[row][col])
    return total


def train_block_budgeted_delta(
    task: LinearTask,
    *,
    parameter_budget: int,
    block_size: int = 2,
    steps: int = 240,
    learning_rate: float = 0.35,
    name: str = "block_budgeted_delta",
) -> TrainingResult:
    """Train full block deltas selected by initial gradient sensitivity."""

    start = perf_counter()
    rows, cols = shape(task.base_weight)
    delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, delta)
    blocks = sorted(
        (
            (_block_score(initial_gradient, row, col, block_size), row, col)
            for row in range(0, rows, block_size)
            for col in range(0, cols, block_size)
        ),
        reverse=True,
    )
    params_per_block = block_size * block_size
    block_budget = max(1, parameter_budget // params_per_block)
    trainable_blocks = {
        (row, col)
        for _score, row, col in blocks[: min(block_budget, len(blocks))]
    }

    def is_trainable(row: int, col: int) -> bool:
        return ((row // block_size) * block_size, (col // block_size) * block_size) in trainable_blocks

    for _ in range(steps):
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row in range(rows):
            for col in range(cols):
                if is_trainable(row, col):
                    delta[row][col] -= learning_rate * gradient[row][col]

    parameter_count = len(trainable_blocks) * params_per_block
    return training_result(
        name,
        task,
        delta,
        parameter_count,
        start,
        {
            "parameter_budget": parameter_budget,
            "block_size": block_size,
            "trainable_blocks": len(trainable_blocks),
        },
    )


__all__ = ["train_block_budgeted_delta"]
