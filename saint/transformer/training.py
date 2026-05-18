"""Phase-5 mini-transformer training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from saint.reconstruction.matrix_ops import shape, zeros
from saint.transformer.model import (
    MatrixDict,
    MiniTransformerTask,
    combine_weights,
    distillation_loss,
    zero_deltas,
)


@dataclass(frozen=True)
class MiniTransformerResult:
    name: str
    train_loss: float
    test_loss: float
    parameter_count: int
    optimizer_state_values: int
    elapsed_s: float
    metadata: dict = field(default_factory=dict)


Coord = tuple[str, int, int]


def _coords(weights: MatrixDict) -> list[Coord]:
    return [
        (name, row, col)
        for name, matrix in weights.items()
        for row in range(shape(matrix)[0])
        for col in range(shape(matrix)[1])
    ]


def _loss(task: MiniTransformerTask, deltas: MatrixDict, *, train: bool) -> float:
    weights = combine_weights(task.base_weights, deltas)
    sequences = task.train_sequences if train else task.test_sequences
    return distillation_loss(weights, task.target_weights, sequences)


def _finite_gradient(
    task: MiniTransformerTask,
    deltas: MatrixDict,
    coord: Coord,
    *,
    epsilon: float,
) -> float:
    name, row, col = coord
    original = deltas[name][row][col]
    deltas[name][row][col] = original + epsilon
    plus = _loss(task, deltas, train=True)
    deltas[name][row][col] = original - epsilon
    minus = _loss(task, deltas, train=True)
    deltas[name][row][col] = original
    return (plus - minus) / (2.0 * epsilon)


def _initial_gradients(
    task: MiniTransformerTask,
    coords: list[Coord],
    *,
    epsilon: float,
) -> dict[Coord, float]:
    deltas = zero_deltas(task.base_weights)
    return {
        coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
        for coord in coords
    }


def _result(
    name: str,
    task: MiniTransformerTask,
    deltas: MatrixDict,
    parameter_count: int,
    start: float,
    metadata: dict | None = None,
) -> MiniTransformerResult:
    base_loss = distillation_loss(
        task.base_weights,
        task.target_weights,
        task.test_sequences,
    )
    test_loss = _loss(task, deltas, train=False)
    gain = max(base_loss - test_loss, 0.0)
    full_metadata = dict(metadata or {})
    full_metadata.setdefault("baseline_test_loss", base_loss)
    full_metadata.setdefault("test_loss_gain", gain)
    full_metadata.setdefault(
        "gain_per_parameter",
        gain / parameter_count if parameter_count else 0.0,
    )
    return MiniTransformerResult(
        name=name,
        train_loss=_loss(task, deltas, train=True),
        test_loss=test_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata=full_metadata,
    )


def train_mini_full_delta(
    task: MiniTransformerTask,
    *,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    trainable = _coords(task.base_weights)
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result("mini_full_delta", task, deltas, len(trainable), start)


def train_mini_budgeted_delta(
    task: MiniTransformerTask,
    *,
    parameter_budget: int = 48,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    name: str = "mini_budgeted_delta",
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    all_coords = _coords(task.base_weights)
    initial = _initial_gradients(task, all_coords, epsilon=epsilon)
    trainable = sorted(all_coords, key=lambda coord: abs(initial[coord]), reverse=True)
    trainable = trainable[: max(1, min(parameter_budget, len(trainable)))]
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result(
        name,
        task,
        deltas,
        len(trainable),
        start,
        {"parameter_budget": parameter_budget},
    )


def _blocks(weights: MatrixDict, block_size: int) -> list[tuple[str, int, int]]:
    return [
        (name, row, col)
        for name, matrix in weights.items()
        for row in range(0, shape(matrix)[0], block_size)
        for col in range(0, shape(matrix)[1], block_size)
    ]


def _block_coords(
    name: str,
    row_start: int,
    col_start: int,
    rows: int,
    cols: int,
    block_size: int,
) -> list[Coord]:
    return [
        (name, row, col)
        for row in range(row_start, min(row_start + block_size, rows))
        for col in range(col_start, min(col_start + block_size, cols))
    ]


def train_mini_block_budgeted_delta(
    task: MiniTransformerTask,
    *,
    parameter_budget: int = 48,
    block_size: int = 2,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    name: str = "mini_block_budgeted_delta",
) -> MiniTransformerResult:
    start = perf_counter()
    deltas = zero_deltas(task.base_weights)
    all_coords = _coords(task.base_weights)
    initial = _initial_gradients(task, all_coords, epsilon=epsilon)
    ranked = []
    for matrix_name, row, col in _blocks(task.base_weights, block_size):
        rows, cols = shape(task.base_weights[matrix_name])
        coords = _block_coords(matrix_name, row, col, rows, cols, block_size)
        ranked.append((sum(abs(initial[coord]) for coord in coords), coords))
    block_budget = max(1, parameter_budget // (block_size * block_size))
    selected = sorted(ranked, reverse=True)[:block_budget]
    trainable = [coord for _score, coords in selected for coord in coords]
    for _ in range(steps):
        gradients = {
            coord: _finite_gradient(task, deltas, coord, epsilon=epsilon)
            for coord in trainable
        }
        for matrix_name, row, col in trainable:
            deltas[matrix_name][row][col] -= learning_rate * gradients[(matrix_name, row, col)]
    return _result(
        name,
        task,
        deltas,
        len(trainable),
        start,
        {"parameter_budget": parameter_budget, "block_size": block_size},
    )
