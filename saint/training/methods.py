"""Linear delta training methods used in phase 4."""

from __future__ import annotations

from random import Random
from time import perf_counter

from saint.blocks.partition import Matrix
from saint.reconstruction.matrix_ops import shape, zeros
from saint.training.data import LinearTask
from saint.training.ops import (
    TrainingResult,
    loss_and_delta_gradient,
    top_gradient_positions,
    training_result,
)


def train_full_delta(
    task: LinearTask,
    *,
    steps: int = 240,
    learning_rate: float = 0.35,
) -> TrainingResult:
    start = perf_counter()
    rows, cols = shape(task.base_weight)
    delta = zeros(rows, cols)
    for _ in range(steps):
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row in range(rows):
            for col in range(cols):
                delta[row][col] -= learning_rate * gradient[row][col]
    return training_result("full_delta", task, delta, rows * cols, start)


def train_sparse_sensitivity_delta(
    task: LinearTask,
    *,
    trainable_fraction: float = 0.25,
    steps: int = 240,
    learning_rate: float = 0.35,
) -> TrainingResult:
    start = perf_counter()
    rows, cols = shape(task.base_weight)
    delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, delta)
    budget = max(1, int(rows * cols * trainable_fraction))
    trainable = top_gradient_positions(initial_gradient, budget)
    for _ in range(steps):
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row, col in trainable:
            delta[row][col] -= learning_rate * gradient[row][col]
    return training_result(
        "sparse_sensitivity_delta",
        task,
        delta,
        len(trainable),
        start,
        {"trainable_fraction": trainable_fraction},
    )


def train_budgeted_full_delta(
    task: LinearTask,
    *,
    parameter_budget: int,
    steps: int = 240,
    learning_rate: float = 0.35,
    name: str = "budgeted_full_delta",
) -> TrainingResult:
    start = perf_counter()
    rows, cols = shape(task.base_weight)
    delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, delta)
    budget = max(1, min(rows * cols, parameter_budget))
    trainable = top_gradient_positions(initial_gradient, budget)
    for _ in range(steps):
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row, col in trainable:
            delta[row][col] -= learning_rate * gradient[row][col]
    return training_result(
        name,
        task,
        delta,
        len(trainable),
        start,
        {"parameter_budget": parameter_budget},
    )


def train_block_scalar_delta(
    task: LinearTask,
    *,
    block_size: int = 2,
    steps: int = 240,
    learning_rate: float = 0.22,
) -> TrainingResult:
    start = perf_counter()
    rows, cols = shape(task.base_weight)
    block_rows = (rows + block_size - 1) // block_size
    block_cols = (cols + block_size - 1) // block_size
    scalars = [[0.0 for _ in range(block_cols)] for _ in range(block_rows)]

    def materialize() -> list[list[float]]:
        return [
            [scalars[row // block_size][col // block_size] for col in range(cols)]
            for row in range(rows)
        ]

    for _ in range(steps):
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        scalar_gradient = [[0.0 for _ in range(block_cols)] for _ in range(block_rows)]
        for row in range(rows):
            for col in range(cols):
                scalar_gradient[row // block_size][col // block_size] += gradient[row][col]
        for row in range(block_rows):
            for col in range(block_cols):
                scalars[row][col] -= learning_rate * scalar_gradient[row][col]
    return training_result(
        f"block_scalar_{block_size}",
        task,
        materialize(),
        block_rows * block_cols,
        start,
        {"block_size": block_size},
    )


def block_signature_from_gradient(
    gradient: Matrix,
    row_start: int,
    col_start: int,
    block_size: int,
    quantization_step: float,
) -> tuple[int, ...]:
    rows, cols = shape(gradient)
    values = []
    for row in range(row_start, min(row_start + block_size, rows)):
        for col in range(col_start, min(col_start + block_size, cols)):
            values.append(float(gradient[row][col]))
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0.0:
        return tuple(0 for _ in values)
    return tuple(round((value / norm) / quantization_step) for value in values)


def _signature_bucket(signature: tuple[int, ...], bucket_count: int) -> int:
    value = sum((index + 1) * item for index, item in enumerate(signature))
    return abs(value) % bucket_count


def train_codebook_delta(
    task: LinearTask,
    *,
    block_size: int = 2,
    quantization_step: float = 0.25,
    steps: int = 240,
    learning_rate: float = 0.35,
) -> TrainingResult:
    start = perf_counter()
    rows, cols = shape(task.base_weight)
    zero_delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, zero_delta)
    block_positions = [
        (row, col)
        for row in range(0, rows, block_size)
        for col in range(0, cols, block_size)
    ]
    signature_to_id: dict[tuple[int, ...], int] = {}
    assignments: dict[tuple[int, int], int] = {}
    for row, col in block_positions:
        signature = block_signature_from_gradient(
            initial_gradient,
            row,
            col,
            block_size,
            quantization_step,
        )
        signature_to_id.setdefault(signature, len(signature_to_id))
        assignments[(row, col)] = signature_to_id[signature]
    prototypes = [zeros(block_size, block_size) for _ in signature_to_id]

    def materialize() -> list[list[float]]:
        delta = zeros(rows, cols)
        for row_start, col_start in block_positions:
            prototype = prototypes[assignments[(row_start, col_start)]]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    delta[row_start + r_offset][col_start + c_offset] = prototype[r_offset][c_offset]
        return delta

    for _ in range(steps):
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        prototype_gradients = [zeros(block_size, block_size) for _ in prototypes]
        for row_start, col_start in block_positions:
            prototype_id = assignments[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    prototype_gradients[prototype_id][r_offset][c_offset] += (
                        gradient[row_start + r_offset][col_start + c_offset]
                    )
        for prototype_id, prototype in enumerate(prototypes):
            for row in range(block_size):
                for col in range(block_size):
                    prototype[row][col] -= learning_rate * prototype_gradients[prototype_id][row][col]
    parameter_count = len(prototypes) * block_size * block_size + len(block_positions)
    return training_result(
        f"codebook_delta_{block_size}",
        task,
        materialize(),
        parameter_count,
        start,
        {
            "block_size": block_size,
            "prototype_count": len(prototypes),
            "assignment_count": len(block_positions),
        },
    )


def train_lora_delta(
    task: LinearTask,
    *,
    rank: int = 2,
    steps: int = 320,
    learning_rate: float = 0.55,
    seed: int = 17,
) -> TrainingResult:
    start = perf_counter()
    rng = Random(seed)
    rows, cols = shape(task.base_weight)
    left = zeros(rows, rank)
    right = [[rng.uniform(-0.02, 0.02) for _ in range(cols)] for _ in range(rank)]

    def materialize() -> list[list[float]]:
        delta = zeros(rows, cols)
        for row in range(rows):
            for hidden in range(rank):
                left_value = left[row][hidden]
                for col in range(cols):
                    delta[row][col] += left_value * right[hidden][col]
        return delta

    for _ in range(steps):
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        left_gradient = zeros(rows, rank)
        right_gradient = zeros(rank, cols)
        for row in range(rows):
            for hidden in range(rank):
                left_gradient[row][hidden] = sum(
                    gradient[row][col] * right[hidden][col] for col in range(cols)
                )
        for hidden in range(rank):
            for col in range(cols):
                right_gradient[hidden][col] = sum(
                    left[row][hidden] * gradient[row][col] for row in range(rows)
                )
        for row in range(rows):
            for hidden in range(rank):
                left[row][hidden] -= learning_rate * left_gradient[row][hidden]
        for hidden in range(rank):
            for col in range(cols):
                right[hidden][col] -= learning_rate * right_gradient[hidden][col]
    return training_result(
        f"lora_rank_{rank}",
        task,
        materialize(),
        rank * (rows + cols),
        start,
        {"rank": rank},
    )


def train_saint_routed_delta(
    task: LinearTask,
    *,
    block_size: int = 2,
    region_size: int = 4,
    quantization_step: float = 0.0869,
    free_region_fraction: float = 0.25,
    codebook_region_fraction: float = 0.50,
    steps: int = 260,
    learning_rate: float = 0.35,
    name: str = "saint_routed_delta",
    max_free_regions: int | None = None,
    max_codebook_regions: int | None = None,
    max_prototypes: int | None = None,
) -> TrainingResult:
    """Train deltas with freeze + codebook + free regions selected by sensitivity."""

    start = perf_counter()
    rows, cols = shape(task.base_weight)
    zero_delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, zero_delta)
    regions = []
    for row_start in range(0, rows, region_size):
        for col_start in range(0, cols, region_size):
            sensitivity = 0.0
            for row in range(row_start, min(row_start + region_size, rows)):
                for col in range(col_start, min(col_start + region_size, cols)):
                    sensitivity += abs(initial_gradient[row][col])
            regions.append((sensitivity, row_start, col_start))
    regions.sort(reverse=True)
    free_count = max(1, int(len(regions) * free_region_fraction))
    codebook_count = max(1, int(len(regions) * codebook_region_fraction))
    if max_free_regions is not None:
        free_count = min(free_count, max_free_regions)
    if max_codebook_regions is not None:
        codebook_count = min(codebook_count, max_codebook_regions)
    free_regions = {(row, col) for _s, row, col in regions[:free_count]}
    codebook_regions = {
        (row, col)
        for _s, row, col in regions[free_count:free_count + codebook_count]
    }
    free_values = zeros(rows, cols)
    block_positions = []
    signature_to_id: dict[tuple[int, ...], int] = {}
    assignments: dict[tuple[int, int], int] = {}
    for region_row, region_col in codebook_regions:
        for row in range(region_row, min(region_row + region_size, rows), block_size):
            for col in range(region_col, min(region_col + region_size, cols), block_size):
                block_positions.append((row, col))
                signature = block_signature_from_gradient(
                    initial_gradient,
                    row,
                    col,
                    block_size,
                    quantization_step,
                )
                if signature in signature_to_id:
                    prototype_id = signature_to_id[signature]
                elif max_prototypes is None or len(signature_to_id) < max_prototypes:
                    prototype_id = len(signature_to_id)
                    signature_to_id[signature] = prototype_id
                else:
                    prototype_id = _signature_bucket(signature, max_prototypes)
                    signature_to_id[signature] = prototype_id
                assignments[(row, col)] = prototype_id
    prototype_count = (
        max(signature_to_id.values()) + 1
        if signature_to_id
        else 0
    )
    prototypes = [zeros(block_size, block_size) for _ in range(prototype_count)]

    def is_free(row: int, col: int) -> bool:
        return ((row // region_size) * region_size, (col // region_size) * region_size) in free_regions

    def materialize() -> list[list[float]]:
        delta = zeros(rows, cols)
        for row in range(rows):
            for col in range(cols):
                if is_free(row, col):
                    delta[row][col] = free_values[row][col]
        for row_start, col_start in block_positions:
            prototype = prototypes[assignments[(row_start, col_start)]]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    delta[row_start + r_offset][col_start + c_offset] = prototype[r_offset][c_offset]
        return delta

    for _ in range(steps):
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row in range(rows):
            for col in range(cols):
                if is_free(row, col):
                    free_values[row][col] -= learning_rate * gradient[row][col]
        prototype_gradients = [zeros(block_size, block_size) for _ in prototypes]
        for row_start, col_start in block_positions:
            prototype_id = assignments[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    prototype_gradients[prototype_id][r_offset][c_offset] += (
                        gradient[row_start + r_offset][col_start + c_offset]
                    )
        for prototype_id, prototype in enumerate(prototypes):
            for row in range(block_size):
                for col in range(block_size):
                    prototype[row][col] -= learning_rate * prototype_gradients[prototype_id][row][col]
    parameter_count = (
        len(free_regions) * region_size * region_size
        + len(prototypes) * block_size * block_size
        + len(block_positions)
    )
    return training_result(
        name,
        task,
        materialize(),
        parameter_count,
        start,
        {
            "block_size": block_size,
            "region_size": region_size,
            "quantization_step": quantization_step,
            "free_regions": len(free_regions),
            "codebook_regions": len(codebook_regions),
            "frozen_regions": len(regions) - len(free_regions) - len(codebook_regions),
            "free_region_fraction": free_region_fraction,
            "codebook_region_fraction": codebook_region_fraction,
            "max_free_regions": max_free_regions,
            "max_codebook_regions": max_codebook_regions,
            "max_prototypes": max_prototypes,
            "prototype_count": len(prototypes),
        },
    )
