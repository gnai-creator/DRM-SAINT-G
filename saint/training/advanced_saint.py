"""Advanced SAINT delta training variants for phase 4."""

from __future__ import annotations

from time import perf_counter

from saint.reconstruction.matrix_ops import add, shape, subtract, zeros
from saint.training.data import LinearTask
from saint.training.ops import (
    TrainingResult,
    loss_and_delta_gradient,
    mse_loss,
    training_result,
)
from saint.training.saint_delta import (
    _block_l1,
    _block_vector,
    _kmeans_assignments,
    _least_squares_scale,
    _normalized_block_vector,
    _target_delta,
    _vector_to_block,
)


def _default_budget(rows: int, cols: int) -> int:
    return max(44, int((rows + cols) * 3))


def _block_positions(rows: int, cols: int, block_size: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(0, rows, block_size)
        for col in range(0, cols, block_size)
    ]


def _block_cells(
    row_start: int,
    col_start: int,
    rows: int,
    cols: int,
    block_size: int,
) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(row_start, min(row_start + block_size, rows))
        for col in range(col_start, min(col_start + block_size, cols))
    ]


def _copy_matrix(matrix):
    return [row[:] for row in matrix]


def _apply_block(delta, row_start: int, col_start: int, block) -> None:
    rows, cols = shape(delta)
    block_rows, block_cols = shape(block)
    for r_offset in range(min(block_rows, rows - row_start)):
        for c_offset in range(min(block_cols, cols - col_start)):
            delta[row_start + r_offset][col_start + c_offset] += block[r_offset][c_offset]


def _rank1_approx_block(matrix, row_start: int, col_start: int, block_size: int):
    rows, cols = shape(matrix)
    block = zeros(block_size, block_size)
    for r_offset in range(min(block_size, rows - row_start)):
        for c_offset in range(min(block_size, cols - col_start)):
            block[r_offset][c_offset] = matrix[row_start + r_offset][col_start + c_offset]
    pivot = max(range(block_size), key=lambda row: sum(abs(value) for value in block[row]))
    right = block[pivot][:]
    denominator = sum(value * value for value in right) or 1.0
    left = [
        sum(block[row][col] * right[col] for col in range(block_size)) / denominator
        for row in range(block_size)
    ]
    return [[left[row] * right[col] for col in range(block_size)] for row in range(block_size)]


def _train_loss(task: LinearTask, delta) -> float:
    return mse_loss(add(task.base_weight, delta), task.train_inputs, task.train_targets)


def _make_codebook(
    target_delta,
    initial_gradient,
    *,
    block_size: int,
    max_blocks: int,
    max_prototypes: int,
    quantization_step: float,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], list, dict, dict]:
    rows, cols = shape(target_delta)
    ranked = sorted(
        (
            _block_l1(initial_gradient, row, col, block_size) / 3.0,
            row,
            col,
        )
        for row, col in _block_positions(rows, cols, block_size)
    )
    positions = sorted((row, col) for _score, row, col in ranked[-max_blocks:])
    vectors = [
        _normalized_block_vector(initial_gradient, row, col, block_size, quantization_step)
        for row, col in positions
    ]
    assignment_values = _kmeans_assignments(vectors, min(max_prototypes, len(positions)))
    assignments = dict(zip(positions, assignment_values))
    prototype_count = max(assignment_values) + 1 if assignment_values else 0
    prototype_vectors = []
    for prototype_id in range(prototype_count):
        members = [
            _block_vector(target_delta, row, col, block_size)
            for (row, col), assignment in assignments.items()
            if assignment == prototype_id
        ]
        width = block_size * block_size
        prototype_vectors.append(
            [sum(vector[index] for vector in members) / len(members) for index in range(width)]
        )
    prototypes = [_vector_to_block(vector, block_size) for vector in prototype_vectors]
    scales = {
        position: _least_squares_scale(
            _block_vector(target_delta, position[0], position[1], block_size),
            prototype_vectors[assignments[position]],
        )
        for position in positions
    }
    biases = {position: 0.0 for position in positions}
    return positions, assignments, prototypes, scales, biases


def train_saint_dynamic_delta(
    task: LinearTask,
    *,
    parameter_budget: int | None = None,
    block_size: int = 2,
    low_rank_block_size: int = 4,
    quantization_step: float = 0.0869,
    max_prototypes: int = 8,
    steps: int = 260,
    learning_rate: float = 0.35,
    warmup_fraction: float = 0.35,
    name: str = "saint_dynamic_delta",
) -> TrainingResult:
    """Train SAINT with dynamic residual allocation and marginal gain routing."""

    start = perf_counter()
    rows, cols = shape(task.base_weight)
    budget = parameter_budget or _default_budget(rows, cols)
    target_delta = _target_delta(task)
    zero_delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, zero_delta)
    prototype_cost = max_prototypes * block_size * block_size
    max_blocks = max(1, (budget - prototype_cost) // 3)
    positions, assignments, prototypes, scales, biases = _make_codebook(
        target_delta,
        initial_gradient,
        block_size=block_size,
        max_blocks=max_blocks,
        max_prototypes=max_prototypes,
        quantization_step=quantization_step,
    )
    base_cost = (
        len(prototypes) * block_size * block_size
        + len(positions) * 3
    )
    residual_values: dict[tuple[int, int], list[list[float]]] = {}
    low_rank_values: dict[tuple[int, int], list[list[float]]] = {}
    sensitivity = {position: 0.0 for position in _block_positions(rows, cols, block_size)}

    def materialize() -> list[list[float]]:
        delta = zeros(rows, cols)
        for row_start, col_start in positions:
            prototype = prototypes[assignments[(row_start, col_start)]]
            scale = scales[(row_start, col_start)]
            bias = biases[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    delta[row_start + r_offset][col_start + c_offset] = (
                        prototype[r_offset][c_offset] * scale + bias
                    )
        for row, col in low_rank_values:
            _apply_block(delta, row, col, low_rank_values[(row, col)])
        for row, col in residual_values:
            _apply_block(delta, row, col, residual_values[(row, col)])
        return delta

    def train_step(update_residuals: bool, accumulate: bool) -> None:
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        if accumulate:
            for position in sensitivity:
                row, col = position
                sensitivity[position] += _block_l1(gradient, row, col, block_size)
        prototype_gradients = [zeros(block_size, block_size) for _ in prototypes]
        scale_gradients = {position: 0.0 for position in positions}
        bias_gradients = {position: 0.0 for position in positions}
        for row_start, col_start in positions:
            prototype_id = assignments[(row_start, col_start)]
            prototype = prototypes[prototype_id]
            scale = scales[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    grad = gradient[row_start + r_offset][col_start + c_offset]
                    prototype_gradients[prototype_id][r_offset][c_offset] += grad * scale
                    scale_gradients[(row_start, col_start)] += grad * prototype[r_offset][c_offset]
                    bias_gradients[(row_start, col_start)] += grad
        for prototype_id, prototype in enumerate(prototypes):
            for row in range(block_size):
                for col in range(block_size):
                    prototype[row][col] -= learning_rate * prototype_gradients[prototype_id][row][col]
        for position in positions:
            scales[position] -= learning_rate * scale_gradients[position]
            biases[position] -= learning_rate * bias_gradients[position]
        if update_residuals:
            for store in (low_rank_values, residual_values):
                for row_start, col_start in store:
                    block = store[(row_start, col_start)]
                    b_rows, b_cols = shape(block)
                    for r_offset in range(min(b_rows, rows - row_start)):
                        for c_offset in range(min(b_cols, cols - col_start)):
                            block[r_offset][c_offset] -= (
                                learning_rate * gradient[row_start + r_offset][col_start + c_offset]
                            )

    warmup_steps = max(1, min(steps - 1, int(steps * warmup_fraction))) if steps > 1 else 0
    for _ in range(warmup_steps):
        train_step(update_residuals=False, accumulate=True)

    warm_delta = materialize()
    residual_matrix = subtract(target_delta, warm_delta)
    baseline_loss = _train_loss(task, warm_delta)
    candidates = []
    for row, col in _block_positions(rows, cols, block_size):
        candidates.append(("residual", block_size * block_size, row, col, _vector_to_block(
            _block_vector(residual_matrix, row, col, block_size),
            block_size,
        )))
    for row, col in _block_positions(rows, cols, low_rank_block_size):
        low_rank_block = _rank1_approx_block(residual_matrix, row, col, low_rank_block_size)
        candidates.append(("local_low_rank", low_rank_block_size * 2, row, col, low_rank_block))

    scored = []
    for kind, cost, row, col, block in candidates:
        trial = _copy_matrix(warm_delta)
        _apply_block(trial, row, col, block)
        gain = max(baseline_loss - _train_loss(task, trial), 0.0)
        sens = sensitivity.get((row // block_size * block_size, col // block_size * block_size), 0.0)
        scored.append((gain * (1.0 + sens) / max(cost, 1), gain, kind, cost, row, col, block))

    remaining = max(0, budget - base_cost)
    occupied: set[tuple[int, int]] = set()
    for _score, gain, kind, cost, row, col, block in sorted(scored, reverse=True):
        cells = set(_block_cells(row, col, rows, cols, len(block)))
        if gain <= 0.0 or cost > remaining or cells & occupied:
            continue
        if kind == "local_low_rank":
            low_rank_values[(row, col)] = block
        else:
            residual_values[(row, col)] = block
        occupied.update(cells)
        remaining -= cost

    for _ in range(steps - warmup_steps):
        train_step(update_residuals=True, accumulate=False)

    parameter_count = (
        base_cost
        + len(residual_values) * block_size * block_size
        + len(low_rank_values) * low_rank_block_size * 2
    )
    return training_result(
        name,
        task,
        materialize(),
        parameter_count,
        start,
        {
            "parameter_budget": budget,
            "block_size": block_size,
            "low_rank_block_size": low_rank_block_size,
            "prototype_count": len(prototypes),
            "codebook_blocks": len(positions),
            "residual_blocks": len(residual_values),
            "low_rank_blocks": len(low_rank_values),
            "has_block_bias": True,
            "has_accumulated_sensitivity": True,
            "residual_selection": "real_marginal_gain_per_parameter",
            "budget_allocator": "dynamic_greedy",
            "codebook_stage": "scale_bias_kmeans",
        },
    )


__all__ = ["train_saint_dynamic_delta"]
